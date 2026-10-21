// optimize.cu
// 独立工具：读取 bucket.cu 产出的 vector_knn.bin 中间 KNN 图 + 原始数据集，
// 调用 RAFT CAGRA 的 sort_knn_graph + optimize，输出 CAGRA 风格的剪枝图。
//
// 流程：
//   1) 读取原始数据集 (.fbin/.u8bin/.i8bin/.ibin) → host float32
//   2) 读取 vector_knn.bin → host uint32 (N, M_in)
//   3) sort_knn_graph_host: 按 L2 升序排序每行邻居（CPU/OpenMP；bucket2 的
//        vector_knn.bin 已经按距离升序，pipeline 传 --skip-sort 跳过）
//   4) cuvs::neighbors::cagra::helpers::optimize (cuVS 26.10 public API)
//        剪枝到目标度数 K_out （CAGRA 反向边平衡 + 距离剪枝）
//   5) 写出结果到磁盘
//
// 使用：
//   ./optimize \
//       --input data.fbin \
//       --knn-graph output_dir/vector_knn.bin \
//       --output   output_dir/cagra_graph.bin \
//       --output-degree 32 \
//       [--save-npy]
//
// 输出格式 (cagra_graph.bin):
//   int64_t  N
//   int32_t  output_degree (K_out)
//   uint32_t graph[N * K_out]  (row-major, CAGRA 剪枝图)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <filesystem>
#include <algorithm>
#include <omp.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include "load.hpp"

namespace po = boost::program_options;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("CUDA error: ") +                \
                                     cudaGetErrorString(err) +                    \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                         \
    } while (0)

// ---------- vector_knn.bin reader ----------
//
// Format (写入端见 bucket.cu :: write_vector_knn_to_disk):
//   int64_t  N
//   int32_t  M
//   int32_t  neighbors[N*M]   (row-major, -1 表示 padding)
static void read_vector_knn_bin(const std::string& path,
                                std::vector<int32_t>& neighbors,
                                int64_t& N, int32_t& M)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open KNN graph file: " + path);

    in.read(reinterpret_cast<char*>(&N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&M), sizeof(int32_t));
    if (!in.good() || N <= 0 || M <= 0)
        throw std::runtime_error("Invalid vector_knn.bin header");

    const size_t cnt = static_cast<size_t>(N) * static_cast<size_t>(M);
    neighbors.resize(cnt);
    in.read(reinterpret_cast<char*>(neighbors.data()),
            static_cast<std::streamsize>(cnt * sizeof(int32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read vector_knn.bin payload");
}

// 把 int32 邻居（含 -1 padding）转为 uint32，并修复 padding 与越界。
// CAGRA 的 sort/optimize 要求每个邻居是 [0, N) 范围内的合法 id；
// -1 (padding) 会被替换为该行内已有的合法邻居，找不到时回退到 (row+1)%N。
// 原地转换 (int32 / uint32 同宽，每个元素只在自己的位置读一次写一次；
// signed/unsigned 同类型允许互相 alias)，之后调用方把 buffer 当 uint32 用，
// 不再额外分配一份 N*M 的输出 (SIFT1M: 256 MB alloc + page faults)。
static void sanitize_to_uint32_inplace(
    std::vector<int32_t>& src, int64_t N, int32_t M)
{
    static_assert(sizeof(int32_t) == sizeof(uint32_t), "in-place sanitize needs same width");
    int64_t total_padding = 0;

    #pragma omp parallel for reduction(+:total_padding) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const int32_t* row_in = src.data() + i * M;
        uint32_t* row_out     = reinterpret_cast<uint32_t*>(src.data() + i * M);

        // 第一遍：找该行任意一个合法邻居作为 fallback
        uint32_t fallback = static_cast<uint32_t>((i + 1) % N);
        for (int j = 0; j < M; ++j) {
            int32_t v = row_in[j];
            if (v >= 0 && v < static_cast<int32_t>(N) &&
                static_cast<int64_t>(v) != i) {
                fallback = static_cast<uint32_t>(v);
                break;
            }
        }

        // 第二遍：写出，padding/越界/自环统一替换为 fallback
        for (int j = 0; j < M; ++j) {
            int32_t v = row_in[j];
            if (v < 0 || v >= static_cast<int32_t>(N) ||
                static_cast<int64_t>(v) == i) {
                row_out[j] = fallback;
                if (v < 0) total_padding++;
            } else {
                row_out[j] = static_cast<uint32_t>(v);
            }
        }
    }

    if (total_padding > 0) {
        std::cout << "[Sanitize] Replaced " << total_padding
                  << " padding (-1) entries with fallback neighbors\n";
    }
}

// Lightweight header-only peek at vector_knn.bin (int64 N, int32 M) -- no
// payload read, so the GPU-fit check below can decide whether to bail
// *before* paying for a potentially multi-GB dataset/graph load.
static void peek_vector_knn_header(const std::string& path, int64_t& N, int32_t& M)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open KNN graph file: " + path);
    in.read(reinterpret_cast<char*>(&N), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&M), sizeof(int32_t));
    if (!in.good() || N <= 0 || M <= 0)
        throw std::runtime_error("Invalid vector_knn.bin header: " + path);
}

// Heuristic only, not exact accounting: raft's sort_knn_graph/optimize
// (graph_core.cuh) manage their own internal device working buffers
// (reverse-graph construction, detour counting, etc.) that aren't visible
// from this call site, so the raw dataset+graph+output byte count alone
// would underestimate real usage. `margin` is a blunt safety multiplier to
// cover that unknown overhead -- tune with --gpu-fit-margin if it's
// rejecting runs that would actually fit, or if you still hit a CUDA OOM
// despite this check passing.
static size_t estimate_required_gpu_bytes(int64_t N, int32_t D, int32_t M_in,
                                           int32_t K_out, double margin)
{
    size_t dataset_bytes   = static_cast<size_t>(N) * static_cast<size_t>(D) * sizeof(float);
    size_t candidate_bytes = static_cast<size_t>(N) * static_cast<size_t>(M_in) * sizeof(uint32_t);
    size_t output_bytes    = static_cast<size_t>(N) * static_cast<size_t>(K_out) * sizeof(uint32_t);
    size_t base = dataset_bytes + candidate_bytes + output_bytes;
    return static_cast<size_t>(static_cast<double>(base) * margin);
}

// 输出 CAGRA 图到磁盘
//   int64_t  N
//   int32_t  output_degree
//   uint32_t graph[N * output_degree]
// Per-row L2 sort of the candidate list (replaces raft's internal GPU
// sort_knn_graph, which cuVS 26.10 no longer exposes).
static void sort_knn_graph_host(const float* data, int64_t N, int64_t D,
                                uint32_t* graph, int64_t M)
{
    #pragma omp parallel
    {
        std::vector<std::pair<float, uint32_t>> row(M);
        #pragma omp for schedule(dynamic, 1024)
        for (int64_t i = 0; i < N; ++i) {
            const float* xi = data + i * D;
            for (int64_t j = 0; j < M; ++j) {
                const float* xj = data + static_cast<int64_t>(graph[i * M + j]) * D;
                float d = 0.f;
                for (int64_t t = 0; t < D; ++t) { float e = xi[t] - xj[t]; d += e * e; }
                row[j] = {d, graph[i * M + j]};
            }
            std::stable_sort(row.begin(), row.end(),
                             [](auto& a, auto& b) { return a.first < b.first; });
            for (int64_t j = 0; j < M; ++j) graph[i * M + j] = row[j].second;
        }
    }
}

static void write_cagra_graph_bin(const std::string& path,
                                  const uint32_t* graph,
                                  int64_t N, int32_t K_out)
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open output: " + path);

    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&K_out), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(graph),
              static_cast<size_t>(N) * K_out * sizeof(uint32_t));
    out.close();

    std::cout << "[Write] " << path << " ("
              << static_cast<size_t>(N) * K_out * sizeof(uint32_t) / 1e6
              << " MB)\n";
}

int main(int argc, char** argv)
{
    try {
        po::options_description desc("CAGRA sort + optimize for bucket.cu's vector_knn.bin");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i",       po::value<std::string>()->required(),
                "Original dataset (.fbin/.bin/.u8bin/.i8bin/.ibin) — "
                "needed by sort_knn_graph for L2 distance")
            ("knn-graph,g",   po::value<std::string>()->required(),
                "Input vector_knn.bin (intermediate KNN graph from bucket.cu)")
            ("output,o",      po::value<std::string>()->required(),
                "Output cagra_graph.bin (pruned CAGRA graph)")
            ("output-degree", po::value<int32_t>()->default_value(32),
                "Target output graph degree K_out (must be <= input M)")
            ("skip-sort",     po::bool_switch()->default_value(false),
                "Skip sort_knn_graph (assume input is already sorted by L2)")
            ("save-npy",      po::bool_switch()->default_value(false),
                "Also write a numpy .npy alongside the .bin (int64, padding-free)")
            ("force-chunked", po::bool_switch()->default_value(false),
                "This tool IS the 'fits on one GPU, run raft's native "
                "sort_knn_graph+optimize directly' path -- by default (false) "
                "it auto-checks dataset+graph+output against free GPU memory "
                "and only proceeds if it fits (see --gpu-fit-margin). Pass "
                "true to force it to refuse and point at prune_windowed + "
                "optimize_chunked instead, even on a dataset (e.g. Sift1M) "
                "that would actually fit -- for validating the chunked "
                "pipeline's output against this one's on the same data.")
            ("gpu-fit-margin", po::value<double>()->default_value(3.0),
                "Safety multiplier on the dataset+graph+output byte estimate "
                "before comparing against free GPU memory -- raft's internal "
                "working buffers aren't visible from here, so this is a "
                "heuristic, not exact accounting. Raise it if you hit a CUDA "
                "OOM despite the check passing; lower it if it rejects runs "
                "that would actually fit.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);

        const std::string input_path = vm["input"].as<std::string>();
        const std::string knn_path   = vm["knn-graph"].as<std::string>();
        const std::string out_path   = vm["output"].as<std::string>();
        const int32_t     K_out      = vm["output-degree"].as<int32_t>();
        const bool        skip_sort  = vm["skip-sort"].as<bool>();
        const bool        save_npy   = vm["save-npy"].as<bool>();
        const bool        force_chunked   = vm["force-chunked"].as<bool>();
        const double      gpu_fit_margin  = vm["gpu-fit-margin"].as<double>();

        if (K_out <= 0)
            throw std::runtime_error("--output-degree must be > 0");

        using Clock = std::chrono::steady_clock;
        auto t_start = Clock::now();

        // ---------------------------------------------------------------
        // Step 0: GPU-fit check. This tool IS the "data fits on one GPU,
        // call raft's own sort_knn_graph+optimize directly" path -- the
        // ~3s-on-Sift1M baseline raft gives you is exactly this. Header-only
        // peeks (no payload read) so a reject is cheap, not a multi-GB
        // load followed by a refusal. When it doesn't fit (or
        // --force-chunked asks to validate the other path on purpose), the
        // right tool is prune_windowed + optimize_chunked -- this one
        // doesn't invoke them itself (disjoint, bucket-specific argument
        // set: bucket-offsets/bucket-process-order/centroid-knn/cache-mb/
        // gpu-budget-mb/...), it just says so and exits.
        // ---------------------------------------------------------------
        {
            auto [N32_peek, D_peek] = load::read_fbin_header(input_path);
            int64_t N_peek = 0; int32_t M_peek = 0;
            peek_vector_knn_header(knn_path, N_peek, M_peek);

            size_t required = estimate_required_gpu_bytes(N_peek, D_peek, M_peek, K_out, gpu_fit_margin);
            size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

            std::cout << "=== Step 0: GPU-fit check ===\n";
            std::cout << "  N=" << N32_peek << " D=" << D_peek << " M_in=" << M_peek
                      << " K_out=" << K_out << "\n";
            std::cout << "  Estimated requirement (x" << gpu_fit_margin << " margin): "
                      << required / 1e9 << " GB, GPU free: " << free_bytes / 1e9
                      << " GB / " << total_bytes / 1e9 << " GB\n";

            const bool fits = (required <= free_bytes);
            if (force_chunked || !fits) {
                std::cout << "  -> " << (force_chunked ? "--force-chunked set"
                                                        : "does not fit free GPU memory")
                          << " -- this tool will NOT run. Use the chunked pipeline instead:\n"
                          << "       prune_windowed  (--cache-mb sized to host budget)\n"
                          << "       optimize_chunked -m E  (--gpu-budget-mb sized to this GPU)\n"
                          << "     See Readme.md for the full bucket-pipeline argument list.\n";
                return 2;
            }
            std::cout << "  -> fits, proceeding with raft native path.\n";
        }

        // ---------------------------------------------------------------
        // Step 1: Load original dataset (host float32)
        // ---------------------------------------------------------------
        std::cout << "=== Step 1: Loading dataset ===\n";
        auto t1 = Clock::now();

        // --skip-sort: cuVS optimize 只用 graph，不用 dataset —— 只读 header
        // 拿 N/D，不再整份读入并转 float32 (SIFT1M: 0.5 GB, ~0.55s)。
        std::vector<float> dataset_host;
        int32_t N32 = 0, D = 0;
        std::string ext = std::filesystem::path(input_path).extension().string();
        if (skip_sort) {
            auto [n_hdr, d_hdr] = load::read_fbin_header(input_path);
            N32 = n_hdr; D = d_hdr;
        } else if (ext == ".fbin" || ext == ".bin") {
            load::read_fbin_f32(input_path, dataset_host, N32, D);
        } else if (ext == ".u8bin" || ext == ".i8bin") {
            load::read_u8bin_to_f32(input_path, dataset_host, N32, D);
        } else if (ext == ".ibin") {
            std::vector<int32_t> tmp;
            load::read_ibin_i32(input_path, tmp, N32, D);
            dataset_host.resize(tmp.size());
            for (size_t i = 0; i < tmp.size(); ++i)
                dataset_host[i] = static_cast<float>(tmp[i]);
        } else {
            throw std::runtime_error("Unsupported input extension: " + ext);
        }
        const int64_t N = N32;
        std::cout << "  N=" << N << " D=" << D
                  << " (" << dataset_host.size() * sizeof(float) / 1e9 << " GB)"
                  << " [" << std::chrono::duration<double>(Clock::now() - t1).count()
                  << "s]\n";

        // ---------------------------------------------------------------
        // Step 2: Load vector_knn.bin and sanitize to uint32
        // ---------------------------------------------------------------
        std::cout << "=== Step 2: Loading KNN graph ===\n";
        auto t2 = Clock::now();
        std::vector<int32_t> knn_i32;
        int64_t Ng = 0;
        int32_t M_in = 0;
        read_vector_knn_bin(knn_path, knn_i32, Ng, M_in);
        if (Ng != N)
            throw std::runtime_error("KNN graph N (" + std::to_string(Ng) +
                                     ") != dataset N (" + std::to_string(N) + ")");
        if (K_out > M_in)
            throw std::runtime_error("--output-degree (" + std::to_string(K_out) +
                                     ") > input degree (" + std::to_string(M_in) + ")");

        std::cout << "  Input graph: N=" << N << " M_in=" << M_in
                  << " (" << knn_i32.size() * sizeof(int32_t) / 1e6 << " MB)\n";

        sanitize_to_uint32_inplace(knn_i32, N, M_in);
        uint32_t* knn_u32 = reinterpret_cast<uint32_t*>(knn_i32.data());
        std::cout << "  Sanitize done ["
                  << std::chrono::duration<double>(Clock::now() - t2).count()
                  << "s]\n";

        // ---------------------------------------------------------------
        // Step 3: CAGRA sort_knn_graph (按 L2 排序每行)
        // ---------------------------------------------------------------
        raft::resources res;

        auto knn_graph_view = raft::make_host_matrix_view<uint32_t, int64_t>(
            knn_u32, N, M_in);

        if (!skip_sort) {
            std::cout << "=== Step 3: CAGRA sort_knn_graph ===\n";
            auto t3 = Clock::now();
            sort_knn_graph_host(dataset_host.data(), N, D, knn_u32, M_in);
            std::cout << "  Sort done ["
                      << std::chrono::duration<double>(Clock::now() - t3).count()
                      << "s]\n";
        } else {
            std::cout << "=== Step 3: skip sort (per --skip-sort) ===\n";
        }

        // ---------------------------------------------------------------
        // Step 4: CAGRA optimize (剪枝到 K_out)
        // ---------------------------------------------------------------
        std::cout << "=== Step 4: CAGRA optimize (prune to K_out=" << K_out << ") ===\n";
        auto t4 = Clock::now();

        auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(N, K_out);
        cuvs::neighbors::cagra::helpers::optimize(
            res, knn_graph_view, cagra_graph.view());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "  Optimize done ["
                  << std::chrono::duration<double>(Clock::now() - t4).count()
                  << "s]\n";

        // 中间图已经不需要了
        { auto _drop = std::move(knn_i32); }
        { auto _drop = std::move(dataset_host); }

        // ---------------------------------------------------------------
        // Step 5: Write outputs
        // ---------------------------------------------------------------
        std::cout << "=== Step 5: Writing output ===\n";
        auto t5 = Clock::now();
        write_cagra_graph_bin(out_path, cagra_graph.data_handle(), N, K_out);

        if (save_npy) {
            // 转 int64 写 .npy（与 Python 端 search 脚本兼容）
            std::vector<int64_t> graph_i64(static_cast<size_t>(N) * K_out);
            const uint32_t* src = cagra_graph.data_handle();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < graph_i64.size(); ++i)
                graph_i64[i] = static_cast<int64_t>(src[i]);

            std::string npy_path = out_path;
            auto p = std::filesystem::path(npy_path);
            p.replace_extension(".npy");
            load::write_npy_int64_2d(p.string(), graph_i64.data(), N, K_out);
            std::cout << "[Write] " << p.string() << " (npy)\n";
        }
        std::cout << "  Write done ["
                  << std::chrono::duration<double>(Clock::now() - t5).count()
                  << "s]\n";

        double total = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cout << "\n=== Done ===\n";
        std::cout << "Total: " << std::fixed << std::setprecision(3) << total << "s\n";
        std::cout << "Output: " << out_path << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
