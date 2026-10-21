// prune_windowed.cpp
// Approximate, memory-bounded stand-in for CAGRA prune (raft
// sort_knn_graph + graph_core.cuh's optimize()'s pruning half), for
// datasets too large to hold the candidate graph resident on GPU/host the
// way raft's own implementation requires.
//
// Deliberately CPU-only (OpenMP across points), not a CUDA kernel: the
// scaling problem this tool targets is *memory* (raft needs the whole
// candidate graph resident to resolve "does my neighbor's neighbor connect
// back to me" lookups), not compute throughput. bucket_order.hpp's
// BeladyBucketCache already solves the memory side with a plain host-memory
// budget; layering an untested CUDA kernel on top would only buy speed,
// at the cost of correctness risk that can't be checked without a GPU on
// hand. See prune_windowed_core.hpp for the actual detour-counting /
// selection algorithm, which is unit-tested standalone with g++
// (tests/test_prune_windowed_core.cpp) and is what a future GPU kernel
// would need to reproduce.
//
// Input:
//   --vector-knn-reordered   vector_knn_reordered.bin (from reorder.cpp)
//   --dataset                data_reordered.<ext> (from reorder.cpp)
//   --bucket-offsets         bucket_offsets.bin (from reorder.cpp)
//   --bucket-process-order   bucket_process_order.bin (from reorder.cpp)
//   --centroid-knn           centroid_knn.bin (from bucket.cu --reorder)
// Output:
//   forward_graph_reordered.bin -- pruned graph, still in reordered
//   coordinate space, format matches optimize_chunked's forward-graph input
//   (int64 N, int32 output_degree, uint32 graph[N*output_degree]) -- feed
//   straight into optimize_chunked --method E (or C/D) with the same
//   --bucket-offsets.
//
// Processing order = bucket_process_order (NOT recomputed here -- see
// graph_io.hpp's comment on why re-deriving it risks silent misalignment).
// Visibility window per position = center bucket + its centroid_knn
// neighbor buckets, loaded through a Belady-optimal cache
// (bucket_order.hpp) sized by --cache-mb; a candidate whose neighbor's
// neighbor falls outside the window is conservatively treated as "no
// detour" (can only under-prune, never mis-prune -- see
// prune_windowed_core.hpp's file header).

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <boost/program_options.hpp>

#include "bucket_order.hpp"
#include "graph_io.hpp"
#include "load.hpp"
#include "prune_windowed_core.hpp"
#include "prune_windowed_gpu.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------
// One bucket's worth of window data: candidate rows + vectors, contiguous
// in the reordered id space. This is the BeladyBucketCache's BucketT.
// ---------------------------------------------------------------------
struct BucketWindowBlock {
    int32_t pos = 0;   // processing position (index into bucket_offsets)
    int64_t first_global_id = 0;
    int64_t count = 0;
    int32_t M_in = 0;
    int32_t D = 0;
    std::vector<int32_t> candidates;  // [count * M_in]
    std::vector<float> vectors;       // [count * D]
};

static size_t block_size_bytes(const BucketWindowBlock& b) {
    return b.candidates.size() * sizeof(int32_t) + b.vectors.size() * sizeof(float);
}

// 多线程 pread 一段文件到 dst (每线程读自己的子区间；页面首次触碰也随之
// 并行，而不是一个线程串行 page-fault 几百 MB)。
static void parallel_pread(const std::string& path, off_t offset, size_t bytes, void* dst)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("prune_windowed: cannot open " + path);
    constexpr size_t kPiece = size_t(8) << 20;
    const int64_t n_pieces = static_cast<int64_t>((bytes + kPiece - 1) / kPiece);
    int64_t n_fail = 0;
    #pragma omp parallel for schedule(dynamic) num_threads(16) reduction(+:n_fail)
    for (int64_t i = 0; i < n_pieces; ++i) {
        size_t lo = static_cast<size_t>(i) * kPiece;
        size_t n = std::min(kPiece, bytes - lo);
        char* p = static_cast<char*>(dst) + lo;
        off_t off = offset + static_cast<off_t>(lo);
        while (n > 0) {
            ssize_t r = ::pread(fd, p, n, off);
            if (r <= 0) { ++n_fail; break; }
            p += r; n -= static_cast<size_t>(r); off += r;
        }
    }
    ::close(fd);
    if (n_fail > 0) throw std::runtime_error("prune_windowed: short read from " + path);
}

static void read_dataset_chunk(const std::string& path, const std::string& ext,
                               int64_t start_row, int64_t num_rows,
                               std::vector<float>& out, int32_t& D)
{
    if (ext == ".fbin" || ext == ".bin") {
        load::read_fbin_chunk(path, start_row, num_rows, out, D);
    } else if (ext == ".u8bin" || ext == ".i8bin") {
        load::read_u8bin_chunk_to_f32(path, start_row, num_rows, out, D);
    } else if (ext == ".ibin") {
        load::read_ibin_chunk_to_f32(path, start_row, num_rows, out, D);
    } else {
        throw std::runtime_error("prune_windowed: unsupported dataset extension: " + ext);
    }
}

int main(int argc, char** argv)
{
    try {
        po::options_description desc("Windowed approximate CAGRA prune for oversized graphs");
        desc.add_options()
            ("help,h", "Show help")
            ("vector-knn-reordered,k", po::value<std::string>()->required(),
                "vector_knn_reordered.bin (from reorder.cpp)")
            ("dataset,i", po::value<std::string>()->required(),
                "Reordered dataset, data_reordered.<ext> (from reorder.cpp)")
            ("bucket-offsets", po::value<std::string>()->required(),
                "bucket_offsets.bin (from reorder.cpp)")
            ("bucket-process-order", po::value<std::string>()->required(),
                "bucket_process_order.bin (from reorder.cpp)")
            ("centroid-knn,c", po::value<std::string>()->required(),
                "centroid_knn.bin (from bucket.cu --reorder)")
            ("output,o", po::value<std::string>()->required(),
                "Output pruned forward graph (forward_graph_reordered.bin)")
            ("output-degree", po::value<int32_t>()->default_value(32),
                "Target output graph degree K_out (must be <= input M)")
            ("order-window", po::value<int32_t>()->default_value(0),
                "Visibility window size in buckets (0 = auto, 4*K -- MUST match the "
                "--order-window used when producing bucket_process_order.bin via reorder, "
                "or leave both at their auto default)")
            ("cache-mb", po::value<size_t>()->default_value(4000),
                "Host memory budget (MB) for the Belady bucket cache (candidate rows + vectors)")
            ("save-npy", po::bool_switch()->default_value(false),
                "Also write a .npy alongside the .bin");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);

        const std::string vknn_path   = vm["vector-knn-reordered"].as<std::string>();
        const std::string dataset_path= vm["dataset"].as<std::string>();
        const std::string bo_path     = vm["bucket-offsets"].as<std::string>();
        const std::string bpo_path    = vm["bucket-process-order"].as<std::string>();
        const std::string cknn_path   = vm["centroid-knn"].as<std::string>();
        const std::string out_path    = vm["output"].as<std::string>();
        const int32_t output_degree   = vm["output-degree"].as<int32_t>();
        const int32_t order_window_arg= vm["order-window"].as<int32_t>();
        const size_t cache_mb         = vm["cache-mb"].as<size_t>();
        const bool save_npy           = vm["save-npy"].as<bool>();

        using Clock = std::chrono::steady_clock;
        auto t_start = Clock::now();

        // ================================================================
        // Step 1: bucket structure + adjacency + processing order
        // ================================================================
        std::cout << "=== Step 1: Loading bucket structure ===\n";
        auto offsets_new = graph_io::read_bucket_offsets(bo_path);
        const int32_t n_buckets = static_cast<int32_t>(offsets_new.size()) - 1;
        const int64_t N = offsets_new.back();

        auto order = graph_io::read_bucket_process_order(bpo_path);
        if (static_cast<int32_t>(order.size()) != n_buckets)
            throw std::runtime_error("bucket_process_order size (" + std::to_string(order.size()) +
                                     ") != n_buckets from bucket_offsets (" + std::to_string(n_buckets) + ")");
        std::vector<int32_t> pos_of(static_cast<size_t>(n_buckets));
        for (int32_t pos = 0; pos < n_buckets; ++pos) pos_of[static_cast<size_t>(order[static_cast<size_t>(pos)])] = pos;

        auto ck = graph_io::read_centroid_knn(cknn_path);
        if (ck.n_centroids != n_buckets)
            throw std::runtime_error("centroid_knn n_centroids (" + std::to_string(ck.n_centroids) +
                                     ") != n_buckets (" + std::to_string(n_buckets) + ")");
        auto adj = bucket_order::adjacency_from_flat_graph(ck.graph.data(), ck.n_centroids, ck.K);

        int32_t order_window = (order_window_arg > 0) ? order_window_arg : std::max<int32_t>(4 * ck.K, 16);
        auto future = bucket_order::compute_future_access_lists(order, adj);

        std::cout << "  N=" << N << " n_buckets=" << n_buckets
                  << " centroid_knn K=" << ck.K << " order_window=" << order_window << "\n";

        // ================================================================
        // Step 2: open persistent handles for the two windowed data sources
        // ================================================================
        graph_io::VectorKnnReader knn_reader(vknn_path);
        if (knn_reader.N() != N)
            throw std::runtime_error("vector_knn_reordered N (" + std::to_string(knn_reader.N()) +
                                     ") != bucket_offsets N (" + std::to_string(N) + ")");
        const int32_t M_in = knn_reader.M();
        if (output_degree > M_in)
            throw std::runtime_error("--output-degree (" + std::to_string(output_degree) +
                                     ") > input degree (" + std::to_string(M_in) + ")");

        const std::string dataset_ext = fs::path(dataset_path).extension().string();
        const int32_t dataset_D_peek = load::read_fbin_header(dataset_path).second;   // BIGANN header: (N, D)

        // Belady bucket cache over (candidates, vectors) blocks, still part
        // of Step 2's setup -- no separate print, this is cheap.
        auto loader = [&](int32_t b) -> BucketWindowBlock {
            int32_t pos = pos_of[static_cast<size_t>(b)];
            uint32_t lo = offsets_new[static_cast<size_t>(pos)];
            uint32_t hi = offsets_new[static_cast<size_t>(pos) + 1];

            BucketWindowBlock blk;
            blk.pos = pos;
            blk.first_global_id = lo;
            blk.count = static_cast<int64_t>(hi - lo);
            blk.M_in = M_in;

            std::vector<int32_t> cand;
            knn_reader.read_rows(lo, hi, cand);
            blk.candidates = std::move(cand);

            std::vector<float> vecs;
            int32_t D = 0;
            read_dataset_chunk(dataset_path, dataset_ext, lo, static_cast<int64_t>(hi - lo), vecs, D);
            blk.D = D;
            blk.vectors = std::move(vecs);
            return blk;
        };

        bucket_order::BeladyBucketCache<BucketWindowBlock> cache(
            future, cache_mb << 20, block_size_bytes);

        std::cout << "=== Step 3: Pruning (windowed, " << omp_get_max_threads() << " threads) ===\n";
        auto t2 = Clock::now();

        graph_io::StreamingGraphWriter writer(out_path, N, output_degree);

        uint64_t total_lookups = 0, total_hits = 0;

        // position_scratch holds pointers, not copies -- see the fetch_one
        // lambda below for when a pointer is actually safe to hold.
        // position_scratch_owned is the defensive-copy backing store used
        // only during cache warm-up; reserved once to its max possible size
        // (center + every centroid-knn neighbor) so that filling it within
        // one position's fetch loop can never reallocate and dangle a
        // pointer already pushed into position_scratch for an earlier
        // fetch in the *same* position.
        std::vector<const BucketWindowBlock*> position_scratch;
        std::vector<BucketWindowBlock> position_scratch_owned;
        position_scratch_owned.reserve(static_cast<size_t>(ck.K) + 1);

        // Per-phase wall-clock breakdown of Step 3, since "Prune = 69.95s"
        // alone doesn't say whether the time went into cache fetch (disk +
        // Belady bookkeeping / defensive copies), the *serial* per-position
        // window rebuild, the OpenMP-parallel detour/select compute, or the
        // streaming write. fetch/window/write are inherently serial (one
        // thread, one position at a time) and sit squarely in the critical
        // path even with high thread counts, so they're worth isolating
        // explicitly rather than assuming the OMP region dominates just
        // because it's the only parallel part.
        prune_windowed::BucketWindow window(offsets_new.data(), n_buckets);

        int64_t n_copy_positions = 0;
        double t_fetch = 0, t_window_build = 0, t_compute = 0, t_write = 0;
        int64_t min_center_count = std::numeric_limits<int64_t>::max(), max_center_count = 0;

        // Batched fast path: 每个 bucket 只有 ~100 行 (SIFT1M)，逐 position
        // 开一个 80 线程的 OpenMP 区域时 fork/join + 负载不均占了 compute 的
        // 大头。整个窗口数据集 (candidates + vectors) 放得进 cache 预算时，
        // 一次取 kBatch 个 position 的窗口 (每个 position 仍是自己的
        // BucketWindow，剪枝语义不变)，再用一个并行区域处理这一批的全部行。放不进预算时走下面原来
        // 的逐 position 路径 (可能淘汰，需要防御性拷贝)。
        constexpr int32_t kBatch = 64;
        const size_t total_window_bytes = static_cast<size_t>(N) *
            (static_cast<size_t>(M_in) * sizeof(int32_t) + static_cast<size_t>(dataset_D_peek) * sizeof(float));
        const bool batched = total_window_bytes <= (cache_mb << 20);
        std::cout << "  mode=" << (batched ? "batched" : "per-position")
                  << " (window data " << total_window_bytes / 1e6 << " MB, cache budget " << cache_mb << " MB)\n";
        if (batched) {
            // 整份窗口数据都放得进 cache 预算 → 不再经过 Belady cache 逐 bucket
            // 读 (每个 bucket 两次 seek+read、数据集文件每次重新打开、再拷进
            // BucketWindowBlock)：两次大块顺序读把 candidates / vectors 整份读进
            // 来，每个窗口直接指向其中的切片 (bucket 在新 id 空间里连续)。
            // 内存占用 = total_window_bytes ≤ --cache-mb，跟 cache 装满时一样。
            auto tl0 = Clock::now();
            const int32_t D_all = dataset_D_peek;
            const size_t n_cand = static_cast<size_t>(N) * M_in;
            const size_t n_vec  = static_cast<size_t>(N) * D_all;
            const bool u8_data = (dataset_ext == ".u8bin" || dataset_ext == ".i8bin");
            // GPU 路径只读原始 u8 字节: 先判定，走 GPU 时不再分配 / 转换整份 float 向量
            // (SIFT10M: 5.1 GB 内存 + 一遍转换，GPU kernel 根本不用)
            const int32_t Wmax = static_cast<int32_t>(ck.K) + 1;
            const bool use_gpu = u8_data &&
                prune_windowed_gpu::available(N, M_in, D_all, n_buckets, Wmax, output_degree);
            std::unique_ptr<int32_t[]> all_cand_buf(new int32_t[n_cand]);   // 不初始化，pread 直接填
            std::unique_ptr<float[]>   all_vec_buf(use_gpu ? nullptr : new float[n_vec]);
            std::unique_ptr<uint8_t[]> raw_u8;   // u8 数据集时保留原始字节给 GPU kernel
            // vector_knn.bin: int64 N + int32 M header (12 B)，之后 (N, M) int32
            parallel_pread(vknn_path, static_cast<off_t>(sizeof(int64_t) + sizeof(int32_t)),
                           n_cand * sizeof(int32_t), all_cand_buf.get());
            if (u8_data) {
                // 跟 load::read_u8bin_chunk_to_f32 同样的转换 (按 uint8 读)，并行做
                raw_u8.reset(new uint8_t[n_vec]);
                parallel_pread(dataset_path, static_cast<off_t>(2 * sizeof(int32_t)), n_vec, raw_u8.get());
                if (!use_gpu) {
                    float* out = all_vec_buf.get();
                    #pragma omp parallel for schedule(static)
                    for (int64_t i = 0; i < static_cast<int64_t>(n_vec); ++i)
                        out[i] = static_cast<float>(raw_u8[static_cast<size_t>(i)]);
                }
            } else {
                std::vector<float> tmp;
                int32_t D_tmp = 0;
                read_dataset_chunk(dataset_path, dataset_ext, 0, N, tmp, D_tmp);
                if (D_tmp != D_all || tmp.size() != n_vec)
                    throw std::runtime_error("prune_windowed: bulk load size mismatch");
                std::memcpy(all_vec_buf.get(), tmp.data(), n_vec * sizeof(float));
            }
            const int32_t* all_cand = all_cand_buf.get();
            const float*   all_vec  = all_vec_buf.get();
            t_fetch += std::chrono::duration<double>(Clock::now() - tl0).count();
            struct Slice { int32_t pos; int64_t first, count; const int32_t* cand; const float* vec; };
            auto slice_of = [&](int32_t bucket) {
                const int32_t pos = pos_of[static_cast<size_t>(bucket)];
                const int64_t lo = offsets_new[static_cast<size_t>(pos)];
                const int64_t hi = offsets_new[static_cast<size_t>(pos) + 1];
                return Slice{pos, lo, hi - lo,
                             all_cand + static_cast<size_t>(lo) * M_in,
                             all_vec + static_cast<size_t>(lo) * D_all};
            };
            // ---- GPU fast path (u8 数据, M_in ≤ 64, 放得进显存): 同样的窗口语义，
            // 一个 warp 一个点 (见 prune_windowed_gpu.cu)。窗口 = 处理位置 p 自己
            // + 它的 centroid-knn 近邻桶所在的位置，按位置排序去重。
            if (use_gpu) {
                auto tg0 = Clock::now();
                std::vector<int32_t> win(static_cast<size_t>(n_buckets) * Wmax, std::numeric_limits<int32_t>::max());
                std::vector<int32_t> win_len(static_cast<size_t>(n_buckets), 0);
                for (int32_t pos = 0; pos < n_buckets; ++pos) {
                    int32_t* wp = win.data() + static_cast<size_t>(pos) * Wmax;
                    int32_t len = 0;
                    wp[len++] = pos;
                    for (int32_t nbk : adj[static_cast<size_t>(order[static_cast<size_t>(pos)])]) {
                        if (nbk < 0 || nbk >= n_buckets) continue;
                        if (len >= Wmax) throw std::runtime_error("prune_windowed: window larger than K+1");
                        wp[len++] = pos_of[static_cast<size_t>(nbk)];
                    }
                    std::sort(wp, wp + len);
                    len = static_cast<int32_t>(std::unique(wp, wp + len) - wp);
                    win_len[static_cast<size_t>(pos)] = len;
                }
                // 结果按行块经 pinned 暂存区直接流给 writer (不再先拷进一份 N*out_deg
                // 的 host buffer 再写)；写盘时间跟后续块的 D2H 重叠
                double t_sink = 0;
                prune_windowed_gpu::run(all_cand, raw_u8.get(), N, M_in, D_all,
                                        offsets_new.data(), n_buckets, win, win_len, Wmax,
                                        output_degree,
                                        [&](const uint32_t* rows, int64_t row0, int64_t n) {
                                            auto tw0 = Clock::now();
                                            writer.write_rows(rows, row0, n);
                                            t_sink += std::chrono::duration<double>(Clock::now() - tw0).count();
                                        });
                t_write += t_sink;
                t_compute += std::chrono::duration<double>(Clock::now() - tg0).count() - t_sink;
                min_center_count = max_center_count = 0;
                std::cout << "  mode=batched+GPU\n";
                goto pruned_done;
            }
            std::vector<prune_windowed::BucketWindow> windows(
                static_cast<size_t>(kBatch), prune_windowed::BucketWindow(offsets_new.data(), n_buckets));
            std::vector<Slice> centers(static_cast<size_t>(kBatch));
            std::vector<int64_t> row_start(static_cast<size_t>(kBatch) + 1);
            std::vector<uint32_t> row_block;
            for (int32_t pos0 = 0; pos0 < n_buckets; pos0 += kBatch) {
                const int32_t nb = std::min(kBatch, n_buckets - pos0);

                auto tf0 = Clock::now();
                for (int32_t j = 0; j < nb; ++j) {
                    const int32_t pos = pos0 + j;
                    const int32_t b = order[static_cast<size_t>(pos)];
                    auto& w = windows[static_cast<size_t>(j)];
                    w.clear();
                    (void)pos;
                    auto add = [&](int32_t id) {
                        Slice sl = slice_of(id);
                        w.add_bucket(sl.pos, sl.cand, M_in, sl.vec, D_all);
                        return sl;
                    };
                    centers[static_cast<size_t>(j)] = add(b);   // center
                    for (int32_t nbk : adj[static_cast<size_t>(b)]) {
                        if (nbk < 0 || nbk >= n_buckets) continue;
                        add(nbk);
                    }
                }
                t_fetch += std::chrono::duration<double>(Clock::now() - tf0).count();   // fetch + window build

                row_start[0] = 0;
                for (int32_t j = 0; j < nb; ++j) {
                    const Slice& c = centers[static_cast<size_t>(j)];
                    // 各 position 的 center bucket 在新 id 空间里首尾相接
                    if (c.first != static_cast<int64_t>(offsets_new[static_cast<size_t>(pos0 + j)]))
                        throw std::runtime_error("prune_windowed: center bucket not at its offset");
                    min_center_count = std::min(min_center_count, c.count);
                    max_center_count = std::max(max_center_count, c.count);
                    row_start[static_cast<size_t>(j) + 1] = row_start[static_cast<size_t>(j)] + c.count;
                }
                const int64_t total_rows = row_start[static_cast<size_t>(nb)];
                row_block.resize(static_cast<size_t>(total_rows) * output_degree);

                auto tc0 = Clock::now();
                #pragma omp parallel for schedule(dynamic, 16) reduction(+:total_lookups,total_hits)
                for (int64_t r = 0; r < total_rows; ++r) {
                    const int32_t j = static_cast<int32_t>(
                        std::upper_bound(row_start.begin(), row_start.begin() + nb + 1, r) - row_start.begin()) - 1;
                    const Slice& center = centers[static_cast<size_t>(j)];
                    const prune_windowed::BucketWindow& w = windows[static_cast<size_t>(j)];
                    const int64_t lk = r - row_start[static_cast<size_t>(j)];
                    uint32_t a_id = static_cast<uint32_t>(center.first + lk);
                    const int32_t* raw = center.cand + lk * M_in;

                    for (int32_t k = 0; k < M_in; ++k) {
                        if (raw[k] < 0) continue;
                        total_lookups++;
                        prune_windowed::PointData pd;
                        if (w.get(static_cast<uint32_t>(raw[k]), pd)) total_hits++;
                    }

                    prune_windowed::prune_one_point(
                        a_id, raw, M_in, D_all, w, output_degree,
                        row_block.data() + static_cast<size_t>(r) * output_degree);
                }
                t_compute += std::chrono::duration<double>(Clock::now() - tc0).count();

                auto tr0 = Clock::now();
                writer.write_rows(row_block.data(), offsets_new[static_cast<size_t>(pos0)], total_rows);
                t_write += std::chrono::duration<double>(Clock::now() - tr0).count();
            }
        }

pruned_done:
        for (int32_t pos = 0; !batched && pos < n_buckets; ++pos) {
            int32_t b = order[static_cast<size_t>(pos)];

            // Once the cache holds every bucket (resident_count() ==
            // n_buckets), BeladyBucketCache::make_room() can never run
            // again -- it only runs on a cache *miss*, and there are no
            // bucket ids left to miss on. So every slot_[id] reference
            // handed back by get() from this point on is stable for the
            // rest of the program's lifetime, and the copy-out-immediately
            // contract (needed in general, since a later miss *can* evict
            // an earlier hit within the same position while still warming
            // up) becomes unnecessary -- on the actual run that motivated
            // this file (cache-mb big enough to hold the whole dataset),
            // this fast path covers all but the first handful of positions
            // and turns ~768MB of redundant copying (every bucket gets
            // copied once per window that references it, i.e. ~(1+K)x
            // over the whole run) into plain pointer handoffs.
            // Optimistic pointer handoff: a reference from get() can only be
            // invalidated by an eviction (a later miss at the same position).
            // Fetch the whole window as pointers first; only if the cache's
            // eviction counter moved during this position redo it with
            // defensive copies. Previously copies were made whenever the cache
            // wasn't *fully* resident -- i.e. for almost every position, since
            // the last bucket only becomes resident near the end -- ~(K+1)x
            // the dataset in memcpy even when nothing was ever evicted.
            auto tf0 = Clock::now();
            auto fetch_window = [&](bool copy) {
                position_scratch.clear();
                position_scratch_owned.clear();
                auto fetch_one = [&](int32_t id) {
                    const BucketWindowBlock& blk = cache.get(pos, id, loader);
                    if (!copy) {
                        position_scratch.push_back(&blk);
                    } else {
                        position_scratch_owned.push_back(blk);  // defensive copy
                        position_scratch.push_back(&position_scratch_owned.back());
                    }
                };
                fetch_one(b);   // index 0 = center
                for (int32_t nb : adj[static_cast<size_t>(b)]) {
                    if (nb < 0 || nb >= n_buckets) continue;
                    fetch_one(nb);
                }
            };
            const uint64_t ev0 = cache.evictions();
            fetch_window(false);
            if (cache.evictions() != ev0) { fetch_window(true); ++n_copy_positions; }
            t_fetch += std::chrono::duration<double>(Clock::now() - tf0).count();

            // Bucket-granular window (one slot per bucket, reused across
            // positions): replaces the old per-position unordered_map over
            // every window row, whose serial build + teardown cost more than
            // the parallel compute itself.
            auto tw0 = Clock::now();
            window.clear();
            for (const auto* blkp : position_scratch)
                window.add_bucket(blkp->pos, blkp->candidates.data(), blkp->M_in,
                                  blkp->vectors.data(), blkp->D);
            t_window_build += std::chrono::duration<double>(Clock::now() - tw0).count();

            const BucketWindowBlock& center = *position_scratch[0];
            min_center_count = std::min(min_center_count, center.count);
            max_center_count = std::max(max_center_count, center.count);
            std::vector<uint32_t> row_block(static_cast<size_t>(center.count) * output_degree);

            // static (not dynamic,256): center.count is a single bucket's
            // row count, which averages N/n_buckets and is typically far
            // smaller than a 256-row dynamic chunk (e.g. 100 for Sift1M's
            // 10k buckets) -- with chunk > count, dynamic scheduling hands
            // the *entire* bucket to one thread while the other
            // omp_get_max_threads()-1 threads sit idle through the
            // fork/join, repeated once per bucket (n_buckets times). static
            // always spreads a bucket's rows across all threads instead.
            auto tc0 = Clock::now();
            #pragma omp parallel for schedule(static) reduction(+:total_lookups,total_hits)
            for (int64_t lk = 0; lk < center.count; ++lk) {
                uint32_t a_id = static_cast<uint32_t>(center.first_global_id + lk);
                const int32_t* raw = center.candidates.data() + lk * center.M_in;

                for (int32_t k = 0; k < M_in; ++k) {
                    if (raw[k] < 0) continue;
                    total_lookups++;
                    prune_windowed::PointData pd;
                    if (window.get(static_cast<uint32_t>(raw[k]), pd)) total_hits++;
                }

                prune_windowed::prune_one_point(
                    a_id, raw, M_in, center.D, window, output_degree,
                    row_block.data() + static_cast<size_t>(lk) * output_degree);
            }
            t_compute += std::chrono::duration<double>(Clock::now() - tc0).count();

            auto tr0 = Clock::now();
            writer.write_rows(row_block.data(), center.first_global_id, center.count);
            t_write += std::chrono::duration<double>(Clock::now() - tr0).count();
        }

        if (writer.rows_written() != N)
            throw std::runtime_error("prune_windowed: wrote " + std::to_string(writer.rows_written()) +
                                     " rows, expected " + std::to_string(N));

        double t_prune = std::chrono::duration<double>(Clock::now() - t2).count();
        double pct_hit = total_lookups ? 100.0 * total_hits / total_lookups : 0.0;
        std::cout << "  [Stats] candidate lookups: " << total_lookups
                  << ", in-window hits: " << total_hits
                  << " (" << std::fixed << std::setprecision(2) << pct_hit << "%)\n";
        std::cout << "  [Stats] cache: resident_count=" << cache.resident_count()
                  << " used_bytes=" << cache.used_bytes() / 1e6 << " MB"
                  << " evictions=" << cache.evictions()
                  << " copy_positions=" << n_copy_positions << "\n";
        std::cout << "  [Stats] bucket size: min=" << min_center_count
                  << " max=" << max_center_count
                  << " avg=" << (n_buckets ? static_cast<double>(N) / n_buckets : 0.0) << "\n";
        std::cout << "  [Timing] breakdown: fetch=" << std::fixed << std::setprecision(3) << t_fetch
                  << "s window_build=" << t_window_build
                  << "s compute(parallel)=" << t_compute
                  << "s write=" << t_write << "s (sum=" << (t_fetch + t_window_build + t_compute + t_write)
                  << "s vs wall=" << t_prune << "s)\n";
        std::cout << "  [Timing] Prune = " << t_prune << "s\n";

        if (save_npy) {
            std::vector<uint32_t> out_graph;
            int64_t N2 = 0; int32_t K2 = 0;
            graph_io::read_forward_graph(out_path, out_graph, N2, K2);
            std::vector<int64_t> g64(out_graph.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < out_graph.size(); ++i)
                g64[i] = static_cast<int64_t>(out_graph[i]);
            auto p = fs::path(out_path);
            p.replace_extension(".npy");
            load::write_npy_int64_2d(p.string(), g64.data(), N2, K2);
            std::cout << "  Wrote " << p.string() << " (npy)\n";
        }

        double total = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cout << "\n=== Done ===\n";
        std::cout << "Total: " << std::fixed << std::setprecision(3) << total << "s\n";
        std::cout << "Output: " << out_path << " (feed into optimize_chunked -g <this> "
                  << "--bucket-offsets " << bo_path << ")\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
