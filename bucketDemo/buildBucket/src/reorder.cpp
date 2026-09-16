// reorder.cpp
// 把数据集 + per-vector KNN 图按 bucket 重排成连续 ID，让每个 bucket 占用一段
// 连续 ID 区间。这样 optimize_chunked --method C/D 的 chunk 边界就能对齐 bucket，
// 更多边落在 chunk 内 —— 但前提是"新 ID 空间里相邻的 bucket，在特征空间里也相
// 邻"，这正是 --centroid-knn 要解决的：不给它时退化为原始 bucket 编号顺序（跟
// centroid 生成顺序有关，没有空间局部性保证）；给了它，就用 DiskJoin 风格的
// task ordering（见 bucket_order.hpp）算一个有空间局部性的 bucket 处理顺序。
//
// 输入（来自 bucket.cu 的输出目录）：
//   bucket_index.bin           bucket 元信息 (n_buckets, offset, count)
//   bucket_data.bin            每个 bucket 的原始 int32 全局 ID
//   data.fbin / .u8bin / ...   原始数据集
//   [vector_knn.bin]           可选，per-vector KNN 图
//   [centroid_knn.bin]         可选（bucket.cu --reorder 时会写），centroid KNN
//                              图，用于推导有空间局部性的 bucket 处理顺序
//
// 输出（写入指定输出目录）：
//   data_reordered.<ext>        重排后的数据集（保留原始格式 / 元素类型）
//   bucket_offsets.bin          method C/D/E 和 prune_windowed 用的 bucket 边界（新 ID 空间）
//   bucket_process_order.bin    order[pos] = 原始 bucket id，给 prune_windowed 用来精确复原
//                                "新 ID 区间 [offsets[pos],offsets[pos+1]) 属于哪个原始 bucket"，
//                                不用重新跑一遍 compute_bucket_processing_order（避免 --order-window
//                                两边对不上导致行映射悄悄错位）
//   perm.bin                    uint32[N]，perm[old_id] = new_id
//   inverse_perm.bin            uint32[N]，inverse_perm[new_id] = old_id
//   [vector_knn_reordered.bin]  可选，重排 + 邻居 ID 重映射后的 KNN 图
//
// bucket_offsets.bin 格式（与 optimize_chunked 对应）:
//   int32_t  n_buckets
//   uint32_t offsets[n_buckets + 1]
//
// 搜索阶段：CAGRA 返回的是新 ID，需要 inverse_perm[new_id] 翻译回原始 ID。

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <filesystem>
#include <cstring>
#include <numeric>
#include <omp.h>

#include <boost/program_options.hpp>

#include "load.hpp"
#include "bucket_order.hpp"
#include "graph_io.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

// ---------- Helpers ----------

// 数据文件每行字节数（按扩展名推断元素类型）
static size_t row_bytes_from_ext(const std::string& ext, int32_t D) {
    if (ext == ".fbin" || ext == ".bin")  return static_cast<size_t>(D) * sizeof(float);
    if (ext == ".u8bin" || ext == ".i8bin") return static_cast<size_t>(D);
    if (ext == ".ibin")                    return static_cast<size_t>(D) * sizeof(int32_t);
    throw std::runtime_error("Unsupported dataset extension: " + ext);
}

// 读 bucket_index.bin: (n_centroids, N) header + (offset, count) 列表
struct BucketIndex {
    int64_t n_centroids;
    int64_t N_total;
    std::vector<int64_t> offsets;   // 每个 bucket 在 bucket_data.bin 中的字节偏移
    std::vector<int32_t> counts;    // 每个 bucket 的点数
};

static BucketIndex read_bucket_index(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open: " + path);

    BucketIndex bi;
    in.read(reinterpret_cast<char*>(&bi.n_centroids), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&bi.N_total),     sizeof(int64_t));
    if (!in.good() || bi.n_centroids <= 0 || bi.N_total <= 0)
        throw std::runtime_error("Invalid bucket_index header");

    bi.offsets.resize(bi.n_centroids);
    bi.counts.resize(bi.n_centroids);
    for (int64_t b = 0; b < bi.n_centroids; ++b) {
        in.read(reinterpret_cast<char*>(&bi.offsets[b]), sizeof(int64_t));
        in.read(reinterpret_cast<char*>(&bi.counts[b]),  sizeof(int32_t));
    }
    if (!in.good())
        throw std::runtime_error("Failed to read bucket_index payload");
    return bi;
}

// 从 bucket_data.bin 读所有 bucket 的点 ID（按 bucket 顺序）
static std::vector<int32_t> read_bucket_data(const std::string& path, int64_t total_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<int32_t> ids(total_count);
    in.read(reinterpret_cast<char*>(ids.data()),
            static_cast<std::streamsize>(total_count * sizeof(int32_t)));
    if (!in.good())
        throw std::runtime_error("Failed to read bucket_data");
    return ids;
}

// centroid_knn.bin / vector_knn.bin readers + writer and the generic
// write_vector_bin now live in graph_io.hpp (shared with optimize_chunked.cu
// and prune_windowed.cpp) as graph_io::CentroidKnn/read_centroid_knn,
// graph_io::VectorKnn/read_vector_knn/write_vector_knn, and
// graph_io::write_vector_bin.

// ---------- Main ----------

int main(int argc, char** argv) {
    try {
        po::options_description desc("Reorder bucket pipeline outputs by bucket layout");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i",         po::value<std::string>()->required(),
                "Original dataset (.fbin/.bin/.u8bin/.i8bin/.ibin)")
            ("bucket-index,x",  po::value<std::string>()->required(),
                "bucket_index.bin (from bucket.cu)")
            ("bucket-data,d",   po::value<std::string>()->required(),
                "bucket_data.bin (from bucket.cu)")
            ("vector-knn,k",    po::value<std::string>()->default_value(""),
                "vector_knn.bin (optional; will be remapped to new ID space)")
            ("centroid-knn,c",  po::value<std::string>()->default_value(""),
                "centroid_knn.bin (optional, from bucket.cu --reorder). When given, buckets "
                "are laid out in a DiskJoin-style spatial processing order instead of raw "
                "bucket-index order, so optimize_chunked --method C/D chunk boundaries "
                "actually land on spatially coherent groups of buckets.")
            ("order-window",    po::value<int32_t>()->default_value(0),
                "Sliding-window size for --centroid-knn's bucket ordering (0 = auto: 4*K)")
            ("output-dir,o",    po::value<std::string>()->required(),
                "Output directory");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);

        const std::string input_path = vm["input"].as<std::string>();
        const std::string bidx_path  = vm["bucket-index"].as<std::string>();
        const std::string bdat_path  = vm["bucket-data"].as<std::string>();
        const std::string vknn_path  = vm["vector-knn"].as<std::string>();
        const std::string cknn_path  = vm["centroid-knn"].as<std::string>();
        const int32_t order_window_arg = vm["order-window"].as<int32_t>();
        const std::string out_dir    = vm["output-dir"].as<std::string>();

        fs::create_directories(out_dir);

        using Clock = std::chrono::steady_clock;
        auto t_start = Clock::now();

        // ============================================================
        // Step 1: Load bucket structure → derive permutation
        // ============================================================
        std::cout << "=== Step 1: Reading bucket structure ===\n";
        auto t1 = Clock::now();
        auto bi = read_bucket_index(bidx_path);
        const int64_t N = bi.N_total;
        const int64_t n_buckets = bi.n_centroids;

        // 累计 bucket 总点数（不一定等于 N，可能有未分配点；但通常 = N）
        int64_t total_in_buckets = 0;
        for (auto c : bi.counts) total_in_buckets += c;
        std::cout << "  N=" << N << " n_buckets=" << n_buckets
                  << " total_in_buckets=" << total_in_buckets << "\n";

        // 读所有 bucket 的原始 ID（按 bucket 顺序）
        auto bucket_data = read_bucket_data(bdat_path, total_in_buckets);

        // bucket 处理顺序：给定 --centroid-knn 时，用 DiskJoin 风格 task ordering
        // (bucket_order.hpp) 算一个让相邻 bucket 在特征空间也相邻的顺序；否则退化
        // 为原始 bucket 编号顺序（旧行为，chunk 边界不保证贴合空间分布）。
        std::vector<int32_t> bucket_process_order(n_buckets);
        if (!cknn_path.empty()) {
            std::cout << "  Loading centroid_knn from " << cknn_path << " ...\n";
            auto ck = graph_io::read_centroid_knn(cknn_path);
            if (ck.n_centroids != n_buckets)
                throw std::runtime_error("centroid_knn n_centroids (" +
                    std::to_string(ck.n_centroids) + ") != bucket-index n_buckets (" +
                    std::to_string(n_buckets) + ")");
            int32_t order_window = (order_window_arg > 0)
                ? order_window_arg
                : std::max<int32_t>(4 * ck.K, 16);
            bucket_process_order = bucket_order::compute_bucket_processing_order(
                bucket_order::adjacency_from_flat_graph(ck.graph.data(), ck.n_centroids, ck.K),
                order_window);
            std::cout << "  Bucket processing order: DiskJoin task ordering (K=" << ck.K
                      << ", window=" << order_window << ")\n";
        } else {
            std::iota(bucket_process_order.begin(), bucket_process_order.end(), 0);
            std::cout << "  [Warn] --centroid-knn not given: falling back to raw bucket-index "
                         "order (chunk boundaries won't be spatially aligned)\n";
        }

        // 推导 perm / inverse_perm
        // 新 ID 空间按 bucket_process_order 排列：
        // bucket_process_order[0] 占 [0, counts[.]) → bucket_process_order[1] 占下一段 → ...
        std::vector<uint32_t> perm(N, 0xFFFFFFFFu);
        std::vector<uint32_t> inverse_perm(total_in_buckets);
        std::vector<uint32_t> offsets_new(n_buckets + 1, 0);

        // bucket_data 在文件中是按 bucket 0,1,2,... 顺序紧凑排列的，但我们按
        // bucket_process_order 给每个 bucket 分配新 ID 区间。
        for (int64_t pos = 0; pos < n_buckets; ++pos) {
            int64_t b = bucket_process_order[pos];
            uint32_t base_new = offsets_new[pos];
            int64_t  base_old = bi.offsets[b] / sizeof(int32_t);
            for (int32_t k = 0; k < bi.counts[b]; ++k) {
                int32_t old_id = bucket_data[base_old + k];
                if (old_id < 0 || static_cast<int64_t>(old_id) >= N)
                    throw std::runtime_error("Out-of-range id in bucket_data: " +
                                             std::to_string(old_id));
                uint32_t new_id = base_new + k;
                if (perm[old_id] != 0xFFFFFFFFu)
                    throw std::runtime_error("Point " + std::to_string(old_id) +
                                             " appears in multiple buckets");
                perm[old_id] = new_id;
                inverse_perm[new_id] = static_cast<uint32_t>(old_id);
            }
            offsets_new[pos + 1] = base_new + static_cast<uint32_t>(bi.counts[b]);
        }

        // 检查未被分配的点（perm 仍是 sentinel）
        int64_t unassigned = 0;
        for (int64_t i = 0; i < N; ++i)
            if (perm[i] == 0xFFFFFFFFu) unassigned++;
        if (unassigned > 0) {
            std::cout << "  [Warn] " << unassigned
                      << " points are not assigned to any bucket. "
                      << "They keep sentinel perm = 0xFFFFFFFF and are NOT in the reordered dataset.\n";
        }

        std::cout << "  Permutation derived ["
                  << std::chrono::duration<double>(Clock::now() - t1).count() << "s]\n";

        // ============================================================
        // Step 2: Reorder dataset (generic byte-level)
        // ============================================================
        std::cout << "=== Step 2: Reordering dataset ===\n";
        auto t2 = Clock::now();

        std::string ext = fs::path(input_path).extension().string();

        // 读 header
        std::ifstream din(input_path, std::ios::binary);
        if (!din.is_open()) throw std::runtime_error("Cannot open: " + input_path);
        int32_t fileN = 0, D = 0;
        din.read(reinterpret_cast<char*>(&fileN), sizeof(int32_t));
        din.read(reinterpret_cast<char*>(&D),     sizeof(int32_t));
        if (fileN != static_cast<int32_t>(N))
            throw std::runtime_error("Dataset N (" + std::to_string(fileN) +
                                     ") != bucket-index N (" + std::to_string(N) + ")");

        const size_t row_bytes  = row_bytes_from_ext(ext, D);
        const size_t total_rows = total_in_buckets;   // 输出只包含被分配的点
        std::cout << "  D=" << D << " row_bytes=" << row_bytes
                  << " output_rows=" << total_rows
                  << " (" << total_rows * row_bytes / 1e9 << " GB)\n";

        // 读全部 payload 到内存（假设主机内存足够）
        std::vector<char> raw_in(static_cast<size_t>(N) * row_bytes);
        din.read(raw_in.data(),
                 static_cast<std::streamsize>(raw_in.size()));
        if (!din.good())
            throw std::runtime_error("Failed to read dataset payload");
        din.close();

        std::vector<char> raw_out(total_rows * row_bytes);
        #pragma omp parallel for schedule(static)
        for (size_t new_id = 0; new_id < total_rows; ++new_id) {
            uint32_t old_id = inverse_perm[new_id];
            std::memcpy(raw_out.data() + new_id * row_bytes,
                        raw_in.data()  + (size_t)old_id * row_bytes,
                        row_bytes);
        }
        { auto _drop = std::move(raw_in); }

        // 写出（保留原始扩展名）
        std::string out_data_path = out_dir + "/data_reordered" + ext;
        std::ofstream dout(out_data_path, std::ios::binary);
        int32_t outN = static_cast<int32_t>(total_rows);
        dout.write(reinterpret_cast<const char*>(&outN), sizeof(int32_t));
        dout.write(reinterpret_cast<const char*>(&D),    sizeof(int32_t));
        dout.write(raw_out.data(), static_cast<std::streamsize>(raw_out.size()));
        dout.close();
        { auto _drop = std::move(raw_out); }

        std::cout << "  Wrote " << out_data_path
                  << " [" << std::chrono::duration<double>(Clock::now() - t2).count() << "s]\n";

        // ============================================================
        // Step 3: Reorder vector_knn.bin (optional)
        // ============================================================
        if (!vknn_path.empty()) {
            std::cout << "=== Step 3: Reordering vector_knn.bin ===\n";
            auto t3 = Clock::now();
            auto vk = graph_io::read_vector_knn(vknn_path);
            if (vk.N != N)
                throw std::runtime_error("vector_knn N != dataset N");

            std::vector<int32_t> new_graph(static_cast<size_t>(total_rows) * vk.M);
            int64_t bad_neighbors = 0;
            #pragma omp parallel for schedule(static) reduction(+:bad_neighbors)
            for (size_t new_id = 0; new_id < total_rows; ++new_id) {
                uint32_t old_id = inverse_perm[new_id];
                const int32_t* src = vk.graph.data() + (size_t)old_id * vk.M;
                int32_t*       dst = new_graph.data() + new_id * vk.M;
                for (int k = 0; k < vk.M; ++k) {
                    int32_t old_nb = src[k];
                    if (old_nb < 0) {
                        dst[k] = -1;  // 保留 padding
                    } else if (old_nb >= static_cast<int32_t>(N)
                               || perm[old_nb] == 0xFFFFFFFFu) {
                        // 邻居指向未分配点（极少见）→ 标记为 padding
                        dst[k] = -1;
                        bad_neighbors++;
                    } else {
                        dst[k] = static_cast<int32_t>(perm[old_nb]);
                    }
                }
            }
            if (bad_neighbors > 0) {
                std::cout << "  [Warn] " << bad_neighbors
                          << " neighbor entries pointed to unassigned points → set to -1\n";
            }

            std::string out_knn_path = out_dir + "/vector_knn_reordered.bin";
            graph_io::write_vector_knn(out_knn_path, new_graph,
                             static_cast<int64_t>(total_rows), vk.M);
            std::cout << "  Wrote " << out_knn_path
                      << " (" << new_graph.size() * sizeof(int32_t) / 1e6 << " MB) ["
                      << std::chrono::duration<double>(Clock::now() - t3).count() << "s]\n";
        } else {
            std::cout << "=== Step 3: vector_knn.bin not provided, skipping ===\n";
        }

        // ============================================================
        // Step 4: Write bucket_offsets / perm / inverse_perm
        // ============================================================
        std::cout << "=== Step 4: Writing metadata ===\n";
        auto t4 = Clock::now();

        // bucket_offsets.bin
        {
            std::string p = out_dir + "/bucket_offsets.bin";
            graph_io::write_bucket_offsets(p, offsets_new);
            std::cout << "  Wrote " << p << " (n_buckets=" << n_buckets << ")\n";
        }

        // bucket_process_order.bin: order[pos] = original bucket id at
        // position pos. Written unconditionally (even the --centroid-knn-less
        // identity-order fallback) so downstream tools (prune_windowed) can
        // always recover "which original bucket owns new-id range
        // [offsets[pos], offsets[pos+1])" by reading this file instead of
        // recomputing compute_bucket_processing_order a second time -- that
        // recomputation would silently produce a different order (and thus
        // corrupt every row mapping) if its --order-window doesn't exactly
        // match what was used here.
        {
            std::string p = out_dir + "/bucket_process_order.bin";
            graph_io::write_bucket_process_order(p, bucket_process_order);
            std::cout << "  Wrote " << p << "\n";
        }

        // perm.bin / inverse_perm.bin
        graph_io::write_vector_bin(out_dir + "/perm.bin", perm);
        graph_io::write_vector_bin(out_dir + "/inverse_perm.bin", inverse_perm);
        std::cout << "  Wrote perm.bin (" << perm.size() * 4 / 1e6 << " MB), "
                  << "inverse_perm.bin (" << inverse_perm.size() * 4 / 1e6 << " MB)\n";

        std::cout << "  ["
                  << std::chrono::duration<double>(Clock::now() - t4).count() << "s]\n";

        double total = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cout << "\n=== Done ===\n";
        std::cout << "Total: " << std::fixed << std::setprecision(3) << total << "s\n";
        std::cout << "Output dir: " << out_dir << "\n";
        std::cout << "  data_reordered" << ext << "        — reordered dataset\n";
        if (!vknn_path.empty())
            std::cout << "  vector_knn_reordered.bin    — remapped KNN graph\n";
        std::cout << "  bucket_offsets.bin          — for optimize_chunked --method C/D/E, prune_windowed\n";
        std::cout << "  bucket_process_order.bin    — for prune_windowed (bucket id at each position)\n";
        std::cout << "  perm.bin / inverse_perm.bin — id translation tables\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
