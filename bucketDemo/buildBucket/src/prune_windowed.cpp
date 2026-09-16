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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "bucket_order.hpp"
#include "graph_io.hpp"
#include "load.hpp"
#include "prune_windowed_core.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------
// One bucket's worth of window data: candidate rows + vectors, contiguous
// in the reordered id space. This is the BeladyBucketCache's BucketT.
// ---------------------------------------------------------------------
struct BucketWindowBlock {
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

        // Belady bucket cache over (candidates, vectors) blocks, still part
        // of Step 2's setup -- no separate print, this is cheap.
        auto loader = [&](int32_t b) -> BucketWindowBlock {
            int32_t pos = pos_of[static_cast<size_t>(b)];
            uint32_t lo = offsets_new[static_cast<size_t>(pos)];
            uint32_t hi = offsets_new[static_cast<size_t>(pos) + 1];

            BucketWindowBlock blk;
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
        std::vector<BucketWindowBlock> position_scratch;  // owned copies for this position only

        for (int32_t pos = 0; pos < n_buckets; ++pos) {
            int32_t b = order[static_cast<size_t>(pos)];

            // Fetch center + neighbors one at a time, copying each out of
            // the cache immediately -- BeladyBucketCache::get()'s contract
            // says a reference is only valid until the *next* get() call
            // for a different id at this position, so we can't hold
            // multiple such references across calls.
            position_scratch.clear();
            position_scratch.push_back(cache.get(pos, b, loader));   // index 0 = center
            for (int32_t nb : adj[static_cast<size_t>(b)]) {
                if (nb < 0 || nb >= n_buckets) continue;
                position_scratch.push_back(cache.get(pos, nb, loader));
            }

            prune_windowed::WindowLookup window;
            for (const auto& blk : position_scratch) {
                for (int64_t k = 0; k < blk.count; ++k) {
                    window.add(static_cast<uint32_t>(blk.first_global_id + k),
                              blk.candidates.data() + k * blk.M_in,
                              blk.vectors.data() + k * blk.D);
                }
            }

            const BucketWindowBlock& center = position_scratch[0];
            std::vector<uint32_t> row_block(static_cast<size_t>(center.count) * output_degree);

            #pragma omp parallel for schedule(dynamic, 256) reduction(+:total_lookups,total_hits)
            for (int64_t lk = 0; lk < center.count; ++lk) {
                uint32_t a_id = static_cast<uint32_t>(center.first_global_id + lk);
                const int32_t* raw = center.candidates.data() + lk * center.M_in;

                for (int32_t k = 0; k < M_in; ++k) {
                    if (raw[k] < 0) continue;
                    total_lookups++;
                    if (window.find(static_cast<uint32_t>(raw[k]))) total_hits++;
                }

                prune_windowed::prune_one_point(
                    a_id, raw, M_in, center.D, window, output_degree,
                    row_block.data() + static_cast<size_t>(lk) * output_degree);
            }

            writer.write_rows(row_block.data(), center.first_global_id, center.count);
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
                  << " used_bytes=" << cache.used_bytes() / 1e6 << " MB\n";
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
