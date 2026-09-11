#include <iostream>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <limits>
#include <iomanip>
#include <atomic>
#include <unordered_set>
#include <cstring>
#include <type_traits>
#include <omp.h>
#include <chrono>
#include <future>
#include <memory>

// Boost
#include <boost/program_options.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

// RAFT
#include <raft/core/resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/distance/distance.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/neighbors/detail/nn_descent.cuh>
#include <raft/neighbors/detail/cagra/graph_core.cuh>
#include <raft/neighbors/cagra.cuh>
#include <raft/random/rng.cuh>
#include <raft/matrix/select_k.cuh>
#include <rmm/device_uvector.hpp>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>

// Local headers
#include "utils.hpp"
#include "load.hpp"
#include "bucket_build.cuh"
#include "bucket_order.hpp"

namespace po = boost::program_options;
using namespace bucket;

// ============== Phase 1: LoadConfig & Memory Management ==============

/**
 * Configuration for flexible data loading and GPU KMeans++ centroid generation
 * Based on user requirements and DiskANN design patterns
 */
struct LoadConfig {
    // Memory limits
    size_t cpu_limit_bytes;        // CPU memory limit, default 16GB
    size_t gpu_limit_bytes;        // GPU memory limit, default 0 (auto-detect via cudaMemGetInfo)

    // Sampling and Centroid parameters
    float sample_rate;             // Sampling ratio [0, 1], default 0.1 (10%)
    float centroid_ratio;          // Centroid ratio relative to sampled data, default 0.01 (1%)

    // PQ parameters (DiskANN Per-Subspace Quantization style)
    uint32_t pq_bits_start;        // PQ starting bits, default 8
    uint32_t pq_bits_min;          // PQ minimum bits, default 4 (ensures compression)
    uint32_t pq_dim;               // PQ dimension, default 0 (auto = D/4)
    float pq_train_fraction;       // PQ training data fraction, default 0.05 (5%)
    uint32_t pq_train_max_rows;    // PQ training data row limit, default 65536

    // Control
    bool use_pq;                   // Force PQ quantization (skip judgment)
    uint32_t seed;                 // Random seed

    // Safety
    float gpu_safety_margin;       // Reserve fraction of GPU memory, default 0.1 (10%)

    // KNN graph parameters
    uint32_t knn_k;                // KNN graph degree for centroids, range [1, 1000], default 32

    static constexpr uint32_t MAX_KNN_K = 1000;

    // Constructor with safe defaults
    LoadConfig() :
        cpu_limit_bytes(16UL << 30),    // 16GB
        gpu_limit_bytes(0),             // Auto-detect
        sample_rate(0.1f),
        centroid_ratio(0.01f),
        pq_bits_start(8),
        pq_bits_min(4),
        pq_dim(0),
        pq_train_fraction(0.05f),
        pq_train_max_rows(65536),
        use_pq(false),
        seed(42),
        gpu_safety_margin(0.1f),
        knn_k(32)
    {}
};

/**
 * Auto-detect GPU memory if gpu_limit_bytes == 0
 * Uses cudaMemGetInfo to query actual available GPU memory
 */
void init_gpu_limit_if_needed(LoadConfig& config) {
    if (config.gpu_limit_bytes == 0) {  // 0 means auto-detect
        size_t free_bytes, total_bytes;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        // Use 95% of available memory, reserve 5% for system
        config.gpu_limit_bytes = static_cast<size_t>(free_bytes * 0.95);
        std::cout << "Auto-detected GPU memory: " << std::fixed << std::setprecision(2)
                  << total_bytes / 1e9 << " GB total, "
                  << config.gpu_limit_bytes / 1e9 << " GB available\n";
    }
}

/**
 * Memory requirement estimation structure
 * Key insight: Must include KMeans++ working space (distance matrix)
 */
struct MemoryEstimate {
    // Raw dataset
    size_t raw_data_bytes;
    int64_t raw_data_rows;

    // KMeans working space (CRITICAL!)
    // Distance matrix: n_trials × n_samples × sizeof(float)
    // where n_trials = 2 + ceil(log(n_centroids))
    size_t kmeans_working_bytes;
    int32_t kmeans_n_trials;

    // Sampled data (when sampling needed)
    size_t sampled_data_bytes;
    int64_t sampled_data_rows;

    // PQ related (when PQ needed)
    size_t pq_train_data_bytes;
    int64_t pq_train_rows;
    size_t pq_codebook_bytes;
    size_t pq_encoded_data_bytes;
    uint32_t final_pq_bits;

    // Centroid data
    size_t centroid_data_bytes;
    int64_t centroid_rows;

    // Decision flags
    bool fits_in_gpu;              // Full dataset fits in GPU (Step 1)
    bool sampled_fits_with_kmeans; // Sampled data + KMeans workspace fits (Step 2)
    bool need_pq;                  // PQ quantization required (Step 3)
};

/**
 * Estimate memory requirements for different scenarios
 *
 * Three-stage judgment flow (as per user requirement):
 * Step 1: Check if full dataset fits in GPU (with KMeans working space)
 * Step 2: Check if sampled data fits in GPU (with KMeans working space)
 * Step 3: Check if PQ quantized data fits (with dynamic pq_bits adjustment)
 */
MemoryEstimate estimate_memory_requirement(
    int64_t N,
    int64_t D,
    int64_t n_centroids,
    const LoadConfig& config,
    const std::string& dtype = "float32") {

    MemoryEstimate est{};

    est.raw_data_rows = N;
    size_t dtype_size = (dtype == "float32") ? 4 : ((dtype == "uint8" || dtype == "int8") ? 1 : 4);
    est.raw_data_bytes = N * D * dtype_size;

    // Calculate KMeans+ working space
    est.kmeans_n_trials = 2 + static_cast<int32_t>(std::ceil(std::log2(std::max(static_cast<int64_t>(2), n_centroids))));
    est.kmeans_working_bytes = static_cast<size_t>(est.kmeans_n_trials) * N * sizeof(float);

    size_t safety_bytes = static_cast<size_t>(config.gpu_limit_bytes * config.gpu_safety_margin);

    // Step 1: Check if full dataset fits in GPU (with KMeans workspace)
    size_t total_with_kmeans = est.raw_data_bytes + est.kmeans_working_bytes + safety_bytes;
    est.fits_in_gpu = total_with_kmeans <= config.gpu_limit_bytes;

    if (est.fits_in_gpu) {
        est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
        est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);
        est.sampled_fits_with_kmeans = true;
        est.need_pq = false;
        return est;
    }

    // Step 2: Try sampling
    est.sampled_data_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.sample_rate));
    est.sampled_data_bytes = est.sampled_data_rows * D * sizeof(float);

    size_t kmeans_working_bytes_sampled =
        static_cast<size_t>(est.kmeans_n_trials) * est.sampled_data_rows * sizeof(float);

    size_t total_sampled = est.sampled_data_bytes + kmeans_working_bytes_sampled + safety_bytes;
    est.sampled_fits_with_kmeans = total_sampled <= config.gpu_limit_bytes;

    if (est.sampled_fits_with_kmeans) {
        est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(est.sampled_data_rows * config.centroid_ratio));
        est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);
        est.need_pq = false;
        return est;
    }

    // Step 3: Sampled data still too large, must use PQ with adaptive compression
    // Strategy: first reduce pq_bits, then reduce pq_dim if needed
    est.need_pq = true;

    // Initial PQ parameters (DiskANN style)
    uint32_t initial_pq_dim = (config.pq_dim == 0) ?
        ((D / 4 + 7) / 8 * 8) : config.pq_dim;

    uint32_t pq_bits = config.pq_bits_start;
    uint32_t pq_dim = initial_pq_dim;
    bool found_valid_config = false;

    // PQ training data: 5% of sampled data, capped at 65536 rows
    est.pq_train_rows = std::min(
        static_cast<int64_t>(est.sampled_data_rows * config.pq_train_fraction),
        static_cast<int64_t>(config.pq_train_max_rows)
    );
    est.pq_train_data_bytes = est.pq_train_rows * D * sizeof(float);

    // Outer loop: try different pq_dim values (smaller pq_dim = more subspaces = better compression)
    // Minimum viable pq_dim is 4 (at least 1 subspace with 4 dims)
    while (pq_dim >= 4 && !found_valid_config) {
        uint32_t n_subspaces = D / pq_dim;

        // Inner loop: try different pq_bits for current pq_dim
        // No longer restricted to config.pq_bits_min - allow flexible range
        pq_bits = config.pq_bits_start;
        while (!found_valid_config) {
            // Codebook size: n_subspaces × 2^pq_bits × pq_dim × sizeof(float)
            size_t codebook_size = static_cast<size_t>(n_subspaces) * (1 << pq_bits) * pq_dim;
            est.pq_codebook_bytes = codebook_size * sizeof(float);

            // Encoded data size: n_sampled × n_subspaces × ceil(pq_bits / 8.0)
            uint32_t bytes_per_code = (pq_bits + 7) / 8;
            est.pq_encoded_data_bytes = static_cast<size_t>(est.sampled_data_rows) * n_subspaces * bytes_per_code;

            // Decoded (reconstructed) data will be float
            size_t pq_decoded_bytes = static_cast<size_t>(est.sampled_data_rows) * D * sizeof(float);

            // Total GPU requirement for PQ pipeline
            size_t total_pq = est.pq_encoded_data_bytes + est.pq_codebook_bytes +
                             pq_decoded_bytes + kmeans_working_bytes_sampled + safety_bytes;

            if (total_pq <= config.gpu_limit_bytes) {
                est.final_pq_bits = pq_bits;
                found_valid_config = true;
                break;
            }

            if (pq_bits <= config.pq_bits_min) {  // Minimum is pq_bits_min bit
                break;
            }
            pq_bits--;
        }

        // If no valid config found with current pq_dim, try larger pq_dim to reduce subspaces
        if (!found_valid_config) {
            // Increase pq_dim to reduce number of subspaces and achieve better compression
            uint32_t new_pq_dim = pq_dim * 2;
            if (new_pq_dim > D || new_pq_dim == pq_dim) {
                break;  // Can't increase further (pq_dim cannot exceed D)
            }
            // Round to power of 2 for efficiency
            uint32_t power = 1;
            while ((power * 2) <= new_pq_dim) power *= 2;
            pq_dim = power;
            // Continue outer loop to retry with new pq_dim and increased pq_bits range
        }
    }

    if (!found_valid_config) {
        std::cerr << "ERROR: Cannot fit data in GPU even with aggressive PQ compression\n";
        std::cerr << "  Tried pq_dim from " << initial_pq_dim << " down to 4\n";
        std::cerr << "  Tried pq_bits from " << config.pq_bits_start << " down to 1\n";
        std::cerr << "  Minimum required: " << (est.pq_encoded_data_bytes + est.pq_codebook_bytes) / 1e9 << " GB\n";
        std::cerr << "  Available: " << config.gpu_limit_bytes / 1e9 << " GB\n";
        throw std::runtime_error("Data too large for GPU memory even with maximum compression");
    }

    est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(est.sampled_data_rows * config.centroid_ratio));
    est.centroid_data_bytes = est.centroid_rows * D * sizeof(float);

    return est;
}

/**
 * Validate LoadConfig parameters
 */
void validate_load_config(const LoadConfig& config, int64_t N) {
    // Check: centroid_ratio < sample_rate
    if (config.centroid_ratio >= config.sample_rate) {
        throw std::runtime_error(
            "centroid_ratio (" + std::to_string(config.centroid_ratio) +
            ") must be < sample_rate (" + std::to_string(config.sample_rate) + ")"
        );
    }

    // Check: sample_rate range
    if (config.sample_rate <= 0.0f || config.sample_rate > 1.0f) {
        throw std::runtime_error("sample_rate must be in (0, 1]");
    }

    // Check: centroid_ratio range
    if (config.centroid_ratio <= 0.0f || config.centroid_ratio > 1.0f) {
        throw std::runtime_error("centroid_ratio must be in (0, 1]");
    }

    // Check: pq_bits range
    // if (config.pq_bits_start < 4 || config.pq_bits_start > 8) {
    //     throw std::runtime_error("pq_bits_start must be in [4, 8]");
    // }
    if (config.pq_bits_start < config.pq_bits_min) {
        throw std::runtime_error("pq_bits_start must be greater than or equal to pq_bits_min");
    }
    if (config.pq_bits_min >= config.pq_bits_start) {
        throw std::runtime_error("pq_bits_min must be < pq_bits_start");
    }

    // Check: knn_k range
    if (config.knn_k == 0 || config.knn_k > LoadConfig::MAX_KNN_K) {
        throw std::runtime_error(
            "knn_k must be in [1, " + std::to_string(LoadConfig::MAX_KNN_K) +
            "], got " + std::to_string(config.knn_k));
    }
}

/**
 * Print memory estimate in human-readable format
 */
void print_memory_estimate(const MemoryEstimate& est) {
    auto format_bytes = [](size_t bytes) {
        if (bytes < 1024) return std::to_string(bytes) + " B";
        if (bytes < 1024*1024) return std::to_string(bytes/1024) + " KB";
        if (bytes < 1024*1024*1024) return std::to_string(bytes/(1024*1024)) + " MB";
        return std::to_string(bytes/(1024.0*1024*1024)) + " GB";
    };

    std::cout << "\n=== Memory Estimate ===\n";
    std::cout << "Raw data:           " << format_bytes(est.raw_data_bytes) << "\n";
    std::cout << "Fits in GPU:        " << (est.fits_in_gpu ? "YES" : "NO") << "\n";

    if (!est.fits_in_gpu) {
        std::cout << "Sampled data:       " << format_bytes(est.sampled_data_bytes)
                  << " (" << (100.0 * est.sampled_data_rows / est.raw_data_rows) << "%)\n";
        std::cout << "Fits with KMeans:   " << (est.sampled_fits_with_kmeans ? "YES" : "NO") << "\n";

        if (est.need_pq) {
            std::cout << "PQ Quantization:    REQUIRED\n";
            std::cout << "  Final pq_bits:    " << est.final_pq_bits << "\n";
            std::cout << "  Encoded data:     " << format_bytes(est.pq_encoded_data_bytes) << "\n";
            std::cout << "  Codebook:         " << format_bytes(est.pq_codebook_bytes) << "\n";
        }
    }

    std::cout << "KMeans workspace:   " << format_bytes(est.kmeans_working_bytes) << "\n";
    std::cout << "  n_trials:         " << est.kmeans_n_trials << "\n";
    std::cout << "Centroids:          " << est.centroid_rows << " points\n";
    std::cout << "========================\n\n";
}

// ============== Phase 2: Sampling Functions ==============

/**
 * Sample using fast random sampling (with replacement)
 */
std::vector<int64_t> sample_without_replacement(
    int64_t N,
    int64_t n_samples,
    uint32_t seed) {

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, N - 1);
    std::vector<int64_t> indices(n_samples);

    // Fast random sampling: directly generate random indices
    for (int64_t i = 0; i < n_samples; ++i) {
        indices[i] = dist(rng);
    }

    return indices;
}

// ============== Phase 3: PQ Quantization (DiskANN style) ==============

/**
 * Simple quantizer using per-dimension quantization
 * (Simplified version for quick deployment)
 */
struct SimpleQuantizer {
    std::vector<float> scale;        // [D]
    std::vector<float> offset;       // [D]
    uint32_t bits;
};

/**
 * Train simple quantizer from sampled data
 */
SimpleQuantizer train_simple_quantizer(
    const std::vector<float>& X,
    int64_t N, int D,
    uint32_t bits) {

    SimpleQuantizer q;
    q.bits = bits;
    q.scale.resize(D);
    q.offset.resize(D);

    uint32_t n_levels = 1 << bits;

    #pragma omp parallel for
    for (int d = 0; d < D; ++d) {
        float min_val = X[d];
        float max_val = X[d];

        for (int64_t i = 0; i < N; ++i) {
            float val = X[i * D + d];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        q.offset[d] = min_val;
        q.scale[d] = (max_val - min_val) / (n_levels - 1);
        if (q.scale[d] < 1e-6f) q.scale[d] = 1.0f;
    }

    return q;
}

/**
 * Encode float data to uint8 codes using simple quantizer
 */
std::vector<uint8_t> simple_encode(
    const std::vector<float>& X,
    int64_t N, int D,
    const SimpleQuantizer& q) {

    std::vector<uint8_t> codes(N * D);
    uint32_t n_levels = 1 << q.bits;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            float val = X[i * D + d];
            float normalized = (val - q.offset[d]) / q.scale[d];
            normalized = std::max(0.0f, std::min((float)(n_levels - 1), normalized));
            codes[i * D + d] = static_cast<uint8_t>(normalized);
        }
    }

    return codes;
}

/**
 * Decode uint8 codes back to float data
 */
std::vector<float> simple_decode(
    const std::vector<uint8_t>& codes,
    int64_t N, int D,
    const SimpleQuantizer& q) {

    std::vector<float> X_decoded(N * D);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            uint8_t code = codes[i * D + d];
            X_decoded[i * D + d] = q.offset[d] + code * q.scale[d];
        }
    }

    return X_decoded;
}

// ============== Phase 5: CLI and Main Integration ==============

/**
 * Parse LoadConfig from command-line arguments
 */
LoadConfig parse_load_config(const po::variables_map& vm) {
    LoadConfig config;

    if (vm.count("cpu-limit")) {
        config.cpu_limit_bytes = vm["cpu-limit"].as<size_t>();
    }
    if (vm.count("gpu-limit")) {
        config.gpu_limit_bytes = vm["gpu-limit"].as<size_t>();
    }
    if (vm.count("sample-rate")) {
        config.sample_rate = vm["sample-rate"].as<float>();
    }
    if (vm.count("centroid-ratio")) {
        config.centroid_ratio = vm["centroid-ratio"].as<float>();
    }
    if (vm.count("use-pq")) {
        config.use_pq = vm["use-pq"].as<bool>();
    }
    if (vm.count("pq-bits-start")) {
        config.pq_bits_start = vm["pq-bits-start"].as<uint32_t>();
    }
    if (vm.count("pq-bits-min")) {
        config.pq_bits_min = vm["pq-bits-min"].as<uint32_t>();
    }
    if (vm.count("seed")) {
        config.seed = vm["seed"].as<uint32_t>();
    }
    if (vm.count("knn-k")) {
        config.knn_k = vm["knn-k"].as<uint32_t>();
        if (config.knn_k > LoadConfig::MAX_KNN_K) {
            throw std::runtime_error(
                "knn-k must be <= " + std::to_string(LoadConfig::MAX_KNN_K) +
                ", got " + std::to_string(config.knn_k));
        }
    }

    return config;
}

// ============== Phase 7: Greedy Graph Search Kernel (template, 支持 uint8/float 等) ==============

template <typename T>
__device__ float device_l2_dist(const T* a, const T* b, int D)
{
    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = static_cast<float>(a[d]) - static_cast<float>(b[d]);
        dist += diff * diff;
    }
    return dist;
}

/**
 * 每个线程处理一个 query，在 KNN 图上做贪心搜索，返回 top-2 最近 centroid。
 * 支持任意数值类型 T (uint8_t, float, uint32_t, ...)
 */
template <typename T>
__global__ void greedy_graph_search_top2_kernel(
    const T*        dataset,     // (n, D) centroid data on GPU
    const uint32_t* graph,       // (n, K) KNN graph on GPU
    const T*        queries,     // (batch, D) query data
    uint32_t*       out_top2,    // (batch, 2) output: [top1, top2] per query
    int n, int D, int K, int batch, int max_iters)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= batch) return;

    const T* q = queries + static_cast<int64_t>(qid) * D;

    // 用伪随机的 entry point 开始搜索
    uint32_t cur = static_cast<uint32_t>(static_cast<uint64_t>(qid) * 7919ULL
                                         % static_cast<uint64_t>(n));

    float best1 = device_l2_dist(q, dataset + static_cast<int64_t>(cur) * D, D);
    uint32_t top1 = cur;
    float best2 = 1e30f;
    uint32_t top2 = cur;

    for (int iter = 0; iter < max_iters; ++iter) {
        bool improved = false;
        for (int k = 0; k < K; ++k) {
            uint32_t nb = graph[static_cast<int64_t>(cur) * K + k];
            if (nb >= static_cast<uint32_t>(n)) continue;

            float d = device_l2_dist(q, dataset + static_cast<int64_t>(nb) * D, D);
            if (d < best1) {
                best2 = best1; top2 = top1;
                best1 = d;     top1 = nb;
                improved = true;
            } else if (d < best2 && nb != top1) {
                best2 = d; top2 = nb;
            }
        }
        if (!improved) break;
        cur = top1;
    }

    out_top2[static_cast<int64_t>(qid) * 2]     = top1;
    out_top2[static_cast<int64_t>(qid) * 2 + 1] = top2;
}

// CAGRA-style beam width 估算:
//   - 参考 raft::neighbors::cagra::search_params 默认 itopk_size = 64
//   - floor 64, 上限 1024, 至少 max(K, 5)
//   - 大 K 时让 beam 适当宽于 K (1.5x)
//   - 例: K=2  → 64;  K=10 → 64;  K=64 → 96;  K=512 → 768;  K≥683 → 1024(cap)
inline uint32_t compute_beam_width(uint32_t k) {
    constexpr uint32_t MIN_BEAM    = 5;
    constexpr uint32_t MAX_BEAM    = 1024;
    constexpr uint32_t CAGRA_ITOPK = 64;
    uint32_t beam = std::max(CAGRA_ITOPK, (3u * k + 1u) / 2u);
    beam = std::max(beam, std::max(k, MIN_BEAM));
    beam = std::min(beam, MAX_BEAM);
    return beam;
}

/**
 * Greedy best-first 图搜索 (CAGRA search 的 single-thread-per-query 简化版),
 * 在 navigable graph 上为每个 query 找 top-K_out 近邻。
 *
 * 算法 (per qid):
 *   1) 维护 sorted candidate buffer 大小 = beam_width (≥ K_out, 余量提升 recall)
 *   2) 起始: self-search 用 graph[qid][0]; query 模式用 hash 落到 db
 *   3) 每轮: 取 buffer 里最近的未访问候选 → 扩展其 K_graph 邻居 → try_insert
 *   4) 直到 max_iters 或 buffer 全部访问
 *   5) self-search 时排除 idx == qid (distance=0 自环)
 *   6) 输出 buffer 中前 K_out 个作为最终结果
 *
 * 工作 buffer (idx/dist/visited) 全部由 caller 在 GPU global memory 预分配,
 * 大小均为 n_q * beam_width.
 *
 * 两种调用模式:
 *   self-search:  queries == dataset, n_q == n_db, exclude_self_idx == true
 *   query mode:   queries 与 dataset 不同, exclude_self_idx == false
 */
template <typename T>
__global__ void greedy_graph_search_topK_kernel(
    const T*        dataset,       // (n_db, D)
    const uint32_t* graph,         // (n_db, K_graph)
    const T*        queries,       // (n_q, D); self-search 时传 dataset
    uint32_t*       out_topK,      // (n_q, K_out)  最终输出 idx
    uint32_t*       buf_idx,       // (n_q, beam_width)  candidate idx
    float*          buf_dist,      // (n_q, beam_width)  candidate dist
    uint8_t*        buf_visited,   // (n_q, beam_width)  visited flags (0/1)
    int n_db, int n_q, int D, int K_graph, int K_out, int beam_width, int max_iters,
    bool exclude_self_idx)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= n_q) return;

    const T* q = queries + static_cast<int64_t>(qid) * D;
    uint32_t* idx_buf = buf_idx     + static_cast<int64_t>(qid) * beam_width;
    float*    d_buf   = buf_dist    + static_cast<int64_t>(qid) * beam_width;
    uint8_t*  vis_buf = buf_visited + static_cast<int64_t>(qid) * beam_width;

    // 初始化 visited
    for (int i = 0; i < beam_width; ++i) vis_buf[i] = 0;

    int filled = 0;

    // 在 sorted candidate buffer 里插入 (idx, d); idx/dist/visited 三数组同步移位
    auto try_insert = [&](uint32_t idx, float d) {
        if (exclude_self_idx && idx == static_cast<uint32_t>(qid)) return;
        // 去重 (线性扫描 buffer)
        for (int i = 0; i < filled; ++i) {
            if (idx_buf[i] == idx) return;
        }
        if (filled < beam_width) {
            int pos = filled;
            while (pos > 0 && d_buf[pos-1] > d) {
                d_buf[pos]   = d_buf[pos-1];
                idx_buf[pos] = idx_buf[pos-1];
                vis_buf[pos] = vis_buf[pos-1];
                --pos;
            }
            d_buf[pos]   = d;
            idx_buf[pos] = idx;
            vis_buf[pos] = 0;
            ++filled;
        } else if (d < d_buf[beam_width - 1]) {
            int pos = beam_width - 1;
            while (pos > 0 && d_buf[pos-1] > d) {
                d_buf[pos]   = d_buf[pos-1];
                idx_buf[pos] = idx_buf[pos-1];
                vis_buf[pos] = vis_buf[pos-1];
                --pos;
            }
            d_buf[pos]   = d;
            idx_buf[pos] = idx;
            vis_buf[pos] = 0;
        }
    };

    // 起始点
    uint32_t cur;
    if (exclude_self_idx) {
        // self-search: qid 既是 query 也是 db 索引, 用 qid 的图邻居作种
        cur = graph[static_cast<int64_t>(qid) * K_graph];
        if (cur >= static_cast<uint32_t>(n_db) || cur == static_cast<uint32_t>(qid)) {
            cur = static_cast<uint32_t>((qid + 1) % n_db);
        }
    } else {
        // query mode: query 不在 db 里, hash 选一个 db 节点作种
        cur = static_cast<uint32_t>(static_cast<uint64_t>(qid) * 7919ULL
                                    % static_cast<uint64_t>(n_db));
    }
    float d_cur = device_l2_dist(q, dataset + static_cast<int64_t>(cur) * D, D);
    try_insert(cur, d_cur);

    // Greedy best-first 扩展
    for (int iter = 0; iter < max_iters; ++iter) {
        int next_pos = -1;
        for (int i = 0; i < filled; ++i) {
            if (!vis_buf[i]) { next_pos = i; break; }
        }
        if (next_pos < 0) break;  // buffer 全部访问完 → 收敛

        uint32_t expand_node = idx_buf[next_pos];
        vis_buf[next_pos] = 1;

        for (int k = 0; k < K_graph; ++k) {
            uint32_t nb = graph[static_cast<int64_t>(expand_node) * K_graph + k];
            if (nb >= static_cast<uint32_t>(n_db)) continue;
            float d = device_l2_dist(q, dataset + static_cast<int64_t>(nb) * D, D);
            try_insert(nb, d);
        }
    }

    // 输出 buffer 中前 K_out 个 (sorted by L2)
    uint32_t* out_buf = out_topK + static_cast<int64_t>(qid) * K_out;
    for (int i = 0; i < K_out; ++i) {
        out_buf[i] = (i < filled) ? idx_buf[i] : 0xFFFFFFFFu;
    }
}

// ============== Phase 8: Batch Assignment via Graph ANNS ==============

/**
 * 分批处理原始数据集，在 GPU 上基于 centroid KNN 图做 ANNS 检索，
 * 为每个非 centroid 数据点找到最近的 2 个 centroid，在 CPU 上做均衡分配。
 *
 * centroid 数据和 KNN 图由 caller 提前上传至 GPU，本函数直接复用，不再重复上传。
 *
 * @tparam T                     原始数据元素类型 (float, uint8_t, uint32_t 等)
 * @tparam CentroidT             centroid 在 GPU 上的数据类型 (float 或 uint8_t)
 * @param X_full                 完整原始数据集 (N * D, T, row-major, CPU)
 * @param N                      数据点总数
 * @param D                      向量维度
 * @param n_centroids            centroid 数量
 * @param centroid_global_indices 每个 centroid 在原始数据中的全局索引
 * @param d_centroids            centroid 数据 (n_centroids * D, CentroidT, 已在 GPU)
 * @param centroid_gpu_bytes     centroid 数据在 GPU 上占用的字节数
 * @param d_graph                centroid KNN 图 (n_centroids * K, uint32, 已在 GPU)
 * @param graph_bytes            KNN 图在 GPU 上占用的字节数
 * @param K                      KNN 图度数
 * @param quantizer              量化器（仅 CentroidT=uint8_t 时使用）
 * @param Mbatch_bytes           GPU 剩余可用空间 (bytes), 默认 10GB
 * @param search_max_iters       图搜索最大迭代次数, 默认 64
 *
 * @return  (N,) 每个点分配到的 local centroid index [0, n_centroids)
 */
template <typename T, typename CentroidT>
std::vector<int64_t> batch_assign_with_cagra_anns(
    const T* X_full,
    int64_t N,
    int64_t D,
    int64_t n_centroids,
    const std::vector<int64_t>& centroid_global_indices,
    CentroidT* d_centroids,
    size_t centroid_gpu_bytes,
    uint32_t* d_graph,
    size_t graph_bytes,
    uint32_t K,
    const SimpleQuantizer& quantizer,
    size_t Mbatch_bytes = 10ULL * 1024 * 1024 * 1024,
    int search_max_iters = 64)
{
    constexpr bool is_pq = std::is_same<CentroidT, uint8_t>::value;

    // ================================================================
    // Step 0: 建立 centroid 集合，收集非 centroid 点索引
    // ================================================================
    std::unordered_set<int64_t> centroid_set(
        centroid_global_indices.begin(), centroid_global_indices.end());

    std::vector<int64_t> non_centroid_indices;
    non_centroid_indices.reserve(N - n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        if (centroid_set.find(i) == centroid_set.end()) {
            non_centroid_indices.push_back(i);
        }
    }
    int64_t N_nc = static_cast<int64_t>(non_centroid_indices.size());

    std::cout << "[BatchAssign] N=" << N
              << ", centroids=" << n_centroids
              << ", non-centroid=" << N_nc
              << ", is_pq=" << is_pq << "\n";

    // ================================================================
    // Step 1: 估算每批大小 Nv
    //         centroid 和 graph 已在 GPU，扣除其占用后计算剩余空间
    // ================================================================
    size_t constant_gpu = graph_bytes + centroid_gpu_bytes;
    size_t remaining = (Mbatch_bytes > constant_gpu) ? (Mbatch_bytes - constant_gpu) : 0;

    std::cout << "[BatchAssign] GPU constant: centroids=" << centroid_gpu_bytes / 1e6
              << "MB, graph=" << graph_bytes / 1e6
              << "MB, remaining=" << remaining / 1e9 << "GB\n";

    // 用 cagra::search 替代手写 kernel; queries != centroids, 不需 self-exclusion.
    constexpr uint32_t TOP_K = 2;
    // CAGRA itopk_size 至少 TOP_K, 32 对齐, floor 64
    const uint32_t itopk = std::max<uint32_t>(64, ((TOP_K + 31) / 32) * 32);

    // Per-point GPU (CAGRA 不再需要 buf_idx/dist/visited):
    //   query:       D * sizeof(CentroidT)
    //   neighbors:   TOP_K * 4
    //   distances:   TOP_K * 4
    size_t per_point_bytes = static_cast<size_t>(D) * sizeof(CentroidT)
                           + TOP_K * (sizeof(uint32_t) + sizeof(float));

    int64_t Nv = static_cast<int64_t>(remaining / per_point_bytes);
    Nv = std::max(static_cast<int64_t>(1), std::min(Nv, N_nc));

    std::cout << "[BatchAssign] per_point=" << per_point_bytes << "B"
              << ", itopk=" << itopk
              << ", Nv=" << Nv
              << ", batches=" << (N_nc + Nv - 1) / Nv << "\n";

    // ================================================================
    // Step 3: 初始化分配结果和聚类大小
    // ================================================================
    std::vector<int64_t> assignments(N, -1);

    // centroid 点预分配给自身
    for (int64_t c = 0; c < n_centroids; ++c) {
        assignments[centroid_global_indices[c]] = c;
    }

    // 聚类大小：每个 centroid 初始大小为 1
    std::vector<std::atomic<int64_t>> cluster_sizes(n_centroids);
    for (auto& s : cluster_sizes) s.store(1, std::memory_order_relaxed);

    // ================================================================
    // Step 4: 构造 cagra::index<CentroidT, uint32_t> (zero-copy view, 只构造一次)
    // ================================================================
    raft::resources res;
    auto dataset_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
        d_centroids, n_centroids, static_cast<int64_t>(D));
    auto graph_view = raft::make_device_matrix_view<const uint32_t, int64_t>(
        d_graph, n_centroids, static_cast<int64_t>(K));
    raft::neighbors::cagra::index<CentroidT, uint32_t> cagra_idx(
        res, raft::distance::DistanceType::L2Expanded,
        dataset_view, graph_view);

    raft::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;
    sp.algo           = raft::neighbors::cagra::search_algo::SINGLE_CTA;
    (void)search_max_iters;  // CAGRA 自己决定 iter, 参数保留为接口兼容

    // ================================================================
    // Step 5: 分批处理
    // ================================================================
    // 分配 per-batch GPU buffer (复用)
    CentroidT* d_queries    = nullptr;
    uint32_t*  d_top2       = nullptr;
    float*     d_distances  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_queries,    Nv * D * sizeof(CentroidT)));
    CUDA_CHECK(cudaMalloc(&d_top2,       Nv * TOP_K * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances,  Nv * TOP_K * sizeof(float)));

    std::vector<uint32_t> h_top2(Nv * TOP_K);

    for (int64_t batch_start = 0; batch_start < N_nc; batch_start += Nv) {
        int64_t batch_end = std::min(batch_start + Nv, N_nc);
        int64_t batch_size = batch_end - batch_start;

        // ---- 4a: 读取 batch 数据，转换为 CentroidT，上传到 GPU ----
        if constexpr (is_pq) {
            // PQ 模式：读取原始 T 数据 -> 转为 float32 -> encode uint8 -> 上传
            std::vector<float> batch_f32(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                for (int64_t d = 0; d < D; ++d) {
                    batch_f32[i * D + d] = static_cast<float>(X_full[gi * D + d]);
                }
            }

            auto codes = simple_encode(batch_f32, batch_size, D, quantizer);

            CUDA_CHECK(cudaMemcpy(d_queries, codes.data(),
                                  batch_size * D * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        } else {
            // 非 PQ 模式：读取原始 T 数据 -> 转为 CentroidT -> 上传
            std::vector<CentroidT> batch_data(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                for (int64_t d = 0; d < D; ++d) {
                    batch_data[i * D + d] = static_cast<CentroidT>(X_full[gi * D + d]);
                }
            }

            CUDA_CHECK(cudaMemcpy(d_queries, batch_data.data(),
                                  batch_size * D * sizeof(CentroidT),
                                  cudaMemcpyHostToDevice));
        }

        // ---- 4b: cagra::search (k=2, queries 不在 dataset 里, 无 self-exclusion) ----
        auto queries_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
            d_queries, batch_size, static_cast<int64_t>(D));
        auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
            d_top2, batch_size, static_cast<int64_t>(TOP_K));
        auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            d_distances, batch_size, static_cast<int64_t>(TOP_K));

        raft::neighbors::cagra::search(res, sp, cagra_idx,
            queries_view, neighbors_view, distances_view);

        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- 4c: 下载 top-2 local centroid indices ----
        CUDA_CHECK(cudaMemcpy(h_top2.data(), d_top2,
                              batch_size * TOP_K * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        // ---- 4d: CPU 并行: local idx -> global idx 映射 + 均衡分配 ----
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < batch_size; ++i) {
            uint32_t local_idx1 = h_top2[i * TOP_K];      // 最近
            uint32_t local_idx2 = h_top2[i * TOP_K + 1];  // 第二近

            // 映射: local centroid idx -> global dataset idx
            // (caller 可通过 centroid_global_indices[local_idx] 得到全局)

            // 比较两个聚类大小，做均衡分配
            int64_t s1 = cluster_sizes[local_idx1].load(std::memory_order_relaxed);
            int64_t s2 = cluster_sizes[local_idx2].load(std::memory_order_relaxed);

            uint32_t chosen;
            if (static_cast<double>(s2) < 0.8 * static_cast<double>(s1)) {
                chosen = local_idx2;  // 聚类 2 太小，分配到聚类 2
            } else {
                chosen = local_idx1;  // 分配到最近的聚类 1
            }

            int64_t gi = non_centroid_indices[batch_start + i];
            assignments[gi] = chosen;
            cluster_sizes[chosen].fetch_add(1, std::memory_order_relaxed);
        }

        if ((batch_start / Nv) % 10 == 0 || batch_end == N_nc) {
            std::cout << "[BatchAssign] " << batch_end << "/" << N_nc << "\n";
        }
    }

    // ================================================================
    // Step 6: 释放 per-batch GPU buffer（centroid 和 graph 保留）
    // ================================================================
    if (d_queries)    CUDA_CHECK(cudaFree(d_queries));
    if (d_top2)       CUDA_CHECK(cudaFree(d_top2));
    if (d_distances)  CUDA_CHECK(cudaFree(d_distances));
    // NOTE: d_centroids 和 d_graph 由 caller 管理，本函数不释放

    // 聚类大小统计
    int64_t min_sz = std::numeric_limits<int64_t>::max(), max_sz = 0;
    double avg_sz = 0;
    for (int64_t c = 0; c < n_centroids; ++c) {
        int64_t sz = cluster_sizes[c].load();
        min_sz = std::min(min_sz, sz);
        max_sz = std::max(max_sz, sz);
        avg_sz += sz;
    }
    avg_sz /= n_centroids;
    std::cout << "[BatchAssign] Done. Cluster size: min=" << min_sz
              << " max=" << max_sz
              << " avg=" << std::fixed << std::setprecision(1) << avg_sz << "\n";

    return assignments;
}

// ============== Phase 9: GPU helper kernels ==============

// Functor: 按 stride 访问 min_dist 数组，用于子采样估计 sum
struct StridedAccessOp {
    const float* d_min_dist;
    int64_t stride;
    __host__ __device__ float operator()(int64_t i) const {
        return d_min_dist[i * stride];
    }
};

/**
 * CUDA kernel: 计算两个 flag 数组的差异。
 *   diff[i] = 1 if curr[i] && !prev[i], else 0
 * 用于在 GPU 上找出本轮新增的候选点，避免将 flags 拷回 CPU。
 */
__global__ void compute_flag_diff(
    const uint8_t* __restrict__ curr,
    const uint8_t* __restrict__ prev,
    uint8_t* __restrict__ diff,
    int64_t N)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    diff[i] = (curr[i] && !prev[i]) ? 1 : 0;
}

/**
 * CUDA kernel: 按索引从源矩阵中 gather 行到目标矩阵。
 *   dst[k, :] = src[idx[k], :]   for k in [0, n_dst_rows)
 */
__global__ void gather_rows_by_index(
    const float* __restrict__ src,
    const int64_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;

    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = src[idx[row] * D + col];
}

// ============== Phase 9b: KMeans|| (Scalable KMeans++) Centroid Selection ==============

/**
 * 在 GPU 上用 Scalable KMeans++ (KMeans||, Bahmani et al. 2012) 选取 centroid。
 * 结果保留在 GPU 上不拷贝回 CPU。
 *
 * 调用方负责采样和 index 管理; 本函数只处理 GPU 上的 centroid 选取。
 *
 * 三层索引关系 (由调用方维护):
 *   raw_idx  ←→ sample_idx   : sampled_indices[sample_idx] = raw_idx
 *   sample_idx ←→ centroid_idx: 本函数返回值 [centroid_idx] = sample_idx
 *
 * @param X_sampled       采样后的数据 (CPU, float32, working_N × D)，可以是全量或子集
 * @param sampled_indices 采样索引表: sampled_indices[i] = raw dataset index
 * @param working_N       采样后的行数
 * @param D               向量维度
 * @param config          配置
 * @param mem_est         内存估算
 * @param out_quantizer   [out] PQ 量化器 (如果启用)
 * @param d_centroids_f32 [out] GPU 上的 centroid 数据 (caller 管理)
 * @return  (n_centroids,) 每个 centroid 的 sample_idx
 */
std::vector<int64_t> select_centroids_on_gpu_kmeans_parallel(
    const float* X_sampled,
    const std::vector<int64_t>& sampled_indices,
    int64_t working_N, int D,
    const LoadConfig& config,
    const MemoryEstimate& mem_est,
    SimpleQuantizer* out_quantizer,
    float** d_centroids_f32)
{
    std::cout << "[GPU Centroid Selection - KMeans|| + KMeans++]\n";

    float* allocated_centroids = nullptr;

    try {
        // Stage 1: PQ quantization if needed
        const float* upload_data = X_sampled;   // 直接指向调用方数据，零拷贝
        std::vector<float> X_pq_buf;            // 仅 PQ 时持有数据
        SimpleQuantizer quantizer{};
        if (mem_est.need_pq) {
            std::cout << "  Training PQ quantizer (" << mem_est.final_pq_bits << " bits)...\n";
            std::vector<float> pq_input(upload_data, upload_data + static_cast<size_t>(working_N) * D);
            quantizer = train_simple_quantizer(pq_input, working_N, D, mem_est.final_pq_bits);
            auto codes = simple_encode(pq_input, working_N, D, quantizer);
            X_pq_buf = simple_decode(codes, working_N, D, quantizer);
            upload_data = X_pq_buf.data();
            if (out_quantizer) *out_quantizer = quantizer;
        }

        int64_t n_centroids = mem_est.centroid_rows;
        std::cout << "  N_work=" << working_N << ", D=" << D
                  << ", n_centroids=" << n_centroids << "\n";

        // Stage 3: Upload working data to GPU & precompute L2 norms
        //          CPU 侧只有一个指针 upload_data，无额外副本
        raft::resources res;
        cudaStream_t stream = raft::resource::get_cuda_stream(res);

        auto dataset_gpu = raft::make_device_matrix<float, int64_t>(res, working_N, D);
        raft::copy(dataset_gpu.data_handle(), upload_data,
                   working_N * D, stream);
        raft::resource::sync_stream(res);

        auto X_view = raft::make_device_matrix_view<const float, int64_t>(
            dataset_gpu.data_handle(), working_N, static_cast<int64_t>(D));

        // L2 norms required by minClusterDistanceCompute (fused L2 path)
        auto L2NormX = raft::make_device_vector<float, int64_t>(res, working_N);
        raft::linalg::rowNorm(L2NormX.data_handle(),
                              dataset_gpu.data_handle(),
                              static_cast<int64_t>(D), working_N,
                              raft::linalg::L2Norm, true, stream);

        // ================================================================
        // Stage 4: KMeans|| oversampling
        // ================================================================
        std::cout << "  Running KMeans|| oversampling...\n";
        raft::random::RngState rng(config.seed);
        std::mt19937 gen(config.seed);

        // 优化策略 (针对 ~1% centroid_ratio, 仅需粗略分组):
        //   - niter = 3 轮 (而非 RAFT 默认的 8 轮)
        //   - 每轮期望选出 l*k 个候选, niter 轮总共 ≈ niter * l * k 个
        //   - 目标: 总候选 ≈ 1.5 * n_centroids (轻度过采样, 减少 GPU 工作量)
        //   - 因此 l = 1.5 / niter ≈ 0.5
        constexpr int FIXED_NITER = 3;
        constexpr double TOTAL_OVERSHOOT = 1.5;  // 总候选数倍数
        double oversampling_factor = TOTAL_OVERSHOOT / FIXED_NITER;

        // Step 1: Pick first centroid uniformly at random
        std::uniform_int_distribution<int64_t> uniform(0, working_N - 1);
        int64_t cIdx = uniform(gen);

        // isSampleCentroid[i] = 1 if point i is a candidate
        auto isSampleCentroid = raft::make_device_vector<uint8_t, int64_t>(res, working_N);
        CUDA_CHECK(cudaMemsetAsync(isSampleCentroid.data_handle(), 0,
                                   working_N * sizeof(uint8_t), stream));
        uint8_t one_val = 1;
        CUDA_CHECK(cudaMemcpyAsync(
            isSampleCentroid.data_handle() + cIdx, &one_val, 1,
            cudaMemcpyHostToDevice, stream));

        // Growing candidate data buffer on GPU
        rmm::device_uvector<float> centroidsBuf(D, stream);
        raft::copy(centroidsBuf.data(),
                   dataset_gpu.data_handle() + cIdx * D, D, stream);

        auto potentialCentroids = raft::make_device_matrix_view<float, int64_t>(
            centroidsBuf.data(), static_cast<int64_t>(1), static_cast<int64_t>(D));

        // Track candidate local indices — order matches centroidsBuf rows
        std::vector<int64_t> candidate_local_indices;
        candidate_local_indices.push_back(cIdx);

        // GPU-side prev_flags for round-by-round diffing (避免每轮 D2H)
        auto d_prev_flags = raft::make_device_vector<uint8_t, int64_t>(res, working_N);
        CUDA_CHECK(cudaMemsetAsync(d_prev_flags.data_handle(), 0,
                                   working_N * sizeof(uint8_t), stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_prev_flags.data_handle() + cIdx, &one_val, 1,
            cudaMemcpyHostToDevice, stream));
        auto d_diff_flags = raft::make_device_vector<uint8_t, int64_t>(res, working_N);

        // Buffers for RAFT distance computation
        rmm::device_uvector<float> L2NormBuf_OR_DistBuf(0, stream);
        rmm::device_uvector<char> workspace(0, stream);
        auto minClusterDistVec = raft::make_device_vector<float, int64_t>(res, working_N);
        auto uniformRands = raft::make_device_vector<float, int64_t>(res, working_N);
        rmm::device_scalar<float> clusterCost(stream);

        // Step 2: Compute initial cost psi = phi_X(C)
        raft::cluster::detail::minClusterDistanceCompute<float, int64_t>(
            res, X_view, potentialCentroids, minClusterDistVec.view(),
            L2NormX.view(), L2NormBuf_OR_DistBuf,
            raft::distance::DistanceType::L2Expanded,
            1 << 15, 0, workspace);

        raft::cluster::detail::computeClusterCost(
            res, minClusterDistVec.view(), workspace,
            raft::make_device_scalar_view(clusterCost.data()),
            raft::identity_op{}, raft::add_op{});

        float psi = clusterCost.value(stream);
        raft::resource::sync_stream(res, stream);

        // 写死 niter (而非 log(psi)), 避免对 ~1% centroid_ratio 的过度迭代
        int niter = FIXED_NITER;
        std::cout << "  KMeans||: psi=" << psi << ", niter=" << niter
                  << ", l=" << oversampling_factor << "\n";

        // 子采样估计 psi 的参数
        // 对 minClusterDistVec 做 strided 采样 (~1%)，估计 sum 而不做全 N reduction
        constexpr int64_t PSI_SUBSAMPLE_TARGET = 16384;  // 目标采样点数
        int64_t psi_stride = std::max<int64_t>(1, working_N / PSI_SUBSAMPLE_TARGET);
        int64_t psi_sub_n = working_N / psi_stride;     // 实际采样点数
        float psi_scale = static_cast<float>(working_N) / static_cast<float>(psi_sub_n);

        // Step 3-6: Oversampling rounds
        for (int iter = 0; iter < niter; ++iter) {
            // Recompute min distances to ALL accumulated candidates (batched GEMM)
            raft::cluster::detail::minClusterDistanceCompute<float, int64_t>(
                res, X_view, potentialCentroids, minClusterDistVec.view(),
                L2NormX.view(), L2NormBuf_OR_DistBuf,
                raft::distance::DistanceType::L2Expanded,
                1 << 15, 0, workspace);

            // 子采样估计 psi: 仅对 ~1% 的点求和，再放大回估计值
            // strided iterator 访问 min_dist[0], min_dist[stride], min_dist[2*stride], ...
            StridedAccessOp stride_op{minClusterDistVec.data_handle(), psi_stride};
            auto strided_iter = thrust::make_transform_iterator(
                thrust::make_counting_iterator<int64_t>(0), stride_op);
            float sub_sum = thrust::reduce(
                thrust::cuda::par.on(stream),
                strided_iter, strided_iter + psi_sub_n,
                0.0f, thrust::plus<float>());
            psi = sub_sum * psi_scale;

            // Independent D² sampling: prob(x) = l * k * d²(x,C) / psi
            raft::random::uniform(
                res, rng, uniformRands.data_handle(), working_N, 0.0f, 1.0f);

            raft::cluster::detail::SamplingOp<float, int64_t> select_op(
                psi, oversampling_factor, n_centroids,
                uniformRands.data_handle(),
                isSampleCentroid.data_handle());

            // CUB DeviceSelect::If — parallel filtering, one kernel, no per-point sync
            rmm::device_uvector<float> CpRaw(0, stream);
            raft::cluster::detail::sampleCentroids<float, int64_t>(
                res, X_view, minClusterDistVec.view(),
                isSampleCentroid.view(), select_op, CpRaw, workspace);

            int64_t n_new = CpRaw.size() / D;
            if (n_new == 0) {
                std::cout << "  Round " << iter << ": no new candidates, stopping early\n";
                break;
            }

            // Append new candidate data to growing buffer
            size_t old_size = centroidsBuf.size();
            centroidsBuf.resize(old_size + CpRaw.size(), stream);
            raft::copy(centroidsBuf.data() + old_size,
                       CpRaw.data(), CpRaw.size(), stream);

            int64_t tot = potentialCentroids.extent(0) + n_new;
            potentialCentroids = raft::make_device_matrix_view<float, int64_t>(
                centroidsBuf.data(), tot, static_cast<int64_t>(D));

            // GPU-side flag diff + ordered compact — 避免 working_N 字节 D2H
            // sampleCentroids 标记新点 → isSampleCentroid; CUB 保序
            {
                int diff_threads = 256;
                int diff_blocks = static_cast<int>((working_N + diff_threads - 1) / diff_threads);
                compute_flag_diff<<<diff_blocks, diff_threads, 0, stream>>>(
                    isSampleCentroid.data_handle(),
                    d_prev_flags.data_handle(),
                    d_diff_flags.data_handle(),
                    working_N);
                CUDA_CHECK(cudaGetLastError());

                // thrust::copy_if 保序提取新增索引 (输出 n_new 个 int64)
                rmm::device_uvector<int64_t> d_new_idx(n_new, stream);
                auto cnt_begin = thrust::make_counting_iterator<int64_t>(0);
                thrust::copy_if(
                    thrust::cuda::par.on(stream),
                    cnt_begin, cnt_begin + working_N,
                    thrust::device_pointer_cast(d_diff_flags.data_handle()),
                    d_new_idx.begin(),
                    [] __device__ (uint8_t f) { return f > 0; });

                // 仅传回 n_new 个 int64 (几 KB)，而非 working_N 字节
                std::vector<int64_t> h_new_idx(n_new);
                raft::copy(h_new_idx.data(), d_new_idx.data(), n_new, stream);
                raft::resource::sync_stream(res, stream);
                candidate_local_indices.insert(
                    candidate_local_indices.end(), h_new_idx.begin(), h_new_idx.end());

                // prev_flags = curr_flags (GPU D2D，无 CPU 参与)
                raft::copy(d_prev_flags.data_handle(),
                           isSampleCentroid.data_handle(), working_N, stream);
            }

            std::cout << "  Round " << iter << ": +" << n_new
                      << " candidates (total=" << tot << "), psi=" << psi << "\n";
        }

        int64_t n_candidates = potentialCentroids.extent(0);
        std::cout << "  KMeans|| done: " << n_candidates << " candidates oversampled\n";

        // ================================================================
        // Stage 5: 从候选集中选 n_centroids 个实际数据点
        // 候选集已经是 D² 加权过采样的结果，直接 uniform subsample 即可
        // ================================================================
        size_t centroid_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        CUDA_CHECK(cudaMalloc(d_centroids_f32, centroid_bytes));
        allocated_centroids = *d_centroids_f32;

        std::vector<int64_t> sub_indices;

        if (n_candidates <= n_centroids) {
            std::cout << "  [WARN] Only " << n_candidates << " candidates <= "
                      << n_centroids << " requested. Using all.\n";
            CUDA_CHECK(cudaMemcpyAsync(
                *d_centroids_f32, centroidsBuf.data(),
                n_candidates * D * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sub_indices.resize(n_candidates);
            std::iota(sub_indices.begin(), sub_indices.end(), 0LL);
        } else {
            // 候选集已由 KMeans|| D² 加权采样产生，分布良好
            // 直接 uniform 随机选 n_centroids 个，O(n_centroids)，无 GPU sync
            std::cout << "  Selecting " << n_centroids << " from "
                      << n_candidates << " candidates (uniform subsample)...\n";

            // Fisher-Yates 前 n_centroids 步，生成不重复的候选索引
            std::vector<int64_t> perm(n_candidates);
            std::iota(perm.begin(), perm.end(), 0LL);
            std::mt19937 sub_gen(config.seed + 42);
            for (int64_t i = 0; i < n_centroids; ++i) {
                std::uniform_int_distribution<int64_t> d(i, n_candidates - 1);
                std::swap(perm[i], perm[d(sub_gen)]);
            }
            sub_indices.assign(perm.begin(), perm.begin() + n_centroids);

            // GPU gather: 按选中的行号从 centroidsBuf 拷贝到 d_centroids_f32
            // 上传 sub_indices 到 GPU，用 gather kernel 一次完成
            rmm::device_uvector<int64_t> d_sub_indices(n_centroids, stream);
            raft::copy(d_sub_indices.data(), sub_indices.data(), n_centroids, stream);

            int64_t total_elems = n_centroids * static_cast<int64_t>(D);
            int gthreads = 256;
            int gblocks = static_cast<int>((total_elems + gthreads - 1) / gthreads);
            gather_rows_by_index<<<gblocks, gthreads, 0, stream>>>(
                centroidsBuf.data(), d_sub_indices.data(),
                *d_centroids_f32, n_centroids, static_cast<int64_t>(D));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Release working data — only caller-managed centroids remain
        { auto _drop = std::move(dataset_gpu); }

        // Stage 6: centroid_idx → candidate_idx → sample_idx
        //   sub_indices[k]                        : centroid k 在 candidate 集中的行号
        //   candidate_local_indices[sub_indices[k]]: 该候选点的 sample_idx
        //   调用方可用 sampled_indices[sample_idx] 得到 raw_idx
        int64_t n_selected = static_cast<int64_t>(sub_indices.size());
        std::vector<int64_t> centroid_sample_indices(n_selected);
        for (int64_t k = 0; k < n_selected; ++k) {
            centroid_sample_indices[k] = candidate_local_indices[sub_indices[k]];
        }

        std::cout << "  Centroids on GPU: " << centroid_bytes / 1e6 << " MB\n";
        std::cout << "  Generated " << n_selected << " centroids\n";

        return centroid_sample_indices;

    } catch (const std::exception& e) {
        std::cerr << "[select_centroids_on_gpu_kmeans_parallel] Error: " << e.what() << "\n";

        if (allocated_centroids) {
            cudaFree(allocated_centroids);
            *d_centroids_f32 = nullptr;
        }

        throw;
    }
}

// ============== Phase 10: GPU-Resident KNN Graph Building ==============

/**
 * 在 GPU 上构建 centroid KNN 图，结果直接保留在 GPU 上。
 *
 * @param d_centroids_f32   centroid 数据 (已在 GPU, float32, n_centroids*D)
 * @param n_centroids       centroid 数量
 * @param D                 向量维度
 * @param K                 KNN 图度数
 * @param d_graph           [out] KNN 图在 GPU 上的指针 (uint32, n_centroids*K)
 * @param max_iterations    NN Descent 最大迭代次数
 * @param termination_threshold 收敛阈值
 */
void build_centroid_knn_on_gpu(
    const float* d_centroids_f32,
    int64_t n_centroids,
    int64_t D,
    uint32_t K,
    uint32_t** d_graph,
    size_t max_iterations = 20,
    double termination_threshold = 0.0001)
{
    if (K == 0 || K > LoadConfig::MAX_KNN_K)
        throw std::runtime_error("K must be in [1, " + std::to_string(LoadConfig::MAX_KNN_K) + "]");
    if (static_cast<int64_t>(K) >= n_centroids)
        throw std::runtime_error("K must be < n_centroids");

    uint32_t intermediate_graph_degree = std::max(K * 2, (uint32_t)64);
    if (static_cast<int64_t>(intermediate_graph_degree) >= n_centroids)
        intermediate_graph_degree = static_cast<uint32_t>(n_centroids) - 1;
    if (intermediate_graph_degree < K)
        intermediate_graph_degree = K;

    std::cout << "[CAGRA KNN - GPU Resident] n=" << n_centroids
              << ", K=" << K
              << ", intermediate=" << intermediate_graph_degree << "\n";

    // Copy centroid data from GPU to CPU (NN Descent requires host data)
    std::vector<float> centroids_host(n_centroids * D);
    CUDA_CHECK(cudaMemcpy(centroids_host.data(), d_centroids_f32,
                          n_centroids * D * sizeof(float), cudaMemcpyDeviceToHost));

    raft::neighbors::experimental::nn_descent::index_params params;
    params.graph_degree              = intermediate_graph_degree;
    params.intermediate_graph_degree = static_cast<size_t>(1.5 * intermediate_graph_degree);
    params.max_iterations            = max_iterations;
    params.termination_threshold     = termination_threshold;
    params.return_distances          = false;

    auto dataset = raft::make_host_matrix_view<const float, int64_t>(
        centroids_host.data(), n_centroids, D);

    raft::resources res;

    std::cout << "  Step 1/3: NN Descent...\n";
    auto idx = raft::neighbors::experimental::nn_descent::detail::build<float, uint32_t>(
        res, params, dataset);

    std::cout << "  Step 2/3: Sort by L2...\n";
    raft::neighbors::cagra::detail::graph::sort_knn_graph(res, dataset, idx.graph());

    std::cout << "  Step 3/3: CAGRA prune to K=" << K << "...\n";
    auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(n_centroids, K);
    raft::neighbors::cagra::detail::graph::optimize(
        res, idx.graph(), cagra_graph.view());

    // Upload pruned graph to GPU
    size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(d_graph, graph_bytes));
    CUDA_CHECK(cudaMemcpy(*d_graph, cagra_graph.data_handle(),
                          graph_bytes, cudaMemcpyHostToDevice));

    std::cout << "  KNN graph on GPU: " << graph_bytes / 1e6 << " MB\n";
}

// ============== Phase 11: Bucket Disk Writer ==============

/**
 * 将每个 centroid 的 bucket（包含的数据点 global index）写入磁盘。
 *
 * 磁盘格式（紧凑二进制，方便随机读取单个 bucket）：
 *
 * 文件: <output_dir>/bucket_index.bin
 *   header:
 *     int64_t  n_centroids
 *     int64_t  N   (总数据点数)
 *   body (n_centroids 条记录):
 *     int64_t  offset     // 在 bucket_data.bin 中的字节偏移
 *     int32_t  count      // 该 bucket 包含的点数
 *
 * 文件: <output_dir>/bucket_data.bin
 *   连续存储每个 bucket 的 int32_t point_ids[]
 *   (用 int32 而非 int64，因为 N < 2^31 时节省一半空间)
 *
 * 读取 bucket c 的方法:
 *   1. 从 bucket_index.bin 读取 offset[c] 和 count[c]
 *   2. seek 到 bucket_data.bin 的 offset[c]
 *   3. 读取 count[c] 个 int32_t
 */
void write_buckets_to_disk(
    const std::string& output_dir,
    const std::vector<int64_t>& assignments,
    int64_t N,
    int64_t n_centroids)
{
    std::cout << "[WriteBuckets] Writing to " << output_dir << "\n";

    std::filesystem::create_directories(output_dir);

    // Step 1: Gather points per centroid
    std::vector<std::vector<int32_t>> buckets(n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        int64_t c = assignments[i];
        if (c >= 0 && c < n_centroids) {
            buckets[c].push_back(static_cast<int32_t>(i));
        }
    }

    // Step 2: Write bucket_data.bin (contiguous int32 arrays)
    std::string data_path = output_dir + "/bucket_data.bin";
    std::ofstream data_out(data_path, std::ios::binary);
    if (!data_out.is_open())
        throw std::runtime_error("Cannot open: " + data_path);

    std::vector<int64_t> offsets(n_centroids);
    std::vector<int32_t> counts(n_centroids);
    int64_t current_offset = 0;

    for (int64_t c = 0; c < n_centroids; ++c) {
        offsets[c] = current_offset;
        counts[c] = static_cast<int32_t>(buckets[c].size());

        if (!buckets[c].empty()) {
            data_out.write(reinterpret_cast<const char*>(buckets[c].data()),
                           buckets[c].size() * sizeof(int32_t));
        }
        current_offset += static_cast<int64_t>(buckets[c].size()) * sizeof(int32_t);
    }
    data_out.close();

    // Step 3: Write bucket_index.bin (header + offset/count table)
    std::string index_path = output_dir + "/bucket_index.bin";
    std::ofstream index_out(index_path, std::ios::binary);
    if (!index_out.is_open())
        throw std::runtime_error("Cannot open: " + index_path);

    // Header
    index_out.write(reinterpret_cast<const char*>(&n_centroids), sizeof(int64_t));
    index_out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));

    // Offset + count per centroid
    for (int64_t c = 0; c < n_centroids; ++c) {
        index_out.write(reinterpret_cast<const char*>(&offsets[c]), sizeof(int64_t));
        index_out.write(reinterpret_cast<const char*>(&counts[c]), sizeof(int32_t));
    }
    index_out.close();

    // Stats
    size_t data_bytes = static_cast<size_t>(current_offset);
    size_t index_bytes = sizeof(int64_t) * 2 + n_centroids * (sizeof(int64_t) + sizeof(int32_t));
    std::cout << "[WriteBuckets] Done. data=" << data_bytes / 1e6
              << "MB, index=" << index_bytes / 1e3 << "KB\n";
}

// ============== Phase 11.5: Centroid Neighbor Expansion (K → nprobe) ==============

/**
 * GPU greedy graph search: 为每个 centroid 找 top-nprobe 个最近 centroid。
 *
 * 流程:
 *   1) 上传 centroids + graph 到 GPU
 *   2) 启动 greedy_graph_search_topK_kernel (每个 centroid 一个线程)
 *   3) 下载结果到 CPU
 *
 * 注意: 输入的 graph_host 是 CAGRA-pruned navigable graph (不是 sorted KNN),
 *       所以即使 nprobe <= K 也必须在图上 search, 不能直接截前 nprobe 个。
 *
 * @param centroids_host  (n_centroids, D) CPU centroid 坐标
 * @param graph_host      (n_centroids, K) CPU 上的 navigable graph
 * @param n_centroids     centroid 数
 * @param D               维度
 * @param K               图度数
 * @param nprobe          目标邻居数
 * @param max_iters       图搜索最大迭代轮数
 * @return                (n_centroids, nprobe) 真实近似 KNN 邻居 idx
 */
std::vector<uint32_t> expand_centroid_neighbors_gpu(
    const std::vector<float>& centroids_host,
    const std::vector<uint32_t>& graph_host,
    int64_t n_centroids,
    int D,
    uint32_t K,
    uint32_t nprobe,
    int /*max_iters*/)
{
    constexpr uint32_t MAX_NPROBE = 1024;
    if (nprobe > MAX_NPROBE) {
        throw std::runtime_error(
            "nprobe (" + std::to_string(nprobe) +
            ") exceeds MAX_NPROBE=" + std::to_string(MAX_NPROBE));
    }
    if (static_cast<int64_t>(nprobe) >= n_centroids) {
        throw std::runtime_error(
            "nprobe (" + std::to_string(nprobe) +
            ") must be < n_centroids=" + std::to_string(n_centroids));
    }

    // CAGRA self-search: 多搜 1 个候选, 用于 host 端剥掉自环 (验证 idx==qid 才剥)
    const uint32_t k_search = nprobe + 1;
    // itopk_size 至少 k_search, 按 32 对齐, floor 64
    uint32_t itopk = std::max<uint32_t>(64, ((k_search + 31) / 32) * 32);

    std::cout << "[CentroidTopK] CAGRA search (self): K=" << K
              << " → nprobe=" << nprobe
              << ", k_search=" << k_search
              << ", itopk_size=" << itopk << "\n";

    // 1) 上传 centroids 和 graph 到 GPU
    float*    d_dataset    = nullptr;
    uint32_t* d_graph      = nullptr;
    uint32_t* d_neighbors  = nullptr;
    float*    d_distances  = nullptr;

    size_t dataset_bytes    = static_cast<size_t>(n_centroids) * D * sizeof(float);
    size_t graph_bytes      = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
    size_t neighbors_bytes  = static_cast<size_t>(n_centroids) * k_search * sizeof(uint32_t);
    size_t distances_bytes  = static_cast<size_t>(n_centroids) * k_search * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_dataset,    dataset_bytes));
    CUDA_CHECK(cudaMalloc(&d_graph,      graph_bytes));
    CUDA_CHECK(cudaMalloc(&d_neighbors,  neighbors_bytes));
    CUDA_CHECK(cudaMalloc(&d_distances,  distances_bytes));

    CUDA_CHECK(cudaMemcpy(d_dataset, centroids_host.data(),
                          dataset_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_graph, graph_host.data(),
                          graph_bytes, cudaMemcpyHostToDevice));

    // 2) 构造 cagra::index (zero-copy view, 不重建图)
    raft::resources res;
    auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(
        d_dataset, n_centroids, static_cast<int64_t>(D));
    auto graph_view   = raft::make_device_matrix_view<const uint32_t, int64_t>(
        d_graph, n_centroids, static_cast<int64_t>(K));
    raft::neighbors::cagra::index<float, uint32_t> idx(
        res, raft::distance::DistanceType::L2Expanded,
        dataset_view, graph_view);

    // 3) cagra::search (queries == dataset, k = nprobe+1)
    raft::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;  // 0 = auto
    sp.algo           = raft::neighbors::cagra::search_algo::SINGLE_CTA;

    auto queries_view = raft::make_device_matrix_view<const float, int64_t>(
        d_dataset, n_centroids, static_cast<int64_t>(D));
    auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        d_neighbors, n_centroids, static_cast<int64_t>(k_search));
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
        d_distances, n_centroids, static_cast<int64_t>(k_search));

    raft::neighbors::cagra::search(res, sp, idx,
        queries_view, neighbors_view, distances_view);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4) 下载 (n_centroids, k_search) 邻居
    std::vector<uint32_t> raw(static_cast<size_t>(n_centroids) * k_search);
    CUDA_CHECK(cudaMemcpy(raw.data(), d_neighbors,
                          neighbors_bytes, cudaMemcpyDeviceToHost));

    // 5) Host 端剥自环: 扫描每行, 找到 idx == qid 才 skip 一次, 取剩下 nprobe 个
    std::vector<uint32_t> result(static_cast<size_t>(n_centroids) * nprobe,
                                 0xFFFFFFFFu);
    #pragma omp parallel for schedule(static)
    for (int64_t qid = 0; qid < n_centroids; ++qid) {
        const uint32_t* row_in  = raw.data() + qid * k_search;
        uint32_t*       row_out = result.data() + qid * nprobe;
        uint32_t self_qid = static_cast<uint32_t>(qid);
        bool self_skipped = false;
        uint32_t written = 0;
        for (uint32_t i = 0; i < k_search && written < nprobe; ++i) {
            uint32_t nb = row_in[i];
            if (!self_skipped && nb == self_qid) {
                self_skipped = true;   // 验证 idx == qid 才剥
                continue;
            }
            row_out[written++] = nb;
        }
    }

    // 释放
    cudaFree(d_dataset);
    cudaFree(d_graph);
    cudaFree(d_neighbors);
    cudaFree(d_distances);

    return result;
}

// ============== Phase 12: Per-Vector KNN via Tensor Core Bucket MatMul ==============

// 计算 X 中每行的 L2 范数平方: norms[i] = sum_d X[i,d]^2
// 模板：DataT 是输入元素类型 (uint8 / int8 / int32 / uint32 / float / half)，
//       内部累加和输出仍用 float (norms 始终 fp32)。
template <typename DataT>
__global__ void compute_row_norms_kernel(
    const DataT* __restrict__ X,
    float* __restrict__ norms,
    int64_t N, int64_t D)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const DataT* row = X + i * D;
    float s = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        float v = static_cast<float>(row[d]);
        s += v * v;
    }
    norms[i] = s;
}

// 按 int32 索引从 src (DataT) 中 gather 行，并转换成 float 写入 dst
//   dst[k, :] = float(src[idx[k], :])
// 这样 d_X_full 可以以 DataT (省 GPU 显存)，per-bucket A/B 仍是 float (走 cuBLAS fp32)。
template <typename DataT>
__global__ void gather_rows_int32(
    const DataT* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;
    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = static_cast<float>(src[static_cast<int64_t>(idx[row]) * D + col]);
}

// === Stage 2: INT8 IMMA path helpers ===
// gather rows 不做类型转换：dst 与 src 同 DataT，cuBLAS INT8 GEMM 直接吃。
template <typename DataT>
__global__ void gather_rows_raw(
    const DataT* __restrict__ src,
    const int32_t* __restrict__ idx,
    DataT* __restrict__ dst,
    int64_t n_dst_rows, int64_t D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = n_dst_rows * D;
    if (tid >= total) return;
    int64_t row = tid / D;
    int64_t col = tid % D;
    dst[tid] = src[static_cast<int64_t>(idx[row]) * D + col];
}

// uint8 → int8 in-place 平移：每个 byte 减 128，结果按 int8 解释。
//   uint8 0   → -128  (0x80 unchanged byte 模式)
//   uint8 128 →  0
//   uint8 255 →  127
// 调用方在 shift 之后用 reinterpret_cast<int8_t*> 拿同一块显存的 int8 视图。
// L2 距离平移不变，只要 norms 和 dot 都从同一个 (shifted) 视图算出来即可。
__global__ void shift_uint8_to_int8_inplace(
    uint8_t* __restrict__ data, int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    // (int)data - 128 然后回写 uint8：byte pattern 与 (int8)(int - 128) 一致
    int v = static_cast<int>(data[tid]) - 128;
    data[tid] = static_cast<uint8_t>(v);
}

// ====================================================================
// RAFT select_k path: dots → dist 原地转换 + RAFT select_k + gather global id
// ====================================================================

// 原地把 dots（int32 或 float）改写成 float dist = norms_pool[j] - 2*dots[i,j]，
// 顺便把 self-loop（cand_gid == my_gid）置 +inf 让 select_k 自动跳过。
// 复用 d_dots 的同一段显存（4 字节宽度相同）→ 零额外 buffer。
template<typename DotsT>
__global__ void dots_to_dist_inplace_kernel(
    DotsT* dots,                                 // (bucket_size, pool_size)
    const float* __restrict__ norms_pool,
    const int32_t* __restrict__ global_ids_bucket,
    const int32_t* __restrict__ global_ids_pool,
    int bucket_size, int pool_size)
{
    static_assert(sizeof(DotsT) == sizeof(float),
                  "DotsT must be 4 bytes for in-place reuse");
    int64_t tid   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(bucket_size) * pool_size;
    if (tid >= total) return;

    int i = static_cast<int>(tid / pool_size);
    int j = static_cast<int>(tid - static_cast<int64_t>(i) * pool_size);

    float dot = static_cast<float>(dots[tid]);
    float dist = norms_pool[j] - 2.0f * dot;
    if (global_ids_pool[j] == global_ids_bucket[i]) dist = INFINITY;

    reinterpret_cast<float*>(dots)[tid] = dist;
}

// RAFT select_k 输出的是 pool 内的 local index（0..pool_size-1）；
// 这个 kernel 把它映射成 global id 写到最终 out_neighbors。
__global__ void gather_global_ids_kernel(
    const int32_t* __restrict__ out_local_idx,    // (bucket_size, K)
    const int32_t* __restrict__ global_ids_pool,  // (pool_size,)
    int32_t* __restrict__ out_neighbors,          // (bucket_size, K)
    int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int32_t local = out_local_idx[tid];
    out_neighbors[tid] = (local >= 0) ? global_ids_pool[local] : -1;
}

// RAFT-based dispatcher：复用现有 d_dots buffer 当 dist 输入。
// 比自己写的 warp kernel 性能更好（RAFT 自动在 radix-select / warp-sort 之间选）。
// 唯一额外开销：dots→dist 转换（每 bucket ~10 MB DRAM 写）+ select_k 内部读
// + global_id gather（K × bucket_size 个 int32）。
//
// 调用方需要分配两个额外 per-slot buffer：
//   - d_select_idx : (max_bucket_size × M) int32 — RAFT 输出的 local pool idx
//   - d_select_dist: (max_bucket_size × M) float — RAFT 必填距离输出（我们不用）
template<typename DotsT>
static inline void launch_extract_topM_raft(
    raft::resources& res,
    cudaStream_t stream,
    DotsT*         d_dots,                  // 原地改写为 float dist
    const float*   d_norms_pool,
    const int32_t* d_ids_bucket,
    const int32_t* d_ids_pool,
    int32_t* d_select_idx,                  // RAFT idx 输出 scratch
    float*   d_select_dist,                 // RAFT dist 输出 scratch (也是最终距离输出)
    int32_t* d_out_neighbors,               // 最终输出 (邻居 global id)
    float*   d_out_distances,               // 可选, nullptr 表示不导出距离
    int bucket_size, int pool_size, int actual_M)
{
    // 1) dots → dist (in-place, 含 self-loop +inf)
    {
        constexpr int threads = 256;
        int64_t total = static_cast<int64_t>(bucket_size) * pool_size;
        int64_t blocks = (total + threads - 1) / threads;
        dots_to_dist_inplace_kernel<DotsT><<<blocks, threads, 0, stream>>>(
            d_dots, d_norms_pool, d_ids_bucket, d_ids_pool,
            bucket_size, pool_size);
    }

    // 2) RAFT select_k: kAuto 会按 (batch, len, k) 自动选 radix-select / warp-sort
    raft::resource::set_cuda_stream(res, stream);

    auto in_val = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
        reinterpret_cast<const float*>(d_dots),
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(pool_size));
    auto out_val = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
        d_select_dist,
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(actual_M));
    auto out_idx = raft::make_device_matrix_view<int32_t, int64_t, raft::row_major>(
        d_select_idx,
        static_cast<int64_t>(bucket_size),
        static_cast<int64_t>(actual_M));

    raft::matrix::select_k<float, int32_t>(
        res,
        in_val,
        std::nullopt,        // in_idx: 隐式 0..pool_size-1
        out_val,
        out_idx,
        /*select_min=*/ true,
        /*sorted=*/    true);  // 输出按距离升序，跟原 kernel 行为一致

    // 3) local pool idx → global id
    {
        constexpr int threads = 256;
        int64_t total = static_cast<int64_t>(bucket_size) * actual_M;
        int64_t blocks = (total + threads - 1) / threads;
        gather_global_ids_kernel<<<blocks, threads, 0, stream>>>(
            d_select_idx, d_ids_pool, d_out_neighbors, total);
    }

    // 4) (可选) 把 RAFT 选出的距离拷到调用方提供的 d_out_distances
    if (d_out_distances != nullptr) {
        size_t nbytes = static_cast<size_t>(bucket_size) * actual_M * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_out_distances, d_select_dist,
                                   nbytes, cudaMemcpyDeviceToDevice, stream));
    }
}

// ====================================================================
// Warp-cooperative top-K kernel
// ====================================================================
// 一个 warp（32 lane）共同处理一个 query。每 lane 在寄存器里持有
// K_PER_LANE = K_OUT/32 个元素，全展开排序——彻底干掉 local memory。
//
// Invariants:
//   - lane k 的 reg_dist[0..K_PER_LANE-1] 升序
//   - 跨 lane: lane 0 持最小的 K_PER_LANE 个，lane 31 持最大的 K_PER_LANE 个
//   - threshold = lane 31 的 reg_dist[K_PER_LANE-1]，broadcast 给所有 lane
//
// 主循环每次取 32 个 candidate（每 lane 一个），先 ballot 快速过滤
// （绝大多数 batch 0 个 qualified，整批跳过）；少数 qualified 的按
// cascade 模式插入：找到目标 lane → 该 lane 全展开 bubble → 被挤掉的
// max 通过 __shfl_up_sync 传给下一 lane → 一路到 lane 31 把全局 max 丢掉。
//
// 模板参数 K_OUT 必须是 32 的倍数（K_OUT ∈ {32,64,128,256,512,1024}）。
// K_PER_LANE = K_OUT/32, 最大 32（K_OUT=1024）→ ~94 reg/lane, V100 占用 ~50%。
// DotsT = int32_t（INT8 IMMA 路径）或 float（FP32 TF32 路径）。
template<int K_OUT, typename DotsT>
__global__ void warp_topK_from_dots_kernel(
    const DotsT* __restrict__ dots,
    const float* __restrict__ norms_pool,
    const int32_t* __restrict__ global_ids_bucket,
    const int32_t* __restrict__ global_ids_pool,
    int32_t* __restrict__ out_neighbors,
    float*   __restrict__ out_dists,   // 可选, NULL 表示不写距离
    int bucket_size, int pool_size, int K)
{
    static_assert(K_OUT % 32 == 0 && K_OUT >= 32,
                  "K_OUT must be a multiple of 32 (>=32)");
    constexpr int LANES = 32;
    constexpr int K_PER_LANE = K_OUT / LANES;
    constexpr unsigned ALL = 0xFFFFFFFFu;

    const int warps_per_block = blockDim.x / LANES;
    const int warp_in_block   = threadIdx.x / LANES;
    const int lane            = threadIdx.x & (LANES - 1);
    const int query_idx       = blockIdx.x * warps_per_block + warp_in_block;
    if (query_idx >= bucket_size) return;

    const int32_t my_gid = global_ids_bucket[query_idx];

    // 每 lane 的 K_PER_LANE 个 slot 进寄存器（K_PER_LANE 编译期已知，
    // 索引在全展开后都是常量）。ptxas 报告里这部分应当 STACK:0。
    float   reg_dist[K_PER_LANE];
    int32_t reg_id  [K_PER_LANE];

    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        reg_dist[p] = INFINITY;
        reg_id  [p] = -1;
    }

    float threshold = INFINITY;

    const DotsT* dots_row = dots + static_cast<int64_t>(query_idx) * pool_size;

    for (int j_base = 0; j_base < pool_size; j_base += LANES) {
        int  j     = j_base + lane;
        bool valid = (j < pool_size);

        float   my_dist;
        int32_t my_id;
        if (valid) {
            float d = static_cast<float>(dots_row[j]);
            my_dist = norms_pool[j] - 2.0f * d;
            my_id   = global_ids_pool[j];
            if (my_id == my_gid) my_dist = INFINITY;
        } else {
            my_dist = INFINITY;
            my_id   = -1;
        }

        unsigned mask = __ballot_sync(ALL, my_dist < threshold);
        if (mask == 0) continue;

        while (mask) {
            int src = __ffs(mask) - 1;
            mask &= ~(1u << src);

            float   c_dist = __shfl_sync(ALL, my_dist, src);
            int32_t c_id   = __shfl_sync(ALL, my_id,   src);
            if (c_dist >= threshold) continue;

            unsigned accept = __ballot_sync(ALL, c_dist <= reg_dist[K_PER_LANE - 1]);
            int target = __ffs(accept) - 1;

            float   old_max    = reg_dist[K_PER_LANE - 1];
            int32_t old_max_id = reg_id  [K_PER_LANE - 1];

            float   in_dist = __shfl_up_sync(ALL, old_max,    1);
            int32_t in_id   = __shfl_up_sync(ALL, old_max_id, 1);

            if (lane > target) {
                #pragma unroll
                for (int p = K_PER_LANE - 1; p > 0; --p) {
                    reg_dist[p] = reg_dist[p - 1];
                    reg_id  [p] = reg_id  [p - 1];
                }
                reg_dist[0] = in_dist;
                reg_id  [0] = in_id;
            } else if (lane == target) {
                reg_dist[K_PER_LANE - 1] = c_dist;
                reg_id  [K_PER_LANE - 1] = c_id;
                #pragma unroll
                for (int p = K_PER_LANE - 1; p > 0; --p) {
                    bool sw = reg_dist[p] < reg_dist[p - 1];
                    float   td  = sw ? reg_dist[p - 1] : reg_dist[p];
                    float   te  = sw ? reg_dist[p]     : reg_dist[p - 1];
                    int32_t tdi = sw ? reg_id  [p - 1] : reg_id  [p];
                    int32_t tei = sw ? reg_id  [p]     : reg_id  [p - 1];
                    reg_dist[p]     = td;
                    reg_dist[p - 1] = te;
                    reg_id  [p]     = tdi;
                    reg_id  [p - 1] = tei;
                }
            }

            threshold = __shfl_sync(ALL, reg_dist[K_PER_LANE - 1], LANES - 1);
        }
    }

    int base = lane * K_PER_LANE;
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        if (base + p < K) {
            out_neighbors[static_cast<int64_t>(query_idx) * K + base + p] = reg_id[p];
            if (out_dists != nullptr) {
                out_dists[static_cast<int64_t>(query_idx) * K + base + p] = reg_dist[p];
            }
        }
    }
}

// 按 int32 索引从 src 中 gather 标量: dst[k] = src[idx[k]]
__global__ void gather_floats_int32(
    const float* __restrict__ src,
    const int32_t* __restrict__ idx,
    float* __restrict__ dst,
    int64_t n)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = src[static_cast<int64_t>(idx[tid])];
}

// Host dispatcher: 按 actual_M 选最小的 K_OUT ∈ {32,64,128,256,512}。
// 用 warp-cooperative kernel: 4 warps/block → 4 queries/block，
// blocks = ceil(bucket_size/4)。avg bucket=100 → 25 blocks，比原 per-thread
// 的 1 block 高 25×，SM 利用率从 1/80 → ~25/80。
template<typename DotsT>
static inline void launch_extract_topM(
    cudaStream_t stream,
    const DotsT*   d_dots,
    const float*   d_norms_pool,
    const int32_t* d_ids_bucket,
    const int32_t* d_ids_pool,
    int32_t* d_out_neighbors,
    float*   d_out_distances,   // 可选, nullptr 表示不写距离
    int bucket_size, int pool_size, int actual_M)
{
    constexpr int LANES = 32;
    // 按 bucket 大小自适应 warps_per_block：
    //   小 bucket → 1 warp/block，让 block 散到尽量多 SM 上（单 query 也独立调度）
    //   大 bucket → 多 warp/block，减少 launch overhead，单 SM occupancy 更高
    int warps_per_block;
    if      (bucket_size <= 64)    warps_per_block = 1;
    else if (bucket_size <= 256)   warps_per_block = 2;
    else if (bucket_size <= 1024)  warps_per_block = 4;
    else                            warps_per_block = 8;
    int threads_per_block = LANES * warps_per_block;
    int blocks = (bucket_size + warps_per_block - 1) / warps_per_block;

#define LAUNCH_KOUT(KV) do {                                                                  \
    if constexpr (std::is_same<DotsT, int32_t>::value) {                                      \
        warp_topK_from_dots_kernel<(KV), int32_t>                                             \
            <<<blocks, threads_per_block, 0, stream>>>(                                       \
            reinterpret_cast<const int32_t*>(d_dots), d_norms_pool,                           \
            d_ids_bucket, d_ids_pool,                                                         \
            d_out_neighbors, d_out_distances,                                                 \
            bucket_size, pool_size, actual_M);                                                \
    } else {                                                                                  \
        warp_topK_from_dots_kernel<(KV), float>                                               \
            <<<blocks, threads_per_block, 0, stream>>>(                                       \
            reinterpret_cast<const float*>(d_dots), d_norms_pool,                             \
            d_ids_bucket, d_ids_pool,                                                         \
            d_out_neighbors, d_out_distances,                                                 \
            bucket_size, pool_size, actual_M);                                                \
    }                                                                                         \
} while (0)

    if      (actual_M <= 32)   LAUNCH_KOUT(32);
    else if (actual_M <= 64)   LAUNCH_KOUT(64);
    else if (actual_M <= 128)  LAUNCH_KOUT(128);
    else if (actual_M <= 256)  LAUNCH_KOUT(256);
    else if (actual_M <= 512)  LAUNCH_KOUT(512);
    else                       LAUNCH_KOUT(1024);

#undef LAUNCH_KOUT
}

// ============== Chunked disk-backed running per-vector KNN ==============
//
// 把 [0, N) 切成若干个固定大小、连续的 id 区间 (chunk)。Step 6 阶段每个点
// 算出的 M 个候选，按 (gid / chunk_size) 路由到对应 chunk 的内存 write buffer
// 里 (只按到达顺序追加，不要求有序)；buffer 攒满就整块顺序 flush 到这一轮专属
// 的 chunk 文件 (chunk_<c>_iter<t>.bin)——全程只有顺序追加写，没有随机 seek。
//
// 每轮 iteration 结束后 (merge_iteration)：对每个 chunk，把磁盘上已有的
// running 结果 (running_chunk_<c>_{nbrs,dists}.bin，是前面所有轮合并去重后
// 的状态) 整块顺序读进内存；再顺序扫一遍这一轮的 arrival-order 文件，每条
// 记录按 (gid - chunk_start) 算出的位置，跟 running 里对应位置的 M 个候选就
// 地合并去重、取 top-M；最后把更新后的 running 整块顺序写回磁盘，删掉这一轮
// 的 arrival 文件。第 0 轮没有已有的 running，直接当成全 sentinel (-1/+inf)
// 处理，走同一条合并逻辑。不同 chunk 相互独立，用 OpenMP 并行处理。
//
// 内存峰值只跟 chunk_size 有关 (running 的两个 dense 数组，各 chunk_size*M*
// 4 字节)，与 N 无关——只要 chunk_size 选得够小，数据集多大都不会撑爆内存；
// 用磁盘换内存的同时，全程没有随机 I/O。
struct ChunkedKnnAccumulator {
    std::string dir;
    int64_t N = 0;
    int     M = 0;
    int64_t chunk_size = 1;
    int64_t num_chunks = 1;
    int     merge_parallelism = 1;

    static constexpr size_t record_bytes(int M) {
        return sizeof(int64_t) + static_cast<size_t>(M) * (sizeof(int32_t) + sizeof(float));
    }

    struct ChunkBuf {
        std::vector<char> data;     // capacity_records * record_bytes(M)
        size_t capacity_records = 0;
        size_t count = 0;
        std::ofstream file;
    };
    std::vector<ChunkBuf> bufs;

    static std::string iter_path(const std::string& dir, int64_t c, int iter) {
        return dir + "/chunk_" + std::to_string(c) + "_iter" + std::to_string(iter) + ".bin";
    }
    static std::string running_nbrs_path(const std::string& dir, int64_t c) {
        return dir + "/running_chunk_" + std::to_string(c) + "_nbrs.bin";
    }
    static std::string running_dists_path(const std::string& dir, int64_t c) {
        return dir + "/running_chunk_" + std::to_string(c) + "_dists.bin";
    }

    // dir 必须已存在 (调用方负责创建); cpu_mem_budget_bytes 用来限制 merge
    // 阶段并行处理多少个 chunk (每个 chunk 需要 chunk_size*M*4B*2 常驻内存)。
    static ChunkedKnnAccumulator create(const std::string& dir, int64_t N, int M,
                                        size_t cpu_mem_budget_bytes) {
        ChunkedKnnAccumulator acc;
        acc.dir = dir;
        acc.N = N;
        acc.M = M;

        // chunk 数量目标 ~500k 点/chunk，但不超过 512 个 (同时打开的文件句柄
        // 数上限，避免撞 ulimit -n)；N 很大时 chunk_size 会自动放大来满足这
        // 个上限。
        constexpr int64_t kTargetChunkPoints = 500000;
        constexpr int64_t kMaxOpenChunks = 512;
        int64_t n_by_target = std::max<int64_t>(1, (N + kTargetChunkPoints - 1) / kTargetChunkPoints);
        acc.num_chunks = std::min(n_by_target, kMaxOpenChunks);
        acc.chunk_size = (N + acc.num_chunks - 1) / acc.num_chunks;

        size_t per_chunk_bytes = static_cast<size_t>(acc.chunk_size) * M * 2 * sizeof(int32_t);
        int max_threads = omp_get_max_threads();
        int budget_threads = static_cast<int>(std::max<size_t>(1,
            cpu_mem_budget_bytes / std::max<size_t>(1, per_chunk_bytes)));
        acc.merge_parallelism = std::max(1, std::min(max_threads, budget_threads));

        constexpr size_t kBufferBytes = 256 * 1024;
        size_t rec_bytes = record_bytes(M);
        size_t cap_records = std::max<size_t>(1, kBufferBytes / rec_bytes);

        acc.bufs.resize(static_cast<size_t>(acc.num_chunks));
        for (auto& b : acc.bufs) {
            b.capacity_records = cap_records;
            b.data.resize(cap_records * rec_bytes);
        }

        std::cout << "[ChunkedKNN] N=" << N << " M=" << M
                  << " num_chunks=" << acc.num_chunks
                  << " chunk_size=" << acc.chunk_size
                  << " write_buffer=" << (cap_records * rec_bytes) / 1024 << "KB/chunk"
                  << " merge_parallelism=" << acc.merge_parallelism << "\n";
        return acc;
    }

    void start_iteration(int iter) {
        for (int64_t c = 0; c < num_chunks; ++c) {
            auto& b = bufs[static_cast<size_t>(c)];
            b.count = 0;
            b.file.open(iter_path(dir, c, iter),
                       std::ios::binary | std::ios::out | std::ios::trunc);
            if (!b.file.is_open())
                throw std::runtime_error("ChunkedKnnAccumulator: cannot open " + iter_path(dir, c, iter));
        }
    }

    void flush_chunk(int64_t c) {
        auto& b = bufs[static_cast<size_t>(c)];
        if (b.count == 0) return;
        size_t rec_bytes = record_bytes(M);
        b.file.write(b.data.data(), static_cast<std::streamsize>(b.count * rec_bytes));
        if (!b.file.good())
            throw std::runtime_error("ChunkedKnnAccumulator: flush failed for chunk " + std::to_string(c));
        b.count = 0;
    }

    // Step 6 的 scatter_pending 里，每个点调用一次。nbrs/dists 长度均为 M
    // (已按距离升序排列，不足 M 的部分是 -1/+inf sentinel)。只做内存追加 +
    // 偶尔的整块顺序 flush，没有 seek。
    void add(int64_t gid, const int32_t* nbrs, const float* dists) {
        int64_t c = std::min(gid / chunk_size, num_chunks - 1);
        auto& b = bufs[static_cast<size_t>(c)];
        size_t rec_bytes = record_bytes(M);
        size_t off = b.count * rec_bytes;
        std::memcpy(b.data.data() + off, &gid, sizeof(int64_t));
        std::memcpy(b.data.data() + off + sizeof(int64_t), nbrs,
                   static_cast<size_t>(M) * sizeof(int32_t));
        std::memcpy(b.data.data() + off + sizeof(int64_t) + static_cast<size_t>(M) * sizeof(int32_t),
                   dists, static_cast<size_t>(M) * sizeof(float));
        if (++b.count == b.capacity_records) flush_chunk(c);
    }

    void finish_iteration_writes() {
        for (int64_t c = 0; c < num_chunks; ++c) {
            flush_chunk(c);
            bufs[static_cast<size_t>(c)].file.close();
        }
    }

    // 把新算出的 M 个候选(new_*)就地合并进 running 位置的 M 个候选(run_*)，
    // 去重、按距离升序取前 M。scratch 由调用方按线程复用，避免每点分配。
    //
    // run_*/new_* 两边各自都已经是"有效候选在前、按距离升序排列，多余的槽位
    // 是 -1/+inf sentinel"的布局(GPU top-M 输出和上一次 merge_one_record 的
    // 输出都保证这一点)——所以不需要拼一起整体 std::sort (O(2M log 2M))，
    // 直接对两个已排序列表做双指针归并 (O(M)) 就够了。
    //
    // 去重故意不用 unordered_set: M 很小 (通常几十到一百多，上限 1024)，
    // node-based 的 unordered_set 每次 insert 都会 malloc 一个新节点——即使
    // 复用同一个 set 对象、每次调用后 clear()，clear() 也会把内部节点整个
    // 释放掉，下次 insert 照样重新 malloc。实测这才是 merge_cpu 里的真正大头。
    //
    // na>0 且 nb>0 时(两边都有数据，只有这种情况才需要真正判重复)：run_*
    // 侧自己没有内部重复(它本身就是上一次 merge_one_record 的输出)，所以把
    // 它的 id 排序一份，new_* 侧每个 id 用二分查找判断是否在 run_* 里出现过
    // 即可——重复只可能是"同一个 id 两边都有"，不会是 run_* 或 new_* 内部
    // 自己有重复。run_* 侧的元素直接按距离顺序原样取用，不需要查重(它是否
    // 重复，会在 new_* 那一侧的二分查找里被发现并跳过)；两边距离值对同一个
    // id 是确定性复现的(相同公式)，所以留哪一侧的副本不影响结果。这样只需
    // 一次 O(na log na) 排序 + 最多 nb 次 O(log na) 二分查找，比对 scratch
    // 做 O(M^2) 线性扫描快得多。
    static void merge_one_record(
        int32_t* run_nbrs, float* run_dists,
        const int32_t* new_nbrs, const float* new_dists, int M,
        std::vector<std::pair<float,int32_t>>& scratch,
        std::vector<int32_t>& id_sort_buf) {
        int na = 0; while (na < M && run_nbrs[na] >= 0) ++na;
        int nb = 0; while (nb < M && new_nbrs[nb] >= 0) ++nb;
        if (nb == 0) return;  // 这一轮没算出新候选 (bucket 被跳过等)，running 保持不变
        if (na == 0) {
            // running 侧还没有数据 (常见于 iter==0，或者这个点之前某轮所在的
            // bucket 被跳过)。new_* 本身保证无重复 id (bucket 互不相交 +
            // search pool 不重复，见 batch_assign_with_cagra_anns 的说明)，
            // 直接拷贝即可，不需要判重。
            std::memcpy(run_nbrs, new_nbrs, static_cast<size_t>(nb) * sizeof(int32_t));
            std::memcpy(run_dists, new_dists, static_cast<size_t>(nb) * sizeof(float));
            for (int k = nb; k < M; ++k) {
                run_nbrs[k] = -1;
                run_dists[k] = std::numeric_limits<float>::infinity();
            }
            return;
        }

        id_sort_buf.assign(run_nbrs, run_nbrs + na);
        std::sort(id_sort_buf.begin(), id_sort_buf.end());

        scratch.clear();
        int i = 0, j = 0;
        while (static_cast<int>(scratch.size()) < M && (i < na || j < nb)) {
            bool take_a = (j >= nb) || (i < na && run_dists[i] <= new_dists[j]);
            if (take_a) {
                scratch.push_back({run_dists[i], run_nbrs[i]});
                ++i;
            } else {
                int32_t id = new_nbrs[j];
                bool dup = std::binary_search(id_sort_buf.begin(), id_sort_buf.end(), id);
                if (!dup) scratch.push_back({new_dists[j], id});
                ++j;
            }
        }

        int written = static_cast<int>(scratch.size());
        for (int k = 0; k < written; ++k) {
            run_dists[k] = scratch[k].first;
            run_nbrs[k]  = scratch[k].second;
        }
        for (int k = written; k < M; ++k) {
            run_nbrs[k] = -1;
            run_dists[k] = std::numeric_limits<float>::infinity();
        }
    }

    // 每轮结束后调用一次: 对每个 chunk 做"读 running -> 顺序扫这一轮 arrival
    // 文件逐条就地合并 -> 写回 running"，chunk 之间用 OpenMP 并行。
    //
    // 每个点确定性地产生: 读 arrival 记录(record_bytes) + 写 running(2*M*4B)
    // + (iter>0 时还要) 读 running(2*M*4B)。这个量是精确算出来的，不是估计
    // 的——所以下面打印的 GB/s 就是这一步磁盘读写的真实吞吐，可以直接用来判断
    // merge 这一段是不是被磁盘带宽/IOPS 卡住了。
    // 每个 chunk 自己的耗时明细，按 c 直接索引写入 (不同线程处理不同 c，天然
    // 无竞争，不需要原子操作/锁)——不再用"总和除以线程数"这种在 chunk 数量比
    // 线程数少时会严重失真的平均法，每个 chunk 到底花了多少时间一目了然。
    struct ChunkTiming {
        double alloc = 0, read_running = 0, read_arrival = 0, merge_cpu = 0, write_running = 0, total = 0;
        int64_t n_records = 0;
        int thread_id = -1;
        double start_ms = 0, end_ms = 0;  // 相对 t_merge_start 的偏移，用来肉眼判断
                                           // 不同 chunk 的执行区间在墙钟上有没有真的重叠
    };

    void merge_iteration(int iter) {
        auto t_merge_start = std::chrono::high_resolution_clock::now();
        size_t rec_bytes = record_bytes(M);
        std::vector<ChunkTiming> timing(static_cast<size_t>(num_chunks));
        int actual_num_threads = 0;

        #pragma omp parallel num_threads(merge_parallelism)
        {
            #pragma omp single
            { actual_num_threads = omp_get_num_threads(); }
            std::vector<std::pair<float,int32_t>> scratch;
            scratch.reserve(static_cast<size_t>(2) * M);
            std::vector<int32_t> id_sort_buf;
            id_sort_buf.reserve(static_cast<size_t>(M));
            std::vector<int32_t> new_nbrs(static_cast<size_t>(M));
            std::vector<float>   new_dists(static_cast<size_t>(M));
            std::vector<char>    arrival_buf;
            // 按线程复用的 running 缓冲区，容量按最大 chunk (chunk_size) 一次
            // 分配到位；后面每个 chunk 只 resize (只有第一次真正触发分配，
            // 之后 resize 到更小/相等的大小不会重新分配)，避免每个 chunk 都
            // 重新 malloc 这两块大内存——chunk 数一多 (真实规模下有几百个)，
            // 这笔重复分配的开销不是小数。
            std::vector<int32_t> run_nbrs;
            std::vector<float>   run_dists;
            run_nbrs.reserve(static_cast<size_t>(chunk_size) * M);
            run_dists.reserve(static_cast<size_t>(chunk_size) * M);
            auto now = [] { return std::chrono::high_resolution_clock::now(); };
            auto secs_between = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

            #pragma omp for schedule(dynamic)
            for (int64_t c = 0; c < num_chunks; ++c) {
                auto t0 = now();
                int64_t chunk_start = c * chunk_size;
                int64_t chunk_c_size = std::min(chunk_size, N - chunk_start);
                if (chunk_c_size <= 0) continue;
                ChunkTiming& tm = timing[static_cast<size_t>(c)];
                tm.thread_id = omp_get_thread_num();
                tm.start_ms = secs_between(t_merge_start, t0) * 1000.0;

                run_nbrs.resize(static_cast<size_t>(chunk_c_size) * M);
                run_dists.resize(static_cast<size_t>(chunk_c_size) * M);
                auto t1 = now();
                tm.alloc = secs_between(t0, t1);

                std::string np = running_nbrs_path(dir, c);
                std::string dp = running_dists_path(dir, c);
                if (iter == 0) {
                    std::fill(run_nbrs.begin(), run_nbrs.end(), -1);
                    std::fill(run_dists.begin(), run_dists.end(),
                              std::numeric_limits<float>::infinity());
                } else {
                    std::ifstream nin(np, std::ios::binary);
                    std::ifstream din(dp, std::ios::binary);
                    if (!nin.is_open() || !din.is_open())
                        throw std::runtime_error("ChunkedKnnAccumulator: cannot read running chunk " + std::to_string(c));
                    nin.read(reinterpret_cast<char*>(run_nbrs.data()),
                             static_cast<std::streamsize>(run_nbrs.size() * sizeof(int32_t)));
                    din.read(reinterpret_cast<char*>(run_dists.data()),
                             static_cast<std::streamsize>(run_dists.size() * sizeof(float)));
                    if (!nin.good() || !din.good())
                        throw std::runtime_error("ChunkedKnnAccumulator: short read on running chunk " + std::to_string(c));
                }
                auto t2 = now();
                tm.read_running = secs_between(t1, t2);

                // 整个 arrival 文件一次性读进内存 (一次 read 系统调用，而不是
                // 每条记录一次)，读磁盘和后面逐条 merge 的 CPU 计算彻底分开计时。
                std::string ip = iter_path(dir, c, iter);
                std::ifstream in(ip, std::ios::binary);
                size_t n_records = 0;
                if (in.is_open()) {
                    in.seekg(0, std::ios::end);
                    std::streamoff sz = in.tellg();
                    in.seekg(0, std::ios::beg);
                    arrival_buf.resize(static_cast<size_t>(sz));
                    in.read(arrival_buf.data(), sz);
                    in.close();
                    n_records = arrival_buf.size() / rec_bytes;
                }
                tm.n_records = static_cast<int64_t>(n_records);
                auto t3 = now();
                tm.read_arrival = secs_between(t2, t3);

                for (size_t r = 0; r < n_records; ++r) {
                    const char* rp = arrival_buf.data() + r * rec_bytes;
                    int64_t gid;
                    std::memcpy(&gid, rp, sizeof(int64_t));
                    std::memcpy(new_nbrs.data(), rp + sizeof(int64_t),
                               static_cast<size_t>(M) * sizeof(int32_t));
                    std::memcpy(new_dists.data(),
                               rp + sizeof(int64_t) + static_cast<size_t>(M) * sizeof(int32_t),
                               static_cast<size_t>(M) * sizeof(float));
                    int64_t pos = gid - chunk_start;
                    if (pos < 0 || pos >= chunk_c_size) continue;  // 不应该发生，保险起见
                    merge_one_record(
                        run_nbrs.data() + static_cast<size_t>(pos) * M,
                        run_dists.data() + static_cast<size_t>(pos) * M,
                        new_nbrs.data(), new_dists.data(), M,
                        scratch, id_sort_buf);
                }
                if (n_records > 0) std::filesystem::remove(ip);
                auto t4 = now();
                tm.merge_cpu = secs_between(t3, t4);

                std::ofstream nout(np, std::ios::binary | std::ios::trunc);
                std::ofstream dout(dp, std::ios::binary | std::ios::trunc);
                nout.write(reinterpret_cast<const char*>(run_nbrs.data()),
                          static_cast<std::streamsize>(run_nbrs.size() * sizeof(int32_t)));
                dout.write(reinterpret_cast<const char*>(run_dists.data()),
                          static_cast<std::streamsize>(run_dists.size() * sizeof(float)));
                if (!nout.good() || !dout.good())
                    throw std::runtime_error("ChunkedKnnAccumulator: write failed for running chunk " + std::to_string(c));
                nout.close(); dout.close();
                auto t5 = now();
                tm.write_running = secs_between(t4, t5);
                tm.total = secs_between(t0, t5);
                tm.end_ms = secs_between(t_merge_start, t5) * 1000.0;
            }
        }

        double secs = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t_merge_start).count();
        size_t running_rw_bytes = static_cast<size_t>(M) * sizeof(int32_t) * 2;  // nbrs+dists 各 M*4B
        size_t per_point_bytes = rec_bytes + running_rw_bytes + (iter > 0 ? running_rw_bytes : 0);
        double total_gb = static_cast<double>(N) * static_cast<double>(per_point_bytes) / 1e9;

        // 汇总: 每个阶段跨 chunk 求和 (sum) 和取最慢的一个 chunk (max，用来看
        // 负载是否均衡)。sum 是"如果单线程串行跑完所有 chunk"需要的时间，不
        // 是墙钟时间——真正的墙钟时间是上面的 secs，两者的差距 (secs 明显小于
        // sum) 说明并行确实生效了；如果 secs 接近甚至大于 sum，说明没并行起来
        // 或者有锁/资源争用。
        ChunkTiming sum, mx;
        for (int64_t c = 0; c < num_chunks; ++c) {
            const auto& t = timing[static_cast<size_t>(c)];
            sum.alloc += t.alloc; sum.read_running += t.read_running;
            sum.read_arrival += t.read_arrival; sum.merge_cpu += t.merge_cpu;
            sum.write_running += t.write_running; sum.total += t.total;
            mx.alloc = std::max(mx.alloc, t.alloc); mx.read_running = std::max(mx.read_running, t.read_running);
            mx.read_arrival = std::max(mx.read_arrival, t.read_arrival); mx.merge_cpu = std::max(mx.merge_cpu, t.merge_cpu);
            mx.write_running = std::max(mx.write_running, t.write_running); mx.total = std::max(mx.total, t.total);
        }
        std::cout << "[ChunkedKNN] merge_iteration(iter=" << iter << "): "
                  << total_gb << " GB moved, wall=" << secs << "s "
                  << "(num_chunks=" << num_chunks << ", merge_parallelism=" << merge_parallelism
                  << ", actual_num_threads=" << actual_num_threads << ")\n"
                  << "  sum-over-chunks (serial-equivalent): alloc=" << sum.alloc
                  << "s read_running=" << sum.read_running << "s read_arrival=" << sum.read_arrival
                  << "s merge_cpu=" << sum.merge_cpu << "s write_running=" << sum.write_running
                  << "s total=" << sum.total << "s\n"
                  << "  slowest single chunk:                 alloc=" << mx.alloc
                  << "s read_running=" << mx.read_running << "s read_arrival=" << mx.read_arrival
                  << "s merge_cpu=" << mx.merge_cpu << "s write_running=" << mx.write_running
                  << "s total=" << mx.total << "s\n";
        // chunk 数不多时 (通常是小数据集测试)，直接把每个 chunk 的明细打出来，
        // 不需要靠猜。
        if (num_chunks <= 16) {
            for (int64_t c = 0; c < num_chunks; ++c) {
                const auto& t = timing[static_cast<size_t>(c)];
                std::cout << "    chunk " << c << ": thread=" << t.thread_id
                          << " [" << t.start_ms << "ms - " << t.end_ms << "ms] n_records=" << t.n_records
                          << " alloc=" << t.alloc << "s read_running=" << t.read_running
                          << "s read_arrival=" << t.read_arrival << "s merge_cpu=" << t.merge_cpu
                          << "s write_running=" << t.write_running << "s total=" << t.total << "s\n";
            }
        }
    }

    // 全部 iteration 跑完后调用一次: 按 chunk 顺序把 running_chunk_* 顺序拼
    // 接成最终的 vector_knn.bin / vector_dists.bin (格式跟原来 RunningKnnFile
    // 的输出完全一致: int64 N + int32 M header，后面跟 (N,M) flat body)。
    void finalize_to_file(const std::string& knn_path, const std::string& dist_path) {
        std::ofstream kout(knn_path, std::ios::binary | std::ios::trunc);
        std::ofstream dout(dist_path, std::ios::binary | std::ios::trunc);
        int32_t M32 = static_cast<int32_t>(M);
        kout.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
        kout.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));
        dout.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
        dout.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));

        for (int64_t c = 0; c < num_chunks; ++c) {
            int64_t chunk_start = c * chunk_size;
            int64_t chunk_c_size = std::min(chunk_size, N - chunk_start);
            if (chunk_c_size <= 0) continue;
            std::string np = running_nbrs_path(dir, c);
            std::string dp = running_dists_path(dir, c);
            std::vector<int32_t> run_nbrs(static_cast<size_t>(chunk_c_size) * M);
            std::vector<float>   run_dists(static_cast<size_t>(chunk_c_size) * M);
            std::ifstream nin(np, std::ios::binary);
            std::ifstream din(dp, std::ios::binary);
            if (!nin.is_open() || !din.is_open())
                throw std::runtime_error("ChunkedKnnAccumulator: cannot read running chunk " + std::to_string(c) + " for finalize");
            nin.read(reinterpret_cast<char*>(run_nbrs.data()),
                     static_cast<std::streamsize>(run_nbrs.size() * sizeof(int32_t)));
            din.read(reinterpret_cast<char*>(run_dists.data()),
                     static_cast<std::streamsize>(run_dists.size() * sizeof(float)));
            nin.close(); din.close();
            kout.write(reinterpret_cast<const char*>(run_nbrs.data()),
                      static_cast<std::streamsize>(run_nbrs.size() * sizeof(int32_t)));
            dout.write(reinterpret_cast<const char*>(run_dists.data()),
                      static_cast<std::streamsize>(run_dists.size() * sizeof(float)));
            std::filesystem::remove(np);
            std::filesystem::remove(dp);
        }
        if (!kout.good() || !dout.good())
            throw std::runtime_error("ChunkedKnnAccumulator: finalize write failed");
    }
};

constexpr size_t knn_file_header_bytes() { return sizeof(int64_t) + sizeof(int32_t); }

// 把 vector_knn.bin (RunningKnnFile 的 int64 N + int32 M header, flat int32
// body) 顺序分块转换成 neighbors.npy (int64)，给 Python 端评测用。纯顺序拷贝
// + 类型转换，不需要整份常驻内存。只在全部 iteration 跑完、running 文件已经
// close 之后调用一次。
inline void convert_vector_knn_to_npy(const std::string& knn_path, const std::string& npy_path,
                                      int64_t N, int M, size_t chunk_bytes_budget) {
    std::ifstream in(knn_path, std::ios::binary);
    if (!in.is_open()) throw std::runtime_error("Cannot open: " + knn_path);
    in.seekg(static_cast<std::streamoff>(knn_file_header_bytes()));

    size_t header_bytes = load::create_npy_int64_2d(npy_path, N, M);
    std::fstream out(npy_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out.is_open()) throw std::runtime_error("Cannot open for write: " + npy_path);
    out.seekp(static_cast<std::streamoff>(header_bytes));

    int64_t chunk_rows = std::max<int64_t>(1,
        static_cast<int64_t>(chunk_bytes_budget / (static_cast<size_t>(M) * sizeof(int64_t))));
    chunk_rows = std::min(chunk_rows, N);
    std::vector<int32_t> buf32(static_cast<size_t>(chunk_rows) * M);
    std::vector<int64_t> buf64(static_cast<size_t>(chunk_rows) * M);

    for (int64_t start = 0; start < N; start += chunk_rows) {
        int64_t cur = std::min(chunk_rows, N - start);
        size_t cnt = static_cast<size_t>(cur) * M;
        in.read(reinterpret_cast<char*>(buf32.data()), static_cast<std::streamsize>(cnt * sizeof(int32_t)));
        if (!in.good()) throw std::runtime_error("Failed reading " + knn_path);
        for (size_t i = 0; i < cnt; ++i) buf64[i] = static_cast<int64_t>(buf32[i]);
        out.write(reinterpret_cast<const char*>(buf64.data()),
                 static_cast<std::streamsize>(cnt * sizeof(int64_t)));
        if (!out.good()) throw std::runtime_error("Failed writing " + npy_path);
    }
}

/**
 * 为每个向量在其 bucket 及 K 个最近邻 bucket 内，用 Tensor Core 矩阵乘法
 * 计算距离并找到 M 个最近邻。
 *
 * 流程 (per bucket c):
 *   1) 收集 search pool: bucket c 自身 + K 个最近邻 bucket 的所有点
 *   2) 将 bucket c 的向量 (A) 和 pool 向量 (B) 上传 GPU
 *   3) cuBLAS GEMM: dots = A * B^T (Tensor Core, FP16 compute)
 *   4) CUDA kernel: 从 dots 矩阵中为 bucket 内每个点提取 top-M 最近邻
 *   5) 下载结果
 *
 * @param X_full              完整数据集 (CPU, float32, N*D)
 * @param N                   数据点总数
 * @param D                   向量维度
 * @param assignments         (N,) 每个点的 bucket (local centroid index)
 * @param centroid_knn_graph  (n_centroids, K) centroid KNN 图 (CPU, uint32)
 * @param n_centroids         centroid / bucket 数量
 * @param K                   centroid KNN 图度数
 * @param M                   每个向量要找的邻居数
 *
 * @param acc                 分块的 disk-backed 累积器 (见文件顶部 ChunkedKnnAccumulator
 *                            的说明)。每算完一个 bucket，就把该 bucket 里每个点新算出
 *                            的 M 个候选，按 gid 路由进对应 chunk 的内存 write buffer
 *                            (纯追加，不读旧值)；真正的跨 iteration 合并延后到调用方
 *                            在每轮结束后调 acc.merge_iteration() 时才做。
 *
 * 距离对合并是必需的 (要按距离排序去重)，所以内部始终按 want_distances=true
 * 的路径跑；不再对外暴露"不算距离"这个选项。
 */
template <typename DataT>
void build_vector_knn_with_tensorcore(
    const DataT* X_full,
    int64_t N,
    int64_t D,
    const std::vector<int64_t>& assignments,
    const std::vector<int64_t>& centroid_global_indices,
    const uint32_t* centroid_knn_graph,  // (n_centroids, K) row-major
    int64_t n_centroids,
    uint32_t K,
    int M,
    ChunkedKnnAccumulator& acc)
{
    constexpr bool want_distances = true;  // 合并去重总是需要距离
    // ============= Stage 2 路径选择（INT8 IMMA / fp32 fallback）=============
    // - int8/uint8 → 走 INT8 IMMA Tensor Core（uint8 入口先减 128 转 int8）
    // - 其它 (float/half/uint32/int32 等) → 走 fp32 cuBLAS GEMM（原行为）
    constexpr bool kIsInt8Path =
        std::is_same<DataT, int8_t>::value || std::is_same<DataT, uint8_t>::value;
    using GemmInT  = typename std::conditional<kIsInt8Path, int8_t, float>::type;
    using GemmOutT = typename std::conditional<kIsInt8Path, int32_t, float>::type;

    std::cout << "[VectorKNN] Building per-vector KNN with Tensor Core matmul\n"
              << "  N=" << N << ", D=" << D
              << ", n_centroids=" << n_centroids
              << ", K=" << K << ", M=" << M
              << ", path=" << (kIsInt8Path ? "INT8 IMMA" : "FP32 TF32") << "\n";

    // launch_extract_topM 最大特化档位是 K_OUT=1024；超过会跑错。
    if (M > 1024) {
        throw std::runtime_error(
            "build_vector_knn_with_tensorcore: M=" + std::to_string(M)
            + " exceeds kernel max K_OUT=1024. Lower M or add a larger K_OUT case in launch_extract_topM.");
    }

    // ================================================================
    // Step 0: Build in-memory bucket lists & ensure centroid is in its bucket
    // ================================================================
    std::vector<std::vector<int32_t>> buckets(n_centroids);
    for (int64_t i = 0; i < N; ++i) {
        int64_t c = assignments[i];
        if (c >= 0 && c < n_centroids) {
            buckets[c].push_back(static_cast<int32_t>(i));
        }
    }

    // 确认每个 centroid 在自己的 bucket 中
    for (int64_t c = 0; c < n_centroids; ++c) {
        int32_t centroid_gid = static_cast<int32_t>(centroid_global_indices[c]);
        bool found = false;
        for (int32_t pid : buckets[c]) {
            if (pid == centroid_gid) { found = true; break; }
        }
        if (!found) {
            std::cout << "  [WARN] Centroid " << c << " (global=" << centroid_gid
                      << ") not in its bucket, inserting.\n";
            buckets[c].push_back(centroid_gid);
        }
    }

    // ================================================================
    // Step 1: cuBLAS handle 初始化
    // ================================================================
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create cuBLAS handle");

    // 启用 Tensor Core (TF32 for FP32 inputs — 自动利用 Tensor Core)
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // RAFT 资源句柄；每次调 select_k 前 set_cuda_stream 切到对应 slot stream
    raft::resources raft_res;

    // 查询可用 GPU 显存来确定处理策略
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t usable_bytes = static_cast<size_t>(free_bytes * 0.85);

    std::cout << "  GPU memory available: " << free_bytes / 1e9 << " GB\n";

    // ================================================================
    // Step 2: 计算所有 bucket 的尺寸上界，一次性预分配 GPU/CPU buffer
    // ================================================================
    // 关键优化:
    //   1) 取消每 bucket cudaMalloc/cudaFree (原本每 bucket 7 次同步分配)
    //   2) X_full 整体常驻 GPU，A/B 改为 GPU gather (省去 CPU memcpy + 大块 H2D)
    //   3) ‖x‖² 全局只算一次
    //   4) Double-buffer + 双 stream: bucket c 的 GEMM/topM/D2H 与 bucket c+1 的
    //      CPU prep + H2D + gather 并行；CPU 的 sort/unique 与 GPU 工作完全重叠。
    //
    // 关于 (2) 的一个已知权衡（暂不改，先记录）：
    // 之所以要求 X_full/d_X_full 整份常驻 CPU+GPU，是因为 gather A/B 这一步是
    // GPU kernel 直接按 id 去 d_X_full 里抠数据（gather_rows_int32/gather_rows_
    // raw），CUDA kernel 只能解引用显存指针，没法在 kernel 内部临时去读磁盘或
    // CPU 内存，而任意一个 bucket 的近邻桶都可能覆盖数据集里的任意点，没法只常
    // 驻一部分。理论上可以换成 bucket_build.cu 那种做法：需要哪个 bucket 就现读
    // 磁盘、CPU 端拼好向量再整块 H2D，完全不需要 d_X_full 常驻——但一个 bucket
    // 里的点在原文件里是随机散布的（分桶本身就打乱了顺序），这样读会退化成"每个
    // 点一次 seek"，而不是几次大块顺序读；把这种随机 I/O 插进现在这条为吞吐量
    // 设计的热循环（tensor core + 多 slot 流水线），可能会让 I/O 耗时超过 GPU
    // 计算耗时，反而拖慢整体。要让它划算，需要先把数据集按桶重排到磁盘上一份
    // （assign 阶段顺便生成，或者复用现成的 reorder 逻辑），这样每个 bucket 和它
    // 的近邻桶就变成几段连续区间，读起来才是大块顺序读而不是随机 seek。这块目前
    // 没有实现，先维持 d_X_full 整份常驻的现状，把这个方案记在这里，之后要做的
    // 话再单独展开。
    size_t max_bucket_size = 0;
    size_t max_pool_size_ub = 0;  // 上界(未去重)
    for (int64_t c = 0; c < n_centroids; ++c) {
        max_bucket_size = std::max(max_bucket_size, buckets[c].size());
        size_t ps = buckets[c].size();
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids)) {
                ps += buckets[nb_c].size();
            }
        }
        max_pool_size_ub = std::max(max_pool_size_ub, ps);
    }
    // bucket 平均大小由 centroid 率直接得出: assignment 把 N 点均分到 n_centroids 个 bucket
    size_t avg_bucket_size = (n_centroids > 0)
        ? static_cast<size_t>(N / n_centroids) : size_t{1};
    if (avg_bucket_size == 0) avg_bucket_size = 1;

    if (max_bucket_size == 0 || max_pool_size_ub == 0) {
        cublasDestroy(cublas_handle);
        std::cout << "[VectorKNN] All buckets empty, nothing to do.\n";
        return;
    }

    // ---- 显存预算分两块: 全局常驻 + per-slot ----
    // d_X_full 以 DataT 存（uint8/int8 时省 4×）；per-bucket A/B 用 GemmInT (int8 时也省 4×)
    size_t bytes_X            = static_cast<size_t>(N) * D * sizeof(DataT);
    size_t bytes_norms_full   = static_cast<size_t>(N) * sizeof(float);
    size_t bytes_A            = max_bucket_size * D * sizeof(GemmInT);
    size_t bytes_B            = max_pool_size_ub * D * sizeof(GemmInT);
    size_t bytes_dots         = max_bucket_size * max_pool_size_ub * sizeof(GemmOutT);
    size_t bytes_norms_pool   = max_pool_size_ub * sizeof(float);
    size_t bytes_ids_bucket   = max_bucket_size * sizeof(int32_t);
    size_t bytes_ids_pool     = max_pool_size_ub * sizeof(int32_t);
    size_t bytes_out          = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    // 距离输出 buffer 仅在 want_distances 时分配 (multi-iteration merge 需要)
    size_t bytes_out_dists    = want_distances ? max_bucket_size * static_cast<size_t>(M) * sizeof(float) : 0;

    size_t bytes_global   = bytes_X + bytes_norms_full;
    size_t bytes_per_slot = bytes_A + bytes_B + bytes_dots + bytes_norms_pool
                          + bytes_ids_bucket + bytes_ids_pool + bytes_out + bytes_out_dists;

    // ---- 动态 num_slots ----
    // 目标：让 num_slots × avg_blocks_per_kernel ≈ num_sm × OVERSUB，正好填满 SM 并留点
    // 富余 hide launch overhead。bucket 大单 kernel 已塞满 SM → num_slots 小；
    // bucket 小单 kernel 占 SM 少 → num_slots 大。
    int dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev_id);
    const int num_sm = prop.multiProcessorCount;

    // 用 avg bucket 估算 warps_per_block (跟 launch_extract_topM 里那段一致)
    int est_wpb;
    if      (avg_bucket_size <= 64)   est_wpb = 1;
    else if (avg_bucket_size <= 256)  est_wpb = 2;
    else if (avg_bucket_size <= 1024) est_wpb = 4;
    else                              est_wpb = 8;
    int est_blocks_per_kernel = std::max(1,
        static_cast<int>((avg_bucket_size + est_wpb - 1) / est_wpb));

    constexpr int MAX_SLOTS = 8;
    constexpr double OVERSUB = 1.5;  // 1.5× SM 利用率目标，hide launch overhead
    int target_streams = std::max(2,
        static_cast<int>(std::ceil(num_sm * OVERSUB / est_blocks_per_kernel)));
    target_streams = std::min(target_streams, MAX_SLOTS);

    // 内存上限：从 target_streams 往下试，挑能装下的最大值
    int num_slots = 1;
    for (int try_slots = target_streams; try_slots >= 1; --try_slots) {
        if (bytes_global + try_slots * bytes_per_slot <= usable_bytes) {
            num_slots = try_slots;
            break;
        }
    }
    if (bytes_global + bytes_per_slot > usable_bytes) {
        throw std::runtime_error(
            "[VectorKNN] Need at least " + std::to_string((bytes_global + bytes_per_slot) / 1e9)
            + " GB GPU mem but only " + std::to_string(usable_bytes / 1e9)
            + " GB usable. Increase n_centroids (smaller buckets) or lower K.");
    }

    std::cout << "  [VectorKNN] buffer plan: X_full=" << bytes_X/1e9
              << "GB, dots(max)=" << bytes_dots/1e9 << "GB, "
              << "per-slot=" << bytes_per_slot/1e9 << "GB × " << num_slots
              << " + global=" << bytes_global/1e9 << "GB / usable="
              << usable_bytes/1e9 << "GB"
              << " [streams=" << num_slots
              << ", num_sm=" << num_sm
              << ", avg_bucket=" << avg_bucket_size
              << ", est_blocks/kernel=" << est_blocks_per_kernel
              << ", target_streams(pre-mem)=" << target_streams << "]\n";

    // ---- 全局常驻 GPU buffer ----
    DataT* d_X_full     = nullptr;          // 数据集本体保留原始 element type
    float* d_norms_full = nullptr;          // 范数始终 fp32
    CUDA_CHECK(cudaMalloc(&d_X_full,     bytes_X));
    CUDA_CHECK(cudaMalloc(&d_norms_full, bytes_norms_full));

    // ---- per-slot 持久化 buffer (最多 8 slot 轮转, 实际用 num_slots 个) ----
    // d_A/d_B/d_dots 类型由 GemmInT/GemmOutT 决定（INT8 路径 / FP32 路径不同）
    // C++ partial init: {nullptr, nullptr} 后面的 slot 会被零初始化为 nullptr
    GemmInT*     d_A[MAX_SLOTS]             = {nullptr};
    GemmInT*     d_B[MAX_SLOTS]             = {nullptr};
    GemmOutT*    d_dots[MAX_SLOTS]          = {nullptr};
    float*       d_norms_pool[MAX_SLOTS]    = {nullptr};
    int32_t*     d_ids_bucket[MAX_SLOTS]    = {nullptr};
    int32_t*     d_ids_pool[MAX_SLOTS]      = {nullptr};
    int32_t*     d_out_neighbors[MAX_SLOTS] = {nullptr};
    float*       d_out_dists[MAX_SLOTS]     = {nullptr};  // 仅 want_distances 时使用
    int32_t*     d_select_idx[MAX_SLOTS]    = {nullptr};  // RAFT idx 输出
    float*       d_select_dist[MAX_SLOTS]   = {nullptr};  // RAFT dist 输出 scratch
    int32_t*     h_ids_bucket[MAX_SLOTS]    = {nullptr};
    int32_t*     h_ids_pool[MAX_SLOTS]      = {nullptr};
    int32_t*     h_out[MAX_SLOTS]           = {nullptr};
    float*       h_out_dists[MAX_SLOTS]     = {nullptr};  // 仅 want_distances 时使用
    cudaStream_t streams[MAX_SLOTS]         = {nullptr};

    // RAFT path 的额外 buffer 大小：(max_bucket_size × M) × 4 bytes，K=1024 时 ~4MB/slot，可忽略
    size_t bytes_select_idx  = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    size_t bytes_select_dist = max_bucket_size * static_cast<size_t>(M) * sizeof(float);

    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaMalloc(&d_A[s],             bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B[s],             bytes_B));
        CUDA_CHECK(cudaMalloc(&d_dots[s],          bytes_dots));
        CUDA_CHECK(cudaMalloc(&d_norms_pool[s],    bytes_norms_pool));
        CUDA_CHECK(cudaMalloc(&d_ids_bucket[s],    bytes_ids_bucket));
        CUDA_CHECK(cudaMalloc(&d_ids_pool[s],      bytes_ids_pool));
        CUDA_CHECK(cudaMalloc(&d_out_neighbors[s], bytes_out));
        CUDA_CHECK(cudaMalloc(&d_select_idx[s],    bytes_select_idx));
        CUDA_CHECK(cudaMalloc(&d_select_dist[s],   bytes_select_dist));
        CUDA_CHECK(cudaMallocHost(&h_ids_bucket[s], bytes_ids_bucket));
        CUDA_CHECK(cudaMallocHost(&h_ids_pool[s],   bytes_ids_pool));
        CUDA_CHECK(cudaMallocHost(&h_out[s],        bytes_out));
        if (want_distances) {
            CUDA_CHECK(cudaMalloc(&d_out_dists[s],  bytes_out_dists));
            CUDA_CHECK(cudaMallocHost(&h_out_dists[s], bytes_out_dists));
        }
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    // ---- 一次性上传 X_full + (uint8 时) shift → int8 + 计算所有点的 ‖x‖² (用 stream 0) ----
    // norms 必须从 GEMM 实际看到的数据视图算出来（uint8 路径下要在 shift 之后用 int8 视图算）。
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(d_X_full, X_full, bytes_X,
                                   cudaMemcpyHostToDevice, streams[0]));
        int threads = 256;

        // uint8 → int8 in-place 平移 (仅 uint8 路径)
        if constexpr (std::is_same<DataT, uint8_t>::value) {
            int64_t total = static_cast<int64_t>(N) * D;
            int64_t shift_blocks = (total + threads - 1) / threads;
            shift_uint8_to_int8_inplace<<<shift_blocks, threads, 0, streams[0]>>>(
                reinterpret_cast<uint8_t*>(d_X_full), total);
            CUDA_CHECK(cudaGetLastError());
        }

        int64_t blocks = (N + threads - 1) / threads;
        if constexpr (kIsInt8Path) {
            // 用 int8 视图算 norms（uint8 已 shift；int8 直接）
            compute_row_norms_kernel<int8_t><<<blocks, threads, 0, streams[0]>>>(
                reinterpret_cast<const int8_t*>(d_X_full), d_norms_full, N, D);
        } else {
            compute_row_norms_kernel<DataT><<<blocks, threads, 0, streams[0]>>>(
                d_X_full, d_norms_full, N, D);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        auto t1 = std::chrono::high_resolution_clock::now();
        double upload_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  [VectorKNN] X_full upload + norms: " << upload_ms << " ms\n";
    }

    // ================================================================
    // Step 3: 逐 bucket ping-pong 处理
    // ================================================================
    // 每个 slot 上有未消费 D2H 时记一笔 pending；下一次该 slot 被复用前必须:
    //   1) cudaStreamSynchronize (drain D2H)
    //   2) 把 h_out[slot] 中的结果逐点 merge 进磁盘上的 running 文件
    // 之后才能安全覆写 h_out[slot] / d_*[slot]。
    struct Pending {
        int64_t c           = -1;   // bucket index in this slot's last submitted iteration; -1 = empty
        int     bucket_size = 0;
        int     actual_M    = 0;
    };
    Pending pending[MAX_SLOTS]{};   // 全部默认初始化为 {-1, 0, 0}

    // ---- GEMM 纯计算时间统计 (GPU side, via cudaEvent on stream) ----
    cudaEvent_t ev_gemm_start[MAX_SLOTS] = {nullptr};
    cudaEvent_t ev_gemm_stop[MAX_SLOTS]  = {nullptr};
    bool gemm_evt_valid[MAX_SLOTS] = {false};
    double gemm_total_ms = 0.0;
    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaEventCreate(&ev_gemm_start[s]));
        CUDA_CHECK(cudaEventCreate(&ev_gemm_stop[s]));
    }
    auto consume_gemm_timing = [&](int slot) {
        if (!gemm_evt_valid[slot]) return;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev_gemm_start[slot], ev_gemm_stop[slot]);
        gemm_total_ms += ms;
        gemm_evt_valid[slot] = false;
    };

    // 每个点的新候选先铺成 M 宽（不足 aM 的部分补 -1/inf），再追加进对应 chunk
    // 的内存 write buffer —— 复用同一块 scratch buffer，避免每个点都分配一次；
    // acc.add() 只做内存 memcpy + 偶尔的整块顺序 flush，不读旧值、没有 seek。
    std::vector<int32_t> scatter_new_n(M);
    std::vector<float>   scatter_new_d(M);

    auto scatter_pending = [&](int slot) {
        if (pending[slot].c < 0) return;
        const auto& bk_prev = buckets[pending[slot].c];
        int bs = pending[slot].bucket_size;
        int aM = pending[slot].actual_M;
        const int32_t* hp = h_out[slot];
        const float*   hd = h_out_dists[slot];  // 始终已分配 (want_distances 内部恒为 true)
        for (int i = 0; i < bs; ++i) {
            int32_t gid = bk_prev[i];
            std::fill(scatter_new_n.begin(), scatter_new_n.end(), -1);
            std::fill(scatter_new_d.begin(), scatter_new_d.end(),
                     std::numeric_limits<float>::infinity());
            for (int m = 0; m < aM && m < M; ++m) {
                scatter_new_n[m] = hp[static_cast<size_t>(i) * aM + m];
                scatter_new_d[m] = hd[static_cast<size_t>(i) * aM + m];
            }
            acc.add(gid, scatter_new_n.data(), scatter_new_d.data());
        }
        pending[slot].c = -1;
    };

    int64_t processed_buckets = 0;
    auto loop_t0 = std::chrono::high_resolution_clock::now();

    for (int64_t c = 0; c < n_centroids; ++c) {
        if (buckets[c].empty()) continue;

        int slot = static_cast<int>(c % num_slots);

        // ---- 3a: 等本 slot 上一轮 D2H 落地，scatter 老结果，腾出 buffer ----
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
        consume_gemm_timing(slot);
        scatter_pending(slot);

        // ---- 3b: CPU 端拼 pool_ids ----
        // BatchAssign 里 line 791 强制 assignments[centroid_global_indices[c]] = c,
        // 加上每个非 centroid 点只在 assignments[] 里有一个值 → buckets 互不相交,
        // 拼出来的 pool 不会重复，无需 sort+unique。
        const auto& bk = buckets[c];
        size_t ofs = 0;
        std::memcpy(h_ids_pool[slot] + ofs, bk.data(), bk.size() * sizeof(int32_t));
        ofs += bk.size();
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids) && !buckets[nb_c].empty()) {
                std::memcpy(h_ids_pool[slot] + ofs, buckets[nb_c].data(),
                            buckets[nb_c].size() * sizeof(int32_t));
                ofs += buckets[nb_c].size();
            }
        }
        int pool_size   = static_cast<int>(ofs);
        int bucket_size = static_cast<int>(bk.size());
        int actual_M    = std::min(M, pool_size - 1);
        if (actual_M <= 0) continue;

        std::memcpy(h_ids_bucket[slot], bk.data(), bk.size() * sizeof(int32_t));

        // ---- 3c: 上传 id 列表 (async on slot stream) ----
        CUDA_CHECK(cudaMemcpyAsync(d_ids_bucket[slot], h_ids_bucket[slot],
                                   bucket_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(d_ids_pool[slot], h_ids_pool[slot],
                                   pool_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));

        // ---- 3d: GPU gather A / B / norms_pool ----
        // INT8 路径: gather_rows_raw 不转换 type；FP32 路径: gather_rows_int32 cast 到 fp32
        {
            int threads = 256;
            int64_t total_A  = static_cast<int64_t>(bucket_size) * D;
            int64_t blocks_A = (total_A + threads - 1) / threads;
            int64_t total_B  = static_cast<int64_t>(pool_size) * D;
            int64_t blocks_B = (total_B + threads - 1) / threads;

            if constexpr (kIsInt8Path) {
                // d_X_full 已含 int8 byte pattern (uint8 path 已在 shift 阶段就位)
                auto d_X_int8 = reinterpret_cast<const int8_t*>(d_X_full);
                gather_rows_raw<int8_t><<<blocks_A, threads, 0, streams[slot]>>>(
                    d_X_int8, d_ids_bucket[slot],
                    reinterpret_cast<int8_t*>(d_A[slot]), bucket_size, D);
                gather_rows_raw<int8_t><<<blocks_B, threads, 0, streams[slot]>>>(
                    d_X_int8, d_ids_pool[slot],
                    reinterpret_cast<int8_t*>(d_B[slot]), pool_size, D);
            } else {
                gather_rows_int32<DataT><<<blocks_A, threads, 0, streams[slot]>>>(
                    d_X_full, d_ids_bucket[slot],
                    reinterpret_cast<float*>(d_A[slot]), bucket_size, D);
                gather_rows_int32<DataT><<<blocks_B, threads, 0, streams[slot]>>>(
                    d_X_full, d_ids_pool[slot],
                    reinterpret_cast<float*>(d_B[slot]), pool_size, D);
            }

            int64_t blocks_n = (pool_size + threads - 1) / threads;
            gather_floats_int32<<<blocks_n, threads, 0, streams[slot]>>>(
                d_norms_full, d_ids_pool[slot], d_norms_pool[slot], pool_size);
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- 3e: cuBLAS GEMM 在 slot stream 上 (event 包夹做纯 GEMM 计时) ----
        cublasSetStream(cublas_handle, streams[slot]);
        CUDA_CHECK(cudaEventRecord(ev_gemm_start[slot], streams[slot]));
        {
            cublasStatus_t stat;
            if constexpr (kIsInt8Path) {
                // INT8 IMMA: int8 in × int8 in → int32 out, Tensor Core
                int alpha_i = 1, beta_i = 0;
                stat = cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, bucket_size, D,
                    &alpha_i,
                    d_B[slot], CUDA_R_8I, D,
                    d_A[slot], CUDA_R_8I, D,
                    &beta_i,
                    d_dots[slot], CUDA_R_32I, pool_size,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error(
                        "cublasGemmEx (INT8 IMMA) failed, status=" + std::to_string(stat));
            } else {
                float alpha = 1.0f, beta = 0.0f;
                stat = cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, bucket_size, D,
                    &alpha,
                    reinterpret_cast<const float*>(d_B[slot]), D,
                    reinterpret_cast<const float*>(d_A[slot]), D,
                    &beta,
                    reinterpret_cast<float*>(d_dots[slot]), pool_size);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("cublasSgemm failed, status=" + std::to_string(stat));
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_gemm_stop[slot], streams[slot]));
        gemm_evt_valid[slot] = true;

        // ---- 3f: top-M ----
        // 切换实现：把下面 USE_RAFT_SELECT_K 改 0 即用自写 warp-cooperative kernel，
        //          改 1 即用 RAFT select_k (radix / warp-sort auto)。
        // 两个 dispatcher 都是模板化 GemmOutT (int32 INT8 路径 / float TF32 路径)。
        #ifndef USE_RAFT_SELECT_K
        #define USE_RAFT_SELECT_K 1
        #endif
        {
        #if USE_RAFT_SELECT_K
            // d_dots 被 dots_to_dist_inplace_kernel 原地改写为 float dist,
            // RAFT 输出落 d_select_idx/dist scratch, 最后 gather 到 d_out_neighbors
            launch_extract_topM_raft<GemmOutT>(
                raft_res,
                streams[slot],
                d_dots[slot],
                d_norms_pool[slot],
                d_ids_bucket[slot], d_ids_pool[slot],
                d_select_idx[slot], d_select_dist[slot],
                d_out_neighbors[slot],
                want_distances ? d_out_dists[slot] : nullptr,
                bucket_size, pool_size, actual_M);
        #else
            // 自写 warp-cooperative kernel: 一 warp 处理一 query,
            // K_PER_LANE = K_OUT/32 个 slot 全在寄存器
            launch_extract_topM<GemmOutT>(
                streams[slot],
                d_dots[slot],
                d_norms_pool[slot],
                d_ids_bucket[slot], d_ids_pool[slot],
                d_out_neighbors[slot],
                want_distances ? d_out_dists[slot] : nullptr,
                bucket_size, pool_size, actual_M);
        #endif
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- 3g: 异步 D2H, 不 sync ----
        CUDA_CHECK(cudaMemcpyAsync(h_out[slot], d_out_neighbors[slot],
                                   static_cast<size_t>(bucket_size) * actual_M * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, streams[slot]));
        if (want_distances) {
            CUDA_CHECK(cudaMemcpyAsync(h_out_dists[slot], d_out_dists[slot],
                                       static_cast<size_t>(bucket_size) * actual_M * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[slot]));
        }

        pending[slot] = {c, bucket_size, actual_M};
        ++processed_buckets;
    }

    // ---- Tail flush: 处理两 slot 上的最后未消费结果 ----
    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        consume_gemm_timing(s);
        scatter_pending(s);
    }

    auto loop_t1 = std::chrono::high_resolution_clock::now();
    double loop_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
    std::cout << "  [VectorKNN] bucket loop: " << loop_ms << " ms over "
              << processed_buckets << " buckets ("
              << (loop_ms / std::max<int64_t>(1, processed_buckets))
              << " ms/bucket avg, slots=" << num_slots << ")\n";
    std::cout << "  [VectorKNN] pure GEMM: " << gemm_total_ms << " ms total, "
              << (gemm_total_ms / std::max<int64_t>(1, processed_buckets))
              << " ms/bucket avg, "
              << (gemm_total_ms / std::max(1e-6, loop_ms) * 100.0)
              << "% of loop wall-time\n";

    // ---- 释放 ----
    cudaFree(d_X_full);
    cudaFree(d_norms_full);
    for (int s = 0; s < num_slots; ++s) {
        cudaFree(d_A[s]);
        cudaFree(d_B[s]);
        cudaFree(d_dots[s]);
        cudaFree(d_norms_pool[s]);
        cudaFree(d_ids_bucket[s]);
        cudaFree(d_ids_pool[s]);
        cudaFree(d_out_neighbors[s]);
        cudaFree(d_select_idx[s]);
        cudaFree(d_select_dist[s]);
        if (d_out_dists[s])  cudaFree(d_out_dists[s]);
        cudaFreeHost(h_ids_bucket[s]);
        cudaFreeHost(h_ids_pool[s]);
        cudaFreeHost(h_out[s]);
        if (h_out_dists[s])  cudaFreeHost(h_out_dists[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(ev_gemm_start[s]);
        cudaEventDestroy(ev_gemm_stop[s]);
    }
    cublasDestroy(cublas_handle);

    std::cout << "[VectorKNN] Done. Output shape: (" << N << ", " << M << ")"
              << " merged into running KNN file on disk\n";
}

/**
 * 增量合并: 把一轮新的 per-vector KNN 结果 (new_n/new_d) 并入"运行中"的累积
 * 结果 (running_n/running_d, 原地更新)。
 *
 * 语义上等价于对 {running 之前已含的所有轮, new 这一轮} 调 dedupe+top-M，
 * 但每次只处理 2*M 个候选/点 (而不是 T*M)，所以可以在每轮 iteration 结束后
 * 立刻调用一次，逐轮把结果并进来 —— 不需要等全部 T 轮跑完再一次性合并。
 * 这让 CPU 侧的 merge 可以和下一轮 iteration 的 GPU 计算 (Step 2-6) 重叠:
 * 主线程 std::async 出去后立刻继续发起下一轮的 GPU 工作，merge 在后台线程
 * 用 CPU 核心跑，两者互不阻塞（只要不同时读写同一份 running_* 缓冲区）。
 *
 * 输入: running_n/d 和 new_n/d 都是 (N, M) row-major，且各自已经是"每行内
 *       按距离升序、已去重"的结果 (build_vector_knn_with_tensorcore /
 *       上一次 merge_two_per_vector_knn 的输出都满足这个不变式)。
 * 输出: running_n/d 原地更新为二者按距离升序 dedupe 后的 top-M。
 *
 * 注: 相同 (query, neighbor) pair 在不同轮的距离是确定性复现的 (公式相同 +
 *     浮点可复现)，dedupe 时按先遇到的为准即可，不需要比较取 min。
 */
inline void merge_two_per_vector_knn(
    std::vector<int32_t>& running_n, std::vector<float>& running_d,
    const std::vector<int32_t>& new_n, const std::vector<float>& new_d,
    int64_t N, int M)
{
    if (running_n.size() != static_cast<size_t>(N) * M ||
        running_d.size() != static_cast<size_t>(N) * M ||
        new_n.size()     != static_cast<size_t>(N) * M ||
        new_d.size()     != static_cast<size_t>(N) * M) {
        throw std::runtime_error("merge_two_per_vector_knn: shape mismatch");
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        // 候选/去重缓冲区按线程复用，避免每行都重新分配
        std::vector<std::pair<float, int32_t>> cand;
        cand.reserve(static_cast<size_t>(2) * M);
        std::unordered_set<int32_t> seen;
        seen.reserve(static_cast<size_t>(M) * 2);

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            cand.clear();
            seen.clear();

            int32_t* rn = running_n.data() + i * M;
            float*   rd = running_d.data() + i * M;
            const int32_t* nn = new_n.data() + i * M;
            const float*   nd = new_d.data() + i * M;

            for (int m = 0; m < M; ++m) if (rn[m] >= 0) cand.push_back({rd[m], rn[m]});
            for (int m = 0; m < M; ++m) if (nn[m] >= 0) cand.push_back({nd[m], nn[m]});
            if (cand.empty()) continue;

            // 只有 2*M 个候选 (M 一般 <= 1024)，直接排序足够快
            std::sort(cand.begin(), cand.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            int written = 0;
            for (const auto& [dist, id] : cand) {
                if (seen.insert(id).second) {
                    rn[written] = id;
                    rd[written] = dist;
                    if (++written >= M) break;
                }
            }
            for (; written < M; ++written) {
                rn[written] = -1;
                rd[written] = std::numeric_limits<float>::infinity();
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[MergeKNN] Incremental merge into (N=" << N << ", M=" << M
              << ") in " << ms << " ms\n";
}

/**
 * 将 per-vector KNN 结果写入磁盘。
 *
 * 格式: 二进制文件
 *   header:
 *     int64_t  N
 *     int32_t  M
 *   body:
 *     int32_t  neighbors[N * M]   // row-major, 每行 M 个邻居 global id
 */
void write_vector_knn_to_disk(
    const std::string& output_path,
    const std::vector<int32_t>& neighbors,
    int64_t N, int M)
{
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open: " + output_path);

    int32_t M32 = static_cast<int32_t>(M);
    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(neighbors.data()),
              static_cast<size_t>(N) * M * sizeof(int32_t));
    out.close();

    std::cout << "[WriteVectorKNN] Written to " << output_path
              << " (" << static_cast<size_t>(N) * M * sizeof(int32_t) / 1e6 << " MB)\n";
}

// ============== Phase 12: Optional bucket-aligned ID reorder ==============

/**
 * Compute the bucket-aligned permutation: new_id 区间 [offsets[pos], offsets[pos+1])
 * 占据 bucket_order[pos] 这个桶的所有点 —— 也就是说 offsets 是按 `bucket_order`
 * (DiskJoin 风格的 task ordering, 见 bucket_order.hpp) 的顺序排列的，不是按原始
 * 桶编号。optimize_chunked 的 method C/D 只按 offsets 数组顺序把连续几个桶划进
 * 一个 chunk，从不关心某个 chunk 对应哪个原始桶编号，所以这个改动对它完全透明；
 * 而这正是重排真正要解决的问题：只有当"新 ID 空间里相邻"的桶在特征空间里也相邻，
 * chunk 边界才会真的贴合数据分布，method C/D 文档里"相邻 chunk 在特征空间也相邻"
 * 的假设才成立。
 *
 * 输入 assignments (老 ID 空间)，输出:
 *   perm[old_id]         = new_id    (未分配的点为 0xFFFFFFFFu)
 *   inverse_perm[new_id] = old_id
 *   bucket_offsets       = bucket 边界（新 ID 空间，按 bucket_order 排列），长度 n_buckets+1
 */
struct ReorderInfo {
    std::vector<uint32_t> perm;
    std::vector<uint32_t> inverse_perm;
    std::vector<uint32_t> bucket_offsets;
    int64_t total_in_buckets;
};

static ReorderInfo compute_bucket_reorder(
    const std::vector<int64_t>& assignments, int64_t N, int64_t n_buckets,
    const std::vector<int32_t>& bucket_order)
{
    ReorderInfo r;
    r.bucket_offsets.assign(n_buckets + 1, 0);

    // Count per bucket (indexed by raw bucket id, same as `assignments`).
    std::vector<int64_t> counts(n_buckets, 0);
    for (int64_t i = 0; i < N; ++i) {
        int64_t b = assignments[i];
        if (b >= 0 && b < n_buckets) counts[b]++;
    }

    // Lay ranges out in bucket_order sequence (position pos -> raw bucket
    // bucket_order[pos]), not raw bucket id order.
    std::vector<uint32_t> new_id_range_start(n_buckets, 0);
    uint32_t running = 0;
    for (int64_t pos = 0; pos < n_buckets; ++pos) {
        int32_t b = bucket_order[pos];
        new_id_range_start[b] = running;
        running += static_cast<uint32_t>(counts[b]);
        r.bucket_offsets[pos + 1] = running;
    }

    r.total_in_buckets = static_cast<int64_t>(running);
    r.perm.assign(N, 0xFFFFFFFFu);
    r.inverse_perm.assign(r.total_in_buckets, 0);

    std::vector<uint32_t> cursor = new_id_range_start;
    for (int64_t old_id = 0; old_id < N; ++old_id) {
        int64_t b = assignments[old_id];
        if (b >= 0 && b < n_buckets) {
            uint32_t new_id = cursor[b]++;
            r.perm[old_id] = new_id;
            r.inverse_perm[new_id] = static_cast<uint32_t>(old_id);
        }
    }
    return r;
}

/**
 * 写出所有 reorder 产物（与 reorder.cpp 对齐，可被 optimize_chunked --method C/D 直接消费）：
 *   data_reordered.<ext>         重排后的 dataset，保留原始 DataT element type
 *   vector_knn_reordered.bin     重排 + 邻居 ID 重映射后的 KNN 图（仅当 neighbors_m > 0）
 *   bucket_offsets.bin           bucket 边界 (新 ID 空间)
 *   perm.bin / inverse_perm.bin  ID 翻译表 (uint32[N])
 *
 * X_full 现在是 DataT，原本格式直写一遍 row 重排即可（无类型转换）。
 */
template <typename DataT>
static void write_reordered_outputs(
    const std::string& output_dir,
    const std::string& input_ext,
    const std::vector<DataT>& X_full, int64_t /*N*/, int D,
    const std::vector<int32_t>& vector_knn, int M_neighbors,
    const ReorderInfo& r, int64_t n_buckets)
{
    const int64_t N = static_cast<int64_t>(X_full.size() / D);
    const int64_t total = r.total_in_buckets;

    // 1) data_reordered.<ext>：直接 row-level memcpy，element type = DataT
    {
        std::string path = output_dir + "/data_reordered" + input_ext;
        std::ofstream out(path, std::ios::binary);
        int32_t Nh = static_cast<int32_t>(total), Dh = static_cast<int32_t>(D);
        out.write(reinterpret_cast<const char*>(&Nh), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(&Dh), sizeof(int32_t));

        std::vector<DataT> reord(static_cast<size_t>(total) * D);
        const size_t row_bytes = static_cast<size_t>(D) * sizeof(DataT);
        #pragma omp parallel for schedule(static)
        for (int64_t new_id = 0; new_id < total; ++new_id) {
            uint32_t old_id = r.inverse_perm[new_id];
            std::memcpy(reord.data() + static_cast<size_t>(new_id) * D,
                        X_full.data() + static_cast<size_t>(old_id) * D,
                        row_bytes);
        }
        out.write(reinterpret_cast<const char*>(reord.data()),
                  static_cast<std::streamsize>(reord.size() * sizeof(DataT)));
        std::cout << "  Wrote " << path
                  << " (" << reord.size() * sizeof(DataT) / 1e9 << " GB)\n";
    }

    // 2) vector_knn_reordered.bin（如果有 vector_knn）
    if (M_neighbors > 0 && !vector_knn.empty()) {
        std::vector<int32_t> reord_knn(static_cast<size_t>(total) * M_neighbors);
        int64_t bad = 0;
        #pragma omp parallel for schedule(static) reduction(+:bad)
        for (int64_t new_id = 0; new_id < total; ++new_id) {
            uint32_t old_id = r.inverse_perm[new_id];
            const int32_t* src = vector_knn.data() + static_cast<size_t>(old_id) * M_neighbors;
            int32_t* dst = reord_knn.data() + static_cast<size_t>(new_id) * M_neighbors;
            for (int k = 0; k < M_neighbors; ++k) {
                int32_t old_nb = src[k];
                if (old_nb < 0) {
                    dst[k] = -1;
                } else if (static_cast<int64_t>(old_nb) >= N
                           || r.perm[old_nb] == 0xFFFFFFFFu) {
                    dst[k] = -1; bad++;
                } else {
                    dst[k] = static_cast<int32_t>(r.perm[old_nb]);
                }
            }
        }
        if (bad > 0)
            std::cout << "  [Warn] " << bad
                      << " neighbor entries → -1 (unassigned/out-of-range)\n";
        write_vector_knn_to_disk(
            output_dir + "/vector_knn_reordered.bin", reord_knn, total, M_neighbors);
    }

    // 3) bucket_offsets.bin / perm.bin / inverse_perm.bin
    {
        std::string p = output_dir + "/bucket_offsets.bin";
        std::ofstream out(p, std::ios::binary);
        int32_t nb32 = static_cast<int32_t>(n_buckets);
        out.write(reinterpret_cast<const char*>(&nb32), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(r.bucket_offsets.data()),
                  static_cast<std::streamsize>(r.bucket_offsets.size() * sizeof(uint32_t)));
        std::cout << "  Wrote " << p << "\n";
    }
    {
        std::ofstream out(output_dir + "/perm.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(r.perm.data()),
                  static_cast<std::streamsize>(r.perm.size() * sizeof(uint32_t)));
    }
    {
        std::ofstream out(output_dir + "/inverse_perm.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(r.inverse_perm.data()),
                  static_cast<std::streamsize>(r.inverse_perm.size() * sizeof(uint32_t)));
    }
    std::cout << "  Wrote perm.bin / inverse_perm.bin\n";
}

// ============== Main ==============

// run_pipeline_impl<DataT> 是真正的工作函数：根据输入文件 element type 实例化一份。
// main() 解析 CLI 后据 ext 分派到对应实例。
//
// iterations > 1: 重复跑 Step 2-6 (centroid 选取/分桶/邻居), 每次 seed 不同;
//                 把 T 次 per-vector KNN 结果按距离 dedupe-merge 输出。
//                 Bucket 文件 (Step 5) 仅保留最后一次 iteration 的结果。
template <typename DataT>
int run_pipeline_impl(
    const std::string& input_path,
    const std::string& output_root,
    int search_max_iters,
    int neighbors_m,
    uint32_t nprobe,
    bool do_reorder,
    int iterations,
    LoadConfig config,
    const std::string& ext,
    int32_t order_window_arg)
{
    try {
        init_gpu_limit_if_needed(config);

        // 在 output_root 下建子目录: k<knn_k>p<nprobe>m<neighbors_m>
        // knn_k = 图度数 K; nprobe = Step 6 邻居扩展数 (0 时回退到 knn_k)
        uint32_t output_nprobe = (nprobe > 0) ? nprobe : config.knn_k;
        std::string output_dir = output_root
                               + "/k" + std::to_string(config.knn_k)
                               + "p" + std::to_string(output_nprobe)
                               + "m" + std::to_string(neighbors_m);
        std::filesystem::create_directories(output_dir);
        std::cout << "Output subdir: " << output_dir << "\n";

        using Clock = std::chrono::high_resolution_clock;
        auto t_total_start = Clock::now();
        double elapsed_step1 = 0, elapsed_step2 = 0, elapsed_step3 = 0;
        double elapsed_step3p5 = 0, elapsed_step4 = 0, elapsed_step5 = 0;
        double elapsed_step6 = 0, elapsed_step7 = 0;
        double elapsed_write_knn = 0;

        // ================================================================
        // Step 1: Read header + prepare sampled data for centroid selection
        // ================================================================
        std::cout << "=== Step 1: Preparing data from " << input_path << " ===\n";
        auto t1 = Clock::now();
        int32_t N = 0, D = 0;

        // ext 由 caller 传入；此处仅校验
        if (ext != ".fbin" && ext != ".bin" && ext != ".u8bin" && ext != ".i8bin"
            && ext != ".ibin" && ext != ".ubin") {
            throw std::runtime_error("Unsupported file format: " + ext);
        }

        // 只读 header 获取 N, D
        auto [hdr_n, hdr_d] = load::read_fbin_header(input_path);
        N = hdr_n; D = hdr_d;
        std::cout << "  Header: N=" << N << ", D=" << D << "\n";

        validate_load_config(config, N);

        // Memory estimation
        static constexpr int64_t MAX_CENTROIDS = 1000000;
        int64_t n_centroids = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
        if (n_centroids > MAX_CENTROIDS) {
            std::cout << "  Capping n_centroids to " << MAX_CENTROIDS << "\n";
            n_centroids = MAX_CENTROIDS;
        }
        MemoryEstimate mem_est = estimate_memory_requirement(N, D, n_centroids, config);
        print_memory_estimate(mem_est);

        // 采样索引: sampled_indices[sample_idx] = raw_idx
        std::vector<int64_t> sampled_indices;
        std::vector<float> X_sampled;
        int64_t working_N;

        if (!mem_est.fits_in_gpu) {
            // 只从 disk 读取采样行，不加载完整数据集
            working_N = mem_est.sampled_data_rows;
            sampled_indices = sample_without_replacement(N, working_N, config.seed);
            // 排序以实现顺序磁盘读取 (SSD/NVMe 友好)
            auto sorted_order = sampled_indices;
            std::sort(sorted_order.begin(), sorted_order.end());

            if (ext == ".fbin" || ext == ".bin") {
                int32_t n_tmp, d_tmp;
                load::read_fbin_sampled(input_path, sorted_order, X_sampled, n_tmp, d_tmp);
            } else {
                // u8bin/ibin: fallback 到全量读取再采样
                // TODO: 为其他格式实现 sampled reader
                std::vector<float> X_full_tmp;
                if (ext == ".u8bin" || ext == ".i8bin") {
                    int32_t n2, d2;
                    load::read_u8bin_to_f32(input_path, X_full_tmp, n2, d2);
                } else {
                    std::vector<int32_t> itmp;
                    int32_t n2, d2;
                    load::read_ibin_i32(input_path, itmp, n2, d2);
                    X_full_tmp.resize(itmp.size());
                    for (size_t i = 0; i < itmp.size(); ++i) X_full_tmp[i] = static_cast<float>(itmp[i]);
                }
                X_sampled.resize(static_cast<size_t>(working_N) * D);
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < working_N; ++i) {
                    std::memcpy(X_sampled.data() + i * D,
                                X_full_tmp.data() + sorted_order[i] * D,
                                D * sizeof(float));
                }
            }

            // sampled_indices 需要和 X_sampled 行顺序一致 (sorted_order)
            sampled_indices = std::move(sorted_order);
            std::cout << "  Sampled " << working_N << " rows from disk ("
                      << X_sampled.size() * sizeof(float) / 1e9 << " GB)\n";
        } else {
            // 全量数据可放入 GPU — 此处仍需全量读取 (后续 step 也需要)
            working_N = N;
            sampled_indices.resize(N);
            std::iota(sampled_indices.begin(), sampled_indices.end(), 0LL);

            if (ext == ".fbin" || ext == ".bin") {
                load::read_fbin_f32(input_path, X_sampled, N, D);
            } else if (ext == ".u8bin" || ext == ".i8bin") {
                load::read_u8bin_to_f32(input_path, X_sampled, N, D);
            } else {
                std::vector<int32_t> tmp;
                load::read_ibin_i32(input_path, tmp, N, D);
                X_sampled.resize(tmp.size());
                for (size_t i = 0; i < tmp.size(); ++i) X_sampled[i] = static_cast<float>(tmp[i]);
            }
            std::cout << "  Loaded full dataset: " << X_sampled.size() * sizeof(float) / 1e9 << " GB\n";
        }

        elapsed_step1 = std::chrono::duration<double>(Clock::now() - t1).count();
        std::cout << "  Step 1 done [" << std::fixed << std::setprecision(3) << elapsed_step1 << "s]\n";

        // ================================================================
        // Per-iteration outer loop (Steps 2-6 may repeat with varying seed)
        // ================================================================
        // running per-vector KNN 现在活在磁盘上，按固定连续 id 区间分 chunk
        // (见文件顶部 ChunkedKnnAccumulator 的说明)。Step 6 内部
        // (scatter_pending -> acc.add) 只做按 chunk 路由的内存追加写；每轮
        // iteration 结束后调 acc.merge_iteration()，把这一轮的结果跟磁盘上的
        // running 状态就地合并。
        //
        // merge_iteration 是纯 CPU/磁盘操作，跟下一轮的 GPU 计算 (Step 2~6
        // 的 build 部分) 完全没有数据依赖 (下一轮的 acc.add() 写的是这一轮
        // 专属的 chunk_<c>_iter<t>.bin，跟 merge_iteration 读写的
        // running_chunk_<c>_*.bin 是两组不同的文件)——所以用 pending_knn_merge
        // 这个 future 把当前这一轮的 merge 扔到后台线程，主线程立刻去跑下一轮
        // 的 Step 2~6，把 merge 的耗时藏到下一轮 GPU 计算的背后。唯一需要同步
        // 的地方是:下一轮真正要发起自己的 merge 之前，必须先等上一轮的 merge
        // 跑完(两轮的 merge 都会读写同一份 running_chunk 文件，不能同时跑)。
        std::unique_ptr<ChunkedKnnAccumulator> knn_acc;
        std::future<void> pending_knn_merge;
        std::string knn_chunk_tmp_dir = output_dir + "/knn_chunks_tmp";
        if (neighbors_m > 0) {
            std::filesystem::create_directories(knn_chunk_tmp_dir);
            knn_acc = std::make_unique<ChunkedKnnAccumulator>(
                ChunkedKnnAccumulator::create(
                    knn_chunk_tmp_dir, N, static_cast<int>(neighbors_m),
                    config.cpu_limit_bytes / 4));
        }

        // 这些值由最后一次 iteration 决定 (用于 Step 5/7)
        std::vector<int64_t> assignments;
        std::vector<int64_t> centroid_global_indices;
        std::vector<uint32_t> centroid_knn_graph_host;  // (n_centroids, K), row-major
        uint32_t             K = config.knn_k;

        // X_full 在 Step 3.5 加载, 跨 iteration 复用 (大数据)
        // 在第一次 iteration 的 Step 4 之前加载
        std::vector<DataT> X_full;
        bool X_full_loaded = false;

        for (int iter = 0; iter < iterations; ++iter) {
            // 每次 iteration 用不同 seed (centroid 选取多样化)
            uint32_t iter_seed = config.seed + static_cast<uint32_t>(iter);
            LoadConfig iter_config = config;
            iter_config.seed = iter_seed;

            if (iterations > 1) {
                std::cout << "\n========== Iteration " << (iter + 1) << " / " << iterations
                          << " (seed=" << iter_seed << ") ==========\n";
            }

        // ================================================================
        // Step 2: Select centroids — results stay on GPU
        // ================================================================
        std::cout << "=== Step 2: Selecting centroids (GPU-resident) ===\n";
        auto t2 = Clock::now();
        float* d_centroids_f32 = nullptr;
        SimpleQuantizer quantizer{};

        // 返回 centroid 的 sample_idx
        std::vector<int64_t> centroid_sample_indices = select_centroids_on_gpu_kmeans_parallel(
            X_sampled.data(), sampled_indices,
            working_N, D, iter_config, mem_est, &quantizer, &d_centroids_f32);

        // sample_idx → raw_idx
        centroid_global_indices.assign(centroid_sample_indices.size(), 0);
        for (size_t k = 0; k < centroid_sample_indices.size(); ++k) {
            centroid_global_indices[k] = sampled_indices[centroid_sample_indices[k]];
        }

        n_centroids = static_cast<int64_t>(centroid_global_indices.size());
        cudaDeviceSynchronize();
        double iter_step2 = std::chrono::duration<double>(Clock::now() - t2).count();
        elapsed_step2 += iter_step2;
        std::cout << "  Step 2 done [" << std::fixed << std::setprecision(3) << iter_step2 << "s]\n";

        // ================================================================
        // Step 3: Build KNN graph on centroids — results stay on GPU
        // ================================================================
        std::cout << "=== Step 3: Building centroid KNN graph (GPU-resident) ===\n";
        auto t3 = Clock::now();
        uint32_t* d_graph = nullptr;
        K = config.knn_k;

        build_centroid_knn_on_gpu(
            d_centroids_f32, n_centroids, D, K, &d_graph);

        size_t centroid_gpu_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
        cudaDeviceSynchronize();
        double iter_step3 = std::chrono::duration<double>(Clock::now() - t3).count();
        elapsed_step3 += iter_step3;
        std::cout << "  Step 3 done [" << std::fixed << std::setprecision(3) << iter_step3 << "s]\n";

        // ================================================================
        // Step 3.5: Lazy-load full dataset if not already loaded
        //           (采样模式下 X_sampled 只有子集，assignment/KNN 需要全量数据)
        // ================================================================
        // X_full 以 DataT 存（保留原始 element type，省 host RAM/PCIe，u8 时 4×）
        // 多 iteration 时只在第一次 iteration 加载, 后续复用
        if (!X_full_loaded) {
            auto t_load = Clock::now();
            if (!mem_est.fits_in_gpu) {
                std::cout << "=== Step 3.5: Loading full dataset for assignment ===\n";
                // 不做类型转换，直接以 DataT 读 BIGANN payload
                int32_t fullN = 0, fullD = 0;
                load::read_bigann_raw<DataT>(input_path, X_full, fullN, fullD);
                if (fullN != N || fullD != D)
                    throw std::runtime_error("Full-load header mismatches sampled header");
                std::cout << "  Full data loaded: " << X_full.size() * sizeof(DataT) / 1e9 << " GB"
                          << " [" << std::chrono::duration<double>(Clock::now() - t_load).count() << "s]\n";
            } else {
                // fits_in_gpu: X_sampled (fp32) 是全量；转成 DataT 给 Step 4/6 用
                X_full.resize(X_sampled.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < X_sampled.size(); ++i)
                    X_full[i] = static_cast<DataT>(X_sampled[i]);
            }
            // 多 iteration: X_sampled 不再需要 (centroid 选取从 iter 1 起会重复用 X_sampled,
            // 但我们改为始终用 X_full → 跨 iteration 复用)。
            // 不过 select_centroids_on_gpu_kmeans_parallel 需要 fp32 X_sampled,
            // 多 iter 时需要保留 X_sampled. 单 iter 时可以 drop.
            if (iterations <= 1) {
                auto _drop = std::move(X_sampled);
            }
            X_full_loaded = true;
            elapsed_step3p5 = std::chrono::duration<double>(Clock::now() - t_load).count();
        }

        // ================================================================
        // Step 4: Batch assign — GPU centroids + graph reused, no re-upload
        // ================================================================
        std::cout << "=== Step 4: Batch assignment ===\n";
        auto t4 = Clock::now();

        // Step 6 需要 centroid 坐标做邻居扩展; 在 Step 4 释放 GPU centroids 前先存 CPU
        // (centroid KNN 图是 CAGRA-pruned navigable graph, 不是 sorted KNN, 即使
        //  effective_nprobe == K 也必须重新 search, 因此只要 Step 6 会跑就都需要)
        std::vector<float> centroids_host_for_step6;
        bool need_centroids_for_step6 = (neighbors_m > 0);
        if (need_centroids_for_step6) {
            centroids_host_for_step6.resize(static_cast<size_t>(n_centroids) * D);
            CUDA_CHECK(cudaMemcpy(centroids_host_for_step6.data(), d_centroids_f32,
                                  centroid_gpu_bytes, cudaMemcpyDeviceToHost));
        }

        // Compute available GPU memory for batch processing
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        size_t Mbatch_bytes = static_cast<size_t>(free_bytes * 0.9);

        assignments.clear();
        if (mem_est.need_pq) {
            // PQ mode: centroids need to be encoded to uint8 on GPU
            // Encode centroid f32 -> uint8 on CPU, then upload
            std::vector<float> centroids_host(n_centroids * D);
            CUDA_CHECK(cudaMemcpy(centroids_host.data(), d_centroids_f32,
                                  centroid_gpu_bytes, cudaMemcpyDeviceToHost));

            auto centroid_codes = simple_encode(centroids_host, n_centroids, D, quantizer);

            uint8_t* d_centroids_u8 = nullptr;
            size_t u8_bytes = static_cast<size_t>(n_centroids) * D * sizeof(uint8_t);
            CUDA_CHECK(cudaMalloc(&d_centroids_u8, u8_bytes));
            CUDA_CHECK(cudaMemcpy(d_centroids_u8, centroid_codes.data(),
                                  u8_bytes, cudaMemcpyHostToDevice));

            // Free f32 centroids from GPU — no longer needed in PQ mode
            CUDA_CHECK(cudaFree(d_centroids_f32));
            d_centroids_f32 = nullptr;

            assignments = batch_assign_with_cagra_anns<DataT, uint8_t>(
                X_full.data(), N, D, n_centroids,
                centroid_global_indices,
                d_centroids_u8, u8_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters);

            CUDA_CHECK(cudaFree(d_centroids_u8));
        } else {
            // Non-PQ mode: use float32 centroids directly
            assignments = batch_assign_with_cagra_anns<DataT, float>(
                X_full.data(), N, D, n_centroids,
                centroid_global_indices,
                d_centroids_f32, centroid_gpu_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters);

            CUDA_CHECK(cudaFree(d_centroids_f32));
        }

        // Download centroid KNN graph to CPU before freeing (needed for Step 6,
        // and for Step 7's bucket processing order when --reorder is set).
        if (neighbors_m > 0 || do_reorder) {
            centroid_knn_graph_host.resize(static_cast<size_t>(n_centroids) * K);
            CUDA_CHECK(cudaMemcpy(centroid_knn_graph_host.data(), d_graph,
                                  graph_bytes, cudaMemcpyDeviceToHost));
        }

        // Free KNN graph from GPU
        CUDA_CHECK(cudaFree(d_graph));
        cudaDeviceSynchronize();
        double iter_step4 = std::chrono::duration<double>(Clock::now() - t4).count();
        elapsed_step4 += iter_step4;
        std::cout << "  Step 4 done [" << std::fixed << std::setprecision(3) << iter_step4 << "s]\n";

        // ================================================================
        // Step 5: Write bucket assignments to disk (last iteration only,
        //         多 iter 时仅最后一次的 bucket 结构落盘)
        // ================================================================
        if (iter == iterations - 1) {
            std::cout << "=== Step 5: Writing buckets to disk ===\n";
            auto t5 = Clock::now();
            write_buckets_to_disk(output_dir, assignments, N, n_centroids);

            // Also save centroid global indices for later reference
            std::string centroid_idx_path = output_dir + "/centroid_global_indices.bin";
            {
                std::ofstream out(centroid_idx_path, std::ios::binary);
                int64_t nc = n_centroids;
                out.write(reinterpret_cast<const char*>(&nc), sizeof(int64_t));
                out.write(reinterpret_cast<const char*>(centroid_global_indices.data()),
                          n_centroids * sizeof(int64_t));
            }

            // Save the centroid KNN graph too, so reorder.cpp (the standalone,
            // post-hoc equivalent of Step 7) can recompute the same DiskJoin-
            // style bucket processing order without rebuilding it from scratch.
            if (!centroid_knn_graph_host.empty()) {
                std::string p = output_dir + "/centroid_knn.bin";
                std::ofstream out(p, std::ios::binary);
                int64_t nc = n_centroids;
                int32_t Kw = static_cast<int32_t>(K);
                out.write(reinterpret_cast<const char*>(&nc), sizeof(int64_t));
                out.write(reinterpret_cast<const char*>(&Kw), sizeof(int32_t));
                out.write(reinterpret_cast<const char*>(centroid_knn_graph_host.data()),
                          static_cast<std::streamsize>(centroid_knn_graph_host.size() * sizeof(uint32_t)));
                std::cout << "  Wrote " << p << " (K=" << K << ")\n";
            }

            elapsed_step5 = std::chrono::duration<double>(Clock::now() - t5).count();
            std::cout << "  Step 5 done [" << std::fixed << std::setprecision(3) << elapsed_step5 << "s]\n";
        }

        // ================================================================
        // Step 6: Per-vector KNN via Tensor Core bucket matmul (optional)
        // ================================================================
        if (neighbors_m > 0) {
            std::cout << "=== Step 6: Building per-vector KNN (M=" << neighbors_m << ") ===\n";
            auto t6 = Clock::now();

            // 解析 nprobe: 若用户没指定 (=0)，回退到图度数 K
            uint32_t effective_nprobe = (nprobe > 0) ? nprobe : K;

            // CAGRA-pruned graph 不是 sorted KNN, 必须用 greedy graph search
            // 在导航图上为每个 centroid 找真实的 top-nprobe 近邻 (含 nprobe == K)
            std::vector<uint32_t> centroid_topK = expand_centroid_neighbors_gpu(
                centroids_host_for_step6,
                centroid_knn_graph_host,
                n_centroids, D, K, effective_nprobe,
                search_max_iters);
            const uint32_t* graph_ptr = centroid_topK.data();
            uint32_t graph_K = effective_nprobe;

            // 这一轮的写入目标是专属的 chunk 文件 (chunk_<c>_iter<iter>.bin)，
            // 每轮开始前先开新文件。
            knn_acc->start_iteration(iter);

            build_vector_knn_with_tensorcore(
                X_full.data(), N, D,
                assignments,
                centroid_global_indices,
                graph_ptr,
                n_centroids, graph_K, neighbors_m,
                *knn_acc);

            // flush 掉这一轮剩余的 write buffer；merge_iteration 本身扔到
            // 后台线程做，让下一轮的 Step 2~6 立刻开始跑，不用等 merge 跑完
            // ——两轮的 merge 不能同时跑 (都要读写同一份 running_chunk 文件)，
            // 所以先等上一轮的 merge (如果还没完) 再发起这一轮的。不管
            // iterations 是 1 还是多轮，都是同一条代码路径——第 0 轮直接当
            // 全 sentinel 处理，不需要为它单独分支。
            knn_acc->finish_iteration_writes();
            if (pending_knn_merge.valid()) {
                auto t_wait = Clock::now();
                pending_knn_merge.get();
                double wait_s = std::chrono::duration<double>(Clock::now() - t_wait).count();
                std::cout << "  [ChunkedKNN] waited " << wait_s
                          << "s for previous iteration's background merge\n";
            }
            {
                ChunkedKnnAccumulator* acc_ptr = knn_acc.get();
                pending_knn_merge = std::async(std::launch::async,
                    [acc_ptr, iter]() { acc_ptr->merge_iteration(iter); });
            }

            cudaDeviceSynchronize();
            double iter_step6 = std::chrono::duration<double>(Clock::now() - t6).count();
            elapsed_step6 += iter_step6;
            std::cout << "  Step 6 done [" << std::fixed << std::setprecision(3) << iter_step6 << "s]\n";
        }

        }   // end of per-iteration loop

        // 最后一轮的 merge 还在后台跑，finalize 之前必须等它跑完，否则
        // running_chunk_*.bin 可能还没写完整就被读走。这段等待没有下一轮 GPU
        // 工作可以覆盖，是完成 Step 6 全部工作必须付出的代价，因此计入
        // elapsed_step6 (否则 Timing Summary 里各 Step 之和会小于 Total)。
        if (pending_knn_merge.valid()) {
            auto t_wait = Clock::now();
            pending_knn_merge.get();
            double wait_s = std::chrono::duration<double>(Clock::now() - t_wait).count();
            std::cout << "[ChunkedKNN] waited " << wait_s
                      << "s for the last iteration's background merge\n";
            elapsed_step6 += wait_s;
        }

        // 把按 chunk 分片的 running 状态 (跨全部 iteration 合并去重后的最终
        // 结果) 按 chunk 顺序拼接成 vector_knn.bin / vector_dists.bin。
        if (neighbors_m > 0) {
            auto t_write = Clock::now();
            knn_acc->finalize_to_file(
                output_dir + "/vector_knn.bin", output_dir + "/vector_dists.bin");
            std::filesystem::remove(knn_chunk_tmp_dir);  // 此时应已清空

            // 分块把 vector_knn.bin 转成 neighbors.npy，给 Python 端评测用
            // (纯顺序拷贝+类型转换，不需要整份常驻内存)
            convert_vector_knn_to_npy(
                output_dir + "/vector_knn.bin", output_dir + "/neighbors.npy",
                N, neighbors_m, config.cpu_limit_bytes / 4);
            std::cout << "[WriteVectorKNN] Saved neighbors.npy\n";
            elapsed_write_knn = std::chrono::duration<double>(Clock::now() - t_write).count();
        }

        // ================================================================
        // Step 7: Bucket-aligned ID reorder (optional, --reorder)
        //
        // 已知限制: assignments/centroid_global_indices (以及 bucket_index.bin/
        // bucket_data.bin) 都只保留"最后一轮" iteration 的分桶结果 (见上面
        // "这些值由最后一次 iteration 决定" 的注释)。当 iterations > 1 时，
        // 最终合并出的 vector_knn 里的边可能来自任意一轮的分桶，并不都落在
        // "最后一轮"划出的桶边界内——所以这里按最后一轮的桶结构重排，只是让
        // 那一轮产生的边对齐，其余轮的边不保证对齐，reorder 的"bucket-aligned"
        // 效果在多轮场景下是打了折扣的近似，不是严格保证。暂时按这个近似做，
        // 没有为多轮场景单独设计一套对齐方案。
        // ================================================================
        if (do_reorder) {
            std::cout << "=== Step 7: Bucket-aligned reorder ===\n";
            auto t7 = Clock::now();

            // Order buckets so ranges that end up adjacent in the new ID space
            // are also adjacent in feature space (DiskJoin's task ordering,
            // Algorithm 2 - see bucket_order.hpp), using the centroid KNN
            // graph already built in Step 3 as the bucket dependency graph.
            int32_t order_window = (order_window_arg > 0)
                ? order_window_arg
                : std::max<int32_t>(4 * static_cast<int32_t>(K), 16);
            auto bucket_process_order = bucket_order::compute_bucket_processing_order(
                bucket_order::adjacency_from_flat_graph(
                    centroid_knn_graph_host.data(), n_centroids, static_cast<int32_t>(K)),
                order_window);
            std::cout << "  order_window=" << order_window << "\n";

            auto reorder_info = compute_bucket_reorder(assignments, N, n_centroids,
                                                        bucket_process_order);
            std::cout << "  N=" << N
                      << " total_in_buckets=" << reorder_info.total_in_buckets
                      << " (unassigned=" << (N - reorder_info.total_in_buckets) << ")\n";

            // write_reordered_outputs 需要整份 vector_knn 在内存里做按 id 重排；
            // 现在它活在磁盘上，这里读回来（顺序读，一次性，只有 --reorder 时
            // 才会触发）。
            std::vector<int32_t> vector_knn;
            if (neighbors_m > 0) {
                std::ifstream in(output_dir + "/vector_knn.bin", std::ios::binary);
                if (!in.is_open())
                    throw std::runtime_error("Cannot open vector_knn.bin for reorder");
                in.seekg(static_cast<std::streamoff>(knn_file_header_bytes()));
                vector_knn.resize(static_cast<size_t>(N) * neighbors_m);
                in.read(reinterpret_cast<char*>(vector_knn.data()),
                       static_cast<std::streamsize>(vector_knn.size() * sizeof(int32_t)));
                if (!in.good())
                    throw std::runtime_error("Failed reading vector_knn.bin for reorder");
            }

            write_reordered_outputs(output_dir, ext, X_full, N, D,
                                    vector_knn, neighbors_m,
                                    reorder_info, n_centroids);

            elapsed_step7 = std::chrono::duration<double>(Clock::now() - t7).count();
            std::cout << "  Step 7 done [" << std::fixed << std::setprecision(3)
                      << elapsed_step7 << "s]\n";
        }

        // X_full no longer needed
        { auto _drop = std::move(X_full); }

        double elapsed_total = std::chrono::duration<double>(Clock::now() - t_total_start).count();

        std::cout << "\n=== All done! ===\n";
        std::cout << "\n=== Timing Summary"
                  << (iterations > 1 ? (" (Step 2-6 sums over T=" + std::to_string(iterations) + " iterations)") : "")
                  << " ===\n";
        std::cout << "  Step 1   (Load sampled data):  " << std::fixed << std::setprecision(3) << elapsed_step1 << "s\n";
        std::cout << "  Step 2   (Select centroids):    " << elapsed_step2 << "s\n";
        std::cout << "  Step 3   (Centroid KNN graph):  " << elapsed_step3 << "s\n";
        std::cout << "  Step 3.5 (Load full dataset):   " << elapsed_step3p5 << "s\n";
        std::cout << "  Step 4   (Batch assignment):    " << elapsed_step4 << "s\n";
        std::cout << "  Step 5   (Write buckets):       " << elapsed_step5 << "s\n";
        if (neighbors_m > 0) {
            std::cout << "  Step 6   (Per-vector KNN, merge included):      " << elapsed_step6 << "s\n";
            std::cout << "  Write    (close + .npy convert):    " << elapsed_write_knn << "s\n";
        }
        if (do_reorder)
            std::cout << "  Step 7   (Bucket reorder):      " << elapsed_step7 << "s\n";
        std::cout << "  --------------------------------\n";
        std::cout << "  Total:                        " << elapsed_total << "s\n";
        std::cout << "  Output: " << output_dir << "/\n";
        std::cout << "    bucket_index.bin  — offset/count table for each centroid\n";
        std::cout << "    bucket_data.bin   — packed int32 point IDs per bucket\n";
        std::cout << "    centroid_global_indices.bin — centroid-to-original-data mapping\n";
        if (!centroid_knn_graph_host.empty()) {
            std::cout << "    centroid_knn.bin  — centroid KNN graph (K=" << K
                      << "), lets reorder.cpp reproduce Step 7's bucket order\n";
        }
        if (neighbors_m > 0) {
            std::cout << "    vector_knn.bin    — per-vector " << neighbors_m << "-NN index\n";
            std::cout << "    neighbors.npy     — same as above in NumPy format (for search.py)\n";
        }
        if (do_reorder) {
            std::cout << "    data_reordered" << ext
                      << "       — bucket-aligned dataset (preserves input format)\n";
            if (neighbors_m > 0)
                std::cout << "    vector_knn_reordered.bin  — KNN graph in new ID space\n";
            std::cout << "    bucket_offsets.bin        — bucket boundaries (new ID space)\n";
            std::cout << "    perm.bin / inverse_perm.bin — ID translation tables\n";
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

// ============== Real main: CLI parse + DataT dispatch ==============
//
// 根据输入文件扩展名选择合适的 DataT，调用对应的 run_pipeline_impl<DataT> 实例。
// 支持的格式 → DataT:
//   .fbin/.bin → float
//   .u8bin     → uint8_t
//   .i8bin     → int8_t
//   .ibin      → int32_t
//   .ubin      → uint32_t
int main(int argc, char** argv) {
    try {
        // ---- CLI parsing ----
        po::options_description desc("Bucket Builder Options");
        desc.add_options()
            ("help,h",        "Show help")
            ("input,i",       po::value<std::string>()->required(),
                "Input data file (.fbin/.bin/.u8bin/.i8bin/.ibin/.ubin)")
            ("output,o",      po::value<std::string>()->required(),
                "Output directory for bucket files")
            ("cpu-limit",     po::value<size_t>(),     "CPU memory limit (bytes)")
            ("gpu-limit",     po::value<size_t>(),     "GPU memory limit (bytes, 0=auto)")
            ("sample-rate",   po::value<float>(),      "Sampling ratio (default 0.1)")
            ("centroid-ratio", po::value<float>(),     "Centroid ratio (default 0.01)")
            ("use-pq",        po::value<bool>(),       "Force PQ quantization")
            ("pq-bits-start", po::value<uint32_t>(),   "PQ starting bits (default 8)")
            ("pq-bits-min",   po::value<uint32_t>(),   "PQ minimum bits (default 4)")
            ("seed",          po::value<uint32_t>(),   "Random seed (default 42)")
            ("knn-k",         po::value<uint32_t>(),   "KNN graph degree (default 32)")
            ("nprobe",        po::value<uint32_t>()->default_value(0),
                "Per-bucket neighbor count for Step 6 (FAISS-style nprobe; 0=use knn-k; max 256)")
            ("search-iters",  po::value<int>()->default_value(64),
                "Graph search max iterations")
            ("neighbors-m",   po::value<int>()->default_value(0),
                "Per-vector KNN neighbor count M (0=skip)")
            ("iterations,t",  po::value<int>()->default_value(1),
                "Run Step 2-6 multiple times with seed+iter, dedupe-merge per-vector KNN. "
                "Bucket files (Step 5) reflect the last iteration only. Default 1.")
            ("reorder",       po::bool_switch()->default_value(false),
                "Also output bucket-aligned reordered files (data_reordered.<ext>, "
                "vector_knn_reordered.bin, bucket_offsets.bin, perm.bin, inverse_perm.bin) "
                "for optimize_chunked --method C/D")
            ("order-window",  po::value<int32_t>()->default_value(0),
                "Sliding-window size for the bucket processing order used by --reorder "
                "(DiskJoin-style task ordering over the centroid KNN graph; 0 = auto: 4*knn-k)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);

        std::string input_path  = vm["input"].as<std::string>();
        std::string output_root = vm["output"].as<std::string>();
        int search_max_iters    = vm["search-iters"].as<int>();
        int neighbors_m         = vm["neighbors-m"].as<int>();
        uint32_t nprobe         = vm["nprobe"].as<uint32_t>();
        bool do_reorder         = vm["reorder"].as<bool>();
        int iterations          = vm["iterations"].as<int>();
        int32_t order_window_arg = vm["order-window"].as<int32_t>();

        constexpr uint32_t MAX_NPROBE = 256;
        if (nprobe > MAX_NPROBE) {
            throw std::runtime_error(
                "nprobe (" + std::to_string(nprobe) + ") exceeds MAX_NPROBE=" +
                std::to_string(MAX_NPROBE) +
                ". Use --nprobe <= " + std::to_string(MAX_NPROBE) + ".");
        }

        if (iterations < 1) {
            throw std::runtime_error(
                "iterations (" + std::to_string(iterations) + ") must be >= 1");
        }
        if (iterations > 1 && neighbors_m <= 0) {
            std::cerr << "[Warn] iterations=" << iterations
                      << " but neighbors-m=0 (Step 6 skipped) → no merge will happen, "
                         "reverting to iterations=1\n";
            iterations = 1;
        }

        LoadConfig config = parse_load_config(vm);

        // ---- 据 ext 分派到对应 DataT 的 run_pipeline_impl ----
        std::string ext = std::filesystem::path(input_path).extension().string();
        std::cout << "[main] Dispatching DataT for ext=" << ext << "\n";

        if (ext == ".fbin" || ext == ".bin") {
            return run_pipeline_impl<float>(input_path, output_root, search_max_iters,
                                            neighbors_m, nprobe, do_reorder, iterations,
                                            config, ext, order_window_arg);
        } else if (ext == ".u8bin") {
            return run_pipeline_impl<uint8_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, do_reorder, iterations,
                                              config, ext, order_window_arg);
        } else if (ext == ".i8bin") {
            return run_pipeline_impl<int8_t>(input_path, output_root, search_max_iters,
                                             neighbors_m, nprobe, do_reorder, iterations,
                                             config, ext, order_window_arg);
        } else if (ext == ".ibin") {
            return run_pipeline_impl<int32_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, do_reorder, iterations,
                                              config, ext, order_window_arg);
        } else if (ext == ".ubin") {
            return run_pipeline_impl<uint32_t>(input_path, output_root, search_max_iters,
                                               neighbors_m, nprobe, do_reorder, iterations,
                                               config, ext, order_window_arg);
        } else {
            throw std::runtime_error("Unsupported file extension: " + ext +
                ". Supported: .fbin/.bin (float), .u8bin (uint8), .i8bin (int8), "
                ".ibin (int32), .ubin (uint32).");
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
