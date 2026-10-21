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
#include <unordered_map>
#include <cstring>
#include <type_traits>
#include <omp.h>
#include <optional>
#include <fcntl.h>
#include <unistd.h>
#include <exception>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <future>
#include <functional>
#include <memory>
#include <sys/resource.h>

// Boost
#include <boost/program_options.hpp>

// CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// RAFT / cuVS 26.10 (public API only)
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/random/rng.cuh>
#include <raft/matrix/select_k.cuh>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
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
#include "graph_io.hpp"
#include "reordered_dataset_io.hpp"

namespace po = boost::program_options;
using namespace bucket;

/**
 * cuVS 26.10 CAGRA wants a padded (16B-aligned rows) dataset: zero-copy view when the
 * rows already satisfy it (e.g. D=128), otherwise an owning padded copy kept in `owner`.
 */
template <typename T>
cuvs::neighbors::cagra::device_padded_index<T, uint32_t> make_padded_cagra_index(
    raft::resources const& res,
    raft::device_matrix_view<const T, int64_t> dataset,
    raft::device_matrix_view<const uint32_t, int64_t> graph,
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, int64_t>>& owner)
{
    constexpr auto L2 = cuvs::distance::DistanceType::L2Expanded;
    if ((dataset.extent(1) * sizeof(T)) % 16 == 0)
        return {res, L2, cuvs::neighbors::make_device_padded_dataset_view(res, dataset), graph};
    owner = cuvs::neighbors::make_device_padded_dataset(res, dataset);
    return {res, L2, owner->as_dataset_view(), graph};
}


// ============== Phase 1: LoadConfig & Memory Management ==============

/**
 * Configuration for flexible data loading and GPU KMeans++ centroid generation
 * Based on user requirements and DiskANN design patterns
 */
struct LoadConfig {
    // Memory limits
    size_t cpu_limit_bytes;        // CPU memory limit, default 16GB
    size_t gpu_limit_bytes;        // GPU memory limit, default 0 (auto-detect via cudaMemGetInfo)
    size_t bucket_vec_buffer_bytes; // Step 4 BucketVectorAccumulator total write-buffer budget,
                                    // default 1GB. Not per-bucket -- split across all n_centroids
                                    // buckets, so it stays flat regardless of dataset/bucket count.

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

    // Output
    bool save_npy;                 // 额外写 neighbors.npy (int64, 给 Python 端评测用), default true;
                                   // 构建流水线本身不读它

    static constexpr uint32_t MAX_KNN_K = 1000;

    // Constructor with safe defaults
    LoadConfig() :
        cpu_limit_bytes(16UL << 30),    // 16GB
        gpu_limit_bytes(0),             // Auto-detect
        bucket_vec_buffer_bytes(1UL << 30),  // 1GB
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
        knn_k(32),
        save_npy(true)
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
    if (vm.count("bucket-vec-buffer")) {
        config.bucket_vec_buffer_bytes = vm["bucket-vec-buffer"].as<size_t>();
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
    if (vm.count("skip-npy")) {
        config.save_npy = !vm["skip-npy"].as<bool>();
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

// ============== Per-bucket disk-backed original-vector cache ==============
//
// 目的: Step 4 算完每个点的 bucket 分配后，把它的原始向量 (D 个 DataT 分量)
// 连同 gid 一起写进"这个 bucket 专属"的字节区间里，供 Step 6/7 顺序读取。
//
// 每条记录格式: int64_t gid + D 个 DataT 分量。记录自带 gid，Step 6/7 读回
// 来的时候不需要依赖"文件第 i 行对应某个外部列表第 i 个 id"这种隐式顺序假设。
//
// 写缓冲区大小: 不能像 ChunkedKnnAccumulator 那样每个 key 固定分配一块 (那是
// 因为 num_chunks 量级是几十；这里 n_centroids 可能是几千上万、未来测大数据集
// 时还会再扩大百倍千倍)，而是先定一个总缓冲区内存预算，再除以 n_centroids
// 分给每个 bucket，这样总内存不会随桶数线性暴涨。
//
// 整个 iteration 全程只用一个磁盘文件 (merged_path)，不会有"每个 bucket 一个
// 小文件、最后再合并"这一步——这不只是 Step 6 的读侧需要 (n_centroids*(1+K)
// 量级的 open()/close() 本身就是瓶颈，远超实际读盘字节数对应的带宽耗时，这件
// 事在读侧已经通过单一持久句柄解决了)，写侧如果也按"每桶一个小文件、结束时
// 读回来拷进合并文件再删掉"来做，一样是 n_centroids 量级的小文件创建/打开/
// 关闭/删除，而且这个量级直接随 bucket 数线性增长、跟数据集本身大小无关——
// bucket 数一旦扩大百倍千倍 (未来大数据集测试的场景)，这部分系统调用开销会
// 比实际搬动的数据量大得多。
//
// 做法: start_iteration() 要求调用方先给出这一轮每个 bucket 最终会收到多少
// 条记录 (bucket_counts，来自对完整 assignments[] 做一次 histogram——分配
// 结果本来就要在任何向量真正落盘之前全部算完，这个 histogram 只是多扫一遍
// 一个 int64 数组，代价和数据集大小、bucket 数都无关，可以忽略)。有了每个
// bucket 的最终大小，就能提前算出它在合并文件里的字节区间 (前缀和)，直接把
// 合并文件撑到最终总大小；每个 bucket 自己的写缓冲区攒满或者 finish_iteration
// 收尾时，用 seekp 定位到这个 bucket 自己算好的游标位置写进去——数据从第一个
//字节开始就已经躺在最终合并文件的正确位置上，不会有任何"先写小文件、再读
// 回来、再拷贝、再删除"的回头路。add() 的调用方只保证"写之前 assignments[]
// 已经全部算完"，不要求写入顺序跟 bucket id 顺序一致。
template <typename DataT>
struct BucketVectorAccumulator {
    std::string dir;
    int64_t n_centroids = 0;
    int64_t D = 0;

    struct BucketBuf {
        std::vector<char> data;
        size_t capacity_records = 0;
        size_t count = 0;           // 当前缓冲区里还没 flush 的记录数
        int64_t total_count = 0;    // histogram 给出的、这个 bucket 本轮最终的记录数
        int64_t added_so_far = 0;   // 实际 add() 调用次数，用来校验跟 histogram 是否一致
        int64_t write_cursor = 0;   // 下次 flush 要 seekp 到的合并文件绝对字节偏移
    };
    std::vector<BucketBuf> bufs;

    std::string merged_path;
    std::vector<int64_t> merged_offset;  // 每个 bucket 在合并文件里的起始字节偏移
    std::fstream merged_out;             // 本轮写句柄 (随机位置 seekp+write)
    std::ifstream merged_in;             // Step 6/7 共用的持久读句柄

    static constexpr size_t record_bytes(int64_t D) {
        return sizeof(int64_t) + static_cast<size_t>(D) * sizeof(DataT);
    }

    // total_buffer_budget_bytes: 所有 bucket 的写缓冲区加起来的总内存预算
    // (不是每个 bucket 各自这么多)。
    static BucketVectorAccumulator create(const std::string& dir, int64_t n_centroids,
                                           int64_t D, size_t total_buffer_budget_bytes) {
        BucketVectorAccumulator acc;
        acc.dir = dir;
        acc.n_centroids = n_centroids;
        acc.D = D;
        acc.merged_path = dir + "/bucket_vecs_merged.bin";

        constexpr size_t kMinBufferBytesPerBucket = 4096;  // ~1 个文件系统块，下限
        size_t rec_bytes = record_bytes(D);
        size_t nb = std::max<int64_t>(1, n_centroids);
        size_t buf_bytes_per_bucket = std::max<size_t>(
            kMinBufferBytesPerBucket, total_buffer_budget_bytes / nb);
        size_t cap_records = std::max<size_t>(1, buf_bytes_per_bucket / rec_bytes);

        acc.bufs.resize(static_cast<size_t>(n_centroids));
        for (auto& b : acc.bufs) {
            b.capacity_records = cap_records;
            b.data.resize(cap_records * rec_bytes);
        }

        std::cout << "[BucketVecCache] n_centroids=" << n_centroids << " D=" << D
                  << " write_buffer=" << (cap_records * rec_bytes) / 1024
                  << "KB/bucket (" << (cap_records * rec_bytes * nb) / 1e6 << "MB total)\n";
        return acc;
    }

    // bucket_counts[c] = 这一轮 bucket c 最终会被 add() 多少次 (来自对完整
    // assignments[] 做 histogram，必须在任何 add() 调用之前就已知)。
    void start_iteration(const std::vector<int64_t>& bucket_counts) {
        size_t rec_bytes = record_bytes(D);
        merged_offset.assign(static_cast<size_t>(n_centroids), 0);
        int64_t pos = 0;
        for (int64_t c = 0; c < n_centroids; ++c) {
            auto& b = bufs[static_cast<size_t>(c)];
            b.count = 0;
            b.total_count = bucket_counts[static_cast<size_t>(c)];
            b.added_so_far = 0;
            merged_offset[static_cast<size_t>(c)] = pos;
            b.write_cursor = pos;
            pos += b.total_count * static_cast<int64_t>(rec_bytes);
        }

        if (merged_out.is_open()) merged_out.close();
        merged_out.clear();
        merged_out.open(merged_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!merged_out.is_open())
            throw std::runtime_error("BucketVectorAccumulator: cannot open " + merged_path + " for write");
        if (pos > 0) {
            // 提前把文件撑到这一轮的最终总大小，这样任何 bucket (包括最后一个,
            // 即使它恰好是 0 条) 的 seekp 都落在一个早已存在、大小正确的文件里,
            // 不依赖"写到哪文件才自动涨到哪"这种隐式行为。
            merged_out.seekp(static_cast<std::streamoff>(pos - 1));
            merged_out.put('\0');
        }
    }

    void flush_bucket(int64_t c) {
        auto& b = bufs[static_cast<size_t>(c)];
        if (b.count == 0) return;
        size_t rec_bytes = record_bytes(D);
        size_t nbytes = b.count * rec_bytes;
        merged_out.seekp(static_cast<std::streamoff>(b.write_cursor));
        merged_out.write(b.data.data(), static_cast<std::streamsize>(nbytes));
        if (!merged_out.good())
            throw std::runtime_error("BucketVectorAccumulator: flush failed for bucket " + std::to_string(c));
        b.write_cursor += static_cast<int64_t>(nbytes);
        b.count = 0;
    }

    // vec 必须指向恰好 D 个 DataT 分量。调用前 bucket_id 的 total_count (这一轮
    // 最终会收到多少条) 必须已经通过 start_iteration 的 histogram 定好——add()
    // 调用次数超过 histogram 给的数会直接越界写进下一个 bucket 的区间，所以这里
    // 主动校验、宁可报错也不要静默写坏数据。
    void add(int64_t bucket_id, int64_t gid, const DataT* vec) {
        auto& b = bufs[static_cast<size_t>(bucket_id)];
        if (b.added_so_far >= b.total_count)
            throw std::runtime_error(
                "BucketVectorAccumulator: add() exceeded histogram count for bucket "
                + std::to_string(bucket_id) + " (histogram said "
                + std::to_string(b.total_count) + ")");
        ++b.added_so_far;

        size_t rec_bytes = record_bytes(D);
        size_t off = b.count * rec_bytes;
        std::memcpy(b.data.data() + off, &gid, sizeof(int64_t));
        std::memcpy(b.data.data() + off + sizeof(int64_t), vec, static_cast<size_t>(D) * sizeof(DataT));
        ++b.count;
        if (b.count == b.capacity_records) flush_bucket(bucket_id);
    }

    // flush 每个 bucket 剩余的缓冲区尾部 (已经写在它自己算好的最终位置上，不需要
    // 再合并/拷贝/删除任何东西)，校验每个 bucket 实际收到的记录数跟 histogram
    // 一致 (否则说明调用方的 assignments/histogram 跟实际 add() 不匹配，数据
    // 已经错位，必须立刻报错而不是让 Step 6 读到看似正常但错位的内容)，然后
    // 打开一个持久的读句柄供 Step 6/7 使用。
    void finish_iteration() {
        for (int64_t c = 0; c < n_centroids; ++c) {
            auto& b = bufs[static_cast<size_t>(c)];
            flush_bucket(c);
            if (b.added_so_far != b.total_count)
                throw std::runtime_error(
                    "BucketVectorAccumulator: bucket " + std::to_string(c) + " got "
                    + std::to_string(b.added_so_far) + " add() calls, expected "
                    + std::to_string(b.total_count) + " from histogram");
        }
        merged_out.close();

        // merged_in may already be open from a previous iteration when this
        // object is being reused across rounds (not freshly make_unique'd) -
        // open() on an already-open stream is a no-op (fails silently,
        // leaves the stale fd/eofbit in place) per the standard, so it must
        // be closed and have its error flags cleared first.
        if (merged_in.is_open()) merged_in.close();
        merged_in.clear();
        merged_in.open(merged_path, std::ios::binary);
        if (!merged_in.is_open())
            throw std::runtime_error("BucketVectorAccumulator: cannot open " + merged_path + " for read");
    }

    int64_t count(int64_t c) const { return bufs[static_cast<size_t>(c)].total_count; }

    // 顺序整块读出 bucket c 的全部 (gid, vector) 记录 (从合并文件里按已知偏移 seek)。
    void read_bucket(int64_t c, std::vector<int64_t>& ids_out, std::vector<DataT>& vecs_out) {
        int64_t cnt = count(c);
        ids_out.resize(static_cast<size_t>(cnt));
        vecs_out.resize(static_cast<size_t>(cnt) * static_cast<size_t>(D));
        if (cnt == 0) return;

        size_t rec_bytes = record_bytes(D);
        std::vector<char> raw(static_cast<size_t>(cnt) * rec_bytes);
        merged_in.seekg(static_cast<std::streamoff>(merged_offset[static_cast<size_t>(c)]));
        merged_in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
        if (!merged_in.good())
            throw std::runtime_error("BucketVectorAccumulator: read failed for bucket " + std::to_string(c));

        for (int64_t i = 0; i < cnt; ++i) {
            const char* p = raw.data() + static_cast<size_t>(i) * rec_bytes;
            std::memcpy(&ids_out[static_cast<size_t>(i)], p, sizeof(int64_t));
            std::memcpy(vecs_out.data() + static_cast<size_t>(i) * static_cast<size_t>(D),
                        p + sizeof(int64_t), static_cast<size_t>(D) * sizeof(DataT));
        }
    }

    // 跟 read_bucket 一样，但直接写进调用方提供的缓冲区 (ids_out_ptr 至少
    // count(c) 个 int64_t，vecs_out_ptr 至少 count(c)*D 个 DataT)，省掉每次
    // 调用都新分配 vector 的开销——热循环 (Step 6 逐 bucket) 里用这个。
    void read_bucket_into(int64_t c, int64_t* ids_out_ptr, DataT* vecs_out_ptr) {
        int64_t cnt = count(c);
        if (cnt == 0) return;

        size_t rec_bytes = record_bytes(D);
        std::vector<char> raw(static_cast<size_t>(cnt) * rec_bytes);
        merged_in.seekg(static_cast<std::streamoff>(merged_offset[static_cast<size_t>(c)]));
        merged_in.read(raw.data(), static_cast<std::streamsize>(raw.size()));
        if (!merged_in.good())
            throw std::runtime_error("BucketVectorAccumulator: read failed for bucket " + std::to_string(c));

        for (int64_t i = 0; i < cnt; ++i) {
            const char* p = raw.data() + static_cast<size_t>(i) * rec_bytes;
            std::memcpy(ids_out_ptr + i, p, sizeof(int64_t));
            std::memcpy(vecs_out_ptr + static_cast<size_t>(i) * static_cast<size_t>(D),
                        p + sizeof(int64_t), static_cast<size_t>(D) * sizeof(DataT));
        }
    }

    void remove_all_files() {
        if (merged_out.is_open()) merged_out.close();
        if (merged_in.is_open()) merged_in.close();
        std::filesystem::remove(merged_path);
    }
};

// 配合 BucketVectorAccumulator，作为 bucket_order::BeladyBucketCache<T> 的缓存值
// 类型：一份已经从磁盘读出来、常驻内存的 (gid, 向量) 副本 (对应 bucket_build.cu
// 里的 Bucket，只是 vecs 是模板化的 DataT 而不是固定 float)。
template <typename DataT>
struct VecBucket {
    std::vector<int64_t> ids;
    std::vector<DataT> vecs;
};

// ============== Phase 8: Batch Assignment via Graph ANNS ==============

// 定义在后面 (Step 6 的 gather kernel)；Step 4 out-of-core 路径复用它在 GPU 上
// 做 "按行 gather + 转 float"。
template <typename DataT>
__global__ void gather_rows_int32(const DataT* __restrict__ src, const int32_t* __restrict__ idx,
                                  float* __restrict__ dst, int64_t n_dst_rows, int64_t D);

/**
 * 分批处理原始数据集，在 GPU 上基于 centroid KNN 图做 ANNS 检索，
 * 为每个非 centroid 数据点找到最近的 2 个 centroid，在 CPU 上做均衡分配。
 *
 * centroid 数据和 KNN 图由 caller 提前上传至 GPU，本函数直接复用，不再重复上传。
 *
 * @tparam T                     原始数据元素类型 (float, uint8_t, uint32_t 等)
 * @tparam CentroidT             centroid 在 GPU 上的数据类型 (float 或 uint8_t)
 * @param input_path             原始数据集文件路径 (磁盘上, BIGANN 格式)；不再
 *                               整份读进内存，按 batch 顺序读取需要的连续区间
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
 * 本函数只负责算出 assignments[]，不写任何 bucket 向量数据——写入需要先知道
 * 完整的 assignments[] 才能对每个 bucket 做 histogram、算出它在合并文件里的
 * 字节区间 (见 BucketVectorAccumulator 的注释)，所以单独放在
 * write_bucket_vectors_from_assignments() 里，由 caller 在拿到这个函数的
 * 返回值之后再调用。
 *
 * @return  (N,) 每个点分配到的 local centroid index [0, n_centroids)
 */
template <typename T, typename CentroidT>
std::vector<int64_t> batch_assign_with_cagra_anns(
    const std::string& input_path,
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
    int search_max_iters = 64,
    // 非空时 (GPU-resident 路径，仅非 PQ 模式)：直接从显存常驻数据集把这批
    // 点 gather 进 d_queries，不再读盘 + CPU 转型 + H2D。
    const std::function<void(const int64_t* idx, int64_t n, CentroidT* d_out)>& gpu_query_loader = {})
{
    constexpr bool is_pq = std::is_same<CentroidT, uint8_t>::value;

    using Clock = std::chrono::high_resolution_clock;
    auto ba_t0 = Clock::now();
    double t_setup = 0;
    double t_read = 0, t_convert_upload = 0, t_search = 0, t_download = 0, t_cpu_assign = 0;

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
    auto t_setup0 = Clock::now();
    raft::resources res;
    auto dataset_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
        d_centroids, n_centroids, static_cast<int64_t>(D));
    auto graph_view = raft::make_device_matrix_view<const uint32_t, int64_t>(
        d_graph, n_centroids, static_cast<int64_t>(K));
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<CentroidT, int64_t>> padded_owner;
    auto cagra_idx = make_padded_cagra_index<CentroidT>(res, dataset_view, graph_view, padded_owner);

    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;
    sp.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
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

    // out-of-core 非 PQ: 原始 T 区间 + 行偏移直接上传，在 GPU 上 gather + 转
    // float (原来 CPU 转 float 再上传 4x 字节的 pageable 数据, ~0.5-0.6s/轮)。
    T*       d_raw_range   = nullptr;
    size_t   d_raw_cap     = 0;     // elements
    int32_t* d_local_rows  = nullptr;
    std::vector<int32_t> h_local_rows;

    // GPU-resident 路径 (非 PQ, queries 直接从显存 gather): 用 exact brute-force
    // top-2 (cuVS brute_force = 分块 GEMM + select_k，对应 cuVS 26.12
    // detail/knn_brute_force.cuh::tiled_brute_force_knn) 代替近似的 CAGRA 图
    // 搜索——10K centroids × 990K 点的 GEMM 在 V100 上比 CAGRA single-CTA
    // 搜索 (~0.7s/轮) 快，而且结果是精确 top-2。out-of-core / PQ 路径不变。
    // 非 PQ 时 out-of-core 也用 (centroids 很小、常驻 GPU；queries 本来就按
    // Nv 分批，显存仍受 Mbatch_bytes 约束，不依赖整份数据集在 GPU 上)。
    const bool use_bf = !is_pq;
    std::optional<cuvs::neighbors::brute_force::index<float, float>> bf_idx;
    int64_t* d_top2_i64 = nullptr;
    std::vector<int64_t> h_top2_i64;
    if constexpr (!is_pq) {
        if (use_bf) {
            cuvs::neighbors::brute_force::index_params bp;
            bp.metric = cuvs::distance::DistanceType::L2Expanded;
            bf_idx.emplace(cuvs::neighbors::brute_force::build(res, bp,
                raft::make_device_matrix_view<const float, int64_t>(
                    d_centroids, n_centroids, static_cast<int64_t>(D))));
            CUDA_CHECK(cudaMalloc(&d_top2_i64, Nv * TOP_K * sizeof(int64_t)));
            h_top2_i64.resize(static_cast<size_t>(Nv) * TOP_K);
        }
    }

    t_setup = std::chrono::duration<double>(Clock::now() - t_setup0).count();

    for (int64_t batch_start = 0; batch_start < N_nc; batch_start += Nv) {
        int64_t batch_end = std::min(batch_start + Nv, N_nc);
        int64_t batch_size = batch_end - batch_start;

        // ---- 4a: 读取 batch 数据 (一次连续区间读)，转换为 CentroidT，上传到 GPU ----
        // non_centroid_indices 在这个 batch 内基本连续 (只跳过零星的 centroid
        // id)，所以整段 [lo, hi] 一次顺序读比逐点 seek 快得多；读出来多覆盖的
        // 那几个 centroid 行的数据直接不用，忽略即可。
        auto tb0 = Clock::now();
        int64_t batch_lo = non_centroid_indices[batch_start];
        int64_t batch_hi = non_centroid_indices[batch_end - 1];
        std::vector<T> raw_batch;
        if (!is_pq && gpu_query_loader) {
            gpu_query_loader(non_centroid_indices.data() + batch_start, batch_size, d_queries);
        } else {
            int32_t N_tmp, D_tmp;
            load::read_bigann_raw_range<T>(input_path, batch_lo, batch_hi - batch_lo + 1,
                                            raw_batch, N_tmp, D_tmp);
        }
        auto tb1 = Clock::now();
        t_read += std::chrono::duration<double>(tb1 - tb0).count();

        if constexpr (is_pq) {
            // PQ 模式：读取原始 T 数据 -> 转为 float32 -> encode uint8 -> 上传
            std::vector<float> batch_f32(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                int64_t local = gi - batch_lo;
                for (int64_t d = 0; d < D; ++d) {
                    batch_f32[i * D + d] = static_cast<float>(raw_batch[local * D + d]);
                }
            }

            auto codes = simple_encode(batch_f32, batch_size, D, quantizer);

            CUDA_CHECK(cudaMemcpy(d_queries, codes.data(),
                                  batch_size * D * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        } else if (!gpu_query_loader && std::is_same<CentroidT, float>::value) {
            // 非 PQ 模式：原始 T 区间整段上传，GPU 上按非 centroid 行 gather + 转 float
            const int64_t n_raw = batch_hi - batch_lo + 1;
            const size_t raw_elems = static_cast<size_t>(n_raw) * D;
            if (raw_elems > d_raw_cap) {
                if (d_raw_range) CUDA_CHECK(cudaFree(d_raw_range));
                CUDA_CHECK(cudaMalloc(&d_raw_range, raw_elems * sizeof(T)));
                d_raw_cap = raw_elems;
            }
            if (!d_local_rows) CUDA_CHECK(cudaMalloc(&d_local_rows, Nv * sizeof(int32_t)));
            h_local_rows.resize(static_cast<size_t>(batch_size));
            int64_t n_bad = 0;
            #pragma omp parallel for schedule(static) reduction(+:n_bad)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t local = non_centroid_indices[batch_start + i] - batch_lo;
                if (local < 0 || local >= n_raw) { ++n_bad; local = 0; }   // bounds check
                h_local_rows[static_cast<size_t>(i)] = static_cast<int32_t>(local);
            }
            if (n_bad > 0) throw std::runtime_error("batch_assign: row offset out of read range");
            CUDA_CHECK(cudaMemcpy(d_raw_range, raw_batch.data(), raw_elems * sizeof(T),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_local_rows, h_local_rows.data(), batch_size * sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
            const int threads = 256;
            const int64_t total = batch_size * D;
            gather_rows_int32<T><<<static_cast<unsigned>((total + threads - 1) / threads), threads>>>(
                d_raw_range, d_local_rows, reinterpret_cast<float*>(d_queries), batch_size, D);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else if (!gpu_query_loader) {
            // 非 PQ 模式：读取原始 T 数据 -> 转为 CentroidT -> 上传
            std::vector<CentroidT> batch_data(batch_size * D);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < batch_size; ++i) {
                int64_t gi = non_centroid_indices[batch_start + i];
                int64_t local = gi - batch_lo;
                for (int64_t d = 0; d < D; ++d) {
                    batch_data[i * D + d] = static_cast<CentroidT>(raw_batch[local * D + d]);
                }
            }

            CUDA_CHECK(cudaMemcpy(d_queries, batch_data.data(),
                                  batch_size * D * sizeof(CentroidT),
                                  cudaMemcpyHostToDevice));
        }
        auto tb2 = Clock::now();
        t_convert_upload += std::chrono::duration<double>(tb2 - tb1).count();

        // ---- 4b: cagra::search (k=2, queries 不在 dataset 里, 无 self-exclusion) ----
        auto queries_view = raft::make_device_matrix_view<const CentroidT, int64_t>(
            d_queries, batch_size, static_cast<int64_t>(D));
        auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
            d_top2, batch_size, static_cast<int64_t>(TOP_K));
        auto distances_view = raft::make_device_matrix_view<float, int64_t>(
            d_distances, batch_size, static_cast<int64_t>(TOP_K));

        bool searched_bf = false;
        if constexpr (!is_pq) {
            if (use_bf) {
                cuvs::neighbors::brute_force::search(res, *bf_idx,
                    raft::make_device_matrix_view<const float, int64_t>(
                        d_queries, batch_size, static_cast<int64_t>(D)),
                    raft::make_device_matrix_view<int64_t, int64_t>(
                        d_top2_i64, batch_size, static_cast<int64_t>(TOP_K)),
                    distances_view);
                searched_bf = true;
            }
        }
        if (!searched_bf) {
            cuvs::neighbors::cagra::search(res, sp, cagra_idx,
                queries_view, neighbors_view, distances_view);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto tb3 = Clock::now();
        t_search += std::chrono::duration<double>(tb3 - tb2).count();

        // ---- 4c: 下载 top-2 local centroid indices ----
        if (searched_bf) {
            CUDA_CHECK(cudaMemcpy(h_top2_i64.data(), d_top2_i64,
                                  batch_size * TOP_K * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost));
            // bounds check: brute-force 结果必须是 [0, n_centroids) 内的 local idx
            int64_t n_bad = 0;
            #pragma omp parallel for schedule(static) reduction(+:n_bad)
            for (int64_t i = 0; i < batch_size * static_cast<int64_t>(TOP_K); ++i) {
                int64_t v = h_top2_i64[static_cast<size_t>(i)];
                if (v < 0 || v >= n_centroids) { ++n_bad; v = 0; }
                h_top2[static_cast<size_t>(i)] = static_cast<uint32_t>(v);
            }
            if (n_bad > 0)
                throw std::runtime_error("batch_assign: brute-force top-2 returned " +
                                         std::to_string(n_bad) + " out-of-range centroid ids");
        } else {
            CUDA_CHECK(cudaMemcpy(h_top2.data(), d_top2,
                                  batch_size * TOP_K * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        auto tb4 = Clock::now();
        t_download += std::chrono::duration<double>(tb4 - tb3).count();

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
        t_cpu_assign += std::chrono::duration<double>(Clock::now() - tb4).count();

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
    if (d_top2_i64)   CUDA_CHECK(cudaFree(d_top2_i64));
    if (d_raw_range)  CUDA_CHECK(cudaFree(d_raw_range));
    if (d_local_rows) CUDA_CHECK(cudaFree(d_local_rows));
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
    double ba_total = std::chrono::duration<double>(Clock::now() - ba_t0).count();
    std::cout << "  [Step4 timing] setup=" << std::fixed << std::setprecision(3) << t_setup
              << "s read=" << t_read
              << "s convert_upload=" << t_convert_upload
              << "s search=" << t_search
              << "s download=" << t_download
              << "s cpu_assign=" << t_cpu_assign
              << "s total=" << ba_total << "s\n";

    return assignments;
}

/**
 * 把完整的 assignments[]（batch_assign_with_cagra_anns 的返回值，这一轮每个
 * 点最终分到哪个 bucket 已经全部拍板）写进 bucket_vecs：先对 assignments[]
 * 做一次 histogram 得到每个 bucket 最终的记录数，用它调用
 * BucketVectorAccumulator::start_iteration()（这样每个 bucket 在合并文件里
 * 的字节区间提前算好，add() 直接写到最终位置，不会有任何小文件/合并/删除的
 * 来回——见 BucketVectorAccumulator 顶部的注释），再顺序分块重新读一遍原始
 * 数据集（纯 I/O，不需要 GPU，所以跟 batch_assign_with_cagra_anns 的 GPU
 * batch 大小无关，这里用单独的、只受 host 内存约束的 chunk 大小），对每个点
 * 调用 add()。
 *
 * 之所以要再读一遍磁盘而不是复用 batch_assign 内部已经读过的 raw_batch：
 * 对点 i 的 bucket 分配是否已经最终确定，取决于之后所有点的分配结果（负载
 * 均衡用的 cluster_sizes 是全程累计的），所以在 batch_assign 返回、
 * assignments[] 完全确定之前，不能提前对任何一个 bucket 做 histogram、也
 * 不能对任何一条记录定下它在合并文件里的最终位置——这意味着"边算边写"
 * (原来的做法) 和"先算完、histogram、再写"这两种方式，天然要么牺牲掉
 * histogram（原来的小文件方案），要么多付一次顺序读的代价（这里）。多一次
 * 顺序读的开销只跟数据集总大小成正比，而小文件方案的系统调用开销是跟桶的
 * 数量成正比——未来桶数量再扩大百倍千倍时，顺序读的代价不变规律，小文件的
 * 代价会线性放大，所以这个取舍在大规模场景下是划算的。
 */
template <typename DataT>
void write_bucket_vectors_from_assignments(
    const std::string& input_path,
    int64_t N,
    int64_t D,
    int64_t n_centroids,
    const std::vector<int64_t>& centroid_global_indices,
    const std::vector<int64_t>& assignments,
    BucketVectorAccumulator<DataT>& bucket_vecs,
    int64_t chunk_bytes_budget = 512LL * 1024 * 1024)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    // ---- histogram + start_iteration: 提前给每个 bucket 定好最终字节区间 ----
    std::vector<int64_t> bucket_counts(static_cast<size_t>(n_centroids), 0);
    for (int64_t i = 0; i < N; ++i) {
        int64_t c = assignments[static_cast<size_t>(i)];
        if (c >= 0) ++bucket_counts[static_cast<size_t>(c)];
    }
    bucket_vecs.start_iteration(bucket_counts);
    double t_hist_start = std::chrono::duration<double>(Clock::now() - t0).count();

    // ---- centroid 自己的向量: n_centroids 条，一次性小读 ----
    auto t_pre0 = Clock::now();
    {
        std::vector<DataT> centroid_vecs_raw;
        int32_t N_tmp, D_tmp;
        load::read_bigann_raw_sampled<DataT>(input_path, centroid_global_indices,
                                              centroid_vecs_raw, N_tmp, D_tmp);
        for (int64_t c = 0; c < n_centroids; ++c) {
            bucket_vecs.add(c, centroid_global_indices[c],
                             centroid_vecs_raw.data() + static_cast<size_t>(c) * D);
        }
    }
    double t_centroid_write = std::chrono::duration<double>(Clock::now() - t_pre0).count();

    // ---- 剩下的点: 顺序分块重新读磁盘 + add()，纯 I/O，不需要 GPU batching 的
    //      那套内存约束，chunk 大小只受 host 内存预算控制 ----
    std::unordered_set<int64_t> centroid_set(
        centroid_global_indices.begin(), centroid_global_indices.end());
    int64_t chunk_points = std::max<int64_t>(
        1, chunk_bytes_budget / std::max<int64_t>(1, D * static_cast<int64_t>(sizeof(DataT))));

    double t_read = 0, t_write = 0;
    for (int64_t lo = 0; lo < N; lo += chunk_points) {
        int64_t hi = std::min(lo + chunk_points, N) - 1;

        auto tr0 = Clock::now();
        std::vector<DataT> raw_chunk;
        int32_t N_tmp, D_tmp;
        load::read_bigann_raw_range<DataT>(input_path, lo, hi - lo + 1, raw_chunk, N_tmp, D_tmp);
        t_read += std::chrono::duration<double>(Clock::now() - tr0).count();

        auto tw0 = Clock::now();
        for (int64_t gi = lo; gi <= hi; ++gi) {
            if (centroid_set.find(gi) != centroid_set.end()) continue;  // 上面已经写过
            bucket_vecs.add(assignments[static_cast<size_t>(gi)], gi,
                             raw_chunk.data() + static_cast<size_t>(gi - lo) * D);
        }
        t_write += std::chrono::duration<double>(Clock::now() - tw0).count();
    }

    double total = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  [BucketWrite timing] hist_start=" << std::fixed << std::setprecision(3) << t_hist_start
              << "s centroid_write=" << t_centroid_write
              << "s read=" << t_read
              << "s write=" << t_write
              << "s total=" << total << "s\n";
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

/**
 * KMeans|| D² sampling (replaces raft::cluster::detail::SamplingOp/sampleCentroids):
 *   select i if !flag[i] && rand[i] < l * k * d²(i) / psi; marks flag[i] = 1, sel[i] = 1.
 */
__global__ void kmeans_par_sample_kernel(
    const float* __restrict__ min_dist,
    const float* __restrict__ rands,
    uint8_t* __restrict__ flags,
    uint8_t* __restrict__ sel,
    float psi, float l_times_k, int64_t N)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    bool take = !flags[i] && rands[i] < l_times_k * min_dist[i] / psi;
    sel[i] = take ? 1 : 0;
    if (take) flags[i] = 1;
}

/**
 * min_dist[i] = min_c ||X[i] - C[c]||² (replaces raft::cluster::detail::minClusterDistanceCompute),
 * via cuvs brute_force k=1, row-batched to bound the distance tile.
 */
static void min_cluster_distance(
    raft::resources const& res,
    raft::device_matrix_view<const float, int64_t> X,
    raft::device_matrix_view<const float, int64_t> C,
    float* d_min_dist)
{
    auto bf = cuvs::neighbors::brute_force::build(
        res, cuvs::neighbors::brute_force::index_params{}, C);
    constexpr int64_t BATCH = 1 << 16;
    auto nb = raft::make_device_matrix<int64_t, int64_t>(res, std::min(BATCH, X.extent(0)), 1);
    for (int64_t off = 0; off < X.extent(0); off += BATCH) {
        int64_t n = std::min(BATCH, X.extent(0) - off);
        cuvs::neighbors::brute_force::search(
            res, cuvs::neighbors::brute_force::search_params{}, bf,
            raft::make_device_matrix_view<const float, int64_t>(
                X.data_handle() + off * X.extent(1), n, X.extent(1)),
            raft::make_device_matrix_view<int64_t, int64_t>(nb.data_handle(), n, 1),
            raft::make_device_matrix_view<float, int64_t>(d_min_dist + off, n, 1));
    }
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

    using Clock = std::chrono::high_resolution_clock;
    auto step2_t0 = Clock::now();
    auto step2_elapsed = [&]() { return std::chrono::duration<double>(Clock::now() - step2_t0).count(); };
    double t_upload = 0, t_kmeans_par = 0, t_select = 0;

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
        cudaStream_t stream = raft::resource::get_cuda_stream(res).get();

        // 裸 cudaMalloc (不走 RMM 池): working set 可达数 GB (SIFT10M 全量 5.1GB)，
        // 远超 2GB 的池上限；池只留给 RAFT/cuVS 内部的小块 scratch。
        struct CudaFree { void operator()(float* p) const { cudaFree(p); } };
        std::unique_ptr<float, CudaFree> dataset_gpu_buf([&] {
            float* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, static_cast<size_t>(working_N) * D * sizeof(float)));
            return p;
        }());
        auto dataset_gpu = raft::make_device_matrix_view<float, int64_t>(
            dataset_gpu_buf.get(), working_N, static_cast<int64_t>(D));
        raft::copy(dataset_gpu.data_handle(), upload_data,
                   working_N * D, stream);
        raft::resource::sync_stream(res);

        auto X_view = raft::make_device_matrix_view<const float, int64_t>(
            dataset_gpu.data_handle(), working_N, static_cast<int64_t>(D));

        raft::resource::sync_stream(res, stream);
        t_upload = step2_elapsed();

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

        auto minClusterDistVec = raft::make_device_vector<float, int64_t>(res, working_N);
        auto uniformRands = raft::make_device_vector<float, int64_t>(res, working_N);
        auto d_sel_flags = raft::make_device_vector<uint8_t, int64_t>(res, working_N);

        // Step 2: Compute initial cost psi = phi_X(C)
        min_cluster_distance(res, X_view, raft::make_const_mdspan(potentialCentroids),
                             minClusterDistVec.data_handle());
        float psi = thrust::reduce(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(minClusterDistVec.data_handle()),
            thrust::device_pointer_cast(minClusterDistVec.data_handle()) + working_N,
            0.0f, thrust::plus<float>());
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
            min_cluster_distance(res, X_view, raft::make_const_mdspan(potentialCentroids),
                                 minClusterDistVec.data_handle());

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

            // flag + ordered compact + row gather (one kernel each, no per-point sync)
            rmm::device_uvector<float> CpRaw(0, stream);
            {
                int st = 256;
                int sb = static_cast<int>((working_N + st - 1) / st);
                kmeans_par_sample_kernel<<<sb, st, 0, stream>>>(
                    minClusterDistVec.data_handle(), uniformRands.data_handle(),
                    isSampleCentroid.data_handle(), d_sel_flags.data_handle(),
                    psi, static_cast<float>(oversampling_factor * n_centroids), working_N);
                CUDA_CHECK(cudaGetLastError());
                rmm::device_uvector<int64_t> d_sel_idx(working_N, stream);
                auto cnt0 = thrust::make_counting_iterator<int64_t>(0);
                auto sel_end = thrust::copy_if(
                    thrust::cuda::par.on(stream), cnt0, cnt0 + working_N,
                    thrust::device_pointer_cast(d_sel_flags.data_handle()),
                    thrust::device_pointer_cast(d_sel_idx.data()),
                    [] __device__ (uint8_t f) { return f > 0; });
                int64_t n_sel = sel_end - thrust::device_pointer_cast(d_sel_idx.data());
                CpRaw.resize(static_cast<size_t>(n_sel) * D, stream);
                if (n_sel > 0) {
                    int64_t tot_el = n_sel * static_cast<int64_t>(D);
                    gather_rows_by_index<<<static_cast<int>((tot_el + 255) / 256), 256, 0, stream>>>(
                        dataset_gpu.data_handle(), d_sel_idx.data(), CpRaw.data(),
                        n_sel, static_cast<int64_t>(D));
                    CUDA_CHECK(cudaGetLastError());
                }
            }

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
        t_kmeans_par = step2_elapsed() - t_upload;

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
        dataset_gpu_buf.reset();

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
        t_select = step2_elapsed() - t_upload - t_kmeans_par;
        std::cout << "  [Step2 timing] upload+norms=" << std::fixed << std::setprecision(3) << t_upload
                  << "s kmeans_parallel=" << t_kmeans_par
                  << "s select_final=" << t_select
                  << "s total=" << step2_elapsed() << "s\n";

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
 * 精确的 centroid 自 KNN (非 PQ 路径用): brute-force (分块 GEMM + select_k，
 * cuVS 26.12 detail/knn_brute_force.cuh) 一次算出每个 centroid 最近的 k_out
 * 个其它 centroid (去掉自己)。10K centroids 只要几十 ms；原来 Step 3 每轮
 * 跑一次完整的 cagra::build (NN-Descent + sort + prune, 还要先拷回 host)
 * ~0.2–0.5 s，Step 6 再对那张剪过枝的导航图做一次 CAGRA self-search 才得到
 * nprobe 近邻。这里一次拿到前 max(K, nprobe) 个精确近邻，前 K 列当 centroid
 * KNN 图 (bucket 顺序 / 重排 / 窗口)，前 nprobe 列直接当 Step 6 的近邻表。
 * 返回 host (n_centroids, k_out) row-major, 按距离升序; 越界 id 直接报错。
 */
std::vector<uint32_t> centroid_self_knn_bf(const float* d_centroids_f32, int64_t n_centroids,
                                           int64_t D, uint32_t k_out)
{
    if (static_cast<int64_t>(k_out) >= n_centroids)
        throw std::runtime_error("centroid_self_knn_bf: k_out must be < n_centroids");
    const int64_t ks = static_cast<int64_t>(k_out) + 1;   // 多取 1 个，剥掉自己
    raft::resources res;
    auto cview = raft::make_device_matrix_view<const float, int64_t>(d_centroids_f32, n_centroids, D);
    cuvs::neighbors::brute_force::index_params bp;
    bp.metric = cuvs::distance::DistanceType::L2Expanded;
    auto bf = cuvs::neighbors::brute_force::build(res, bp, cview);
    auto nb = raft::make_device_matrix<int64_t, int64_t>(res, n_centroids, ks);
    auto ds = raft::make_device_matrix<float, int64_t>(res, n_centroids, ks);
    cuvs::neighbors::brute_force::search(res, bf, cview, nb.view(), ds.view());
    std::vector<int64_t> h_nb(static_cast<size_t>(n_centroids * ks));
    raft::copy(h_nb.data(), nb.data_handle(), h_nb.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::vector<uint32_t> out(static_cast<size_t>(n_centroids) * k_out);
    int64_t n_bad = 0;
    #pragma omp parallel for schedule(static) reduction(+:n_bad)
    for (int64_t c = 0; c < n_centroids; ++c) {
        const int64_t* row = h_nb.data() + c * ks;
        uint32_t* o = out.data() + static_cast<size_t>(c) * k_out;
        uint32_t w = 0;
        bool self_seen = false;
        for (int64_t j = 0; j < ks && w < k_out; ++j) {
            int64_t v = row[j];
            if (v < 0 || v >= n_centroids) { ++n_bad; continue; }
            if (v == c && !self_seen) { self_seen = true; continue; }   // 自己 (距离相同的重复向量时不一定在第 0 位)
            o[w++] = static_cast<uint32_t>(v);
        }
        if (w < k_out) ++n_bad;
    }
    if (n_bad > 0)
        throw std::runtime_error("centroid_self_knn_bf: " + std::to_string(n_bad) + " bad rows");
    return out;
}

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

    cuvs::neighbors::cagra::index_params ip;
    ip.graph_degree              = K;
    ip.intermediate_graph_degree = intermediate_graph_degree;
    auto nnd = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(intermediate_graph_degree);
    nnd.graph_degree              = intermediate_graph_degree;
    nnd.intermediate_graph_degree = static_cast<size_t>(1.5 * intermediate_graph_degree);
    nnd.max_iterations            = max_iterations;
    nnd.termination_threshold     = termination_threshold;
    ip.graph_build_params         = nnd;

    auto dataset = cuvs::neighbors::make_host_standard_dataset_view(
        raft::make_host_matrix_view<const float, int64_t>(centroids_host.data(), n_centroids, D));

    raft::resources res;

    std::cout << "  cagra::build (NN Descent + sort + prune to K=" << K << ")...\n";
    auto idx = cuvs::neighbors::cagra::build(res, ip, dataset);
    auto cagra_graph = raft::make_host_matrix<uint32_t, int64_t>(n_centroids, K);
    raft::copy(cagra_graph.data_handle(), idx.graph().data_handle(), cagra_graph.size(),
               raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

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
    std::unique_ptr<cuvs::neighbors::device_padded_dataset<float, int64_t>> padded_owner;
    auto idx = make_padded_cagra_index<float>(res, dataset_view, graph_view, padded_owner);

    // 3) cagra::search (queries == dataset, k = nprobe+1)
    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size     = itopk;
    sp.search_width   = 1;
    sp.max_iterations = 0;  // 0 = auto
    sp.algo           = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;

    auto queries_view = raft::make_device_matrix_view<const float, int64_t>(
        d_dataset, n_centroids, static_cast<int64_t>(D));
    auto neighbors_view = raft::make_device_matrix_view<uint32_t, int64_t>(
        d_neighbors, n_centroids, static_cast<int64_t>(k_search));
    auto distances_view = raft::make_device_matrix_view<float, int64_t>(
        d_distances, n_centroids, static_cast<int64_t>(k_search));

    cuvs::neighbors::cagra::search(res, sp, idx,
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
    int bucket_size, int pool_size, int K,
    int out_ld = 0,                    // 0: 第 query_idx 行、行宽 K (原行为)；>0: 第 my_gid 行、行宽 out_ld
    const float* __restrict__ q_norms = nullptr)  // 非空: dist 加上 ||q||²，输出真实 L2²
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
    const int64_t out_row = (out_ld > 0) ? static_cast<int64_t>(my_gid) * out_ld
                                         : static_cast<int64_t>(query_idx) * K;
    const float q_norm = (q_norms != nullptr) ? q_norms[query_idx] : 0.0f;
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        if (base + p < K) {
            out_neighbors[out_row + base + p] = reg_id[p];
            if (out_dists != nullptr) {
                out_dists[out_row + base + p] = reg_dist[p] + q_norm;
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
    int bucket_size, int pool_size, int actual_M,
    int out_ld = 0,                    // 见 warp_topK_from_dots_kernel
    const float* d_q_norms = nullptr)
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
            bucket_size, pool_size, actual_M, out_ld, d_q_norms);                             \
    } else {                                                                                  \
        warp_topK_from_dots_kernel<(KV), float>                                               \
            <<<blocks, threads_per_block, 0, stream>>>(                                       \
            reinterpret_cast<const float*>(d_dots), d_norms_pool,                             \
            d_ids_bucket, d_ids_pool,                                                         \
            d_out_neighbors, d_out_distances,                                                 \
            bucket_size, pool_size, actual_M, out_ld, d_q_norms);                             \
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

// 一个 query 一个 block (G 个 warp)：warp_topK_from_dots_kernel 一 warp 扫一整行
// pool (~25000 候选)，一个 bucket 只有 bucket_size (~100) 个 warp，SM 占用率极低，
// 实测占了 GPU-resident Step 6 的 ~70% GPU 时间。这里 G 个 warp 交错扫同一行
// (仍然合并访存)，各自维护 K_OUT 寄存器 top-K，最后经 shared memory 由 warp 0
// 归并另外 G-1 份列表。另外: query 自己就是 pool 的第 query_idx 行 (pool 前
// bucket_size 行是 bucket 自己)，自环判断改成比较位置，主循环不再读 id；寄存器
// 里存 pool 内下标，写出时才查 gid。只用于 GPU-resident 路径 (行号 = gid)。
template <int K_PER_LANE>
__device__ __forceinline__ void warp_topk_insert(
    float (&reg_dist)[K_PER_LANE], int32_t (&reg_id)[K_PER_LANE],
    float c_dist, int32_t c_id, int lane)
{
    constexpr unsigned ALL = 0xFFFFFFFFu;
    unsigned accept = __ballot_sync(ALL, c_dist <= reg_dist[K_PER_LANE - 1]);
    int target = __ffs(accept) - 1;
    float   in_dist = __shfl_up_sync(ALL, reg_dist[K_PER_LANE - 1], 1);
    int32_t in_id   = __shfl_up_sync(ALL, reg_id  [K_PER_LANE - 1], 1);
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
}

template <int K_OUT, int G>
__global__ void __launch_bounds__(G * 32) block_topK_from_dots_kernel(
    const float*   __restrict__ dots,        // (bucket_size, ldd)，ldd 是 4 的倍数 (float4 对齐)
    const float*   __restrict__ norms_pool,  // (pool_size,)  前 bucket_size 个即 ‖q‖²
    const int32_t* __restrict__ ids_pool,    // (pool_size,)  前 bucket_size 个即 query gid
    int32_t* __restrict__ out_neighbors,     // 第 gid 行、行宽 out_ld
    float*   __restrict__ out_dists,         // 同上，真实 L2²
    int pool_size, int ldd, int K, int out_ld)
{
    constexpr int LANES = 32;
    constexpr int K_PER_LANE = K_OUT / LANES;
    constexpr unsigned ALL = 0xFFFFFFFFu;
    const int q    = blockIdx.x;
    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x & (LANES - 1);

    float   reg_dist[K_PER_LANE];
    int32_t reg_id  [K_PER_LANE];
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) { reg_dist[p] = INFINITY; reg_id[p] = -1; }
    float threshold = INFINITY;

    // 每 lane 每次读 4 个候选 (float4)，提高在途访存量——一个 bucket 只有
    // bucket_size 个 block，占用率天然不高，标量读时这个 kernel 是延迟瓶颈。
    const float* row = dots + static_cast<int64_t>(q) * ldd;
    for (int jb = warp * LANES * 4; jb < pool_size; jb += G * LANES * 4) {
        int j0 = jb + lane * 4;
        float dd[4];
        if (j0 + 3 < pool_size) {
            float4 dv = *reinterpret_cast<const float4*>(row + j0);
            float4 nv = *reinterpret_cast<const float4*>(norms_pool + j0);
            dd[0] = nv.x - 2.0f * dv.x; dd[1] = nv.y - 2.0f * dv.y;
            dd[2] = nv.z - 2.0f * dv.z; dd[3] = nv.w - 2.0f * dv.w;
        } else {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                dd[e] = (j0 + e < pool_size) ? norms_pool[j0 + e] - 2.0f * row[j0 + e] : INFINITY;
        }
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            if (j0 + e == q) dd[e] = INFINITY;
            unsigned mask = __ballot_sync(ALL, dd[e] < threshold);
            while (mask) {
                int src = __ffs(mask) - 1;
                mask &= mask - 1;
                float c_dist = __shfl_sync(ALL, dd[e], src);
                if (c_dist >= threshold) continue;
                warp_topk_insert<K_PER_LANE>(reg_dist, reg_id, c_dist, jb + src * 4 + e, lane);
                threshold = __shfl_sync(ALL, reg_dist[K_PER_LANE - 1], LANES - 1);
            }
        }
    }

    __shared__ float   s_dist[G][K_OUT];
    __shared__ int32_t s_id  [G][K_OUT];
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        s_dist[warp][lane * K_PER_LANE + p] = reg_dist[p];
        s_id  [warp][lane * K_PER_LANE + p] = reg_id[p];
    }
    __syncthreads();
    if (warp != 0) return;

    for (int w = 1; w < G; ++w) {
        for (int base = 0; base < K_OUT; base += LANES) {
            float my_dist = s_dist[w][base + lane];
            int32_t my_id = s_id[w][base + lane];
            unsigned mask = __ballot_sync(ALL, my_dist < threshold);
            while (mask) {
                int src = __ffs(mask) - 1;
                mask &= mask - 1;
                float   c_dist = __shfl_sync(ALL, my_dist, src);
                int32_t c_id   = __shfl_sync(ALL, my_id, src);
                if (c_dist >= threshold) continue;
                warp_topk_insert<K_PER_LANE>(reg_dist, reg_id, c_dist, c_id, lane);
                threshold = __shfl_sync(ALL, reg_dist[K_PER_LANE - 1], LANES - 1);
            }
        }
    }

    const int64_t out_row = static_cast<int64_t>(ids_pool[q]) * out_ld;
    const float q_norm = norms_pool[q];
    int base = lane * K_PER_LANE;
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        if (base + p < K) {
            int32_t local = reg_id[p];
            out_neighbors[out_row + base + p] = (local >= 0) ? ids_pool[local] : -1;
            out_dists[out_row + base + p]     = (local >= 0) ? reg_dist[p] + q_norm : INFINITY;
        }
    }
}

// GPU-resident 路径的 top-M dispatcher；M > 256 时 shared memory 不够，回退到
// 一 warp 一 query 的 warp_topK_from_dots_kernel。
static inline void launch_block_topM_resident(
    cudaStream_t stream, const float* d_dots, const float* d_norms_pool, const int32_t* d_ids_pool,
    int32_t* d_out_neighbors, float* d_out_dists,
    int bucket_size, int pool_size, int ldd, int actual_M, int out_ld)
{
    constexpr int G = 8;
#define LAUNCH_BLOCK_KOUT(KV) block_topK_from_dots_kernel<(KV), G><<<bucket_size, G * 32, 0, stream>>>( \
        d_dots, d_norms_pool, d_ids_pool, d_out_neighbors, d_out_dists, pool_size, ldd, actual_M, out_ld)
    if      (actual_M <= 32)  LAUNCH_BLOCK_KOUT(32);
    else if (actual_M <= 64)  LAUNCH_BLOCK_KOUT(64);
    else if (actual_M <= 128) LAUNCH_BLOCK_KOUT(128);
    else if (actual_M <= 256) LAUNCH_BLOCK_KOUT(256);
    else
        throw std::runtime_error("GPU-resident path supports neighbors-m <= 256 (got "
                                 + std::to_string(actual_M) + "); use --out-of-core");
#undef LAUNCH_BLOCK_KOUT
}

// ============== GPU-resident fast path (数据集 + running KNN 都放得进显存时) ==============
//
// 下面的 build_vector_knn_with_tensorcore + ChunkedKnnAccumulator 是 out-of-core
// 路径：每个 bucket 的 pool 在 CPU 上拼、H2D 上传，结果 D2H 后按 chunk 落盘，
// 每轮再从磁盘读回来合并。数据集能整个放进显存时 (SIFT1M 这种规模)，这些
// CPU/磁盘往返就是纯开销——实测 pool_build + scatter + 磁盘 merge + 最终写盘
// 占了 Step 6 墙钟的 85% 以上，纯 GEMM 只占 ~12%。这里把整份数据集 (GEMM
// 输入格式)、‖x‖²、running KNN 全部常驻显存 (跟 CAGRA 的 IVF-PQ / NN-Descent
// build 一样，整个 build 只在 GPU 上跑，只在最后下载一次结果)：
//   - pool 用 GPU gather kernel 按 bucket 成员表 (CSR) 从常驻数据集直接拼
//   - top-M 结果按 gid 直接写进本轮的 (N, M) 结果表，不 D2H
//   - 跨轮合并是一个 per-point GPU kernel，不落盘
//   - bucket 循环里没有任何 host 同步，多 stream 轮转，只有 kernel launch
// 8-bit 输入 (uint8/int8) 存成 fp16、走 fp16 Tensor Core GEMM (fp32 累加)：
// |v| ≤ 255 在 fp16 里精确，乘积 ≤ 65025，128 维求和 ≤ 8.3M < 2^24，fp32 累加
// 精确，所以距离跟 INT8 路径完全一致；V100 (sm_70) 没有 INT8 Tensor Core，
// INT8 GemmEx 在它上面走的是 dp4a，比 fp16 Tensor Core 慢一倍。其它类型 fp32 SGEMM。

template <typename DataT> struct ResidentGemmType { using type = float; };
template <> struct ResidentGemmType<uint8_t> { using type = __half; };
template <> struct ResidentGemmType<int8_t>  { using type = __half; };

template <typename T> __device__ __forceinline__ float resident_to_f32(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float resident_to_f32<__half>(__half v) { return __half2float(v); }
template <typename T> __device__ __forceinline__ T resident_from_f32(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half resident_from_f32<__half>(float v) { return __float2half_rn(v); }

template <typename DataT, typename GemmT>
__global__ void resident_convert_kernel(const DataT* __restrict__ src, GemmT* __restrict__ dst, int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < total) dst[tid] = resident_from_f32<GemmT>(static_cast<float>(src[tid]));
}

// 一 warp 一行
template <typename GemmT>
__global__ void resident_row_norms_kernel(const GemmT* __restrict__ X, float* __restrict__ norms,
                                          int64_t N, int D)
{
    int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (row >= N) return;
    float s = 0.0f;
    for (int d = lane; d < D; d += 32) {
        float v = resident_to_f32(X[row * D + d]);
        s += v * v;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xFFFFFFFFu, s, o);
    if (lane == 0) norms[row] = s;
}

// dst[i, :] = src[idx[i], :]
template <typename SrcT, typename DstT, typename IdxT>
__global__ void resident_gather_rows_kernel(const SrcT* __restrict__ src, const IdxT* __restrict__ idx,
                                            DstT* __restrict__ dst, int64_t n_rows, int D)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_rows * D) return;
    int64_t row = tid / D;
    int64_t col = tid - row * D;
    dst[tid] = resident_from_f32<DstT>(resident_to_f32(src[static_cast<int64_t>(idx[row]) * D + col]));
}

// 拼 bucket c 的 pool：一个 block 负责一个段 (段 0 = c 自己，段 1..P = 它的
// nprobe 个近邻 bucket)，按 CSR 成员表把行从常驻数据集拷到 pool 的段偏移处，
// 顺带拷 gid / ‖x‖²。段 0 在最前面，所以 pool 的前 bucket_size 行就是 GEMM
// 的 A 操作数 (跟 out-of-core 路径同一个约定)。
template <typename GemmT>
__global__ void resident_gather_pool_kernel(
    const GemmT*   __restrict__ data,
    const float*   __restrict__ norms,
    const int32_t* __restrict__ members,         // (N,) 按 bucket 分组的 gid
    const int64_t* __restrict__ member_offsets,  // (n_centroids+1,)
    const int32_t* __restrict__ seg_bucket,      // (n_seg,) 这个 pool 每段的 bucket id, -1 = 无效
    const int32_t* __restrict__ seg_ofs,         // (n_seg,) 每段在 pool 里的起始行
    GemmT*   __restrict__ pool,
    float*   __restrict__ pool_norms,
    int32_t* __restrict__ pool_ids,
    int D)
{
    int b = seg_bucket[blockIdx.x];
    if (b < 0) return;
    int64_t lo = member_offsets[b];
    int64_t cnt = member_offsets[b + 1] - lo;
    int64_t dst0 = seg_ofs[blockIdx.x];
    size_t row_bytes = static_cast<size_t>(D) * sizeof(GemmT);
    if (row_bytes % 16 == 0) {
        int vec_per_row = static_cast<int>(row_bytes / 16);
        for (int64_t t = threadIdx.x; t < cnt * vec_per_row; t += blockDim.x) {
            int64_t r = t / vec_per_row;
            int v = static_cast<int>(t - r * vec_per_row);
            int64_t gid = members[lo + r];
            reinterpret_cast<int4*>(pool + (dst0 + r) * D)[v] =
                reinterpret_cast<const int4*>(data + gid * D)[v];
        }
    } else {
        for (int64_t t = threadIdx.x; t < cnt * D; t += blockDim.x) {
            int64_t r = t / D;
            int64_t gid = members[lo + r];
            pool[(dst0 + r) * D + (t - r * D)] = data[gid * D + (t - r * D)];
        }
    }
    for (int64_t r = threadIdx.x; r < cnt; r += blockDim.x) {
        int32_t gid = members[lo + r];
        pool_ids[dst0 + r]   = gid;
        pool_norms[dst0 + r] = norms[gid];
    }
}

__global__ void resident_fill_sentinel_kernel(int32_t* __restrict__ n, float* __restrict__ d, int64_t total)
{
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    n[tid] = -1;
    d[tid] = INFINITY;
}

// 跟 ChunkedKnnAccumulator::merge_one_record 同一语义：两边都按距离升序、
// -1/+inf 结尾；双指针归并取 top-M，new 侧的 id 如果在 running 侧出现过就跳过
// (running 侧自己无重复；new 侧来自互不相交的 bucket 拼成的 pool，也无重复)。
// 一线程一个点。
__global__ void resident_merge_kernel(
    const int32_t* __restrict__ run_n, const float* __restrict__ run_d,
    const int32_t* __restrict__ new_n, const float* __restrict__ new_d,
    int32_t* __restrict__ out_n, float* __restrict__ out_d,
    int64_t N, int M)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const int32_t* an = run_n + row * M; const float* ad = run_d + row * M;
    const int32_t* bn = new_n + row * M; const float* bd = new_d + row * M;
    int32_t* on = out_n + row * M; float* od = out_d + row * M;
    int na = 0; while (na < M && an[na] >= 0) ++na;
    int nb = 0; while (nb < M && bn[nb] >= 0) ++nb;
    int i = 0, j = 0, w = 0;
    while (w < M && (i < na || j < nb)) {
        if (j >= nb || (i < na && ad[i] <= bd[j])) {
            on[w] = an[i]; od[w] = ad[i]; ++w; ++i;
        } else {
            int32_t id = bn[j];
            bool dup = false;
            for (int k = 0; k < na; ++k) {
                if (an[k] == id) { dup = true; break; }
            }
            if (!dup) { on[w] = id; od[w] = bd[j]; ++w; }
            ++j;
        }
    }
    for (; w < M; ++w) { on[w] = -1; od[w] = INFINITY; }
}

template <typename DataT>
struct GpuResidentKnn {
    using GemmT = typename ResidentGemmType<DataT>::type;
    static constexpr bool kHalf = std::is_same<GemmT, __half>::value;
    static constexpr int kStreams = 4;

    int64_t N = 0;
    int     D = 0;
    int     M = 0;
    GemmT*   d_data  = nullptr;   // (N, D) 当前坐标系
    float*   d_norms = nullptr;   // (N,)
    // 三份 (N, M) 表: run = 跨轮累积, cur = 本轮按 gid 写入, tmp = merge 输出 (之后跟 run 交换)
    int32_t* d_run_n = nullptr; float* d_run_d = nullptr;
    int32_t* d_cur_n = nullptr; float* d_cur_d = nullptr;
    int32_t* d_tmp_n = nullptr; float* d_tmp_d = nullptr;
    bool has_run = false;

    // per-stream 工作区，按需扩容、跨轮复用
    struct Slot {
        cudaStream_t   stream = nullptr;
        cublasHandle_t cublas = nullptr;
        GemmT*   pool       = nullptr;
        float*   pool_norms = nullptr;
        int32_t* pool_ids   = nullptr;
        float*   dots       = nullptr;
        size_t   pool_cap = 0, norms_cap = 0, ids_cap = 0, dots_cap = 0;   // 元素数
    };
    Slot slots[kStreams];

    // 本轮的 bucket 结构 (device)
    int32_t* d_members = nullptr;   size_t members_cap = 0;
    int64_t* d_member_offsets = nullptr; size_t offsets_cap = 0;
    int32_t* d_seg_bucket = nullptr; size_t seg_bucket_cap = 0;
    int32_t* d_seg_ofs = nullptr;    size_t seg_ofs_cap = 0;

    static size_t bytes_needed(int64_t N, int D, int M) {
        return static_cast<size_t>(N) * D * sizeof(GemmT) + static_cast<size_t>(N) * sizeof(float)
             + 3 * static_cast<size_t>(N) * M * (sizeof(int32_t) + sizeof(float))
             + static_cast<size_t>(N) * (sizeof(int32_t) + sizeof(int64_t));
    }

    GpuResidentKnn(const DataT* h_data, int64_t N_, int D_, int M_) : N(N_), D(D_), M(M_) {
        size_t total = static_cast<size_t>(N) * D;
        DataT* d_raw = nullptr;
        CUDA_CHECK(cudaMalloc(&d_raw, total * sizeof(DataT)));
        CUDA_CHECK(cudaMemcpy(d_raw, h_data, total * sizeof(DataT), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(GemmT)));
        resident_convert_kernel<DataT, GemmT><<<(total + 255) / 256, 256>>>(d_raw, d_data, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_raw));
        CUDA_CHECK(cudaMalloc(&d_norms, N * sizeof(float)));
        compute_norms();

        size_t nm = static_cast<size_t>(N) * M;
        CUDA_CHECK(cudaMalloc(&d_run_n, nm * sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&d_run_d, nm * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cur_n, nm * sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&d_cur_d, nm * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tmp_n, nm * sizeof(int32_t))); CUDA_CHECK(cudaMalloc(&d_tmp_d, nm * sizeof(float)));

        for (auto& s : slots) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
            if (cublasCreate(&s.cublas) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("GpuResidentKnn: cublasCreate failed");
            cublasSetStream(s.cublas, s.stream);
            cublasSetMathMode(s.cublas, CUBLAS_TENSOR_OP_MATH);
        }
    }

    ~GpuResidentKnn() {
        for (auto& s : slots) {
            cudaFree(s.pool); cudaFree(s.pool_norms); cudaFree(s.pool_ids); cudaFree(s.dots);
            if (s.cublas) cublasDestroy(s.cublas);
            if (s.stream) cudaStreamDestroy(s.stream);
        }
        cudaFree(d_members); cudaFree(d_member_offsets); cudaFree(d_seg_bucket); cudaFree(d_seg_ofs);
        cudaFree(d_data); cudaFree(d_norms);
        cudaFree(d_run_n); cudaFree(d_run_d); cudaFree(d_cur_n); cudaFree(d_cur_d);
        cudaFree(d_tmp_n); cudaFree(d_tmp_d);
    }
    GpuResidentKnn(const GpuResidentKnn&) = delete;
    GpuResidentKnn& operator=(const GpuResidentKnn&) = delete;

    void compute_norms() {
        int64_t threads = static_cast<int64_t>(N) * 32;
        resident_row_norms_kernel<GemmT><<<(threads + 255) / 256, 256>>>(d_data, d_norms, N, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // round 1 重排之后换到新坐标系: data[new] = data[inverse_perm[new]]
    void permute(const std::vector<uint32_t>& inverse_perm) {
        if (static_cast<int64_t>(inverse_perm.size()) != N)
            throw std::runtime_error("GpuResidentKnn::permute: inverse_perm size != N");
        uint32_t* d_inv = nullptr;
        GemmT* d_new = nullptr;
        size_t total = static_cast<size_t>(N) * D;
        CUDA_CHECK(cudaMalloc(&d_inv, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_new, total * sizeof(GemmT)));
        CUDA_CHECK(cudaMemcpy(d_inv, inverse_perm.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        resident_gather_rows_kernel<GemmT, GemmT, uint32_t><<<(total + 255) / 256, 256>>>(
            d_data, d_inv, d_new, N, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_inv));
        d_data = d_new;
        compute_norms();
    }

    // Step 4 的 query 直接从常驻数据集 gather 成 fp32，不再读盘/CPU 转型/H2D
    void gather_queries_f32(const int64_t* h_idx, int64_t n, float* d_out) {
        int64_t* d_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_idx, n * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(d_idx, h_idx, n * sizeof(int64_t), cudaMemcpyHostToDevice));
        int64_t total = n * D;
        resident_gather_rows_kernel<GemmT, float, int64_t><<<(total + 255) / 256, 256>>>(
            d_data, d_idx, d_out, n, D);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_idx));
    }

    template <typename T>
    static void ensure_device(T*& p, size_t& cap, size_t n) {
        if (n <= cap) return;
        if (p) CUDA_CHECK(cudaFree(p));
        CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
        cap = n;
    }

    // 一轮: assignments 是当前坐标系下每个点的 bucket id (每个点恰好属于一个
    // bucket)；centroid_topK 是 (n_centroids, P) 的 nprobe 近邻 bucket 表。
    void run_round(const std::vector<int64_t>& assignments, int64_t n_centroids,
                   const uint32_t* centroid_topK, uint32_t P) {
        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();
        if (static_cast<int64_t>(assignments.size()) != N)
            throw std::runtime_error("GpuResidentKnn::run_round: assignments size != N");

        // ---- 1) CSR: bucket -> 成员 gid ----
        std::vector<int64_t> offsets(n_centroids + 1, 0);
        for (int64_t i = 0; i < N; ++i) {
            int64_t b = assignments[i];
            if (b < 0 || b >= n_centroids)
                throw std::runtime_error("GpuResidentKnn::run_round: point " + std::to_string(i) + " unassigned");
            ++offsets[b + 1];
        }
        for (int64_t c = 0; c < n_centroids; ++c) offsets[c + 1] += offsets[c];
        std::vector<int32_t> members(N);
        {
            std::vector<int64_t> cursor(offsets.begin(), offsets.end() - 1);
            for (int64_t i = 0; i < N; ++i) members[cursor[assignments[i]]++] = static_cast<int32_t>(i);
        }

        // ---- 2) 每个 bucket 的 pool 段表 (段 0 = 自己, 段 1..P = 近邻 bucket) ----
        const int64_t S = static_cast<int64_t>(P) + 1;
        std::vector<int32_t> seg_bucket(n_centroids * S), seg_ofs(n_centroids * S);
        std::vector<int64_t> pool_size(n_centroids);
        #pragma omp parallel for schedule(static)
        for (int64_t c = 0; c < n_centroids; ++c) {
            int64_t ofs = 0;
            for (int64_t j = 0; j < S; ++j) {
                int64_t b = (j == 0) ? c : static_cast<int64_t>(centroid_topK[c * P + (j - 1)]);
                if (b >= n_centroids) b = -1;
                seg_bucket[c * S + j] = static_cast<int32_t>(b);
                seg_ofs[c * S + j]    = static_cast<int32_t>(ofs);
                if (b >= 0) ofs += offsets[b + 1] - offsets[b];
            }
            pool_size[c] = ofs;
        }
        int64_t max_pool = 0, max_bucket = 0;
        for (int64_t c = 0; c < n_centroids; ++c) {
            max_pool   = std::max(max_pool, pool_size[c]);
            max_bucket = std::max(max_bucket, offsets[c + 1] - offsets[c]);
        }

        ensure_device(d_members, members_cap, static_cast<size_t>(N));
        ensure_device(d_member_offsets, offsets_cap, static_cast<size_t>(n_centroids + 1));
        ensure_device(d_seg_bucket, seg_bucket_cap, static_cast<size_t>(n_centroids * S));
        ensure_device(d_seg_ofs, seg_ofs_cap, static_cast<size_t>(n_centroids * S));
        CUDA_CHECK(cudaMemcpy(d_members, members.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_member_offsets, offsets.data(), (n_centroids + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seg_bucket, seg_bucket.data(), n_centroids * S * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seg_ofs, seg_ofs.data(), n_centroids * S * sizeof(int32_t), cudaMemcpyHostToDevice));

        for (auto& s : slots) {
            ensure_device(s.pool, s.pool_cap, static_cast<size_t>(max_pool) * D);
            ensure_device(s.pool_norms, s.norms_cap, static_cast<size_t>(max_pool));
            ensure_device(s.pool_ids, s.ids_cap, static_cast<size_t>(max_pool));
            ensure_device(s.dots, s.dots_cap, static_cast<size_t>(max_bucket) * ((max_pool + 3) / 4 * 4));
        }

        size_t nm = static_cast<size_t>(N) * M;
        resident_fill_sentinel_kernel<<<(nm + 255) / 256, 256>>>(d_cur_n, d_cur_d, nm);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double t_setup = std::chrono::duration<double>(Clock::now() - t0).count();

        // ---- 3) bucket 循环: gather -> GEMM -> top-M (按 gid 写 cur)，无 host 同步 ----
        auto t1 = Clock::now();
        int64_t launched = 0;
        for (int64_t c = 0; c < n_centroids; ++c) {
            int bucket_size = static_cast<int>(offsets[c + 1] - offsets[c]);
            int ps = static_cast<int>(pool_size[c]);
            int ldd = (ps + 3) / 4 * 4;   // dots 行宽对齐到 float4
            int actual_M = std::min(M, ps - 1);
            if (bucket_size == 0 || actual_M <= 0) continue;
            Slot& s = slots[launched % kStreams];
            ++launched;

            resident_gather_pool_kernel<GemmT><<<static_cast<unsigned>(S), 256, 0, s.stream>>>(
                d_data, d_norms, d_members, d_member_offsets,
                d_seg_bucket + c * S, d_seg_ofs + c * S,
                s.pool, s.pool_norms, s.pool_ids, D);
            CUDA_CHECK(cudaGetLastError());

            // dots (bucket_size, ps) row-major = pool[0:bucket_size] · pool^T
            float alpha = 1.0f, beta = 0.0f;
            cublasStatus_t stat;
            if constexpr (kHalf) {
                stat = cublasGemmEx(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    ps, bucket_size, D, &alpha,
                                    s.pool, CUDA_R_16F, D,
                                    s.pool, CUDA_R_16F, D, &beta,
                                    s.dots, CUDA_R_32F, ldd,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            } else {
                stat = cublasSgemm(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                   ps, bucket_size, D, &alpha,
                                   s.pool, D, s.pool, D, &beta, s.dots, ldd);
            }
            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("GpuResidentKnn: GEMM failed, status=" + std::to_string(stat));

            // pool 前 bucket_size 行就是 query 自己: ids / ‖q‖² 都直接取 pool 的前缀
            launch_block_topM_resident(s.stream, s.dots, s.pool_norms, s.pool_ids,
                                       d_cur_n, d_cur_d, bucket_size, ps, ldd, actual_M, /*out_ld=*/M);
            CUDA_CHECK(cudaGetLastError());
        }
        double t_launch = std::chrono::duration<double>(Clock::now() - t1).count();
        CUDA_CHECK(cudaDeviceSynchronize());
        double t_loop = std::chrono::duration<double>(Clock::now() - t1).count();

        // ---- 4) 跨轮合并 ----
        auto t2 = Clock::now();
        if (!has_run) {
            std::swap(d_run_n, d_cur_n); std::swap(d_run_d, d_cur_d);
            has_run = true;
        } else {
            resident_merge_kernel<<<(N + 127) / 128, 128>>>(
                d_run_n, d_run_d, d_cur_n, d_cur_d, d_tmp_n, d_tmp_d, N, M);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(d_run_n, d_tmp_n); std::swap(d_run_d, d_tmp_d);
        }
        double t_merge = std::chrono::duration<double>(Clock::now() - t2).count();

        std::cout << "  [GpuResidentKNN] buckets=" << launched << " max_pool=" << max_pool
                  << " max_bucket=" << max_bucket << " gemm=" << (kHalf ? "FP16-TC" : "FP32")
                  << " | setup=" << std::fixed << std::setprecision(3) << t_setup
                  << "s loop=" << t_loop << "s (host launch " << t_launch
                  << "s) merge=" << t_merge << "s\n";
    }

    // 最终结果只下载一次，三个文件并行写 (npy_path 为空则不写 .npy)
    void write_outputs(const std::string& knn_path, const std::string& dist_path,
                       const std::string& npy_path) const {
        size_t nm = static_cast<size_t>(N) * M;
        std::vector<int32_t> h_n(nm);
        std::vector<float>   h_d(nm);
        CUDA_CHECK(cudaMemcpy(h_n.data(), d_run_n, nm * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_d.data(), d_run_d, nm * sizeof(float), cudaMemcpyDeviceToHost));

        int32_t M32 = M;
        auto write_bin = [&](const std::string& path, const void* body, size_t bytes) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(&M32), sizeof(int32_t));
            out.write(reinterpret_cast<const char*>(body), static_cast<std::streamsize>(bytes));
            if (!out.good()) throw std::runtime_error("GpuResidentKnn: write failed: " + path);
        };
        auto f_knn  = std::async(std::launch::async, [&] { write_bin(knn_path, h_n.data(), nm * sizeof(int32_t)); });
        auto f_dist = std::async(std::launch::async, [&] { write_bin(dist_path, h_d.data(), nm * sizeof(float)); });
        if (!npy_path.empty()) {
            std::vector<int64_t> h64(nm);
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(nm); ++i) h64[i] = h_n[i];
            size_t header_bytes = load::create_npy_int64_2d(npy_path, N, M);
            std::fstream out(npy_path, std::ios::binary | std::ios::in | std::ios::out);
            out.seekp(static_cast<std::streamoff>(header_bytes));
            out.write(reinterpret_cast<const char*>(h64.data()), static_cast<std::streamsize>(nm * sizeof(int64_t)));
            if (!out.good()) throw std::runtime_error("GpuResidentKnn: write failed: " + npy_path);
        }
        f_knn.get();
        f_dist.get();
    }
};

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

    // 最后一轮的 merge 直接把每个 chunk 的合并结果 pwrite 到最终的
    // vector_knn.bin / vector_dists.bin (chunk c 的行在文件里就是连续的一段)，
    // 不再先写 running_chunk_* 再由 finalize_to_file 整份读回、拼接、重写一遍
    // ——省掉一整遍 N*M*8 B 的写 + 读 + 写 (out-of-core 时 page cache 早已被
    // dirty 页顶满，这些写都是按磁盘带宽限速的)。
    std::string final_knn_path, final_dist_path;
    int final_iter = -1;
    bool final_written = false;
    void set_final_output(const std::string& knn_path, const std::string& dist_path, int iter) {
        final_knn_path = knn_path; final_dist_path = dist_path; final_iter = iter;
    }

    // dir 必须已存在 (调用方负责创建); cpu_mem_budget_bytes 用来限制 merge
    // 阶段并行处理多少个 chunk (每个 chunk 需要 chunk_size*M*4B*2 常驻内存)。
    static ChunkedKnnAccumulator create(const std::string& dir, int64_t N, int M,
                                        size_t cpu_mem_budget_bytes) {
        ChunkedKnnAccumulator acc;
        acc.dir = dir;
        acc.N = N;
        acc.M = M;

        // ---- chunk 数量/大小：由内存预算 + 本机核心数反推，不再写死常量 ----
        //
        // 目标 chunk_size 反过来算：如果想让 merge 阶段真的能同时调动
        // desired_parallelism (= 本机 omp_get_max_threads()) 个线程各自处理
        // 一个 chunk，在 cpu_mem_budget_bytes 这个总预算下，每个 chunk 能分
        // 到多少常驻内存 (每点 M*2*4B，nbrs+dists)？反推出 chunk_size，再算
        // num_chunks = ceil(N/chunk_size)。这样 chunk 粒度会随数据集大小、
        // --cpu-limit、以及机器核心数自动变化，不需要手调常量。
        //
        // kMinChunkPoints 是下限，防止内存预算特别紧或核心数特别多时 chunk
        // 碎成不合理的小块 (per-chunk 固定开销、文件数量会失控)；这种情况下
        // 实际能达到的并发数会小于核心数，这是合理的降级，比强行切碎更好。
        //
        // num_chunks 还要再按 RLIMIT_NOFILE 的实际余量封顶 (Step 6 build 期间
        // 每个 chunk 常驻打开一个 arrival 文件句柄；main() 已尽量把软限制提到
        // 硬限制附近)。撞顶时 chunk_size 会相应放大，这是唯一还需要的兜底。
        constexpr int64_t kMinChunkPoints = 10000;
        int desired_parallelism = std::max(1, omp_get_max_threads());
        size_t bytes_per_point = static_cast<size_t>(M) * 2 * sizeof(int32_t);  // nbrs+dists
        size_t mem_per_chunk_budget = std::max<size_t>(1,
            cpu_mem_budget_bytes / static_cast<size_t>(desired_parallelism));
        int64_t chunk_size_from_mem = std::max<int64_t>(kMinChunkPoints,
            static_cast<int64_t>(mem_per_chunk_budget / std::max<size_t>(1, bytes_per_point)));

        int64_t kMaxOpenChunks = 512;  // getrlimit 失败时的兜底默认值
        {
            struct rlimit rl;
            if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
                constexpr rlim_t kFdReserve = 64;
                rlim_t usable = (rl.rlim_cur > kFdReserve) ? (rl.rlim_cur - kFdReserve) : 1;
                kMaxOpenChunks = std::max<int64_t>(1, static_cast<int64_t>(usable));
            }
        }
        int64_t n_by_mem = std::max<int64_t>(1,
            (N + chunk_size_from_mem - 1) / chunk_size_from_mem);
        acc.num_chunks = std::min(n_by_mem, kMaxOpenChunks);
        acc.chunk_size = (N + acc.num_chunks - 1) / acc.num_chunks;

        size_t per_chunk_bytes = static_cast<size_t>(acc.chunk_size) * bytes_per_point;
        int budget_threads = static_cast<int>(std::max<size_t>(1,
            cpu_mem_budget_bytes / std::max<size_t>(1, per_chunk_bytes)));
        acc.merge_parallelism = std::max(1, std::min(desired_parallelism, budget_threads));

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

        const bool is_final = (iter == final_iter);
        int fd_knn = -1, fd_dist = -1;
        const off_t hdr = static_cast<off_t>(sizeof(int64_t) + sizeof(int32_t));
        if (is_final) {
            auto open_out = [&](const std::string& path) {
                int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
                if (fd < 0) throw std::runtime_error("ChunkedKnnAccumulator: cannot open " + path);
                char h[sizeof(int64_t) + sizeof(int32_t)];
                int32_t M32 = static_cast<int32_t>(M);
                std::memcpy(h, &N, sizeof(int64_t));
                std::memcpy(h + sizeof(int64_t), &M32, sizeof(int32_t));
                if (::pwrite(fd, h, sizeof(h), 0) != static_cast<ssize_t>(sizeof(h)) ||
                    ::ftruncate(fd, hdr + static_cast<off_t>(N) * M * 4) != 0) {
                    ::close(fd);
                    throw std::runtime_error("ChunkedKnnAccumulator: header/ftruncate failed for " + path);
                }
                return fd;
            };
            fd_knn = open_out(final_knn_path);
            fd_dist = open_out(final_dist_path);
        }
        auto pwrite_all = [](int fd, const void* buf, size_t n, off_t off) {
            const char* p = static_cast<const char*>(buf);
            while (n > 0) {
                ssize_t w = ::pwrite(fd, p, n, off);
                if (w <= 0) return false;
                p += w; n -= static_cast<size_t>(w); off += w;
            }
            return true;
        };

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

                if (is_final) {
                    // chunk c 在最终文件里是第 [chunk_start, chunk_start+chunk_c_size) 行
                    const off_t ofs = hdr + static_cast<off_t>(chunk_start) * M * 4;
                    if (!pwrite_all(fd_knn, run_nbrs.data(), run_nbrs.size() * sizeof(int32_t), ofs) ||
                        !pwrite_all(fd_dist, run_dists.data(), run_dists.size() * sizeof(float), ofs))
                        throw std::runtime_error("ChunkedKnnAccumulator: final pwrite failed for chunk " + std::to_string(c));
                    if (iter > 0) { std::filesystem::remove(np); std::filesystem::remove(dp); }
                } else {
                    std::ofstream nout(np, std::ios::binary | std::ios::trunc);
                    std::ofstream dout(dp, std::ios::binary | std::ios::trunc);
                    nout.write(reinterpret_cast<const char*>(run_nbrs.data()),
                              static_cast<std::streamsize>(run_nbrs.size() * sizeof(int32_t)));
                    dout.write(reinterpret_cast<const char*>(run_dists.data()),
                              static_cast<std::streamsize>(run_dists.size() * sizeof(float)));
                    if (!nout.good() || !dout.good())
                        throw std::runtime_error("ChunkedKnnAccumulator: write failed for running chunk " + std::to_string(c));
                    nout.close(); dout.close();
                }
                auto t5 = now();
                tm.write_running = secs_between(t4, t5);
                tm.total = secs_between(t0, t5);
                tm.end_ms = secs_between(t_merge_start, t5) * 1000.0;
            }
        }

        if (is_final) {
            if (::close(fd_knn) != 0 || ::close(fd_dist) != 0)
                throw std::runtime_error("ChunkedKnnAccumulator: close failed for final outputs");
            final_written = true;
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
        if (final_written) {   // 最后一轮 merge 已经直接写好了
            if (knn_path != final_knn_path || dist_path != final_dist_path)
                throw std::runtime_error("ChunkedKnnAccumulator: finalize path differs from set_final_output");
            return;
        }
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
 * @param N                   数据点总数
 * @param D                   向量维度
 * @param bucket_vecs         每个 bucket 的 (gid, 原始向量) 磁盘缓存 (Step 4 写好的，
 *                            见文件顶部 BucketVectorAccumulator 的说明)；bucket/pool
 *                            的原始向量直接从这里顺序读，不需要整份数据集常驻内存
 * @param centroid_knn_graph  (n_centroids, K) centroid KNN 图，nprobe 展开后的、Step 6
 *                            实际搜索用的近邻集合 (CPU, uint32)——决定每个 bucket 真正
 *                            会读哪些近邻 bucket，也是 Belady 缓存 future-access 预测
 *                            必须严格对应的那个访问模式
 * @param n_centroids         centroid / bucket 数量
 * @param K                   centroid_knn_graph 的度数 (= nprobe 展开后的近邻数，可能
 *                            远大于 Step 3 原始 CAGRA 图度数)
 * @param M                   每个向量要找的邻居数
 *
 * @param acc                 分块的 disk-backed 累积器 (见文件顶部 ChunkedKnnAccumulator
 *                            的说明)。每算完一个 bucket，就把该 bucket 里每个点新算出
 *                            的 M 个候选，按 gid 路由进对应 chunk 的内存 write buffer
 *                            (纯追加，不读旧值)；真正的跨 iteration 合并延后到调用方
 *                            在每轮结束后调 acc.merge_iteration() 时才做。
 * @param order_graph         Step 3 原始 (未 nprobe 展开的) centroid KNN 图，只用来算
 *                            桶处理顺序——compute_bucket_processing_order 是 O(B*K'^2)
 *                            的，K' 用 nprobe 展开后的度数 (可能到 250+) 会直接爆炸，
 *                            用原始、小得多的图度数 (order_graph_K) 求序足够了：顺序只
 *                            要能让"近邻集合重叠多的桶排在一起"这个大方向对，不需要
 *                            跟真实访问的近邻集合完全一致 (那是 future_access 的职责，
 *                            用的是上面 centroid_knn_graph/K 那个真实展开后的图)
 * @param order_graph_K       order_graph 的度数 (= Step 3 的 knn-k)
 * @param order_window_arg    DiskJoin 风格桶处理顺序的滑动窗口大小 (见 bucket_order.hpp)，
 *                            0 = 自动 (4*order_graph_K)
 * @param cache_bytes         Belady 桶缓存的字节预算——按处理顺序把"桶自己 + K 个
 *                            近邻桶"的原始向量缓存住，命中就不用再碰磁盘
 *
 * 距离对合并是必需的 (要按距离排序去重)，所以内部始终按 want_distances=true
 * 的路径跑；不再对外暴露"不算距离"这个选项。
 *
 * bucket_vecs 的类型是模板参数 BucketReaderT 而不是写死 BucketVectorAccumulator<DataT>——
 * 函数体只通过 count(bid) / read_bucket(bid, ids_out, vecs_out) 这两个方法访问它
 * (见下面 bucket_loader)，round 2..t 传 BucketVectorAccumulator<DataT>&（跟以前一样），
 * round 1 传 ReorderedDatasetBucketReader<DataT>&（见 reordered_dataset_io.hpp）——
 * round 1 分配完就已经把数据集重排写进 data_reordered.<ext> 了，直接按 offset 读，
 * 不需要 BucketVectorAccumulator 那份按桶散写的临时文件。
 */
template <typename DataT, typename BucketReaderT>
void build_vector_knn_with_tensorcore(
    int64_t N,
    int64_t D,
    BucketReaderT& bucket_vecs,
    const uint32_t* centroid_knn_graph,  // (n_centroids, K) row-major, nprobe 展开后
    int64_t n_centroids,
    uint32_t K,
    int M,
    ChunkedKnnAccumulator& acc,
    const uint32_t* order_graph,   // (n_centroids, order_graph_K) row-major, 原始 CAGRA 图
    uint32_t order_graph_K,
    int32_t order_window_arg,
    size_t cache_bytes,
    const std::vector<int32_t>* precomputed_order = nullptr)
{
    auto func_t0 = std::chrono::high_resolution_clock::now();
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
    // Step 0: bucket 大小表 (点的原始向量已经在 Step 4 里按 bucket 顺序落盘了，
    // 这里只需要知道每个 bucket 有多少点，不用再重建/常驻整份 id 列表)。
    // centroid 必在自己 bucket 里 (Step 4 已经保证)，不需要再检查/补写。
    // ================================================================
    std::vector<int64_t> bucket_count(n_centroids);
    for (int64_t c = 0; c < n_centroids; ++c) {
        bucket_count[c] = bucket_vecs.count(c);
    }
    double t_setup_bucket_count = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count();

    // ================================================================
    // Step 0.5: DiskJoin 风格的桶处理顺序 + Belady 最优离线缓存 (见
    // bucket_order.hpp，跟 bucket_build.cu Step 4 用的是同一套机制)。
    //
    // 主循环按 c 的自然顺序遍历时，同一个桶平均会被 ~K 个不同的查询桶当作
    // "近邻"重复读——处理顺序如果是随便的 0,1,2,...，相邻处理的桶之间的
    // 近邻集合基本不重叠，没法靠缓存复用。compute_bucket_processing_order
    // 贪心地把"近邻集合重叠多"的桶排在一起处理，配合下面按这个顺序算出的
    // "每个桶未来会在哪些位置被用到"，BeladyBucketCache 就能做到（给定这个
    // 顺序时）理论最优的缓存命中率——大部分近邻桶的读会直接命中缓存，不用
    // 再碰磁盘 (哪怕现在磁盘读本身已经很便宜，见 BucketVectorAccumulator 的
    // 单文件+持久句柄设计，命中缓存还是比 seekg+read 更快，且完全不占用
    // 磁盘带宽)。
    //
    // 算 order 用的是原始 (未 nprobe 展开) 的小度数图 order_graph/order_graph_K，
    // 不是 centroid_knn_graph/K 那个真实展开后的大度数图——
    // compute_bucket_processing_order 是 O(B*度数^2) 的，nprobe 展开后度数可能
    // 到 250+，代进去会直接算不动；用原始度数 (通常 32 量级) 求出的顺序，"近邻
    // 集合重叠多的桶排一起"这个大方向依然成立，足够给缓存用。
    // future_access 则必须用 centroid_knn_graph/K (真实展开后的图)，因为它要
    // 精确预测"每个桶未来会在哪些位置被访问"，跟主循环里 resolve_bucket 实际
    // 读取的近邻集合必须一致，否则 Belady 缓存的淘汰决策会跟真实访问模式脱节。
    int32_t order_window = (order_window_arg > 0)
        ? order_window_arg
        : std::max<int32_t>(4 * static_cast<int32_t>(order_graph_K), 16);
    // round1 调用方 (iter==0) 已经用完全相同的 centroid_knn_graph/K/order_window
    // 在 Step 4.5 里算过一遍 bucket_process_order 了 (结果必然逐位相同，输入
    // 一样)，这时候 caller 会把它传进来，这里直接复用，连 order_adj 都不用建。
    std::vector<int32_t> bucket_order_seq = precomputed_order
        ? *precomputed_order
        : bucket_order::compute_bucket_processing_order(
              bucket_order::adjacency_from_flat_graph(
                  order_graph, n_centroids, static_cast<int32_t>(order_graph_K)),
              order_window);
    double t_setup_order = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count() - t_setup_bucket_count;

    auto access_adj = bucket_order::adjacency_from_flat_graph(
        centroid_knn_graph, n_centroids, static_cast<int32_t>(K));
    auto future_access = bucket_order::compute_future_access_lists(bucket_order_seq, access_adj);
    double t_setup_future_access = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count() - t_setup_bucket_count - t_setup_order;

    auto vec_bucket_bytes = [](const VecBucket<DataT>& b) {
        return b.ids.size() * sizeof(int64_t) + b.vecs.size() * sizeof(DataT);
    };
    bucket_order::BeladyBucketCache<VecBucket<DataT>> bucket_cache(
        std::move(future_access), cache_bytes, vec_bucket_bytes);
    auto bucket_loader = [&](int32_t bid) {
        VecBucket<DataT> vb;
        bucket_vecs.read_bucket(bid, vb.ids, vb.vecs);
        return vb;
    };
    std::cout << "  [VectorKNN] bucket order_window=" << order_window
              << ", cache_budget=" << (cache_bytes / 1e6) << " MB\n";
    double t_setup_cache_ctor = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count()
        - t_setup_bucket_count - t_setup_order - t_setup_future_access;

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
    //   2) bucket/pool 的原始向量从磁盘上"按 bucket 分组"的文件顺序整块读取
    //      (Step 4 写的，见 BucketVectorAccumulator)，不需要整份数据集常驻
    //      CPU/GPU，也不需要 GPU kernel 按 id gather——数据已经是需要的行。
    //   3) ‖x‖² 按 bucket 现算 (从刚读进来的 pool 数据算，不再有全局常驻表)
    //   4) Double-buffer + 双 stream: bucket c 的 GEMM/topM/D2H 与 bucket c+1 的
    //      CPU prep + H2D 并行；CPU 端的磁盘读取与 GPU 工作完全重叠。
    size_t max_bucket_size = 0;
    size_t max_pool_size_ub = 0;  // 上界(未去重)
    for (int64_t c = 0; c < n_centroids; ++c) {
        max_bucket_size = std::max<size_t>(max_bucket_size, static_cast<size_t>(bucket_count[c]));
        size_t ps = static_cast<size_t>(bucket_count[c]);
        for (uint32_t k = 0; k < K; ++k) {
            uint32_t nb_c = centroid_knn_graph[c * K + k];
            if (nb_c < static_cast<uint32_t>(n_centroids)) {
                ps += static_cast<size_t>(bucket_count[nb_c]);
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

    // ---- 显存预算: 不再有整份数据集常驻的全局 buffer，只有 per-slot ----
    // d_B_raw 存原始 DataT (从磁盘/缓存读来的 pool 数据, 未转型)；d_B 是转型/
    // 位移之后喂给 cuBLAS 的 GemmInT 视图。A (bucket 自己) 不再单独占一份
    // buffer——pool 的前 bucket_size 行本来就是 A 的数据 (resolve_bucket 总是
    // 先 resolve 自己)，GEMM 直接拿 d_B 的前缀当 A 操作数即可，省掉一份重复的
    // CPU memcpy + H2D 传输 + gather/shift kernel。
    size_t bytes_B_raw        = max_pool_size_ub * D * sizeof(DataT);
    size_t bytes_B            = max_pool_size_ub * D * sizeof(GemmInT);
    size_t bytes_dots         = max_bucket_size * max_pool_size_ub * sizeof(GemmOutT);
    size_t bytes_norms_pool   = max_pool_size_ub * sizeof(float);
    size_t bytes_ids_bucket   = max_bucket_size * sizeof(int32_t);
    size_t bytes_ids_pool     = max_pool_size_ub * sizeof(int32_t);
    size_t bytes_out          = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    // 距离输出 buffer 仅在 want_distances 时分配 (multi-iteration merge 需要)
    size_t bytes_out_dists    = want_distances ? max_bucket_size * static_cast<size_t>(M) * sizeof(float) : 0;

    // identity index (0,1,2,...) 全局共用一份，只跟 max_pool_size_ub 相关
    size_t bytes_identity_idx = max_pool_size_ub * sizeof(int32_t);

    size_t bytes_global   = bytes_identity_idx;
    size_t bytes_per_slot = bytes_B_raw + bytes_B + bytes_dots + bytes_norms_pool
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

    std::cout << "  [VectorKNN] buffer plan: dots(max)=" << bytes_dots/1e9 << "GB, "
              << "per-slot=" << bytes_per_slot/1e9 << "GB × " << num_slots
              << " + global=" << bytes_global/1e9 << "GB / usable="
              << usable_bytes/1e9 << "GB"
              << " [streams=" << num_slots
              << ", num_sm=" << num_sm
              << ", avg_bucket=" << avg_bucket_size
              << ", est_blocks/kernel=" << est_blocks_per_kernel
              << ", target_streams(pre-mem)=" << target_streams << "]\n";

    // ---- 全局常驻 GPU buffer: 只有 identity index (0,1,2,...)，跟数据集大小
    // 无关，只跟 max_pool_size_ub 有关——用来把"刚从磁盘读上来、已经是需要的
    // 那些行、且已经按 bucket+pool 顺序紧密排列"的原始数据，直接喂给原有的
    // gather_rows_raw/gather_rows_int32 kernel 做类型转换 (identity 索引 =
    // 不做任何重排，只做逐元素转型/位移)，复用现成 kernel，不用另写转型 kernel。
    // ----
    int32_t* d_identity_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_identity_idx, bytes_identity_idx));
    {
        std::vector<int32_t> h_identity(max_pool_size_ub);
        std::iota(h_identity.begin(), h_identity.end(), 0);
        CUDA_CHECK(cudaMemcpy(d_identity_idx, h_identity.data(), bytes_identity_idx,
                              cudaMemcpyHostToDevice));
    }

    // ---- per-slot 持久化 buffer (最多 8 slot 轮转, 实际用 num_slots 个) ----
    // d_B/d_dots 类型由 GemmInT/GemmOutT 决定（INT8 路径 / FP32 路径不同）
    // C++ partial init: {nullptr, nullptr} 后面的 slot 会被零初始化为 nullptr
    // A (bucket 自己) 没有独立 buffer：GEMM 直接用 d_B 的前 bucket_size 行当
    // A 操作数 (resolve_bucket 总是先 resolve 自己，pool 前缀天然就是 A 的数据)。
    DataT*       d_B_raw[MAX_SLOTS]         = {nullptr};
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
    DataT*       h_B_raw[MAX_SLOTS]         = {nullptr};
    int64_t*     h_ids_bucket64[MAX_SLOTS]  = {nullptr};  // bucket_vecs 读出来的 gid (int64)
    int64_t*     h_ids_pool64[MAX_SLOTS]    = {nullptr};
    int32_t*     h_out[MAX_SLOTS]           = {nullptr};
    float*       h_out_dists[MAX_SLOTS]     = {nullptr};  // 仅 want_distances 时使用
    cudaStream_t streams[MAX_SLOTS]         = {nullptr};

    // RAFT path 的额外 buffer 大小：(max_bucket_size × M) × 4 bytes，K=1024 时 ~4MB/slot，可忽略
    size_t bytes_select_idx  = max_bucket_size * static_cast<size_t>(M) * sizeof(int32_t);
    size_t bytes_select_dist = max_bucket_size * static_cast<size_t>(M) * sizeof(float);

    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaMalloc(&d_B_raw[s],         bytes_B_raw));
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
        CUDA_CHECK(cudaMallocHost(&h_B_raw[s],      bytes_B_raw));
        h_ids_bucket64[s] = new int64_t[max_bucket_size];
        h_ids_pool64[s]   = new int64_t[max_pool_size_ub];
        CUDA_CHECK(cudaMallocHost(&h_out[s],        bytes_out));
        if (want_distances) {
            CUDA_CHECK(cudaMalloc(&d_out_dists[s],  bytes_out_dists));
            CUDA_CHECK(cudaMallocHost(&h_out_dists[s], bytes_out_dists));
        }
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
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

    // pool 拼装 (3b, 生产者线程) 的 resolve/memcpy 两趟之间传递的 job 描述：
    // resolve 趟顺序调 bucket_cache.get()，只记源指针/大小/目标偏移；memcpy
    // 趟再用 OpenMP 并行做实际搬运。
    struct CopyJob {
        const int64_t* src_ids;
        const DataT*    src_vecs;
        size_t count;
        size_t dest_ofs;
    };

    auto scatter_pending = [&](int slot) {
        if (pending[slot].c < 0) return;
        // h_ids_bucket[slot] 这时候还是"上一次派到这个 slot 的那个 bucket"的
        // gid 数组 (本次循环要到下面 3b 才会覆写它)，直接用，不用额外记录。
        int bs = pending[slot].bucket_size;
        int aM = pending[slot].actual_M;
        const int32_t* hp = h_out[slot];
        const float*   hd = h_out_dists[slot];  // 始终已分配 (want_distances 内部恒为 true)
        for (int i = 0; i < bs; ++i) {
            int32_t gid = h_ids_bucket[slot][i];
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
    double t_setup_before_loop = std::chrono::duration<double>(loop_t0 - func_t0).count();
    double t_setup_gpu_alloc = t_setup_before_loop
        - t_setup_bucket_count - t_setup_order - t_setup_future_access - t_setup_cache_ctor;
    std::cout << "  [VectorKNN] setup before bucket loop: " << std::fixed << std::setprecision(3)
              << t_setup_before_loop << "s (bucket_count=" << t_setup_bucket_count
              << "s, order=" << t_setup_order
              << "s, future_access=" << t_setup_future_access
              << "s, cache_ctor=" << t_setup_cache_ctor
              << "s, gpu_buffer_alloc=" << t_setup_gpu_alloc << "s)\n";

    // ---- TEMP DIAGNOSTIC: fine-grained bucket-loop breakdown ----
    // pure GEMM is already timed via cudaEvent above; this splits the
    // remaining ~88-91% of loop wall-time into: stream-sync wait,
    // scatter_pending (CPU merge of previous slot's results into acc),
    // CPU-side pool_ids assembly (up to K memcpy's per bucket), and GPU
    // submission (H2D + gather kernels + GEMM launch + select_k launch +
    // D2H enqueue). Remove once we know where the time actually goes.
    auto diag_now = [] { return std::chrono::high_resolution_clock::now(); };
    auto diag_secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    double t_sync = 0, t_scatter = 0, t_poolbuild = 0, t_submit = 0;
    // pool_build 内部再拆一层: cache.get() (命中时的记账 + 没命中时真正调
    // loader 读盘) vs. 从 cache 命中的数据往 pool buffer 里 memcpy 的时间。
    // 想确认这两版优化之后剩下的 pool_build 时间到底是"访问 cache 本身"还是
    // "纯数据搬运量大"——先测出来，不再靠估算。跑完可以再删。
    double t_cache_get = 0, t_pool_memcpy = 0;

    // ---- 3b 拆到生产者线程 ----
    // 原来每个 pos 串行做: sync → scatter (上一轮结果) → pool_build (cache.get
    // + memcpy, ~0.24 ms/bucket) → submit。pool_build 只依赖 bucket 顺序，跟
    // scatter/submit 无关，所以交给一个独占 bucket_cache 的生产者线程提前做
    // (cache 仍然只被一个线程访问, Belady 访问顺序不变)，主线程只做 sync /
    // scatter / submit —— 跟 cuVS 26.12 cagra_build.cuh::build_knn_graph 里
    // "后台线程准备下一批、前台提交当前批" 的 double buffer 同一思路。
    // 暂存区就是原来的 per-slot pinned buffer (h_ids_pool64/h_B_raw/h_ids_pool)，
    // 生产者最多领先 num_slots 个 pos：写 slot 之前必须等主线程已经提交过
    // 这个 slot 上一次的 H2D (slot_ready_pos[slot] < 0)，再等那次 H2D 真正
    // 完成 (ev_h2d[slot])。bucket 自己的 gid 先放 staged_ids_bucket[slot]，主
    // 线程在 scatter 完旧结果之后才拷进 h_ids_bucket[slot] (scatter 还要读旧值)。
    struct Staged { int pool_size = 0, bucket_size = 0, actual_M = 0; };
    Staged staged[MAX_SLOTS]{};
    std::vector<int32_t> staged_ids_bucket[MAX_SLOTS];
    cudaEvent_t ev_h2d[MAX_SLOTS] = {nullptr};
    bool h2d_pending[MAX_SLOTS] = {false};          // guarded by stage_mtx
    int32_t slot_ready_pos[MAX_SLOTS];              // guarded by stage_mtx; -1 = 空
    for (int s = 0; s < num_slots; ++s) {
        staged_ids_bucket[s].resize(max_bucket_size);
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d[s], cudaEventDisableTiming));
        slot_ready_pos[s] = -1;
    }
    std::mutex stage_mtx;
    std::condition_variable stage_cv;
    bool stage_abort = false, prod_failed = false;  // guarded by stage_mtx
    std::exception_ptr prod_exc;
    double t_wait_producer = 0;

    auto producer = [&]() {
        try {
            std::vector<CopyJob> jobs;
            jobs.reserve(static_cast<size_t>(K) + 1);
            for (int32_t pos = 0; pos < static_cast<int32_t>(n_centroids); ++pos) {
                int64_t c = bucket_order_seq[pos];
                if (bucket_count[c] == 0) continue;
                int slot = static_cast<int>(pos % num_slots);
                bool wait_h2d = false;
                {
                    std::unique_lock<std::mutex> lk(stage_mtx);
                    stage_cv.wait(lk, [&] { return slot_ready_pos[slot] < 0 || stage_abort; });
                    if (stage_abort) return;
                    wait_h2d = h2d_pending[slot];
                    h2d_pending[slot] = false;
                }
                if (wait_h2d) CUDA_CHECK(cudaEventSynchronize(ev_h2d[slot]));
                auto pb0 = diag_now();

                // Step 4 已经保证每个非 centroid 点只属于一个 bucket、centroid 必在
                // 自己 bucket 里 → 各 bucket 互不相交，拼出来的 pool 不会重复。
                // 先顺序 get() (cache 非线程安全；同一 pos 内的桶彼此不会互相淘汰)，
                // 只记源指针，再并行 memcpy (见 #6: 线程数封顶 8)。
                int64_t* ids_pool64 = h_ids_pool64[slot];
                DataT*   vecs_pool  = h_B_raw[slot];
                size_t ofs = 0;  // 以"点数"计，不是字节
                jobs.clear();
                auto resolve_bucket = [&](int64_t bid) {
                    if (bucket_count[bid] == 0) return;
                    auto ga0 = diag_now();
                    const VecBucket<DataT>& vb = bucket_cache.get(pos, static_cast<int32_t>(bid), bucket_loader);
                    t_cache_get += diag_secs(ga0, diag_now());
                    size_t cnt = vb.ids.size();
                    if (ofs + cnt > max_pool_size_ub)   // bounds check: pool buffer 容量
                        throw std::runtime_error("VectorKNN: pool overflow at pos " + std::to_string(pos));
                    jobs.push_back(CopyJob{vb.ids.data(), vb.vecs.data(), cnt, ofs});
                    ofs += cnt;
                };
                resolve_bucket(c);
                for (uint32_t k = 0; k < K; ++k) {
                    uint32_t nb_c = centroid_knn_graph[c * K + k];
                    if (nb_c < static_cast<uint32_t>(n_centroids)) {
                        resolve_bucket(static_cast<int64_t>(nb_c));
                    }
                }
                {
                    auto gm0 = diag_now();
                    // 线程数封顶 8：每个 pool 只有 ~K+1 个小 job (SIFT1M 每个 ~100 行
                    // ≈ 13 KB)，80 线程 fork/join + 跨 NUMA 抢带宽反而更慢。实测
                    // (SIFT1M --out-of-core, 每轮 pool_memcpy): 80 线程 2.1–2.9s，
                    // 32: 1.6s, 16: 1.2–1.6s, 8: 1.3–1.6s, 4: 1.7s, 1: 5.3s。
                    static const int pool_copy_threads = std::min(8, omp_get_max_threads());
                    #pragma omp parallel for schedule(dynamic) num_threads(pool_copy_threads)
                    for (int64_t j = 0; j < static_cast<int64_t>(jobs.size()); ++j) {
                        const CopyJob& job = jobs[static_cast<size_t>(j)];
                        std::memcpy(ids_pool64 + job.dest_ofs, job.src_ids, job.count * sizeof(int64_t));
                        std::memcpy(vecs_pool + job.dest_ofs * D, job.src_vecs,
                                    job.count * static_cast<size_t>(D) * sizeof(DataT));
                    }
                    t_pool_memcpy += diag_secs(gm0, diag_now());
                }
                Staged& st = staged[slot];
                st.pool_size   = static_cast<int>(ofs);
                st.bucket_size = static_cast<int>(bucket_count[c]);
                st.actual_M    = std::min(M, st.pool_size - 1);
                if (st.actual_M > 0) {
                    // bucket 自己的这部分 (pool 最前面 bucket_size 条) 就是 A；gid 从
                    // int64 truncate 成 int32 (沿用项目里 N<2^31 的既有假设)。
                    for (int i = 0; i < st.bucket_size; ++i)
                        staged_ids_bucket[slot][static_cast<size_t>(i)] = static_cast<int32_t>(ids_pool64[i]);
                    for (int i = 0; i < st.pool_size; ++i)
                        h_ids_pool[slot][i] = static_cast<int32_t>(ids_pool64[i]);
                }
                t_poolbuild += diag_secs(pb0, diag_now());
                {
                    std::lock_guard<std::mutex> lk(stage_mtx);
                    slot_ready_pos[slot] = pos;
                }
                stage_cv.notify_all();
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lk(stage_mtx);
                prod_exc = std::current_exception();
                prod_failed = true;
            }
            stage_cv.notify_all();
        }
    };
    std::thread prod_thread(producer);
    struct ProducerJoin {
        std::thread& t; std::mutex& m; std::condition_variable& cv; bool& abort;
        ~ProducerJoin() {
            { std::lock_guard<std::mutex> lk(m); abort = true; }
            cv.notify_all();
            if (t.joinable()) t.join();
        }
    } prod_join{prod_thread, stage_mtx, stage_cv, stage_abort};

    for (int32_t pos = 0; pos < static_cast<int32_t>(n_centroids); ++pos) {
        int64_t c = bucket_order_seq[pos];
        if (bucket_count[c] == 0) continue;

        // slot 必须按遍历位置 pos 轮转 (不是按桶 id c)——c 现在是按 DiskJoin
        // 顺序跳着走的，只有 pos 才是真正连续递增的循环步数，slot 轮转要靠它
        // 才能维持"这一 slot 的上一次提交已经隔了 num_slots 步"这个 double-
        // buffer 假设。
        int slot = static_cast<int>(pos % num_slots);

        // ---- 3a: 等本 slot 上一轮 D2H 落地，scatter 老结果，腾出 buffer ----
        auto dt0 = diag_now();
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
        auto dt1 = diag_now();
        t_sync += diag_secs(dt0, dt1);
        consume_gemm_timing(slot);
        scatter_pending(slot);
        auto dt2 = diag_now();
        t_scatter += diag_secs(dt1, dt2);

        // ---- 3b: 取生产者已经拼好的 pool ----
        {
            std::unique_lock<std::mutex> lk(stage_mtx);
            stage_cv.wait(lk, [&] { return slot_ready_pos[slot] == pos || prod_failed; });
            if (slot_ready_pos[slot] != pos) std::rethrow_exception(prod_exc);
        }
        const Staged st = staged[slot];
        int pool_size   = st.pool_size;
        int bucket_size = st.bucket_size;
        int actual_M    = st.actual_M;
        if (actual_M <= 0) {
            {
                std::lock_guard<std::mutex> lk(stage_mtx);
                slot_ready_pos[slot] = -1;
            }
            stage_cv.notify_all();
            t_wait_producer += diag_secs(dt2, diag_now());
            continue;
        }
        std::memcpy(h_ids_bucket[slot], staged_ids_bucket[slot].data(),
                    static_cast<size_t>(bucket_size) * sizeof(int32_t));
        auto dt3 = diag_now();
        t_wait_producer += diag_secs(dt2, dt3);

        // ---- 3c: 上传 id 列表 + 原始向量 (async on slot stream) ----
        CUDA_CHECK(cudaMemcpyAsync(d_ids_bucket[slot], h_ids_bucket[slot],
                                   bucket_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(d_ids_pool[slot], h_ids_pool[slot],
                                   pool_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, streams[slot]));
        CUDA_CHECK(cudaMemcpyAsync(d_B_raw[slot], h_B_raw[slot],
                                   static_cast<size_t>(pool_size) * D * sizeof(DataT),
                                   cudaMemcpyHostToDevice, streams[slot]));
        // 生产者在 ev_h2d[slot] 完成之后才能再写这个 slot 的 pinned 暂存区
        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot]));
        {
            std::lock_guard<std::mutex> lk(stage_mtx);
            h2d_pending[slot] = true;
            slot_ready_pos[slot] = -1;
        }
        stage_cv.notify_all();

        // ---- 3d: 转型/位移 B (identity 索引，不做重排只做逐元素转型) + 现算 norms_pool ----
        // 数据已经是磁盘/缓存按 bucket 顺序读出来的、恰好需要的那些行，不需要
        // 再按 id gather 一遍——复用原有 kernel，只是把"按 gid 查表"换成
        // "identity 索引"。A 不再单独转型：d_B 转好之后，它的前 bucket_size
        // 行本来就是转型/位移后的 A。
        {
            int threads = 256;
            int64_t total_B  = static_cast<int64_t>(pool_size) * D;
            int64_t blocks_B = (total_B + threads - 1) / threads;

            if constexpr (kIsInt8Path) {
                gather_rows_raw<DataT><<<blocks_B, threads, 0, streams[slot]>>>(
                    d_B_raw[slot], d_identity_idx,
                    reinterpret_cast<DataT*>(d_B[slot]), pool_size, D);
                // uint8 → int8 in-place 平移 (仅 uint8 路径；int8 数据本来就对)
                if constexpr (std::is_same<DataT, uint8_t>::value) {
                    shift_uint8_to_int8_inplace<<<blocks_B, threads, 0, streams[slot]>>>(
                        reinterpret_cast<uint8_t*>(d_B[slot]), total_B);
                }
            } else {
                gather_rows_int32<DataT><<<blocks_B, threads, 0, streams[slot]>>>(
                    d_B_raw[slot], d_identity_idx,
                    reinterpret_cast<float*>(d_B[slot]), pool_size, D);
            }
            CUDA_CHECK(cudaGetLastError());

            // norms 必须从 GEMM 实际看到的 (转型/位移之后的) 数据视图算，直接
            // 对刚转好的 d_B[slot] 现算，不再有全局常驻的 norms 表可查。
            int64_t blocks_n = (pool_size + threads - 1) / threads;
            compute_row_norms_kernel<GemmInT><<<blocks_n, threads, 0, streams[slot]>>>(
                d_B[slot], d_norms_pool[slot], pool_size, D);
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
                // A 操作数直接传 d_B[slot] 本身: A 是 pool 的前 bucket_size
                // 行，跟 B 是同一份已转型数据，lda=D 不变，cuBLAS 只读前
                // bucket_size 行，不需要一份独立的 A buffer。
                stat = cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, bucket_size, D,
                    &alpha_i,
                    d_B[slot], CUDA_R_8I, D,
                    d_B[slot], CUDA_R_8I, D,
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
                    reinterpret_cast<const float*>(d_B[slot]), D,
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

        t_submit += diag_secs(dt3, diag_now());

        pending[slot] = {c, bucket_size, actual_M};
        ++processed_buckets;
    }

    prod_thread.join();   // 已经全部消费完；ProducerJoin 析构时 joinable() 为 false

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
    {
        double loop_s = loop_ms / 1000.0;
        double sum_s = t_sync + t_scatter + t_wait_producer + t_submit;
        std::cout << "  [VectorKNN] bucket loop breakdown: stream_sync=" << t_sync
                  << "s scatter_pending=" << t_scatter
                  << "s wait_producer=" << t_wait_producer
                  << "s gpu_submit=" << t_submit
                  << "s (sum=" << sum_s << "s, loop_wall=" << loop_s << "s)\n";
        std::cout << "  [VectorKNN] pool_build breakdown: cache_get="
                  << t_cache_get << "s pool_memcpy=" << t_pool_memcpy
                  << "s (sum=" << (t_cache_get + t_pool_memcpy)
                  << "s vs pool_build=" << t_poolbuild << "s, on the producer thread)\n";
    }

    // ---- 释放 ----
    cudaFree(d_identity_idx);
    for (int s = 0; s < num_slots; ++s) {
        cudaFree(d_B_raw[s]);
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
        cudaFreeHost(h_B_raw[s]);
        delete[] h_ids_bucket64[s];
        delete[] h_ids_pool64[s];
        cudaFreeHost(h_out[s]);
        if (h_out_dists[s])  cudaFreeHost(h_out_dists[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(ev_h2d[s]);
        cudaEventDestroy(ev_gemm_start[s]);
        cudaEventDestroy(ev_gemm_stop[s]);
    }
    cublasDestroy(cublas_handle);

    double func_total = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count();
    double t_cleanup = func_total - t_setup_before_loop - (loop_ms / 1000.0);
    std::cout << "  [VectorKNN] function total: " << std::fixed << std::setprecision(3) << func_total
              << "s (setup=" << t_setup_before_loop
              << "s, bucket_loop=" << (loop_ms / 1000.0)
              << "s, cleanup=" << t_cleanup << "s)\n";
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
 *   inverse_perm[new_id] = old_id
 *   bucket_offsets       = bucket 边界（新 ID 空间，按 bucket_order 排列），长度 n_buckets+1
 *   centroid_new_ids[k]  = centroid_global_indices[k] 的新 ID
 *
 * 没有 perm[old_id]=new_id 这张全量表——它从来没被任何下游读过（inverse_perm
 * 才是 search 阶段真正需要的方向：新 ID 翻回旧 ID），而且信息上是 inverse_perm
 * 的完全冗余（谁真需要 old->new，反着扫一遍 inverse_perm 就有），没必要为了一
 * 个从没用过的方向多背 N 个 uint32。调用方唯一需要的 old->new 查询只有
 * centroid_global_indices 那几个具体的点（n_centroids 个，远小于 N），所以在
 * 下面这个本来就要跑一遍的 O(N) 循环里，顺手用一个只有 n_centroids 个条目的
 * hash map 把它们摘出来，不必再为全量 old->new 分配一个 O(N) 数组。
 */
struct ReorderInfo {
    std::vector<uint32_t> inverse_perm;
    std::vector<uint32_t> bucket_offsets;
    std::vector<uint32_t> centroid_new_ids;  // 跟 centroid_global_indices 一一对应
    int64_t total_in_buckets;
};

static ReorderInfo compute_bucket_reorder(
    const std::vector<int64_t>& assignments, int64_t N, int64_t n_buckets,
    const std::vector<int32_t>& bucket_order,
    const std::vector<int64_t>& centroid_global_indices)
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
    r.inverse_perm.assign(r.total_in_buckets, 0);
    r.centroid_new_ids.assign(centroid_global_indices.size(), 0xFFFFFFFFu);

    // old_id -> index into centroid_global_indices；只有 n_centroids 个条目，
    // 不是 N 个——这就是不再需要全量 perm[] 数组的关键。
    std::unordered_map<int64_t, size_t> centroid_old_id_to_k;
    centroid_old_id_to_k.reserve(centroid_global_indices.size() * 2);
    for (size_t k = 0; k < centroid_global_indices.size(); ++k)
        centroid_old_id_to_k[centroid_global_indices[k]] = k;

    std::vector<uint32_t> cursor = new_id_range_start;
    for (int64_t old_id = 0; old_id < N; ++old_id) {
        int64_t b = assignments[old_id];
        if (b >= 0 && b < n_buckets) {
            uint32_t new_id = cursor[b]++;
            r.inverse_perm[new_id] = static_cast<uint32_t>(old_id);
            auto it = centroid_old_id_to_k.find(old_id);
            if (it != centroid_old_id_to_k.end())
                r.centroid_new_ids[it->second] = new_id;
        }
    }
    return r;
}

// write_reordered_outputs（事后重排，读 bucket_vecs 的按桶散写临时文件，重映射
// vector_knn 邻居 ID）已经删掉了——它做的事现在在主循环 round 1 的 Step 4.5
// 里内联完成（见 ReorderedDatasetWriter / compute_bucket_reorder 的调用处），
// 不再需要事后单独一遍。

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
    int iterations,
    LoadConfig config,
    const std::string& ext,
    int32_t order_window_arg,
    size_t cache_mb_arg,
    bool force_out_of_core)
{
    try {
        init_gpu_limit_if_needed(config);

        // Step 6 的 Belady 桶缓存预算：0 = 自动，退化成 cpu-limit 的一半
        // (跟 bucket_build.cu 的 --cache-mb 是同一个约定)。
        size_t step6_cache_bytes = (cache_mb_arg > 0)
            ? cache_mb_arg * 1024ULL * 1024ULL
            : config.cpu_limit_bytes / 2;

        // 在 output_root 下建子目录: k<knn_k>p<nprobe>m<neighbors_m>t<iterations>
        // knn_k = 图度数 K; nprobe = Step 6 邻居扩展数 (0 时回退到 knn_k);
        // iterations (t) 必须编码进目录名——它决定了 Step2-6 重复轮数/去重
        // 合并结果，跟 k/p/m 一样是影响 vector_knn.bin 内容的参数，否则不同
        // --iterations 的两次跑会覆盖同一个目录。
        uint32_t output_nprobe = (nprobe > 0) ? nprobe : config.knn_k;
        std::string output_dir = output_root
                               + "/k" + std::to_string(config.knn_k)
                               + "p" + std::to_string(output_nprobe)
                               + "m" + std::to_string(neighbors_m)
                               + "t" + std::to_string(iterations);
        std::filesystem::create_directories(output_dir);
        std::cout << "Output subdir: " << output_dir << "\n";

        using Clock = std::chrono::high_resolution_clock;
        auto t_total_start = Clock::now();
        double elapsed_step1 = 0, elapsed_step2 = 0, elapsed_step3 = 0;
        double elapsed_step3p5 = 0, elapsed_step4 = 0, elapsed_step45 = 0, elapsed_step5 = 0;
        double elapsed_step6 = 0;
        double elapsed_write_knn = 0;
        double elapsed_final_merge_wait = 0;

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
        int64_t working_N = 0;  // X_sampled 的行数；Step 2 等 lambda 外的代码要用

        // 采样逻辑抽成 lambda：round 1 重排完之后，如果 iterations > 1，要对着
        // 新坐标系的 data_reordered.<ext> 重新采一遍（否则第 2..t 轮选出来的
        // centroid 会是旧坐标系的 ID，去新数据集里找点就全错了）——用同一份
        // 逻辑读不同路径，不能各写一份容易漂移。N/D/mem_est 采样前后不变
        // （round 1 重排不改变点数/维度，只改变 ID 的物理排列），可以直接复用。
        auto sample_from = [&](const std::string& path) {
            if (!mem_est.fits_in_gpu) {
                // 只从 disk 读取采样行，不加载完整数据集
                working_N = mem_est.sampled_data_rows;
                sampled_indices = sample_without_replacement(N, working_N, config.seed);
                // 排序以实现顺序磁盘读取 (SSD/NVMe 友好)
                auto sorted_order = sampled_indices;
                std::sort(sorted_order.begin(), sorted_order.end());

                if (ext == ".fbin" || ext == ".bin") {
                    int32_t n_tmp, d_tmp;
                    load::read_fbin_sampled(path, sorted_order, X_sampled, n_tmp, d_tmp);
                } else {
                    // u8bin/ibin: fallback 到全量读取再采样
                    // TODO: 为其他格式实现 sampled reader
                    std::vector<float> X_full_tmp;
                    if (ext == ".u8bin" || ext == ".i8bin") {
                        int32_t n2, d2;
                        load::read_u8bin_to_f32(path, X_full_tmp, n2, d2);
                    } else {
                        std::vector<int32_t> itmp;
                        int32_t n2, d2;
                        load::read_ibin_i32(path, itmp, n2, d2);
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
                    load::read_fbin_f32(path, X_sampled, N, D);
                } else if (ext == ".u8bin" || ext == ".i8bin") {
                    load::read_u8bin_to_f32(path, X_sampled, N, D);
                } else {
                    std::vector<int32_t> tmp;
                    load::read_ibin_i32(path, tmp, N, D);
                    X_sampled.resize(tmp.size());
                    for (size_t i = 0; i < tmp.size(); ++i) X_sampled[i] = static_cast<float>(tmp[i]);
                }
                std::cout << "  Loaded full dataset: " << X_sampled.size() * sizeof(float) / 1e9 << " GB\n";
            }
        };

        sample_from(input_path);

        // 数据集 + 3 份 (N, M) running KNN 表都放得进显存时，走 GPU-resident 路径
        // (见 GpuResidentKnn 的说明)：Step 4 的 query、Step 6 的 pool、跨轮合并
        // 都不再经过 CPU/磁盘。raw_host 留到 round 1 重排写 data_reordered 时用。
        std::unique_ptr<GpuResidentKnn<DataT>> gres;
        std::vector<DataT> raw_host;
        if (neighbors_m > 0 && neighbors_m <= 256 && mem_est.fits_in_gpu && !force_out_of_core) {
            size_t need = GpuResidentKnn<DataT>::bytes_needed(N, D, neighbors_m);
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            if (need <= static_cast<size_t>(free_b * 0.6)) {
                int32_t n_tmp, d_tmp;
                load::read_bigann_raw_range<DataT>(input_path, 0, N, raw_host, n_tmp, d_tmp);
                gres = std::make_unique<GpuResidentKnn<DataT>>(raw_host.data(), N, D, neighbors_m);
                std::cout << "  [GpuResidentKNN] enabled: " << need / 1e9 << " GB resident (free "
                          << free_b / 1e9 << " GB)\n";
            } else {
                std::cout << "  [GpuResidentKNN] disabled: needs " << need / 1e9 << " GB, free "
                          << free_b / 1e9 << " GB -> out-of-core path\n";
            }
        }

        elapsed_step1 = std::chrono::duration<double>(Clock::now() - t1).count();
        std::cout << "  Step 1 done [" << std::fixed << std::setprecision(3) << elapsed_step1 << "s]\n";

        // input_path 参数是 const&，round 1 分配完之后要把它换成
        // data_reordered.<ext>——从这里往下（包括 Step4 的调用点）一律用这个
        // 可变的本地副本，不要再直接用 input_path 参数。round 2..t 全程读的
        // 就是这份重排后的文件。
        std::string working_input_path = input_path;

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
        if (neighbors_m > 0 && !gres) {
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

        // 每个 bucket 的 (gid, 原始向量) 磁盘缓存：Step 4 算完一个点的分配就
        // 顺手把它的原始向量写进这里 (见文件顶部 BucketVectorAccumulator 的
        // 说明)；Step 6 直接从这里按 bucket 顺序整块读，不需要整份数据集常驻
        // 内存/显存。n_centroids 每轮 Step 2 选完才知道 (不同轮的 KMeans||
        // 采样可能选出稍微不同的数量)，所以在循环内、Step 2 之后才 create()。
        std::string bucket_vec_dir = output_dir + "/bucket_vecs_tmp";
        std::filesystem::create_directories(bucket_vec_dir);
        std::unique_ptr<BucketVectorAccumulator<DataT>> bucket_vecs_ptr;

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
        double t2_select_call = std::chrono::duration<double>(Clock::now() - t2).count();

        // sample_idx → raw_idx
        centroid_global_indices.assign(centroid_sample_indices.size(), 0);
        for (size_t k = 0; k < centroid_sample_indices.size(); ++k) {
            centroid_global_indices[k] = sampled_indices[centroid_sample_indices[k]];
        }

        n_centroids = static_cast<int64_t>(centroid_global_indices.size());
        double t2_idx_map = std::chrono::duration<double>(Clock::now() - t2).count() - t2_select_call;

        // bucket 磁盘缓存跟着这一轮的 n_centroids 重建 (bufs 大小依赖它)——但
        // n_centroids/D 在实践中逐轮不变 (KMeans|| 候选数只要够用，最终选出的
        // 数量恒等于 config.centroid_ratio 定的目标值)，这时候整个对象重新
        // make_unique/create 纯粹是重新分配一遍跟上一轮内容无关的 ~1GB host
        // buffer，白白浪费。只有维度真的变了 (或者还没创建过) 才重新 create。
        // start_iteration() 现在需要知道这一轮每个 bucket 最终的记录数 (histogram)
        // 才能把它们在合并文件里的字节区间提前算好——这个数要等 Step 4 的
        // assignments[] 全部算完才知道，所以挪到 Step 4 里调用，这里只管 create。
        auto t2_bvc0 = Clock::now();
        if (!gres && (!bucket_vecs_ptr || bucket_vecs_ptr->n_centroids != n_centroids || bucket_vecs_ptr->D != D)) {
            bucket_vecs_ptr = std::make_unique<BucketVectorAccumulator<DataT>>(
                BucketVectorAccumulator<DataT>::create(
                    bucket_vec_dir, n_centroids, D, config.bucket_vec_buffer_bytes));
        }
        double t2_bucket_vec_create = std::chrono::duration<double>(Clock::now() - t2_bvc0).count();

        cudaDeviceSynchronize();
        double iter_step2 = std::chrono::duration<double>(Clock::now() - t2).count();
        elapsed_step2 += iter_step2;
        std::cout << "  [Step2 outer timing] select_call=" << std::fixed << std::setprecision(3) << t2_select_call
                  << "s idx_map=" << t2_idx_map
                  << "s bucket_vec_create=" << t2_bucket_vec_create
                  << "s total=" << iter_step2 << "s\n";
        std::cout << "  Step 2 done [" << std::fixed << std::setprecision(3) << iter_step2 << "s]\n";

        // ================================================================
        // Step 3: Build KNN graph on centroids — results stay on GPU
        // ================================================================
        std::cout << "=== Step 3: Building centroid KNN graph (GPU-resident) ===\n";
        auto t3 = Clock::now();
        uint32_t* d_graph = nullptr;
        K = config.knn_k;

        // 非 PQ: 精确 brute-force 自 KNN，一次取 max(K, nprobe) 列 (见
        // centroid_self_knn_bf)；PQ 路径 Step 4 仍要在 centroid 图上做 CAGRA
        // 搜索，保留原来的 cagra::build 导航图。
        std::vector<uint32_t> centroid_exact_knn;   // (n_centroids, exact_knn_k)，非 PQ 时有效
        uint32_t exact_knn_k = 0;
        if (!mem_est.need_pq) {
            exact_knn_k = std::max<uint32_t>(K, (nprobe > 0) ? nprobe : K);
            centroid_exact_knn = centroid_self_knn_bf(d_centroids_f32, n_centroids, D, exact_knn_k);
            std::vector<uint32_t> g(static_cast<size_t>(n_centroids) * K);
            for (int64_t c = 0; c < n_centroids; ++c)
                std::memcpy(g.data() + static_cast<size_t>(c) * K,
                            centroid_exact_knn.data() + static_cast<size_t>(c) * exact_knn_k,
                            K * sizeof(uint32_t));
            CUDA_CHECK(cudaMalloc(&d_graph, g.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_graph, g.data(), g.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        } else {
            build_centroid_knn_on_gpu(
                d_centroids_f32, n_centroids, D, K, &d_graph);
        }

        size_t centroid_gpu_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
        cudaDeviceSynchronize();
        double iter_step3 = std::chrono::duration<double>(Clock::now() - t3).count();
        elapsed_step3 += iter_step3;
        std::cout << "  Step 3 done [" << std::fixed << std::setprecision(3) << iter_step3 << "s]\n";

        // Step 3.5 (原来的"整份加载数据集"步骤) 已经不需要了——Step 4/6 都改成
        // 按需从磁盘读了，不再要求整份数据集常驻内存。elapsed_step3p5 保留在
        // Timing Summary 里，恒为 0。
        //
        // select_centroids_on_gpu_kmeans_parallel 需要 fp32 X_sampled，多轮时
        // 每轮 Step 2 都要用，只有 iterations<=1 (只有这一轮、以后不会再用)
        // 时才能提前释放。
        if (iter == 0 && iterations <= 1) {
            auto _drop = std::move(X_sampled);
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
        double t4_centroids_dl = std::chrono::duration<double>(Clock::now() - t4).count();

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
                working_input_path, N, D, n_centroids,
                centroid_global_indices,
                d_centroids_u8, u8_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters);

            CUDA_CHECK(cudaFree(d_centroids_u8));
        } else {
            // Non-PQ mode: use float32 centroids directly
            std::function<void(const int64_t*, int64_t, float*)> loader;
            if (gres) {
                loader = [&](const int64_t* idx, int64_t n, float* d_out) {
                    gres->gather_queries_f32(idx, n, d_out);
                };
            }
            assignments = batch_assign_with_cagra_anns<DataT, float>(
                working_input_path, N, D, n_centroids,
                centroid_global_indices,
                d_centroids_f32, centroid_gpu_bytes,
                d_graph, graph_bytes,
                K, quantizer, Mbatch_bytes, search_max_iters, loader);

            CUDA_CHECK(cudaFree(d_centroids_f32));
        }
        double t4_assign_call = std::chrono::duration<double>(Clock::now() - t4).count() - t4_centroids_dl;

        // Download centroid KNN graph to CPU before freeing (needed for Step 6,
        // and for round 1's inline reorder just below, which always runs).
        if (neighbors_m > 0 || iter == 0) {
            centroid_knn_graph_host.resize(static_cast<size_t>(n_centroids) * K);
            CUDA_CHECK(cudaMemcpy(centroid_knn_graph_host.data(), d_graph,
                                  graph_bytes, cudaMemcpyDeviceToHost));
        }

        // Free KNN graph from GPU
        CUDA_CHECK(cudaFree(d_graph));
        cudaDeviceSynchronize();
        double t4_graph_dl_free = std::chrono::duration<double>(Clock::now() - t4).count()
                                   - t4_centroids_dl - t4_assign_call;

        // round 1 (iter==0) 分配完之后马上要做一次性重排，直接把数据写进
        // data_reordered.<ext>（见本轮 Step4 之后的新增逻辑）；BucketVectorAccumulator
        // 那份按桶散写的缓存对 round 1 来说从头到尾都用不上，只把它置成"本轮
        // 全空"状态 (histogram 全 0)，不花时间真的重新读一遍磁盘去写。
        // round 2..t 才真正调用 write_bucket_vectors_from_assignments 写数据。
        auto t4_bw0 = Clock::now();
        if (gres) {
            // GPU-resident: Step 6 直接用显存里的数据集 + assignments，不需要按桶落盘
        } else if (iter == 0) {
            bucket_vecs_ptr->start_iteration(std::vector<int64_t>(static_cast<size_t>(n_centroids), 0));
        } else {
            write_bucket_vectors_from_assignments<DataT>(
                working_input_path, N, D, n_centroids,
                centroid_global_indices, assignments, *bucket_vecs_ptr);
        }
        double t4_bucket_write = std::chrono::duration<double>(Clock::now() - t4_bw0).count();

        // flush 掉这一轮所有 bucket 剩余的写缓冲区尾部，Step 6 才能安全地整块
        // 读它们。
        auto t4_fin0 = Clock::now();
        if (!gres) bucket_vecs_ptr->finish_iteration();
        double t4_finish_iteration = std::chrono::duration<double>(Clock::now() - t4_fin0).count();

        double iter_step4 = std::chrono::duration<double>(Clock::now() - t4).count();
        elapsed_step4 += iter_step4;
        std::cout << "  [Step4 outer timing] centroids_dl=" << std::fixed << std::setprecision(3) << t4_centroids_dl
                  << "s assign_call=" << t4_assign_call
                  << "s graph_dl_free=" << t4_graph_dl_free
                  << "s bucket_write=" << t4_bucket_write
                  << "s finish_iteration=" << t4_finish_iteration
                  << "s total=" << iter_step4 << "s\n";
        std::cout << "  Step 4 done [" << std::fixed << std::setprecision(3) << iter_step4 << "s]\n";

        // ================================================================
        // Step 4.5 (round 1 only): inline reorder. Used to happen as a
        // separate post-hoc Step 7 after all `iterations` rounds, using
        // whichever round ran last; now it happens right after round 1's
        // Step 4 (assignments for the *whole* dataset are known, which is
        // the earliest point at which every bucket's final size -- and
        // hence every point's final id -- can be computed at all), so
        // rounds 2..t and the final merged vector_knn.bin are in this
        // coordinate system from the moment they're produced, and no
        // separate neighbor-id remapping pass is ever needed.
        // ================================================================
        std::vector<uint32_t> reorder_offsets_new;   // 处理位置索引；round1 的 Step6 要用
        std::vector<int32_t>  reorder_pos_of;         // 原始 bucket id -> 处理位置；round1 的 Step6 要用
        // round1 自己算出来的 bucket 处理顺序，直接传给下面 Step6 复用——Step6
        // 内部原本会用同一份 centroid_knn_graph_host/K/order_window 重新算一遍
        // compute_bucket_processing_order，结果跟这里必然逐位相同 (输入完全
        // 一样)，纯属重复计算 (实测 ~3s)，没必要算两次。
        std::vector<int32_t> bucket_process_order_round1;
        if (iter == 0) {
            std::cout << "=== Step 4.5: Inline reorder (round 1) ===\n";
            auto t45 = Clock::now();
            double t45_order = 0, t45_reorder_info = 0, t45_assign_range = 0;
            double t45_pass3 = 0, t45_artifacts = 0, t45_resample = 0;

            // Pass 2：复用现成的 compute_bucket_reorder，只是这里喂的是 round 1
            // 的 assignments（不是"最后一轮"的）。
            int32_t order_window = (order_window_arg > 0)
                ? order_window_arg
                : std::max<int32_t>(4 * static_cast<int32_t>(K), 16);
            bucket_process_order_round1 = bucket_order::compute_bucket_processing_order(
                bucket_order::adjacency_from_flat_graph(
                    centroid_knn_graph_host.data(), n_centroids, static_cast<int32_t>(K)),
                order_window);
            const auto& bucket_process_order = bucket_process_order_round1;
            {
                auto tnow = Clock::now();
                t45_order = std::chrono::duration<double>(tnow - t45).count();
            }
            auto reorder_info = compute_bucket_reorder(assignments, N, n_centroids,
                                                        bucket_process_order, centroid_global_indices);
            {
                auto tnow = Clock::now();
                t45_reorder_info = std::chrono::duration<double>(tnow - t45).count() - t45_order;
            }
            std::cout << "  N=" << N
                      << " total_in_buckets=" << reorder_info.total_in_buckets
                      << " (unassigned=" << (N - reorder_info.total_in_buckets) << ")\n";
            if (reorder_info.total_in_buckets != N) {
                // batch_assign_with_cagra_anns 的分配逻辑（4d 步，最近/次近两个
                // centroid 之间做均衡）总是给每个点分配一个桶，不存在拒绝分支，
                // 这里应当恒不触发；触发说明别处有 bug，直接报错而不是悄悄丢点、
                // 让后面所有轮次在一个变小的 N 上跑出一个不一致的 pipeline。
                throw std::runtime_error(
                    "Inline reorder: " + std::to_string(N - reorder_info.total_in_buckets) +
                    " point(s) were not assigned to any bucket in round 1. "
                    "batch_assign_with_cagra_anns is expected to always assign every point.");
            }

            reorder_pos_of.assign(static_cast<size_t>(n_centroids), 0);
            for (int64_t pos = 0; pos < n_centroids; ++pos)
                reorder_pos_of[static_cast<size_t>(bucket_process_order[static_cast<size_t>(pos)])] =
                    static_cast<int32_t>(pos);
            reorder_offsets_new = reorder_info.bucket_offsets;

            // assignments 的新 ID 版本（只有 iterations==1 时 Step5 会用得上，
            // 见下面）不需要经过 assignments[old_id]/perm 这层逐点查找——重排
            // 之后每个桶在新坐标系里天然是连续区间，区间
            // [offsets[pos],offsets[pos+1)) 里的新 ID 全都属于
            // bucket_process_order[pos] 这个桶，直接按区间填充就行。只依赖
            // bucket_process_order/reorder_offsets_new，跟 Pass 3、跟旧坐标系
            // 的 assignments 完全没关系，所以可以放在 Pass 3 之前算——但下面
            // "换到 assignments 这个变量名上"那一步还是要等 Pass 3 用完旧版本
            // 才能做（见 Pass 3 之后的注释）。
            std::vector<int64_t> assignments_by_bucket_range;
            if (iterations == 1 || gres) {
                assignments_by_bucket_range.resize(static_cast<size_t>(N));
                for (int64_t pos = 0; pos < n_centroids; ++pos) {
                    int64_t b = bucket_process_order[static_cast<size_t>(pos)];
                    uint32_t lo = reorder_offsets_new[static_cast<size_t>(pos)];
                    uint32_t hi = reorder_offsets_new[static_cast<size_t>(pos) + 1];
                    for (uint32_t new_id = lo; new_id < hi; ++new_id)
                        assignments_by_bucket_range[new_id] = b;
                }
            }
            {
                auto tnow = Clock::now();
                t45_assign_range = std::chrono::duration<double>(tnow - t45).count()
                                    - t45_order - t45_reorder_info;
            }

            // Pass 3：单遍顺序扫一遍原始数据集（一个 ifstream 全程开着，不是
            // 每批重开一次），按 assignments[] 分桶缓冲，精确定位 pwrite 进
            // data_reordered.<ext>（见 reordered_dataset_io.hpp 的设计说明：
            // Pass2 已经知道每个桶的精确最终大小，不需要事后合并）。
            std::string data_reordered_path = output_dir + "/data_reordered" + ext;
            if (gres) {
                // 原始数据已经在内存里: 按 inverse_perm 原地重排后一次顺序写出，
                // 显存里的常驻数据集同步换到新坐标系。
                std::vector<DataT> reordered(raw_host.size());
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < N; ++i) {
                    std::memcpy(reordered.data() + static_cast<size_t>(i) * D,
                                raw_host.data() + static_cast<size_t>(reorder_info.inverse_perm[i]) * D,
                                static_cast<size_t>(D) * sizeof(DataT));
                }
                raw_host = std::vector<DataT>();
                std::ofstream out(data_reordered_path, std::ios::binary | std::ios::trunc);
                int32_t hdr[2] = {N, D};
                out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
                out.write(reinterpret_cast<const char*>(reordered.data()),
                          static_cast<std::streamsize>(reordered.size() * sizeof(DataT)));
                if (!out.good())
                    throw std::runtime_error("Inline reorder: failed writing " + data_reordered_path);
                gres->permute(reorder_info.inverse_perm);
            } else {
                auto writer = reordered_io::ReorderedDatasetWriter<DataT>::create(
                    data_reordered_path, D, n_centroids,
                    reorder_offsets_new, reorder_pos_of, config.bucket_vec_buffer_bytes);

                std::ifstream scan_in(working_input_path, std::ios::binary);
                if (!scan_in.is_open())
                    throw std::runtime_error("Inline reorder: cannot open " + working_input_path);
                scan_in.seekg(2 * sizeof(int32_t));  // 跳过 header (int32 N, int32 D)

                constexpr int64_t kScanBatchRows = 1 << 16;
                std::vector<DataT> scan_batch;
                for (int64_t lo = 0; lo < N; lo += kScanBatchRows) {
                    int64_t rows = std::min<int64_t>(kScanBatchRows, N - lo);
                    scan_batch.resize(static_cast<size_t>(rows) * D);
                    scan_in.read(reinterpret_cast<char*>(scan_batch.data()),
                                static_cast<std::streamsize>(scan_batch.size() * sizeof(DataT)));
                    if (!scan_in.good())
                        throw std::runtime_error(
                            "Inline reorder: failed reading dataset batch at row " + std::to_string(lo));
                    for (int64_t r = 0; r < rows; ++r) {
                        writer.add(assignments[static_cast<size_t>(lo + r)],
                                  scan_batch.data() + static_cast<size_t>(r) * D);
                    }
                }
                writer.finish();
            }
            std::cout << "  Wrote " << data_reordered_path << "\n";
            {
                auto tnow = Clock::now();
                t45_pass3 = std::chrono::duration<double>(tnow - t45).count()
                            - t45_order - t45_reorder_info - t45_assign_range;
            }

            // assignments[old_id]（原变量）到这里为止都还是 Pass 3 需要的
            // *旧* 坐标系（Pass 3 按原始 input_path 的行序扫描，靠
            // assignments[old_id] 分桶，必须保持旧坐标系直到这里）——现在
            // Pass 3 已经用完它了，才能把上面按桶区间算好的新坐标系版本换过来。
            // 如果 iterations==1，这一轮同时也是 Step5 要用的"最后一轮"，
            // Step5 会直接拿 assignments/centroid_global_indices 写
            // bucket_index.bin/bucket_data.bin/centroid_global_indices.bin，
            // 而这时候 data_reordered.<ext>/vector_knn.bin/inverse_perm.bin
            // 已经全都是新坐标系了，不换的话这几个诊断文件会跟其它产物对不上。
            // iterations>1 时不需要：最后一轮的 Step4 本来就是在
            // data_reordered.<ext> 上跑的，算出来的 assignments 天然已经是
            // 新坐标系。
            if (iterations == 1 || gres) {
                assignments = std::move(assignments_by_bucket_range);
            }
            if (iterations == 1) {

                // centroid_global_indices 是几个具体的点（不是一段区间），没法
                // 像上面 assignments 那样按桶区间直接填——但也不需要为此专门
                // 查一张全量 perm 表，compute_bucket_reorder 已经在它自己那个
                // O(N) 循环里顺手把这几个点的新 ID 摘出来了（centroid_new_ids，
                // 只有 n_centroids 个），直接用就行。
                for (int64_t k = 0; k < n_centroids; ++k)
                    centroid_global_indices[static_cast<size_t>(k)] =
                        static_cast<int64_t>(reorder_info.centroid_new_ids[static_cast<size_t>(k)]);
            }

            // 永久性的重排产物：这份 bucket_offsets/bucket_process_order/
            // centroid_knn 从这里固定下来，round 2..t 不会再碰这几个文件名
            // （Step 5 已经去掉了写 centroid_knn.bin 那部分，见下面）。
            // 不写 perm.bin：它是 inverse_perm 的完全冗余信息（谁真需要
            // old->new，反着扫一遍 inverse_perm 就有），而且现在整个代码库里
            // 没有任何地方读它。
            graph_io::write_bucket_offsets(output_dir + "/bucket_offsets.bin", reorder_offsets_new);
            graph_io::write_bucket_process_order(output_dir + "/bucket_process_order.bin", bucket_process_order);
            graph_io::write_vector_bin(output_dir + "/inverse_perm.bin", reorder_info.inverse_perm);
            {
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
            {
                auto tnow = Clock::now();
                t45_artifacts = std::chrono::duration<double>(tnow - t45).count()
                                - t45_order - t45_reorder_info - t45_assign_range - t45_pass3;
            }

            working_input_path = data_reordered_path;

            // Step 2 选 centroid 用的 X_sampled/sampled_indices 是对着旧坐标系
            // 采的样；数据集坐标系已经变了，round 2..t 要用的话必须重新采一遍
            // (见 sample_from 定义处的注释)，否则第 2..t 轮选出来的
            // centroid_global_indices 会是错坐标系下的 ID。
            if (iterations > 1 && gres) {
                // fits_in_gpu 时 X_sampled 就是整份数据集 (sampled_indices = iota)，
                // 直接在内存里按 inverse_perm 重排，不用重新读盘。
                auto tresample0 = Clock::now();
                std::vector<float> X_new(X_sampled.size());
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < N; ++i) {
                    std::memcpy(X_new.data() + static_cast<size_t>(i) * D,
                                X_sampled.data() + static_cast<size_t>(reorder_info.inverse_perm[i]) * D,
                                static_cast<size_t>(D) * sizeof(float));
                }
                X_sampled = std::move(X_new);
                t45_resample = std::chrono::duration<double>(Clock::now() - tresample0).count();
            } else if (iterations > 1) {
                std::cout << "  Re-sampling for rounds 2.." << iterations
                          << " from the reordered dataset...\n";
                auto tresample0 = Clock::now();
                sample_from(working_input_path);
                t45_resample = std::chrono::duration<double>(Clock::now() - tresample0).count();
            }

            elapsed_step45 = std::chrono::duration<double>(Clock::now() - t45).count();
            std::cout << "  [Step4.5 timing] order=" << std::fixed << std::setprecision(3) << t45_order
                      << "s reorder_info=" << t45_reorder_info
                      << "s assign_range=" << t45_assign_range
                      << "s pass3_write=" << t45_pass3
                      << "s artifacts=" << t45_artifacts
                      << "s resample=" << t45_resample
                      << "s total=" << elapsed_step45 << "s\n";
            std::cout << "  Step 4.5 done [" << std::fixed << std::setprecision(3) << elapsed_step45 << "s]\n";
        }

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

            // centroid_knn.bin is NOT (re-)written here: round 1's Step 4.5
            // already wrote the permanent copy tied to bucket_offsets.bin/
            // bucket_process_order.bin. Overwriting it here with whichever
            // round happens to run last would silently break prune_windowed
            // (its bucket_process_order.bin would no longer match this file).

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

            // 非 PQ: Step 3 的精确自 KNN 已经算好前 max(K, nprobe) 个近邻，
            // 直接取前 nprobe 列。PQ: CAGRA-pruned graph 不是 sorted KNN, 必须用
            // greedy graph search 在导航图上为每个 centroid 找 top-nprobe 近邻。
            std::vector<uint32_t> centroid_topK;
            if (!centroid_exact_knn.empty() && effective_nprobe <= exact_knn_k) {
                centroid_topK.resize(static_cast<size_t>(n_centroids) * effective_nprobe);
                for (int64_t c = 0; c < n_centroids; ++c)
                    std::memcpy(centroid_topK.data() + static_cast<size_t>(c) * effective_nprobe,
                                centroid_exact_knn.data() + static_cast<size_t>(c) * exact_knn_k,
                                effective_nprobe * sizeof(uint32_t));
            } else {
                centroid_topK = expand_centroid_neighbors_gpu(
                    centroids_host_for_step6,
                    centroid_knn_graph_host,
                    n_centroids, D, K, effective_nprobe,
                    search_max_iters);
            }
            const uint32_t* graph_ptr = centroid_topK.data();
            uint32_t graph_K = effective_nprobe;
            double t6_topk_expand = std::chrono::duration<double>(Clock::now() - t6).count();

            if (gres) {
                gres->run_round(assignments, n_centroids, graph_ptr, graph_K);
                double iter_step6 = std::chrono::duration<double>(Clock::now() - t6).count();
                elapsed_step6 += iter_step6;
                std::cout << "  [Step6 timing] centroid_topk_expand=" << std::fixed << std::setprecision(3)
                          << t6_topk_expand << "s gpu_resident=" << (iter_step6 - t6_topk_expand)
                          << "s total=" << iter_step6 << "s\n";
                std::cout << "  Step 6 done [" << iter_step6 << "s]\n";
                continue;
            }

            // 这一轮的写入目标是专属的 chunk 文件 (chunk_<c>_iter<iter>.bin)，
            // 每轮开始前先开新文件。
            knn_acc->start_iteration(iter);

            // round 1: 数据集已经在 Step 4.5 里重排写进 data_reordered.<ext> 了，
            // 直接按 offset 读，不用碰 BucketVectorAccumulator 那份全空的临时
            // 文件（batch_assign_with_cagra_anns 在 round 1 传了 nullptr 跳过
            // 了它）。round 2..t 照旧用 BucketVectorAccumulator。
            if (iter == 0) {
                reordered_io::ReorderedDatasetBucketReader<DataT> reader(
                    working_input_path, D, reorder_offsets_new, reorder_pos_of);
                build_vector_knn_with_tensorcore<DataT>(
                    N, D, reader, graph_ptr,
                    n_centroids, graph_K, neighbors_m,
                    *knn_acc,
                    centroid_knn_graph_host.data(), K,
                    order_window_arg, step6_cache_bytes,
                    &bucket_process_order_round1);
            } else {
                build_vector_knn_with_tensorcore<DataT>(
                    N, D, *bucket_vecs_ptr, graph_ptr,
                    n_centroids, graph_K, neighbors_m,
                    *knn_acc,
                    centroid_knn_graph_host.data(), K,
                    order_window_arg, step6_cache_bytes);
            }
            double t6_after_build = std::chrono::duration<double>(Clock::now() - t6).count();
            double t6_build_vector_knn = t6_after_build - t6_topk_expand;

            // flush 掉这一轮剩余的 write buffer；merge_iteration 本身扔到
            // 后台线程做，让下一轮的 Step 2~6 立刻开始跑，不用等 merge 跑完
            // ——两轮的 merge 不能同时跑 (都要读写同一份 running_chunk 文件)，
            // 所以先等上一轮的 merge (如果还没完) 再发起这一轮的。不管
            // iterations 是 1 还是多轮，都是同一条代码路径——第 0 轮直接当
            // 全 sentinel 处理，不需要为它单独分支。
            knn_acc->finish_iteration_writes();
            double t6_finish_writes = std::chrono::duration<double>(Clock::now() - t6).count() - t6_after_build;
            double t6_wait = 0;
            if (pending_knn_merge.valid()) {
                auto t_wait = Clock::now();
                pending_knn_merge.get();
                t6_wait = std::chrono::duration<double>(Clock::now() - t_wait).count();
                std::cout << "  [ChunkedKNN] waited " << t6_wait
                          << "s for previous iteration's background merge\n";
            }
            {
                ChunkedKnnAccumulator* acc_ptr = knn_acc.get();
                if (iter == iterations - 1)
                    acc_ptr->set_final_output(output_dir + "/vector_knn.bin",
                                              output_dir + "/vector_dists.bin", iter);
                pending_knn_merge = std::async(std::launch::async,
                    [acc_ptr, iter]() { acc_ptr->merge_iteration(iter); });
            }

            cudaDeviceSynchronize();
            double iter_step6 = std::chrono::duration<double>(Clock::now() - t6).count();
            elapsed_step6 += iter_step6;
            std::cout << "  [Step6 timing] centroid_topk_expand=" << std::fixed << std::setprecision(3) << t6_topk_expand
                      << "s build_vector_knn=" << t6_build_vector_knn
                      << "s finish_writes=" << t6_finish_writes
                      << "s merge_wait=" << t6_wait
                      << "s total=" << iter_step6 << "s\n";
            std::cout << "  Step 6 done [" << std::fixed << std::setprecision(3) << iter_step6 << "s]\n";
        }

        }   // end of per-iteration loop

        // 最后一轮的 merge 还在后台跑，finalize 之前必须等它跑完，否则
        // running_chunk_*.bin 可能还没写完整就被读走。这段等待没有下一轮 GPU
        // 工作可以覆盖，是结构性的尾部开销，单独算一项 (不并入 Step 6)，这样
        // Step 6 的数字反映的是"build+pipeline 后仍暴露出来的 merge 时间"，
        // 跟这段"收尾必须等的时间"分开看。
        if (pending_knn_merge.valid()) {
            auto t_wait = Clock::now();
            pending_knn_merge.get();
            double wait_s = std::chrono::duration<double>(Clock::now() - t_wait).count();
            std::cout << "[ChunkedKNN] waited " << wait_s
                      << "s for the last iteration's background merge\n";
            elapsed_final_merge_wait = wait_s;
        }

        // 把按 chunk 分片的 running 状态 (跨全部 iteration 合并去重后的最终
        // 结果) 按 chunk 顺序拼接成 vector_knn.bin / vector_dists.bin。
        if (gres) {
            auto t_write = Clock::now();
            gres->write_outputs(output_dir + "/vector_knn.bin", output_dir + "/vector_dists.bin",
                                config.save_npy ? output_dir + "/neighbors.npy" : std::string());
            std::cout << "[WriteVectorKNN] Saved vector_knn.bin / vector_dists.bin"
                      << (config.save_npy ? " / neighbors.npy" : " (--skip-npy)") << "\n";
            elapsed_write_knn = std::chrono::duration<double>(Clock::now() - t_write).count();
        } else if (neighbors_m > 0) {
            auto t_write = Clock::now();
            knn_acc->finalize_to_file(
                output_dir + "/vector_knn.bin", output_dir + "/vector_dists.bin");
            std::filesystem::remove(knn_chunk_tmp_dir);  // 此时应已清空

            // 分块把 vector_knn.bin 转成 neighbors.npy，给 Python 端评测用
            // (纯顺序拷贝+类型转换，不需要整份常驻内存)
            if (config.save_npy) {
                convert_vector_knn_to_npy(
                    output_dir + "/vector_knn.bin", output_dir + "/neighbors.npy",
                    N, neighbors_m, config.cpu_limit_bytes / 4);
                std::cout << "[WriteVectorKNN] Saved neighbors.npy\n";
            }
            elapsed_write_knn = std::chrono::duration<double>(Clock::now() - t_write).count();
        }

        // Step 7（事后重排）已经不存在了——它的工作现在在 round 1 的
        // Step 4.5 里就做完了，round 2..t 和最终合并出的 vector_knn.bin
        // 从诞生起就已经是重排后的坐标系。

        // bucket_vecs_ptr 这份按桶散写的临时目录到这里可以清掉了（round 1
        // 全程是空文件，round 2..t 的内容 Step 6 已经读完不再需要）。用
        // remove_all 而不是 bucket_vecs_ptr->remove_all_files()，因为不同轮
        // n_centroids 可能不完全一样，remove_all_files() 只按最后一轮的
        // n_centroids 删，可能漏掉更早某轮 (n_centroids 更大时) 留下的文件；
        // 直接删整个临时目录更彻底。
        std::filesystem::remove_all(bucket_vec_dir);

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
        std::cout << "  Step 4.5 (Inline reorder, round 1 only): " << elapsed_step45 << "s\n";
        std::cout << "  Step 5   (Write buckets):       " << elapsed_step5 << "s\n";
        if (neighbors_m > 0) {
            std::cout << "  Step 6   (Per-vector KNN, merge included):      " << elapsed_step6 << "s\n";
            std::cout << "  Wait     (final iteration's background merge):  " << elapsed_final_merge_wait << "s\n";
            std::cout << "  Write    (close + .npy convert):    " << elapsed_write_knn << "s\n";
        }
        std::cout << "  --------------------------------\n";
        std::cout << "  Total:                        " << elapsed_total << "s\n";
        std::cout << "  Output: " << output_dir << "/\n";
        std::cout << "    data_reordered" << ext
                  << "            — round-1 bucket-aligned dataset (preserves input format)\n";
        std::cout << "    bucket_offsets.bin           — bucket boundaries (new ID space)\n";
        std::cout << "    bucket_process_order.bin     — for prune_windowed (bucket id at each position)\n";
        std::cout << "    inverse_perm.bin  — ID translation table (new id -> original id; "
                     "see translate_ids for translating a final graph back to original order)\n";
        std::cout << "    centroid_knn.bin             — round-1 centroid KNN graph (K=" << K
                  << "), for prune_windowed\n";
        std::cout << "    bucket_index.bin / bucket_data.bin — last-round bucket membership (diagnostic; "
                     "IDs are already in the new/reordered space)\n";
        std::cout << "    centroid_global_indices.bin  — last-round centroid-to-data mapping (diagnostic)\n";
        if (neighbors_m > 0) {
            std::cout << "    vector_knn.bin    — per-vector " << neighbors_m
                      << "-NN index, already in the reordered ID space\n";
            std::cout << "    neighbors.npy     — same as above in NumPy format (for search.py)\n";
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
        // ---- 提高 RLIMIT_NOFILE ----
        // ChunkedKnnAccumulator 在 Step 6 build 期间给每个 chunk 常驻打开一个
        // arrival 文件句柄 (num_chunks 个)。默认 soft limit 常常只有 1024，
        // 但 hard limit 通常大得多；soft limit 在 hard limit 范围内可以无需
        // root 权限直接提升，只影响本进程。这里尽量提到 hard limit (封顶
        // 65536)，让 ChunkedKnnAccumulator::create 算出来的 kMaxOpenChunks
        // 能有足够余量，不用再手动 `ulimit -n` 之后才跑。
        {
            struct rlimit rl;
            if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
                rlim_t target = std::min<rlim_t>(65536, rl.rlim_max);
                if (rl.rlim_cur < target) {
                    rlim_t old_cur = rl.rlim_cur;
                    rl.rlim_cur = target;
                    if (setrlimit(RLIMIT_NOFILE, &rl) == 0) {
                        std::cout << "[main] Raised RLIMIT_NOFILE soft limit: "
                                  << old_cur << " -> " << target << "\n";
                    } else {
                        std::cout << "[main] Warning: failed to raise RLIMIT_NOFILE "
                                     "(soft=" << old_cur << ", hard=" << rl.rlim_max
                                  << "); num_chunks will be sized conservatively.\n";
                    }
                }
            }
        }

        // ---- RMM 内存池 ----
        // RAFT 内部 (select_k / KMeans / CAGRA 等) 大量用 rmm::device_uvector
        // 做临时 scratch 分配；不装内存池的话，默认每次都是裸的
        // cudaMalloc/cudaFree —— 这两个调用同步且昂贵。Step 6 的 bucket 循环
        // 每 iteration 调用 10000 次 select_k，profiling 显示这段"提交 GPU
        // 工作"的耗时占了 loop wall time 的 65-78%，远超纯 kernel launch 应有
        // 的开销，怀疑就是这里。池子选比较保守的初始/上限大小，避免跟
        // build_vector_knn_with_tensorcore 里基于 cudaMemGetInfo 做的显存
        // 预算打架 (那部分是裸 cudaMalloc，不走 RMM，不受这个池影响)。
        rmm::mr::set_current_device_resource(rmm::mr::pool_memory_resource(
            rmm::mr::cuda_memory_resource{},
            256UL * 1024 * 1024,         // initial pool size: 256MB
            2UL * 1024 * 1024 * 1024));  // max pool size: 2GB, grows on demand

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
            ("bucket-vec-buffer", po::value<size_t>(),  "Step 4 bucket-vector-cache total write-buffer budget across all buckets (bytes, default 1GB)")
            ("sample-rate",   po::value<float>(),      "Sampling ratio (default 0.1)")
            ("centroid-ratio", po::value<float>(),     "Centroid ratio (default 0.01)")
            ("use-pq",        po::value<bool>(),       "Force PQ quantization")
            ("skip-npy",      po::bool_switch()->default_value(false),
                "Don't write neighbors.npy (vector_knn.bin as int64 NumPy, only for Python-side eval; "
                "the build pipeline never reads it)")
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
            ("order-window",  po::value<int32_t>()->default_value(0),
                "Sliding-window size for the round-1 bucket processing order (always computed -- "
                "the dataset is always reordered inline after round 1's assignment; DiskJoin-style "
                "task ordering over the centroid KNN graph; 0 = auto: 4*knn-k). Also reused for "
                "Step6's own Belady bucket-cache ordering.")
            ("cache-mb",      po::value<size_t>()->default_value(0),
                "Step6 Belady bucket-vector read cache budget in MB, keyed on the same "
                "DiskJoin-style processing order as --order-window (0 = auto: cpu-limit/2)")
            ("out-of-core",   po::bool_switch()->default_value(false),
                "Force the disk-backed Step 4/6 path even when the dataset + running KNN fit in GPU "
                "memory (default: auto-select the GPU-resident path when they fit).");

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
        int iterations          = vm["iterations"].as<int>();
        int32_t order_window_arg = vm["order-window"].as<int32_t>();
        size_t cache_mb_arg      = vm["cache-mb"].as<size_t>();
        bool force_out_of_core   = vm["out-of-core"].as<bool>();

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
                                            neighbors_m, nprobe, iterations,
                                            config, ext, order_window_arg, cache_mb_arg, force_out_of_core);
        } else if (ext == ".u8bin") {
            return run_pipeline_impl<uint8_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, iterations,
                                              config, ext, order_window_arg, cache_mb_arg, force_out_of_core);
        } else if (ext == ".i8bin") {
            return run_pipeline_impl<int8_t>(input_path, output_root, search_max_iters,
                                             neighbors_m, nprobe, iterations,
                                             config, ext, order_window_arg, cache_mb_arg, force_out_of_core);
        } else if (ext == ".ibin") {
            return run_pipeline_impl<int32_t>(input_path, output_root, search_max_iters,
                                              neighbors_m, nprobe, iterations,
                                              config, ext, order_window_arg, cache_mb_arg, force_out_of_core);
        } else if (ext == ".ubin") {
            return run_pipeline_impl<uint32_t>(input_path, output_root, search_max_iters,
                                               neighbors_m, nprobe, iterations,
                                               config, ext, order_window_arg, cache_mb_arg, force_out_of_core);
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
