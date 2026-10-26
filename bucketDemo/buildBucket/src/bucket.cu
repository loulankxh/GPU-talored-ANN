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
#include <deque>
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
#include <mma.h>
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

    // Step 2 (全量放得进 GPU 时) KMeans|| 只在这么大比例的随机子集上训练
    // (centroid 数仍按 N*centroid_ratio)，default 1.0 = 用全部点
    float kmeans_train_fraction;

    // out-of-core Step 6: GPU 端 bucket 缓存预算 (MB, 0 = 关闭)，见 DevBucket
    size_t gpu_bucket_cache_mb;
    // round 2..t 的 bucket 记录 (BucketVectorAccumulator) 放内存的预算 (MB, 0 = cpu_limit/4)
    size_t bucket_mem_mb = 0;

    // Output
    bool save_dists;               // 写 vector_dists.bin (default true)；流水线后续阶段不读它
    bool save_npy;                 // 额外写 neighbors.npy (int64, 给 Python 端评测用), default true;
                                   // 构建流水线本身不读它
    // > 0: GPU-resident 路径在 bucket2 进程内接着做 prune + 反向边 (cuVS
    // helpers::optimize，跟独立的 optimize 一样) + id 翻译，直接写 final_graph.bin
    // (= optimize + translate_ids 两个阶段的输出)；out-of-core 路径忽略。default 0
    int fused_optimize_degree;

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
        kmeans_train_fraction(1.0f),
        gpu_bucket_cache_mb(2048),
        save_dists(true),
        save_npy(true),
        fused_optimize_degree(0)
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
        // centroid_ratio 是相对全量 N 的 (validate 要求 centroid_ratio < sample_rate)；
        // 原来按采样行数乘 → SIFT100M 只有 N/1000 个 centroid、bucket 平均 1000 点，Step 6 显存需求 29 GB
        est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
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

    // centroid_ratio 是相对全量 N 的 (validate 要求 centroid_ratio < sample_rate)；
    // 原来按采样行数乘 → SIFT100M 只有 N/1000 个 centroid、bucket 平均 1000 点，Step 6 显存需求 29 GB
    est.centroid_rows = std::max(static_cast<int64_t>(1), static_cast<int64_t>(N * config.centroid_ratio));
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
 * n_samples 个互不相同的行号 (升序)。原来名字叫 without_replacement、实际是有放回
 * 抽样：SIFT100M 抽 10M/100M 会有 ~50 万个重复行 → KMeans|| 可能把同一个点选成两个
 * centroid → 两个 bucket 的 centroid 是同一个 gid，只有一个能拿到它的分配，另一个
 * bucket 直方图为 0，round 2 的 BucketVectorAccumulator::add 直接报错。
 * 先有放回地抽、排序去重，再补抽不够的部分 (每轮补的期望重复率 ≈ n/N，几轮收敛)。
 */
std::vector<int64_t> sample_without_replacement(
    int64_t N,
    int64_t n_samples,
    uint32_t seed) {

    if (n_samples > N) throw std::runtime_error("sample_without_replacement: n_samples > N");
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, N - 1);
    std::vector<int64_t> indices;
    indices.reserve(static_cast<size_t>(n_samples));
    while (static_cast<int64_t>(indices.size()) < n_samples) {
        const int64_t need = n_samples - static_cast<int64_t>(indices.size());
        for (int64_t i = 0; i < need; ++i) indices.push_back(dist(rng));
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
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
    if (vm.count("kmeans-train-fraction")) {
        config.kmeans_train_fraction = vm["kmeans-train-fraction"].as<float>();
        if (!(config.kmeans_train_fraction > 0.0f && config.kmeans_train_fraction <= 1.0f))
            throw std::runtime_error("--kmeans-train-fraction must be in (0, 1]");
    }
    if (vm.count("gpu-bucket-cache-mb")) {
        config.gpu_bucket_cache_mb = vm["gpu-bucket-cache-mb"].as<size_t>();
    }
    if (vm.count("bucket-mem-mb")) {
        config.bucket_mem_mb = vm["bucket-mem-mb"].as<size_t>();
    }
    if (vm.count("skip-dists")) {
        config.save_dists = !vm["skip-dists"].as<bool>();
    }
    if (vm.count("skip-npy")) {
        config.save_npy = !vm["skip-npy"].as<bool>();
    }
    if (vm.count("fused-optimize-degree")) {
        config.fused_optimize_degree = vm["fused-optimize-degree"].as<int>();
        if (config.fused_optimize_degree < 0)
            throw std::runtime_error("--fused-optimize-degree must be >= 0");
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
        // 不初始化 (原来 vector<char>::resize 把 ~1 GB 写缓冲整份清零，SIFT1M
        // 第一轮 bucket_vec_create ≈ 0.75 s)；只按 capacity_records 读写
        std::unique_ptr<char[]> data;
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

    // 内存模式: 本轮合并后的总字节数 (N*(8+D*sizeof)) 不超过 mem_budget 时，
    // 每条记录直接写到它在合并缓冲区里的最终位置 (histogram 已经给出每个桶
    // 的偏移)，不经过 per-bucket 写缓冲/文件/flush，读桶就是 memcpy。原来
    // round ≥2 每轮 bucket_write + finish_iteration (落盘, 被 dirty 页限速)
    // ~0.8–1.0 s。超预算时走文件，跟原来一样。
    size_t mem_budget = 0;
    bool in_memory = false;
    std::unique_ptr<char[]> mem;
    size_t mem_cap = 0;
    void set_in_memory_budget(size_t bytes) { mem_budget = bytes; }

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
            b.data.reset(new char[cap_records * rec_bytes]);
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

        in_memory = static_cast<size_t>(pos) <= mem_budget;
        if (in_memory) {
            if (static_cast<size_t>(pos) > mem_cap) {
                mem.reset(new char[static_cast<size_t>(std::max<int64_t>(pos, 1))]);   // 不初始化: 每个字节都会被 add 写一次
                mem_cap = static_cast<size_t>(pos);
            }
            return;
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
        merged_out.write(b.data.get(), static_cast<std::streamsize>(nbytes));
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
        if (in_memory) {
            char* dst = mem.get() + b.write_cursor;
            std::memcpy(dst, &gid, sizeof(int64_t));
            std::memcpy(dst + sizeof(int64_t), vec, static_cast<size_t>(D) * sizeof(DataT));
            b.write_cursor += static_cast<int64_t>(rec_bytes);
            return;
        }
        size_t off = b.count * rec_bytes;
        std::memcpy(b.data.get() + off, &gid, sizeof(int64_t));
        std::memcpy(b.data.get() + off + sizeof(int64_t), vec, static_cast<size_t>(D) * sizeof(DataT));
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
            if (!in_memory) flush_bucket(c);
            if (b.added_so_far != b.total_count)
                throw std::runtime_error(
                    "BucketVectorAccumulator: bucket " + std::to_string(c) + " got "
                    + std::to_string(b.added_so_far) + " add() calls, expected "
                    + std::to_string(b.total_count) + " from histogram");
        }
        if (in_memory) return;
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

    // 内存模式下批量写一段连续 gid [lo, lo+n) (跳过 is_centroid 的点)，多线程且
    // 结果跟逐条 add() 完全一样：每个线程管一段连续 gid，先按桶计数，再按
    // (桶, 线程) 前缀和定出每段的起始位置 → 桶内顺序仍是 gid 递增。
    void add_range_parallel(int64_t lo, int64_t n, const DataT* data,
                            const std::vector<int64_t>& assignments, const std::vector<char>& is_centroid) {
        if (!in_memory) throw std::runtime_error("add_range_parallel: only in memory mode");
        const size_t rec_bytes = record_bytes(D);
        const int T = std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>((n + 4095) / 4096)));
        const int64_t per = (n + T - 1) / T;
        std::vector<int64_t> cnt(static_cast<size_t>(T) * n_centroids, 0);
        int64_t n_bad = 0;
        #pragma omp parallel for num_threads(T) schedule(static) reduction(+:n_bad)
        for (int t = 0; t < T; ++t) {
            int64_t* ct = cnt.data() + static_cast<size_t>(t) * n_centroids;
            const int64_t a = lo + t * per, b = std::min(lo + n, a + per);
            for (int64_t gi = a; gi < b; ++gi) {
                if (is_centroid[static_cast<size_t>(gi)]) continue;
                int64_t c = assignments[static_cast<size_t>(gi)];
                if (c < 0 || c >= n_centroids) { ++n_bad; continue; }   // bounds check
                ++ct[c];
            }
        }
        if (n_bad > 0) throw std::runtime_error("add_range_parallel: invalid bucket id in assignments");
        // 每个 (线程, 桶) 段的起始字节偏移；桶的 write_cursor / added_so_far 一次推进
        for (int64_t c = 0; c < n_centroids; ++c) {
            auto& bk = bufs[static_cast<size_t>(c)];
            int64_t cur = bk.write_cursor, add = 0;
            for (int t = 0; t < T; ++t) {
                int64_t k = cnt[static_cast<size_t>(t) * n_centroids + c];
                cnt[static_cast<size_t>(t) * n_centroids + c] = cur;
                cur += k * static_cast<int64_t>(rec_bytes);
                add += k;
            }
            if (bk.added_so_far + add > bk.total_count)
                throw std::runtime_error("add_range_parallel: exceeded histogram count for bucket " + std::to_string(c));
            bk.write_cursor = cur;
            bk.added_so_far += add;
        }
        #pragma omp parallel for num_threads(T) schedule(static)
        for (int t = 0; t < T; ++t) {
            int64_t* off = cnt.data() + static_cast<size_t>(t) * n_centroids;
            const int64_t a = lo + t * per, b = std::min(lo + n, a + per);
            for (int64_t gi = a; gi < b; ++gi) {
                if (is_centroid[static_cast<size_t>(gi)]) continue;
                int64_t c = assignments[static_cast<size_t>(gi)];
                char* dst = mem.get() + off[c];
                std::memcpy(dst, &gi, sizeof(int64_t));
                std::memcpy(dst + sizeof(int64_t), data + static_cast<size_t>(gi - lo) * D,
                            static_cast<size_t>(D) * sizeof(DataT));
                off[c] += static_cast<int64_t>(rec_bytes);
            }
        }
    }

    // 桶 c 的 cnt 条记录的起始地址: 内存模式直接指向合并缓冲区，否则读进 raw
    const char* bucket_bytes(int64_t c, int64_t cnt, std::vector<char>& raw) {
        const size_t nbytes = static_cast<size_t>(cnt) * record_bytes(D);
        const int64_t off = merged_offset[static_cast<size_t>(c)];
        if (in_memory) {
            if (static_cast<size_t>(off) + nbytes > mem_cap)   // bounds check
                throw std::runtime_error("BucketVectorAccumulator: bucket " + std::to_string(c) + " out of range");
            return mem.get() + off;
        }
        raw.resize(nbytes);
        merged_in.seekg(static_cast<std::streamoff>(off));
        merged_in.read(raw.data(), static_cast<std::streamsize>(nbytes));
        if (!merged_in.good())
            throw std::runtime_error("BucketVectorAccumulator: read failed for bucket " + std::to_string(c));
        return raw.data();
    }

    // 内存模式下整份记录缓冲区就是一块连续 host blob (每条 = int64 gid + D 个 DataT)，
    // Step 6 可以整块上传 GPU、gather 直接按偏移取 (见 build_vector_knn 的 blob 模式)
    bool blob_layout(const char*& base, size_t& bytes, int& mode, int& stride) const {
        if (!in_memory) return false;
        base = mem.get(); bytes = mem_cap; mode = 1; stride = static_cast<int>(record_bytes(D));
        return true;
    }
    void blob_locate(int64_t c, size_t& byte_off, int64_t& gid0) const {
        byte_off = static_cast<size_t>(merged_offset[static_cast<size_t>(c)]); gid0 = 0;
    }

    // 顺序整块读出 bucket c 的全部 (gid, vector) 记录 (从合并文件里按已知偏移 seek)。
    void read_bucket(int64_t c, std::vector<int64_t>& ids_out, std::vector<DataT>& vecs_out) {
        int64_t cnt = count(c);
        ids_out.resize(static_cast<size_t>(cnt));
        vecs_out.resize(static_cast<size_t>(cnt) * static_cast<size_t>(D));
        if (cnt == 0) return;

        size_t rec_bytes = record_bytes(D);
        std::vector<char> raw;
        const char* src = bucket_bytes(c, cnt, raw);
        for (int64_t i = 0; i < cnt; ++i) {
            const char* p = src + static_cast<size_t>(i) * rec_bytes;
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
        std::vector<char> raw;
        const char* src = bucket_bytes(c, cnt, raw);
        for (int64_t i = 0; i < cnt; ++i) {
            const char* p = src + static_cast<size_t>(i) * rec_bytes;
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

// ---- Step 4 精确 top-2 (8-bit 数据): fp16 Tensor Core GEMM + 一 warp 一行 top-2 ----
// uint8/int8 在 fp16 里精确，乘积 ≤ 65025，D=128 求和 ≤ 8.3M < 2^24，fp32 累加
// 精确 (跟 GpuResidentKnn 的 fp16 GEMM 同一论证)；d = ‖c‖² − 2 q·c (省掉每行常数
// ‖q‖²) 的每一步也都是 < 2^24 的整数 → 跟 fp32 brute force 的排序完全一致。
// 距离相同按 centroid id 小者优先 (确定性)。
__global__ void step4_f32_to_f16_kernel(const float* __restrict__ in, __half* __restrict__ out, int64_t n)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

__global__ void step4_row_norms_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n, int D)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) { float v = x[i * D + d]; acc += v * v; }
    out[i] = acc;
}

__device__ __forceinline__ bool step4_less(float da, int ia, float db, int ib)
{
    return da < db || (da == db && ia < ib);
}

__global__ void step4_top2_kernel(const float* __restrict__ dots,   // (n_rows, C) row-major
                                  const float* __restrict__ cnorm,  // (C,)
                                  int64_t n_rows, int C,
                                  uint32_t* __restrict__ out,       // (n_rows, 2)
                                  const float* __restrict__ q = nullptr, int D = 0,   // A6a: 真实距离要 ‖q‖²
                                  float* __restrict__ out_d = nullptr)                // (n_rows, 2) 平方 L2
{
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const float* r = dots + row * static_cast<int64_t>(C);
    float b1 = INFINITY, b2 = INFINITY;
    int i1 = 0x7fffffff, i2 = 0x7fffffff;
    for (int c = lane; c < C; c += 32) {
        float d = cnorm[c] - 2.0f * r[c];
        if (step4_less(d, c, b1, i1)) { b2 = b1; i2 = i1; b1 = d; i1 = c; }
        else if (step4_less(d, c, b2, i2)) { b2 = d; i2 = c; }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ob1 = __shfl_down_sync(0xffffffffu, b1, off);
        int   oi1 = __shfl_down_sync(0xffffffffu, i1, off);
        float ob2 = __shfl_down_sync(0xffffffffu, b2, off);
        int   oi2 = __shfl_down_sync(0xffffffffu, i2, off);
        // 两个已排序的 (b1,b2) / (ob1,ob2) 归并取前两个
        if (step4_less(ob1, oi1, b1, i1)) {
            if (step4_less(b1, i1, ob2, oi2)) { b2 = b1; i2 = i1; } else { b2 = ob2; i2 = oi2; }
            b1 = ob1; i1 = oi1;
        } else if (step4_less(ob1, oi1, b2, i2)) {
            b2 = ob1; i2 = oi1;
        }
    }
    float qn = 0.f;
    if (out_d) {   // ‖q‖² (warp 归约)，距离 = ‖q‖² + ‖c‖² - 2 q·c
        for (int e = lane; e < D; e += 32) { const float v = q[row * D + e]; qn += v * v; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) qn += __shfl_xor_sync(0xffffffffu, qn, off);
    }
    if (lane == 0) {
        out[row * 2]     = static_cast<uint32_t>(i1);
        out[row * 2 + 1] = static_cast<uint32_t>(i2);
        if (out_d) { out_d[row * 2] = b1 + qn; out_d[row * 2 + 1] = b2 + qn; }
    }
}

// float32 Step 4 (#39, 用户优先级 1): fp16 Tensor Core 打分后每行留近似前 R 个 centroid，再用 fp32
// 精确重算这 R 个的 ‖c‖² - 2·q·c 取前 2 (同距离取下标小的)。
template <typename SrcT>
__global__ void step4_to_f16_pad_kernel(const SrcT* __restrict__ x, __half* __restrict__ out,
                                        int64_t n, int D, int Dh)   // (n, D) → (n, Dh)，补位 0
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n * Dh) return;
    const int64_t r = i / Dh; const int c = static_cast<int>(i - r * Dh);
    out[i] = c < D ? __float2half_rn(static_cast<float>(x[r * D + c])) : __float2half_rn(0.0f);
}

// 4 个有序 (d, i) 寄存器槽的插入 (同 step4_top2_kernel 的 if 链)
__device__ __forceinline__ void step4_ins4(float d, int c, float (&b)[4], int (&ix)[4])
{
    if (!step4_less(d, c, b[3], ix[3])) return;
    if (step4_less(d, c, b[1], ix[1])) {
        b[3] = b[2]; ix[3] = ix[2]; b[2] = b[1]; ix[2] = ix[1];
        if (step4_less(d, c, b[0], ix[0])) { b[1] = b[0]; ix[1] = ix[0]; b[0] = d; ix[0] = c; }
        else { b[1] = d; ix[1] = c; }
    } else if (step4_less(d, c, b[2], ix[2])) { b[3] = b[2]; ix[3] = ix[2]; b[2] = d; ix[2] = c; }
    else { b[3] = d; ix[3] = c; }
}

// fp16 近似分数取前 4，fp32 精确重算这 4 个取前 2
__global__ void step4_top4_rerank_kernel(const float* __restrict__ dots,   // (n_rows, C) 近似点积
                                         const float* __restrict__ cnorm,  // (C,) 精确
                                         const float* __restrict__ cent,   // (C, D) fp32
                                         const float* __restrict__ q,      // (n_rows, D) fp32
                                         int64_t n_rows, int C, int D,
                                         uint32_t* __restrict__ out,       // (n_rows, 2)
                                         float* __restrict__ out_d = nullptr)   // A6a: (n_rows, 2) 平方 L2
{
    const int64_t row = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (row >= n_rows) return;
    const float* r = dots + row * static_cast<int64_t>(C);
    float b[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    int ix[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    for (int c = lane; c < C; c += 32) step4_ins4(cnorm[c] - 2.0f * r[c], c, b, ix);
    for (int off = 16; off > 0; off >>= 1) {   // 两个有序 4 元组归并取前 4
        float ob[4]; int oi[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) { ob[k] = __shfl_down_sync(0xffffffffu, b[k], off); oi[k] = __shfl_down_sync(0xffffffffu, ix[k], off); }
        #pragma unroll
        for (int k = 0; k < 4; ++k) step4_ins4(ob[k], oi[k], b, ix);
    }
    const float* qv = q + row * static_cast<int64_t>(D);
    float e1 = INFINITY, e2 = INFINITY; int j1 = 0x7fffffff, j2 = 0x7fffffff;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int c = __shfl_sync(0xffffffffu, ix[k], 0);
        if (c < 0 || c >= C) continue;   // bounds check (不足 4 个)
        const float* cv = cent + static_cast<int64_t>(c) * D;
        float part = 0.0f;
        for (int e = lane; e < D; e += 32) part += qv[e] * cv[e];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        const float d = cnorm[c] - 2.0f * part;
        if (step4_less(d, c, e1, j1)) { e2 = e1; j2 = j1; e1 = d; j1 = c; }
        else if (step4_less(d, c, e2, j2)) { e2 = d; j2 = c; }
    }
    float qn = 0.f;
    if (out_d) {
        for (int e = lane; e < D; e += 32) { const float v = qv[e]; qn += v * v; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) qn += __shfl_xor_sync(0xffffffffu, qn, off);
    }
    if (lane == 0) {
        out[row * 2]     = static_cast<uint32_t>(j1);
        out[row * 2 + 1] = static_cast<uint32_t>(j2);
        if (out_d) { out_d[row * 2] = e1 + qn; out_d[row * 2 + 1] = e2 + qn; }
    }
}

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
    const std::function<void(const int64_t* idx, int64_t n, CentroidT* d_out)>& gpu_query_loader = {},
    // 非空时 (out-of-core 路径、原始数据集整份放得进 CPU 预算)：(N, D) 原始数据
    // 已在 host 内存里 (当前坐标系)，直接按区间取指针，不再读盘。
    const T* host_raw = nullptr)
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
    static const uint32_t itopk_floor = [] {
        const char* e = std::getenv("BUCKET2_ASSIGN_ITOPK");
        // 只要 top-2: itopk 32 (cuVS 允许的最小 32 对齐值) 比 64 快一倍
        // (SIFT10M 100K centroids: 9.3 → 5.2 s/轮)，BUCKET2_ASSIGN_ITOPK 可调
        return e ? static_cast<uint32_t>(std::atoi(e)) : 32u;
    }();
    const uint32_t itopk = std::max<uint32_t>(itopk_floor, ((TOP_K + 31) / 32) * 32);

    // Per-point GPU (CAGRA 不再需要 buf_idx/dist/visited):
    //   query:       D * sizeof(CentroidT)
    //   neighbors:   TOP_K * 4
    //   distances:   TOP_K * 4
    //   (非 PQ、host 区间上传路径) 原始 T 区间 + local row 下标: D * sizeof(T) + 4
    //   (Nv 大时这块 SIFT100M: 26M × 132 B = 3.4 GB，原来没算进来 → OOM)
    size_t per_point_bytes = static_cast<size_t>(D) * sizeof(CentroidT)
                           + TOP_K * (sizeof(uint32_t) + sizeof(float));
    if (!is_pq && !gpu_query_loader && std::is_same<CentroidT, float>::value)
        per_point_bytes += static_cast<size_t>(D) * sizeof(T) + sizeof(int32_t);

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
    // 8-bit 数据: 精确 fp16 GEMM + top-2 (见 step4_top2_kernel)，比 cuVS fp32
    // brute force 快 ~4x；其它数据类型仍走 brute_force。
    constexpr bool kExactF16 = !is_pq &&
        (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value);
    constexpr int64_t kF16Tile = 4096;   // query tile: dots = 4096 × n_centroids × 4 B (10K centroids: 164 MB)
    // 精确 GEMM 的代价 ∝ N × n_centroids (每个 query tile 还要写回再读一遍
    // tile × n_centroids 的 dots)，centroid 多了以后 (SIFT10M: 100K) 反而比
    // CAGRA 图搜索 (代价 ∝ N × log n_centroids) 慢；超过阈值走 CAGRA。
    static const int64_t exact_max = [] {
        const char* e = std::getenv("BUCKET2_EXACT_ASSIGN_MAX");
        // SIFT1M (10K centroids) 精确 0.12 s/轮 vs CAGRA ~0.7 s；SIFT10M (100K)
        // 精确 11.1 s/轮 vs CAGRA (itopk 32) 5.2 s。BUCKET2_EXACT_ASSIGN_MAX 可调
        return e ? std::atoll(e) : static_cast<int64_t>(32768);
    }();
    const bool exact_f16 = kExactF16 && n_centroids <= exact_max;
    // float32 数据: 精确 fp32 SGEMM + 同一个 top-2 kernel (P2.4)，代替 cuVS brute_force
    // (L2Expanded 分块 GEMM + select_k，msTuring10M 40K centroids 17 s/轮)。同样是精确
    // top-2；只是 fp32 舍入顺序不同 (不加每行常数 ‖q‖²)，距离几乎相等的点可能换一个 centroid。
    // query tile 按 dots ≤ 512 MB 定 (centroid 多时 tile 变小)。BUCKET2_F32_ASSIGN=0 关掉。
    constexpr bool kExactF32 = !is_pq && std::is_same<T, float>::value;
    const bool exact_f32 = kExactF32 && n_centroids >= 2 &&
        !(std::getenv("BUCKET2_F32_ASSIGN") && std::getenv("BUCKET2_F32_ASSIGN")[0] == '0');
    const int64_t f32_tile = std::max<int64_t>(32, std::min<int64_t>(kF16Tile,
        (int64_t(512) << 20) / (std::max<int64_t>(n_centroids, 1) * static_cast<int64_t>(sizeof(float)))));
    cublasHandle_t f16_cublas = nullptr;
    __half*   d_cent_h  = nullptr;
    float*    d_cnorm   = nullptr;
    __half*   d_q_h     = nullptr;
    float*    d_f16_dots = nullptr;
    uint32_t* d_top2_u32 = nullptr;
    // A6a (user-approved 2026-10-25): DiskJoin 的两选一规则 (add2choice): 第二近的桶更小 且 d2 < 1.2·d1 才去
    // 第二近的，否则最近的。要 top-2 的真实平方 L2 距离。BUCKET2_ASSIGN_RULE=diskjoin 打开 (默认原规则 s2 < 0.8·s1)
    const bool rule_dj = std::getenv("BUCKET2_ASSIGN_RULE") && std::string(std::getenv("BUCKET2_ASSIGN_RULE")) == "diskjoin";
    float* d_top2_d = nullptr;
    std::vector<float> h_top2_d;
    if (kExactF16 && exact_f16) {
        if (n_centroids < 2) throw std::runtime_error("batch_assign: need >= 2 centroids");
        if (cublasCreate(&f16_cublas) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch_assign: cublasCreate failed");
        cublasSetMathMode(f16_cublas, CUBLAS_TENSOR_OP_MATH);
        const int64_t cd = n_centroids * D;
        CUDA_CHECK(cudaMalloc(&d_cent_h, cd * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_cnorm, n_centroids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q_h, kF16Tile * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_f16_dots, kF16Tile * n_centroids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top2_u32, Nv * TOP_K * sizeof(uint32_t)));
        if (rule_dj && !d_top2_d) CUDA_CHECK(cudaMalloc(&d_top2_d, Nv * TOP_K * sizeof(float)));
        const float* d_cent_f = reinterpret_cast<const float*>(d_centroids);
        step4_f32_to_f16_kernel<<<static_cast<unsigned>((cd + 255) / 256), 256>>>(d_cent_f, d_cent_h, cd);
        step4_row_norms_kernel<<<static_cast<unsigned>((n_centroids + 255) / 256), 256>>>(
            d_cent_f, d_cnorm, n_centroids, static_cast<int>(D));
        CUDA_CHECK(cudaGetLastError());
    }
    // #39: fp32 数据的精确 top-2 用 fp16 Tensor Core 打分 + fp32 重算前 kStep4R 个 (BUCKET2_F32_F16SCORE=0 关掉)
    const bool f32_f16score = exact_f32 &&
        !(std::getenv("BUCKET2_F32_F16SCORE") && std::getenv("BUCKET2_F32_F16SCORE")[0] == '0');
    const int64_t Dh4 = (static_cast<int64_t>(D) + 7) / 8 * 8;   // fp16 行宽补到 8 的倍数 (align8 Tensor Core kernel)
    if (exact_f32) {
        if (cublasCreate(&f16_cublas) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch_assign: cublasCreate failed");
        if (f32_f16score) {
            cublasSetMathMode(f16_cublas, CUBLAS_TENSOR_OP_MATH);
            const int64_t ch = n_centroids * Dh4;
            CUDA_CHECK(cudaMalloc(&d_cent_h, ch * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_q_h, f32_tile * Dh4 * sizeof(__half)));
            step4_to_f16_pad_kernel<float><<<static_cast<unsigned>((ch + 255) / 256), 256>>>(
                reinterpret_cast<const float*>(d_centroids), d_cent_h, n_centroids, static_cast<int>(D), static_cast<int>(Dh4));
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMalloc(&d_cnorm, n_centroids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_f16_dots, f32_tile * n_centroids * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top2_u32, Nv * TOP_K * sizeof(uint32_t)));
        if (rule_dj && !d_top2_d) CUDA_CHECK(cudaMalloc(&d_top2_d, Nv * TOP_K * sizeof(float)));
        step4_row_norms_kernel<<<static_cast<unsigned>((n_centroids + 255) / 256), 256>>>(
            reinterpret_cast<const float*>(d_centroids), d_cnorm, n_centroids, static_cast<int>(D));
        CUDA_CHECK(cudaGetLastError());
    }
    std::optional<cuvs::neighbors::brute_force::index<float, float>> bf_idx;
    int64_t* d_top2_i64 = nullptr;
    std::vector<int64_t> h_top2_i64;
    if constexpr (!is_pq && !kExactF16) {
        if (use_bf && !exact_f32) {
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

    std::vector<T> raw_batch;   // 跨 batch 复用 (每批重新分配会单线程清零一遍)
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
        const T* raw_ptr = nullptr;
        if (!is_pq && gpu_query_loader) {
            gpu_query_loader(non_centroid_indices.data() + batch_start, batch_size, d_queries);
        } else if (host_raw) {
            if (batch_lo < 0 || batch_hi >= N || batch_lo > batch_hi)   // bounds check
                throw std::runtime_error("batch_assign: host_raw range out of bounds");
            raw_ptr = host_raw + static_cast<size_t>(batch_lo) * D;
        } else {
            int32_t N_tmp, D_tmp;
            load::read_bigann_raw_range<T>(input_path, batch_lo, batch_hi - batch_lo + 1,
                                            raw_batch, N_tmp, D_tmp);
            raw_ptr = raw_batch.data();
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
                    batch_f32[i * D + d] = static_cast<float>(raw_ptr[local * D + d]);
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
            CUDA_CHECK(cudaMemcpy(d_raw_range, raw_ptr, raw_elems * sizeof(T),
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
                    batch_data[i * D + d] = static_cast<CentroidT>(raw_ptr[local * D + d]);
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
        bool searched_f16 = false;
        if (kExactF16 && exact_f16) {
            const float alpha = 1.0f, beta = 0.0f;
            const float* d_q_f = reinterpret_cast<const float*>(d_queries);
            for (int64_t t0 = 0; t0 < batch_size; t0 += kF16Tile) {
                const int64_t tn = std::min(kF16Tile, batch_size - t0);
                const int64_t qe = tn * D;
                step4_f32_to_f16_kernel<<<static_cast<unsigned>((qe + 255) / 256), 256>>>(
                    d_q_f + t0 * D, d_q_h, qe);
                // dots (tn, C) row-major = q_tile · centroidsᵀ
                cublasStatus_t st = cublasGemmEx(f16_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(n_centroids), static_cast<int>(tn), static_cast<int>(D), &alpha,
                    d_cent_h, CUDA_R_16F, static_cast<int>(D),
                    d_q_h, CUDA_R_16F, static_cast<int>(D), &beta,
                    d_f16_dots, CUDA_R_32F, static_cast<int>(n_centroids),
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (st != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("batch_assign: fp16 GEMM failed, status=" + std::to_string(st));
                const int64_t threads = tn * 32;
                step4_top2_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(
                    d_f16_dots, d_cnorm, tn, static_cast<int>(n_centroids), d_top2_u32 + t0 * TOP_K,
                    d_q_f + t0 * D, static_cast<int>(D), d_top2_d ? d_top2_d + t0 * TOP_K : nullptr);
                CUDA_CHECK(cudaGetLastError());
            }
            searched_f16 = true;
        }
        if (exact_f32) {
            // 同上，fp32: dots (tn, C) = q_tile · centroidsᵀ，结果走 fp16 路径的下载 / 检查
            const float alpha = 1.0f, beta = 0.0f;
            const float* d_q_f = reinterpret_cast<const float*>(d_queries);
            const float* d_c_f = reinterpret_cast<const float*>(d_centroids);
            for (int64_t t0 = 0; t0 < batch_size; t0 += f32_tile) {
                const int64_t tn = std::min(f32_tile, batch_size - t0);
                if (f32_f16score) {
                    const int64_t qe = tn * Dh4;
                    step4_to_f16_pad_kernel<float><<<static_cast<unsigned>((qe + 255) / 256), 256>>>(
                        d_q_f + t0 * D, d_q_h, tn, static_cast<int>(D), static_cast<int>(Dh4));
                    cublasStatus_t st = cublasGemmEx(f16_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        static_cast<int>(n_centroids), static_cast<int>(tn), static_cast<int>(Dh4), &alpha,
                        d_cent_h, CUDA_R_16F, static_cast<int>(Dh4),
                        d_q_h, CUDA_R_16F, static_cast<int>(Dh4), &beta,
                        d_f16_dots, CUDA_R_32F, static_cast<int>(n_centroids),
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    if (st != CUBLAS_STATUS_SUCCESS)
                        throw std::runtime_error("batch_assign: fp16 scoring GEMM failed, status=" + std::to_string(st));
                    const int64_t threads = tn * 32;
                    step4_top4_rerank_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(
                        d_f16_dots, d_cnorm, d_c_f, d_q_f + t0 * D, tn, static_cast<int>(n_centroids),
                        static_cast<int>(D), d_top2_u32 + t0 * TOP_K, d_top2_d ? d_top2_d + t0 * TOP_K : nullptr);
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }
                cublasStatus_t st = cublasSgemm(f16_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(n_centroids), static_cast<int>(tn), static_cast<int>(D), &alpha,
                    d_c_f, static_cast<int>(D), d_q_f + t0 * D, static_cast<int>(D), &beta,
                    d_f16_dots, static_cast<int>(n_centroids));
                if (st != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("batch_assign: fp32 GEMM failed, status=" + std::to_string(st));
                const int64_t threads = tn * 32;
                if (std::getenv("BUCKET2_F32_RERANK_REF"))   // 验证用: SGEMM 分数 + 同一个精确重算
                    step4_top4_rerank_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(
                        d_f16_dots, d_cnorm, d_c_f, d_q_f + t0 * D, tn, static_cast<int>(n_centroids),
                        static_cast<int>(D), d_top2_u32 + t0 * TOP_K, d_top2_d ? d_top2_d + t0 * TOP_K : nullptr);
                else
                step4_top2_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256>>>(
                    d_f16_dots, d_cnorm, tn, static_cast<int>(n_centroids), d_top2_u32 + t0 * TOP_K,
                    d_q_f + t0 * D, static_cast<int>(D), d_top2_d ? d_top2_d + t0 * TOP_K : nullptr);
                CUDA_CHECK(cudaGetLastError());
            }
            searched_f16 = true;   // 跟 fp16 路径共用下载 + 越界检查
        }
        if constexpr (!is_pq && !kExactF16) {
            if (use_bf && !exact_f32) {
                cuvs::neighbors::brute_force::search(res, *bf_idx,
                    raft::make_device_matrix_view<const float, int64_t>(
                        d_queries, batch_size, static_cast<int64_t>(D)),
                    raft::make_device_matrix_view<int64_t, int64_t>(
                        d_top2_i64, batch_size, static_cast<int64_t>(TOP_K)),
                    distances_view);
                searched_bf = true;
            }
        }
        if (!searched_bf && !searched_f16) {
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
        } else if (searched_f16) {
            CUDA_CHECK(cudaMemcpy(h_top2.data(), d_top2_u32,
                                  batch_size * TOP_K * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
            // bounds check
            int64_t n_bad = 0;
            #pragma omp parallel for schedule(static) reduction(+:n_bad)
            for (int64_t i = 0; i < batch_size * static_cast<int64_t>(TOP_K); ++i)
                if (h_top2[static_cast<size_t>(i)] >= static_cast<uint64_t>(n_centroids)) { ++n_bad; h_top2[static_cast<size_t>(i)] = 0; }
            if (n_bad > 0)
                throw std::runtime_error("batch_assign: fp16 top-2 returned " +
                                         std::to_string(n_bad) + " out-of-range centroid ids");
        } else {
            CUDA_CHECK(cudaMemcpy(h_top2.data(), d_top2,
                                  batch_size * TOP_K * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
        }
        if (rule_dj) {   // 精确 top-2 路径: kernel 写的真实距离；CAGRA / brute force: distances_view (平方 L2)
            h_top2_d.resize(static_cast<size_t>(batch_size) * TOP_K);
            CUDA_CHECK(cudaMemcpy(h_top2_d.data(), searched_f16 ? d_top2_d : d_distances,
                                  batch_size * TOP_K * sizeof(float), cudaMemcpyDeviceToHost));
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
            if (rule_dj) {
                const float d1 = h_top2_d[static_cast<size_t>(i) * TOP_K], d2 = h_top2_d[static_cast<size_t>(i) * TOP_K + 1];
                chosen = (s2 < s1 && d2 < 1.2f * d1) ? local_idx2 : local_idx1;   // DiskJoin add2choice
            } else if (static_cast<double>(s2) < 0.8 * static_cast<double>(s1)) {
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
    if (d_top2_d)     CUDA_CHECK(cudaFree(d_top2_d));
    if (d_top2_i64)   CUDA_CHECK(cudaFree(d_top2_i64));
    if (d_cent_h)     CUDA_CHECK(cudaFree(d_cent_h));
    if (d_cnorm)      CUDA_CHECK(cudaFree(d_cnorm));
    if (d_q_h)        CUDA_CHECK(cudaFree(d_q_h));
    if (d_f16_dots)   CUDA_CHECK(cudaFree(d_f16_dots));
    if (d_top2_u32)   CUDA_CHECK(cudaFree(d_top2_u32));
    if (f16_cublas)   cublasDestroy(f16_cublas);
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
    int64_t chunk_bytes_budget = 512LL * 1024 * 1024,
    // 非空时：(N, D) 原始数据已在 host 内存 (当前坐标系)，不再读盘
    const DataT* host_raw = nullptr)
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
    if (host_raw && bucket_vecs.in_memory) {
        // 内存模式下 add() 只碰 bucket c 自己的游标和字节区间，centroid c 恰好进 bucket c
        // → 各 c 互不相干，并行 (SIFT10M 10 万个 centroid 串行 0.2-0.4 s/轮)
        int64_t n_bad = 0;
        #pragma omp parallel for schedule(static) reduction(+:n_bad)
        for (int64_t c = 0; c < n_centroids; ++c) {
            const int64_t gi = centroid_global_indices[c];
            if (gi < 0 || gi >= N) { ++n_bad; continue; }   // bounds check
            bucket_vecs.add(c, gi, host_raw + static_cast<size_t>(gi) * D);
        }
        if (n_bad > 0) throw std::runtime_error("write_bucket_vectors: centroid id out of range");
    } else if (host_raw) {
        for (int64_t c = 0; c < n_centroids; ++c) {
            const int64_t gi = centroid_global_indices[c];
            if (gi < 0 || gi >= N)   // bounds check
                throw std::runtime_error("write_bucket_vectors: centroid id out of range");
            bucket_vecs.add(c, gi, host_raw + static_cast<size_t>(gi) * D);
        }
    } else {
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
    std::vector<char> is_centroid;   // 内存模式下代替 centroid_set 的逐点哈希查找
    std::vector<DataT> raw_chunk;    // 跨 chunk 复用 (每块重新分配会单线程清零一遍)
    for (int64_t lo = 0; lo < N; lo += chunk_points) {
        int64_t hi = std::min(lo + chunk_points, N) - 1;

        auto tr0 = Clock::now();
        const DataT* chunk_ptr = nullptr;
        if (host_raw) {
            chunk_ptr = host_raw + static_cast<size_t>(lo) * D;
        } else {
            int32_t N_tmp, D_tmp;
            load::read_bigann_raw_range<DataT>(input_path, lo, hi - lo + 1, raw_chunk, N_tmp, D_tmp);
            chunk_ptr = raw_chunk.data();
        }
        t_read += std::chrono::duration<double>(Clock::now() - tr0).count();

        auto tw0 = Clock::now();
        if (bucket_vecs.in_memory) {
            if (is_centroid.empty()) {
                is_centroid.assign(static_cast<size_t>(N), 0);
                for (int64_t g : centroid_global_indices) {
                    if (g < 0 || g >= N)   // bounds check
                        throw std::runtime_error("write_bucket_vectors: centroid id out of range");
                    is_centroid[static_cast<size_t>(g)] = 1;
                }
            }
            bucket_vecs.add_range_parallel(lo, hi - lo + 1, chunk_ptr, assignments, is_centroid);
        } else {
            for (int64_t gi = lo; gi <= hi; ++gi) {
                if (centroid_set.find(gi) != centroid_set.end()) continue;  // 上面已经写过
                bucket_vecs.add(assignments[static_cast<size_t>(gi)], gi,
                                 chunk_ptr + static_cast<size_t>(gi - lo) * D);
            }
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

// KMeans|| 的 d(x, C) (8-bit 数据): 跟 Step 4 一样的精确 fp16 Tensor Core GEMM
// (见 step4_f32_to_f16_kernel 处的论证: 8-bit 值在 fp16 精确，乘积和 < 2^24 在
// fp32 累加里精确) + 一 warp 一行求 min。L2 值跟 cuVS fp32 brute_force
// (L2Expanded: ‖x‖² + ‖c‖² − 2x·c, 负数截到 0) 逐位相同，所以 D² 采样结果不变，
// 只是 GEMM 走 Tensor Core (V100 fp32 brute force 是 CUDA core)。
__global__ void kmeans_row_min_kernel(const float* __restrict__ dots, const float* __restrict__ cnorm,
                                      const float* __restrict__ xnorm, int64_t rows, int n,
                                      float* __restrict__ out)
{
    const int64_t r = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* row = dots + r * n;
    float m = INFINITY;
    for (int j = lane; j < n; j += 32) m = fminf(m, cnorm[j] - 2.0f * row[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fminf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off));
    if (lane == 0) out[r] = fmaxf(xnorm[r] + m, 0.0f);
}

// GEMM + 逐行 min 融合 (Tensor Core WMMA)：原来 cuBLAS fp16 GEMM 把 tile × n 的
// dots (SIFT10M 子集 1M × 50K 新候选 = 200 GB fp32) 写回显存再由
// kmeans_row_min_kernel 读一遍，带宽占了大头；这里 dots 只活在寄存器 / 每
// warp 1 KB 的 shared scratch 里，直接归约成每行 min (cuVS 26.12
// detail/fused_l2_knn.cuh / fused distance+argmin 的思路, k=1)。
// block: 128 行 X × 每次 64 个候选，8 warp = 4 行组 (32 行) × 2 列组 (32 列)。
// 结果对每行 atomicMin (非负 float 的位序 = 数值序)，out 须先置 +inf。
constexpr int kFmBM = 128, kFmBN = 64, kFmLD = 136;   // LD: 128 + 8 half, 避 bank 冲突
constexpr size_t kFmSmem = (static_cast<size_t>(kFmBM + kFmBN) * kFmLD) * sizeof(__half)
                         + 8 * 256 * sizeof(float);
__global__ void __launch_bounds__(256) kmeans_fused_min_kernel(
    const __half* __restrict__ X, const float* __restrict__ xn, int64_t N,
    const __half* __restrict__ C, const float* __restrict__ cn, int n, int D,
    unsigned* __restrict__ out)
{
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char fm_smem[];
    __half* sA = reinterpret_cast<__half*>(fm_smem);
    __half* sB = sA + kFmBM * kFmLD;
    float* scratch = reinterpret_cast<float*>(sB + kFmBN * kFmLD);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wr = warp >> 1, wc = warp & 1;
    float* sc = scratch + warp * 256;
    const int64_t row0 = static_cast<int64_t>(blockIdx.x) * kFmBM;
    const int v8 = D >> 3;                 // 每行 uint4 数
    const int ksteps = D >> 4;

    for (int idx = tid; idx < kFmBM * v8; idx += 256) {
        const int r = idx / v8, c = idx - r * v8;
        const int64_t gr = row0 + r;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gr < N) v = reinterpret_cast<const uint4*>(X + gr * D)[c];
        reinterpret_cast<uint4*>(sA + r * kFmLD)[c] = v;
    }
    // 下一块候选先读进寄存器 (sm70 没有 cp.async)，算当前块时预取
    uint4 pre[4];
    auto load_b = [&](int j0) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + q * 256;
            pre[q] = make_uint4(0, 0, 0, 0);
            if (idx < kFmBN * v8) {
                const int r = idx / v8, c = idx - r * v8;
                const int64_t gc = static_cast<int64_t>(j0) + r;
                if (gc < n) pre[q] = reinterpret_cast<const uint4*>(C + gc * D)[c];
            }
        }
    };
    auto store_b = [&]() {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int idx = tid + q * 256;
            if (idx < kFmBN * v8) {
                const int r = idx / v8, c = idx - r * v8;
                reinterpret_cast<uint4*>(sB + r * kFmLD)[c] = pre[q];
            }
        }
    };

    float rmin[2] = {INFINITY, INFINITY};
    const int jstep = gridDim.y * kFmBN;
    int j0 = blockIdx.y * kFmBN;
    if (j0 < n) load_b(j0);
    for (; j0 < n; j0 += jstep) {
        __syncthreads();                   // 上一块的 sB 用完
        store_b();
        __syncthreads();
        if (j0 + jstep < n) load_b(j0 + jstep);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
        for (int k = 0; k < ksteps; ++k) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(fa[i], sA + (wr * 32 + i * 16) * kFmLD + k * 16, kFmLD);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(fb[j], sB + (wc * 32 + j * 16) * kFmLD + k * 16, kFmLD);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        const int r = lane & 15, h = lane >> 4;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::store_matrix_sync(sc, acc[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                const int cb = j0 + wc * 32 + j * 16 + h * 8;
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int col = cb + c;
                    if (col < n) rmin[i] = fminf(rmin[i], cn[col] - 2.0f * sc[r * 16 + h * 8 + c]);
                }
                __syncwarp();
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float m = fminf(rmin[i], __shfl_xor_sync(0xFFFFFFFFu, rmin[i], 16));
        const int64_t row = row0 + wr * 32 + i * 16 + lane;
        if (lane < 16 && row < N && m < INFINITY)
            atomicMin(out + row, __float_as_uint(fmaxf(xn[row] + m, 0.0f)));
    }
}

__global__ void fill_u32_kernel(unsigned* p, int64_t n, unsigned v)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

struct KmeansMinDistF16 {
    static constexpr int64_t kTile = 4096;
    cublasHandle_t h = nullptr;
    cudaStream_t stream = nullptr;
    int64_t N = 0; int D = 0;
    __half* d_xh = nullptr; float* d_xn = nullptr;
    __half* d_ch = nullptr; float* d_cn = nullptr; int64_t c_cap = 0;
    float* d_dots = nullptr; int64_t dots_cap = 0;   // 元素数
    bool fused = false;                               // kmeans_fused_min_kernel (D % 16 == 0, D ≤ 128)

    KmeansMinDistF16(const float* d_x, int64_t N_, int D_, cudaStream_t st) : stream(st), N(N_), D(D_) {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("KmeansMinDistF16: cublasCreate failed");
        cublasSetStream(h, stream);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        const int64_t nd = N * D;
        CUDA_CHECK(cudaMalloc(&d_xh, nd * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_xn, N * sizeof(float)));
        step4_f32_to_f16_kernel<<<static_cast<unsigned>((nd + 255) / 256), 256, 0, stream>>>(d_x, d_xh, nd);
        step4_row_norms_kernel<<<static_cast<unsigned>((N + 255) / 256), 256, 0, stream>>>(d_x, d_xn, N, D);
        CUDA_CHECK(cudaGetLastError());
        fused = (D % 16 == 0) && D <= 128 && !std::getenv("BUCKET2_KMEANS_NO_FUSED");
        if (fused)
            CUDA_CHECK(cudaFuncSetAttribute(kmeans_fused_min_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFmSmem)));
    }
    ~KmeansMinDistF16() {
        cudaFree(d_xh); cudaFree(d_xn); cudaFree(d_ch); cudaFree(d_cn); cudaFree(d_dots);
        if (h) cublasDestroy(h);
    }
    KmeansMinDistF16(const KmeansMinDistF16&) = delete;
    KmeansMinDistF16& operator=(const KmeansMinDistF16&) = delete;

    // d_min[i] = min_j ‖x_i − c_j‖²，C 是 (n, D) 的 device float
    void run(const float* d_c, int64_t n, float* d_min) {
        if (n <= 0) throw std::runtime_error("KmeansMinDistF16: empty candidate set");
        if (n > c_cap) {
            cudaFree(d_ch); cudaFree(d_cn);
            CUDA_CHECK(cudaMalloc(&d_ch, n * D * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(&d_cn, n * sizeof(float)));
            c_cap = n;
        }
        // dots 缓冲 ≤ ~1 GB: 候选多时 (SIFT100M 1M centroids → 每轮 ~500K 新候选)
        // 4096 × n × 4 B 会到 8 GB，跟数据集本身一起把显存撑爆；SIFT1M/10M 不变
        const int64_t tile = std::min<int64_t>(std::min<int64_t>(kTile, N),
                                               std::max<int64_t>(256, (int64_t(1) << 30) / (n * 4)));
        if (!fused && tile * n > dots_cap) {
            cudaFree(d_dots);
            CUDA_CHECK(cudaMalloc(&d_dots, tile * n * sizeof(float)));
            dots_cap = tile * n;
        }
        step4_f32_to_f16_kernel<<<static_cast<unsigned>((n * D + 255) / 256), 256, 0, stream>>>(d_c, d_ch, n * D);
        step4_row_norms_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(d_c, d_cn, n, D);
        CUDA_CHECK(cudaGetLastError());
        if (fused) {
            unsigned* o = reinterpret_cast<unsigned*>(d_min);
            fill_u32_kernel<<<static_cast<unsigned>((N + 255) / 256), 256, 0, stream>>>(o, N, 0x7f800000u);
            const unsigned gx = static_cast<unsigned>((N + kFmBM - 1) / kFmBM);
            // 行块不够铺满 SM 时把候选再切几份 (各自 atomicMin)
            const unsigned gy = static_cast<unsigned>(std::max<int64_t>(1, std::min<int64_t>(
                (n + kFmBN - 1) / kFmBN, (4 * 80 + gx - 1) / gx)));
            kmeans_fused_min_kernel<<<dim3(gx, gy), 256, kFmSmem, stream>>>(
                d_xh, d_xn, N, d_ch, d_cn, static_cast<int>(n), D, o);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const float alpha = 1.0f, beta = 0.0f;
        for (int64_t t0 = 0; t0 < N; t0 += tile) {
            const int64_t tn = std::min(tile, N - t0);
            // dots (tn, n) row-major = x_tile · Cᵀ
            cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(tn), D, &alpha,
                d_ch, CUDA_R_16F, D, d_xh + t0 * D, CUDA_R_16F, D, &beta,
                d_dots, CUDA_R_32F, static_cast<int>(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("KmeansMinDistF16: fp16 GEMM failed, status=" + std::to_string(st));
            const int64_t threads = tn * 32;
            kmeans_row_min_kernel<<<static_cast<unsigned>((threads + 255) / 256), 256, 0, stream>>>(
                d_dots, d_cn, d_xn + t0, tn, static_cast<int>(n), d_min + t0);
            CUDA_CHECK(cudaGetLastError());
        }
    }
};

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
    float** d_centroids_f32,
    bool exact8 = false)   // 数据是 8-bit (uint8/int8) 且没 PQ: d(x, C) 走精确 fp16 GEMM
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
        // 预留期望的最终候选数 (~TOTAL_OVERSHOOT × n_centroids, 加 20% 余量)：逐轮
        // resize 时新旧两份同时存在，1M centroids (SIFT100M) 要 ~1.5 GB，超过
        // RMM 池 2 GB 上限；一次 reserve 之后 resize 不再搬家
        centroidsBuf.reserve(static_cast<size_t>(std::min<double>(
            static_cast<double>(working_N), TOTAL_OVERSHOOT * 1.2 * n_centroids) + 1) * D, stream);
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

        std::unique_ptr<KmeansMinDistF16> f16_mcd;
        if (exact8 && !mem_est.need_pq)
            f16_mcd = std::make_unique<KmeansMinDistF16>(dataset_gpu.data_handle(), working_N, D, stream);
        auto min_dist = [&](raft::device_matrix_view<const float, int64_t> C, float* out) {
            if (f16_mcd) f16_mcd->run(C.data_handle(), C.extent(0), out);
            else min_cluster_distance(res, X_view, C, out);
        };

        // Step 2: Compute initial cost psi = phi_X(C)
        min_dist(raft::make_const_mdspan(potentialCentroids), minClusterDistVec.data_handle());
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
            // d(x, C) 是增量维护的 (cuVS 26.12 cluster/detail/kmeans.cuh::
            // initScalableKMeansPlusPlus 同样只对新候选算距离再逐点取 min)：进入
            // 本轮时 minClusterDistVec 已经是到全部已有候选的最小距离 (第 0 轮 =
            // 上面对初始点算的那份，之后每轮末尾对新候选更新)，不再每轮对全部
            // 候选 (1 → ~5K → ~10K) 重算。

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

            // 下一轮要用的 d(x, C): 只对本轮新候选算一遍，跟已有的逐点取 min
            // (uniformRands 下一轮开头才重新生成，这里借来当临时 buffer)；最后一轮
            // 之后不再需要
            if (iter + 1 < niter) {
                min_dist(raft::make_device_matrix_view<const float, int64_t>(
                             CpRaw.data(), n_new, static_cast<int64_t>(D)),
                         uniformRands.data_handle());
                thrust::transform(thrust::cuda::par.on(stream),
                    thrust::device_pointer_cast(minClusterDistVec.data_handle()),
                    thrust::device_pointer_cast(minClusterDistVec.data_handle()) + working_N,
                    thrust::device_pointer_cast(uniformRands.data_handle()),
                    thrust::device_pointer_cast(minClusterDistVec.data_handle()),
                    thrust::minimum<float>());
            }

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
// dist[r][j] = ‖c_j‖² − 2·dot[r][j] (行常数 ‖c_r‖² 不影响排序)；自己 (j == row0 + r) 置 +inf
__global__ void centroid_self_dist_kernel(float* __restrict__ dots, const float* __restrict__ cn,
                                          int64_t rows, int64_t n, int64_t row0)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const int64_t r = i / n, j = i - r * n;
    dots[i] = (j == row0 + r) ? INFINITY : cn[j] - 2.0f * dots[i];
}

std::vector<uint32_t> centroid_self_knn_bf(const float* d_centroids_f32, int64_t n_centroids,
                                           int64_t D, uint32_t k_out, bool exact8 = false)
{
    if (static_cast<int64_t>(k_out) >= n_centroids)
        throw std::runtime_error("centroid_self_knn_bf: k_out must be < n_centroids");
    // 8-bit 数据: centroid 就是数据点 (整数值)，fp16 表示精确、fp32 累加
    // (D·255² < 2^24) 也精确 → Tensor Core fp16 GEMM 得到跟 fp32 一样的距离
    // (同 step4_top2_kernel / KmeansMinDistF16)。分块 GEMM + 原地转距离 (自己
    // 置 +inf) + select_k，即 cuVS detail/knn_brute_force.cuh 的结构，只是 GEMM
    // 换成 fp16 Tensor Core (V100 fp32 brute force 走 CUDA core)。
    if (exact8 && D <= 128) {
        const int64_t n = n_centroids;
        cudaStream_t st = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        cublasHandle_t h = nullptr;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("centroid_self_knn_bf: cublasCreate failed");
        cublasSetStream(h, st);
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        const int64_t tile = std::max<int64_t>(1, std::min<int64_t>(n, (512ll << 20) / (n * 4)));
        __half* d_ch = nullptr; float* d_cn = nullptr; float* d_dots = nullptr;
        float* d_sd = nullptr; int32_t* d_si = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ch, n * D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_cn, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dots, tile * n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_sd, tile * k_out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_si, n * k_out * sizeof(int32_t)));
        step4_f32_to_f16_kernel<<<static_cast<unsigned>((n * D + 255) / 256), 256, 0, st>>>(d_centroids_f32, d_ch, n * D);
        step4_row_norms_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, st>>>(d_centroids_f32, d_cn, n, static_cast<int>(D));
        CUDA_CHECK(cudaGetLastError());
        raft::resources res;
        raft::resource::set_cuda_stream(res, st);
        const float alpha = 1.0f, beta = 0.0f;
        for (int64_t t0 = 0; t0 < n; t0 += tile) {
            const int64_t tn = std::min(tile, n - t0);
            cublasStatus_t cs = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(tn), static_cast<int>(D), &alpha,
                d_ch, CUDA_R_16F, static_cast<int>(D), d_ch + t0 * D, CUDA_R_16F, static_cast<int>(D), &beta,
                d_dots, CUDA_R_32F, static_cast<int>(n),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (cs != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("centroid_self_knn_bf: fp16 GEMM failed, status=" + std::to_string(cs));
            const int64_t tot = tn * n;
            centroid_self_dist_kernel<<<static_cast<unsigned>((tot + 255) / 256), 256, 0, st>>>(d_dots, d_cn, tn, n, t0);
            CUDA_CHECK(cudaGetLastError());
            raft::matrix::select_k<float, int32_t>(res,
                raft::make_device_matrix_view<const float, int64_t, raft::row_major>(d_dots, tn, n),
                std::nullopt,
                raft::make_device_matrix_view<float, int64_t, raft::row_major>(d_sd, tn, k_out),
                raft::make_device_matrix_view<int32_t, int64_t, raft::row_major>(d_si + t0 * k_out, tn, k_out),
                /*select_min=*/true, /*sorted=*/true);
        }
        std::vector<int32_t> h_si(static_cast<size_t>(n) * k_out);
        CUDA_CHECK(cudaMemcpyAsync(h_si.data(), d_si, h_si.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        cudaFree(d_ch); cudaFree(d_cn); cudaFree(d_dots); cudaFree(d_sd); cudaFree(d_si);
        cublasDestroy(h);
        cudaStreamDestroy(st);
        std::vector<uint32_t> out(h_si.size());
        int64_t n_bad = 0;
        #pragma omp parallel for schedule(static) reduction(+:n_bad)
        for (int64_t c = 0; c < n; ++c) {
            for (uint32_t j = 0; j < k_out; ++j) {
                const int32_t v = h_si[static_cast<size_t>(c) * k_out + j];
                if (v < 0 || v >= n || v == c) { ++n_bad; out[static_cast<size_t>(c) * k_out + j] = 0; continue; }
                out[static_cast<size_t>(c) * k_out + j] = static_cast<uint32_t>(v);
            }
        }
        if (n_bad > 0)
            throw std::runtime_error("centroid_self_knn_bf: " + std::to_string(n_bad) + " bad ids (fp16 path)");
        return out;
    }
    const int64_t ks = static_cast<int64_t>(k_out) + 1;   // 多取 1 个，剥掉自己
    raft::resources res;
    auto cview = raft::make_device_matrix_view<const float, int64_t>(d_centroids_f32, n_centroids, D);
    cuvs::neighbors::brute_force::index_params bp;
    bp.metric = cuvs::distance::DistanceType::L2Expanded;
    auto bf = cuvs::neighbors::brute_force::build(res, bp, cview);
    // query 分批: 输出 (n, ks) int64+float 在 1M centroids (SIFT100M) 时 2.4 GB，
    // 超过 RMM 池 2 GB 上限；只在输出 > 512 MB 时按 64K 行一批 (~160 MB)，结果跟
    // 一次搜完全一样 (SIFT1M/10M 的 10K/100K centroids 仍一次搜完，分批每轮多 ~0.15 s)
    constexpr int64_t kQB = 65536;
    const bool split_q = n_centroids * ks * static_cast<int64_t>(sizeof(int64_t) + sizeof(float)) > (int64_t(512) << 20);
    const int64_t qb = split_q ? std::min<int64_t>(kQB, n_centroids) : n_centroids;
    auto nb = raft::make_device_matrix<int64_t, int64_t>(res, qb, ks);
    auto ds = raft::make_device_matrix<float, int64_t>(res, qb, ks);
    std::vector<int64_t> h_nb(static_cast<size_t>(n_centroids * ks));
    for (int64_t q0 = 0; q0 < n_centroids; q0 += qb) {
        const int64_t qn = std::min(qb, n_centroids - q0);
        auto qv = raft::make_device_matrix_view<const float, int64_t>(d_centroids_f32 + q0 * D, qn, D);
        auto nbv = raft::make_device_matrix_view<int64_t, int64_t>(nb.data_handle(), qn, ks);
        auto dsv = raft::make_device_matrix_view<float, int64_t>(ds.data_handle(), qn, ks);
        cuvs::neighbors::brute_force::search(res, bf, qv, nbv, dsv);
        raft::copy(h_nb.data() + q0 * ks, nb.data_handle(), static_cast<size_t>(qn * ks),
                   raft::resource::get_cuda_stream(res));
        raft::resource::sync_stream(res);
    }

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

// DotT = float (fp16/fp32 GEMM) 或 int32 (out-of-core INT8 GEMM，|dot| ≤ 128*128*128
// < 2^24，转 float 精确)。kRowByQuery: 输出第 q 行 (out-of-core 每 bucket 一块
// 本地输出)；否则输出第 ids_pool[q] 行 (GPU-resident 全局表)。
template <int K_OUT, int G, typename DotT = float, bool kRowByQuery = false>
__global__ void __launch_bounds__(G * 32) block_topK_from_dots_kernel(
    const DotT*    __restrict__ dots,        // (bucket_size, ldd)，ldd 是 4 的倍数 (16 B 对齐)
    const float*   __restrict__ norms_pool,  // (pool_size,)  前 bucket_size 个即 ‖q‖²
    const int32_t* __restrict__ ids_pool,    // (pool_size,)  前 bucket_size 个即 query gid
    int32_t* __restrict__ out_neighbors,     // 第 gid 行、行宽 out_ld
    float*   __restrict__ out_dists,         // 同上，真实 L2²
    int pool_size, int ldd, int K, int out_ld,
    // 非空 (GPU-resident round 2..t): 跨轮 running top-M (行宽 out_ld，升序)。
    // 本轮候选的真实距离 ≥ 该点 running 第 M 个就进不了合并后的 top-M
    // (resident_merge_kernel 取两者并集去重后的前 M)，拿它当行阈值初值
    const float* __restrict__ seed_d = nullptr,
    // 非空 (out-of-core round 2..t): 按 gid 索引的 running 第 M 个距离 (ChunkedKnnAccumulator::kth)
    const float* __restrict__ seed_gid = nullptr, int64_t seed_n = 0,
    // 行分块 (out-of-core Step 6): dots 只装 bucket 的 [row0, row0 + gridDim.x) 行，q 仍是 bucket 内行号
    int row0 = 0)
{
    constexpr int LANES = 32;
    constexpr int K_PER_LANE = K_OUT / LANES;
    constexpr unsigned ALL = 0xFFFFFFFFu;
    const int q    = blockIdx.x + row0;
    const int warp = threadIdx.x / LANES;
    const int lane = threadIdx.x & (LANES - 1);

    float   reg_dist[K_PER_LANE];
    int32_t reg_id  [K_PER_LANE];
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) { reg_dist[p] = INFINITY; reg_id[p] = -1; }
    float threshold = INFINITY;

    // G 个 warp 共享阈值: 任何一个 warp 当前的第 K 小都是整行第 K 小的上界，
    // 比它大的候选不可能进最终 top-K。每个 warp 插入后把自己的阈值 atomicMin
    // 进 shared memory，每轮外循环取全 block 最小值过滤——其它 warp 不用各自从
    // inf 开始热身，串行插入次数大减 (同 #45 融合 kernel 里的跨段共享阈值)。
    // 距离 (‖p‖² - 2·dot) 可能为负，用保序的 float→uint 键比较。
    __shared__ unsigned s_th;
    auto f2key = [](float f) {
        unsigned u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };
    auto key2f = [](unsigned k) {
        return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
    };
    if (threadIdx.x == 0) {
        s_th = 0xFFFFFFFFu;
        float sk = INFINITY;
        if (seed_d) {
            const int64_t srow = static_cast<int64_t>(kRowByQuery ? q : ids_pool[q]);
            sk = seed_d[srow * out_ld + out_ld - 1];
        } else if (seed_gid) {
            const int64_t g = ids_pool[q];
            if (g >= 0 && g < seed_n) sk = seed_gid[g];   // bounds check
        }
        if (sk < INFINITY) s_th = f2key(sk - norms_pool[q]);   // kernel 内距离不含 ‖q‖²
    }
    __syncthreads();

    // 每 lane 每次读 4 个候选 (float4)，提高在途访存量——一个 bucket 只有
    // bucket_size 个 block，占用率天然不高，标量读时这个 kernel 是延迟瓶颈。
    const DotT* row = dots + static_cast<int64_t>(blockIdx.x) * ldd;   // dots 行是分块内行号
    for (int jb = warp * LANES * 4; jb < pool_size; jb += G * LANES * 4) {
        threshold = fminf(threshold, key2f(*reinterpret_cast<volatile unsigned*>(&s_th)));
        int j0 = jb + lane * 4;
        float dd[4];
        if (j0 + 3 < pool_size) {
            float4 dv;
            if constexpr (std::is_same<DotT, float>::value) {
                dv = *reinterpret_cast<const float4*>(row + j0);
            } else {
                int4 iv = *reinterpret_cast<const int4*>(row + j0);
                dv = make_float4(static_cast<float>(iv.x), static_cast<float>(iv.y),
                                 static_cast<float>(iv.z), static_cast<float>(iv.w));
            }
            float4 nv = *reinterpret_cast<const float4*>(norms_pool + j0);
            dd[0] = nv.x - 2.0f * dv.x; dd[1] = nv.y - 2.0f * dv.y;
            dd[2] = nv.z - 2.0f * dv.z; dd[3] = nv.w - 2.0f * dv.w;
        } else {
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                dd[e] = (j0 + e < pool_size) ? norms_pool[j0 + e] - 2.0f * static_cast<float>(row[j0 + e]) : INFINITY;
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
                const float own = __shfl_sync(ALL, reg_dist[K_PER_LANE - 1], LANES - 1);
                if (own < threshold) {
                    threshold = own;
                    if (lane == 0) atomicMin(&s_th, f2key(own));
                }
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

    // G 份各自升序的 top-K 列表做树形归并 (G → G/2 → … → 1)：每层每对列表由
    // 一个 warp 用 merge-path (co-rank 二分) 并行算出合并后的前 K_OUT 个，
    // 每 lane 负责 K_PER_LANE 个输出位置。原来是 warp 0 把其余 (G-1)*K_OUT 个
    // 候选逐个 ballot + 移位插入，串行 ~几百次插入，占了这个 kernel 的大头。
    // 同距离时 A (低编号 warp) 在前，结果集合跟原来一致，只有并列项的先后可能不同。
    for (int step = 1; step < G; step <<= 1) {
        float   o_dist[K_PER_LANE];
        int32_t o_id  [K_PER_LANE];
        const bool active = (warp % (2 * step) == 0) && (warp + step < G);
        if (active) {
            const float*   Ad = s_dist[warp];        const int32_t* Ai = s_id[warp];
            const float*   Bd = s_dist[warp + step]; const int32_t* Bi = s_id[warp + step];
            #pragma unroll
            for (int p = 0; p < K_PER_LANE; ++p) {
                const int k = lane * K_PER_LANE + p;   // 输出第 k 小 (0-based)
                // i = 前 k+1 个合并结果里来自 A 的个数
                int lo = max(0, k + 1 - K_OUT), hi = min(k + 1, K_OUT);
                while (lo < hi) {
                    int i = (lo + hi) >> 1;
                    int j = k + 1 - i;
                    // A 取 i 个太少 ⇔ B[j-1] 严格大于 A[i] (A 优先)
                    if (j > 0 && i < K_OUT && Ad[i] <= Bd[j - 1]) lo = i + 1;
                    else hi = i;
                }
                const int i = lo, j = k + 1 - i;
                // 第 k 个 = A[i-1] 与 B[j-1] 中较晚出现的那个
                // (相等时 B 在合并序里更靠后，所以取 B)
                const bool fromA = (j == 0) || (i > 0 && Ad[i - 1] > Bd[j - 1]);
                o_dist[p] = fromA ? Ad[i - 1] : Bd[j - 1];
                o_id[p]   = fromA ? Ai[i - 1] : Bi[j - 1];
            }
        }
        __syncthreads();
        if (active) {
            #pragma unroll
            for (int p = 0; p < K_PER_LANE; ++p) {
                s_dist[warp][lane * K_PER_LANE + p] = o_dist[p];
                s_id  [warp][lane * K_PER_LANE + p] = o_id[p];
            }
        }
        __syncthreads();
    }
    if (warp != 0) return;

    const int64_t out_row = static_cast<int64_t>(kRowByQuery ? q : ids_pool[q]) * out_ld;
    const float q_norm = norms_pool[q];
    int base = lane * K_PER_LANE;
    #pragma unroll
    for (int p = 0; p < K_PER_LANE; ++p) {
        if (base + p < K) {
            int32_t local = s_id[0][base + p];
            out_neighbors[out_row + base + p] = (local >= 0) ? ids_pool[local] : -1;
            out_dists[out_row + base + p]     = (local >= 0) ? s_dist[0][base + p] + q_norm : INFINITY;
        }
    }
}

// GPU-resident 路径的 top-M dispatcher；M > 256 时 shared memory 不够，回退到
// 一 warp 一 query 的 warp_topK_from_dots_kernel。
static inline void launch_block_topM_resident(
    cudaStream_t stream, const float* d_dots, const float* d_norms_pool, const int32_t* d_ids_pool,
    int32_t* d_out_neighbors, float* d_out_dists,
    int bucket_size, int pool_size, int ldd, int actual_M, int out_ld,
    const float* d_seed = nullptr, int row0 = 0)
{
    constexpr int G = 8;
    // bucket_size = 本次 dots 分块的行数 (bucket 的 [row0, row0 + bucket_size) 行)
#define LAUNCH_BLOCK_KOUT(KV) block_topK_from_dots_kernel<(KV), G><<<bucket_size, G * 32, 0, stream>>>( \
        d_dots, d_norms_pool, d_ids_pool, d_out_neighbors, d_out_dists, pool_size, ldd, actual_M, out_ld, d_seed, \
        nullptr, 0, row0)
    if      (actual_M <= 32)  LAUNCH_BLOCK_KOUT(32);
    else if (actual_M <= 64)  LAUNCH_BLOCK_KOUT(64);
    else if (actual_M <= 128) LAUNCH_BLOCK_KOUT(128);
    else if (actual_M <= 256) LAUNCH_BLOCK_KOUT(256);
    else
        throw std::runtime_error("GPU-resident path supports neighbors-m <= 256 (got "
                                 + std::to_string(actual_M) + "); use --out-of-core");
#undef LAUNCH_BLOCK_KOUT
}

// out-of-core 路径用的同一个 block top-M kernel：输出按 bucket 内行号 q 写 (行宽
// out_ld)，邻居 id 取 ids_pool (global id)，距离是真实 L2²——跟原来
// dots_to_dist + raft select_k + gather (3 个 kernel) 的输出一致。返回 false
// 表示 actual_M > 256 (shared memory 不够)，调用方回退到 raft select_k。
template <typename DotT>
static inline bool launch_block_topM_bucket(
    cudaStream_t stream, const DotT* d_dots, const float* d_norms_pool, const int32_t* d_ids_pool,
    int32_t* d_out_neighbors, float* d_out_dists,
    int rows, int pool_size, int ldd, int actual_M, int out_ld,
    const float* d_seed_gid = nullptr, int64_t seed_n = 0, int row0 = 0)
{
    constexpr int G = 8;
#define LAUNCH_BUCKET_KOUT(KV) block_topK_from_dots_kernel<(KV), G, DotT, true><<<rows, G * 32, 0, stream>>>( \
        d_dots, d_norms_pool, d_ids_pool, d_out_neighbors, d_out_dists, pool_size, ldd, actual_M, out_ld, \
        nullptr, d_seed_gid, seed_n, row0)
    if      (actual_M <= 32)  LAUNCH_BUCKET_KOUT(32);
    else if (actual_M <= 64)  LAUNCH_BUCKET_KOUT(64);
    else if (actual_M <= 128) LAUNCH_BUCKET_KOUT(128);
    else if (actual_M <= 256) LAUNCH_BUCKET_KOUT(256);
    else return false;
#undef LAUNCH_BUCKET_KOUT
    return true;
}

// ---------------- A1-R (用户批准的 A1，resident 路径): local join 精修 ----------------
// GNN-Descent 的 local join 换成 "拉" 的写法：点 a 的候选 = 它前 S 个邻居 u 各自的前 S 个邻居
// (邻居的邻居)，算精确距离，小于 a 当前第 M 近的才参与合并；新表写到另一份缓冲 (本趟所有点读的
// 都是上一趟的表)。TAGORE GNN-Descent 是把 u 的采样邻居两两配对推给对方 (nn_descent_opt_cal /
// _merge)；这里候选集合同为 2 跳邻居，按点独立计算，不需要 atomic 写别人的表。
constexpr int kA1MaxCand = 1024;   // S × S 上限 (S ≤ 32)
struct A1Counters { unsigned long long cand, pass, inserted; };

template <typename GemmT>
__global__ void __launch_bounds__(256) a1_pull_join_kernel(
    const GemmT* __restrict__ data, const float* __restrict__ norms, int64_t N, int D,
    const int32_t* __restrict__ in_n, const float* __restrict__ in_d, int M, int S,
    int32_t* __restrict__ out_n, float* __restrict__ out_d, A1Counters* __restrict__ ctr)
{
    const int64_t a = blockIdx.x;
    if (a >= N) return;
    __shared__ float s_a[256];                     // a 的向量 (D ≤ 256)
    __shared__ int32_t s_cid[kA1MaxCand + 256];    // 通过阈值的候选 + 原表
    __shared__ float   s_cd[kA1MaxCand + 256];
    __shared__ int32_t s_old[256];
    __shared__ int s_np;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int e = tid; e < D; e += blockDim.x) s_a[e] = static_cast<float>(data[a * D + e]);
    for (int k = tid; k < M; k += blockDim.x) s_old[k] = in_n[a * M + k];
    if (tid == 0) s_np = 0;
    __syncthreads();
    const float kth = in_d[a * M + M - 1];
    const float na = norms[a];
    const int nc = S * S;
    // 候选距离: 一个 warp 一个候选，lane 跨维
    for (int c = warp; c < nc; c += nw) {
        const int32_t u = s_old[c / S];
        int32_t v = (u >= 0 && u < N) ? in_n[static_cast<int64_t>(u) * M + (c % S)] : -1;
        if (v < 0 || v >= N || v == a) continue;   // bounds check
        bool dup = false;                           // 已在 a 的表里
        for (int k = lane; k < M; k += 32) dup |= (s_old[k] == v);
        if (__any_sync(0xffffffffu, dup)) continue;
        float part = 0.0f;
        const GemmT* pv = data + static_cast<int64_t>(v) * D;
        for (int e = lane; e < D; e += 32) part += s_a[e] * static_cast<float>(pv[e]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        const float d = na + norms[v] - 2.0f * part;
        if (lane == 0 && d < kth) {
            const int p = atomicAdd(&s_np, 1);
            if (p < kA1MaxCand) { s_cid[p] = v; s_cd[p] = d; }
        }
    }
    __syncthreads();
    const int np = min(s_np, kA1MaxCand);
    // 候选之间去重: 同一个 v 的距离是同一种算法算的，值相同；保留下标最小的那份
    for (int p = tid; p < np; p += blockDim.x) {
        const int32_t v = s_cid[p];
        for (int o = 0; o < p; ++o) if (s_cid[o] == v) { s_cd[p] = INFINITY; break; }
    }
    __syncthreads();
    for (int p = tid; p < np; p += blockDim.x) if (!(s_cd[p] < INFINITY)) s_cid[p] = INT_MAX;   // 重复的排到最后
    // 原表接在后面
    for (int k = tid; k < M; k += blockDim.x) { s_cid[np + k] = in_n[a * M + k]; s_cd[np + k] = in_d[a * M + k]; }
    __syncthreads();
    // 按 (距离, id) 排名，前 M 个写出 (无效 / 重复的是 INF，排在后面)
    const int tot = np + M;
    int ins = 0;
    for (int p = tid; p < tot; p += blockDim.x) {
        const float d = s_cd[p]; const int32_t id = s_cid[p];
        if (!(d < INFINITY) && p < np) continue;
        int rank = 0;
        for (int o = 0; o < tot; ++o) {
            const float od = s_cd[o]; const int32_t oid = s_cid[o];
            rank += (od < d) || (od == d && (oid < id || (oid == id && o < p)));
        }
        if (rank < M) {
            out_n[a * M + rank] = id; out_d[a * M + rank] = d;
            if (p < np) ++ins;
        }
    }
    if (ins) atomicAdd(&ctr->inserted, static_cast<unsigned long long>(ins));
    if (tid == 0) { atomicAdd(&ctr->cand, static_cast<unsigned long long>(nc)); atomicAdd(&ctr->pass, static_cast<unsigned long long>(np)); }
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

        // dots 按 query 行分块 (同 out-of-core Step 6 的 kDotsRowTile)：每个 slot 的 dots 不超过
        // kDotsSlotBytes。不分块时是 max_bucket × max_pool × 4 B / slot，大 bucket + 大 pool 时
        // 显存爆掉 (msTuring1M B 512 / NPROBE 256: 4643 × 133K × 4 B = 2.5 GB × 4 slot → OOM)。
        // 默认参数下一块就装下整个 bucket，跟原来完全一样。
        constexpr size_t kDotsSlotBytes = size_t(512) << 20;
        const int64_t max_ldd = (max_pool + 3) / 4 * 4;
        const int64_t row_tile = std::max<int64_t>(1, std::min<int64_t>(max_bucket,
            static_cast<int64_t>(kDotsSlotBytes / (sizeof(float) * std::max<int64_t>(1, max_ldd)))));
        for (auto& s : slots) {
            ensure_device(s.pool, s.pool_cap, static_cast<size_t>(max_pool) * D);
            ensure_device(s.pool_norms, s.norms_cap, static_cast<size_t>(max_pool));
            ensure_device(s.pool_ids, s.ids_cap, static_cast<size_t>(max_pool));
            ensure_device(s.dots, s.dots_cap, static_cast<size_t>(row_tile) * max_ldd);
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

            for (int r0 = 0; r0 < bucket_size; r0 += static_cast<int>(row_tile)) {
                const int rows = std::min(static_cast<int>(row_tile), bucket_size - r0);
                // dots (rows, ps) row-major = pool[r0:r0+rows] · pool^T
                float alpha = 1.0f, beta = 0.0f;
                cublasStatus_t stat;
                if constexpr (kHalf) {
                    stat = cublasGemmEx(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                        ps, rows, D, &alpha,
                                        s.pool, CUDA_R_16F, D,
                                        s.pool + static_cast<size_t>(r0) * D, CUDA_R_16F, D, &beta,
                                        s.dots, CUDA_R_32F, ldd,
                                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                } else {
                    stat = cublasSgemm(s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                       ps, rows, D, &alpha,
                                       s.pool, D, s.pool + static_cast<size_t>(r0) * D, D, &beta, s.dots, ldd);
                }
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("GpuResidentKnn: GEMM failed, status=" + std::to_string(stat));

                // pool 前 bucket_size 行就是 query 自己: ids / ‖q‖² 都直接取 pool 的前缀 (第 r0 + 块内行)
                launch_block_topM_resident(s.stream, s.dots, s.pool_norms, s.pool_ids,
                                           d_cur_n, d_cur_d, rows, ps, ldd, actual_M, /*out_ld=*/M,
                                           has_run ? d_run_d : nullptr, r0);
                CUDA_CHECK(cudaGetLastError());
            }
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
                  << " max_bucket=" << max_bucket << " row_tile=" << row_tile << " gemm=" << (kHalf ? "FP16-TC" : "FP32")
                  << " | setup=" << std::fixed << std::setprecision(3) << t_setup
                  << "s loop=" << t_loop << "s (host launch " << t_launch
                  << "s) merge=" << t_merge << "s\n";
    }

    // 最终结果只下载一次，三个文件并行写 (npy_path 为空则不写 .npy)
    // after_download (可选): D2H 完成、写盘线程已经起来之后在主线程上调用，拿到
    // host 上的 (N, M) 邻居表——GPU-resident 收尾 (prune/反向边/id 翻译) 跟
    // vector_knn.bin 的写盘重叠。回调里可以释放本对象的显存 (之后不再访问显存)。
    void write_outputs(const std::string& knn_path, const std::string& dist_path,
                       const std::string& npy_path,
                       const std::function<void(const std::vector<int32_t>&)>& after_download = {}) const {
        size_t nm = static_cast<size_t>(N) * M;
        std::vector<int32_t> h_n(nm);
        std::vector<float>   h_d(dist_path.empty() ? 0 : nm);   // dist_path 为空 = 不写 vector_dists
        CUDA_CHECK(cudaMemcpy(h_n.data(), d_run_n, nm * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (!dist_path.empty())
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
        auto f_dist = std::async(std::launch::async, [&] {
            if (!dist_path.empty()) write_bin(dist_path, h_d.data(), nm * sizeof(float));
        });
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
        if (after_download) after_download(h_n);
        f_knn.get();
        f_dist.get();
    }

    // 释放全部显存 (收尾阶段把显存让给 cuVS optimize)；之后对象只能析构
    // A1-R: passes 趟 local join (S = 每点取前 S 个邻居、每个邻居再取前 S 个)，打印 A1_stats
    void refine_local_join(int passes, int S) {
        if (passes <= 0 || !has_run) return;
        if (D > 256 || S < 1 || S > 32 || S > M) throw std::runtime_error("A1-R: needs D <= 256 and 1 <= S <= min(32, M)");
        A1Counters* d_ctr = nullptr;
        CUDA_CHECK(cudaMalloc(&d_ctr, sizeof(A1Counters)));
        for (int it = 0; it < passes; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemset(d_ctr, 0, sizeof(A1Counters)));
            a1_pull_join_kernel<GemmT><<<static_cast<unsigned>(N), 256>>>(
                d_data, d_norms, N, D, d_run_n, d_run_d, M, S, d_tmp_n, d_tmp_d, d_ctr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::swap(d_run_n, d_tmp_n); std::swap(d_run_d, d_tmp_d);
            A1Counters h{};
            CUDA_CHECK(cudaMemcpy(&h, d_ctr, sizeof(A1Counters), cudaMemcpyDeviceToHost));
            const double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "A1_stats: path=resident variant=pull2hop pass=" << it + 1 << " sample=" << S
                      << " points=" << N << " dist_computed_avg=" << static_cast<double>(h.cand) / N
                      << " cand_pass_avg=" << static_cast<double>(h.pass) / N
                      << " inserted_avg=" << static_cast<double>(h.inserted) / N
                      << " seconds=" << std::fixed << std::setprecision(3) << s << "\n";
        }
        cudaFree(d_ctr);
    }

    void release_device() {
        for (auto& s : slots) {
            cudaFree(s.pool); cudaFree(s.pool_norms); cudaFree(s.pool_ids); cudaFree(s.dots);
            s.pool = nullptr; s.pool_norms = nullptr; s.pool_ids = nullptr; s.dots = nullptr;
            s.pool_cap = s.norms_cap = s.ids_cap = s.dots_cap = 0;
        }
        cudaFree(d_members); cudaFree(d_member_offsets); cudaFree(d_seg_bucket); cudaFree(d_seg_ofs);
        d_members = nullptr; d_member_offsets = nullptr; d_seg_bucket = nullptr; d_seg_ofs = nullptr;
        members_cap = offsets_cap = seg_bucket_cap = seg_ofs_cap = 0;
        cudaFree(d_data); cudaFree(d_norms);
        cudaFree(d_run_n); cudaFree(d_run_d); cudaFree(d_cur_n); cudaFree(d_cur_d);
        cudaFree(d_tmp_n); cudaFree(d_tmp_d);
        d_data = nullptr; d_norms = nullptr;
        d_run_n = d_cur_n = d_tmp_n = nullptr; d_run_d = d_cur_d = d_tmp_d = nullptr;
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
    // arrival 文件里的变长记录: [int64 gid][uint32 cnt][cnt × int32 nbr][cnt × float dist]，
    // cnt = 最后一个有效 (nbr ≥ 0) 位置 + 1。后面全是 -1/+inf sentinel，merge 读回时补齐成 M 宽，
    // 合并结果跟定长记录完全一样。round 2..t 的 top-M 用 running 第 M 个距离做阈值，大部分行
    // 只剩少数有效候选 → 写盘量大减 (SIFT100M round 2 的 scatter 卡在 ~250 MB/s 的盘写上)。
    static constexpr size_t kVarHdr = sizeof(int64_t) + sizeof(uint32_t);
    static size_t var_record_bytes(uint32_t cnt) {
        return kVarHdr + static_cast<size_t>(cnt) * (sizeof(int32_t) + sizeof(float));
    }
    uint64_t arrival_bytes = 0;   // 本轮写出的 arrival 字节 (统计；只在写 arrival 的线程上累加)

    struct ChunkBuf {
        std::vector<char> data;     // ≥ 一条最长的变长记录
        size_t capacity_records = 0;
        size_t count = 0;           // data 里已用的字节数
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
    // 倒数第二轮的 merge 只算 kth (下一轮 top-M 的阈值)，不写 running 文件，arrival 文件留着给最终
    // merge 再合一遍 (结果完全一样)：省掉一整份 running (N·M·8 B，SIFT100M 51 GB) 的写盘——它在下一轮
    // Step 6 期间还在回写，把 scatter 的 arrival 写卡在 ~250 MB/s 的盘上。参考 TAGORE order_merge_new:
    // 子图留在缓冲/磁盘里，最后一趟才统一合并输出。
    int deferred_iter = -1;
    int planned_final_iter = -1;   // 最后一轮的 iter (每次 merge 前由调用方设置)
    bool final_written = false;
    // 非最终轮 merge 完之后每个点 running 第 M 个距离 (真实 L2²，不足 M 个为 inf)，
    // 下一轮 Step 6 拿来当 block top-M 的行阈值初值 (见 #61/#62)
    // merge 线程写 kth_work，整轮写完后在 kth_mtx 下换成 kth_done (最近一次完成的
    // 那份)。Step 6 只读 kth_done、不等 merge: 本轮还没完成就用更早一轮的——
    // running 第 M 个距离只会越来越小，旧的仍是合法上界 (只是松一些)。
    std::vector<float> kth_work, kth_done;
    std::unique_ptr<std::mutex> kth_mtx = std::make_unique<std::mutex>();   // unique_ptr: 对象要能 move
    bool kth_valid = false;   // kth_done 里有数据 (guarded by kth_mtx)
    // 拷一份最近完成的 kth (没有则返回 false)
    bool snapshot_kth(std::vector<float>& out) {
        std::lock_guard<std::mutex> lk(*kth_mtx);
        if (!kth_valid) return false;
        out = kth_done;
        return true;
    }
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
        // 至少切成 desired_parallelism 个 chunk: merge 按 chunk 并行，chunk 数
        // 少于线程数时只用得上 num_chunks 个线程 (SIFT1M: 预算推出 10 个 chunk，
        // 80 核只用 10 个，最后一轮 merge 的等待 ~0.9 s 全露在外面)。chunk 更小
        // 只会让每个 merge 线程的常驻内存更少，不改变内存上限。
        chunk_size_from_mem = std::min<int64_t>(chunk_size_from_mem,
            std::max<int64_t>(1, (N + desired_parallelism - 1) / desired_parallelism));

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
            b.data.resize(std::max(cap_records * rec_bytes, var_record_bytes(static_cast<uint32_t>(M))));
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
        b.file.write(b.data.data(), static_cast<std::streamsize>(b.count));
        arrival_bytes += b.count;
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
        uint32_t cnt = static_cast<uint32_t>(M);
        while (cnt > 0 && nbrs[cnt - 1] < 0) --cnt;
        const size_t need = var_record_bytes(cnt);
        if (b.count + need > b.data.size()) flush_chunk(c);
        char* dst = b.data.data() + b.count;
        std::memcpy(dst, &gid, sizeof(int64_t));
        std::memcpy(dst + sizeof(int64_t), &cnt, sizeof(uint32_t));
        std::memcpy(dst + kVarHdr, nbrs, static_cast<size_t>(cnt) * sizeof(int32_t));
        std::memcpy(dst + kVarHdr + static_cast<size_t>(cnt) * sizeof(int32_t), dists,
                    static_cast<size_t>(cnt) * sizeof(float));
        b.count += need;
    }

    void finish_iteration_writes() {
        for (int64_t c = 0; c < num_chunks; ++c) {
            flush_chunk(c);
            bufs[static_cast<size_t>(c)].file.close();
        }
        std::cout << "  [ChunkedKNN] arrival records written: " << arrival_bytes / 1e9 << " GB (fixed-width would be "
                  << static_cast<double>(N) * record_bytes(M) / 1e9 << " GB)\n";
        arrival_bytes = 0;
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
        const bool defer = !is_final && planned_final_iter >= 0 && iter + 1 == planned_final_iter;
        const int kept_iter = (deferred_iter == iter - 1) ? iter - 1 : -1;   // 上一轮留下的 arrival
        // running 文件里是第 base_iter 轮 merge 后的状态 (没有 = 全 sentinel)
        const int base_iter = (kept_iter >= 0) ? kept_iter - 1 : iter - 1;
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
            if (!final_dist_path.empty()) fd_dist = open_out(final_dist_path);   // 空 = 不写 vector_dists
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

        if (!is_final && kth_work.size() != static_cast<size_t>(N)) kth_work.resize(static_cast<size_t>(N));
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
                if (base_iter < 0) {
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
                // 先合上一轮留下的 arrival (kept_iter)，再合这一轮的。
                size_t n_records = 0;
                auto t3 = now();
                for (int ai : {kept_iter, iter}) {
                if (ai < 0) continue;
                auto ta0 = now();
                std::string ip = iter_path(dir, c, ai);
                std::ifstream in(ip, std::ios::binary);
                size_t n_file = 0;
                if (in.is_open()) {
                    in.seekg(0, std::ios::end);
                    std::streamoff sz = in.tellg();
                    in.seekg(0, std::ios::beg);
                    arrival_buf.resize(static_cast<size_t>(sz));
                    in.read(arrival_buf.data(), sz);
                    in.close();
                } else {
                    arrival_buf.clear();
                }
                auto ta1 = now();
                tm.read_arrival += secs_between(ta0, ta1);
                t3 = ta1;

                for (size_t off = 0; off < arrival_buf.size(); ++n_file) {
                    // 变长记录 (见 var_record_bytes)，读回后补齐成 M 宽 (-1/+inf)
                    if (off + kVarHdr > arrival_buf.size())   // bounds check
                        throw std::runtime_error("ChunkedKnnAccumulator: truncated arrival record in chunk " + std::to_string(c));
                    const char* rp = arrival_buf.data() + off;
                    int64_t gid; uint32_t cnt;
                    std::memcpy(&gid, rp, sizeof(int64_t));
                    std::memcpy(&cnt, rp + sizeof(int64_t), sizeof(uint32_t));
                    if (cnt > static_cast<uint32_t>(M) || off + var_record_bytes(cnt) > arrival_buf.size())   // bounds check
                        throw std::runtime_error("ChunkedKnnAccumulator: bad arrival record in chunk " + std::to_string(c));
                    std::memcpy(new_nbrs.data(), rp + kVarHdr, static_cast<size_t>(cnt) * sizeof(int32_t));
                    std::memcpy(new_dists.data(), rp + kVarHdr + static_cast<size_t>(cnt) * sizeof(int32_t),
                                static_cast<size_t>(cnt) * sizeof(float));
                    std::fill(new_nbrs.begin() + cnt, new_nbrs.end(), -1);
                    std::fill(new_dists.begin() + cnt, new_dists.end(), std::numeric_limits<float>::infinity());
                    off += var_record_bytes(cnt);
                    int64_t pos = gid - chunk_start;
                    if (pos < 0 || pos >= chunk_c_size) continue;  // 不应该发生，保险起见
                    merge_one_record(
                        run_nbrs.data() + static_cast<size_t>(pos) * M,
                        run_dists.data() + static_cast<size_t>(pos) * M,
                        new_nbrs.data(), new_dists.data(), M,
                        scratch, id_sort_buf);
                }
                n_records += n_file;
                // 这一轮要留给最终 merge 的 arrival 文件不删
                if (n_file > 0 && !(defer && ai == iter)) std::filesystem::remove(ip);
                }   // ai
                tm.n_records = static_cast<int64_t>(n_records);
                if (!is_final) {
                    for (int64_t pos = 0; pos < chunk_c_size; ++pos)
                        kth_work[static_cast<size_t>(chunk_start + pos)] =
                            run_dists[static_cast<size_t>(pos) * M + static_cast<size_t>(M - 1)];
                }
                auto t4 = now();
                tm.merge_cpu = secs_between(t3, t4);

                if (is_final) {
                    // chunk c 在最终文件里是第 [chunk_start, chunk_start+chunk_c_size) 行
                    const off_t ofs = hdr + static_cast<off_t>(chunk_start) * M * 4;
                    if (!pwrite_all(fd_knn, run_nbrs.data(), run_nbrs.size() * sizeof(int32_t), ofs) ||
                        (fd_dist >= 0 && !pwrite_all(fd_dist, run_dists.data(), run_dists.size() * sizeof(float), ofs)))
                        throw std::runtime_error("ChunkedKnnAccumulator: final pwrite failed for chunk " + std::to_string(c));
                    if (base_iter >= 0) { std::filesystem::remove(np); std::filesystem::remove(dp); }
                } else if (defer) {
                    // 不写 running: 旧 running 文件 (base_iter ≥ 0 时) 和这一轮的 arrival 都留着
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

        if (!is_final) {
            std::lock_guard<std::mutex> lk(*kth_mtx);
            kth_done.swap(kth_work);
            kth_valid = true;
        }
        if (is_final) {
            if (::close(fd_knn) != 0 || (fd_dist >= 0 && ::close(fd_dist) != 0))
                throw std::runtime_error("ChunkedKnnAccumulator: close failed for final outputs");
            final_written = true;
        }
        deferred_iter = defer ? iter : -1;
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
        if (defer) std::cout << "[ChunkedKNN] merge_iteration(iter=" << iter << "): running not written (deferred to the final merge)\n";
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
        std::ofstream dout(dist_path.empty() ? std::string("/dev/null") : dist_path,   // 空 = 不写 vector_dists
                           std::ios::binary | std::ios::trunc);
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
// ---- out-of-core Step 6: GPU 端 bucket 缓存 ----
// 原来每个 bucket 的 pool (自己 + nprobe 个近邻桶, NPROBE=200 时 ~2.7 MB) 都在
// CPU 上 memcpy 拼好再 H2D——相邻处理的桶近邻集合大量重叠 (DiskJoin 顺序的
// 本意)，同一个桶的数据每轮被拷/传 ~nprobe 次 (SIFT1M ~27 GB/轮)。现在每个桶
// 在 GPU 缓存里只传一次 (预算内，Belady 淘汰，跟 host 缓存用同一份未来访问
// 表)，pool 由 GPU gather kernel 按 job 表从缓存里拼出来 (顺便 u8→fp16)。
struct DevPoolJob {
    const int32_t* ids;   // 桶的 gid (device)；mode 1/2 不用
    const void*    vecs;  // mode 0/2: 桶的原始向量 (device, DataT)；mode 1: 第一条记录
    int32_t count;
    int32_t dest;         // 在 pool 里的起始行
    int64_t gid0;         // mode 2: 第 r 行的 gid = gid0 + r
    int32_t stride;       // mode 1: 记录字节数 (int64 gid + D 个 DataT)
    int32_t mode;         // 0 = ids 数组 + 向量行; 1 = 交错记录; 2 = 连续行 (gid 隐式)
};

// 一个 block 一个桶 (job)，一个 warp 一行：写 fp16 行 + gid，顺手算 ‖row‖²
// (8-bit 值在 fp16/fp32 里都精确，跟 compute_row_norms_kernel 从 d_B 读回来
// 算的结果完全一样)，省掉单独的 norms kernel。
// GPU bucket 缓存未命中的批量装载 (参考 TAGORE gpu_construct: 整块一次 H2D 再在显存里分发)：
// producer 把一个 position 的全部未命中桶按 d_buf 布局写进 pinned 暂存区，末尾附 MissJob 表，
// 一次 cudaMemcpyAsync 到显存暂存区，再由这个 kernel 把每段拷到各自的 d_buf (一个 block 一段)。
struct MissJob {
    char*    dst;
    uint64_t src_off;   // 在暂存区里的偏移 (16B 对齐)
    uint64_t bytes;
};

__global__ void devcache_scatter_misses_kernel(const char* __restrict__ stage, const MissJob* __restrict__ jobs)
{
    const MissJob j = jobs[blockIdx.x];
    const char* src = stage + j.src_off;
    const uint64_t n16 = j.bytes / 16;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(j.dst);
    for (uint64_t i = threadIdx.x; i < n16; i += blockDim.x) d4[i] = s4[i];
    for (uint64_t i = n16 * 16 + threadIdx.x; i < j.bytes; i += blockDim.x) j.dst[i] = src[i];
}

// OutT = __half (8-bit 数据, fp16 Tensor Core GEMM) 或 float (float32 数据, fp32 GEMM; P2.3)。
template <typename DataT, typename OutT>
__global__ void devcache_gather_pool_kernel(const DevPoolJob* __restrict__ jobs,
                                            OutT* __restrict__ B, int32_t* __restrict__ ids_pool,
                                            float* __restrict__ norms_pool, int D)
{
    const DevPoolJob jb = jobs[blockIdx.x];
    const DataT* v = static_cast<const DataT*>(jb.vecs);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    if constexpr (std::is_same<OutT, float>::value) {
        // fp32: 按行块经 shared memory 中转 (行宽 D+1 错开 bank)：global 读写都按扁平下标合并访问，
        // 每个线程再从 shared 里串行算一行的 ‖row‖²，按 compute_row_norms_kernel 完全相同的顺序累加
        // (逐维 s += v*v)，跟不走 dev 缓存时从 d_B 现算的 norms 逐位相同。原来一个 warp 一行、lane 0
        // 串行累加 (31 个 lane 空等、行间 400 B 跨度的非合并读)，这个 kernel 占 msTuring Step 6 GPU
        // 时间的 41% (nsys, #31)。
        (void)lane; (void)warp; (void)nwarps;
        constexpr int kTileFloats = 12288;   // 48 KB
        __shared__ float tile[kTileFloats];
        const float* src0 = (jb.mode == 1)
            ? reinterpret_cast<const float*>(static_cast<const char*>(jb.vecs) + sizeof(int64_t))
            : reinterpret_cast<const float*>(v);
        const int src_ld = (jb.mode == 1) ? jb.stride / static_cast<int>(sizeof(float)) : D;
        float* dst0 = reinterpret_cast<float*>(B) + static_cast<int64_t>(jb.dest) * D;
        const int n = jb.count, P = D + 1, bd = blockDim.x;
        const int TR = kTileFloats / P;   // 每块行数 (D=100: 121)
        auto id_of = [&](int r) -> int32_t {
            return (jb.mode == 0) ? jb.ids[r]
                 : (jb.mode == 1) ? static_cast<int32_t>(*reinterpret_cast<const int64_t*>(
                       static_cast<const char*>(jb.vecs) + static_cast<int64_t>(r) * jb.stride))
                 : static_cast<int32_t>(jb.gid0 + r);
        };
        if (TR == 0) {   // D 太大放不下一行: 直接逐线程一行 (慢路径，顺序不变)
            for (int r = threadIdx.x; r < n; r += bd) {
                const float* s = src0 + static_cast<int64_t>(r) * src_ld;
                float* d = dst0 + static_cast<int64_t>(r) * D;
                float sq = 0.0f;
                for (int c = 0; c < D; ++c) { float x = s[c]; d[c] = x; sq += x * x; }
                norms_pool[jb.dest + r] = sq;
                ids_pool[jb.dest + r] = id_of(r);
            }
            return;
        }
        for (int t0 = 0; t0 < n; t0 += TR) {
            const int rows = min(TR, n - t0);
            const int total = rows * D;
            const float* s = src0 + static_cast<int64_t>(t0) * src_ld;
            float* d = dst0 + static_cast<int64_t>(t0) * D;
            int k = threadIdx.x;
            for (; k + 3 * bd < total; k += 4 * bd) {   // 4 路展开: 多个 load 在途
                float x[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kk = k + u * bd, r = kk / D, c = kk - r * D;
                    x[u] = s[r * src_ld + c];
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kk = k + u * bd, r = kk / D, c = kk - r * D;
                    tile[r * P + c] = x[u];
                    d[kk] = x[u];
                }
            }
            for (; k < total; k += bd) {
                const int r = k / D, c = k - r * D;
                const float x = s[r * src_ld + c];
                tile[r * P + c] = x;
                d[k] = x;
            }
            __syncthreads();
            for (int r = threadIdx.x; r < rows; r += bd) {
                const float* row = tile + r * P;
                float sq = 0.0f;
                for (int c = 0; c < D; ++c) {
                    float x = row[c];
                    sq += x * x;
                }
                norms_pool[jb.dest + t0 + r] = sq;
                ids_pool[jb.dest + t0 + r] = id_of(t0 + r);
            }
            __syncthreads();
        }
        return;
    }
    for (int r = warp; r < jb.count; r += nwarps) {
        const char* rec = static_cast<const char*>(jb.vecs) + static_cast<int64_t>(r) * jb.stride;
        const DataT* src = (jb.mode == 1) ? reinterpret_cast<const DataT*>(rec + sizeof(int64_t))
                                          : v + static_cast<int64_t>(r) * D;
        OutT* dst = B + static_cast<int64_t>(jb.dest + r) * D;
        float acc = 0.f;
        if constexpr (std::is_same<OutT, float>::value) {
            // fp32 已在上面按 32 行一组处理
        } else if (sizeof(DataT) == 1 && (D & 3) == 0) {
            // 8-bit 且 D%4==0: 每 lane 一次读 4 个字节 (uchar4)、写 2 个 half2，
            // D=128 一个 warp 一趟读完一行 (原来逐字节读、逐 half 写要 4 趟)。
            // 对齐: 桶向量区 16B 对齐 (见 DevBucket)、行宽 D 字节是 4 的倍数；
            // B 的行宽 2D 字节是 8 的倍数。
            // int8 按有符号读 (char4)，uint8 按无符号 (uchar4)；原来一律 uchar4，int8 负值读成 128…255
            using V4 = typename std::conditional<std::is_same<DataT, int8_t>::value, char4, uchar4>::type;
            const V4* src4 = reinterpret_cast<const V4*>(src);
            __half2* dst2 = reinterpret_cast<__half2*>(dst);
            for (int q = lane; q < (D >> 2); q += 32) {
                V4 u = src4[q];
                // 8-bit 值在 fp16 里精确
                __half2 h01 = __floats2half2_rn(static_cast<float>(u.x), static_cast<float>(u.y));
                __half2 h23 = __floats2half2_rn(static_cast<float>(u.z), static_cast<float>(u.w));
                dst2[2 * q]     = h01;
                dst2[2 * q + 1] = h23;
                float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
                acc += f01.x * f01.x + f01.y * f01.y + f23.x * f23.x + f23.y * f23.y;
            }
        } else {
            for (int e = lane; e < D; e += 32) {
                float x = static_cast<float>(src[e]);
                __half h = __float2half_rn(x);
                dst[e] = h;
                float xh = __half2float(h);
                acc += xh * xh;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) {
            norms_pool[jb.dest + r] = acc;
            ids_pool[jb.dest + r] = (jb.mode == 0) ? jb.ids[r]
                                  : (jb.mode == 1) ? static_cast<int32_t>(*reinterpret_cast<const int64_t*>(rec))
                                  : static_cast<int32_t>(jb.gid0 + r);
        }
    }
}

// 释放前必须等到最后一次读它的 gather 已经完成 (由 ctx 提供: 等主线程提交到
// last_pos，再让 copy_stream 等那个位置的事件)，然后 stream-ordered free。
struct DevCacheCtx {
    cudaStream_t copy_stream = nullptr;
    std::function<void(int32_t last_pos)> before_free;   // 在 producer 线程上调用
    // 设置后淘汰不再当场等待/释放，只把 (d_buf, last_pos) 交出去延迟释放 (见 Step 6 producer 的
    // drain_pending_free)：被淘汰的桶常常刚在 1-3 个还没提交的 position 里用过，当场等会卡住 producer
    std::function<void(void* d_buf, int32_t last_pos, size_t bytes)> defer_free;
};

struct DevBucket {
    void*   d_buf  = nullptr;   // [int32 ids (count, 16B 对齐)] [DataT vecs (count*D)]
    const int32_t* d_ids = nullptr;
    const void*    d_vecs = nullptr;
    int64_t count = 0;
    size_t  bytes = 0;
    int64_t gid0 = 0;     // 指向整块上传的 blob 时 (d_buf 为空、不拥有显存) 的布局，见 DevPoolJob
    int32_t stride = 0;
    int32_t mode = 0;
    mutable int32_t last_pos = -1;   // 最近一次被拼进 pool 的处理位置
    DevCacheCtx* ctx = nullptr;

    DevBucket() = default;
    DevBucket(const DevBucket&) = delete;
    DevBucket& operator=(const DevBucket&) = delete;
    DevBucket(DevBucket&& o) noexcept { take(o); }
    DevBucket& operator=(DevBucket&& o) noexcept {
        if (this != &o) { release(); take(o); }
        return *this;
    }
    ~DevBucket() { release(); }
private:
    void take(DevBucket& o) {
        d_buf = o.d_buf; d_ids = o.d_ids; d_vecs = o.d_vecs; count = o.count; bytes = o.bytes;
        gid0 = o.gid0; stride = o.stride; mode = o.mode;
        last_pos = o.last_pos; ctx = o.ctx;
        o.d_buf = nullptr; o.d_ids = nullptr; o.d_vecs = nullptr; o.count = 0; o.bytes = 0; o.last_pos = -1;
    }
    void release() {
        if (!d_buf) return;
        if (ctx && ctx->defer_free) {
            ctx->defer_free(d_buf, last_pos, bytes);
        } else {
            if (ctx && ctx->before_free) ctx->before_free(last_pos);
            cudaFreeAsync(d_buf, ctx ? ctx->copy_stream : nullptr);
        }
        d_buf = nullptr; d_ids = nullptr; d_vecs = nullptr; count = 0; bytes = 0;
    }
};

// 大块 H2D: 两块 64 MB pinned 暂存区双缓冲 (多线程拷进暂存区的同时上一块在 DMA)，
// 跟 prune_windowed_gpu.cu 的 Staging 同一做法 (cuVS 26.12 cagra_build.cuh 处理 host batch)
static void h2d_staged(void* d, const void* h, size_t bytes)
{
    constexpr size_t kChunk = size_t(64) << 20;
    void* pin[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    cudaStream_t st = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&pin[b], kChunk));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
    }
    bool used[2] = {false, false};
    for (size_t off = 0, k = 0; off < bytes; off += kChunk, ++k) {
        const int b = static_cast<int>(k & 1);
        const size_t n = std::min(kChunk, bytes - off);
        if (used[b]) CUDA_CHECK(cudaEventSynchronize(ev[b]));
        const int64_t pieces = static_cast<int64_t>((n + (size_t(1) << 20) - 1) >> 20);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < pieces; ++i) {
            const size_t o = static_cast<size_t>(i) << 20;
            std::memcpy(static_cast<char*>(pin[b]) + o, static_cast<const char*>(h) + off + o,
                        std::min(size_t(1) << 20, n - o));
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(d) + off, pin[b], n, cudaMemcpyHostToDevice, st));
        CUDA_CHECK(cudaEventRecord(ev[b], st));
        used[b] = true;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    for (int b = 0; b < 2; ++b) { cudaFreeHost(pin[b]); cudaEventDestroy(ev[b]); }
    cudaStreamDestroy(st);
}

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
    const std::vector<int32_t>* precomputed_order = nullptr,
    size_t gpu_cache_bytes = 0,
    const float* seed_kth = nullptr)   // (N,) 上一轮 merge 后每点 running 第 M 个距离 (#62)
{
    auto func_t0 = std::chrono::high_resolution_clock::now();
    constexpr bool want_distances = true;  // 合并去重总是需要距离
    // ============= Stage 2 路径选择（fp16 Tensor Core / fp32 fallback）=============
    // - int8/uint8 → fp16 Tensor Core GEMM (fp32 累加)，跟 GPU-resident 路径一样:
    //   |v| ≤ 255 在 fp16 里精确，D=128 点积 ≤ 8.3M < 2^24 → 距离精确。原来走
    //   INT8 IMMA (uint8 先减 128)，但 V100 (sm_70) 没有 INT8 Tensor Core，
    //   INT8 GemmEx 落到 dp4a，比 fp16 Tensor Core 慢一倍。
    // - 其它 (float/half/uint32/int32 等) → 走 fp32 cuBLAS GEMM（原行为）
    constexpr bool kIsInt8Path =
        std::is_same<DataT, int8_t>::value || std::is_same<DataT, uint8_t>::value;
    using GemmInT  = typename std::conditional<kIsInt8Path, __half, float>::type;
    using GemmOutT = float;

    std::cout << "[VectorKNN] Building per-vector KNN with Tensor Core matmul\n"
              << "  N=" << N << ", D=" << D
              << ", n_centroids=" << n_centroids
              << ", K=" << K << ", M=" << M
              << ", path=" << (kIsInt8Path ? "fp16 Tensor Core (8-bit exact)" : "fp32 SGEMM") << "\n";   // V100 没有 TF32

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

    // GPU 端 bucket 缓存跟 host 缓存用同一份未来访问表 (8-bit / float32 数据、预算 > 0 时用)
    // float32 数据也用 (P2.3: gather kernel 写 fp32 行，结果跟逐桶上传路径逐位相同)
    constexpr bool kDevCacheType = kIsInt8Path || std::is_same<DataT, float>::value;
    const bool use_dev_cache = kDevCacheType && gpu_cache_bytes > 0;
    // blob 模式: 这一轮的桶数据整块就在 host 内存里 (round 1: 重排后的 raw_host；
    // round 2..t: BucketVectorAccumulator 内存模式的记录缓冲区)，而且放得进 GPU bucket
    // 缓存预算 → 整块上传一次 (pinned 双缓冲)，dev 缓存的 loader 只发出指向 blob
    // 的 DevBucket (不再逐桶 host 缓存 get + 拆记录 + cudaMallocAsync + 两次 pageable
    // H2D: SIFT10M 每桶 ~58 µs、每轮 10 万桶，都在 producer 线程上)。放不下就照旧逐桶
    // 加载 (Belady)，大路径不依赖数据集放进显存。BUCKET2_BLOB=0 关掉。
    const char* h_blob = nullptr; size_t blob_bytes = 0; int blob_mode = 0, blob_stride = 0;
    char* d_blob = nullptr;
    if (use_dev_cache && bucket_vecs.blob_layout(h_blob, blob_bytes, blob_mode, blob_stride) &&
        blob_bytes > 0 && blob_bytes <= gpu_cache_bytes) {
        const char* e = std::getenv("BUCKET2_BLOB");
        if (!(e && e[0] == '0')) {
            auto tb0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMalloc(&d_blob, blob_bytes));
            h2d_staged(d_blob, h_blob, blob_bytes);
            std::cout << "  [VectorKNN] bucket data uploaded as one blob: " << blob_bytes / 1e6 << " MB (mode "
                      << blob_mode << ") in " << std::chrono::duration<double>(
                             std::chrono::high_resolution_clock::now() - tb0).count() << "s\n";
        }
    }
    // blob 模式下两个 Belady 缓存都不用 (job 直接指向 blob，中心桶 gid 也从 host blob
    // 取)，未来访问表 (SIFT10M 每轮 0.8 s) 和缓存构造 (拷表 0.5 s) 都省掉: 给空表
    std::vector<std::vector<int32_t>> future_access;
    if (d_blob) {
        future_access.assign(static_cast<size_t>(n_centroids), {});
    } else {
        auto access_adj = bucket_order::adjacency_from_flat_graph(
            centroid_knn_graph, n_centroids, static_cast<int32_t>(K));
        future_access = bucket_order::compute_future_access_lists(bucket_order_seq, access_adj);
    }
    double t_setup_future_access = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - func_t0).count() - t_setup_bucket_count - t_setup_order;

    auto vec_bucket_bytes = [](const VecBucket<DataT>& b) {
        return b.ids.size() * sizeof(int64_t) + b.vecs.size() * sizeof(DataT);
    };
    // GPU 端 bucket 缓存跟 host 缓存用同一份未来访问表 (8-bit / float32 数据、预算 > 0 时用)
    std::vector<std::vector<int32_t>> future_access_dev;
    if (use_dev_cache) future_access_dev = future_access;
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
    // dots 行宽对齐到 4 (block top-M kernel 按 16 B 向量读)
    // dots 按 bucket 行分块 (每行的 top-M 互相独立，分块结果完全一样)：dots buffer 只要
    // kDotsRowTile 行，不随最大 bucket 变大 (参考 TAGORE gpu_construct: 缓冲区按固定上界分配)。
    // M > 256 的 raft 回退路径不分块 (只在 bucket ≤ 分块行数时可用，见 3e 的检查)。
    constexpr size_t kDotsRowTile = 256;
    const size_t dots_rows    = std::min(max_bucket_size, kDotsRowTile);
    size_t bytes_dots         = dots_rows * ((max_pool_size_ub + 3) / 4 * 4) * sizeof(GemmOutT);
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
    // 2.0× SM 利用率目标，hide launch overhead。#40 实测 (SIFT1M --out-of-core,
    // GPU bucket 缓存之后): 3 slot (=1.5x) 1.59 s/轮, 4 slot (=2.0x) 1.43 s/轮,
    // 6 slot 1.38 s/轮但显存 +0.8 GB。
    constexpr double OVERSUB = 2.0;
    int target_streams = std::max(2,
        static_cast<int>(std::ceil(num_sm * OVERSUB / est_blocks_per_kernel)));
    target_streams = std::min(target_streams, MAX_SLOTS);
    // GPU bucket 缓存逐桶加载 (非 blob) 时 producer 是瓶颈: slot 少了 producer 最多只能领先 GPU
    // 两个 position，主线程在 stream_sync / wait_producer 上来回等。多给几个 slot 加深流水 (参考 TAGORE
    // largeIndex: graph_pool 多个缓冲槽，GPU 构建和 CPU 合并互不等待)。#3 的 dots 行分块后每个 slot
    // 只要 ~0.2 GB。SIFT10M 非 blob 实测 (缓存 200 MB): 2 slot 12.8 s/轮, 4: 10.4, 6: 9.4, 8: 9.2。
    // blob 模式 (数据整块在显存) 没有这个瓶颈，保持原来的 slot 数。
    constexpr int kMinSlotsCacheMiss = 6;
    if (use_dev_cache && !d_blob) target_streams = std::min(MAX_SLOTS, std::max(target_streams, kMinSlotsCacheMiss));
    // 至少 4 个 slot (显存放得下时): B 512 (#33) 以后 est_blocks/kernel = 128 → OVERSUB 公式只给 2 个，
    // blob 模式下 GPU 在相邻两个桶之间空等。msTuring10M Step 6 每轮 2 → 4 slot: 17.6 → 15.7 s (#47)
    constexpr int kMinSlots = 4;
    target_streams = std::min(MAX_SLOTS, std::max(target_streams, kMinSlots));

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
    // round 2..t: 每点 running 第 M 个距离 (N floats，SIFT10M 40 MB) 上传一次，
    // block top-M 按 gid 取行阈值初值
    float* d_seed_kth = nullptr;
    if (seed_kth) {
        CUDA_CHECK(cudaMalloc(&d_seed_kth, static_cast<size_t>(N) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_seed_kth, seed_kth, static_cast<size_t>(N) * sizeof(float), cudaMemcpyHostToDevice));
    }
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
        // ids 后面紧跟 dists (block top-M 路径一次 D2H 拷回两者)，所以开两倍
        CUDA_CHECK(cudaMalloc(&d_out_neighbors[s], 2 * bytes_out));
        CUDA_CHECK(cudaMalloc(&d_select_idx[s],    bytes_select_idx));
        CUDA_CHECK(cudaMalloc(&d_select_dist[s],   bytes_select_dist));
        CUDA_CHECK(cudaMallocHost(&h_ids_bucket[s], bytes_ids_bucket));
        CUDA_CHECK(cudaMallocHost(&h_ids_pool[s],   bytes_ids_pool));
        CUDA_CHECK(cudaMallocHost(&h_B_raw[s],      bytes_B_raw));
        h_ids_bucket64[s] = new int64_t[max_bucket_size];
        h_ids_pool64[s]   = new int64_t[max_pool_size_ub];
        CUDA_CHECK(cudaMallocHost(&h_out[s],        2 * bytes_out));
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
        bool    fused_out   = false; // dists 紧跟在 h_out 的 ids 后面 (block top-M 路径)
    };
    Pending pending[MAX_SLOTS]{};   // 全部默认初始化为 {-1, 0, 0}

    // ---- GEMM 纯计算时间统计 (GPU side, via cudaEvent on stream) ----
    cudaEvent_t ev_gemm_start[MAX_SLOTS] = {nullptr};
    cudaEvent_t ev_gemm_stop[MAX_SLOTS]  = {nullptr};
    bool gemm_evt_valid[MAX_SLOTS] = {false};
    double gemm_total_ms = 0.0;
    // 纯 GEMM 计时 (每桶 2 次 event record + elapsed 查询) 只在诊断时开:
    // 主线程每桶的 CUDA API 调用就是 out-of-core 循环的瓶颈 (gpu_submit)
    const bool gemm_timing = std::getenv("BUCKET2_GEMM_TIMING") != nullptr;
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

    // scatter (逐点铺成 M 宽 + acc.add 进 chunk 写缓冲) 交给一个 consumer 线程，
    // 不再串在主线程的 sync → scatter → submit 之间 (~0.7–1.0 s/轮)。主线程只把
    // 这个 slot 的结果 (gid + bucket_size×actual_M 的邻居/距离, ~50 KB) 拷进一个
    // 复用的 job buffer 再入队，然后就可以覆写 h_ids_bucket/h_out[slot] 提交下一
    // 个 bucket。只有 scatter 线程调用 acc.add，按入队顺序处理 (跟原来的顺序一致)。
    struct ScatterJob {
        int bs = 0, aM = 0;
        std::vector<int32_t> gid, nb;
        std::vector<float>   d;
    };
    std::deque<std::unique_ptr<ScatterJob>> scatter_queue, scatter_free;
    std::mutex scatter_mtx;
    std::condition_variable scatter_cv;
    bool scatter_done = false;            // guarded by scatter_mtx
    std::exception_ptr scatter_exc;
    double t_scatter_worker = 0;
    std::thread scatter_thread([&]() {
        try {
            for (;;) {
                std::unique_ptr<ScatterJob> job;
                {
                    std::unique_lock<std::mutex> lk(scatter_mtx);
                    scatter_cv.wait(lk, [&] { return !scatter_queue.empty() || scatter_done; });
                    if (scatter_queue.empty()) return;   // done 且已排空
                    job = std::move(scatter_queue.front());
                    scatter_queue.pop_front();
                }
                auto ts0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < job->bs; ++i) {
                    std::fill(scatter_new_n.begin(), scatter_new_n.end(), -1);
                    std::fill(scatter_new_d.begin(), scatter_new_d.end(),
                              std::numeric_limits<float>::infinity());
                    for (int m = 0; m < job->aM && m < M; ++m) {
                        scatter_new_n[m] = job->nb[static_cast<size_t>(i) * job->aM + m];
                        scatter_new_d[m] = job->d [static_cast<size_t>(i) * job->aM + m];
                    }
                    acc.add(job->gid[static_cast<size_t>(i)], scatter_new_n.data(), scatter_new_d.data());
                }
                t_scatter_worker += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts0).count();
                {
                    std::lock_guard<std::mutex> lk(scatter_mtx);
                    scatter_free.push_back(std::move(job));
                }
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(scatter_mtx);
            scatter_exc = std::current_exception();
            scatter_queue.clear();
        }
    });
    struct ScatterJoin {
        std::thread& t; std::mutex& m; std::condition_variable& cv; bool& done;
        ~ScatterJoin() {
            { std::lock_guard<std::mutex> lk(m); done = true; }
            cv.notify_all();
            if (t.joinable()) t.join();
        }
    } scatter_join{scatter_thread, scatter_mtx, scatter_cv, scatter_done};

    auto scatter_pending = [&](int slot) {
        if (pending[slot].c < 0) return;
        // h_ids_bucket[slot] 这时候还是"上一次派到这个 slot 的那个 bucket"的
        // gid 数组 (本次循环要到下面 3b 才会覆写它)，拷一份给 scatter 线程。
        const int bs = pending[slot].bucket_size;
        const int aM = pending[slot].actual_M;
        std::unique_ptr<ScatterJob> job;
        {
            std::lock_guard<std::mutex> lk(scatter_mtx);
            if (scatter_exc) std::rethrow_exception(scatter_exc);
            if (!scatter_free.empty()) { job = std::move(scatter_free.back()); scatter_free.pop_back(); }
        }
        if (!job) job = std::make_unique<ScatterJob>();
        const size_t n_out = static_cast<size_t>(bs) * aM;
        job->bs = bs; job->aM = aM;
        job->gid.assign(h_ids_bucket[slot], h_ids_bucket[slot] + bs);
        job->nb.assign(h_out[slot], h_out[slot] + n_out);
        const float* hd = pending[slot].fused_out
            ? reinterpret_cast<const float*>(h_out[slot] + n_out)
            : h_out_dists[slot];   // 始终已分配 (want_distances 恒为 true)
        job->d.assign(hd, hd + n_out);
        {
            std::lock_guard<std::mutex> lk(scatter_mtx);
            scatter_queue.push_back(std::move(job));
        }
        scatter_cv.notify_one();
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

    // ---- GPU 端 bucket 缓存 (use_dev_cache) ----
    // 每个 slot: pinned job 表 + device job 表 + 加载完成事件 (producer 在
    // copy_stream 上记，主线程让 slot stream 等它)。gather 完成事件按处理位置
    // 放进一个带 tag 的环形表 (主线程记)，淘汰时 copy_stream 等最后一次读该桶
    // 的那个位置；位置被环形表覆盖前主线程会先同步它，所以 tag 不匹配 = 早已完成。
    constexpr int kPosRing = 1024;
    struct PosEvent { cudaEvent_t ev = nullptr; int32_t pos = -1; };
    std::vector<PosEvent> pos_ring;
    int32_t submitted_pos = -1;                   // guarded by stage_mtx
    cudaStream_t copy_stream = nullptr;
    DevPoolJob* h_jobs[MAX_SLOTS] = {nullptr};
    DevPoolJob* d_jobs[MAX_SLOTS] = {nullptr};
    int n_jobs_staged[MAX_SLOTS] = {0};
    cudaEvent_t ev_load[MAX_SLOTS] = {nullptr};
    DevCacheCtx dev_ctx;
    std::unique_ptr<bucket_order::BeladyBucketCache<DevBucket>> dev_cache;
    // job 表直接放 pinned host 内存让 gather kernel 经 UVA 读 (每桶几十条 × 32 B)，
    // 省掉主线程每桶一次 cudaMemcpyAsync；不支持映射时退回 H2D 拷贝。
    bool jobs_mapped = false;
    // 未命中桶的 pinned 暂存区 (每 slot 一块，按 d_buf 布局 [int32 ids 16B 对齐][vecs] 写好，一次
    // 异步 H2D)。参考 TAGORE gpu_construct: 缓冲区按上界一次分配、整块拷贝，不走 pageable 小拷贝。
    // 一个 position 的未命中最多就是整个 pool，所以按 pool 上界 + 每桶 16B 对齐余量分配。
    // 淘汰的 device 桶: 等最后读它的 position 提交后 (copy_stream 等那个 gather 事件) 再 cudaFreeAsync。
    // 只在 producer 线程上读写 (淘汰都发生在 producer 的 get() 里)；收尾时 dev_cache.reset() 在主线程，
    // 那时 producer 已结束。缓存的字节账不变，最多多占几个 position 的淘汰量。
    std::vector<std::tuple<void*, int32_t, size_t>> pending_free;
    double t_free_drain = 0;
    // device 桶块复用 (参考 TAGORE gpu_construct: 显存分配一次、反复复用，不在热循环里 malloc/free)：
    // 桶按 1 KB 粒度取整，淘汰的块进同尺寸类的空闲链表 (copy_stream 已等过最后一个读者，之后对它的写
    // 都在 copy_stream 上，按流顺序安全)，下次同尺寸 (或最多大 50%) 的未命中直接拿来用；链表里的总量
    // 超过 gpu_cache_bytes/8 时才真的 cudaFreeAsync，显存占用 ≤ 缓存预算 + 1/8。
    constexpr size_t kArenaGran = 1024;
    std::unordered_map<size_t, std::vector<char*>> arena_free;   // 尺寸 (字节, kArenaGran 的倍数) → 空闲块
    size_t arena_free_bytes = 0;
    const size_t arena_free_cap = gpu_cache_bytes / 8;
    int64_t n_arena_fallback = 0;
    // bytes 已按 kArenaGran 取整；返回块地址，bytes 改成块的实际大小 (淘汰时按它回到对应链表，缓存记账也用它)
    auto arena_alloc = [&](size_t& bytes) -> char* {
        // 同尺寸类优先，其次最多大 50% 的空闲块 (尺寸类之间不合并，靠这个减少 arena 碎片导致的退回)
        for (size_t sz = bytes; sz <= bytes + bytes / 2; sz += kArenaGran) {
            auto it = arena_free.find(sz);
            if (it != arena_free.end() && !it->second.empty()) {
                char* b = it->second.back(); it->second.pop_back(); bytes = sz; arena_free_bytes -= sz; return b;
            }
        }
        void* b = nullptr;
        CUDA_CHECK(cudaMallocAsync(&b, bytes, copy_stream));
        ++n_arena_fallback;
        return static_cast<char*>(b);
    };
    auto arena_release = [&](void* buf, size_t bytes) {   // 调用前 copy_stream 已等过最后的读者
        if (arena_free_bytes + bytes <= arena_free_cap) {
            arena_free[bytes].push_back(static_cast<char*>(buf));
            arena_free_bytes += bytes;
        } else {
            cudaFreeAsync(buf, copy_stream);
        }
    };
    auto drain_pending_free = [&](bool all) {
        if (pending_free.empty()) return;
        auto td0 = std::chrono::high_resolution_clock::now();
        size_t keep = 0;
        {
            std::lock_guard<std::mutex> lk(stage_mtx);
            int32_t waited = -1;
            for (size_t i = 0; i < pending_free.size(); ++i) {
                auto [buf, lp, nbytes] = pending_free[i];
                if (!all && lp > submitted_pos) { pending_free[keep++] = pending_free[i]; continue; }
                if (lp >= 0 && lp != waited) {
                    const PosEvent& pe = pos_ring[static_cast<size_t>(lp % kPosRing)];
                    if (pe.pos == lp) cudaStreamWaitEvent(copy_stream, pe.ev, 0);
                    waited = lp;
                }
                arena_release(buf, nbytes);
            }
        }
        pending_free.resize(keep);
        t_free_drain += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - td0).count();
    };
    char* h_dstage[MAX_SLOTS] = {nullptr};
    char* d_dstage[MAX_SLOTS] = {nullptr};
    MissJob* h_mjobs[MAX_SLOTS] = {nullptr};
    size_t dstage_cap = 0;
    if (use_dev_cache) {
        int dev = 0, uva = 0, can_map = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, dev));
        jobs_mapped = uva && can_map;
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        pos_ring.resize(kPosRing);
        for (auto& pe : pos_ring) CUDA_CHECK(cudaEventCreateWithFlags(&pe.ev, cudaEventDisableTiming));
        for (int sl = 0; sl < num_slots; ++sl) {
            CUDA_CHECK(cudaMallocHost(&h_jobs[sl], (static_cast<size_t>(K) + 1) * sizeof(DevPoolJob)));
            CUDA_CHECK(cudaMalloc(&d_jobs[sl], (static_cast<size_t>(K) + 1) * sizeof(DevPoolJob)));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_load[sl], cudaEventDisableTiming));
        }
        dev_ctx.copy_stream = copy_stream;
        dev_ctx.before_free = [&](int32_t last_pos) {
            if (last_pos < 0) return;
            std::unique_lock<std::mutex> lk(stage_mtx);
            stage_cv.wait(lk, [&] { return submitted_pos >= last_pos || stage_abort; });
            const PosEvent& pe = pos_ring[static_cast<size_t>(last_pos % kPosRing)];
            if (pe.pos == last_pos) cudaStreamWaitEvent(copy_stream, pe.ev, 0);
        };
        dev_ctx.defer_free = [&](void* buf, int32_t last_pos, size_t bytes) { pending_free.emplace_back(buf, last_pos, bytes); };
        dev_cache = std::make_unique<bucket_order::BeladyBucketCache<DevBucket>>(
            std::move(future_access_dev), gpu_cache_bytes,
            [](const DevBucket& b) { return b.bytes; });
        dev_cache->set_pin_current_position(true);   // 本 position 的桶都在 job 表里，不能被淘汰
        if (!d_blob) {
            dstage_cap = max_pool_size_ub * (sizeof(int32_t) + static_cast<size_t>(D) * sizeof(DataT))
                       + (static_cast<size_t>(K) + 1) * 16;
            dstage_cap = (dstage_cap + 255) / 256 * 256;   // 后面接 MissJob 表 (8B 字段)，必须对齐
            for (int sl = 0; sl < num_slots; ++sl) {
                // [未命中数据 dstage_cap][MissJob 表 (K+1) 条]，一次拷过去
                const size_t tot = dstage_cap + (static_cast<size_t>(K) + 1) * sizeof(MissJob);
                CUDA_CHECK(cudaMallocHost(&h_dstage[sl], tot));
                CUDA_CHECK(cudaMalloc(&d_dstage[sl], tot));
                h_mjobs[sl] = reinterpret_cast<MissJob*>(h_dstage[sl] + dstage_cap);
            }
            std::cout << "  [VectorKNN] miss staging: " << num_slots << " x " << dstage_cap / 1e6 << " MB pinned\n";
        }
        std::cout << "  [VectorKNN] GPU bucket cache: " << gpu_cache_bytes / 1e6 << " MB\n";
    }
    double t_dev_upload = 0, t_dev_hostget = 0, t_dev_malloc = 0, t_dev_copy = 0;
    int64_t n_dev_loads = 0;
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
                if (use_dev_cache) {
                    // pool 由 GPU 从 bucket 缓存 gather；这里只解析/加载桶、写 job 表
                    size_t dofs = 0;
                    int nj = 0;
                    size_t stage_off = 0;
                    int n_miss = 0;
                    // 这个 slot 上一次 (num_slots 个 position 之前) 的未命中拷贝完成后才能重写暂存区
                    if (h_dstage[slot]) CUDA_CHECK(cudaEventSynchronize(ev_load[slot]));
                    drain_pending_free(false);   // 已提交 position 的淘汰桶在这里释放 (不阻塞)
                    auto dev_loader = [&](int32_t bid) {
                        auto tu0 = diag_now();
                        if (d_blob) {
                            DevBucket db;
                            db.ctx = &dev_ctx;
                            db.count = bucket_count[bid];
                            size_t off = 0; int64_t gid0 = 0;
                            bucket_vecs.blob_locate(bid, off, gid0);
                            const size_t row_bytes = (blob_mode == 1) ? static_cast<size_t>(blob_stride)
                                                                      : static_cast<size_t>(D) * sizeof(DataT);
                            if (off + static_cast<size_t>(db.count) * row_bytes > blob_bytes)   // bounds check
                                throw std::runtime_error("VectorKNN: bucket " + std::to_string(bid) + " outside the blob");
                            db.d_vecs = d_blob + off;
                            db.gid0 = gid0; db.stride = blob_stride; db.mode = blob_mode;
                            ++n_dev_loads;
                            t_dev_upload += diag_secs(tu0, diag_now());
                            return db;   // d_buf 为空: 不拥有显存，淘汰时什么都不做
                        }
                        // 这一轮的记录整块在 host 内存里 (h_blob，但放不进 GPU 缓存，例如 SIFT100M round 2):
                        // 直接从 blob 拷进 pinned 暂存，不经过 host Belady 缓存 (不重复占内存、不读盘)。
                        // 参考 TAGORE largeIndex: graph_pool 常驻 host 内存，GPU 侧按需整块取 (cuVS-loop #75 的思路)。
                        const VecBucket<DataT>* hbp = nullptr;
                        size_t blob_off = 0; int64_t blob_gid0 = 0; size_t blob_row = 0;
                        int64_t cnt_b = 0;
                        if (h_blob) {
                            bucket_vecs.blob_locate(bid, blob_off, blob_gid0);
                            cnt_b = bucket_count[bid];
                            blob_row = (blob_mode == 1) ? static_cast<size_t>(blob_stride) : static_cast<size_t>(D) * sizeof(DataT);
                            if (blob_off + static_cast<size_t>(cnt_b) * blob_row > blob_bytes)   // bounds check
                                throw std::runtime_error("VectorKNN: bucket " + std::to_string(bid) + " outside the host blob");
                        } else {
                            hbp = &bucket_cache.get(pos, bid, bucket_loader);
                            cnt_b = static_cast<int64_t>(hbp->ids.size());
                        }
                        t_dev_hostget += diag_secs(tu0, diag_now());
                        DevBucket db;
                        db.ctx = &dev_ctx;
                        db.count = cnt_b;
                        const size_t ids_bytes = (static_cast<size_t>(db.count) * sizeof(int32_t) + 15) / 16 * 16;
                        const size_t vec_bytes = static_cast<size_t>(db.count) * D * sizeof(DataT);
                        const size_t real_bytes = ids_bytes + vec_bytes;
                        db.bytes = (std::max<size_t>(real_bytes, 16) + kArenaGran - 1) / kArenaGran * kArenaGran;
                        auto tm0 = diag_now();
                        db.d_buf = arena_alloc(db.bytes);
                        t_dev_malloc += diag_secs(tm0, diag_now());
                        char* base = static_cast<char*>(db.d_buf);
                        db.d_ids  = reinterpret_cast<const int32_t*>(base);
                        db.d_vecs = base + ids_bytes;
                        if (db.count > 0) {
                            if (stage_off + real_bytes > dstage_cap)   // bounds check (未命中总量 ≤ pool 上界)
                                throw std::runtime_error("VectorKNN: miss staging overflow at pos " + std::to_string(pos));
                            char* hs = h_dstage[slot] + stage_off;
                            int32_t* hs_ids = reinterpret_cast<int32_t*>(hs);
                            if (hbp) {
                                for (int64_t i = 0; i < db.count; ++i) hs_ids[i] = static_cast<int32_t>(hbp->ids[static_cast<size_t>(i)]);
                                std::memcpy(hs + ids_bytes, hbp->vecs.data(), vec_bytes);
                            } else if (blob_mode == 1) {   // 交错记录 [int64 gid][D 个 DataT]
                                const char* src = h_blob + blob_off;
                                char* dv = hs + ids_bytes;
                                for (int64_t i = 0; i < db.count; ++i) {
                                    int64_t g; std::memcpy(&g, src + static_cast<size_t>(i) * blob_row, sizeof(int64_t));
                                    if (g < 0 || g >= N) throw std::runtime_error("VectorKNN: blob gid out of range");   // bounds check
                                    hs_ids[i] = static_cast<int32_t>(g);
                                    std::memcpy(dv + static_cast<size_t>(i) * D * sizeof(DataT),
                                                src + static_cast<size_t>(i) * blob_row + sizeof(int64_t), static_cast<size_t>(D) * sizeof(DataT));
                                }
                            } else {                       // 连续行，gid = gid0 + i
                                for (int64_t i = 0; i < db.count; ++i) hs_ids[i] = static_cast<int32_t>(blob_gid0 + i);
                                std::memcpy(hs + ids_bytes, h_blob + blob_off, vec_bytes);
                            }
                            if (n_miss > static_cast<int>(K))   // bounds check (一个 position 最多 K+1 个桶)
                                throw std::runtime_error("VectorKNN: miss job table overflow at pos " + std::to_string(pos));
                            h_mjobs[slot][n_miss++] = MissJob{base, static_cast<uint64_t>(stage_off), static_cast<uint64_t>(real_bytes)};
                            stage_off += (real_bytes + 15) / 16 * 16;   // 下一块 16B 对齐 (余量已算进 dstage_cap)
                        }
                        ++n_dev_loads;
                        t_dev_upload += diag_secs(tu0, diag_now());
                        return db;
                    };
                    auto add_dev = [&](int64_t bid) {
                        if (bucket_count[bid] == 0) return;
                        if (d_blob) {   // 直接按偏移出 job，不经过 dev 缓存
                            size_t off = 0; int64_t gid0 = 0;
                            bucket_vecs.blob_locate(bid, off, gid0);
                            const int64_t cnt = bucket_count[bid];
                            const size_t row_bytes = (blob_mode == 1) ? static_cast<size_t>(blob_stride)
                                                                      : static_cast<size_t>(D) * sizeof(DataT);
                            if (off + static_cast<size_t>(cnt) * row_bytes > blob_bytes ||
                                dofs + static_cast<size_t>(cnt) > max_pool_size_ub)   // bounds check
                                throw std::runtime_error("VectorKNN: blob bucket / pool out of range at pos " + std::to_string(pos));
                            h_jobs[slot][nj++] = DevPoolJob{nullptr, d_blob + off, static_cast<int32_t>(cnt),
                                                            static_cast<int32_t>(dofs), gid0, blob_stride, blob_mode};
                            dofs += static_cast<size_t>(cnt);
                            return;
                        }
                        auto ga0 = diag_now();
                        const DevBucket& db = dev_cache->get(pos, static_cast<int32_t>(bid), dev_loader);
                        t_cache_get += diag_secs(ga0, diag_now());
                        db.last_pos = pos;
                        if (dofs + static_cast<size_t>(db.count) > max_pool_size_ub)   // bounds check
                            throw std::runtime_error("VectorKNN: pool overflow at pos " + std::to_string(pos));
                        h_jobs[slot][nj++] = DevPoolJob{db.d_ids, db.d_vecs, static_cast<int32_t>(db.count),
                                                        static_cast<int32_t>(dofs), db.gid0, db.stride, db.mode};
                        dofs += static_cast<size_t>(db.count);
                    };
                    add_dev(c);
                    for (uint32_t k = 0; k < K; ++k) {
                        uint32_t nb_c = centroid_knn_graph[c * K + k];
                        if (nb_c < static_cast<uint32_t>(n_centroids)) add_dev(static_cast<int64_t>(nb_c));
                    }
                    if (n_miss > 0) {   // 这个 position 的未命中: 一次 H2D (数据 + job 表) + 一个分发 kernel
                        auto tc0 = diag_now();
                        CUDA_CHECK(cudaMemcpyAsync(d_dstage[slot], h_dstage[slot], stage_off, cudaMemcpyHostToDevice, copy_stream));
                        CUDA_CHECK(cudaMemcpyAsync(d_dstage[slot] + dstage_cap, h_mjobs[slot],
                                                   static_cast<size_t>(n_miss) * sizeof(MissJob), cudaMemcpyHostToDevice, copy_stream));
                        devcache_scatter_misses_kernel<<<n_miss, 256, 0, copy_stream>>>(
                            d_dstage[slot], reinterpret_cast<const MissJob*>(d_dstage[slot] + dstage_cap));
                        CUDA_CHECK(cudaGetLastError());
                        t_dev_copy += diag_secs(tc0, diag_now());
                    }
                    CUDA_CHECK(cudaEventRecord(ev_load[slot], copy_stream));
                    n_jobs_staged[slot] = nj;
                    Staged& st = staged[slot];
                    st.pool_size   = static_cast<int>(dofs);
                    st.bucket_size = static_cast<int>(bucket_count[c]);
                    st.actual_M    = std::min(M, st.pool_size - 1);
                    if (st.actual_M > 0) {
                        // 自己桶的 gid，给 scatter 用 (记录在 host 内存里时直接从 host blob 取)
                        if (h_blob) {
                            size_t off = 0; int64_t gid0 = 0;
                            bucket_vecs.blob_locate(c, off, gid0);
                            for (int i = 0; i < st.bucket_size; ++i) {
                                int64_t g = gid0 + i;
                                if (blob_mode == 1)
                                    std::memcpy(&g, h_blob + off + static_cast<size_t>(i) * blob_stride, sizeof(int64_t));
                                if (g < 0 || g >= N) throw std::runtime_error("VectorKNN: blob gid out of range");   // bounds check
                                staged_ids_bucket[slot][static_cast<size_t>(i)] = static_cast<int32_t>(g);
                            }
                        } else {
                        const VecBucket<DataT>& hc = bucket_cache.get(pos, static_cast<int32_t>(c), bucket_loader);
                        if (static_cast<int>(hc.ids.size()) != st.bucket_size)
                            throw std::runtime_error("VectorKNN: center bucket size mismatch");
                        for (int i = 0; i < st.bucket_size; ++i)
                            staged_ids_bucket[slot][static_cast<size_t>(i)] = static_cast<int32_t>(hc.ids[static_cast<size_t>(i)]);
                        }
                    }
                    // blob 模式: gather 也由 producer 发 (同一 slot stream 上，主线程已经
                    // 发完上一个桶的 GEMM/top-M/D2H 才会放出这个 slot，stream 顺序保证
                    // 不会覆盖还没读完的 pool)；主线程只剩 GEMM + top-M + D2H
                    if (d_blob && st.actual_M > 0) {
                        if (!jobs_mapped)
                            CUDA_CHECK(cudaMemcpyAsync(d_jobs[slot], h_jobs[slot], static_cast<size_t>(nj) * sizeof(DevPoolJob),
                                                       cudaMemcpyHostToDevice, streams[slot]));
                        devcache_gather_pool_kernel<DataT, GemmInT><<<nj, 256, 0, streams[slot]>>>(
                            jobs_mapped ? h_jobs[slot] : d_jobs[slot], reinterpret_cast<GemmInT*>(d_B[slot]),
                            d_ids_pool[slot], d_norms_pool[slot], static_cast<int>(D));
                        CUDA_CHECK(cudaGetLastError());
                        CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot]));   // job 表读完 (下次重写前等)
                    }
                    t_poolbuild += diag_secs(pb0, diag_now());
                    {
                        std::lock_guard<std::mutex> lk(stage_mtx);
                        if (d_blob && st.actual_M > 0) h2d_pending[slot] = true;
                        slot_ready_pos[slot] = pos;
                    }
                    stage_cv.notify_all();
                    continue;
                }

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
                submitted_pos = pos;   // 这个位置没有 gather (见 dev 缓存的淘汰等待)
            }
            stage_cv.notify_all();
            t_wait_producer += diag_secs(dt2, diag_now());
            continue;
        }
        std::memcpy(h_ids_bucket[slot], staged_ids_bucket[slot].data(),
                    static_cast<size_t>(bucket_size) * sizeof(int32_t));
        auto dt3 = diag_now();
        t_wait_producer += diag_secs(dt2, dt3);

        if (use_dev_cache && d_blob) {
            // blob 模式: gather 已经由 producer 发到这个 slot 的 stream 上了；slot 等
            // 本桶的 GEMM/top-M/D2H 都发完再放出 (见 3g 之后)
            if (actual_M > 256)   // 只有 raft select_k 回退路径读 d_ids_bucket
                CUDA_CHECK(cudaMemcpyAsync(d_ids_bucket[slot], h_ids_bucket[slot],
                                           bucket_size * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, streams[slot]));
        } else if (use_dev_cache) {
            // ---- 3c/3d (dev 缓存): 等这个位置的桶加载完，按 job 表 gather pool ----
            const int nj = n_jobs_staged[slot];
            CUDA_CHECK(cudaStreamWaitEvent(streams[slot], ev_load[slot], 0));
            if (actual_M > 256)   // 只有 raft select_k 回退路径读 d_ids_bucket
                CUDA_CHECK(cudaMemcpyAsync(d_ids_bucket[slot], h_ids_bucket[slot],
                                           bucket_size * sizeof(int32_t),
                                           cudaMemcpyHostToDevice, streams[slot]));
            if (!jobs_mapped)
                CUDA_CHECK(cudaMemcpyAsync(d_jobs[slot], h_jobs[slot], static_cast<size_t>(nj) * sizeof(DevPoolJob),
                                           cudaMemcpyHostToDevice, streams[slot]));
            devcache_gather_pool_kernel<DataT, GemmInT><<<nj, 256, 0, streams[slot]>>>(
                jobs_mapped ? h_jobs[slot] : d_jobs[slot], reinterpret_cast<GemmInT*>(d_B[slot]), d_ids_pool[slot],
                d_norms_pool[slot], static_cast<int>(D));
            CUDA_CHECK(cudaGetLastError());
            // 生产者等 ev_h2d 之后才重写 h_jobs[slot]: 映射模式下 kernel 读完才算完
            CUDA_CHECK(cudaEventRecord(ev_h2d[slot], streams[slot]));
            {
                // gather 完成事件 (淘汰等待用)；环形表覆盖前先确认旧位置已完成
                PosEvent& pe = pos_ring[static_cast<size_t>(pos % kPosRing)];
                if (pe.pos >= 0) CUDA_CHECK(cudaEventSynchronize(pe.ev));
                std::lock_guard<std::mutex> lk(stage_mtx);
                CUDA_CHECK(cudaEventRecord(pe.ev, streams[slot]));
                pe.pos = pos;
                h2d_pending[slot] = true;
                slot_ready_pos[slot] = -1;
                submitted_pos = pos;
            }
            stage_cv.notify_all();
            // (norms 已在 gather kernel 里算好)
        } else {
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
                // pool 已经按需要的行序拼好 (identity)，直接逐元素转 fp16
                resident_convert_kernel<DataT, __half><<<blocks_B, threads, 0, streams[slot]>>>(
                    d_B_raw[slot], reinterpret_cast<__half*>(d_B[slot]), total_B);
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
        }   // !use_dev_cache

        bool fused_out = false;   // block top-M 把 dists 写在 ids 后面 (见 3g)
        // dots 行宽: block top-M kernel (actual_M ≤ 256) 要 4 对齐; raft 回退路径按
        // pool_size 紧凑排布
        const int ldd_c = (actual_M <= 256) ? (pool_size + 3) / 4 * 4 : pool_size;

        // ---- 3e: cuBLAS GEMM 在 slot stream 上 (event 包夹做纯 GEMM 计时) ----
        // bucket 行按 dots_rows 分块: 每块一个 GEMM (A = pool 的 [r0, r0+rows) 行) + 一个 top-M
        if (bucket_size > static_cast<int>(dots_rows) && actual_M > 256)   // raft 回退不分块
            throw std::runtime_error("VectorKNN: M > 256 needs bucket_size <= dots row tile");
        cublasSetStream(cublas_handle, streams[slot]);
        if (gemm_timing) CUDA_CHECK(cudaEventRecord(ev_gemm_start[slot], streams[slot]));
        for (int r0 = 0; r0 < bucket_size; r0 += static_cast<int>(dots_rows)) {
        const int rows = std::min(static_cast<int>(dots_rows), bucket_size - r0);
        {
            cublasStatus_t stat;
            if constexpr (kIsInt8Path) {
                // fp16 in × fp16 in → fp32 out (fp32 累加), Tensor Core
                float alpha = 1.0f, beta = 0.0f;
                // A 操作数直接传 d_B[slot] 本身: A 是 pool 的前 bucket_size
                // 行，跟 B 是同一份已转型数据，lda=D 不变，cuBLAS 只读前
                // bucket_size 行，不需要一份独立的 A buffer。
                stat = cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, rows, D,
                    &alpha,
                    d_B[slot], CUDA_R_16F, D,
                    d_B[slot] + static_cast<size_t>(r0) * D, CUDA_R_16F, D,
                    &beta,
                    d_dots[slot], CUDA_R_32F, ldd_c,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error(
                        "cublasGemmEx (fp16 Tensor Core) failed, status=" + std::to_string(stat));
            } else {
                float alpha = 1.0f, beta = 0.0f;
                stat = cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    pool_size, rows, D,
                    &alpha,
                    reinterpret_cast<const float*>(d_B[slot]), D,
                    reinterpret_cast<const float*>(d_B[slot]) + static_cast<size_t>(r0) * D, D,
                    &beta,
                    reinterpret_cast<float*>(d_dots[slot]), ldd_c);
                if (stat != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("cublasSgemm failed, status=" + std::to_string(stat));
            }
        }
        if (gemm_timing) {
            CUDA_CHECK(cudaEventRecord(ev_gemm_stop[slot], streams[slot]));
            gemm_evt_valid[slot] = true;
        }

        // ---- 3f: top-M ----
        // 切换实现：把下面 USE_RAFT_SELECT_K 改 0 即用自写 warp-cooperative kernel，
        //          改 1 即用 RAFT select_k (radix / warp-sort auto)。
        // 两个 dispatcher 都是模板化 GemmOutT (int32 INT8 路径 / float TF32 路径)。
        #ifndef USE_RAFT_SELECT_K
        #define USE_RAFT_SELECT_K 1
        #endif
        {
        #if USE_RAFT_SELECT_K
            // actual_M ≤ 256: 单个 block top-M kernel (跟 GPU-resident 同一个，
            // 输出按 bucket 内行号)，替代 dots_to_dist + raft select_k + gather
            // 三个 kernel；更大的 M 回退到 raft select_k。
            fused_out = launch_block_topM_bucket<GemmOutT>(
                    streams[slot], d_dots[slot], d_norms_pool[slot], d_ids_pool[slot],
                    d_out_neighbors[slot],
                    reinterpret_cast<float*>(d_out_neighbors[slot] + static_cast<size_t>(bucket_size) * actual_M),
                    rows, pool_size, ldd_c, actual_M, /*out_ld=*/actual_M,
                    d_seed_kth, d_seed_kth ? N : 0, r0);
            if (!fused_out)
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
        }   // r0 行分块

        // ---- 3g: 异步 D2H, 不 sync ----
        // block top-M 路径: ids + dists 在同一块连续 buffer 里，一次拷回
        CUDA_CHECK(cudaMemcpyAsync(h_out[slot], d_out_neighbors[slot],
                                   static_cast<size_t>(bucket_size) * actual_M * sizeof(int32_t) * (fused_out ? 2 : 1),
                                   cudaMemcpyDeviceToHost, streams[slot]));
        if (want_distances && !fused_out) {
            CUDA_CHECK(cudaMemcpyAsync(h_out_dists[slot], d_out_dists[slot],
                                       static_cast<size_t>(bucket_size) * actual_M * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[slot]));
        }

        if (use_dev_cache && d_blob) {   // 本桶全部发完，放出 slot 给 producer 发下一个 gather
            {
                std::lock_guard<std::mutex> lk(stage_mtx);
                slot_ready_pos[slot] = -1;
                submitted_pos = pos;
            }
            stage_cv.notify_all();
        }
        t_submit += diag_secs(dt3, diag_now());

        pending[slot] = {c, bucket_size, actual_M, fused_out};
        ++processed_buckets;
    }

    prod_thread.join();   // 已经全部消费完；ProducerJoin 析构时 joinable() 为 false

    // ---- Tail flush: 处理两 slot 上的最后未消费结果 ----
    for (int s = 0; s < num_slots; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        consume_gemm_timing(s);
        scatter_pending(s);
    }
    // 等 scatter 线程把队列处理完 (acc 之后 finish_iteration_writes 才能安全用)
    {
        auto tj0 = std::chrono::high_resolution_clock::now();
        { std::lock_guard<std::mutex> lk(scatter_mtx); scatter_done = true; }
        scatter_cv.notify_all();
        scatter_thread.join();
        if (scatter_exc) std::rethrow_exception(scatter_exc);
        std::cout << "  [VectorKNN] scatter worker: " << std::fixed << std::setprecision(3)
                  << t_scatter_worker << "s (tail drain "
                  << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tj0).count()
                  << "s)\n";
    }

    auto loop_t1 = std::chrono::high_resolution_clock::now();
    double loop_ms = std::chrono::duration<double, std::milli>(loop_t1 - loop_t0).count();
    std::cout << "  [VectorKNN] bucket loop: " << loop_ms << " ms over "
              << processed_buckets << " buckets ("
              << (loop_ms / std::max<int64_t>(1, processed_buckets))
              << " ms/bucket avg, slots=" << num_slots << ")\n";
    if (gemm_timing)
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
    if (use_dev_cache) {
        std::cout << "  [VectorKNN] GPU bucket cache: " << n_dev_loads << " bucket uploads, upload "
                  << std::fixed << std::setprecision(3) << t_dev_upload << "s (host get " << t_dev_hostget
                  << "s, malloc " << t_dev_malloc << "s, h2d call " << t_dev_copy << "s, deferred free " << t_free_drain << "s, device allocs " << n_arena_fallback << "), evictions "
                  << dev_cache->evictions() << "\n";
        dev_cache.reset();                       // 释放全部 device 桶 (都已提交)
        drain_pending_free(true);
        for (auto& kv : arena_free) for (char* b : kv.second) cudaFreeAsync(b, copy_stream);
        arena_free.clear(); arena_free_bytes = 0;
        CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        CUDA_CHECK(cudaStreamDestroy(copy_stream));
        for (auto& pe : pos_ring) cudaEventDestroy(pe.ev);
        for (int sl = 0; sl < num_slots; ++sl) {
            cudaFreeHost(h_jobs[sl]); cudaFree(d_jobs[sl]); cudaEventDestroy(ev_load[sl]);
            if (h_dstage[sl]) { cudaFreeHost(h_dstage[sl]); cudaFree(d_dstage[sl]); }
            if (sl == 0 && d_blob) { cudaFree(d_blob); d_blob = nullptr; }
        }
    }
    cudaFree(d_seed_kth);
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
// GPU-resident 路径的收尾 (--fused-optimize-degree，用户 2026-10-21 同意把
// optimize + translate_ids 合进 bucket2 进程)：跟独立的 optimize / translate_ids
// 做的事完全一样——(N, M) 候选表 sanitize 成 uint32 → cuVS
// cagra::helpers::optimize (CAGRA rank-based prune + 反向边 + merge，跟 optimize.cu
// 同一个调用) → 按 inverse_perm 把行和邻居 id 翻回原始顺序 → 写 final_graph.bin
// (跟 translate_ids 的输出格式相同: int64 N, int32 K, uint32 [N, K])。省掉的是
// 两个进程的启动 (CUDA context)、256 MB 候选表和 128 MB 中间图的写盘/读盘。
static void fused_optimize_translate(const std::vector<int32_t>& knn, int64_t N, int M, int K_out,
                                     const std::vector<uint32_t>& inverse_perm, const std::string& out_path)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    if (K_out <= 0 || K_out > M)
        throw std::runtime_error("fused optimize: --fused-optimize-degree must be in [1, " + std::to_string(M) + "]");
    if (static_cast<int64_t>(knn.size()) != N * M || static_cast<int64_t>(inverse_perm.size()) != N)
        throw std::runtime_error("fused optimize: size mismatch (knn / inverse_perm vs N)");

    // ---- sanitize (同 optimize.cu sanitize_to_uint32_inplace): padding (-1) / 越界 /
    //      自环换成该行第一个合法邻居，没有则 (i+1)%N；knn 还在被写盘线程读，另开一份 ----
    // 不清零的 buffer: 缺页由下面的并行循环 first-touch (std::vector 会单线程清零 256 MB)
    std::unique_ptr<uint32_t[]> g(new uint32_t[static_cast<size_t>(N) * M]);
    int64_t n_pad = 0;
    #pragma omp parallel for reduction(+:n_pad) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const int32_t* row = knn.data() + i * M;
        uint32_t* out = g.get() + i * M;
        uint32_t fallback = static_cast<uint32_t>((i + 1) % N);
        for (int j = 0; j < M; ++j) {
            const int32_t v = row[j];
            if (v >= 0 && v < N && v != i) { fallback = static_cast<uint32_t>(v); break; }
        }
        for (int j = 0; j < M; ++j) {
            const int32_t v = row[j];
            if (v < 0 || v >= N || v == i) { out[j] = fallback; if (v < 0) ++n_pad; }
            else out[j] = static_cast<uint32_t>(v);
        }
    }
    auto t1 = Clock::now();

    // ---- prune + 反向边 (cuVS 26.10 public API，跟 optimize.cu 相同) ----
    auto pruned = raft::make_host_matrix<uint32_t, int64_t>(N, K_out);
    {
        raft::resources res;
        cuvs::neighbors::cagra::helpers::optimize(
            res, raft::make_host_matrix_view<uint32_t, int64_t>(g.get(), N, M), pruned.view());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    g.reset();
    auto t2 = Clock::now();

    // ---- id 翻译 (同 translate_graph_ids.hpp): final[inv[i]][k] = inv[pruned[i][k]] ----
    std::unique_ptr<uint32_t[]> fin(new uint32_t[static_cast<size_t>(N) * K_out]);
    std::vector<uint8_t> hit(static_cast<size_t>(N), 0);
    int64_t n_bad = 0;
    const uint32_t* pr = pruned.data_handle();
    #pragma omp parallel for reduction(+:n_bad) schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const uint32_t old_row = inverse_perm[static_cast<size_t>(i)];
        if (old_row >= static_cast<uint64_t>(N)) { ++n_bad; continue; }   // bounds check
        hit[old_row] = 1;
        uint32_t* dst = fin.get() + static_cast<size_t>(old_row) * K_out;
        const uint32_t* src = pr + static_cast<size_t>(i) * K_out;
        for (int k = 0; k < K_out; ++k) {
            uint32_t v = src[k];
            if (v >= static_cast<uint64_t>(N)) { ++n_bad; v = 0; }   // bounds check
            dst[k] = inverse_perm[v];
        }
    }
    if (n_bad > 0)
        throw std::runtime_error("fused optimize: " + std::to_string(n_bad) + " out-of-range ids during translation");
    int64_t n_hit = 0;
    #pragma omp parallel for reduction(+:n_hit) schedule(static)
    for (int64_t i = 0; i < N; ++i) n_hit += hit[static_cast<size_t>(i)];
    if (n_hit != N)   // inverse_perm 是置换 ⇔ 每个目标行恰好写到一次
        throw std::runtime_error("fused optimize: inverse_perm is not a permutation (" +
                                 std::to_string(n_hit) + " of " + std::to_string(N) + " rows hit)");
    auto t3 = Clock::now();

    {
        std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
        const int32_t K32 = K_out;
        out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
        out.write(reinterpret_cast<const char*>(&K32), sizeof(int32_t));
        out.write(reinterpret_cast<const char*>(fin.get()),
                  static_cast<std::streamsize>(static_cast<size_t>(N) * K_out * sizeof(uint32_t)));
        if (!out.good()) throw std::runtime_error("fused optimize: write failed: " + out_path);
    }
    auto t4 = Clock::now();
    auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    std::cout << "  [FusedOptimize] K_out=" << K_out << " padding=" << n_pad
              << " sanitize=" << std::fixed << std::setprecision(3) << sec(t0, t1)
              << "s optimize=" << sec(t1, t2) << "s translate=" << sec(t2, t3)
              << "s write=" << sec(t3, t4) << "s -> " << out_path << "\n";
}

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
        // 非空时: 当前坐标系 id → X_sampled 的行号 (round 1 重排后 X_sampled 仍是
        // 旧坐标系，只在 KMeans|| 抽训练子集时按它 gather，省掉整份重排)
        std::vector<uint32_t> x_row_of;
        // true: X_sampled 先不建 (out-of-core 且原始 u8 数据集整份留在 raw_host 时，
        // Step 2 直接从 raw_host 抽 kmeans_train_fraction 的子集转 float——不再
        // 整份转一份 N*D*4 B 的 float (SIFT10M 5.1 GB，Step 1 单线程清零 + 转换)
        bool x_lazy = false;

        // 采样逻辑抽成 lambda：round 1 重排完之后，如果 iterations > 1，要对着
        // 新坐标系的 data_reordered.<ext> 重新采一遍（否则第 2..t 轮选出来的
        // centroid 会是旧坐标系的 ID，去新数据集里找点就全错了）——用同一份
        // 逻辑读不同路径，不能各写一份容易漂移。N/D/mem_est 采样前后不变
        // （round 1 重排不改变点数/维度，只改变 ID 的物理排列），可以直接复用。
        // 原始 (N, D) 数据集的 host 副本 (DataT)：GPU-resident 路径建常驻数据集 /
        // round 1 重排时用，out-of-core 预算够时跨轮复用 (见下面 ooc_host_raw)。
        std::vector<DataT> raw_host;
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
                } else if (ext == ".u8bin" || ext == ".i8bin") {
                    // 只读采样行 (原来整份读成 float 再挑: SIFT100M 51 GB 缓冲, Step 1 + Step 4.5 resample 各 ~70 s)
                    int32_t n_tmp, d_tmp;
                    load::read_u8bin_sampled_to_f32(path, sorted_order, X_sampled, n_tmp, d_tmp);
                } else {
                    // ibin: fallback 到全量读取再采样
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
                } else if (ext == ".u8bin" && std::is_same<DataT, uint8_t>::value && raw_host.empty()) {
                    // 原始 u8 只读一次进 raw_host (后面 GPU-resident / out-of-core 都要用)，
                    // 再并行转 float——原来先 read_u8bin_to_f32 (单线程转换) 再整份重读一遍
                    int32_t n_tmp = 0, d_tmp = 0;
                    load::read_bigann_raw_range<DataT>(path, 0, N, raw_host, n_tmp, d_tmp);
                    if (static_cast<int64_t>(raw_host.size()) != static_cast<int64_t>(N) * D)
                        throw std::runtime_error("Step 1: raw dataset size mismatch");
                    if (config.kmeans_train_fraction < 1.0f && !mem_est.need_pq) {
                        x_lazy = true;   // 等确定走不走 out-of-core host raw 再决定 (见下)
                    } else {
                        X_sampled.resize(raw_host.size());
                        #pragma omp parallel for schedule(static)
                        for (int64_t i = 0; i < static_cast<int64_t>(raw_host.size()); ++i)
                            X_sampled[static_cast<size_t>(i)] = static_cast<float>(raw_host[static_cast<size_t>(i)]);
                    }
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
        // --fused-optimize-degree: round 1 重排的 inverse_perm 留到最后做 id 翻译
        std::vector<uint32_t> fused_inv_perm;
        // round 1 重排后的 data_reordered.<ext> 在后台线程写 (进程内后面的步骤读的是
        // raw_host / 显存里的数据，只有 prune 等后续进程要这个文件)；在任何可能读它
        // 的地方之前、释放它所用的缓冲之前、以及进程结束前 join
        std::future<void> reorder_write;
        auto join_reorder_write = [&] { if (reorder_write.valid()) reorder_write.get(); };
        auto write_reordered_async = [&](const std::string& path, const DataT* p, size_t n,
                                         std::shared_ptr<std::vector<DataT>> holder) {
            const int32_t n32 = static_cast<int32_t>(N), d32 = static_cast<int32_t>(D);
            // 文件在这里同步创建: blob 模式下 round 1 的 reader 不 join 就构造 (只 open、不读)，
            // 文件若还没被后台线程建出来就会 "cannot open" (SIFT1M large 偶发)
            std::ofstream out_file(path, std::ios::binary | std::ios::trunc);
            if (!out_file.is_open()) throw std::runtime_error("Inline reorder: cannot create " + path);
            reorder_write = std::async(std::launch::async, [out = std::move(out_file), path, n32, d32, p, n, holder]() mutable {
                int32_t hdr[2] = {n32, d32};
                out.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
                out.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n * sizeof(DataT)));
                if (!out.good()) throw std::runtime_error("Inline reorder: failed writing " + path);
            });
        };
        const bool raw_preloaded = !raw_host.empty();
        if (neighbors_m > 0 && neighbors_m <= 256 && mem_est.fits_in_gpu && !force_out_of_core) {
            size_t need = GpuResidentKnn<DataT>::bytes_needed(N, D, neighbors_m);
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            if (need <= static_cast<size_t>(free_b * 0.6)) {
                int32_t n_tmp, d_tmp;
                if (!raw_preloaded)
                    load::read_bigann_raw_range<DataT>(input_path, 0, N, raw_host, n_tmp, d_tmp);
                gres = std::make_unique<GpuResidentKnn<DataT>>(raw_host.data(), N, D, neighbors_m);
                std::cout << "  [GpuResidentKNN] enabled: " << need / 1e9 << " GB resident (free "
                          << free_b / 1e9 << " GB)\n";
            } else {
                std::cout << "  [GpuResidentKNN] disabled: needs " << need / 1e9 << " GB, free "
                          << free_b / 1e9 << " GB -> out-of-core path\n";
            }
        }
        // out-of-core 但原始数据 (两份: 本体 + round 1 重排时的新副本) 放得进
        // CPU 预算的 1/4 时：整份原始数据集读进 host 内存一次 (round 1 重排后
        // 换成新坐标系那份)，Step 4 的分配和 round 2..t 的 bucket 写入都直接
        // 用它，不再每轮读盘 2~3 遍。超预算时 (SIFT10M 以上) 照旧按区间读盘。
        // 预算: 给了 --bucket-mem-mb (run_pipeline 按数据集算好的 host 记录预算) 就用它，否则
        // cpu-limit/4 (P2.4b: msTuring10M float 两份 8 GB > 16 GB/4，原来 Step 4 每轮重读 3.3 s)
        const size_t host_raw_budget = config.bucket_mem_mb > 0 ? (config.bucket_mem_mb << 20)
                                                                : config.cpu_limit_bytes / 4;
        const bool ooc_host_raw = !gres &&
            static_cast<size_t>(N) * D * sizeof(DataT) * 2 <= host_raw_budget;
        if (x_lazy && !ooc_host_raw) {
            // raw_host 不会跨轮保留 (GPU-resident / 超预算): 照旧整份转 float
            X_sampled.resize(raw_host.size());
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(raw_host.size()); ++i)
                X_sampled[static_cast<size_t>(i)] = static_cast<float>(raw_host[static_cast<size_t>(i)]);
            x_lazy = false;
        }
        if (ooc_host_raw) {
            int32_t n_tmp = 0, d_tmp = 0;
            if (!raw_preloaded)
                load::read_bigann_raw_range<DataT>(input_path, 0, N, raw_host, n_tmp, d_tmp);
            if (static_cast<int64_t>(raw_host.size()) != N * static_cast<int64_t>(D))
                throw std::runtime_error("Step 1: host raw dataset size mismatch");
            std::cout << "  [HostRaw] out-of-core dataset kept in host memory: "
                      << raw_host.size() * sizeof(DataT) / 1e9 << " GB\n";
        } else if (!gres) {
            raw_host = std::vector<DataT>();   // 预读了但用不上 (超预算)
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

        // KMeans|| 训练子集的 host 缓冲：pinned、跨轮复用 (只分配/缺页一次，
        // 不再每轮 resize 时单线程清零 128 MB；pinned 也让 Step 2 的 H2D 更快)。
        struct PinnedFree { void operator()(float* p) const { if (p) cudaFreeHost(p); } };
        std::unique_ptr<float, PinnedFree> X_train_pinned;
        size_t X_train_cap = 0;   // floats

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

        // KMeans|| 训练子集 (跟 cuVS IVF 的 kmeans_trainset_fraction 一样):
        // 按固定 seed 对每个点做 Bernoulli(frac) 抽样 (不重复、确定性)，只在
        // 子集上做 D² 过采样；centroid 仍是真实数据点、数量仍是 N*ratio。
        // KMeans|| 的代价 ∝ 点数 × 候选数，子集越小越快。
        const float* X_train = X_sampled.data();
        int64_t train_N = working_N;
        std::vector<int64_t> train_indices;   // train 行 → raw idx
        const bool sub_train = iter_config.kmeans_train_fraction < 1.0f && !mem_est.need_pq;
        if (sub_train) {
            std::mt19937_64 rng(static_cast<uint64_t>(iter_config.seed) * 0x9E3779B97F4A7C15ULL + 1);
            std::bernoulli_distribution keep(iter_config.kmeans_train_fraction);
            std::vector<int64_t> rows;
            rows.reserve(static_cast<size_t>(working_N * iter_config.kmeans_train_fraction * 1.1) + 16);
            for (int64_t i = 0; i < working_N; ++i) if (keep(rng)) rows.push_back(i);
            // KMeans|| 需要的候选数 ~1.5x centroids，留足余量
            if (static_cast<int64_t>(rows.size()) < 4 * mem_est.centroid_rows) {
                rows.resize(static_cast<size_t>(working_N));
                std::iota(rows.begin(), rows.end(), 0LL);
            }
            train_N = static_cast<int64_t>(rows.size());
            const size_t need_f = static_cast<size_t>(train_N) * D;
            if (need_f > X_train_cap) {
                X_train_pinned.reset();
                float* p = nullptr;
                // 余量 5%: 各轮 Bernoulli 抽到的点数略有不同，避免每轮重新分配
                const size_t cap = need_f + need_f / 20;
                CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&p), cap * sizeof(float), cudaHostAllocDefault));
                X_train_pinned.reset(p);
                X_train_cap = cap;
            }
            float* X_train_dst = X_train_pinned.get();
            train_indices.resize(static_cast<size_t>(train_N));
            if (x_lazy) {
                // raw_host 是当前坐标系的整份原始数据 (round 1 重排后换成新坐标系那份)，
                // sampled_indices = iota: 直接按行号取、转 float
                if (static_cast<int64_t>(raw_host.size()) != static_cast<int64_t>(working_N) * D)
                    throw std::runtime_error("Step 2: lazy train subset needs the whole raw dataset in host memory");
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < train_N; ++i) {
                    const int64_t r = rows[static_cast<size_t>(i)];   // < working_N (Bernoulli over [0, working_N))
                    const DataT* srow = raw_host.data() + static_cast<size_t>(r) * D;
                    float* drow = X_train_dst + static_cast<size_t>(i) * D;
                    for (int64_t d = 0; d < D; ++d) drow[d] = static_cast<float>(srow[d]);
                    train_indices[static_cast<size_t>(i)] = sampled_indices[static_cast<size_t>(r)];
                }
            } else {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < train_N; ++i) {
                std::memcpy(X_train_dst + static_cast<size_t>(i) * D,
                            X_sampled.data() + static_cast<size_t>(x_row_of.empty()
                                ? rows[static_cast<size_t>(i)]
                                : x_row_of[static_cast<size_t>(rows[static_cast<size_t>(i)])]) * D,
                            static_cast<size_t>(D) * sizeof(float));
                train_indices[static_cast<size_t>(i)] = sampled_indices[static_cast<size_t>(rows[static_cast<size_t>(i)])];
            }
            }
            X_train = X_train_dst;
            std::cout << "  KMeans|| train subset: " << train_N << " / " << working_N << " points\n";
        }
        const std::vector<int64_t>& train_map = sub_train ? train_indices : sampled_indices;

        // 返回 centroid 在训练集里的行号
        std::vector<int64_t> centroid_sample_indices = select_centroids_on_gpu_kmeans_parallel(
            X_train, train_map,
            train_N, D, iter_config, mem_est, &quantizer, &d_centroids_f32,
            std::is_same<DataT, uint8_t>::value || std::is_same<DataT, int8_t>::value);
        double t2_select_call = std::chrono::duration<double>(Clock::now() - t2).count();

        // 训练行号 → raw_idx (bounds check)
        centroid_global_indices.assign(centroid_sample_indices.size(), 0);
        for (size_t k = 0; k < centroid_sample_indices.size(); ++k) {
            int64_t r = centroid_sample_indices[k];
            if (r < 0 || r >= train_N)
                throw std::runtime_error("Step 2: centroid row " + std::to_string(r) + " out of range");
            centroid_global_indices[k] = train_map[static_cast<size_t>(r)];
        }

        n_centroids = static_cast<int64_t>(centroid_global_indices.size());
        {   // 每个 centroid 必须是不同的数据点 (重复 → 某个 bucket 拿不到自己的 centroid)
            std::vector<int64_t> chk(centroid_global_indices);
            std::sort(chk.begin(), chk.end());
            if (std::adjacent_find(chk.begin(), chk.end()) != chk.end())
                throw std::runtime_error("Step 2: duplicate centroid ids");
        }
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
            bucket_vecs_ptr->set_in_memory_budget(config.bucket_mem_mb > 0 ? (config.bucket_mem_mb << 20)
                                                                           : config.cpu_limit_bytes / 4);
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
            centroid_exact_knn = centroid_self_knn_bf(d_centroids_f32, n_centroids, D, exact_knn_k,
                std::is_same<DataT, uint8_t>::value || std::is_same<DataT, int8_t>::value);
            std::vector<uint32_t> g(static_cast<size_t>(n_centroids) * K);
            for (int64_t c = 0; c < n_centroids; ++c)
                std::memcpy(g.data() + static_cast<size_t>(c) * K,
                            centroid_exact_knn.data() + static_cast<size_t>(c) * exact_knn_k,
                            K * sizeof(uint32_t));
            CUDA_CHECK(cudaMalloc(&d_graph, g.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(d_graph, g.data(), g.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
            if (iter == 0 && nprobe > 0) {
                // A4 (user-approved): round 1 的 Step 6 pool 表 (每桶前 nprobe 个近邻桶，跟 Step 6 用的是同一张
                // 精确自 KNN 的前 nprobe 列)，给 prune_windowed --pool-graph 当窗口；格式同 centroid_knn.bin
                const std::string p = output_dir + "/centroid_pool.bin";
                std::ofstream out(p, std::ios::binary);
                const int64_t nc = n_centroids; const int32_t Pw = static_cast<int32_t>(nprobe);
                out.write(reinterpret_cast<const char*>(&nc), sizeof(int64_t));
                out.write(reinterpret_cast<const char*>(&Pw), sizeof(int32_t));
                for (int64_t c = 0; c < n_centroids; ++c)
                    out.write(reinterpret_cast<const char*>(centroid_exact_knn.data() + static_cast<size_t>(c) * exact_knn_k),
                              static_cast<std::streamsize>(nprobe * sizeof(uint32_t)));
                if (!out.good()) throw std::runtime_error("write failed: " + p);
            }
        } else {
            build_centroid_knn_on_gpu(
                d_centroids_f32, n_centroids, D, K, &d_graph);
        }

        size_t centroid_gpu_bytes = static_cast<size_t>(n_centroids) * D * sizeof(float);
        size_t graph_bytes = static_cast<size_t>(n_centroids) * K * sizeof(uint32_t);
        cudaDeviceSynchronize();

        // bucket 处理顺序 (DiskJoin 贪心) 只依赖 centroid KNN 图：图一出来就在
        // 后台线程算，跟 Step 4 的分配重叠 (原来 round 1 在 Step 4.5、round ≥2
        // 在 Step 6 setup 里串行算，各 ~0.3 s)。图先下载到 host (原来在 Step 4
        // 之后下载)；输入、order_window 跟两个原调用点完全一样 → 结果逐位相同。
        std::future<std::vector<int32_t>> order_future;
        const bool need_order = (iter == 0) || (neighbors_m > 0 && !gres);
        if (neighbors_m > 0 || iter == 0) {
            centroid_knn_graph_host.resize(static_cast<size_t>(n_centroids) * K);
            CUDA_CHECK(cudaMemcpy(centroid_knn_graph_host.data(), d_graph,
                                  graph_bytes, cudaMemcpyDeviceToHost));
        }
        if (need_order) {
            const int32_t ow = (order_window_arg > 0)
                ? order_window_arg
                : std::max<int32_t>(4 * static_cast<int32_t>(K), 16);
            order_future = std::async(std::launch::async,
                [graph = centroid_knn_graph_host, nc = n_centroids, k = static_cast<int32_t>(K), ow]() {
                    return bucket_order::compute_bucket_processing_order(
                        bucket_order::adjacency_from_flat_graph(graph.data(), nc, k), ow);
                });
        }
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
                K, quantizer, Mbatch_bytes, search_max_iters, {},
                ooc_host_raw ? raw_host.data() : nullptr);

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
                K, quantizer, Mbatch_bytes, search_max_iters, loader,
                ooc_host_raw ? raw_host.data() : nullptr);

            CUDA_CHECK(cudaFree(d_centroids_f32));
        }
        double t4_assign_call = std::chrono::duration<double>(Clock::now() - t4).count() - t4_centroids_dl;

        // Download centroid KNN graph to CPU before freeing (needed for Step 6,
        // and for round 1's inline reorder just below, which always runs).
        // (centroid KNN 图已在 Step 3 之后下载到 centroid_knn_graph_host)

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
                centroid_global_indices, assignments, *bucket_vecs_ptr,
                512LL * 1024 * 1024, ooc_host_raw ? raw_host.data() : nullptr);
            // 最后一轮写完 bucket 之后 Step 6 只读 bucket_vecs，原始数据不再需要
            if (ooc_host_raw && iter + 1 == iterations) { join_reorder_write(); raw_host = std::vector<DataT>(); }
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
            // Step 3 之后已经在后台算好 (见 order_future)
            bucket_process_order_round1 = order_future.get();
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
            if (gres && config.fused_optimize_degree > 0) fused_inv_perm = reorder_info.inverse_perm;

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
                auto holder = std::make_shared<std::vector<DataT>>(std::move(reordered));
                write_reordered_async(data_reordered_path, holder->data(), holder->size(), holder);
                gres->permute(reorder_info.inverse_perm);
            } else if (static_cast<size_t>(N) * D * sizeof(DataT) * 2 <= host_raw_budget) {
                // out-of-core 但原始数据 (两份: 读入 + 重排后) 放得进 CPU 预算的
                // 1/4: 跟 GPU-resident 一样整份读入、按 inverse_perm 并行重排、一次
                // 顺序写出 (SIFT1M 128 MB: 1.0 s → ~0.3 s)。超预算时走下面按桶缓冲
                // 散写的流式 writer (内存只占 bucket_vec_buffer)。
                std::vector<DataT> raw_all;
                if (ooc_host_raw) {
                    raw_all = std::move(raw_host);   // Step 1 已整份读入 (旧坐标系)
                } else {
                    int32_t n_tmp = 0, d_tmp = 0;
                    load::read_bigann_raw_range<DataT>(working_input_path, 0, N, raw_all, n_tmp, d_tmp);
                }
                if (static_cast<int64_t>(raw_all.size()) != N * static_cast<int64_t>(D) ||
                    static_cast<int64_t>(reorder_info.inverse_perm.size()) != N)
                    throw std::runtime_error("Inline reorder: size mismatch in in-memory path");
                std::vector<DataT> reordered(raw_all.size());
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < N; ++i) {
                    std::memcpy(reordered.data() + static_cast<size_t>(i) * D,
                                raw_all.data() + static_cast<size_t>(reorder_info.inverse_perm[i]) * D,
                                static_cast<size_t>(D) * sizeof(DataT));
                }
                raw_all = std::vector<DataT>();
                // 重排后的副本就是新坐标系下的原始数据集：round 2..t 继续用它 (后台写文件
                // 直接用 raw_host 的缓冲，释放 raw_host 之前 join)；否则交给 holder 保活
                if (ooc_host_raw && iterations > 1) {
                    raw_host = std::move(reordered);
                    write_reordered_async(data_reordered_path, raw_host.data(), raw_host.size(), nullptr);
                } else {
                    auto holder = std::make_shared<std::vector<DataT>>(std::move(reordered));
                    write_reordered_async(data_reordered_path, holder->data(), holder->size(), holder);
                }
            } else {
                // 输出就是输入行的一个置换 (新行 i = 旧行 inverse_perm[i])。原来单线程逐行扫输入、
                // 塞进 n_centroids 个每桶 ~10 KB 的小缓冲再零散 pwrite (SIFT100M pass3_write 55 s)。
                // 现在按输出行分窗 (≤ 4 GB 一窗)：每窗并行 pread 整份输入 (8 MB 块)，只把落在窗里的
                // 行按 perm 放进窗缓冲，再一次顺序写出。结果逐字节相同。
                // 参考 TAGORE prep: 直接按目标位置成块写，不走逐条小缓冲。
                if (static_cast<int64_t>(reorder_info.inverse_perm.size()) != N)
                    throw std::runtime_error("Inline reorder: inverse_perm size != N");
                const int64_t row_bytes = static_cast<int64_t>(D) * static_cast<int64_t>(sizeof(DataT));
                std::vector<uint32_t> perm(static_cast<size_t>(N));   // 旧 id → 新 id
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < N; ++i) {
                    const uint32_t o = reorder_info.inverse_perm[static_cast<size_t>(i)];
                    if (static_cast<int64_t>(o) >= N) throw std::runtime_error("Inline reorder: inverse_perm out of range");   // bounds check
                    perm[o] = static_cast<uint32_t>(i);
                }
                const int in_fd = ::open(working_input_path.c_str(), O_RDONLY);
                if (in_fd < 0) throw std::runtime_error("Inline reorder: cannot open " + working_input_path);
                const int out_fd = ::open(data_reordered_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
                if (out_fd < 0) { ::close(in_fd); throw std::runtime_error("Inline reorder: cannot create " + data_reordered_path); }
                auto pio = [](bool wr, int fd, char* p, size_t n, off_t off) {
                    while (n > 0) {
                        ssize_t r = wr ? ::pwrite(fd, p, n, off) : ::pread(fd, p, n, off);
                        if (r <= 0) return false;
                        p += r; n -= static_cast<size_t>(r); off += r;
                    }
                    return true;
                };
                {
                    int32_t hdr[2] = {static_cast<int32_t>(N), static_cast<int32_t>(D)};
                    if (!pio(true, out_fd, reinterpret_cast<char*>(hdr), sizeof(hdr), 0))
                        throw std::runtime_error("Inline reorder: header write failed");
                }
                const off_t hdr_bytes = 2 * sizeof(int32_t);
                const int64_t win_rows = std::max<int64_t>(1, std::min<int64_t>(N, (int64_t(4) << 30) / row_bytes));
                const int64_t chunk_rows = std::max<int64_t>(1, (int64_t(8) << 20) / row_bytes);
                const int64_t n_chunks = (N + chunk_rows - 1) / chunk_rows;
                std::unique_ptr<char[]> win(new char[static_cast<size_t>(win_rows * row_bytes)]);
                bool io_ok = true;
                for (int64_t w0 = 0; w0 < N; w0 += win_rows) {
                    const int64_t w1 = std::min<int64_t>(N, w0 + win_rows);
                    #pragma omp parallel
                    {
                        std::vector<char> cbuf(static_cast<size_t>(chunk_rows * row_bytes));
                        #pragma omp for schedule(dynamic, 1)
                        for (int64_t ch = 0; ch < n_chunks; ++ch) {
                            const int64_t r0 = ch * chunk_rows, r1 = std::min<int64_t>(N, r0 + chunk_rows);
                            // 这一块里有没有落进窗的行 (没有就不读)
                            bool any = false;
                            for (int64_t r = r0; r < r1 && !any; ++r) any = (perm[r] >= w0 && perm[r] < w1);
                            if (!any) continue;
                            if (!pio(false, in_fd, cbuf.data(), static_cast<size_t>((r1 - r0) * row_bytes),
                                     hdr_bytes + static_cast<off_t>(r0 * row_bytes))) {
                                #pragma omp atomic write
                                io_ok = false;
                                continue;
                            }
                            for (int64_t r = r0; r < r1; ++r) {
                                const int64_t p = perm[r];
                                if (p >= w0 && p < w1)
                                    std::memcpy(win.get() + (p - w0) * row_bytes, cbuf.data() + (r - r0) * row_bytes,
                                                static_cast<size_t>(row_bytes));
                            }
                        }
                    }
                    if (!io_ok) break;
                    // 窗内容顺序写出 (按 64 MB 片并行)
                    const int64_t wn = (w1 - w0) * row_bytes;
                    const int64_t piece = int64_t(64) << 20;
                    #pragma omp parallel for schedule(dynamic, 1)
                    for (int64_t o = 0; o < wn; o += piece) {
                        if (!pio(true, out_fd, win.get() + o, static_cast<size_t>(std::min<int64_t>(piece, wn - o)),
                                 hdr_bytes + static_cast<off_t>(w0 * row_bytes + o))) {
                            #pragma omp atomic write
                            io_ok = false;
                        }
                    }
                    if (!io_ok) break;
                }
                ::close(in_fd);
                if (::close(out_fd) != 0) io_ok = false;
                if (!io_ok) throw std::runtime_error("Inline reorder: I/O failed writing " + data_reordered_path);
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
            // fits_in_gpu: X_sampled 是整份数据集、旧坐标系行序 (sampled_indices = iota)
            // —— GPU-resident 和 out-of-core 两条路径都是 (out-of-core 原来要重新
            // 读一遍 data_reordered 并转 float)。
            if (iterations > 1 && x_lazy) {
                // Step 2 直接从 raw_host 取，raw_host 已经换成新坐标系那份 (见上)，不需要映射
            } else if (iterations > 1 && mem_est.fits_in_gpu && config.kmeans_train_fraction < 1.0f && !mem_est.need_pq) {
                // Step 2 只会从 X_sampled 抽 kmeans_train_fraction 的子集 (见
                // Step 2)：不整份重排 (N*D*4 B, SIFT1M ~0.5 s)，只记新 id → 旧行号
                // 的映射，抽子集时按映射 gather。
                auto tresample0 = Clock::now();
                if (static_cast<int64_t>(reorder_info.inverse_perm.size()) != N)
                    throw std::runtime_error("Step 4.5: inverse_perm size != N");
                x_row_of.assign(reorder_info.inverse_perm.begin(), reorder_info.inverse_perm.end());
                t45_resample = std::chrono::duration<double>(Clock::now() - tresample0).count();
            } else if (iterations > 1 && gres) {
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
                // 不能走 blob 模式时 reader 要读 data_reordered 文件: 先等后台写完
                if (!(ooc_host_raw && static_cast<int64_t>(raw_host.size()) == N * static_cast<int64_t>(D)))
                    join_reorder_write();
                reordered_io::ReorderedDatasetBucketReader<DataT> reader(
                    working_input_path, D, reorder_offsets_new, reorder_pos_of);
                // raw_host 这时是重排后的整份数据集 (ooc_host_raw 且多轮时) → blob 模式
                if (ooc_host_raw && static_cast<int64_t>(raw_host.size()) == N * static_cast<int64_t>(D))
                    reader.set_host_data(raw_host.data(), N);
                build_vector_knn_with_tensorcore<DataT>(
                    N, D, reader, graph_ptr,
                    n_centroids, graph_K, neighbors_m,
                    *knn_acc,
                    centroid_knn_graph_host.data(), K,
                    order_window_arg, step6_cache_bytes,
                    &bucket_process_order_round1, config.gpu_bucket_cache_mb << 20);
            } else {
                // bucket 处理顺序在 Step 3 之后已经在后台算好
                std::vector<int32_t> order_rk = order_future.valid() ? order_future.get()
                                                                     : std::vector<int32_t>();
                // 每点 running 第 M 个距离当 block top-M 的行阈值初值: 用最近一次
                // 已经完成的 merge 的结果，不等正在后台跑的那次 (等它会把 merge
                // 放上关键路径: #62 round 3 等了 7.9 s)
                std::vector<float> kth_seed;
                const bool have_seed = knn_acc->snapshot_kth(kth_seed);
                std::cout << "  [Step6] k-th seed: " << (have_seed ? "from the last completed merge" : "none yet") << "\n";
                build_vector_knn_with_tensorcore<DataT>(
                    N, D, *bucket_vecs_ptr, graph_ptr,
                    n_centroids, graph_K, neighbors_m,
                    *knn_acc,
                    centroid_knn_graph_host.data(), K,
                    order_window_arg, step6_cache_bytes,
                    order_rk.empty() ? nullptr : &order_rk, config.gpu_bucket_cache_mb << 20,
                    have_seed ? kth_seed.data() : nullptr);
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
                acc_ptr->planned_final_iter = iterations - 1;
                if (iter == iterations - 1)
                    acc_ptr->set_final_output(output_dir + "/vector_knn.bin",
                                              config.save_dists ? output_dir + "/vector_dists.bin" : std::string(), iter);
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

        join_reorder_write();   // data_reordered.<ext> 必须在进程结束前写完
        // 把按 chunk 分片的 running 状态 (跨全部 iteration 合并去重后的最终
        // 结果) 按 chunk 顺序拼接成 vector_knn.bin / vector_dists.bin。
        if (gres) {
            // A1-R (用户批准): BUCKET2_A1_PASSES 趟 local join，BUCKET2_A1_SAMPLE (默认 32)
            if (const char* e = std::getenv("BUCKET2_A1_PASSES")) {
                const int passes = std::atoi(e);
                const char* es = std::getenv("BUCKET2_A1_SAMPLE");
                auto ta = Clock::now();
                gres->refine_local_join(passes, es ? std::atoi(es) : 32);
                if (passes > 0)
                    std::cout << "  [A1-R] local join: " << std::chrono::duration<double>(Clock::now() - ta).count() << "s\n";
            }
            auto t_write = Clock::now();
            // --fused-optimize-degree: 下载完就在写盘的同时做 prune/反向边/id 翻译
            std::function<void(const std::vector<int32_t>&)> tail;
            double fused_s = 0.0;
            if (config.fused_optimize_degree > 0) {
                tail = [&](const std::vector<int32_t>& h_n) {
                    auto tf = Clock::now();
                    gres->release_device();   // 显存让给 cuVS optimize
                    fused_optimize_translate(h_n, N, static_cast<int>(neighbors_m), config.fused_optimize_degree,
                                             fused_inv_perm, output_dir + "/final_graph.bin");
                    fused_s = std::chrono::duration<double>(Clock::now() - tf).count();
                };
            }
            gres->write_outputs(output_dir + "/vector_knn.bin",
                                config.save_dists ? output_dir + "/vector_dists.bin" : std::string(),
                                config.save_npy ? output_dir + "/neighbors.npy" : std::string(), tail);
            if (tail)
                std::cout << "[FusedOptimize] prune + reverse edges + id translation in-process: "
                          << std::fixed << std::setprecision(3) << fused_s << "s (overlapped with the vector_knn.bin write)\n";
            std::cout << "[WriteVectorKNN] Saved vector_knn.bin / vector_dists.bin"
                      << (config.save_npy ? " / neighbors.npy" : " (--skip-npy)") << "\n";
            elapsed_write_knn = std::chrono::duration<double>(Clock::now() - t_write).count();
        } else if (neighbors_m > 0) {
            auto t_write = Clock::now();
            knn_acc->finalize_to_file(
                output_dir + "/vector_knn.bin",
                config.save_dists ? output_dir + "/vector_dists.bin" : std::string());
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
            ("kmeans-train-fraction", po::value<float>(),
                "Step 2 (dataset fits on GPU): train KMeans|| on this random fraction of the points "
                "(centroid count stays N*centroid-ratio; default 1.0 = all points)")
            ("bucket-mem-mb", po::value<size_t>(),
                "Rounds 2..t: keep the bucket records (Step 4 → Step 6) in host memory when they fit this budget in MB "
                "(default 0 = cpu-limit/4); Step 6 then copies GPU-cache misses straight from it")
            ("gpu-bucket-cache-mb", po::value<size_t>(),
                "Out-of-core Step 6: GPU-side bucket cache budget in MB (Belady eviction; default 2048, 0 = off "
                "→ pools assembled on the host and uploaded per bucket)")
            ("skip-dists",    po::bool_switch()->default_value(false),
                "Don't write vector_dists.bin (distances of vector_knn.bin; no later pipeline stage reads it)")
            ("skip-npy",      po::bool_switch()->default_value(false),
                "Don't write neighbors.npy (vector_knn.bin as int64 NumPy, only for Python-side eval; "
                "the build pipeline never reads it)")
            ("fused-optimize-degree", po::value<int>(),
                "GPU-resident path only: after the last round, prune + add reverse edges (cuVS "
                "helpers::optimize) to this degree and translate ids back to the original order in this "
                "process, writing final_graph.bin (replaces the separate optimize + translate_ids stages; "
                "default 0 = off; the out-of-core path ignores it)")
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
