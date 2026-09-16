// optimize_chunked.cu
// 三种 GPU 内存友好的 reverse-graph + merge 实现，处理"前向图整张放不进 GPU"的场景。
//
// 输入：已剪枝的前向图 forward_graph.bin（可由 optimize.cu --output 或 per-bucket
//       prune 拼接得到）
// 输出：CAGRA 风格的最终图 cagra_graph.bin（forward 边 + reverse 边合并后）
//
// 三种方法（--method）:
//   A: 按目的端均匀分块，kernel 内过滤越界边（越界边在其他 chunk pass 中被捞回）。
//      代码最简单，PCIe 流量 = num_chunks × FwdSize。
//   B: 先在 host 把 (src,dst) 对按目的端预分桶（一次 shuffle），每个 chunk 只送自己
//      的 pair 上 GPU。PCIe 流量 = 2 × FwdSize（送 8B/edge），但需要 N×K_out×8B 暂存。
//   C: bucket-aligned + 单次源行扫描 (PCIe = 1 × FwdSize) + cross-chunk 边走 overflow。
//      要求数据已按 bucket 重排（用 reorder.cpp）。lossless：phase A 处理 in-chunk 边，
//      phase B 在 host 上处理跨 chunk overflow 边并合并。统计 in-chunk / cross-chunk
//      比例，方便判断重排效果。
//   D: C 的升级版（3-slot sliding window）。GPU 同时驻留 3 个 dst chunk 的 rev_graph
//      slot，处理 chunk c 时 slot 持有 {c-1, c, c+1} 的 rev。kernel 把每条边路由到
//      对应 slot；只有跨度 ≥2 的远距边才走 overflow。chunk_size 缩为 C 的 1/3，
//      但相邻 chunk 的 cross-chunk 边直接吸收，phase B 负担显著减小。s_compute 跑
//      kernel + memset，s_drain 异步 D2H rev[c-1] 与下轮 kernel 重叠。
//
// 输入 / 输出二进制格式（与 optimize.cu 对齐）:
//   int64_t  N
//   int32_t  K_out
//   uint32_t graph[N * K_out]
//
// --bucket-offsets 文件格式（method C / D 都需要）:
//   int32_t  n_buckets
//   uint32_t offsets[n_buckets + 1]    // bucket b 占 [offsets[b], offsets[b+1])

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
#include <numeric>
#include <omp.h>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>

#include "load.hpp"
#include "graph_io.hpp"
#include "edge_chunk_accumulator.hpp"

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

// =====================================================================
// I/O -- shared readers/writers now live in graph_io.hpp (used by
// reorder.cpp and prune_windowed.cu too); these are thin aliases so the
// rest of this file doesn't need to change call sites.
// =====================================================================
static void read_forward_graph(const std::string& path,
                               std::vector<uint32_t>& graph,
                               int64_t& N, int32_t& K_out)
{
    graph_io::read_forward_graph(path, graph, N, K_out);
}

static void write_graph(const std::string& path,
                        const uint32_t* graph,
                        int64_t N, int32_t K_out)
{
    graph_io::write_graph(path, graph, N, K_out);
}

static std::vector<uint32_t> read_bucket_offsets(const std::string& path)
{
    return graph_io::read_bucket_offsets(path);
}

// =====================================================================
// GPU kernels
// =====================================================================

// 方案 A 用：扫一批前向图行，对每条边 (i, k) 检查 dest 是否在当前 chunk 范围内，
// 不在的直接丢弃（A 依赖 num_chunks 次扫描在不同的 chunk 把丢边捞回来）。
__global__ void kern_rev_filter_rowbatch(
    const uint32_t* __restrict__ fwd_batch,   // [batch_rows, K_out] row-major
    uint64_t batch_row_offset,                // 该 batch 起始的全局 i
    uint32_t batch_rows,
    uint32_t K_out,
    uint32_t j_lo, uint32_t j_hi,             // 当前 chunk 目的端范围
    uint32_t K_rev_cap,
    uint32_t* __restrict__ rev_graph_chunk,   // [chunk_size, K_rev_cap]
    uint32_t* __restrict__ rev_count_chunk)   // [chunk_size]
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)batch_rows * K_out;
    if (tid >= total) return;

    uint32_t row = tid / K_out;       // batch 内行号
    uint32_t col = tid - row * K_out;
    uint32_t j = fwd_batch[(uint64_t)row * K_out + col];
    if (j < j_lo || j >= j_hi) return;

    uint32_t i = static_cast<uint32_t>(batch_row_offset + row);
    uint32_t local_j = j - j_lo;
    uint32_t pos = atomicAdd(&rev_count_chunk[local_j], 1u);
    if (pos < K_rev_cap)
        rev_graph_chunk[(uint64_t)local_j * K_rev_cap + pos] = i;
}

// 方案 C-correct 用：和上面一样，但越界边不丢弃，而是写入 overflow buffer，
// 同时分别统计 in-chunk 和 cross-chunk 边数。
//   - overflow_count 是个 device singleton 计数器，原子递增
//   - overflow_cap   是 buffer 容量，超过时 kernel 仅累加 count（caller 据此判断溢出）
//   - in_chunk_count 同理统计落在 [j_lo, j_hi) 内的边数
__global__ void kern_rev_filter_overflow(
    const uint32_t* __restrict__ fwd_batch,
    uint64_t batch_row_offset,
    uint32_t batch_rows,
    uint32_t K_out,
    uint32_t j_lo, uint32_t j_hi,
    uint32_t K_rev_cap,
    uint32_t* __restrict__ rev_graph_chunk,
    uint32_t* __restrict__ rev_count_chunk,
    uint32_t* __restrict__ overflow_src,      // [overflow_cap]
    uint32_t* __restrict__ overflow_dst,      // [overflow_cap]
    uint32_t* __restrict__ overflow_count,    // singleton
    uint32_t overflow_cap,
    uint32_t* __restrict__ in_chunk_count)    // singleton
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)batch_rows * K_out;
    if (tid >= total) return;

    uint32_t row = tid / K_out;
    uint32_t col = tid - row * K_out;
    uint32_t j = fwd_batch[(uint64_t)row * K_out + col];
    uint32_t i = static_cast<uint32_t>(batch_row_offset + row);

    if (j >= j_lo && j < j_hi) {
        // in-chunk: 走正常 path
        atomicAdd(in_chunk_count, 1u);
        uint32_t local_j = j - j_lo;
        uint32_t pos = atomicAdd(&rev_count_chunk[local_j], 1u);
        if (pos < K_rev_cap)
            rev_graph_chunk[(uint64_t)local_j * K_rev_cap + pos] = i;
    } else {
        // cross-chunk: 写 overflow
        uint32_t pos = atomicAdd(overflow_count, 1u);
        if (pos < overflow_cap) {
            overflow_src[pos] = i;
            overflow_dst[pos] = j;
        }
        // pos >= overflow_cap：caller 检测到 *overflow_count > overflow_cap 时报警
    }
}

// 方案 D 用：3 个 destination chunk slot 同时驻留 GPU。kernel 检查 j 落在哪个 slot 范围内，
// 写入对应的 rev_graph_slot；都不在则进 overflow。空 slot（如 c=0 时的 prev、c=n-1 时的 next）
// 调用方传 jlo=jhi=0xFFFFFFFF 让范围检查永远不过。
__global__ void kern_rev_3slot(
    const uint32_t* __restrict__ fwd_batch,
    uint64_t batch_row_offset,
    uint32_t batch_rows,
    uint32_t K_out,
    uint32_t K_rev_cap,
    uint32_t s0_jlo, uint32_t s0_jhi,
    uint32_t* __restrict__ s0_rev, uint32_t* __restrict__ s0_cnt,
    uint32_t s1_jlo, uint32_t s1_jhi,
    uint32_t* __restrict__ s1_rev, uint32_t* __restrict__ s1_cnt,
    uint32_t s2_jlo, uint32_t s2_jhi,
    uint32_t* __restrict__ s2_rev, uint32_t* __restrict__ s2_cnt,
    uint32_t* __restrict__ overflow_src,
    uint32_t* __restrict__ overflow_dst,
    uint32_t* __restrict__ overflow_cnt,
    uint32_t overflow_cap,
    uint32_t* __restrict__ in_chunk_cnt)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)batch_rows * K_out;
    if (tid >= total) return;

    uint32_t row = tid / K_out;
    uint32_t col = tid - row * K_out;
    uint32_t j = fwd_batch[(uint64_t)row * K_out + col];
    uint32_t i = static_cast<uint32_t>(batch_row_offset + row);

    if (j >= s0_jlo && j < s0_jhi) {
        atomicAdd(in_chunk_cnt, 1u);
        uint32_t local_j = j - s0_jlo;
        uint32_t pos = atomicAdd(&s0_cnt[local_j], 1u);
        if (pos < K_rev_cap) s0_rev[(uint64_t)local_j * K_rev_cap + pos] = i;
        return;
    }
    if (j >= s1_jlo && j < s1_jhi) {
        atomicAdd(in_chunk_cnt, 1u);
        uint32_t local_j = j - s1_jlo;
        uint32_t pos = atomicAdd(&s1_cnt[local_j], 1u);
        if (pos < K_rev_cap) s1_rev[(uint64_t)local_j * K_rev_cap + pos] = i;
        return;
    }
    if (j >= s2_jlo && j < s2_jhi) {
        atomicAdd(in_chunk_cnt, 1u);
        uint32_t local_j = j - s2_jlo;
        uint32_t pos = atomicAdd(&s2_cnt[local_j], 1u);
        if (pos < K_rev_cap) s2_rev[(uint64_t)local_j * K_rev_cap + pos] = i;
        return;
    }
    // overflow（j 在 chunks {…, c-2, c+2, …} 之外，跨度 ≥ 2 个 chunk）
    uint32_t pos = atomicAdd(overflow_cnt, 1u);
    if (pos < overflow_cap) {
        overflow_src[pos] = i;
        overflow_dst[pos] = j;
    }
}

// 方案 B 用：消费已经按目的端分桶过的 (src, dst) pair 列表
__global__ void kern_rev_scatter_pairs(
    const uint32_t* __restrict__ pairs_src,   // [num_pairs]
    const uint32_t* __restrict__ pairs_dst,   // [num_pairs]
    size_t num_pairs,
    uint32_t j_lo,
    uint32_t K_rev_cap,
    uint32_t* __restrict__ rev_graph_chunk,
    uint32_t* __restrict__ rev_count_chunk)
{
    uint64_t e = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_pairs) return;
    uint32_t i = pairs_src[e];
    uint32_t j = pairs_dst[e];
    uint32_t local_j = j - j_lo;
    uint32_t pos = atomicAdd(&rev_count_chunk[local_j], 1u);
    if (pos < K_rev_cap)
        rev_graph_chunk[(uint64_t)local_j * K_rev_cap + pos] = i;
}

// 方案 E 用：单遍扫描，每条边按 chunk_of(dst) 直接散射进该 chunk 自己的 GPU
// scatter buffer（容量固定，跨多个 batch 持续累积，由 host 侧按批读回计数，
// 写满了才 drain 到磁盘并清零——不是每批都 flush）。本批次恰好把某个 chunk
// 缓冲区写满、多出来的少量边落进一个共享 overflow 缓冲区（大小按
// batch_rows*K_out 最坏情况算，一批边全部挤爆同一个 chunk 也能兜住），host
// 侧每批都会把 overflow drain 出来按 chunk_of 分发进正确的磁盘文件——保证
// 不丢边，同时不需要为了防溢出把每个 chunk 的主 buffer 按最坏情况分配。
//
// rank = col：这条边是 src 这一行的第几个（第几近）邻居，写进
// edge_chunk::EdgeRecord 里带走，Phase 2 按 rank 排序后重建优先级（见
// edge_chunk_accumulator.hpp 顶部注释）。
__global__ void kern_scatter_rank(
    const uint32_t* __restrict__ fwd_batch,   // [batch_rows, K_out]
    uint64_t batch_row_offset,
    uint32_t batch_rows,
    uint32_t K_out,
    const uint32_t* __restrict__ chunk_starts,  // [num_chunks+1]
    uint32_t num_chunks,
    edge_chunk::EdgeRecord* __restrict__ chunk_buf,  // [num_chunks * capacity_per_chunk]
    uint32_t capacity_per_chunk,
    uint32_t* __restrict__ chunk_count,         // [num_chunks], 跨 batch 持续累积
    edge_chunk::EdgeRecord* __restrict__ overflow_buf,  // [overflow_cap]
    uint32_t* __restrict__ overflow_count,      // singleton，每 batch 由 host 清零
    uint32_t overflow_cap)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)batch_rows * K_out;
    if (tid >= total) return;

    uint32_t row = tid / K_out;
    uint32_t col = tid - row * K_out;
    uint32_t j = fwd_batch[(uint64_t)row * K_out + col];
    uint32_t i = static_cast<uint32_t>(batch_row_offset + row);

    uint32_t c = edge_chunk::chunk_of(chunk_starts, num_chunks, j);
    uint32_t pos = atomicAdd(&chunk_count[c], 1u);
    edge_chunk::EdgeRecord rec{j, i, static_cast<uint16_t>(col)};
    if (pos < capacity_per_chunk) {
        chunk_buf[(uint64_t)c * capacity_per_chunk + pos] = rec;
    } else {
        uint32_t opos = atomicAdd(overflow_count, 1u);
        if (opos < overflow_cap) overflow_buf[opos] = rec;
        // opos >= overflow_cap: host 检测 *overflow_count > overflow_cap 时报错
        // (见 run_method_E) -- 意味着 --gpu-budget-mb/--edge-buffer-mb 对当前
        // batch_rows 来说太小，不会静默丢边。
    }
}

// =====================================================================
// CPU merge: 把 reverse 边合并进 output_graph[j]，保护前 K_out/2 条 forward 边
// （等价于 RAFT graph_core.cuh 的 "Replace some edges with reverse edges"）
// =====================================================================
static void merge_rev_into_output_chunk(
    uint32_t* output_graph,           // [N, K_out]，输入时是 forward_graph 的拷贝
    const uint32_t* rev_graph_chunk,  // [chunk_size, K_rev_cap]
    const uint32_t* rev_count_chunk,  // [chunk_size]
    uint32_t j_lo, uint32_t j_hi,
    uint32_t K_out, uint32_t K_rev_cap)
{
    const uint32_t num_protected = K_out / 2;

    #pragma omp parallel for schedule(dynamic, 1024)
    for (uint32_t j = j_lo; j < j_hi; ++j) {
        uint32_t local_j = j - j_lo;
        uint32_t cnt = std::min(rev_count_chunk[local_j], K_rev_cap);
        uint32_t* row = output_graph + static_cast<uint64_t>(j) * K_out;

        // 倒序插入 rev edges（CAGRA 原版逻辑）
        while (cnt > 0) {
            cnt--;
            uint32_t i = rev_graph_chunk[(uint64_t)local_j * K_rev_cap + cnt];

            // 找 i 在 row 中的位置
            uint32_t pos = K_out;
            for (uint32_t p = 0; p < K_out; ++p) {
                if (row[p] == i) { pos = p; break; }
            }
            if (pos < num_protected) continue;

            uint32_t num_shift;
            if (pos == K_out)
                num_shift = K_out - num_protected - 1;
            else
                num_shift = pos - num_protected;

            // row[num_protected .. num_protected + num_shift] 整体右移 1 位
            for (uint32_t s = num_shift; s > 0; --s)
                row[num_protected + s] = row[num_protected + s - 1];
            row[num_protected] = i;
        }
    }
}

// =====================================================================
// Chunk boundary helpers
// =====================================================================
struct ChunkPlan {
    std::vector<uint32_t> starts;   // 长度 num_chunks+1, [starts[c], starts[c+1])
    std::string label;
};

// 方案 A / B：根据 GPU 预算切均匀 chunk
static ChunkPlan plan_uniform_chunks(int64_t N, int32_t K_rev_cap, size_t gpu_budget_bytes)
{
    // 粗略预算分配：rev_graph_chunk + counts + 一些缓冲，留 60% 给 rev_graph_chunk
    size_t per_node_bytes = K_rev_cap * sizeof(uint32_t) + sizeof(uint32_t);
    size_t rev_budget = static_cast<size_t>(gpu_budget_bytes * 0.6);
    uint32_t chunk_size = static_cast<uint32_t>(std::max<size_t>(1, rev_budget / per_node_bytes));
    chunk_size = std::min<uint32_t>(chunk_size, static_cast<uint32_t>(N));

    ChunkPlan plan;
    for (uint32_t s = 0; s < N; s += chunk_size)
        plan.starts.push_back(s);
    plan.starts.push_back(static_cast<uint32_t>(N));
    plan.label = "uniform (chunk_size=" + std::to_string(chunk_size) + ")";
    return plan;
}

// 方案 C：把若干 bucket 合并成一个 chunk，使每个 chunk 容量不超 GPU 预算
static ChunkPlan plan_bucket_aligned_chunks(
    const std::vector<uint32_t>& bucket_offsets, int32_t K_rev_cap, size_t gpu_budget_bytes)
{
    size_t per_node_bytes = K_rev_cap * sizeof(uint32_t) + sizeof(uint32_t);
    size_t rev_budget = static_cast<size_t>(gpu_budget_bytes * 0.6);
    uint32_t chunk_capacity = static_cast<uint32_t>(std::max<size_t>(1, rev_budget / per_node_bytes));

    ChunkPlan plan;
    plan.starts.push_back(0);
    uint32_t accum = 0;
    for (size_t b = 1; b < bucket_offsets.size(); ++b) {
        uint32_t bsz = bucket_offsets[b] - bucket_offsets[b - 1];
        if (bsz > chunk_capacity) {
            throw std::runtime_error(
                "bucket " + std::to_string(b - 1) + " size (" + std::to_string(bsz) +
                ") exceeds chunk capacity " + std::to_string(chunk_capacity) +
                ". Increase --gpu-budget-mb or split this bucket.");
        }
        if (accum + bsz > chunk_capacity) {
            plan.starts.push_back(bucket_offsets[b - 1]);
            accum = 0;
        }
        accum += bsz;
    }
    plan.starts.push_back(bucket_offsets.back());
    plan.label = "bucket-aligned (n_chunks=" + std::to_string(plan.starts.size() - 1) + ")";
    return plan;
}

// =====================================================================
// Method A: kernel-side filtering, row-batch streaming
// 每个 chunk 都过整张前向图，越界边 kernel 内丢弃（依赖其他 chunk pass 捞回）
// =====================================================================
static void run_method_A(
    const uint32_t* fwd_graph_host,    // pinned
    uint32_t* output_graph_host,       // [N, K_out], 预初始化为 fwd 的拷贝
    int64_t N, int32_t K_out, int32_t K_rev_cap,
    const ChunkPlan& plan,
    size_t gpu_budget_bytes)
{
    using Clock = std::chrono::steady_clock;

    // 找最大 chunk_size，按它分配 GPU buffer 一次
    uint32_t max_chunk = 0;
    for (size_t c = 0; c + 1 < plan.starts.size(); ++c)
        max_chunk = std::max(max_chunk, plan.starts[c + 1] - plan.starts[c]);

    // GPU buffers
    uint32_t* d_rev_graph   = nullptr;
    uint32_t* d_rev_count   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rev_graph,
        (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_count,
        (size_t)max_chunk * sizeof(uint32_t)));

    // 剩余预算给 forward batch
    size_t used = (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)
                + (size_t)max_chunk * sizeof(uint32_t);
    size_t fwd_budget = (gpu_budget_bytes > used) ? (gpu_budget_bytes - used) : 0;
    // 至少给 forward batch 留 256MB
    if (fwd_budget < (256ULL << 20))
        fwd_budget = (256ULL << 20);
    uint32_t batch_rows = static_cast<uint32_t>(
        std::max<size_t>(1024, fwd_budget / 2 / (K_out * sizeof(uint32_t))));
    batch_rows = std::min<uint32_t>(batch_rows, static_cast<uint32_t>(N));

    // 双缓冲 forward batch
    uint32_t* d_fwd_batch[2] = {nullptr, nullptr};
    cudaStream_t stream[2];
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&d_fwd_batch[s], (size_t)batch_rows * K_out * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&stream[s]));
    }

    std::cout << "  [Method A] " << plan.label
              << ", batch_rows=" << batch_rows
              << ", num_chunks=" << plan.starts.size() - 1 << "\n";

    // host pinned 暂存（rev_graph chunk D2H 用）
    uint32_t* h_rev_graph_chunk = nullptr;
    uint32_t* h_rev_count_chunk = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_rev_graph_chunk,
        (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_rev_count_chunk,
        (size_t)max_chunk * sizeof(uint32_t)));

    double t_h2d = 0, t_kern = 0, t_d2h = 0, t_merge = 0;

    for (size_t c = 0; c + 1 < plan.starts.size(); ++c) {
        uint32_t j_lo = plan.starts[c];
        uint32_t j_hi = plan.starts[c + 1];
        uint32_t chunk_sz = j_hi - j_lo;

        // 每个 chunk 重置 rev_count / rev_graph
        CUDA_CHECK(cudaMemset(d_rev_count, 0, chunk_sz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_rev_graph, 0xff,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t)));

        // 流式扫前向图
        auto t0 = Clock::now();
        for (uint64_t bs = 0; bs < (uint64_t)N; bs += batch_rows) {
            uint32_t cur_rows = static_cast<uint32_t>(std::min<uint64_t>(batch_rows, N - bs));
            int s = (bs / batch_rows) & 1;

            CUDA_CHECK(cudaMemcpyAsync(
                d_fwd_batch[s],
                fwd_graph_host + bs * K_out,
                (size_t)cur_rows * K_out * sizeof(uint32_t),
                cudaMemcpyHostToDevice, stream[s]));

            uint64_t total = (uint64_t)cur_rows * K_out;
            int threads = 256;
            int blocks = static_cast<int>((total + threads - 1) / threads);
            kern_rev_filter_rowbatch<<<blocks, threads, 0, stream[s]>>>(
                d_fwd_batch[s], bs, cur_rows, (uint32_t)K_out,
                j_lo, j_hi, (uint32_t)K_rev_cap,
                d_rev_graph, d_rev_count);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_kern += std::chrono::duration<double>(Clock::now() - t0).count();

        // D2H rev_graph chunk
        auto t1 = Clock::now();
        CUDA_CHECK(cudaMemcpy(h_rev_count_chunk, d_rev_count,
            chunk_sz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rev_graph_chunk, d_rev_graph,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        t_d2h += std::chrono::duration<double>(Clock::now() - t1).count();

        // CPU merge
        auto t2 = Clock::now();
        merge_rev_into_output_chunk(
            output_graph_host, h_rev_graph_chunk, h_rev_count_chunk,
            j_lo, j_hi, (uint32_t)K_out, (uint32_t)K_rev_cap);
        t_merge += std::chrono::duration<double>(Clock::now() - t2).count();
    }

    std::cout << "  [Timing] H2D+kern=" << std::fixed << std::setprecision(2) << t_kern
              << "s, D2H=" << t_d2h << "s, merge=" << t_merge << "s\n";

    cudaFreeHost(h_rev_graph_chunk);
    cudaFreeHost(h_rev_count_chunk);
    for (int s = 0; s < 2; ++s) {
        cudaFree(d_fwd_batch[s]);
        cudaStreamDestroy(stream[s]);
    }
    cudaFree(d_rev_graph);
    cudaFree(d_rev_count);
}

// =====================================================================
// Method C-correct: bucket-aligned chunks, single-pass over source rows,
// cross-chunk edges captured into overflow buffer (Phase B 处理)
// =====================================================================
static void run_method_C(
    const uint32_t* fwd_graph_host,    // pinned
    uint32_t* output_graph_host,
    int64_t N, int32_t K_out, int32_t K_rev_cap,
    const ChunkPlan& plan,
    size_t gpu_budget_bytes)
{
    using Clock = std::chrono::steady_clock;

    uint32_t max_chunk = 0;
    for (size_t c = 0; c + 1 < plan.starts.size(); ++c)
        max_chunk = std::max(max_chunk, plan.starts[c + 1] - plan.starts[c]);

    // -------- 预算分配 --------
    // 每节点 (rev_graph_chunk) + 4 字节 (rev_count_chunk)
    size_t rev_bytes = (size_t)max_chunk * (K_rev_cap + 1) * sizeof(uint32_t);

    // 双缓冲 forward batch + overflow buffer：合占剩余预算
    size_t remaining = (gpu_budget_bytes > rev_bytes) ? (gpu_budget_bytes - rev_bytes)
                                                       : (256ULL << 20);
    // batch_rows 的每行成本：fwd_batch (K_out*4) + overflow_src/dst worst-case (K_out*8)
    // 双缓冲再 ×2
    size_t per_row_bytes_dual = 2 * (K_out * sizeof(uint32_t)              // fwd
                                     + K_out * 2 * sizeof(uint32_t));       // overflow
    uint32_t batch_rows = static_cast<uint32_t>(
        std::max<size_t>(4096, remaining / per_row_bytes_dual));
    batch_rows = std::min<uint32_t>(batch_rows, max_chunk);

    // overflow_cap_per_batch = batch_rows * K_out （worst case，每条边都 cross-chunk）
    uint32_t overflow_cap = batch_rows * (uint32_t)K_out;

    // -------- 分配 GPU/host buffer --------
    uint32_t* d_rev_graph = nullptr;
    uint32_t* d_rev_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rev_graph, (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_count, (size_t)max_chunk * sizeof(uint32_t)));

    uint32_t* d_fwd_batch[2] = {nullptr, nullptr};
    uint32_t* d_overflow_src[2] = {nullptr, nullptr};
    uint32_t* d_overflow_dst[2] = {nullptr, nullptr};
    uint32_t* d_overflow_cnt[2] = {nullptr, nullptr};
    uint32_t* d_in_chunk_cnt[2] = {nullptr, nullptr};
    cudaStream_t stream[2];
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&d_fwd_batch[s], (size_t)batch_rows * K_out * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_overflow_src[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_overflow_dst[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_overflow_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_in_chunk_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&stream[s]));
    }

    uint32_t* h_rev_graph_chunk = nullptr;
    uint32_t* h_rev_count_chunk = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_rev_graph_chunk,
        (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_rev_count_chunk,
        (size_t)max_chunk * sizeof(uint32_t)));

    uint32_t* h_overflow_src[2] = {nullptr, nullptr};
    uint32_t* h_overflow_dst[2] = {nullptr, nullptr};
    uint32_t* h_overflow_cnt[2] = {nullptr, nullptr};
    uint32_t* h_in_chunk_cnt[2] = {nullptr, nullptr};
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_overflow_src[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_overflow_dst[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_overflow_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_in_chunk_cnt[s], sizeof(uint32_t)));
    }

    std::cout << "  [Method C-correct] " << plan.label
              << ", batch_rows=" << batch_rows
              << ", overflow_cap_per_batch=" << overflow_cap
              << " (~" << overflow_cap * 8 / 1e6 << " MB/stream)\n";

    // 全局 overflow 累加（host）
    std::vector<uint32_t> all_overflow_src, all_overflow_dst;
    uint64_t total_in_chunk     = 0;
    uint64_t total_overflow     = 0;
    uint64_t total_overflow_dropped = 0;

    double t_phaseA = 0, t_phaseB = 0, t_d2h_rev = 0, t_merge = 0;

    auto tA = Clock::now();
    for (size_t c = 0; c + 1 < plan.starts.size(); ++c) {
        uint32_t i_lo = plan.starts[c];
        uint32_t i_hi = plan.starts[c + 1];
        uint32_t j_lo = i_lo;          // C 中 dst 范围 = src 范围
        uint32_t j_hi = i_hi;
        uint32_t chunk_sz = j_hi - j_lo;

        CUDA_CHECK(cudaMemset(d_rev_count, 0, chunk_sz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_rev_graph, 0xff,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t)));

        // ★ 关键：只扫本 chunk 的源行 [i_lo, i_hi)
        for (uint64_t bs = i_lo; bs < i_hi; bs += batch_rows) {
            uint32_t cur_rows = static_cast<uint32_t>(
                std::min<uint64_t>(batch_rows, i_hi - bs));
            int s = (((bs - i_lo) / batch_rows) & 1);

            // 等待该 stream 上一轮 overflow drain 完成
            cudaStreamSynchronize(stream[s]);

            CUDA_CHECK(cudaMemsetAsync(d_overflow_cnt[s], 0, sizeof(uint32_t), stream[s]));
            CUDA_CHECK(cudaMemsetAsync(d_in_chunk_cnt[s], 0, sizeof(uint32_t), stream[s]));

            CUDA_CHECK(cudaMemcpyAsync(d_fwd_batch[s],
                fwd_graph_host + bs * K_out,
                (size_t)cur_rows * K_out * sizeof(uint32_t),
                cudaMemcpyHostToDevice, stream[s]));

            uint64_t total = (uint64_t)cur_rows * K_out;
            int threads = 256;
            int blocks  = static_cast<int>((total + threads - 1) / threads);
            kern_rev_filter_overflow<<<blocks, threads, 0, stream[s]>>>(
                d_fwd_batch[s], bs, cur_rows, (uint32_t)K_out,
                j_lo, j_hi, (uint32_t)K_rev_cap,
                d_rev_graph, d_rev_count,
                d_overflow_src[s], d_overflow_dst[s],
                d_overflow_cnt[s], overflow_cap,
                d_in_chunk_cnt[s]);

            // D2H counts (small, async)
            CUDA_CHECK(cudaMemcpyAsync(h_overflow_cnt[s], d_overflow_cnt[s],
                sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[s]));
            CUDA_CHECK(cudaMemcpyAsync(h_in_chunk_cnt[s], d_in_chunk_cnt[s],
                sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[s]));
            // 同步后才能读 count，决定 overflow data 多大
            cudaStreamSynchronize(stream[s]);

            uint32_t ofc = *h_overflow_cnt[s];
            uint32_t inc = *h_in_chunk_cnt[s];
            total_in_chunk += inc;
            if (ofc > overflow_cap) {
                total_overflow_dropped += (ofc - overflow_cap);
                ofc = overflow_cap;
            }
            total_overflow += ofc;

            if (ofc > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_overflow_src[s], d_overflow_src[s],
                    (size_t)ofc * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[s]));
                CUDA_CHECK(cudaMemcpyAsync(h_overflow_dst[s], d_overflow_dst[s],
                    (size_t)ofc * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[s]));
                cudaStreamSynchronize(stream[s]);
                all_overflow_src.insert(all_overflow_src.end(),
                    h_overflow_src[s], h_overflow_src[s] + ofc);
                all_overflow_dst.insert(all_overflow_dst.end(),
                    h_overflow_dst[s], h_overflow_dst[s] + ofc);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = Clock::now();
        CUDA_CHECK(cudaMemcpy(h_rev_count_chunk, d_rev_count,
            chunk_sz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rev_graph_chunk, d_rev_graph,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        t_d2h_rev += std::chrono::duration<double>(Clock::now() - t1).count();

        auto t2 = Clock::now();
        merge_rev_into_output_chunk(
            output_graph_host, h_rev_graph_chunk, h_rev_count_chunk,
            j_lo, j_hi, (uint32_t)K_out, (uint32_t)K_rev_cap);
        t_merge += std::chrono::duration<double>(Clock::now() - t2).count();
    }
    t_phaseA = std::chrono::duration<double>(Clock::now() - tA).count();

    // ============================================================
    // Phase B: 在 host 上消化 cross-chunk overflow 边
    // ============================================================
    auto tB = Clock::now();
    if (!all_overflow_src.empty()) {
        size_t n_over = all_overflow_src.size();
        const size_t num_chunks = plan.starts.size() - 1;

        // chunk lookup
        auto chunk_of = [&](uint32_t j) -> size_t {
            auto it = std::upper_bound(plan.starts.begin(), plan.starts.end(), j);
            return static_cast<size_t>(it - plan.starts.begin()) - 1;
        };

        // 先统计每个目的 chunk 的边数
        std::vector<size_t> chunk_count(num_chunks, 0);
        std::vector<size_t> edge_chunk(n_over);
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < n_over; ++e)
            edge_chunk[e] = chunk_of(all_overflow_dst[e]);
        for (size_t e = 0; e < n_over; ++e)
            chunk_count[edge_chunk[e]]++;

        // 对每个目的 chunk 构建 mini rev_graph 并 merge
        for (size_t c = 0; c < num_chunks; ++c) {
            if (chunk_count[c] == 0) continue;
            uint32_t cj_lo = plan.starts[c];
            uint32_t cj_hi = plan.starts[c + 1];
            uint32_t cz    = cj_hi - cj_lo;

            std::vector<uint32_t> mini_rev((size_t)cz * K_rev_cap, 0xFFFFFFFFu);
            std::vector<uint32_t> mini_cnt(cz, 0);

            // 顺序 scatter（边数小，单线程已足够；并发也行但要 atomic）
            for (size_t e = 0; e < n_over; ++e) {
                if (edge_chunk[e] != c) continue;
                uint32_t i = all_overflow_src[e];
                uint32_t j = all_overflow_dst[e];
                uint32_t local_j = j - cj_lo;
                uint32_t pos = mini_cnt[local_j]++;
                if (pos < (uint32_t)K_rev_cap)
                    mini_rev[(size_t)local_j * K_rev_cap + pos] = i;
            }

            merge_rev_into_output_chunk(
                output_graph_host, mini_rev.data(), mini_cnt.data(),
                cj_lo, cj_hi, (uint32_t)K_out, (uint32_t)K_rev_cap);
        }
    }
    t_phaseB = std::chrono::duration<double>(Clock::now() - tB).count();

    // -------- 报告 --------
    uint64_t total_edges = total_in_chunk + total_overflow + total_overflow_dropped;
    double pct_in     = total_edges ? 100.0 * total_in_chunk / total_edges : 0.0;
    double pct_over   = total_edges ? 100.0 * total_overflow / total_edges : 0.0;
    double pct_drop   = total_edges ? 100.0 * total_overflow_dropped / total_edges : 0.0;

    std::cout << "  [Stats] in-chunk    edges: " << total_in_chunk
              << " (" << std::fixed << std::setprecision(2) << pct_in << "%)\n";
    std::cout << "  [Stats] cross-chunk edges: " << total_overflow
              << " (" << pct_over << "%)\n";
    if (total_overflow_dropped > 0) {
        std::cout << "  [WARN ] dropped edges:     " << total_overflow_dropped
                  << " (" << pct_drop << "%) — overflow buffer too small. "
                  << "Reduce batch_rows or increase --gpu-budget-mb.\n";
    }
    std::cout << "  [Timing] phase A (GPU+merge)=" << t_phaseA
              << "s, phase B (overflow)=" << t_phaseB
              << "s, [included] D2H rev=" << t_d2h_rev
              << "s, merge=" << t_merge << "s\n";

    // 清理
    for (int s = 0; s < 2; ++s) {
        cudaFree(d_fwd_batch[s]);
        cudaFree(d_overflow_src[s]);
        cudaFree(d_overflow_dst[s]);
        cudaFree(d_overflow_cnt[s]);
        cudaFree(d_in_chunk_cnt[s]);
        cudaFreeHost(h_overflow_src[s]);
        cudaFreeHost(h_overflow_dst[s]);
        cudaFreeHost(h_overflow_cnt[s]);
        cudaFreeHost(h_in_chunk_cnt[s]);
        cudaStreamDestroy(stream[s]);
    }
    cudaFreeHost(h_rev_graph_chunk);
    cudaFreeHost(h_rev_count_chunk);
    cudaFree(d_rev_graph);
    cudaFree(d_rev_count);
}

// =====================================================================
// Method D: 3-slot sliding-window，C-aggressive 的升级版
//
// 核心思想：
//   - GPU 同时驻留 3 个 destination chunk 的 rev_graph slot
//   - 处理 chunk c 时，3 slots 分别持有 rev[c-1], rev[c], rev[c+1]
//   - kernel 把每条边 (i, j) 路由到 j 所属的 slot；都不命中 → overflow
//   - chunk c 处理完毕：rev[c-1] 已收齐所有 writers (chunks c-2, c-1, c)
//     → drain to host + merge → 该 GPU slot 重置为 rev[c+2]
//   - 双 stream：s_compute (kernel + memset) 与 s_drain (D2H) 异步重叠
//
// 相比 method C：
//   chunk_size 缩成 ~1/3（要装 3 份），但 cross-chunk 边的 ±1 邻接被直接吸收掉
//   没浪费到 overflow，phase B 工作量大幅减少（仅 ≥2 跨度的远距边走 overflow）
//
// 假设：bucket 重排后，相邻 chunk 在特征空间也相邻（默认成立 if buckets sorted by
// some spatial layout）。否则 overflow 比例可能较高，看 stats 决定是否值得。
// =====================================================================
static void run_method_D(
    const uint32_t* fwd_graph_host,    // pinned
    uint32_t* output_graph_host,
    int64_t N, int32_t K_out, int32_t K_rev_cap,
    const ChunkPlan& plan,
    size_t gpu_budget_bytes)
{
    using Clock = std::chrono::steady_clock;
    const int num_chunks = static_cast<int>(plan.starts.size()) - 1;

    // -------- 退化处理 --------
    if (num_chunks <= 2) {
        std::cout << "  [Method D] num_chunks=" << num_chunks
                  << " <= 2, falling back to method C\n";
        run_method_C(fwd_graph_host, output_graph_host, N, K_out, K_rev_cap,
                     plan, gpu_budget_bytes);
        return;
    }

    uint32_t max_chunk = 0;
    for (int c = 0; c + 1 < (int)plan.starts.size(); ++c)
        max_chunk = std::max(max_chunk, plan.starts[c + 1] - plan.starts[c]);

    // -------- 预算切分 --------
    // 3 slots ≈ 50%；fwd batch ×2 ≈ 25%；overflow buffers ≈ 25%
    size_t per_slot_bytes = (size_t)max_chunk * (K_rev_cap + 1) * sizeof(uint32_t);
    size_t three_slots_bytes = 3 * per_slot_bytes;

    if (three_slots_bytes >= gpu_budget_bytes)
        throw std::runtime_error(
            "Method D: 3 slots (" + std::to_string(three_slots_bytes / (1ULL << 20)) +
            " MB) exceed --gpu-budget-mb. Make chunks smaller or raise budget.");

    size_t remaining = gpu_budget_bytes - three_slots_bytes;
    size_t per_row_bytes_dual = 2 * (K_out * sizeof(uint32_t)            // fwd
                                     + K_out * 2 * sizeof(uint32_t));     // overflow
    uint32_t batch_rows = static_cast<uint32_t>(
        std::max<size_t>(4096, remaining / per_row_bytes_dual));
    batch_rows = std::min<uint32_t>(batch_rows, max_chunk);
    uint32_t overflow_cap = batch_rows * (uint32_t)K_out;

    // -------- 分配 GPU buffers --------
    uint32_t* d_rev_slot[3] = {};
    uint32_t* d_cnt_slot[3] = {};
    for (int m = 0; m < 3; ++m) {
        CUDA_CHECK(cudaMalloc(&d_rev_slot[m], (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_cnt_slot[m], (size_t)max_chunk * sizeof(uint32_t)));
    }

    uint32_t* d_fwd_batch[2] = {};
    uint32_t* d_over_src[2]  = {};
    uint32_t* d_over_dst[2]  = {};
    uint32_t* d_over_cnt[2]  = {};
    uint32_t* d_in_cnt[2]    = {};
    cudaStream_t s_compute, s_drain;
    CUDA_CHECK(cudaStreamCreate(&s_compute));
    CUDA_CHECK(cudaStreamCreate(&s_drain));

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMalloc(&d_fwd_batch[s], (size_t)batch_rows * K_out * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_over_src[s],  (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_over_dst[s],  (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_over_cnt[s],  sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_in_cnt[s],    sizeof(uint32_t)));
    }

    // -------- 分配 host pinned buffers --------
    // ping-pong drain buffers 让 phase A 的 CPU merge 与下一轮 D2H 重叠
    uint32_t* h_drain_rev[2] = {};
    uint32_t* h_drain_cnt[2] = {};
    for (int p = 0; p < 2; ++p) {
        CUDA_CHECK(cudaMallocHost(&h_drain_rev[p],
            (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_drain_cnt[p],
            (size_t)max_chunk * sizeof(uint32_t)));
    }

    uint32_t* h_over_src[2] = {};
    uint32_t* h_over_dst[2] = {};
    uint32_t* h_over_cnt[2] = {};
    uint32_t* h_in_cnt[2]   = {};
    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_over_src[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_over_dst[s], (size_t)overflow_cap * sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_over_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMallocHost(&h_in_cnt[s],   sizeof(uint32_t)));
    }

    // 事件：cross-stream 同步
    cudaEvent_t evt_kernel, evt_drain;
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_kernel, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evt_drain,  cudaEventDisableTiming));

    // 初始化 3 slots（chunk 0/1/2 的 rev_graph 起始为空）
    for (int m = 0; m < 3; ++m) {
        CUDA_CHECK(cudaMemsetAsync(d_rev_slot[m], 0xff,
            (size_t)max_chunk * K_rev_cap * sizeof(uint32_t), s_compute));
        CUDA_CHECK(cudaMemsetAsync(d_cnt_slot[m], 0,
            (size_t)max_chunk * sizeof(uint32_t), s_compute));
    }

    std::cout << "  [Method D] " << plan.label
              << ", num_chunks=" << num_chunks
              << ", max_chunk=" << max_chunk
              << ", batch_rows=" << batch_rows
              << ", overflow_cap_per_batch=" << overflow_cap << "\n";

    constexpr uint32_t SLOT_NULL_LO = 0xFFFFFFFFu;
    constexpr uint32_t SLOT_NULL_HI = 0xFFFFFFFFu;

    auto get_slot_range = [&](int chunk_id, uint32_t& jlo, uint32_t& jhi,
                              uint32_t*& rev_ptr, uint32_t*& cnt_ptr) {
        if (chunk_id < 0 || chunk_id >= num_chunks) {
            jlo = SLOT_NULL_LO; jhi = SLOT_NULL_HI;
            rev_ptr = d_rev_slot[0];   // 任意有效指针即可（kernel 不会写入）
            cnt_ptr = d_cnt_slot[0];
            return;
        }
        jlo = plan.starts[chunk_id];
        jhi = plan.starts[chunk_id + 1];
        int m = ((chunk_id % 3) + 3) % 3;
        rev_ptr = d_rev_slot[m];
        cnt_ptr = d_cnt_slot[m];
    };

    std::vector<uint32_t> all_overflow_src, all_overflow_dst;
    uint64_t total_in_chunk = 0, total_overflow = 0, total_dropped = 0;

    double t_phaseA = 0, t_phaseB = 0, t_merge = 0;
    auto tA = Clock::now();

    for (int c = 0; c < num_chunks; ++c) {
        // 当前 chunk 的源行范围 & 三个 slot 的 dst 范围
        uint32_t i_lo = plan.starts[c];
        uint32_t i_hi = plan.starts[c + 1];

        uint32_t s0_lo, s0_hi, s1_lo, s1_hi, s2_lo, s2_hi;
        uint32_t *s0_rev, *s0_cnt, *s1_rev, *s1_cnt, *s2_rev, *s2_cnt;
        get_slot_range(c - 1, s0_lo, s0_hi, s0_rev, s0_cnt);
        get_slot_range(c,     s1_lo, s1_hi, s1_rev, s1_cnt);
        get_slot_range(c + 1, s2_lo, s2_hi, s2_rev, s2_cnt);

        // -------- 跑 chunk c 的所有 source-row batches --------
        for (uint64_t bs = i_lo; bs < i_hi; bs += batch_rows) {
            uint32_t cur_rows = static_cast<uint32_t>(
                std::min<uint64_t>(batch_rows, i_hi - bs));
            int s = (((bs - i_lo) / batch_rows) & 1);

            // 等该 stream 上一轮 batch 的 overflow drain 完成
            cudaStreamSynchronize(s_compute);

            CUDA_CHECK(cudaMemsetAsync(d_over_cnt[s], 0, sizeof(uint32_t), s_compute));
            CUDA_CHECK(cudaMemsetAsync(d_in_cnt[s],  0, sizeof(uint32_t), s_compute));

            CUDA_CHECK(cudaMemcpyAsync(d_fwd_batch[s],
                fwd_graph_host + bs * K_out,
                (size_t)cur_rows * K_out * sizeof(uint32_t),
                cudaMemcpyHostToDevice, s_compute));

            uint64_t total = (uint64_t)cur_rows * K_out;
            int threads = 256;
            int blocks  = static_cast<int>((total + threads - 1) / threads);
            kern_rev_3slot<<<blocks, threads, 0, s_compute>>>(
                d_fwd_batch[s], bs, cur_rows, (uint32_t)K_out, (uint32_t)K_rev_cap,
                s0_lo, s0_hi, s0_rev, s0_cnt,
                s1_lo, s1_hi, s1_rev, s1_cnt,
                s2_lo, s2_hi, s2_rev, s2_cnt,
                d_over_src[s], d_over_dst[s], d_over_cnt[s], overflow_cap,
                d_in_cnt[s]);

            // D2H counts on s_compute（同 stream，与 kernel 串行；保证读到正确值）
            CUDA_CHECK(cudaMemcpyAsync(h_over_cnt[s], d_over_cnt[s],
                sizeof(uint32_t), cudaMemcpyDeviceToHost, s_compute));
            CUDA_CHECK(cudaMemcpyAsync(h_in_cnt[s], d_in_cnt[s],
                sizeof(uint32_t), cudaMemcpyDeviceToHost, s_compute));
            cudaStreamSynchronize(s_compute);

            uint32_t ofc = *h_over_cnt[s];
            uint32_t inc = *h_in_cnt[s];
            total_in_chunk += inc;
            if (ofc > overflow_cap) { total_dropped += (ofc - overflow_cap); ofc = overflow_cap; }
            total_overflow += ofc;

            if (ofc > 0) {
                CUDA_CHECK(cudaMemcpyAsync(h_over_src[s], d_over_src[s],
                    (size_t)ofc * sizeof(uint32_t), cudaMemcpyDeviceToHost, s_compute));
                CUDA_CHECK(cudaMemcpyAsync(h_over_dst[s], d_over_dst[s],
                    (size_t)ofc * sizeof(uint32_t), cudaMemcpyDeviceToHost, s_compute));
                cudaStreamSynchronize(s_compute);
                all_overflow_src.insert(all_overflow_src.end(),
                    h_over_src[s], h_over_src[s] + ofc);
                all_overflow_dst.insert(all_overflow_dst.end(),
                    h_over_dst[s], h_over_dst[s] + ofc);
            }
        }
        // 标记 chunk c 的 kernel 全部完成（用于 s_drain 等）
        CUDA_CHECK(cudaEventRecord(evt_kernel, s_compute));

        // -------- 异步 drain rev[c-1] 到 host (s_drain) --------
        if (c >= 1) {
            int slot_prev = ((c - 1) % 3 + 3) % 3;
            int dbuf = (c - 1) & 1;            // ping-pong host buffer
            uint32_t prev_lo = plan.starts[c - 1];
            uint32_t prev_sz = plan.starts[c] - prev_lo;

            // s_drain 等 chunk c kernel 完成
            CUDA_CHECK(cudaStreamWaitEvent(s_drain, evt_kernel, 0));
            CUDA_CHECK(cudaMemcpyAsync(h_drain_rev[dbuf], d_rev_slot[slot_prev],
                (size_t)prev_sz * K_rev_cap * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, s_drain));
            CUDA_CHECK(cudaMemcpyAsync(h_drain_cnt[dbuf], d_cnt_slot[slot_prev],
                (size_t)prev_sz * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, s_drain));
            CUDA_CHECK(cudaEventRecord(evt_drain, s_drain));

            // s_compute 在 D2H 完成后再 reset slot_prev（chunk c+2 会用它）
            CUDA_CHECK(cudaStreamWaitEvent(s_compute, evt_drain, 0));
            CUDA_CHECK(cudaMemsetAsync(d_rev_slot[slot_prev], 0xff,
                (size_t)max_chunk * K_rev_cap * sizeof(uint32_t), s_compute));
            CUDA_CHECK(cudaMemsetAsync(d_cnt_slot[slot_prev], 0,
                (size_t)max_chunk * sizeof(uint32_t), s_compute));

            // host 侧：等 D2H 完成后 CPU merge（这一段与 s_compute 上的 reset
            // + 下轮 c+1 的 batch 重叠运行）
            CUDA_CHECK(cudaEventSynchronize(evt_drain));
            auto tm = Clock::now();
            uint32_t prev_jlo = plan.starts[c - 1];
            uint32_t prev_jhi = plan.starts[c];
            merge_rev_into_output_chunk(
                output_graph_host, h_drain_rev[dbuf], h_drain_cnt[dbuf],
                prev_jlo, prev_jhi, (uint32_t)K_out, (uint32_t)K_rev_cap);
            t_merge += std::chrono::duration<double>(Clock::now() - tm).count();
        }
    }

    // -------- 收尾：drain 最后一个 chunk rev[n-1] --------
    {
        int last = num_chunks - 1;
        int slot_last = ((last % 3) + 3) % 3;
        int dbuf = last & 1;
        uint32_t last_lo = plan.starts[last];
        uint32_t last_sz = plan.starts[last + 1] - last_lo;

        CUDA_CHECK(cudaStreamWaitEvent(s_drain, evt_kernel, 0));
        CUDA_CHECK(cudaMemcpyAsync(h_drain_rev[dbuf], d_rev_slot[slot_last],
            (size_t)last_sz * K_rev_cap * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, s_drain));
        CUDA_CHECK(cudaMemcpyAsync(h_drain_cnt[dbuf], d_cnt_slot[slot_last],
            (size_t)last_sz * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, s_drain));
        CUDA_CHECK(cudaStreamSynchronize(s_drain));

        auto tm = Clock::now();
        merge_rev_into_output_chunk(
            output_graph_host, h_drain_rev[dbuf], h_drain_cnt[dbuf],
            last_lo, plan.starts[last + 1], (uint32_t)K_out, (uint32_t)K_rev_cap);
        t_merge += std::chrono::duration<double>(Clock::now() - tm).count();
    }

    t_phaseA = std::chrono::duration<double>(Clock::now() - tA).count();

    // ============================================================
    // Phase B: 远距 overflow 边在 host 上分桶 + merge
    // ============================================================
    auto tB = Clock::now();
    if (!all_overflow_src.empty()) {
        size_t n_over = all_overflow_src.size();
        const size_t nc = plan.starts.size() - 1;

        auto chunk_of = [&](uint32_t j) -> size_t {
            auto it = std::upper_bound(plan.starts.begin(), plan.starts.end(), j);
            return static_cast<size_t>(it - plan.starts.begin()) - 1;
        };

        std::vector<size_t> chunk_count(nc, 0);
        std::vector<size_t> edge_chunk(n_over);
        #pragma omp parallel for schedule(static)
        for (size_t e = 0; e < n_over; ++e)
            edge_chunk[e] = chunk_of(all_overflow_dst[e]);
        for (size_t e = 0; e < n_over; ++e)
            chunk_count[edge_chunk[e]]++;

        for (size_t c = 0; c < nc; ++c) {
            if (chunk_count[c] == 0) continue;
            uint32_t cj_lo = plan.starts[c];
            uint32_t cj_hi = plan.starts[c + 1];
            uint32_t cz    = cj_hi - cj_lo;

            std::vector<uint32_t> mini_rev((size_t)cz * K_rev_cap, 0xFFFFFFFFu);
            std::vector<uint32_t> mini_cnt(cz, 0);

            for (size_t e = 0; e < n_over; ++e) {
                if (edge_chunk[e] != c) continue;
                uint32_t i = all_overflow_src[e];
                uint32_t j = all_overflow_dst[e];
                uint32_t local_j = j - cj_lo;
                uint32_t pos = mini_cnt[local_j]++;
                if (pos < (uint32_t)K_rev_cap)
                    mini_rev[(size_t)local_j * K_rev_cap + pos] = i;
            }
            merge_rev_into_output_chunk(
                output_graph_host, mini_rev.data(), mini_cnt.data(),
                cj_lo, cj_hi, (uint32_t)K_out, (uint32_t)K_rev_cap);
        }
    }
    t_phaseB = std::chrono::duration<double>(Clock::now() - tB).count();

    // -------- Stats --------
    uint64_t total_edges = total_in_chunk + total_overflow + total_dropped;
    double pct_in   = total_edges ? 100.0 * total_in_chunk / total_edges : 0.0;
    double pct_over = total_edges ? 100.0 * total_overflow / total_edges : 0.0;
    double pct_drop = total_edges ? 100.0 * total_dropped / total_edges : 0.0;

    std::cout << "  [Stats] in 3-slot window: " << total_in_chunk
              << " (" << std::fixed << std::setprecision(2) << pct_in << "%)\n";
    std::cout << "  [Stats] far-overflow:     " << total_overflow
              << " (" << pct_over << "%)\n";
    if (total_dropped > 0)
        std::cout << "  [WARN ] dropped edges:    " << total_dropped
                  << " (" << pct_drop << "%) — overflow_cap exceeded.\n";
    std::cout << "  [Timing] phase A=" << t_phaseA << "s, phase B (overflow)="
              << t_phaseB << "s, [included] CPU merge=" << t_merge << "s\n";

    // -------- 清理 --------
    for (int p = 0; p < 2; ++p) {
        cudaFreeHost(h_drain_rev[p]);
        cudaFreeHost(h_drain_cnt[p]);
    }
    for (int s = 0; s < 2; ++s) {
        cudaFree(d_fwd_batch[s]);
        cudaFree(d_over_src[s]);
        cudaFree(d_over_dst[s]);
        cudaFree(d_over_cnt[s]);
        cudaFree(d_in_cnt[s]);
        cudaFreeHost(h_over_src[s]);
        cudaFreeHost(h_over_dst[s]);
        cudaFreeHost(h_over_cnt[s]);
        cudaFreeHost(h_in_cnt[s]);
    }
    for (int m = 0; m < 3; ++m) {
        cudaFree(d_rev_slot[m]);
        cudaFree(d_cnt_slot[m]);
    }
    cudaEventDestroy(evt_kernel);
    cudaEventDestroy(evt_drain);
    cudaStreamDestroy(s_compute);
    cudaStreamDestroy(s_drain);
}

// =====================================================================
// Method B: pre-shuffle pairs by destination chunk
// =====================================================================
static void run_method_B(
    const uint32_t* fwd_graph_host,
    uint32_t* output_graph_host,
    int64_t N, int32_t K_out, int32_t K_rev_cap,
    const ChunkPlan& plan,
    size_t gpu_budget_bytes)
{
    using Clock = std::chrono::steady_clock;
    const size_t num_chunks = plan.starts.size() - 1;
    const size_t total_edges = (size_t)N * K_out;

    std::cout << "  [Method B] " << plan.label
              << ", num_chunks=" << num_chunks
              << ", total_edges=" << total_edges
              << " (~" << total_edges * 8 / 1e9 << " GB pair buffer)\n";

    // ---- Pass 1a: histogram (每个 chunk 多少条边) ----
    auto t0 = Clock::now();
    std::vector<size_t> chunk_count(num_chunks, 0);

    // chunk lookup: 给定 j，返回 chunk c。用二分（plan.starts 单调）
    auto chunk_of = [&](uint32_t j) -> size_t {
        // upper_bound - 1
        auto it = std::upper_bound(plan.starts.begin(), plan.starts.end(), j);
        return static_cast<size_t>(it - plan.starts.begin()) - 1;
    };

    // OpenMP per-thread 局部直方图，再合并
    int nthreads = omp_get_max_threads();
    std::vector<std::vector<size_t>> local_hist(nthreads, std::vector<size_t>(num_chunks, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& h = local_hist[tid];
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < N; ++i) {
            const uint32_t* row = fwd_graph_host + (uint64_t)i * K_out;
            for (int k = 0; k < K_out; ++k)
                h[chunk_of(row[k])]++;
        }
    }
    for (int t = 0; t < nthreads; ++t)
        for (size_t c = 0; c < num_chunks; ++c)
            chunk_count[c] += local_hist[t][c];

    std::vector<size_t> chunk_offset(num_chunks + 1, 0);
    for (size_t c = 0; c < num_chunks; ++c)
        chunk_offset[c + 1] = chunk_offset[c] + chunk_count[c];

    // ---- Pass 1b: 分配 pair 数组并散写 ----
    // 用两个独立数组（src/dst）便于 GPU coalesced load
    uint32_t* h_pairs_src = nullptr;
    uint32_t* h_pairs_dst = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pairs_src, total_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_pairs_dst, total_edges * sizeof(uint32_t)));

    std::vector<size_t> cursor(chunk_offset.begin(), chunk_offset.end() - 1);
    // 用原子游标避免线程竞争。OpenMP atomic capture 即可。
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < N; ++i) {
        const uint32_t* row = fwd_graph_host + (uint64_t)i * K_out;
        for (int k = 0; k < K_out; ++k) {
            uint32_t j = row[k];
            size_t c = chunk_of(j);
            size_t pos;
            #pragma omp atomic capture
            pos = cursor[c]++;
            h_pairs_src[pos] = static_cast<uint32_t>(i);
            h_pairs_dst[pos] = j;
        }
    }
    double t_shuffle = std::chrono::duration<double>(Clock::now() - t0).count();
    std::cout << "  [Method B] Pass 1 (shuffle) " << t_shuffle << "s\n";

    // ---- Pass 2: 对每个 chunk 跑 kernel ----
    uint32_t max_chunk = 0;
    size_t   max_pairs = 0;
    for (size_t c = 0; c + 1 < plan.starts.size(); ++c) {
        max_chunk = std::max(max_chunk, plan.starts[c + 1] - plan.starts[c]);
        max_pairs = std::max(max_pairs, chunk_count[c]);
    }

    uint32_t* d_rev_graph = nullptr;
    uint32_t* d_rev_count = nullptr;
    uint32_t* d_pairs_src = nullptr;
    uint32_t* d_pairs_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rev_graph, (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_count, (size_t)max_chunk * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pairs_src, max_pairs * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_pairs_dst, max_pairs * sizeof(uint32_t)));

    uint32_t* h_rev_graph_chunk = nullptr;
    uint32_t* h_rev_count_chunk = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_rev_graph_chunk,
        (size_t)max_chunk * K_rev_cap * sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocHost(&h_rev_count_chunk,
        (size_t)max_chunk * sizeof(uint32_t)));

    double t_kern = 0, t_xfer = 0, t_merge = 0;

    for (size_t c = 0; c + 1 < plan.starts.size(); ++c) {
        uint32_t j_lo    = plan.starts[c];
        uint32_t j_hi    = plan.starts[c + 1];
        uint32_t chunk_sz = j_hi - j_lo;
        size_t   n_pairs  = chunk_count[c];
        size_t   p_off    = chunk_offset[c];

        CUDA_CHECK(cudaMemset(d_rev_count, 0, chunk_sz * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_rev_graph, 0xff,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t)));

        auto tx = Clock::now();
        CUDA_CHECK(cudaMemcpy(d_pairs_src, h_pairs_src + p_off,
            n_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pairs_dst, h_pairs_dst + p_off,
            n_pairs * sizeof(uint32_t), cudaMemcpyHostToDevice));
        t_xfer += std::chrono::duration<double>(Clock::now() - tx).count();

        auto tk = Clock::now();
        int threads = 256;
        int blocks = static_cast<int>((n_pairs + threads - 1) / threads);
        if (blocks > 0) {
            kern_rev_scatter_pairs<<<blocks, threads>>>(
                d_pairs_src, d_pairs_dst, n_pairs,
                j_lo, (uint32_t)K_rev_cap, d_rev_graph, d_rev_count);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        t_kern += std::chrono::duration<double>(Clock::now() - tk).count();

        auto td = Clock::now();
        CUDA_CHECK(cudaMemcpy(h_rev_count_chunk, d_rev_count,
            chunk_sz * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rev_graph_chunk, d_rev_graph,
            (size_t)chunk_sz * K_rev_cap * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        t_xfer += std::chrono::duration<double>(Clock::now() - td).count();

        auto tm = Clock::now();
        merge_rev_into_output_chunk(
            output_graph_host, h_rev_graph_chunk, h_rev_count_chunk,
            j_lo, j_hi, (uint32_t)K_out, (uint32_t)K_rev_cap);
        t_merge += std::chrono::duration<double>(Clock::now() - tm).count();
    }

    std::cout << "  [Timing] xfer=" << std::fixed << std::setprecision(2) << t_xfer
              << "s, kern=" << t_kern << "s, merge=" << t_merge << "s\n";

    cudaFreeHost(h_rev_graph_chunk);
    cudaFreeHost(h_rev_count_chunk);
    cudaFreeHost(h_pairs_src);
    cudaFreeHost(h_pairs_dst);
    cudaFree(d_rev_graph);
    cudaFree(d_rev_count);
    cudaFree(d_pairs_src);
    cudaFree(d_pairs_dst);
}

// =====================================================================
// Method E: 单遍扫描 + per-chunk 磁盘 flush 的 scatter buffer（见
// kern_scatter_rank 和 edge_chunk_accumulator.hpp），rank 保序，不要求
// forward/output 整图常驻 host。
//
// 跟 A/B/C/D 不同：这里不吃预先读好的 host 数组，自己流式读
// forward_graph.bin，也流式写 output —— 是这四个方法之外唯一一个不要求
// N*K_out*4B 常驻 host 两份（fwd + output）的路径。main() 对 method "E"
// 单独分支，跳过现有的整图 read_forward_graph/output_graph 拷贝。
//
// Phase 1 在这份文件里都保持单 stream、同步的写法（每个 batch: 读盘 → H2D
// → kernel → 同步 → D2H 计数 → 按需 drain），没有像 A/C/D 那样做双缓冲/
// 多 stream 重叠——本机没有 GPU 没法在这上面调优，先保证正确、简单，异步
// 重叠留作后续在真实硬件上验证过一遍之后再加。
// =====================================================================
static void run_method_E(
    const std::string& fwd_path,
    const std::string& out_path,
    int64_t N, int32_t K_out,
    const ChunkPlan& plan,
    size_t gpu_budget_bytes,
    size_t edge_buffer_bytes,
    const std::string& tmp_dir)
{
    using Clock = std::chrono::steady_clock;
    const int32_t num_chunks = static_cast<int32_t>(plan.starts.size()) - 1;

    std::filesystem::create_directories(tmp_dir);

    // -------- Phase 1 GPU 预算切分 --------
    // 50% 给 chunk scatter buffer（跨 batch 持续累积），剩下的给双缓冲级别
    // 简化后的单份 fwd_batch staging + worst-case overflow buffer。
    size_t chunk_buf_budget = static_cast<size_t>(gpu_budget_bytes * 0.5);
    size_t capacity_per_chunk = std::max<size_t>(
        16, chunk_buf_budget / static_cast<size_t>(num_chunks) / sizeof(edge_chunk::EdgeRecord));
    size_t chunk_buf_bytes = static_cast<size_t>(num_chunks) * capacity_per_chunk * sizeof(edge_chunk::EdgeRecord);

    size_t remaining = (gpu_budget_bytes > chunk_buf_bytes)
        ? (gpu_budget_bytes - chunk_buf_bytes) : (256ULL << 20);
    // 每行成本：fwd_batch (K_out*4) + overflow worst-case (K_out * sizeof(EdgeRecord))
    size_t per_row_bytes = K_out * sizeof(uint32_t) + static_cast<size_t>(K_out) * sizeof(edge_chunk::EdgeRecord);
    uint32_t batch_rows = static_cast<uint32_t>(std::max<size_t>(4096, remaining / per_row_bytes));
    batch_rows = static_cast<uint32_t>(std::min<int64_t>(batch_rows, N));
    uint32_t overflow_cap = batch_rows * static_cast<uint32_t>(K_out);  // 一整批全挤爆同一个 chunk 也能兜住

    std::cout << "  [Method E] " << plan.label
              << ", num_chunks=" << num_chunks
              << ", capacity_per_chunk=" << capacity_per_chunk << " records"
              << " (" << capacity_per_chunk * sizeof(edge_chunk::EdgeRecord) / 1e6 << " MB/chunk)"
              << ", batch_rows=" << batch_rows
              << ", overflow_cap=" << overflow_cap << "\n";

    // -------- GPU 侧分配 --------
    std::vector<uint32_t> starts_u32(plan.starts.begin(), plan.starts.end());
    uint32_t* d_chunk_starts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chunk_starts, starts_u32.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_chunk_starts, starts_u32.data(),
        starts_u32.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    edge_chunk::EdgeRecord* d_chunk_buf = nullptr;
    uint32_t* d_chunk_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chunk_buf,
        static_cast<size_t>(num_chunks) * capacity_per_chunk * sizeof(edge_chunk::EdgeRecord)));
    CUDA_CHECK(cudaMalloc(&d_chunk_count, static_cast<size_t>(num_chunks) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_chunk_count, 0, static_cast<size_t>(num_chunks) * sizeof(uint32_t)));

    edge_chunk::EdgeRecord* d_overflow_buf = nullptr;
    uint32_t* d_overflow_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_overflow_buf, static_cast<size_t>(overflow_cap) * sizeof(edge_chunk::EdgeRecord)));
    CUDA_CHECK(cudaMalloc(&d_overflow_count, sizeof(uint32_t)));

    uint32_t* d_fwd_batch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fwd_batch, static_cast<size_t>(batch_rows) * K_out * sizeof(uint32_t)));

    uint32_t* h_fwd_batch = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_fwd_batch, static_cast<size_t>(batch_rows) * K_out * sizeof(uint32_t)));
    std::vector<uint32_t> h_chunk_count(static_cast<size_t>(num_chunks));
    edge_chunk::EdgeRecord* h_overflow_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_overflow_buf, static_cast<size_t>(overflow_cap) * sizeof(edge_chunk::EdgeRecord)));
    edge_chunk::EdgeRecord* h_drain_buf = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_drain_buf, capacity_per_chunk * sizeof(edge_chunk::EdgeRecord)));

    auto acc = edge_chunk::EdgeChunkAccumulator::create(tmp_dir, num_chunks, edge_buffer_bytes);
    acc.start();

    // -------- Phase 1: 单遍扫描 forward_graph.bin，散射进 per-chunk buffer --------
    auto tA = Clock::now();
    {
        std::ifstream fwd_in(fwd_path, std::ios::binary);
        if (!fwd_in.is_open())
            throw std::runtime_error("Method E: cannot open " + fwd_path);
        fwd_in.seekg(sizeof(int64_t) + sizeof(int32_t), std::ios::beg);  // 跳过 N/K_out header

        uint64_t total_in_chunk = 0, total_overflow = 0;

        for (int64_t bs = 0; bs < N; bs += batch_rows) {
            uint32_t cur_rows = static_cast<uint32_t>(std::min<int64_t>(batch_rows, N - bs));
            size_t n_elems = static_cast<size_t>(cur_rows) * K_out;

            fwd_in.read(reinterpret_cast<char*>(h_fwd_batch),
                        static_cast<std::streamsize>(n_elems * sizeof(uint32_t)));
            if (!fwd_in.good())
                throw std::runtime_error("Method E: failed to read forward graph batch at row " + std::to_string(bs));

            CUDA_CHECK(cudaMemcpy(d_fwd_batch, h_fwd_batch,
                n_elems * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_overflow_count, 0, sizeof(uint32_t)));

            uint64_t total_edges = static_cast<uint64_t>(cur_rows) * K_out;
            int threads = 256;
            int blocks = static_cast<int>((total_edges + threads - 1) / threads);
            kern_scatter_rank<<<blocks, threads>>>(
                d_fwd_batch, static_cast<uint64_t>(bs), cur_rows, static_cast<uint32_t>(K_out),
                d_chunk_starts, static_cast<uint32_t>(num_chunks),
                d_chunk_buf, static_cast<uint32_t>(capacity_per_chunk), d_chunk_count,
                d_overflow_buf, d_overflow_count, overflow_cap);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_chunk_count.data(), d_chunk_count,
                static_cast<size_t>(num_chunks) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t ofc = 0;
            CUDA_CHECK(cudaMemcpy(&ofc, d_overflow_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (ofc > overflow_cap)
                throw std::runtime_error(
                    "Method E: overflow buffer exceeded capacity (" + std::to_string(ofc) +
                    " > " + std::to_string(overflow_cap) +
                    "). Increase --gpu-budget-mb or --edge-buffer-mb.");
            if (ofc > 0) {
                CUDA_CHECK(cudaMemcpy(h_overflow_buf, d_overflow_buf,
                    static_cast<size_t>(ofc) * sizeof(edge_chunk::EdgeRecord), cudaMemcpyDeviceToHost));
                for (uint32_t e = 0; e < ofc; ++e) {
                    uint32_t c = edge_chunk::chunk_of(plan.starts.data(),
                        static_cast<uint32_t>(num_chunks), h_overflow_buf[e].dst);
                    acc.add_records(static_cast<int32_t>(c), &h_overflow_buf[e], 1);
                }
                total_overflow += ofc;
            }

            for (int32_t c = 0; c < num_chunks; ++c) {
                if (h_chunk_count[static_cast<size_t>(c)] >= capacity_per_chunk) {
                    uint32_t take = static_cast<uint32_t>(capacity_per_chunk);
                    CUDA_CHECK(cudaMemcpy(h_drain_buf, d_chunk_buf + static_cast<size_t>(c) * capacity_per_chunk,
                        static_cast<size_t>(take) * sizeof(edge_chunk::EdgeRecord), cudaMemcpyDeviceToHost));
                    acc.add_records(c, h_drain_buf, take);
                    CUDA_CHECK(cudaMemset(d_chunk_count + c, 0, sizeof(uint32_t)));
                    total_in_chunk += take;
                }
            }
        }

        // 收尾：drain 每个 chunk 剩下没写满的部分
        CUDA_CHECK(cudaMemcpy(h_chunk_count.data(), d_chunk_count,
            static_cast<size_t>(num_chunks) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (int32_t c = 0; c < num_chunks; ++c) {
            uint32_t cnt = h_chunk_count[static_cast<size_t>(c)];
            if (cnt == 0) continue;
            uint32_t take = std::min(cnt, static_cast<uint32_t>(capacity_per_chunk));
            CUDA_CHECK(cudaMemcpy(h_drain_buf, d_chunk_buf + static_cast<size_t>(c) * capacity_per_chunk,
                static_cast<size_t>(take) * sizeof(edge_chunk::EdgeRecord), cudaMemcpyDeviceToHost));
            acc.add_records(c, h_drain_buf, take);
            total_in_chunk += take;
        }

        double t_phase1 = std::chrono::duration<double>(Clock::now() - tA).count();
        uint64_t total_edges_all = total_in_chunk + total_overflow;
        double pct_over = total_edges_all ? 100.0 * total_overflow / total_edges_all : 0.0;
        std::cout << "  [Stats] scattered edges: " << total_edges_all
                  << " (overflow path: " << total_overflow << ", " << std::fixed
                  << std::setprecision(2) << pct_over << "%)\n";
        std::cout << "  [Timing] Phase 1 (scatter) = " << t_phase1 << "s\n";
    }

    acc.finish();

    // -------- 清理 Phase 1 的 GPU/host 分配 --------
    cudaFree(d_chunk_starts);
    cudaFree(d_chunk_buf);
    cudaFree(d_chunk_count);
    cudaFree(d_overflow_buf);
    cudaFree(d_overflow_count);
    cudaFree(d_fwd_batch);
    cudaFreeHost(h_fwd_batch);
    cudaFreeHost(h_overflow_buf);
    cudaFreeHost(h_drain_buf);

    // -------- Phase 2: 按 chunk 顺序重新顺序读一遍 forward_graph.bin（这次
    // 是当"底稿"用，不是当边的源表用——两次顺序读，用途不同，参见此前的
    // 讨论：Phase 1 读整张图当 source table，Phase 2 每个 chunk 只读自己
    // 那一段当 output 的初始值），rank 排序合并，直接流式写出。--------
    auto tB = Clock::now();
    {
        std::ifstream fwd_in2(fwd_path, std::ios::binary);
        if (!fwd_in2.is_open())
            throw std::runtime_error("Method E: cannot reopen " + fwd_path);
        fwd_in2.seekg(sizeof(int64_t) + sizeof(int32_t), std::ios::beg);

        graph_io::StreamingGraphWriter writer(out_path, N, K_out);

        std::vector<uint32_t> row_block;
        std::vector<edge_chunk::EdgeRecord> chunk_edges;

        for (int32_t c = 0; c < num_chunks; ++c) {
            uint32_t j_lo = plan.starts[static_cast<size_t>(c)];
            uint32_t j_hi = plan.starts[static_cast<size_t>(c) + 1];
            uint32_t chunk_sz = j_hi - j_lo;

            row_block.resize(static_cast<size_t>(chunk_sz) * K_out);
            fwd_in2.read(reinterpret_cast<char*>(row_block.data()),
                        static_cast<std::streamsize>(row_block.size() * sizeof(uint32_t)));
            if (!fwd_in2.good())
                throw std::runtime_error("Method E Phase 2: failed to read forward rows for chunk " + std::to_string(c));

            acc.read_chunk(c, chunk_edges);
            auto csr = edge_chunk::build_rank_ordered_csr(chunk_edges, j_lo, chunk_sz);

            #pragma omp parallel for schedule(dynamic, 1024)
            for (int64_t lj = 0; lj < static_cast<int64_t>(chunk_sz); ++lj) {
                uint32_t lo = csr.row_offsets[static_cast<size_t>(lj)];
                uint32_t hi = csr.row_offsets[static_cast<size_t>(lj) + 1];
                edge_chunk::merge_rank_ordered_into_row(
                    row_block.data() + static_cast<size_t>(lj) * K_out,
                    csr.src.data() + lo, hi - lo, static_cast<uint32_t>(K_out));
            }

            writer.write_rows(row_block.data(), j_lo, chunk_sz);
        }

        if (writer.rows_written() != N)
            throw std::runtime_error("Method E: wrote " + std::to_string(writer.rows_written()) +
                                     " rows, expected " + std::to_string(N));
    }
    double t_phase2 = std::chrono::duration<double>(Clock::now() - tB).count();
    std::cout << "  [Timing] Phase 2 (merge + stream write) = " << t_phase2 << "s\n";

    std::filesystem::remove_all(tmp_dir);
}

// =====================================================================
// main
// =====================================================================
int main(int argc, char** argv)
{
    try {
        po::options_description desc("CAGRA chunked reverse-graph + merge (5 methods)");
        desc.add_options()
            ("help,h", "Show help")
            ("forward-graph,g", po::value<std::string>()->required(),
                "Input pruned forward graph (cagra_graph.bin format)")
            ("output,o", po::value<std::string>()->required(),
                "Output graph after reverse-edge merge")
            ("method,m", po::value<std::string>()->default_value("A"),
                "Method: A (kernel-filter, multi-pass PCIe), "
                "B (pre-shuffle pairs), "
                "C (bucket-aligned single-pass + overflow capture, lossless), "
                "D (3-slot sliding window over bucket-aligned chunks, "
                "captures ±1 cross-chunk edges directly, lossless), "
                "E (per-chunk disk-backed scatter buffer, rank-priority preserving, "
                "single pass, streams fwd graph + output from/to disk -- doesn't "
                "require the full graph resident in host memory like A/B/C/D do)")
            ("gpu-budget-mb", po::value<size_t>()->default_value(8000),
                "GPU memory budget for chunk buffers (MB)")
            ("edge-buffer-mb", po::value<size_t>()->default_value(1024),
                "Method E only: total host-side staging budget (MB), shared across all "
                "chunks' disk-flush buffers -- same semantics as bucket2's --bucket-vec-buffer")
            ("tmp-dir", po::value<std::string>()->default_value(""),
                "Method E only: scratch directory for per-chunk edge files "
                "(default: <output dir>/.optimize_chunked_e_tmp, removed on success)")
            ("bucket-offsets", po::value<std::string>()->default_value(""),
                "Bucket boundaries file (required for methods C and D; optional for E -- "
                "if given, chunks align to buckets, otherwise E falls back to uniform "
                "GPU-budget-sized chunks since it doesn't need bucket locality for correctness)")
            ("rev-degree-cap", po::value<int32_t>()->default_value(0),
                "Cap on reverse edges per node (0 = K_out, the default). Unused by method E, "
                "which keeps variable-length per-node candidate lists instead of a fixed cap.")
            ("save-npy", po::bool_switch()->default_value(false),
                "Also write a .npy alongside the .bin");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);

        const std::string fwd_path  = vm["forward-graph"].as<std::string>();
        const std::string out_path  = vm["output"].as<std::string>();
        const std::string method    = vm["method"].as<std::string>();
        const size_t gpu_budget_mb  = vm["gpu-budget-mb"].as<size_t>();
        const size_t gpu_budget_b   = gpu_budget_mb << 20;
        const size_t edge_buffer_mb = vm["edge-buffer-mb"].as<size_t>();
        const size_t edge_buffer_b  = edge_buffer_mb << 20;
        std::string tmp_dir_arg     = vm["tmp-dir"].as<std::string>();
        const std::string bo_path   = vm["bucket-offsets"].as<std::string>();
        int32_t rev_cap_arg         = vm["rev-degree-cap"].as<int32_t>();
        const bool save_npy         = vm["save-npy"].as<bool>();

        if (method != "A" && method != "B" && method != "C" && method != "D" && method != "E")
            throw std::runtime_error("--method must be A, B, C, D, or E");
        if ((method == "C" || method == "D") && bo_path.empty())
            throw std::runtime_error("Method " + method + " requires --bucket-offsets");

        using Clock = std::chrono::steady_clock;
        auto t_start = Clock::now();

        // ====================================================================
        // Method E takes a completely different path: it never loads the full
        // forward/output graph into host memory (that's the whole point), so
        // it can't share the "read_forward_graph -> array -> write_graph" flow
        // the other four methods use below. Handle it here and return early.
        // ====================================================================
        if (method == "E") {
            std::cout << "=== Step 1: Reading forward graph header ===\n";
            int64_t N = 0; int32_t K_out = 0;
            graph_io::read_forward_graph_header(fwd_path, N, K_out);
            std::cout << "  N=" << N << " K_out=" << K_out << "\n";

            std::string tmp_dir = tmp_dir_arg.empty()
                ? std::filesystem::path(out_path).parent_path().string() + "/.optimize_chunked_e_tmp"
                : tmp_dir_arg;

            std::cout << "=== Step 2: Planning chunks ===\n";
            ChunkPlan plan;
            if (!bo_path.empty()) {
                auto offsets = read_bucket_offsets(bo_path);
                if (offsets.back() != static_cast<uint32_t>(N))
                    throw std::runtime_error(
                        "bucket_offsets.back() (" + std::to_string(offsets.back()) +
                        ") != N (" + std::to_string(N) + ")");
                // K_out 当"每个节点期望反向边数"的启发式估计，用来复用现有的
                // bucket-aligned 分块公式 -- 分块大小只影响性能（buffer 摊得
                // 多细），不影响正确性（EdgeChunkAccumulator 写满就 flush，
                // 加上共享 overflow buffer 兜底，任意 chunk 数都是对的）。
                plan = plan_bucket_aligned_chunks(offsets, K_out, gpu_budget_b);
            } else {
                plan = plan_uniform_chunks(N, K_out, gpu_budget_b);
            }
            std::cout << "  " << plan.label << ", num_chunks=" << plan.starts.size() - 1 << "\n";

            std::cout << "=== Step 3: Reverse graph + merge (method E) ===\n";
            auto t2 = Clock::now();
            run_method_E(fwd_path, out_path, N, K_out, plan, gpu_budget_b, edge_buffer_b, tmp_dir);
            std::cout << "  Done [" << std::chrono::duration<double>(Clock::now() - t2).count()
                      << "s]\n";

            if (save_npy) {
                // Optional diagnostic/interop output only -- reloads the just-written
                // file whole, unlike the streaming core path above.
                std::vector<uint32_t> out_graph;
                int64_t N2 = 0; int32_t K2 = 0;
                graph_io::read_forward_graph(out_path, out_graph, N2, K2);
                std::vector<int64_t> g64(out_graph.size());
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < out_graph.size(); ++i)
                    g64[i] = static_cast<int64_t>(out_graph[i]);
                auto p = std::filesystem::path(out_path);
                p.replace_extension(".npy");
                load::write_npy_int64_2d(p.string(), g64.data(), N2, K2);
                std::cout << "  Wrote " << p.string() << " (npy)\n";
            }

            double total = std::chrono::duration<double>(Clock::now() - t_start).count();
            std::cout << "\n=== Done ===\n";
            std::cout << "Method: E\n";
            std::cout << "Total: " << std::fixed << std::setprecision(3) << total << "s\n";
            return 0;
        }

        // ---- Load forward graph ----
        std::cout << "=== Step 1: Loading forward graph ===\n";
        auto t1 = Clock::now();
        std::vector<uint32_t> fwd_graph;
        int64_t N = 0; int32_t K_out = 0;
        read_forward_graph(fwd_path, fwd_graph, N, K_out);
        const int32_t K_rev_cap = (rev_cap_arg > 0) ? rev_cap_arg : K_out;
        std::cout << "  N=" << N << " K_out=" << K_out
                  << " K_rev_cap=" << K_rev_cap
                  << " (" << fwd_graph.size() * sizeof(uint32_t) / 1e9 << " GB)"
                  << " [" << std::chrono::duration<double>(Clock::now() - t1).count() << "s]\n";

        // pinned-register the forward graph in place (avoid double allocation)
        CUDA_CHECK(cudaHostRegister(fwd_graph.data(),
            fwd_graph.size() * sizeof(uint32_t), cudaHostRegisterDefault));

        // output_graph initialized to fwd_graph (host)
        std::vector<uint32_t> output_graph = fwd_graph;

        // ---- Plan chunks ----
        // 方案 D 的 GPU 同时驻留 3 个 chunk slot，所以每个 chunk 容量限制要 ÷3。
        size_t plan_budget_b = (method == "D") ? gpu_budget_b / 3 : gpu_budget_b;
        ChunkPlan plan;
        if (method == "C" || method == "D") {
            auto offsets = read_bucket_offsets(bo_path);
            if (offsets.back() != static_cast<uint32_t>(N))
                throw std::runtime_error(
                    "bucket_offsets.back() (" + std::to_string(offsets.back()) +
                    ") != N (" + std::to_string(N) + ")");
            plan = plan_bucket_aligned_chunks(offsets, K_rev_cap, plan_budget_b);
        } else {
            plan = plan_uniform_chunks(N, K_rev_cap, plan_budget_b);
        }

        // ---- Run selected method ----
        std::cout << "=== Step 2: Reverse graph + merge (method " << method << ") ===\n";
        auto t2 = Clock::now();
        if (method == "A") {
            run_method_A(fwd_graph.data(), output_graph.data(),
                         N, K_out, K_rev_cap, plan, gpu_budget_b);
        } else if (method == "B") {
            run_method_B(fwd_graph.data(), output_graph.data(),
                         N, K_out, K_rev_cap, plan, gpu_budget_b);
        } else if (method == "C") {
            run_method_C(fwd_graph.data(), output_graph.data(),
                         N, K_out, K_rev_cap, plan, gpu_budget_b);
        } else /* method == "D" */ {
            run_method_D(fwd_graph.data(), output_graph.data(),
                         N, K_out, K_rev_cap, plan, gpu_budget_b);
        }
        std::cout << "  Done [" << std::chrono::duration<double>(Clock::now() - t2).count()
                  << "s]\n";

        CUDA_CHECK(cudaHostUnregister(fwd_graph.data()));
        { auto _drop = std::move(fwd_graph); }

        // ---- Write output ----
        std::cout << "=== Step 3: Writing output ===\n";
        auto t3 = Clock::now();
        write_graph(out_path, output_graph.data(), N, K_out);
        std::cout << "  Wrote " << out_path << " ("
                  << output_graph.size() * sizeof(uint32_t) / 1e6 << " MB) ["
                  << std::chrono::duration<double>(Clock::now() - t3).count() << "s]\n";

        if (save_npy) {
            std::vector<int64_t> g64(output_graph.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < output_graph.size(); ++i)
                g64[i] = static_cast<int64_t>(output_graph[i]);
            auto p = std::filesystem::path(out_path);
            p.replace_extension(".npy");
            load::write_npy_int64_2d(p.string(), g64.data(), N, K_out);
            std::cout << "  Wrote " << p.string() << " (npy)\n";
        }

        double total = std::chrono::duration<double>(Clock::now() - t_start).count();
        std::cout << "\n=== Done ===\n";
        std::cout << "Method: " << method << "\n";
        std::cout << "Total: " << std::fixed << std::setprecision(3) << total << "s\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
