// prune_windowed_gpu.cu
// GPU version of prune_windowed's batched fast path (see prune_windowed.cpp):
// same windowed approximate CAGRA prune as prune_windowed_core.hpp, one warp
// per point. Used only when the whole window data set is host-resident
// (batched mode), the dataset is uint8 (distances are exact integers) and
// M_in <= 64; prune_windowed falls back to the CPU kernel otherwise.
//
// Semantics per point a (center bucket at processing position p):
//   window W(p) = p plus the positions of p's centroid-knn neighbour buckets
//   1. sort a's candidates by L2 (candidate outside W(p) → +inf, padding last).
//      Ties are ordered by original column (std::sort on the CPU path is
//      unstable, so only the order among exact ties can differ).
//   2. detour count for rank kAB = #{kAD < kAB : D = sorted[kAD] in W(p) and
//      sorted[kAB] appears in D's candidate row} (each (D, B) counted once).
//   3. output the first out_deg candidates by (detour count, rank) — exactly
//      prune_windowed_core's select_pruned_row threshold scan.
// cuVS 26.12 reference: detail/cagra/graph_core.cuh::kern_fused_prune (warp
// per node, detours counted against the node's sorted row in shared memory).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "prune_windowed_gpu.hpp"
#include <cstring>
#include <algorithm>

namespace {

#define PWG_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("prune_windowed_gpu: ") + cudaGetErrorString(e_)); } while (0)

constexpr int kMaxM = 64;
constexpr int kHash = 128;          // open-addressing id → rank table per warp
constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ int find_pos(const uint32_t* __restrict__ offsets, int n_buckets, uint32_t id)
{
    // last p with offsets[p] <= id (id < offsets[n_buckets] checked by caller)
    int lo = 0, hi = n_buckets;   // invariant: offsets[lo] <= id < offsets[hi]
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid] <= id) lo = mid; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ bool in_window(const int32_t* __restrict__ win, int wlen, int q)
{
    int lo = 0, hi = wlen;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (win[mid] < q) lo = mid + 1; else hi = mid;
    }
    return lo < wlen && win[lo] == q;
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32) prune_kernel(
    const int32_t* __restrict__ cand,      // (N, M)
    const uint8_t* __restrict__ vec,       // (N, D)
    const uint32_t* __restrict__ offsets,  // (n_buckets + 1)
    const int32_t* __restrict__ win,       // (n_buckets, W) sorted window positions, padded INT32_MAX
    const int32_t* __restrict__ win_len,   // (n_buckets)
    const int64_t* __restrict__ cbase,     // (n_buckets) row of position q's first point in cand/vec (-1: not loaded)
    int64_t row0, int64_t n_rows,          // points to prune: [row0, row0 + n_rows) (new ids)
    int M, int D, int n_buckets, int W, int out_deg,
    uint32_t* __restrict__ out,            // (n_rows, out_deg)
    int* __restrict__ err)
{
    __shared__ float   s_dist[kWarpsPerBlock][kMaxM];
    __shared__ int32_t s_id  [kWarpsPerBlock][kMaxM];
    __shared__ int32_t s_k   [kWarpsPerBlock][kMaxM];   // original column (tie-break)
    __shared__ int32_t s_hkey[kWarpsPerBlock][kHash];
    __shared__ int32_t s_hval[kWarpsPerBlock][kHash];
    __shared__ int32_t s_cnt [kWarpsPerBlock][kMaxM];
    __shared__ int32_t s_stamp[kWarpsPerBlock][kMaxM];

    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t a = row0 + static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + w;
    if (a >= row0 + n_rows) return;
    const uint32_t total = offsets[n_buckets];
    const int p = find_pos(offsets, n_buckets, static_cast<uint32_t>(a));
    const int32_t* wp = win + static_cast<int64_t>(p) * W;
    const int wl = win_len[p];
    // 点 id → cand/vec 里的行 (整份数据: cbase = offsets → 行 = id；分块: 只装了窗口并集)
    auto row_of = [&](int q, uint32_t id) { return cbase[q] + (static_cast<int64_t>(id) - offsets[q]); };
    const int64_t arow = row_of(p, static_cast<uint32_t>(a));
    if (cbase[p] < 0) { if (lane == 0) atomicExch(err, 2); return; }   // bounds check: center not loaded
    const uint8_t* va = vec + arow * D;

    // ---- 1. distances ----
    for (int k = lane; k < kMaxM; k += 32) {
        float dist = INFINITY;
        int32_t id = -1;
        bool valid = false;
        if (k < M) {
            id = cand[arow * M + k];
            if (id >= 0) {
                valid = true;
                const int q = (static_cast<uint32_t>(id) < total) ? find_pos(offsets, n_buckets, static_cast<uint32_t>(id)) : -1;
                if (q >= 0 && in_window(wp, wl, q) && cbase[q] >= 0) {
                    const uint8_t* vc = vec + row_of(q, static_cast<uint32_t>(id)) * D;
                    float acc = 0.f;
                    for (int d = 0; d < D; ++d) {
                        float df = static_cast<float>(va[d]) - static_cast<float>(vc[d]);
                        acc += df * df;
                    }
                    dist = acc;
                }
            }
        }
        // padding / k >= M sorts after every valid entry (even +inf ones)
        s_dist[w][k] = dist;
        s_id[w][k]   = valid ? id : -1;
        s_k[w][k]    = valid ? k : (kMaxM + k);
    }
    __syncwarp();

    // ---- bitonic sort of 64 entries by (padding, dist, k) ----
    auto less = [&](int i, int j) {
        const bool pi = s_id[w][i] < 0, pj = s_id[w][j] < 0;
        if (pi != pj) return pj;                 // valid < padding
        if (s_dist[w][i] != s_dist[w][j]) return s_dist[w][i] < s_dist[w][j];
        return s_k[w][i] < s_k[w][j];
    };
    for (int size = 2; size <= kMaxM; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = lane; t < kMaxM / 2; t += 32) {
                int i = 2 * stride * (t / stride) + (t % stride);
                int j = i + stride;
                bool up = ((i & size) == 0);
                if (less(j, i) == up) {
                    float td = s_dist[w][i]; s_dist[w][i] = s_dist[w][j]; s_dist[w][j] = td;
                    int32_t ti = s_id[w][i]; s_id[w][i] = s_id[w][j]; s_id[w][j] = ti;
                    int32_t tk = s_k[w][i];  s_k[w][i]  = s_k[w][j];  s_k[w][j]  = tk;
                }
            }
            __syncwarp();
        }
    }

    // ---- hash: id → rank (duplicates kept as separate entries) ----
    for (int h = lane; h < kHash; h += 32) s_hkey[w][h] = -1;
    for (int k = lane; k < kMaxM; k += 32) { s_cnt[w][k] = 0; s_stamp[w][k] = -1; }
    __syncwarp();
    for (int r = lane; r < M; r += 32) {
        int32_t id = s_id[w][r];
        if (id < 0) continue;
        uint32_t h = (static_cast<uint32_t>(id) * 2654435761u) & (kHash - 1);
        while (atomicCAS(&s_hkey[w][h], -1, id) != -1) h = (h + 1) & (kHash - 1);
        s_hval[w][h] = r;
    }
    __syncwarp();

    // ---- 2. detour counts ----
    for (int kAD = 0; kAD < M - 1; ++kAD) {
        const int32_t d_id = s_id[w][kAD];
        if (d_id < 0) break;   // padding is sorted last
        if (static_cast<uint32_t>(d_id) >= total) continue;
        const int qd = find_pos(offsets, n_buckets, static_cast<uint32_t>(d_id));
        if (!in_window(wp, wl, qd) || cbase[qd] < 0) continue;
        const int32_t* drow = cand + row_of(qd, static_cast<uint32_t>(d_id)) * M;
        for (int kDB = lane; kDB < M; kDB += 32) {
            const int32_t b = drow[kDB];
            if (b < 0) continue;
            uint32_t h = (static_cast<uint32_t>(b) * 2654435761u) & (kHash - 1);
            for (int probes = 0; probes < kHash && s_hkey[w][h] >= 0; ++probes, h = (h + 1) & (kHash - 1)) {
                if (s_hkey[w][h] != b) continue;
                const int kAB = s_hval[w][h];
                if (kAB > kAD && atomicExch(&s_stamp[w][kAB], kAD) != kAD)
                    atomicAdd(&s_cnt[w][kAB], 1);
            }
        }
        __syncwarp();
    }

    // ---- 3. select first out_deg by (count, rank) ----
    int n_valid = 0;
    for (int r = 0; r < M; ++r) n_valid += (s_id[w][r] >= 0);
    if (n_valid < out_deg) { if (lane == 0) atomicExch(err, 1); return; }
    for (int r = lane; r < M; r += 32) {
        if (s_id[w][r] < 0) continue;
        const int cr = s_cnt[w][r];
        int pos = 0;
        for (int q = 0; q < M; ++q) {
            if (s_id[w][q] < 0) continue;
            const int cq = s_cnt[w][q];
            pos += (cq < cr) || (cq == cr && q < r);
        }
        if (pos < out_deg) out[(a - row0) * out_deg + pos] = static_cast<uint32_t>(s_id[w][r]);
    }
}

}  // namespace

namespace prune_windowed_gpu {

bool available(int64_t N, int M, int D, int n_buckets, int W, int out_deg)
{
    if (M > kMaxM) return false;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    const size_t need = static_cast<size_t>(N) * (M * 4 + D + out_deg * 4)
                      + static_cast<size_t>(n_buckets) * (W + 2) * 4;
    return need < free_b * 0.8;
}

// 大块传输经两块 pinned 暂存区双缓冲 (cuVS 26.12 cagra_build.cuh::build_knn_graph
// 处理 host batch 的同一做法): 多线程把下一块拷进 pinned 区的同时，上一块在 DMA。
// 原来整份 pageable cudaMemcpy (SIFT10M: 3.84 GB 上传 1.37 s ≈ 2.8 GB/s)，结果再
// 拷回一块新分配的 1.28 GB pageable buffer (清零 0.9 s + D2H 0.8 s) 之后才写盘。
namespace {
struct Staging {
    static constexpr size_t kChunk = size_t(64) << 20;
    void* pin[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
    bool used[2] = {false, false};
    cudaStream_t st = nullptr;
    Staging() {
        PWG_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            PWG_CHECK(cudaMallocHost(&pin[b], kChunk));
            PWG_CHECK(cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming));
        }
    }
    ~Staging() {
        for (int b = 0; b < 2; ++b) { cudaFreeHost(pin[b]); cudaEventDestroy(ev[b]); }
        cudaStreamDestroy(st);
    }
    static void par_copy(void* dst, const void* src, size_t n) {
        const int64_t pieces = static_cast<int64_t>((n + (size_t(1) << 20) - 1) >> 20);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < pieces; ++i) {
            const size_t off = static_cast<size_t>(i) << 20;
            std::memcpy(static_cast<char*>(dst) + off, static_cast<const char*>(src) + off,
                        std::min(size_t(1) << 20, n - off));
        }
    }
    void h2d(void* d, const void* h, size_t bytes) {
        for (size_t off = 0, k = 0; off < bytes; off += kChunk, ++k) {
            const int b = static_cast<int>(k & 1);
            const size_t n = std::min(kChunk, bytes - off);
            if (used[b]) PWG_CHECK(cudaEventSynchronize(ev[b]));   // 这块 pinned 区的上一次 DMA 完成
            par_copy(pin[b], static_cast<const char*>(h) + off, n);
            PWG_CHECK(cudaMemcpyAsync(static_cast<char*>(d) + off, pin[b], n, cudaMemcpyHostToDevice, st));
            PWG_CHECK(cudaEventRecord(ev[b], st));
            used[b] = true;
        }
    }
    // d 上 (N, w) 的 uint32 行 → 按行块经 pinned 区交给 sink (DMA 跟 sink 重叠)
    void d2h_rows(const uint32_t* d, int64_t N, int w,
                  const std::function<void(const uint32_t*, int64_t, int64_t)>& sink) {
        PWG_CHECK(cudaStreamSynchronize(st));
        const int64_t rows_per = std::max<int64_t>(1, static_cast<int64_t>(kChunk / (static_cast<size_t>(w) * 4)));
        auto issue = [&](int64_t r0, int b) {
            const int64_t n = std::min(rows_per, N - r0);
            PWG_CHECK(cudaMemcpyAsync(pin[b], d + r0 * w, static_cast<size_t>(n) * w * 4, cudaMemcpyDeviceToHost, st));
            PWG_CHECK(cudaEventRecord(ev[b], st));
        };
        if (N > 0) issue(0, 0);
        for (int64_t r0 = 0, k = 0; r0 < N; r0 += rows_per, ++k) {
            const int b = static_cast<int>(k & 1);
            if (r0 + rows_per < N) issue(r0 + rows_per, b ^ 1);   // 另一块 pinned 区的 sink 已经返回
            PWG_CHECK(cudaEventSynchronize(ev[b]));
            sink(static_cast<const uint32_t*>(pin[b]), r0, std::min(rows_per, N - r0));
        }
    }
};
}  // namespace

void run(const int32_t* h_cand, const uint8_t* h_vec, int64_t N, int M, int D,
         const uint32_t* h_offsets, int n_buckets,
         const std::vector<int32_t>& h_win, const std::vector<int32_t>& h_win_len, int W,
         int out_deg, const std::function<void(const uint32_t*, int64_t, int64_t)>& sink)
{
    int32_t* d_cand = nullptr; uint8_t* d_vec = nullptr; uint32_t* d_off = nullptr;
    int32_t* d_win = nullptr; int32_t* d_wl = nullptr; uint32_t* d_out = nullptr; int* d_err = nullptr;
    int64_t* d_cbase = nullptr;
    auto cleanup = [&] {
        cudaFree(d_cand); cudaFree(d_vec); cudaFree(d_off); cudaFree(d_win);
        cudaFree(d_wl); cudaFree(d_out); cudaFree(d_err); cudaFree(d_cbase);
    };
    try {
        Staging stg;
        PWG_CHECK(cudaMalloc(&d_cand, static_cast<size_t>(N) * M * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_vec,  static_cast<size_t>(N) * D));
        PWG_CHECK(cudaMalloc(&d_off,  (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_win,  h_win.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_wl,   h_win_len.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_out,  static_cast<size_t>(N) * out_deg * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_err,  sizeof(int)));
        stg.h2d(d_cand, h_cand, static_cast<size_t>(N) * M * sizeof(int32_t));
        stg.h2d(d_vec,  h_vec,  static_cast<size_t>(N) * D);
        stg.h2d(d_off,  h_offsets, (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t));
        stg.h2d(d_win,  h_win.data(), h_win.size() * sizeof(int32_t));
        stg.h2d(d_wl,   h_win_len.data(), h_win_len.size() * sizeof(int32_t));
        std::vector<int64_t> cbase(h_offsets, h_offsets + n_buckets);   // 整份数据: 行 = id
        PWG_CHECK(cudaMalloc(&d_cbase, cbase.size() * sizeof(int64_t)));
        stg.h2d(d_cbase, cbase.data(), cbase.size() * sizeof(int64_t));
        PWG_CHECK(cudaMemsetAsync(d_err, 0, sizeof(int), stg.st));
        const int64_t blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
        prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(
            d_cand, d_vec, d_off, d_win, d_wl, d_cbase, 0, N, M, D, n_buckets, W, out_deg, d_out, d_err);
        PWG_CHECK(cudaGetLastError());
        int h_err = 0;
        PWG_CHECK(cudaMemcpyAsync(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost, stg.st));
        PWG_CHECK(cudaStreamSynchronize(stg.st));
        if (h_err)
            throw std::runtime_error("prune_windowed_gpu: fewer than output_degree valid candidates "
                                     "(check --output-degree <= M_in and padding)");
        stg.d2h_rows(d_out, N, out_deg, sink);
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

// 相邻两块的窗口并集大量重叠: 上一块已经在显存里的桶直接 device→device 搬到新块的布局里，
// 一个 block 一个 run (src 行, dst 行, 行数)，cand 行 M×int32、vec 行 D 字节。
struct RowRun { int64_t src, dst, n; };
__global__ void copy_row_runs_kernel(const RowRun* __restrict__ runs,
                                     const int32_t* __restrict__ cand_src, const uint8_t* __restrict__ vec_src,
                                     int32_t* __restrict__ cand_dst, uint8_t* __restrict__ vec_dst, int M, int D)
{
    const RowRun r = runs[blockIdx.x];
    const int64_t nc = r.n * M;
    const int32_t* cs = cand_src + r.src * M;
    int32_t* cd = cand_dst + r.dst * M;
    for (int64_t i = threadIdx.x; i < nc; i += blockDim.x) cd[i] = cs[i];
    const int64_t nv = r.n * D;
    const uint8_t* vs = vec_src + r.src * D;
    uint8_t* vd = vec_dst + r.dst * D;
    if ((D & 15) == 0) {
        const uint4* vs4 = reinterpret_cast<const uint4*>(vs);
        uint4* vd4 = reinterpret_cast<uint4*>(vd);
        for (int64_t i = threadIdx.x; i < nv / 16; i += blockDim.x) vd4[i] = vs4[i];
    } else {
        for (int64_t i = threadIdx.x; i < nv; i += blockDim.x) vd[i] = vs[i];
    }
}

// 窗口数据整份放不进 host 预算 / 显存时 (SIFT100M: 76.8 GB float 窗口 > 16 GB；GPU 整份
// 需 51 GB) 的分块版本: 按处理位置切块 [p0, p1)，每块只把这些位置窗口的并集
// (bucket 在新 id 空间连续 → 合并成连续行段) 读进来、压紧上传，kernel 用 cbase 把
// 点 id 映射到压紧后的行。每个点的窗口 / 剪枝语义跟整份版本完全一样；显存和 host
// 内存都只跟块大小 (budget_bytes) 有关，不随 N 增长。
void run_chunked(int64_t N, int M, int D, const uint32_t* h_offsets, int n_buckets,
                 const std::vector<int32_t>& h_win, const std::vector<int32_t>& h_win_len, int W,
                 int out_deg, size_t budget_bytes,
                 const std::function<void(int64_t lo, int64_t n, int32_t* cand, uint8_t* vec)>& load_rows,
                 const std::function<void(const uint32_t*, int64_t, int64_t)>& sink,
                 ChunkStats* stats)
{
    const size_t row_bytes = static_cast<size_t>(M) * sizeof(int32_t) + static_cast<size_t>(D);
    int64_t max_bucket = 0;
    for (int q = 0; q < n_buckets; ++q) max_bucket = std::max<int64_t>(max_bucket, h_offsets[q + 1] - h_offsets[q]);
    // 一个位置的窗口最多 W 个桶，块至少要装得下一个窗口
    const int64_t cap_rows = std::max<int64_t>(static_cast<int64_t>(budget_bytes / row_bytes),
                                               static_cast<int64_t>(W) * max_bucket);
    // 两套行缓冲轮换 (上一块 / 这一块)，重叠的桶 device→device 搬，不再从磁盘重读、不再 H2D
    // (参考 TAGORE largeIndex: graph_pool 缓冲里的子图被下一个 cluster 直接复用)
    int32_t* d_candB[2] = {nullptr, nullptr}; uint8_t* d_vecB[2] = {nullptr, nullptr};
    RowRun* d_runs = nullptr;
    uint32_t* d_off = nullptr;
    int32_t* d_win = nullptr; int32_t* d_wl = nullptr; uint32_t* d_out = nullptr; int* d_err = nullptr;
    int64_t* d_cbase = nullptr;
    auto cleanup = [&] {
        for (int b = 0; b < 2; ++b) { cudaFree(d_candB[b]); cudaFree(d_vecB[b]); }
        cudaFree(d_runs); cudaFree(d_off); cudaFree(d_win);
        cudaFree(d_wl); cudaFree(d_out); cudaFree(d_err); cudaFree(d_cbase);
    };
    try {
        Staging stg;
        for (int b = 0; b < 2; ++b) {
            PWG_CHECK(cudaMalloc(&d_candB[b], static_cast<size_t>(cap_rows) * M * sizeof(int32_t)));
            PWG_CHECK(cudaMalloc(&d_vecB[b],  static_cast<size_t>(cap_rows) * D));
        }
        PWG_CHECK(cudaMalloc(&d_runs, static_cast<size_t>(n_buckets) * sizeof(RowRun)));
        PWG_CHECK(cudaMalloc(&d_out,  static_cast<size_t>(cap_rows) * out_deg * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_off,  (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_win,  h_win.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_wl,   h_win_len.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_cbase, static_cast<size_t>(n_buckets) * sizeof(int64_t)));
        PWG_CHECK(cudaMalloc(&d_err,  sizeof(int)));
        stg.h2d(d_off, h_offsets, (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t));
        stg.h2d(d_win, h_win.data(), h_win.size() * sizeof(int32_t));
        stg.h2d(d_wl,  h_win_len.data(), h_win_len.size() * sizeof(int32_t));
        PWG_CHECK(cudaMemsetAsync(d_err, 0, sizeof(int), stg.st));

        // 新读的行直接 pread 进 pinned 缓冲，一次 DMA 上传 (原来 pageable → 64 MB pinned 双缓冲中转，
        // H2D 只有 ~5 GB/s: SIFT10M 强制分块时 h2d 占 prune 计算的一大半)；大小跟原来的 host 缓冲一样
        struct PinnedBuf {
            void* p = nullptr;
            explicit PinnedBuf(size_t n) { PWG_CHECK(cudaMallocHost(&p, std::max<size_t>(n, 1))); }
            ~PinnedBuf() { cudaFreeHost(p); }
        };
        PinnedBuf pin_cand(static_cast<size_t>(cap_rows) * M * sizeof(int32_t));
        PinnedBuf pin_vec(static_cast<size_t>(cap_rows) * D);
        int32_t* h_cand = static_cast<int32_t*>(pin_cand.p);
        uint8_t* h_vec = static_cast<uint8_t*>(pin_vec.p);
        std::vector<int64_t> cbase(static_cast<size_t>(n_buckets), -1);
        std::vector<int64_t> prev_cbase(static_cast<size_t>(n_buckets), -1);   // 上一块里桶的行 (在 d_*B[cur ^ 1])
        std::vector<int32_t> stamp(static_cast<size_t>(n_buckets), -1);
        std::vector<int32_t> U, prevU;
        std::vector<RowRun> h_runs;
        int cur = 0;
        int32_t chunk_id = 0;
        for (int p0 = 0; p0 < n_buckets; ++chunk_id) {
            // ---- 并集 U(p0..p1): 加位置直到行数到上限 ----
            for (int q : prevU) prev_cbase[static_cast<size_t>(q)] = -1;
            for (int q : U) prev_cbase[static_cast<size_t>(q)] = cbase[static_cast<size_t>(q)];   // 上一块的布局
            prevU.swap(U);
            for (int q : prevU) cbase[static_cast<size_t>(q)] = -1;
            U.clear();
            int64_t rows = 0;
            int p1 = p0;
            while (p1 < n_buckets) {
                const int32_t* wp = h_win.data() + static_cast<size_t>(p1) * W;
                const int wl = h_win_len[static_cast<size_t>(p1)];
                int64_t add = 0;
                for (int i = 0; i < wl; ++i)
                    if (stamp[static_cast<size_t>(wp[i])] != chunk_id) add += h_offsets[wp[i] + 1] - h_offsets[wp[i]];
                if (p1 > p0 && rows + add > cap_rows) break;
                for (int i = 0; i < wl; ++i) {
                    const int q = wp[i];
                    if (q < 0 || q >= n_buckets) throw std::runtime_error("prune_windowed_gpu: window position out of range");
                    if (stamp[static_cast<size_t>(q)] != chunk_id) { stamp[static_cast<size_t>(q)] = chunk_id; U.push_back(q); }
                }
                rows += add;
                ++p1;
            }
            // ---- 布局: 上一块已在显存里的桶排前面 (按旧行号，旧布局里连续的仍连续 → run 少)，
            //      其余桶按 id 排在后面 (相邻 id 合并成连续段并行读 + 一次 H2D)。kernel 只经 cbase 找行，
            //      布局顺序不影响结果。
            std::vector<int32_t> reuse, fresh;
            for (int q : U) (prev_cbase[static_cast<size_t>(q)] >= 0 ? reuse : fresh).push_back(q);
            std::sort(reuse.begin(), reuse.end(), [&](int a, int b) { return prev_cbase[static_cast<size_t>(a)] < prev_cbase[static_cast<size_t>(b)]; });
            std::sort(fresh.begin(), fresh.end());
            int64_t dst = 0;
            h_runs.clear();
            for (int q : reuse) {
                const int64_t n = static_cast<int64_t>(h_offsets[q + 1]) - h_offsets[q];
                const int64_t src = prev_cbase[static_cast<size_t>(q)];
                cbase[static_cast<size_t>(q)] = dst;
                if (!h_runs.empty() && h_runs.back().src + h_runs.back().n == src && h_runs.back().dst + h_runs.back().n == dst)
                    h_runs.back().n += n;
                else h_runs.push_back({src, dst, n});
                dst += n;
            }
            const int64_t reuse_rows = dst;
            struct Run { int64_t lo, n, dst; };
            std::vector<Run> runs;
            for (int q : fresh) {
                const int64_t lo = h_offsets[q], n = static_cast<int64_t>(h_offsets[q + 1]) - lo;
                cbase[static_cast<size_t>(q)] = dst;
                if (!runs.empty() && runs.back().lo + runs.back().n == lo) runs.back().n += n;
                else runs.push_back({lo, n, dst});
                dst += n;
            }
            if (dst > cap_rows) throw std::runtime_error("prune_windowed_gpu: chunk rows exceed capacity");
            const int nb = cur ^ 1;   // 这一块写进另一套缓冲 (cur 里是上一块)
            if (!h_runs.empty()) {
                stg.h2d(d_runs, h_runs.data(), h_runs.size() * sizeof(RowRun));
                copy_row_runs_kernel<<<static_cast<unsigned>(h_runs.size()), 256, 0, stg.st>>>(
                    d_runs, d_candB[cur], d_vecB[cur], d_candB[nb], d_vecB[nb], M, D);
                PWG_CHECK(cudaGetLastError());
            }
            #pragma omp parallel for schedule(dynamic, 1)
            for (int64_t i = 0; i < static_cast<int64_t>(runs.size()); ++i) {
                const Run& r = runs[static_cast<size_t>(i)];
                const int64_t hd = r.dst - reuse_rows;   // host 缓冲里只放新读的行
                load_rows(r.lo, r.n, h_cand + static_cast<size_t>(hd) * M, h_vec + static_cast<size_t>(hd) * D);
            }
            const int64_t fresh_rows = dst - reuse_rows;
            // pinned → 直接 DMA (同一 stream: 排在上面的 device 内搬运之后、kernel 之前；
            // 下一块重写 h_cand/h_vec 前，本块的 d2h_rows 已经同步过这个 stream)
            PWG_CHECK(cudaMemcpyAsync(d_candB[nb] + static_cast<size_t>(reuse_rows) * M, h_cand,
                                      static_cast<size_t>(fresh_rows) * M * sizeof(int32_t), cudaMemcpyHostToDevice, stg.st));
            PWG_CHECK(cudaMemcpyAsync(d_vecB[nb] + static_cast<size_t>(reuse_rows) * D, h_vec,
                                      static_cast<size_t>(fresh_rows) * D, cudaMemcpyHostToDevice, stg.st));
            stg.h2d(d_cbase, cbase.data(), cbase.size() * sizeof(int64_t));
            int32_t* d_cand = d_candB[nb]; uint8_t* d_vec = d_vecB[nb];
            cur = nb;
            const int64_t row0 = h_offsets[p0], n_rows = static_cast<int64_t>(h_offsets[p1]) - row0;
            const int64_t blocks = (n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
            if (blocks > 0)
                prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(
                    d_cand, d_vec, d_off, d_win, d_wl, d_cbase, row0, n_rows, M, D, n_buckets, W, out_deg, d_out, d_err);
            PWG_CHECK(cudaGetLastError());
            int h_err = 0;
            PWG_CHECK(cudaMemcpyAsync(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost, stg.st));
            PWG_CHECK(cudaStreamSynchronize(stg.st));
            if (h_err)
                throw std::runtime_error(h_err == 2 ? "prune_windowed_gpu: center bucket not loaded in chunk"
                                                    : "prune_windowed_gpu: fewer than output_degree valid candidates "
                                                      "(check --output-degree <= M_in and padding)");
            stg.d2h_rows(d_out, n_rows, out_deg, [&](const uint32_t* r, int64_t r0, int64_t n) { sink(r, row0 + r0, n); });
            if (stats) { stats->chunks++; stats->loaded_rows += fresh_rows; }
            p0 = p1;
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}


}  // namespace prune_windowed_gpu
