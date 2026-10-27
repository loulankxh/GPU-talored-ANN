// prune_windowed_gpu.cu
// GPU version of prune_windowed's batched fast path (see prune_windowed.cpp):
// same windowed approximate CAGRA prune as prune_windowed_core.hpp, one warp
// per point. Element type T = uint8 / int8 (distances are exact integers) or
// float (fp32 distances, same rule; P2.2 2026-10-24), M_in <= 64;
// prune_windowed falls back to the CPU kernel otherwise.
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

#include <climits>
#include <numeric>
#include <functional>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

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

// ---- A7 (user-approved for 10M, 2026-10-25): 不受窗口限制的 local join (batched: 整份数据在显存, 行 = id) ----
// 点 a 的候选 = 原 M 个 ∪ 它前 S 个邻居各自的前 S 个邻居 (2 跳)，全部算精确距离，取最近 M 个 (升序) 写进 own。
struct A7Counters { unsigned long long points, cand, inserted; };
constexpr int kA7MaxS = 16;
template <typename T>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) a7_refine_kernel(
    const int32_t* __restrict__ cand, const T* __restrict__ vec, int64_t N, int M, int D, int S,
    int32_t* __restrict__ own, A7Counters* __restrict__ ctr)
{
    constexpr int kC = kMaxM + kA7MaxS * kA7MaxS;
    __shared__ float   s_d[kWarpsPerBlock][kC];
    __shared__ int32_t s_id[kWarpsPerBlock][kC];
    __shared__ int s_n[kWarpsPerBlock];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int64_t a = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + w;
    if (a >= N) return;
    const T* va = vec + a * D;
    auto dist_to = [&](int64_t r) {   // 跟 prune_kernel 同样的逐维顺序
        const T* vc = vec + r * D;
        float acc = 0.f;
        for (int d = 0; d < D; ++d) { float df = static_cast<float>(va[d]) - static_cast<float>(vc[d]); acc += df * df; }
        return acc;
    };
    if (lane == 0) s_n[w] = 0;
    __syncwarp();
    for (int k = lane; k < M; k += 32) {   // 原候选
        const int32_t id = cand[a * M + k];
        s_id[w][k] = id;
        s_d[w][k] = (id >= 0 && id < N) ? dist_to(id) : INFINITY;
    }
    __syncwarp();
    const int nc = S * S;
    for (int c = lane; c < nc; c += 32) {   // 2 跳候选
        const int32_t u = cand[a * M + c / S];
        if (u < 0 || u >= N) continue;   // bounds check
        const int32_t v = cand[static_cast<int64_t>(u) * M + (c % S)];
        if (v < 0 || v >= N || v == static_cast<int32_t>(a)) continue;
        bool dup = false;
        for (int k = 0; k < M && !dup; ++k) dup = (s_id[w][k] == v);
        if (dup) continue;
        const int slot = M + atomicAdd(&s_n[w], 1);
        if (slot < kC) { s_id[w][slot] = v; s_d[w][slot] = dist_to(v); }
    }
    __syncwarp();
    const int tot = min(M + s_n[w], kC);
    for (int i = M + lane; i < tot; i += 32) {   // 候选之间去重
        const int32_t v = s_id[w][i];
        for (int o = M; o < i; ++o) if (s_id[w][o] == v) { s_d[w][i] = INFINITY; s_id[w][i] = -1; break; }
    }
    __syncwarp();
    int32_t* orow = own + a * M;
    for (int k = lane; k < M; k += 32) orow[k] = -1;
    __syncwarp();
    int ins = 0;
    for (int e = lane; e < tot; e += 32) {
        const float d = s_d[w][e];
        if (!(d < INFINITY) || s_id[w][e] < 0) continue;
        int rank = 0;
        for (int o = 0; o < tot; ++o) rank += (s_d[w][o] < d) || (s_d[w][o] == d && o < e);
        if (rank < M) { orow[rank] = s_id[w][e]; if (e >= M) ++ins; }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ins += __shfl_xor_sync(0xffffffffu, ins, off);
    if (lane == 0) {
        atomicAdd(&ctr->points, 1ull); atomicAdd(&ctr->cand, static_cast<unsigned long long>(tot));
        atomicAdd(&ctr->inserted, static_cast<unsigned long long>(ins));
    }
}

// B12: 分块 prune 的 GPU 桶池 —— 行按 kSlotRows 行一槽存放 (桶占若干个不必连续的槽，没有碎片)，
// 跨块按 Belady (下一次被用到的块最远的先淘汰) 保留，代替只从上一块复用 (SIFT100M 重读 8.4 × N 行 → 模拟 4.9 × N)
constexpr int kSlotRows = 64;
struct SlotJob { int64_t host_row; int32_t n; int32_t tab0; };   // 新装的桶: pinned 里的起始行、行数、槽表起点
// 新读的行从 pinned (UVA 直接可读) 拷进各自的槽: 一个 block 一个桶
__global__ void scatter_slots_kernel(const SlotJob* __restrict__ jobs, const int32_t* __restrict__ slot_tab,
                                     const int32_t* __restrict__ h_cand, const uint8_t* __restrict__ h_vec,
                                     int32_t* __restrict__ cand, uint8_t* __restrict__ vec, int M, int vec_row_bytes)
{
    const SlotJob j = jobs[blockIdx.x];
    // 行宽是 16 B 的倍数时按 int4 读 pinned (零拷贝读 PCIe: 逐字节读只有一小部分带宽)
    if ((M * 4) % 16 == 0 && vec_row_bytes % 16 == 0) {
        const int cw = M * 4 / 16, vw = vec_row_bytes / 16;
        const int4* hc = reinterpret_cast<const int4*>(h_cand); const int4* hv = reinterpret_cast<const int4*>(h_vec);
        int4* dc = reinterpret_cast<int4*>(cand); int4* dv = reinterpret_cast<int4*>(vec);
        for (int64_t i = threadIdx.x; i < static_cast<int64_t>(j.n) * cw; i += blockDim.x) {
            const int64_t r = i / cw, c = i - r * cw;
            const int64_t dst = static_cast<int64_t>(slot_tab[j.tab0 + r / kSlotRows]) * kSlotRows + r % kSlotRows;
            dc[dst * cw + c] = hc[(j.host_row + r) * cw + c];
        }
        for (int64_t i = threadIdx.x; i < static_cast<int64_t>(j.n) * vw; i += blockDim.x) {
            const int64_t r = i / vw, c = i - r * vw;
            const int64_t dst = static_cast<int64_t>(slot_tab[j.tab0 + r / kSlotRows]) * kSlotRows + r % kSlotRows;
            dv[dst * vw + c] = hv[(j.host_row + r) * vw + c];
        }
        return;
    }
    for (int64_t i = threadIdx.x; i < static_cast<int64_t>(j.n) * M; i += blockDim.x) {
        const int64_t r = i / M, c = i - r * M;
        const int64_t dst = static_cast<int64_t>(slot_tab[j.tab0 + r / kSlotRows]) * kSlotRows + r % kSlotRows;
        cand[dst * M + c] = h_cand[(j.host_row + r) * M + c];
    }
    for (int64_t i = threadIdx.x; i < static_cast<int64_t>(j.n) * vec_row_bytes; i += blockDim.x) {
        const int64_t r = i / vec_row_bytes, c = i - r * vec_row_bytes;
        const int64_t dst = static_cast<int64_t>(slot_tab[j.tab0 + r / kSlotRows]) * kSlotRows + r % kSlotRows;
        vec[dst * vec_row_bytes + c] = h_vec[(j.host_row + r) * vec_row_bytes + c];
    }
}

// 保留下来的边里，候选不在窗口里 (距离未知 = +inf、没做 detour 检查) 的条数 (A4 统计)
__device__ unsigned long long g_unverified_kept = 0;

template <typename T>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) prune_kernel(
    const int32_t* __restrict__ cand,      // (N, M)
    const T* __restrict__ vec,             // (N, D)
    const uint32_t* __restrict__ offsets,  // (n_buckets + 1)
    const int32_t* __restrict__ win,       // (n_buckets, W) sorted window positions, padded INT32_MAX
    const int32_t* __restrict__ win_len,   // (n_buckets)
    const int64_t* __restrict__ cbase,     // (n_buckets) row of position q's first point in cand/vec (-1: not loaded)
    int64_t row0, int64_t n_rows,          // points to prune: [row0, row0 + n_rows) (new ids)
    int M, int D, int n_buckets, int W, int out_deg,
    uint32_t* __restrict__ out,            // (n_rows, out_deg)
    int* __restrict__ err,
    bool unwin = false,                    // A7: 整份数据在显存时不按窗口限制 (所有候选都有真实距离 / detour 检查)
    const int32_t* __restrict__ own = nullptr,   // A7: 中心点自己的精修候选行 (按 a - row0)
    const float* __restrict__ cdist = nullptr,   // B13: 中心点候选的 Step 6 距离 (按 a - row0，行宽 M)
    const int32_t* __restrict__ slot_tab = nullptr)   // B12: 桶 q 的第 j 个 kSlotRows 行槽 = slot_tab[cbase[q] + j]
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
    auto row_of = [&](int q, uint32_t id) -> int64_t {
        const int64_t r = static_cast<int64_t>(id) - offsets[q];
        if (!slot_tab) return cbase[q] + r;
        return static_cast<int64_t>(slot_tab[cbase[q] + r / kSlotRows]) * kSlotRows + r % kSlotRows;
    };
    const int64_t arow = row_of(p, static_cast<uint32_t>(a));
    if (cbase[p] < 0) { if (lane == 0) atomicExch(err, 2); return; }   // bounds check: center not loaded
    const T* va = vec + arow * D;

    // ---- 1. distances ----
    for (int k = lane; k < kMaxM; k += 32) {
        float dist = INFINITY;
        int32_t id = -1;
        bool valid = false;
        if (k < M) {
            id = own ? own[(a - row0) * M + k] : cand[arow * M + k];
            if (id >= 0 && cdist) {   // B13: Step 6 算过的距离 (窗口外的候选也有真实距离)
                valid = true;
                dist = cdist[(a - row0) * M + k];
            } else if (id >= 0) {
                valid = true;
                const int q = (static_cast<uint32_t>(id) < total) ? find_pos(offsets, n_buckets, static_cast<uint32_t>(id)) : -1;
                if (q >= 0 && (unwin || in_window(wp, wl, q)) && cbase[q] >= 0) {
                    const T* vc = vec + row_of(q, static_cast<uint32_t>(id)) * D;
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
        if (!(unwin || in_window(wp, wl, qd)) || cbase[qd] < 0) continue;
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
        if (pos < out_deg) {
            out[(a - row0) * out_deg + pos] = static_cast<uint32_t>(s_id[w][r]);
            if (!(s_dist[w][r] < INFINITY)) atomicAdd(&g_unverified_kept, 1ull);
        }
    }
}

}  // namespace

namespace prune_windowed_gpu {

static std::function<void(int64_t, int64_t, float*)> g_center_dists;
void set_center_dists(std::function<void(int64_t lo, int64_t n, float* out)> f) { g_center_dists = std::move(f); }

static void reset_unverified() { unsigned long long z = 0; cudaMemcpyToSymbol(g_unverified_kept, &z, sizeof(z)); }
static void report_unverified(int64_t n_points, int out_deg)
{
    unsigned long long h = 0;
    cudaMemcpyFromSymbol(&h, g_unverified_kept, sizeof(h));
    std::cout << "  [Prune] unverified kept edges (candidate outside the window): " << h << " = "
              << static_cast<double>(h) / std::max<int64_t>(n_points, 1) << " per row of " << out_deg << "\n";
}

bool available(int64_t N, int M, int D, int n_buckets, int W, int out_deg, int elem_bytes)
{
    if (M > kMaxM) return false;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return false;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    const bool a7 = std::getenv("BUCKET2_A7_SAMPLE") && std::atoi(std::getenv("BUCKET2_A7_SAMPLE")) > 0;
    const size_t need = static_cast<size_t>(N) * (M * 4 + static_cast<size_t>(D) * elem_bytes + out_deg * 4
                                                  + (a7 ? M * 4 : 0))   // A7 精修行
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

template <typename T>
void run(const int32_t* h_cand, const T* h_vec, int64_t N, int M, int D,
         const uint32_t* h_offsets, int n_buckets,
         const std::vector<int32_t>& h_win, const std::vector<int32_t>& h_win_len, int W,
         int out_deg, const std::function<void(const uint32_t*, int64_t, int64_t)>& sink)
{
    int32_t* d_cand = nullptr; T* d_vec = nullptr; uint32_t* d_off = nullptr;
    int32_t* d_win = nullptr; int32_t* d_wl = nullptr; uint32_t* d_out = nullptr; int* d_err = nullptr;
    int64_t* d_cbase = nullptr;
    auto cleanup = [&] {
        cudaFree(d_cand); cudaFree(d_vec); cudaFree(d_off); cudaFree(d_win);
        cudaFree(d_wl); cudaFree(d_out); cudaFree(d_err); cudaFree(d_cbase);
    };
    try {
        Staging stg;
        PWG_CHECK(cudaMalloc(&d_cand, static_cast<size_t>(N) * M * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_vec,  static_cast<size_t>(N) * D * sizeof(T)));
        PWG_CHECK(cudaMalloc(&d_off,  (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_win,  h_win.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_wl,   h_win_len.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_out,  static_cast<size_t>(N) * out_deg * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_err,  sizeof(int)));
        stg.h2d(d_cand, h_cand, static_cast<size_t>(N) * M * sizeof(int32_t));
        stg.h2d(d_vec,  h_vec,  static_cast<size_t>(N) * D * sizeof(T));
        stg.h2d(d_off,  h_offsets, (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t));
        stg.h2d(d_win,  h_win.data(), h_win.size() * sizeof(int32_t));
        stg.h2d(d_wl,   h_win_len.data(), h_win_len.size() * sizeof(int32_t));
        std::vector<int64_t> cbase(h_offsets, h_offsets + n_buckets);   // 整份数据: 行 = id
        PWG_CHECK(cudaMalloc(&d_cbase, cbase.size() * sizeof(int64_t)));
        stg.h2d(d_cbase, cbase.data(), cbase.size() * sizeof(int64_t));
        PWG_CHECK(cudaMemsetAsync(d_err, 0, sizeof(int), stg.st));
        reset_unverified();
        const int64_t blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
        // A7: BUCKET2_A7_SAMPLE = S (≤ 16) → 先做不受窗口限制的 local join；BUCKET2_A7_UNWIN=1 或 S > 0 → prune 不按窗口
        const char* a7e = std::getenv("BUCKET2_A7_SAMPLE");
        const int a7_S = a7e ? std::min(std::atoi(a7e), std::min(kA7MaxS, M)) : 0;
        const bool unwin = a7_S > 0 || (std::getenv("BUCKET2_A7_UNWIN") && std::getenv("BUCKET2_A7_UNWIN")[0] == '1');
        int32_t* d_own = nullptr;
        if (a7_S > 0) {
            A7Counters* d_a7 = nullptr;
            PWG_CHECK(cudaMalloc(&d_own, static_cast<size_t>(N) * M * sizeof(int32_t)));
            PWG_CHECK(cudaMalloc(&d_a7, sizeof(A7Counters)));
            PWG_CHECK(cudaMemsetAsync(d_a7, 0, sizeof(A7Counters), stg.st));
            cudaEvent_t e0, e1; PWG_CHECK(cudaEventCreate(&e0)); PWG_CHECK(cudaEventCreate(&e1));
            PWG_CHECK(cudaEventRecord(e0, stg.st));
            a7_refine_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(d_cand, d_vec, N, M, D, a7_S, d_own, d_a7);
            PWG_CHECK(cudaGetLastError());
            PWG_CHECK(cudaEventRecord(e1, stg.st));
            PWG_CHECK(cudaEventSynchronize(e1));
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            A7Counters h{};
            PWG_CHECK(cudaMemcpy(&h, d_a7, sizeof(A7Counters), cudaMemcpyDeviceToHost));
            cudaFree(d_a7);
            const double np_ = h.points ? static_cast<double>(h.points) : 1.0;
            std::cout << "A1_stats: path=large variant=a7_unwindowed_pull2hop pass=1 sample=" << a7_S << " points=" << h.points
                      << " cand_avg=" << h.cand / np_ << " inserted_avg=" << h.inserted / np_ << " seconds=" << ms / 1e3 << "\n";
        }
        if (unwin) std::cout << "  A7: prune without window restriction (all candidates scored)\n";
        prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(
            d_cand, d_vec, d_off, d_win, d_wl, d_cbase, 0, N, M, D, n_buckets, W, out_deg, d_out, d_err, unwin, d_own);
        PWG_CHECK(cudaGetLastError());
        int h_err = 0;
        PWG_CHECK(cudaMemcpyAsync(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost, stg.st));
        PWG_CHECK(cudaStreamSynchronize(stg.st));
        if (h_err)
            throw std::runtime_error("prune_windowed_gpu: fewer than output_degree valid candidates "
                                     "(check --output-degree <= M_in and padding)");
        stg.d2h_rows(d_out, N, out_deg, sink);
        report_unverified(N, out_deg);
        cudaFree(d_own);
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

// 相邻两块的窗口并集大量重叠: 上一块已经在显存里的桶直接 device→device 搬到新块的布局里，
// 一个 block 一个 run (src 行, dst 行, 行数)，cand 行 M×int32、vec 行 vec_row_bytes 字节。
struct RowRun { int64_t src, dst, n; };
__global__ void copy_row_runs_kernel(const RowRun* __restrict__ runs,
                                     const int32_t* __restrict__ cand_src, const uint8_t* __restrict__ vec_src,
                                     int32_t* __restrict__ cand_dst, uint8_t* __restrict__ vec_dst, int M, int vec_row_bytes)
{
    const RowRun r = runs[blockIdx.x];
    const int64_t nc = r.n * M;
    const int32_t* cs = cand_src + r.src * M;
    int32_t* cd = cand_dst + r.dst * M;
    for (int64_t i = threadIdx.x; i < nc; i += blockDim.x) cd[i] = cs[i];
    const int64_t nv = r.n * vec_row_bytes;
    const uint8_t* vs = vec_src + r.src * vec_row_bytes;
    uint8_t* vd = vec_dst + r.dst * vec_row_bytes;
    if ((vec_row_bytes & 15) == 0) {
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
template <typename T>
void run_chunked(int64_t N, int M, int D, const uint32_t* h_offsets, int n_buckets,
                 const std::vector<int32_t>& h_win, const std::vector<int32_t>& h_win_len, int W,
                 int out_deg, size_t budget_bytes,
                 const std::function<void(int64_t lo, int64_t n, int32_t* cand, T* vec)>& load_rows,
                 const std::function<void(const uint32_t*, int64_t, int64_t)>& sink,
                 ChunkStats* stats)
{
    const size_t vec_row = static_cast<size_t>(D) * sizeof(T);
    const size_t row_bytes = static_cast<size_t>(M) * sizeof(int32_t) + vec_row;
    int64_t max_bucket = 0;
    for (int q = 0; q < n_buckets; ++q) max_bucket = std::max<int64_t>(max_bucket, h_offsets[q + 1] - h_offsets[q]);
    // 一个位置的窗口最多 W 个桶，块至少要装得下一个窗口
    const int64_t cap_rows = std::max<int64_t>(static_cast<int64_t>(budget_bytes / row_bytes),
                                               static_cast<int64_t>(W) * max_bucket);
    // 两套行缓冲轮换 (上一块 / 这一块)，重叠的桶 device→device 搬，不再从磁盘重读、不再 H2D
    // (参考 TAGORE largeIndex: graph_pool 缓冲里的子图被下一个 cluster 直接复用)
    int32_t* d_candB[2] = {nullptr, nullptr}; T* d_vecB[2] = {nullptr, nullptr};
    RowRun* d_runs = nullptr;
    uint32_t* d_off = nullptr;
    int32_t* d_win = nullptr; int32_t* d_wl = nullptr; uint32_t* d_out = nullptr; int* d_err = nullptr;
    int64_t* d_cbase = nullptr;
    auto cleanup = [&] {
        for (int b = 0; b < 2; ++b) { cudaFree(d_candB[b]); cudaFree(d_vecB[b]); }
        cudaFree(d_runs); cudaFree(d_off); cudaFree(d_win);
        cudaFree(d_wl); cudaFree(d_out); cudaFree(d_err); cudaFree(d_cbase);
    };
    // B12: GPU 桶池 (默认开; BUCKET2_PRUNE_POOL=0 回到两套缓冲 + 只从上一块复用)
    const bool pool_mode = !(std::getenv("BUCKET2_PRUNE_POOL") && std::getenv("BUCKET2_PRUNE_POOL")[0] == '0');
    // 池按空闲显存定 (扣掉输出 / 窗口 / 槽表等，留 10% + 512 MB 余量)，至少两套缓冲那么大
    int64_t pool_slots = 2 * cap_rows / kSlotRows;
    if (pool_mode) {
        size_t free_b = 0, total_b = 0;
        PWG_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const double other = 2.0 * static_cast<double>(cap_rows) * out_deg * sizeof(uint32_t)   // 两套输出 (流水线)
                           + static_cast<double>(h_win.size() + h_win_len.size()) * sizeof(int32_t)
                           + static_cast<double>(n_buckets) * (sizeof(int64_t) + sizeof(SlotJob) + 2 * sizeof(int32_t) + sizeof(uint32_t))
                           + static_cast<double>(cap_rows) / kSlotRows * sizeof(int32_t)
                           + (g_center_dists ? static_cast<double>(cap_rows) * M * sizeof(float) : 0.0)
                           + (512.0 * (1 << 20));
        const double room = 0.9 * static_cast<double>(free_b) - other;
        const int64_t fit = room > 0 ? static_cast<int64_t>(room / (static_cast<double>(row_bytes) * kSlotRows)) : 0;
        pool_slots = std::max(pool_slots, fit);
    }
    int32_t* d_candP = nullptr; T* d_vecP = nullptr; int32_t* d_tab = nullptr; SlotJob* d_jobs = nullptr;
    uint32_t* d_out2 = nullptr; cudaEvent_t ev_done[2] = {nullptr, nullptr}, ev_scat[2] = {nullptr, nullptr};
    auto cleanup_pool = [&] {
        cudaFree(d_candP); cudaFree(d_vecP); cudaFree(d_tab); cudaFree(d_jobs); cudaFree(d_out2);
        for (auto& e : ev_done) if (e) cudaEventDestroy(e);
        for (auto& e : ev_scat) if (e) cudaEventDestroy(e);
    };
    try {
        Staging stg;
        if (pool_mode) {
            PWG_CHECK(cudaMalloc(&d_candP, static_cast<size_t>(pool_slots) * kSlotRows * M * sizeof(int32_t)));
            PWG_CHECK(cudaMalloc(&d_vecP,  static_cast<size_t>(pool_slots) * kSlotRows * vec_row));
            PWG_CHECK(cudaMalloc(&d_tab,   (static_cast<size_t>(cap_rows) / kSlotRows + n_buckets) * sizeof(int32_t)));
            PWG_CHECK(cudaMalloc(&d_jobs,  static_cast<size_t>(n_buckets) * sizeof(SlotJob)));
            PWG_CHECK(cudaMalloc(&d_out2,  static_cast<size_t>(cap_rows) * out_deg * sizeof(uint32_t)));
            for (auto& e : ev_done) PWG_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            for (auto& e : ev_scat) PWG_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        } else
        for (int b = 0; b < 2; ++b) {
            PWG_CHECK(cudaMalloc(&d_candB[b], static_cast<size_t>(cap_rows) * M * sizeof(int32_t)));
            PWG_CHECK(cudaMalloc(&d_vecB[b],  static_cast<size_t>(cap_rows) * vec_row));
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
        PinnedBuf pin_vec(static_cast<size_t>(cap_rows) * vec_row);
        int32_t* h_cand = static_cast<int32_t*>(pin_cand.p);
        T* h_vec = static_cast<T*>(pin_vec.p);
        std::vector<int64_t> cbase(static_cast<size_t>(n_buckets), -1);
        std::vector<int64_t> prev_cbase(static_cast<size_t>(n_buckets), -1);   // 上一块里桶的行 (在 d_*B[cur ^ 1])
        std::vector<int32_t> stamp(static_cast<size_t>(n_buckets), -1);
        std::vector<int32_t> U, prevU;
        std::vector<RowRun> h_runs;
        int cur = 0;
        int32_t chunk_id = 0;
        reset_unverified();
        float* d_cdist = nullptr;              // B13
        std::vector<float> h_cdist;
        if (g_center_dists) {
            PWG_CHECK(cudaMalloc(&d_cdist, static_cast<size_t>(cap_rows) * M * sizeof(float)));
            std::cout << "  B13: centers ranked by Step 6 distances (vector_dists)\n";
        }
        if (pool_mode) {
            // ---- 先把所有块规划出来 (跟下面原来的规划完全一样)，记下每个桶被哪些块用到 ----
            struct Plan { int p0, p1; std::vector<int32_t> U; };
            std::vector<Plan> plans;
            for (int p0 = 0; p0 < n_buckets; ++chunk_id) {
                Plan pl; pl.p0 = p0;
                int64_t rows = 0; int p1 = p0;
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
                        if (stamp[static_cast<size_t>(q)] != chunk_id) { stamp[static_cast<size_t>(q)] = chunk_id; pl.U.push_back(q); }
                    }
                    rows += add; ++p1;
                }
                pl.p1 = p1; plans.push_back(std::move(pl)); p0 = p1;
            }
            std::vector<std::vector<int32_t>> uses(static_cast<size_t>(n_buckets));
            for (int32_t c = 0; c < static_cast<int32_t>(plans.size()); ++c)
                for (int q : plans[static_cast<size_t>(c)].U) uses[static_cast<size_t>(q)].push_back(c);
            std::vector<int32_t> use_ptr(static_cast<size_t>(n_buckets), 0);
            std::vector<std::vector<int32_t>> slots_of(static_cast<size_t>(n_buckets));
            std::vector<char> resident(static_cast<size_t>(n_buckets), 0);
            std::vector<int32_t> free_slots(static_cast<size_t>(pool_slots));
            std::iota(free_slots.begin(), free_slots.end(), 0);
            std::vector<int32_t> resident_list;
            std::vector<int32_t> h_tab; std::vector<SlotJob> h_jobs;
            std::vector<int32_t> prevU;
            // 流水线 (TAGORE largeIndex 的 GPU 构建 / CPU 归并重叠): 块 c 的槽规划 + 读盘跟 GPU 上的块 c-1 重叠，
            // 块 c-1 的结果 D2H + 写出跟 GPU 上的块 c 重叠。块 c 的散布 / kernel 跟块 c-1 同一 stream (槽复用按 stream
            // 顺序安全)，输出缓冲两套。pinned 行缓冲仍是一份 (host 内存不变)，当环用: 块 c 的新行接在块 c-1
            // 的后面，放不下就绕回开头；跟块 c-1 的区间重叠时先等块 c-1 的散布完成 (平均新行 < 半个缓冲，很少等)。
            // 每块的剪枝跟串行版本完全一样。
            Staging stg_out;                                   // 结果 D2H 走自己的 stream
            uint32_t* d_outB[2] = {d_out, d_out2};
            int64_t ring_off = 0, prev_off = 0, prev_hd = 0;   // 块 c-1 在环里的区间
            double t_plan = 0, t_load = 0, t_cdist = 0, t_wait = 0, t_d2h = 0;   // 分段计时 (prune 日志; CPU 侧)
            // 读盘线程数: 跟 GPU 散布 (零拷贝读 pinned) 并发时线程多反而慢 (msTuring100M: 8 线程 load 43 s，
            // 16 / 32 / 80 线程 58 / 65 / 57 s)；BUCKET2_PRUNE_LOAD_THREADS 可改
            const int load_threads = std::max(1, std::min(omp_get_max_threads(),
                std::getenv("BUCKET2_PRUNE_LOAD_THREADS") ? std::atoi(std::getenv("BUCKET2_PRUNE_LOAD_THREADS")) : 8));
            auto nslots = [&](int q) { return static_cast<int32_t>((h_offsets[q + 1] - h_offsets[q] + kSlotRows - 1) / kSlotRows); };
            struct Pending { int64_t row0, n_rows; };
            Pending pend[2] = {};
            // 块 c 的结果: 等它的 kernel 完成、查错、D2H + sink
            auto finish = [&](int32_t c) {
                const int b = c & 1;
                double t0 = omp_get_wtime();
                PWG_CHECK(cudaEventSynchronize(ev_done[b]));
                int h_err = 0;
                PWG_CHECK(cudaMemcpyAsync(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost, stg_out.st));
                PWG_CHECK(cudaStreamSynchronize(stg_out.st));
                double t1 = omp_get_wtime(); t_wait += t1 - t0; t0 = t1;
                if (h_err)
                    throw std::runtime_error(h_err == 2 ? "prune_windowed_gpu: center bucket not loaded in chunk"
                                                        : "prune_windowed_gpu: fewer than output_degree valid candidates "
                                                          "(check --output-degree <= M_in and padding)");
                const int64_t row0 = pend[b].row0;
                stg_out.d2h_rows(d_outB[b], pend[b].n_rows, out_deg, [&](const uint32_t* r, int64_t r0, int64_t n) { sink(r, row0 + r0, n); });
                t_d2h += omp_get_wtime() - t0;
            };
            for (int32_t c = 0; c < static_cast<int32_t>(plans.size()); ++c) {
                const Plan& pl = plans[static_cast<size_t>(c)];
                const int b = c & 1;
                double t0 = omp_get_wtime();
                for (int q : prevU) cbase[static_cast<size_t>(q)] = -1;
                for (int q : pl.U) { use_ptr[static_cast<size_t>(q)]++; stamp[static_cast<size_t>(q)] = -2 - c; }   // 本块这次用掉 + 标记 "本块要用" (淘汰时跳过)
                // 未命中的桶 + 需要的槽数
                std::vector<int32_t> fresh;
                int64_t need = 0;
                for (int q : pl.U) if (!resident[static_cast<size_t>(q)]) { fresh.push_back(q); need += nslots(q); }
                if (need > static_cast<int64_t>(free_slots.size())) {   // Belady: 本块用不到、下次用得最晚的先淘汰
                    std::vector<std::pair<int32_t, int32_t>> ev;   // (next use, bucket)
                    for (int q : resident_list) {
                        if (!resident[static_cast<size_t>(q)] || stamp[static_cast<size_t>(q)] == -2 - c) continue;
                        const auto& u = uses[static_cast<size_t>(q)];
                        const int32_t nu = use_ptr[static_cast<size_t>(q)] < static_cast<int32_t>(u.size()) ? u[static_cast<size_t>(use_ptr[static_cast<size_t>(q)])] : INT32_MAX;
                        ev.push_back({nu, q});
                    }
                    std::sort(ev.begin(), ev.end(), [](auto& x, auto& y) { return x.first > y.first; });
                    for (auto& e : ev) {
                        if (need <= static_cast<int64_t>(free_slots.size())) break;
                        const int q = e.second;
                        for (int32_t s : slots_of[static_cast<size_t>(q)]) free_slots.push_back(s);
                        slots_of[static_cast<size_t>(q)].clear();
                        resident[static_cast<size_t>(q)] = 0;
                    }
                    if (need > static_cast<int64_t>(free_slots.size()))
                        throw std::runtime_error("prune_windowed_gpu: B12 pool too small for one chunk");
                    resident_list.erase(std::remove_if(resident_list.begin(), resident_list.end(),
                        [&](int q) { return !resident[static_cast<size_t>(q)]; }), resident_list.end());
                }
                // 新桶分槽，按 id 排好以便合并连续段读盘
                std::sort(fresh.begin(), fresh.end());
                for (int q : fresh) {
                    auto& sl = slots_of[static_cast<size_t>(q)];
                    for (int32_t k = 0; k < nslots(q); ++k) { sl.push_back(free_slots.back()); free_slots.pop_back(); }
                    resident[static_cast<size_t>(q)] = 1; resident_list.push_back(q);
                }
                // 槽表 (本块用到的桶) + cbase = 槽表里的起点
                h_tab.clear();
                for (int q : pl.U) {
                    cbase[static_cast<size_t>(q)] = static_cast<int64_t>(h_tab.size());
                    for (int32_t s : slots_of[static_cast<size_t>(q)]) h_tab.push_back(s);
                }
                // 读盘: 新桶连续段并行 pread 进 pinned 环 (块 c-2 已经 finish，只需避开块 c-1 的区间)
                struct Run { int64_t lo, n, hd; };
                std::vector<Run> runs; h_jobs.clear();
                int64_t hd = 0;
                for (int q : fresh) {
                    const int64_t lo = h_offsets[q], n = static_cast<int64_t>(h_offsets[q + 1]) - lo;
                    if (!runs.empty() && runs.back().lo + runs.back().n == lo) runs.back().n += n;
                    else runs.push_back({lo, n, hd});
                    h_jobs.push_back({hd, static_cast<int32_t>(n), static_cast<int32_t>(cbase[static_cast<size_t>(q)])});
                    hd += n;
                }
                if (hd > cap_rows) throw std::runtime_error("prune_windowed_gpu: B12 fresh rows exceed the pinned buffer");
                if (ring_off + hd > cap_rows) ring_off = 0;
                if (c > 0 && ring_off < prev_off + prev_hd && prev_off < ring_off + hd)
                    PWG_CHECK(cudaEventSynchronize(ev_scat[b ^ 1]));   // 块 c-1 的散布读完了这段 pinned
                for (auto& j : h_jobs) j.host_row += ring_off;
                double t1 = omp_get_wtime(); t_plan += t1 - t0; t0 = t1;
                int32_t* hc = h_cand + static_cast<size_t>(ring_off) * M; T* hv = h_vec + static_cast<size_t>(ring_off) * D;
                #pragma omp parallel for schedule(dynamic, 1) num_threads(load_threads)
                for (int64_t i = 0; i < static_cast<int64_t>(runs.size()); ++i) {
                    const Run& r = runs[static_cast<size_t>(i)];
                    load_rows(r.lo, r.n, hc + static_cast<size_t>(r.hd) * M, hv + static_cast<size_t>(r.hd) * D);
                }
                t1 = omp_get_wtime(); t_load += t1 - t0; t0 = t1;
                const int64_t row0 = h_offsets[pl.p0], n_rows = static_cast<int64_t>(h_offsets[pl.p1]) - row0;
                const int64_t blocks = (n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
                if (d_cdist && n_rows > 0) {
                    if (n_rows > cap_rows) throw std::runtime_error("prune_windowed_gpu: B13 center rows exceed capacity");
                    h_cdist.resize(static_cast<size_t>(n_rows) * M);
                    g_center_dists(row0, n_rows, h_cdist.data());
                    t1 = omp_get_wtime(); t_cdist += t1 - t0; t0 = t1;
                }
                // 以下排在块 c-1 的 kernel 之后 (同一 stream); staging 上传可能等到它完成
                stg.h2d(d_tab, h_tab.data(), h_tab.size() * sizeof(int32_t));
                stg.h2d(d_cbase, cbase.data(), cbase.size() * sizeof(int64_t));
                if (!h_jobs.empty()) {
                    stg.h2d(d_jobs, h_jobs.data(), h_jobs.size() * sizeof(SlotJob));
                    scatter_slots_kernel<<<static_cast<unsigned>(h_jobs.size()), 256, 0, stg.st>>>(
                        d_jobs, d_tab, h_cand, reinterpret_cast<const uint8_t*>(h_vec), d_candP,
                        reinterpret_cast<uint8_t*>(d_vecP), M, static_cast<int>(vec_row));
                    PWG_CHECK(cudaGetLastError());
                }
                PWG_CHECK(cudaEventRecord(ev_scat[b], stg.st));
                prev_off = ring_off; prev_hd = hd; ring_off += hd;
                if (d_cdist && n_rows > 0)
                    PWG_CHECK(cudaMemcpyAsync(d_cdist, h_cdist.data(), h_cdist.size() * sizeof(float), cudaMemcpyHostToDevice, stg.st));
                if (blocks > 0)
                    prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(
                        d_candP, d_vecP, d_off, d_win, d_wl, d_cbase, row0, n_rows, M, D, n_buckets, W, out_deg, d_outB[b], d_err,
                        false, nullptr, d_cdist, d_tab);
                PWG_CHECK(cudaGetLastError());
                PWG_CHECK(cudaEventRecord(ev_done[b], stg.st));
                pend[b] = {row0, n_rows};
                t_wait += omp_get_wtime() - t0;
                if (c > 0) finish(c - 1);   // 块 c-1 的结果写出跟 GPU 上的块 c 重叠
                if (stats) { stats->chunks++; stats->loaded_rows += hd; }
                prevU = pl.U;
            }
            if (!plans.empty()) finish(static_cast<int32_t>(plans.size()) - 1);
            std::cout << "  B12: GPU bucket pool " << pool_slots << " slots x " << kSlotRows << " rows\n";
            std::cout << "  [Chunked timing] plan=" << t_plan << "s load=" << t_load << "s center_dists=" << t_cdist
                      << "s gpu_wait=" << t_wait << "s d2h+sink=" << t_d2h << "s (pipelined)\n";
        } else
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
                    d_runs, d_candB[cur], reinterpret_cast<const uint8_t*>(d_vecB[cur]), d_candB[nb],
                    reinterpret_cast<uint8_t*>(d_vecB[nb]), M, static_cast<int>(vec_row));
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
                                      static_cast<size_t>(fresh_rows) * vec_row, cudaMemcpyHostToDevice, stg.st));
            stg.h2d(d_cbase, cbase.data(), cbase.size() * sizeof(int64_t));
            int32_t* d_cand = d_candB[nb]; T* d_vec = d_vecB[nb];
            cur = nb;
            const int64_t row0 = h_offsets[p0], n_rows = static_cast<int64_t>(h_offsets[p1]) - row0;
            const int64_t blocks = (n_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
            if (d_cdist && n_rows > 0) {
                if (n_rows > cap_rows) throw std::runtime_error("prune_windowed_gpu: B13 center rows exceed capacity");
                h_cdist.resize(static_cast<size_t>(n_rows) * M);
                g_center_dists(row0, n_rows, h_cdist.data());
                PWG_CHECK(cudaMemcpyAsync(d_cdist, h_cdist.data(), h_cdist.size() * sizeof(float), cudaMemcpyHostToDevice, stg.st));
                PWG_CHECK(cudaStreamSynchronize(stg.st));   // h_cdist 下一块会被重写
            }
            if (blocks > 0)
                prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32, 0, stg.st>>>(
                    d_cand, d_vec, d_off, d_win, d_wl, d_cbase, row0, n_rows, M, D, n_buckets, W, out_deg, d_out, d_err,
                    false, nullptr, d_cdist);
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
        report_unverified(N, out_deg);
        cudaFree(d_cdist);
    } catch (...) {
        cleanup(); cleanup_pool();
        throw;
    }
    cleanup(); cleanup_pool();
}

// uint8 / int8 / float32 数据集 (prune_windowed 按扩展名选)
#define PWG_INSTANTIATE(T)                                                                               \
    template void run<T>(const int32_t*, const T*, int64_t, int, int, const uint32_t*, int,              \
                         const std::vector<int32_t>&, const std::vector<int32_t>&, int, int,             \
                         const std::function<void(const uint32_t*, int64_t, int64_t)>&);                \
    template void run_chunked<T>(int64_t, int, int, const uint32_t*, int, const std::vector<int32_t>&,  \
                                 const std::vector<int32_t>&, int, int, size_t,                          \
                                 const std::function<void(int64_t, int64_t, int32_t*, T*)>&,             \
                                 const std::function<void(const uint32_t*, int64_t, int64_t)>&, ChunkStats*);
PWG_INSTANTIATE(uint8_t)
PWG_INSTANTIATE(int8_t)
PWG_INSTANTIATE(float)

}  // namespace prune_windowed_gpu
