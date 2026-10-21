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
    int64_t N, int M, int D, int n_buckets, int W, int out_deg,
    uint32_t* __restrict__ out,            // (N, out_deg)
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
    const int64_t a = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + w;
    if (a >= N) return;
    const uint32_t total = offsets[n_buckets];
    const int p = find_pos(offsets, n_buckets, static_cast<uint32_t>(a));
    const int32_t* wp = win + static_cast<int64_t>(p) * W;
    const int wl = win_len[p];
    const uint8_t* va = vec + a * D;

    // ---- 1. distances ----
    for (int k = lane; k < kMaxM; k += 32) {
        float dist = INFINITY;
        int32_t id = -1;
        bool valid = false;
        if (k < M) {
            id = cand[a * M + k];
            if (id >= 0) {
                valid = true;
                if (static_cast<uint32_t>(id) < total &&
                    in_window(wp, wl, find_pos(offsets, n_buckets, static_cast<uint32_t>(id)))) {
                    const uint8_t* vc = vec + static_cast<int64_t>(id) * D;
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
        if (static_cast<uint32_t>(d_id) >= total ||
            !in_window(wp, wl, find_pos(offsets, n_buckets, static_cast<uint32_t>(d_id)))) continue;
        const int32_t* drow = cand + static_cast<int64_t>(d_id) * M;
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
        if (pos < out_deg) out[a * out_deg + pos] = static_cast<uint32_t>(s_id[w][r]);
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

void run(const int32_t* h_cand, const uint8_t* h_vec, int64_t N, int M, int D,
         const uint32_t* h_offsets, int n_buckets,
         const std::vector<int32_t>& h_win, const std::vector<int32_t>& h_win_len, int W,
         int out_deg, uint32_t* h_out)
{
    int32_t* d_cand = nullptr; uint8_t* d_vec = nullptr; uint32_t* d_off = nullptr;
    int32_t* d_win = nullptr; int32_t* d_wl = nullptr; uint32_t* d_out = nullptr; int* d_err = nullptr;
    auto cleanup = [&] {
        cudaFree(d_cand); cudaFree(d_vec); cudaFree(d_off); cudaFree(d_win);
        cudaFree(d_wl); cudaFree(d_out); cudaFree(d_err);
    };
    try {
        PWG_CHECK(cudaMalloc(&d_cand, static_cast<size_t>(N) * M * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_vec,  static_cast<size_t>(N) * D));
        PWG_CHECK(cudaMalloc(&d_off,  (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_win,  h_win.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_wl,   h_win_len.size() * sizeof(int32_t)));
        PWG_CHECK(cudaMalloc(&d_out,  static_cast<size_t>(N) * out_deg * sizeof(uint32_t)));
        PWG_CHECK(cudaMalloc(&d_err,  sizeof(int)));
        PWG_CHECK(cudaMemcpy(d_cand, h_cand, static_cast<size_t>(N) * M * sizeof(int32_t), cudaMemcpyHostToDevice));
        PWG_CHECK(cudaMemcpy(d_vec,  h_vec,  static_cast<size_t>(N) * D, cudaMemcpyHostToDevice));
        PWG_CHECK(cudaMemcpy(d_off,  h_offsets, (static_cast<size_t>(n_buckets) + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        PWG_CHECK(cudaMemcpy(d_win,  h_win.data(), h_win.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        PWG_CHECK(cudaMemcpy(d_wl,   h_win_len.data(), h_win_len.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
        PWG_CHECK(cudaMemset(d_err, 0, sizeof(int)));
        const int64_t blocks = (N + kWarpsPerBlock - 1) / kWarpsPerBlock;
        prune_kernel<<<static_cast<unsigned>(blocks), kWarpsPerBlock * 32>>>(
            d_cand, d_vec, d_off, d_win, d_wl, N, M, D, n_buckets, W, out_deg, d_out, d_err);
        PWG_CHECK(cudaGetLastError());
        PWG_CHECK(cudaDeviceSynchronize());
        int h_err = 0;
        PWG_CHECK(cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_err)
            throw std::runtime_error("prune_windowed_gpu: fewer than output_degree valid candidates "
                                     "(check --output-degree <= M_in and padding)");
        PWG_CHECK(cudaMemcpy(h_out, d_out, static_cast<size_t>(N) * out_deg * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

}  // namespace prune_windowed_gpu
