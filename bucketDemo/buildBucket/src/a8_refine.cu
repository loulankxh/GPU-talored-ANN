// a8_refine: A8 (user-approved experiment, 2026-10-25) — TAGORE-style local join over a finished graph.
//
// Input: final graph (N, K) in original ids (after prune + reverse-edge merge) and the base vectors.
// For every node a, candidates = forward neighbors N(a) ∪ reverse neighbors R(a) (≤ R_CAP, the first
// R_CAP sources in id order) ∪ 2-hop forward neighbors (first S of each of a's first S forward neighbors),
// as TAGORE's GNN-Descent samples forward + reverse neighbors (nn_descent_opt_reverse_sample /
// merge_reverse_plus). Exact L2 distances, top-M kept → vector_knn.bin (N, M) sorted by distance, ready
// for `optimize` (CAGRA prune + reverse edges) to bring it back to degree K.
//
//   a8_refine -i base.{fbin,u8bin,i8bin} -g final_graph.bin -o refined_knn.bin [--m 64] [--sample 16] [--rev-cap 32]
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <cuda_runtime.h>

#include "graph_io.hpp"
#include "load.hpp"

namespace po = boost::program_options;

#define A8_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) \
    throw std::runtime_error(std::string("a8_refine: ") + cudaGetErrorString(e_)); } while (0)

namespace {
constexpr int kMaxM = 64;
constexpr int kMaxCand = 1024;
struct Counters { unsigned long long cand, inserted; };

// one block (128 threads) per node: gather candidates, exact distances (warp per candidate), rank-select top M
__global__ void __launch_bounds__(128) a8_join_kernel(
    const float* __restrict__ data, const float* __restrict__ norms, int64_t N, int D,
    const uint32_t* __restrict__ fwd, int K,                        // (N, K)
    const int64_t* __restrict__ rev_off, const uint32_t* __restrict__ rev, int S,
    int M, int32_t* __restrict__ out, Counters* __restrict__ ctr)
{
    __shared__ int32_t s_id[kMaxCand];
    __shared__ float   s_d[kMaxCand];
    __shared__ float   s_a[256];
    __shared__ int s_n;
    const int64_t a = blockIdx.x;
    if (a >= N) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    for (int e = tid; e < D; e += blockDim.x) s_a[e] = data[a * D + e];
    if (tid == 0) s_n = 0;
    __syncthreads();
    auto push = [&](uint32_t v) {
        if (v >= static_cast<uint64_t>(N) || v == static_cast<uint64_t>(a)) return;   // bounds check
        const int p = atomicAdd(&s_n, 1);
        if (p < kMaxCand) s_id[p] = static_cast<int32_t>(v);
    };
    const uint32_t* fa = fwd + a * K;
    for (int k = tid; k < K; k += blockDim.x) push(fa[k]);                                 // forward
    for (int64_t r = rev_off[a] + tid; r < rev_off[a + 1]; r += blockDim.x) push(rev[r]); // reverse (capped)
    const int s1 = min(S, K);
    for (int c = tid; c < s1 * s1; c += blockDim.x) {                                      // 2-hop forward
        const uint32_t u = fa[c / s1];
        if (u < static_cast<uint64_t>(N)) push(fwd[static_cast<int64_t>(u) * K + (c % s1)]);
    }
    __syncthreads();
    const int n = min(s_n, kMaxCand);
    // duplicates → keep the first occurrence
    for (int i = tid; i < n; i += blockDim.x) {
        const int32_t v = s_id[i];
        for (int o = 0; o < i; ++o) if (s_id[o] == v) { s_id[i] = -1; break; }
    }
    __syncthreads();
    const float na = norms[a];
    for (int i = warp; i < n; i += nw) {
        const int32_t v = s_id[i];
        float part = 0.0f;
        if (v >= 0) {
            const float* pv = data + static_cast<int64_t>(v) * D;
            for (int e = lane; e < D; e += 32) part += s_a[e] * pv[e];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
        if (lane == 0) s_d[i] = (v >= 0) ? na + norms[v] - 2.0f * part : INFINITY;
    }
    __syncthreads();
    int32_t* orow = out + a * M;
    for (int k = tid; k < M; k += blockDim.x) orow[k] = -1;
    __syncthreads();
    int ins = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const float d = s_d[i];
        if (!(d < INFINITY)) continue;
        int rank = 0;
        for (int o = 0; o < n; ++o) rank += (s_d[o] < d) || (s_d[o] == d && o < i);
        if (rank < M) {
            orow[rank] = s_id[i];
            bool was_fwd = false;   // 原来就是前向邻居？
            for (int k = 0; k < K && !was_fwd; ++k) was_fwd = (fa[k] == static_cast<uint32_t>(s_id[i]));
            if (!was_fwd) ++ins;
        }
    }
    if (ins) atomicAdd(&ctr->inserted, static_cast<unsigned long long>(ins));
    if (tid == 0) atomicAdd(&ctr->cand, static_cast<unsigned long long>(n));
}

__global__ void norms_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n, int D)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.f;
    for (int d = 0; d < D; ++d) { const float v = x[i * D + d]; s += v * v; }
    out[i] = s;
}
}  // namespace

int main(int argc, char** argv)
{
    try {
        po::options_description desc("A8: local join over forward + reverse neighbors of a finished graph");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", po::value<std::string>()->required(), "base vectors (.fbin / .u8bin / .i8bin), original order")
            ("graph,g", po::value<std::string>()->required(), "final graph (N, K) in original ids")
            ("output,o", po::value<std::string>()->required(), "refined vector_knn.bin (N, M)")
            ("m", po::value<int>()->default_value(64), "output list width M (<= 64)")
            ("sample", po::value<int>()->default_value(16), "2-hop sample S (first S of the first S neighbors)")
            ("rev-cap", po::value<int>()->default_value(32), "max reverse neighbors per node");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);
        const std::string in_path = vm["input"].as<std::string>();
        const int M = vm["m"].as<int>(), S = vm["sample"].as<int>(), RC = vm["rev-cap"].as<int>();
        if (M < 1 || M > kMaxM || S < 1 || RC < 0) throw std::runtime_error("bad --m / --sample / --rev-cap");
        auto t0 = std::chrono::steady_clock::now();

        std::vector<float> x; int32_t n32 = 0, D = 0;
        if (in_path.size() > 5 && in_path.substr(in_path.size() - 5) == ".fbin") load::read_fbin_f32(in_path, x, n32, D);
        else load::read_u8bin_to_f32(in_path, x, n32, D);
        if (D > 256) throw std::runtime_error("D > 256 not supported");
        std::vector<uint32_t> g; int64_t N = 0; int32_t K = 0;
        graph_io::read_forward_graph(vm["graph"].as<std::string>(), g, N, K);
        if (N != n32) throw std::runtime_error("graph N != data N");
        if (K + RC + std::min(S, static_cast<int>(K)) * std::min(S, static_cast<int>(K)) > kMaxCand)
            throw std::runtime_error("candidate cap exceeded: lower --sample / --rev-cap");

        // reverse lists (CSR), first RC sources in id order
        std::vector<int64_t> cnt(static_cast<size_t>(N) + 1, 0);
        for (int64_t a = 0; a < N; ++a)
            for (int k = 0; k < K; ++k) { const uint32_t v = g[static_cast<size_t>(a) * K + k]; if (v < N) ++cnt[v + 1]; }
        for (int64_t v = 0; v < N; ++v) cnt[v + 1] = cnt[v] + std::min<int64_t>(cnt[v + 1], RC);
        std::vector<uint32_t> rev(static_cast<size_t>(cnt[N]));
        std::vector<int64_t> fill(cnt.begin(), cnt.end() - 1);
        for (int64_t a = 0; a < N; ++a)
            for (int k = 0; k < K; ++k) {
                const uint32_t v = g[static_cast<size_t>(a) * K + k];
                if (v < N && fill[v] < cnt[v + 1]) rev[static_cast<size_t>(fill[v]++)] = static_cast<uint32_t>(a);
            }

        float *d_x = nullptr, *d_n = nullptr; uint32_t *d_g = nullptr, *d_rev = nullptr; int64_t* d_off = nullptr;
        int32_t* d_out = nullptr; Counters* d_c = nullptr;
        A8_CHECK(cudaMalloc(&d_x, x.size() * sizeof(float)));
        A8_CHECK(cudaMalloc(&d_n, static_cast<size_t>(N) * sizeof(float)));
        A8_CHECK(cudaMalloc(&d_g, g.size() * sizeof(uint32_t)));
        A8_CHECK(cudaMalloc(&d_rev, std::max<size_t>(rev.size(), 1) * sizeof(uint32_t)));
        A8_CHECK(cudaMalloc(&d_off, cnt.size() * sizeof(int64_t)));
        A8_CHECK(cudaMalloc(&d_out, static_cast<size_t>(N) * M * sizeof(int32_t)));
        A8_CHECK(cudaMalloc(&d_c, sizeof(Counters)));
        A8_CHECK(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        A8_CHECK(cudaMemcpy(d_g, g.data(), g.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        A8_CHECK(cudaMemcpy(d_rev, rev.data(), rev.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        A8_CHECK(cudaMemcpy(d_off, cnt.data(), cnt.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        A8_CHECK(cudaMemset(d_c, 0, sizeof(Counters)));
        norms_kernel<<<static_cast<unsigned>((N + 255) / 256), 256>>>(d_x, d_n, N, D);
        auto tk = std::chrono::steady_clock::now();
        a8_join_kernel<<<static_cast<unsigned>(N), 128>>>(d_x, d_n, N, D, d_g, K, d_off, d_rev, S, M, d_out, d_c);
        A8_CHECK(cudaGetLastError());
        A8_CHECK(cudaDeviceSynchronize());
        const double kern_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tk).count();
        std::vector<int32_t> out(static_cast<size_t>(N) * M);
        A8_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        Counters h{};
        A8_CHECK(cudaMemcpy(&h, d_c, sizeof(Counters), cudaMemcpyDeviceToHost));
        cudaFree(d_x); cudaFree(d_n); cudaFree(d_g); cudaFree(d_rev); cudaFree(d_off); cudaFree(d_out); cudaFree(d_c);
        graph_io::write_vector_knn(vm["output"].as<std::string>(), out, N, M);
        std::cout << "A1_stats: path=final variant=a8_fwd_rev_join sample=" << S << " rev_cap=" << RC << " points=" << N
                  << " cand_avg=" << static_cast<double>(h.cand) / N
                  << " new_in_topM_avg=" << static_cast<double>(h.inserted) / N
                  << " kernel_s=" << kern_s
                  << " total_s=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
