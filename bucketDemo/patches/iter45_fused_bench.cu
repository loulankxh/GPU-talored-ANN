#include "fused_knn.cuh"
#include <cstdio>
#include <vector>
int main_bench();
int main() { int r = main_bench(); unsigned long long c=0;
#ifdef FUSED_COUNT
cudaMemcpyFromSymbol(&c, fused_knn::g_flush_count, 8);
#endif
printf("flushes total=%llu per-bucket=%.1f\n", c, c/800.0); return r; }
int main_bench() {
    const int bs = 100, ps = 20000, D = 128, K = 64, NB = 400;
    size_t prow = fused_knn::padded_rows(ps);
    __half* pool; float* nrm; int32_t* ids; float* pd; int32_t* pi; unsigned* cnt; int32_t* on; float* od;
    cudaMalloc(&pool, prow * D * 2); cudaMemset(pool, 0, prow * D * 2);
    std::vector<__half> h(prow * D); for (size_t i = 0; i < h.size(); ++i) h[i] = __float2half((float)((i * 2654435761u >> 7) & 255));
    cudaMemcpy(pool, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&nrm, ps * 4); cudaMemset(nrm, 0, ps * 4);
    cudaMalloc(&ids, ps * 4); cudaMemset(ids, 0, ps * 4);
    cudaMalloc(&pd, fused_knn::partial_elems(bs) * 4 * 4); cudaMalloc(&pi, fused_knn::partial_elems(bs) * 4 * 4);
    cudaMalloc(&cnt, 4096); cudaMemset(cnt, 0, 4096);
    cudaMalloc(&on, bs * 64 * 4); cudaMalloc(&od, bs * 64 * 4);
    cudaStream_t st[4]; for (auto& s : st) cudaStreamCreate(&s);
    fused_knn::Scratch sc[4];
    for (int i = 0; i < 4; ++i) { cudaMalloc(&sc[i].part_d, fused_knn::partial_elems(bs) * 4); cudaMalloc(&sc[i].part_i, fused_knn::partial_elems(bs) * 4); cudaMalloc(&sc[i].counters, 64); cudaMemset(sc[i].counters, 0, 64); cudaMalloc(&sc[i].row_th, bs * 4); cudaMemset(sc[i].row_th, 0xFF, bs * 4); }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int w = 0; w < 2; ++w) {
        cudaEventRecord(a);
        for (int i = 0; i < NB; ++i) fused_knn::launch<true>(st[i % 4], pool, nrm, ids, bs, ps, D, K, sc[i % 4], on, od, 64);
        cudaDeviceSynchronize();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("fused: %.1f us/bucket (4 streams), err=%s\n", ms * 1000 / NB, cudaGetErrorString(cudaGetLastError()));
    }
    return 0;
}
