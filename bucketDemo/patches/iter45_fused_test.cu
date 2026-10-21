// fused_knn 正确性测试: 随机 u8 数据, 对比 CPU 暴力 top-K (距离逐位比较, id 只在并列外比较)
#include "fused_knn.cuh"
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int run(int bs, int ps, int D, int K, bool rowByQuery, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u(0, 255);
    std::vector<uint8_t> x(static_cast<size_t>(ps) * D);
    for (auto& v : x) v = static_cast<uint8_t>(u(rng));
    // 一些重复行制造并列
    for (int r = 0; r < ps / 50; ++r) { int a = rng() % ps, b = rng() % ps; std::copy(x.begin() + (size_t)a * D, x.begin() + (size_t)(a + 1) * D, x.begin() + (size_t)b * D); }
    size_t prow = fused_knn::padded_rows(ps);
    std::vector<__half> h(prow * D, __float2half(0.f));
    std::vector<float> nrm(ps);
    std::vector<int32_t> ids(ps);
    for (int i = 0; i < ps; ++i) {
        double s = 0; for (int d = 0; d < D; ++d) { float f = x[(size_t)i * D + d]; h[(size_t)i * D + d] = __float2half(f); s += f * f; }
        nrm[i] = (float)s; ids[i] = 1000000 + i * 7;
    }
    // padding 行塞 NaN 检查确实被屏蔽
    for (size_t i = (size_t)ps * D; i < prow * D; ++i) h[i] = __float2half(NAN);
    __half* d_pool; float* d_n; int32_t* d_ids; float* pd; int32_t* pi; unsigned* cnt; int32_t* on; float* od;
    int out_rows = rowByQuery ? bs : 1000000 + ps * 7;   // 按 gid 写时行号 = ids_pool[q]
    CK(cudaMalloc(&d_pool, h.size() * 2)); CK(cudaMemcpy(d_pool, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_n, ps * 4)); CK(cudaMemcpy(d_n, nrm.data(), ps * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_ids, ps * 4)); CK(cudaMemcpy(d_ids, ids.data(), ps * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&pd, fused_knn::partial_elems(bs) * 4)); CK(cudaMalloc(&pi, fused_knn::partial_elems(bs) * 4));
    CK(cudaMalloc(&cnt, fused_knn::counter_elems(bs) * 4)); CK(cudaMemset(cnt, 0, fused_knn::counter_elems(bs) * 4));
    int out_ld = 64;
    CK(cudaMalloc(&on, (size_t)out_rows * out_ld * 4)); CK(cudaMalloc(&od, (size_t)out_rows * out_ld * 4));
    unsigned* rth; CK(cudaMalloc(&rth, bs * 4)); CK(cudaMemset(rth, 0xFF, bs * 4));
    fused_knn::Scratch sc{pd, pi, cnt, rth};
    for (int rep = 0; rep < 2; ++rep) {   // 两次: 检查计数器复位
        bool ok = rowByQuery ? fused_knn::launch<true>(0, d_pool, d_n, d_ids, bs, ps, D, K, sc, on, od, out_ld)
                             : fused_knn::launch<false>(0, d_pool, d_n, d_ids, bs, ps, D, K, sc, on, od, out_ld);
        if (!ok) { printf("unsupported\n"); return 1; }
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    }
    std::vector<unsigned> hc(fused_knn::counter_elems(bs));
    CK(cudaMemcpy(hc.data(), cnt, hc.size() * 4, cudaMemcpyDeviceToHost));
    for (unsigned c : hc) if (c) { printf("counter not reset\n"); return 1; }
    std::vector<unsigned> hr(bs); CK(cudaMemcpy(hr.data(), rth, bs * 4, cudaMemcpyDeviceToHost));
    for (unsigned c : hr) if (c != 0xFFFFFFFFu) { printf("row_th not reset\n"); return 1; }
    int bad = 0;
    for (int q = 0; q < bs; ++q) {
        std::vector<std::pair<float, int>> all;
        for (int j = 0; j < ps; ++j) {
            if (j == q) continue;
            double s = 0; for (int d = 0; d < D; ++d) { double t = (double)x[(size_t)q * D + d] - x[(size_t)j * D + d]; s += t * t; }
            all.push_back({(float)s, j});
        }
        std::sort(all.begin(), all.end());
        int64_t row = rowByQuery ? q : ids[q];
        std::vector<int32_t> gn(K); std::vector<float> gd(K);
        CK(cudaMemcpy(gn.data(), on + row * out_ld, K * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gd.data(), od + row * out_ld, K * 4, cudaMemcpyDeviceToHost));
        int kk = std::min<int>(K, (int)all.size());
        for (int k = 0; k < kk; ++k) {
            if (gd[k] != all[k].first) { if (bad < 5) printf("q=%d k=%d dist %f vs %f\n", q, k, gd[k], all[k].first); ++bad; break; }
            // id 必须是同距离的某个点
            int j = (gn[k] - 1000000) / 7;
            if (j < 0 || j >= ps || j == q) { ++bad; printf("bad id q=%d k=%d\n", q, k); break; }
            double s = 0; for (int d = 0; d < D; ++d) { double t = (double)x[(size_t)q * D + d] - x[(size_t)j * D + d]; s += t * t; }
            if ((float)s != gd[k]) { ++bad; printf("id/dist mismatch q=%d k=%d\n", q, k); break; }
        }
        // 重复 id 检查
        std::vector<int32_t> srt(gn.begin(), gn.begin() + kk); std::sort(srt.begin(), srt.end());
        if (std::adjacent_find(srt.begin(), srt.end()) != srt.end()) { ++bad; printf("dup id q=%d\n", q); }
    }
    printf("bs=%d ps=%d D=%d K=%d rowByQuery=%d -> %s (%d bad rows)\n", bs, ps, D, K, (int)rowByQuery, bad ? "FAIL" : "OK", bad);
    cudaFree(d_pool); cudaFree(d_n); cudaFree(d_ids); cudaFree(pd); cudaFree(pi); cudaFree(cnt); cudaFree(on); cudaFree(od);
    return bad ? 1 : 0;
}

int main() {
    int f = 0;
    f |= run(100, 20000, 128, 64, true, 1);
    f |= run(37, 3001, 128, 64, false, 2);
    f |= run(1, 129, 128, 64, true, 3);
    f |= run(33, 500, 128, 32, true, 4);
    f |= run(800, 25000, 128, 64, true, 5);
    f |= run(64, 1000, 128, 64, false, 6);
    f |= run(50, 60, 128, 64, true, 7);   // pool 比 K 小
    printf(f ? "SOME FAILED\n" : "ALL OK\n");
    return f;
}
