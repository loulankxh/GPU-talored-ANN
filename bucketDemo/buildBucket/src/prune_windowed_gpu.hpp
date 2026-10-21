// prune_windowed_gpu.hpp — GPU kernel for prune_windowed's batched path (see prune_windowed_gpu.cu).
#pragma once
#include <cstdint>
#include <vector>

namespace prune_windowed_gpu {
// true if a CUDA device is present, M <= 64 and the working set fits in free GPU memory
bool available(int64_t N, int M, int D, int n_buckets, int W, int out_deg);
// cand (N,M) int32 (-1 padding), vec (N,D) uint8, offsets (n_buckets+1) by processing position,
// win (n_buckets, W) sorted window positions per position (padded), win_len (n_buckets).
// Writes out (N, out_deg). Throws on CUDA errors / too few valid candidates.
void run(const int32_t* cand, const uint8_t* vec, int64_t N, int M, int D,
         const uint32_t* offsets, int n_buckets,
         const std::vector<int32_t>& win, const std::vector<int32_t>& win_len, int W,
         int out_deg, uint32_t* out);
}
