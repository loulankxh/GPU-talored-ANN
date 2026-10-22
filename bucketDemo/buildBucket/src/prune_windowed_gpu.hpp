// prune_windowed_gpu.hpp — GPU kernel for prune_windowed's batched path (see prune_windowed_gpu.cu).
#pragma once
#include <cstdint>
#include <vector>
#include <functional>

namespace prune_windowed_gpu {
// true if a CUDA device is present, M <= 64 and the working set fits in free GPU memory
bool available(int64_t N, int M, int D, int n_buckets, int W, int out_deg);
// cand (N,M) int32 (-1 padding), vec (N,D) uint8, offsets (n_buckets+1) by processing position,
// win (n_buckets, W) sorted window positions per position (padded), win_len (n_buckets).
// Writes out (N, out_deg). Throws on CUDA errors / too few valid candidates.
void run(const int32_t* cand, const uint8_t* vec, int64_t N, int M, int D,
         const uint32_t* offsets, int n_buckets,
         const std::vector<int32_t>& win, const std::vector<int32_t>& win_len, int W,
         int out_deg,
         // 结果按行块交回 (rows 指向 pinned 暂存区，row0 起 n 行，行宽 out_deg)，
         // 依次递增、覆盖 [0, N)；回调返回后暂存区会被复用
         const std::function<void(const uint32_t* rows, int64_t row0, int64_t n)>& sink);
struct ChunkStats { int64_t chunks = 0, loaded_rows = 0; };
// Same prune, out-of-core: positions are processed in chunks whose window union (rows of the
// buckets in the windows of those positions) fits budget_bytes (cand + vec rows); load_rows(lo, n,
// cand, vec) must fill n consecutive rows (new ids lo..lo+n) of candidates (int32, M per row) and
// uint8 vectors (D per row). Results go to sink in increasing row order like run().
void run_chunked(int64_t N, int M, int D, const uint32_t* offsets, int n_buckets,
                 const std::vector<int32_t>& win, const std::vector<int32_t>& win_len, int W,
                 int out_deg, size_t budget_bytes,
                 const std::function<void(int64_t lo, int64_t n, int32_t* cand, uint8_t* vec)>& load_rows,
                 const std::function<void(const uint32_t* rows, int64_t row0, int64_t n)>& sink,
                 ChunkStats* stats = nullptr);
}
