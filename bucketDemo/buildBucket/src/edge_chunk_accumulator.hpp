// edge_chunk_accumulator.hpp
// Core of optimize_chunked.cu's Method E: reverse-edge scatter into
// per-chunk disk-backed buffers (same buffer/flush/merge pattern as
// bucket.cu's BucketVectorAccumulator, just with EdgeRecord instead of
// vectors), plus the Phase-2 per-node rank-priority merge.
//
// This header has no CUDA dependency -- everything here is plain host C++
// so it can be exercised by a standalone g++ unit test
// (tests/test_edge_chunk_accumulator.cpp) without a GPU. optimize_chunked.cu
// includes it from device code too (for chunk_of()), hence the
// ECA_HOST_DEVICE qualifier.
//
// Why rank is carried at all: raft's own reverse-edge merge
// (graph_core.cuh's "Make reverse graph" + "Replace some edges with reverse
// edges") builds each node's reverse-edge list column-by-column (rank 0
// first, rank K_out-1 last) and processes it back-to-front, so the closest
// (rank 0) reverse edges end up with priority when a node has more
// candidates than protected-free slots. A GPU kernel that streams forward
// graph *rows* (not columns) loses that ordering unless it's carried
// explicitly -- so every scattered edge remembers which column (rank) of
// its source row it came from, and Phase 2 restores the priority order with
// a per-node sort before replaying the same shift/protect logic
// optimize_chunked.cu's merge_rev_into_output_chunk already uses.

#pragma once

#include <algorithm>
#include <omp.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define ECA_HOST_DEVICE __host__ __device__
#else
#define ECA_HOST_DEVICE
#endif

namespace edge_chunk {

// ---------------------------------------------------------------------
// Wire record: one reverse edge candidate, as scattered by the GPU kernel
// and flushed to a chunk's disk file.
// ---------------------------------------------------------------------
#pragma pack(push, 1)
struct EdgeRecord {
    uint32_t dst;    // global destination node id (defines which chunk/row)
    uint32_t src;    // global source node id (the candidate reverse edge)
    uint16_t rank;   // column index (0..K_out-1) of dst within src's own
                     // forward row -- 0 = src's closest neighbor
};
#pragma pack(pop)
static_assert(sizeof(EdgeRecord) == 10, "EdgeRecord must be 10 bytes packed");

using EdgeVec = std::vector<EdgeRecord>;

// ---------------------------------------------------------------------
// chunk_of: binary search over sorted chunk boundaries (ChunkPlan::starts
// in optimize_chunked.cu, size num_chunks+1). Returns c such that
// starts[c] <= j < starts[c+1]. __host__ __device__ so the exact same
// implementation runs in the scatter kernel and in Phase 2's host code.
// ---------------------------------------------------------------------
ECA_HOST_DEVICE inline uint32_t chunk_of(const uint32_t* starts, uint32_t num_chunks, uint32_t j) {
    uint32_t lo = 0, hi = num_chunks - 1;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;   // upper mid, avoids infinite loop
        if (starts[mid] <= j) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// ---------------------------------------------------------------------
// EdgeChunkAccumulator: Phase 1's disk-backed scatter target. One entry
// per chunk; each chunk gets a slice of a shared byte budget (mirrors
// BucketVectorAccumulator::create in bucket.cu:795), flushes to its own
// file when its slice fills, and keeps that file handle open for the whole
// run (the exact "don't reopen on every flush" fix from commit f52ba3f,
// applied here from the start instead of retrofitted).
// ---------------------------------------------------------------------
class EdgeChunkAccumulator {
public:
    struct ChunkBuf {
        std::vector<char> data;
        size_t capacity_records = 0;
        size_t count = 0;          // pending, not yet flushed
        int64_t total_count = 0;   // cumulative for this chunk, whole run
        std::ofstream file;
    };

    std::string dir;
    int32_t num_chunks = 0;
    std::vector<ChunkBuf> bufs;

    std::string merged_path;
    std::vector<int64_t> merged_offset;  // byte offset of chunk c in merged file
    std::vector<int64_t> merged_count;   // record count of chunk c
    std::ifstream merged_in;

    static std::string chunk_path(const std::string& dir, int32_t c) {
        return dir + "/edge_chunk_" + std::to_string(c) + ".bin";
    }

    // total_buffer_budget_bytes: shared across all chunks (not per-chunk).
    // max_records_per_chunk: optional cap -- this buffer is a pure
    // write-behind staging area (flushed to disk via flush_chunk() the
    // moment it's full, see add_records()), so it never needs to hold more
    // than one chunk's worth of a run's total edges. Without this cap, a
    // generous total_buffer_budget_bytes (sized for many chunks) handed to
    // a run with few chunks -- e.g. num_chunks=1 -- allocates and
    // zero-initializes the *entire* budget as one std::vector<char>::resize
    // (several seconds for a multi-GB buffer that will only ever hold a
    // small fraction of that before flushing), the same pathology as
    // optimize_chunked.cu's capacity_per_chunk.
    static EdgeChunkAccumulator create(const std::string& dir, int32_t num_chunks,
                                        size_t total_buffer_budget_bytes,
                                        size_t max_records_per_chunk = std::numeric_limits<size_t>::max()) {
        EdgeChunkAccumulator acc;
        acc.dir = dir;
        acc.num_chunks = num_chunks;
        acc.merged_path = dir + "/edge_chunks_merged.bin";

        constexpr size_t kMinBufferBytes = 4096;
        size_t nb = static_cast<size_t>(std::max<int32_t>(1, num_chunks));
        size_t buf_bytes = std::max<size_t>(kMinBufferBytes, total_buffer_budget_bytes / nb);
        size_t cap_records = std::max<size_t>(1, buf_bytes / sizeof(EdgeRecord));
        cap_records = std::min<size_t>(cap_records, max_records_per_chunk);

        acc.bufs.resize(static_cast<size_t>(num_chunks));
        for (auto& b : acc.bufs) {
            b.capacity_records = cap_records;
            b.data.resize(cap_records * sizeof(EdgeRecord));
        }
        return acc;
    }

    void start() {
        for (int32_t c = 0; c < num_chunks; ++c) {
            auto& b = bufs[static_cast<size_t>(c)];
            b.count = 0;
            b.total_count = 0;
            b.file.open(chunk_path(dir, c), std::ios::binary | std::ios::out | std::ios::trunc);
            if (!b.file.is_open())
                throw std::runtime_error("EdgeChunkAccumulator: cannot open " + chunk_path(dir, c));
        }
    }

    void flush_chunk(int32_t c) {
        auto& b = bufs[static_cast<size_t>(c)];
        if (b.count == 0) return;
        b.file.write(b.data.data(), static_cast<std::streamsize>(b.count * sizeof(EdgeRecord)));
        if (!b.file.good())
            throw std::runtime_error("EdgeChunkAccumulator: flush failed for chunk " + std::to_string(c));
        b.count = 0;
    }

    // Appends n records (already D2H'd, or synthetic in tests) into chunk
    // c's buffer, flushing to disk as needed. n may exceed capacity_records
    // (e.g. records drained from the shared overflow buffer in one go) --
    // this loops and flushes as many times as needed.
    void add_records(int32_t c, const EdgeRecord* recs, size_t n) {
        auto& b = bufs[static_cast<size_t>(c)];
        size_t off = 0;
        while (off < n) {
            size_t space = b.capacity_records - b.count;
            size_t take = std::min(space, n - off);
            std::memcpy(b.data.data() + b.count * sizeof(EdgeRecord),
                        recs + off, take * sizeof(EdgeRecord));
            b.count += take;
            b.total_count += static_cast<int64_t>(take);
            off += take;
            if (b.count == b.capacity_records) flush_chunk(c);
        }
    }

    int64_t count(int32_t c) const { return bufs[static_cast<size_t>(c)].total_count; }

    // Flush + close every chunk's file, sequentially concatenate them into
    // one merged file with an offset table (same rationale as
    // BucketVectorAccumulator::finish_iteration: bandwidth-bound sequential
    // copy instead of paying per-file open() overhead in Phase 2), then
    // open a persistent read handle.
    // 每个 chunk 的文件写完就是完整的 (只追加、记录数已知)，不再把它们拷进一个合并文件
    // (SIFT100M: 32 GB 读 + 32 GB 写, acc.finish 107 s)，read_chunk 直接读 chunk 自己的文件。
    // 参考 TAGORE order_merge_new: 每个局部图一个文件，最后一趟逐个流式读。
    void finish() {
        merged_count.assign(static_cast<size_t>(num_chunks), 0);
        for (int32_t c = 0; c < num_chunks; ++c) {
            flush_chunk(c);
            bufs[static_cast<size_t>(c)].file.close();
            merged_count[static_cast<size_t>(c)] = bufs[static_cast<size_t>(c)].total_count;
        }
    }

    // 读回 chunk c 的全部记录 (只读一次: 读完删掉它的文件)
    void read_chunk(int32_t c, EdgeVec& out) {
        int64_t cnt = merged_count[static_cast<size_t>(c)];
        out.resize(static_cast<size_t>(cnt));
        if (cnt == 0) return;
        std::string p = chunk_path(dir, c);
        std::ifstream in(p, std::ios::binary | std::ios::ate);
        if (!in.is_open())
            throw std::runtime_error("EdgeChunkAccumulator: cannot open " + p + " for read");
        const size_t nbytes = static_cast<size_t>(cnt) * sizeof(EdgeRecord);
        if (static_cast<size_t>(in.tellg()) != nbytes)   // bounds check: 文件大小必须跟计数一致
            throw std::runtime_error("EdgeChunkAccumulator: chunk " + std::to_string(c) + " file size mismatch");
        in.seekg(0);
        in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(nbytes));
        if (!in.good())
            throw std::runtime_error("EdgeChunkAccumulator: read_chunk failed for chunk " + std::to_string(c));
        in.close();
        std::filesystem::remove(p);
    }
};

struct ChunkCsr {
    std::vector<uint32_t> row_offsets;  // size chunk_size+1
    std::vector<uint32_t> src;          // size row_offsets.back(), rank-ascending per row
};

//
// Parallel (OpenMP; plain serial loops if compiled without -fopenmp):
// atomic per-row counting + atomic-cursor scatter, then each row's segment
// is sorted by (rank, src). The src tie-break makes the result independent
// of the (non-deterministic) scatter order -- previously ties in rank kept
// the GPU scatter's arrival order. Bounds check: every edge's dst must fall
// inside [j_lo, j_lo + chunk_size).
inline ChunkCsr build_rank_ordered_csr(const EdgeVec& edges,
                                        uint32_t j_lo, uint32_t chunk_size)
{
    struct RankSrc { uint16_t rank; uint32_t src; };

    ChunkCsr csr;
    csr.row_offsets.assign(static_cast<size_t>(chunk_size) + 1, 0);
    const int64_t n_edges = static_cast<int64_t>(edges.size());
    int64_t n_bad = 0;
    uint32_t* cnt = csr.row_offsets.data() + 1;
    #pragma omp parallel for schedule(static) reduction(+:n_bad)
    for (int64_t e = 0; e < n_edges; ++e) {
        uint32_t local_j = edges[static_cast<size_t>(e)].dst - j_lo;
        if (local_j >= chunk_size) { ++n_bad; continue; }
        #pragma omp atomic
        cnt[local_j]++;
    }
    if (n_bad > 0)
        throw std::runtime_error("build_rank_ordered_csr: " + std::to_string(n_bad) +
                                 " edges with dst outside the chunk");
    for (uint32_t i = 0; i < chunk_size; ++i)
        csr.row_offsets[i + 1] += csr.row_offsets[i];

    size_t total = csr.row_offsets[chunk_size];
    std::vector<RankSrc> tmp(total);
    std::vector<uint32_t> cursor(csr.row_offsets.begin(), csr.row_offsets.end() - 1);
    #pragma omp parallel for schedule(static)
    for (int64_t e = 0; e < n_edges; ++e) {
        const EdgeRecord& r = edges[static_cast<size_t>(e)];
        uint32_t local_j = r.dst - j_lo;
        uint32_t slot;
        #pragma omp atomic capture
        slot = cursor[local_j]++;
        tmp[slot] = RankSrc{r.rank, r.src};
    }

    csr.src.resize(total);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < static_cast<int64_t>(chunk_size); ++i) {
        uint32_t lo = csr.row_offsets[static_cast<size_t>(i)], hi = csr.row_offsets[static_cast<size_t>(i) + 1];
        if (hi - lo > 1) {
            std::sort(tmp.begin() + lo, tmp.begin() + hi,
                      [](const RankSrc& a, const RankSrc& b) {
                          return a.rank != b.rank ? a.rank < b.rank : a.src < b.src;
                      });
        }
        for (uint32_t p = lo; p < hi; ++p) csr.src[p] = tmp[p].src;
    }
    return csr;
}

// ---------------------------------------------------------------------
// Phase 2, step 2: single-row version of optimize_chunked.cu's
// merge_rev_into_output_chunk (optimize_chunked.cu:278). Same shift/protect
// math, but reads priority from `rank_sorted_src`'s physical order (index 0
// = strongest) instead of assuming write order encoded priority.
// ---------------------------------------------------------------------
// build_rank_ordered_csr 的 cache 友好版本 (结果逐字节相同)。原版对 1.6B 条边在 50M 行上做
// 原子计数 + 原子 scatter (随机访问整块 row_offsets/cursor)，SIFT100M Phase 2 里约一半时间。
// 这里两趟基数划分: (1) 每个线程按 kRowBlock 行一块做局部直方图 (无原子)，(2) 按块 scatter 成
// 64 位键 ((本地行 << 5 | rank) << 32 | src)，(3) 每块在 cache 里排序 → 行内按 (rank, src) 升序，
// 跟原版 sort 的比较完全一致 ((rank, src) 在一行内唯一，是全序)。参考 TAGORE large_index.cu
// reorder / order_merge_new 的分块处理思路 (先按块分桶，块内再做细活)。
// 前提: K_out ≤ 32 (rank 5 bit)、chunk_size < 2^27；不满足时调用方用原版。
inline bool rank_ordered_csr_fast_ok(uint32_t chunk_size, uint32_t K_out) {
    return K_out <= 32 && chunk_size < (1u << 27);
}

inline ChunkCsr build_rank_ordered_csr_fast(const EdgeVec& edges,
                                             uint32_t j_lo, uint32_t chunk_size)
{
    constexpr uint32_t kRowBlockBits = 13;            // 8192 行一块
    const uint32_t n_blocks = (chunk_size >> kRowBlockBits) + 1;
    const int64_t n_edges = static_cast<int64_t>(edges.size());
    const int T = std::max(1, omp_get_max_threads());

    // (1) 每线程每块的边数 (线程按静态区间分边，跟 (2) 的划分一致)
    std::vector<int64_t> hist(static_cast<size_t>(T) * n_blocks + 1, 0);
    int64_t n_bad = 0;
    #pragma omp parallel num_threads(T) reduction(+:n_bad)
    {
        const int t = omp_get_thread_num();
        const int64_t e0 = n_edges * t / T, e1 = n_edges * (t + 1) / T;
        int64_t* h = hist.data() + static_cast<size_t>(t) * n_blocks;
        for (int64_t e = e0; e < e1; ++e) {
            const EdgeRecord& r = edges[static_cast<size_t>(e)];
            uint32_t lj = r.dst - j_lo;
            if (lj >= chunk_size || r.rank >= 32) { ++n_bad; continue; }   // bounds check
            h[lj >> kRowBlockBits]++;
        }
    }
    if (n_bad > 0)
        throw std::runtime_error("build_rank_ordered_csr_fast: " + std::to_string(n_bad) +
                                 " edges with dst outside the chunk or rank >= 32");
    // 块主序的前缀: block b 的区间 = [blk_start[b], blk_start[b+1])，块内按线程号排
    std::vector<int64_t> blk_start(static_cast<size_t>(n_blocks) + 1, 0);
    std::vector<int64_t> cur(static_cast<size_t>(T) * n_blocks);
    {
        int64_t pos = 0;
        for (uint32_t b = 0; b < n_blocks; ++b) {
            blk_start[b] = pos;
            for (int t = 0; t < T; ++t) {
                cur[static_cast<size_t>(t) * n_blocks + b] = pos;
                pos += hist[static_cast<size_t>(t) * n_blocks + b];
            }
        }
        blk_start[n_blocks] = pos;
    }
    // (2) scatter 成 64 位键
    // 键数组不清零 (只被下面的并行 scatter 写满、只在本函数里用)；读文件的缓冲照旧用 std::vector
    std::unique_ptr<uint64_t[]> keys(new uint64_t[static_cast<size_t>(std::max<int64_t>(n_edges, 1))]);
    #pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const int64_t e0 = n_edges * t / T, e1 = n_edges * (t + 1) / T;
        int64_t* c = cur.data() + static_cast<size_t>(t) * n_blocks;
        for (int64_t e = e0; e < e1; ++e) {
            const EdgeRecord& r = edges[static_cast<size_t>(e)];
            uint32_t lj = r.dst - j_lo;
            uint64_t k = (static_cast<uint64_t>((lj << 5) | r.rank) << 32) | r.src;
            keys[static_cast<size_t>(c[lj >> kRowBlockBits]++)] = k;
        }
    }
    // (3) 块内排序 + 每行计数
    ChunkCsr csr;
    csr.row_offsets.assign(static_cast<size_t>(chunk_size) + 1, 0);
    #pragma omp parallel for schedule(dynamic, 4) num_threads(T)
    for (int64_t b = 0; b < static_cast<int64_t>(n_blocks); ++b) {
        uint64_t* kb = keys.get() + blk_start[static_cast<size_t>(b)];
        uint64_t* ke = keys.get() + blk_start[static_cast<size_t>(b) + 1];
        std::sort(kb, ke);
        for (uint64_t* k = kb; k < ke; ++k) csr.row_offsets[static_cast<size_t>(*k >> 37) + 1]++;
    }
    for (uint32_t i = 0; i < chunk_size; ++i)
        csr.row_offsets[i + 1] += csr.row_offsets[i];
    // 块排好序后 keys 整体就是 (行, rank, src) 升序，跟 row_offsets 一一对应
    csr.src.resize(static_cast<size_t>(n_edges));
    #pragma omp parallel for schedule(static) num_threads(T)
    for (int64_t e = 0; e < n_edges; ++e)
        csr.src[static_cast<size_t>(e)] = static_cast<uint32_t>(keys[static_cast<size_t>(e)]);
    return csr;
}

inline void merge_rank_ordered_into_row(
    uint32_t* row,                     // [K_out], forward row for node j; modified in place
    const uint32_t* rank_sorted_src,   // this row's CSR segment, rank-ascending
    uint32_t count,
    uint32_t K_out)
{
    const uint32_t num_protected = K_out / 2;
    uint32_t k = count;
    while (k > 0) {
        k--;
        uint32_t i = rank_sorted_src[k];

        uint32_t pos = K_out;
        for (uint32_t p = 0; p < K_out; ++p) {
            if (row[p] == i) { pos = p; break; }
        }
        if (pos < num_protected) continue;

        uint32_t num_shift = (pos == K_out) ? (K_out - num_protected - 1) : (pos - num_protected);
        for (uint32_t s = num_shift; s > 0; --s)
            row[num_protected + s] = row[num_protected + s - 1];
        row[num_protected] = i;
    }
}

}  // namespace edge_chunk
