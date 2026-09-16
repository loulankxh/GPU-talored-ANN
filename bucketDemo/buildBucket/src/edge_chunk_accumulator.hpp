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
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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
    static EdgeChunkAccumulator create(const std::string& dir, int32_t num_chunks,
                                        size_t total_buffer_budget_bytes) {
        EdgeChunkAccumulator acc;
        acc.dir = dir;
        acc.num_chunks = num_chunks;
        acc.merged_path = dir + "/edge_chunks_merged.bin";

        constexpr size_t kMinBufferBytes = 4096;
        size_t nb = static_cast<size_t>(std::max<int32_t>(1, num_chunks));
        size_t buf_bytes = std::max<size_t>(kMinBufferBytes, total_buffer_budget_bytes / nb);
        size_t cap_records = std::max<size_t>(1, buf_bytes / sizeof(EdgeRecord));

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
    void finish() {
        for (int32_t c = 0; c < num_chunks; ++c) {
            flush_chunk(c);
            bufs[static_cast<size_t>(c)].file.close();
        }

        merged_offset.assign(static_cast<size_t>(num_chunks), 0);
        merged_count.assign(static_cast<size_t>(num_chunks), 0);
        std::ofstream mout(merged_path, std::ios::binary | std::ios::trunc);
        if (!mout.is_open())
            throw std::runtime_error("EdgeChunkAccumulator: cannot open " + merged_path + " for write");

        int64_t pos = 0;
        std::vector<char> copy_buf;
        for (int32_t c = 0; c < num_chunks; ++c) {
            merged_offset[static_cast<size_t>(c)] = pos;
            int64_t cnt = bufs[static_cast<size_t>(c)].total_count;
            merged_count[static_cast<size_t>(c)] = cnt;
            if (cnt == 0) continue;

            std::string p = chunk_path(dir, c);
            size_t nbytes = static_cast<size_t>(cnt) * sizeof(EdgeRecord);
            copy_buf.resize(nbytes);
            {
                std::ifstream in(p, std::ios::binary);
                if (!in.is_open())
                    throw std::runtime_error("EdgeChunkAccumulator: cannot open " + p + " for merge");
                in.read(copy_buf.data(), static_cast<std::streamsize>(nbytes));
                if (!in.good())
                    throw std::runtime_error("EdgeChunkAccumulator: merge-read failed for chunk " + std::to_string(c));
            }
            mout.write(copy_buf.data(), static_cast<std::streamsize>(nbytes));
            if (!mout.good())
                throw std::runtime_error("EdgeChunkAccumulator: merge-write failed for chunk " + std::to_string(c));
            pos += static_cast<int64_t>(nbytes);

            std::filesystem::remove(p);
        }
        mout.close();

        merged_in.open(merged_path, std::ios::binary);
        if (!merged_in.is_open())
            throw std::runtime_error("EdgeChunkAccumulator: cannot open " + merged_path + " for read");
    }

    void read_chunk(int32_t c, std::vector<EdgeRecord>& out) {
        int64_t cnt = merged_count[static_cast<size_t>(c)];
        out.resize(static_cast<size_t>(cnt));
        if (cnt == 0) return;
        merged_in.seekg(static_cast<std::streamoff>(merged_offset[static_cast<size_t>(c)]));
        merged_in.read(reinterpret_cast<char*>(out.data()),
                       static_cast<std::streamsize>(static_cast<size_t>(cnt) * sizeof(EdgeRecord)));
        if (!merged_in.good())
            throw std::runtime_error("EdgeChunkAccumulator: read_chunk failed for chunk " + std::to_string(c));
    }
};

// ---------------------------------------------------------------------
// Phase 2, step 1: group one chunk's edges by destination node (CSR) and,
// within each node's segment, sort ascending by rank -- so index 0 is the
// closest (rank 0, highest-priority) candidate and index (count-1) is the
// weakest. merge_rank_ordered_into_row below then walks each segment from
// the end backward, which is exactly optimize_chunked.cu's existing
// "process weakest first, closest last so it wins the slot" loop, just
// fed from this sorted-by-rank order instead of raw write order.
//
// Deliberately NOT a K_out-bucket counting sort over the whole chunk: that
// would force resetting a K_out-sized histogram per node regardless of how
// many edges that node actually has, which is wasteful once K_out reaches
// the hundreds. Sorting each node's own (typically much smaller) segment
// independently costs O(edges_for_that_node * log(that count)), summing to
// O(E_c * log(avg segment size)) overall -- cheaper in the common case and
// with no K_out-sized fixed cost per node.
// ---------------------------------------------------------------------
struct ChunkCsr {
    std::vector<uint32_t> row_offsets;  // size chunk_size+1
    std::vector<uint32_t> src;          // size row_offsets.back(), rank-ascending per row
};

inline ChunkCsr build_rank_ordered_csr(const std::vector<EdgeRecord>& edges,
                                        uint32_t j_lo, uint32_t chunk_size)
{
    struct RankSrc { uint16_t rank; uint32_t src; };

    ChunkCsr csr;
    csr.row_offsets.assign(static_cast<size_t>(chunk_size) + 1, 0);
    for (const auto& e : edges) {
        uint32_t local_j = e.dst - j_lo;
        csr.row_offsets[static_cast<size_t>(local_j) + 1]++;
    }
    for (uint32_t i = 0; i < chunk_size; ++i)
        csr.row_offsets[i + 1] += csr.row_offsets[i];

    size_t total = csr.row_offsets[chunk_size];
    std::vector<RankSrc> tmp(total);
    std::vector<uint32_t> cursor(csr.row_offsets.begin(), csr.row_offsets.end() - 1);
    for (const auto& e : edges) {
        uint32_t local_j = e.dst - j_lo;
        tmp[cursor[local_j]++] = RankSrc{e.rank, e.src};
    }

    for (uint32_t i = 0; i < chunk_size; ++i) {
        uint32_t lo = csr.row_offsets[i], hi = csr.row_offsets[i + 1];
        if (hi - lo > 1) {
            std::sort(tmp.begin() + lo, tmp.begin() + hi,
                      [](const RankSrc& a, const RankSrc& b) { return a.rank < b.rank; });
        }
    }

    csr.src.resize(total);
    for (size_t p = 0; p < total; ++p) csr.src[p] = tmp[p].src;
    return csr;
}

// ---------------------------------------------------------------------
// Phase 2, step 2: single-row version of optimize_chunked.cu's
// merge_rev_into_output_chunk (optimize_chunked.cu:278). Same shift/protect
// math, but reads priority from `rank_sorted_src`'s physical order (index 0
// = strongest) instead of assuming write order encoded priority.
// ---------------------------------------------------------------------
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
