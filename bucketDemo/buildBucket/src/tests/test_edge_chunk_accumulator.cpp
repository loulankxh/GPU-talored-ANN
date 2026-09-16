// Standalone g++ unit test for edge_chunk_accumulator.hpp's CPU-only logic
// (no CUDA/raft/boost dependency -- can be compiled on a machine with no
// GPU toolchain at all):
//
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_eca test_edge_chunk_accumulator.cpp
//   /tmp/test_eca
//
// Covers:
//  1. chunk_of() binary search over chunk boundaries.
//  2. EdgeChunkAccumulator: add_records across multiple flushes + finish()
//     merge round-trips back to the exact same records per chunk.
//  3. build_rank_ordered_csr + merge_rank_ordered_into_row produces the
//     EXACT same final row as a direct port of optimize_chunked.cu's
//     existing merge_rev_into_output_chunk (optimize_chunked.cu:278) fed
//     candidates in "rank == original write order" -- i.e. the new
//     rank-based path is provably equivalent to the existing
//     write-order-based path in the case where they're defined to agree.
//  4. A truncation scenario (more reverse-edge candidates than available
//     slots) where the surviving set must be exactly the lowest-rank
//     (closest) candidates -- hand-verified expected result.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "../edge_chunk_accumulator.hpp"

using edge_chunk::EdgeRecord;
using edge_chunk::EdgeChunkAccumulator;
using edge_chunk::build_rank_ordered_csr;
using edge_chunk::merge_rank_ordered_into_row;
using edge_chunk::chunk_of;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------
// Direct port of optimize_chunked.cu:278 merge_rev_into_output_chunk's
// per-row inner loop, operating on candidates in a caller-supplied write
// order (index 0 = written first, back = written last -- matches the
// original's "process from the end of the write buffer backward" logic).
// Used only as a reference to prove the new rank-based path agrees with it.
// ---------------------------------------------------------------------
static void reference_merge_write_order(uint32_t* row, const uint32_t* write_order_src,
                                        uint32_t count, uint32_t K_out) {
    const uint32_t num_protected = K_out / 2;
    uint32_t cnt = count;
    while (cnt > 0) {
        cnt--;
        uint32_t i = write_order_src[cnt];
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

static void test_chunk_of() {
    std::vector<uint32_t> starts = {0, 5, 5, 12, 20, 20, 30};  // 6 chunks (some empty), N=30
    uint32_t num_chunks = static_cast<uint32_t>(starts.size()) - 1;
    // spot-check every j in [0, 30)
    for (uint32_t j = 0; j < 30; ++j) {
        uint32_t c = chunk_of(starts.data(), num_chunks, j);
        CHECK(starts[c] <= j && j < starts[c + 1]);
    }
    printf("test_chunk_of: %s\n", g_failures == 0 ? "ok" : "FAILED");
}

static void test_accumulator_roundtrip() {
    int before = g_failures;
    std::string dir = "/tmp/eca_test_dir";
    std::filesystem::create_directories(dir);

    const int32_t num_chunks = 4;
    // Tiny budget so we're forced through multiple flushes per chunk.
    auto acc = EdgeChunkAccumulator::create(dir, num_chunks, /*total_buffer_budget_bytes=*/200);
    acc.start();

    std::mt19937 rng(42);
    std::vector<std::vector<EdgeRecord>> expected(num_chunks);
    for (int32_t c = 0; c < num_chunks; ++c) {
        int n = 50 + c * 37;  // varying sizes, including forcing several flush cycles
        std::vector<EdgeRecord> batch;
        for (int i = 0; i < n; ++i) {
            EdgeRecord r{static_cast<uint32_t>(1000 + c), static_cast<uint32_t>(i),
                         static_cast<uint16_t>(i % 1024)};
            batch.push_back(r);
            expected[c].push_back(r);
        }
        // feed in a few unevenly-sized calls to exercise the multi-flush loop
        size_t off = 0;
        while (off < batch.size()) {
            size_t take = std::min<size_t>(batch.size() - off, 7);
            acc.add_records(c, batch.data() + off, take);
            off += take;
        }
    }
    acc.finish();

    for (int32_t c = 0; c < num_chunks; ++c) {
        std::vector<EdgeRecord> got;
        acc.read_chunk(c, got);
        CHECK(got.size() == expected[c].size());
        for (size_t i = 0; i < got.size() && i < expected[c].size(); ++i) {
            CHECK(got[i].dst == expected[c][i].dst);
            CHECK(got[i].src == expected[c][i].src);
            CHECK(got[i].rank == expected[c][i].rank);
        }
    }

    std::filesystem::remove_all(dir);
    printf("test_accumulator_roundtrip: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_csr_matches_reference_write_order() {
    int before = g_failures;
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> node_dist(0, 19);   // chunk_size = 20
    std::uniform_int_distribution<uint32_t> src_dist(1000, 9999);

    for (int trial = 0; trial < 200; ++trial) {
        const uint32_t chunk_size = 20;
        const uint32_t j_lo = 500;
        const uint32_t K_out = 8;
        const uint32_t num_protected = K_out / 2;

        // Random initial forward rows (base content before any merge), one
        // per node in the chunk, all distinct ids disjoint from candidate ids.
        std::vector<std::vector<uint32_t>> base_rows(chunk_size);
        for (uint32_t j = 0; j < chunk_size; ++j) {
            base_rows[j].resize(K_out);
            for (uint32_t k = 0; k < K_out; ++k) base_rows[j][k] = j * 100 + k;  // disjoint from src_dist range
        }

        // Random candidate edges targeting this chunk, "rank" = the order
        // in which we generate them per node (0,1,2,... in generation order)
        // so that "rank ascending" == "write order" and the two algorithms
        // are defined to agree.
        int n_edges = 5 + (trial % 15);
        std::vector<uint32_t> per_node_next_rank(chunk_size, 0);
        std::vector<EdgeRecord> edges;
        std::vector<std::vector<uint32_t>> write_order(chunk_size);  // per node, in generation order
        for (int e = 0; e < n_edges; ++e) {
            uint32_t local_j = node_dist(rng) % chunk_size;
            uint32_t src = src_dist(rng);
            uint32_t rank = per_node_next_rank[local_j]++;
            edges.push_back(EdgeRecord{j_lo + local_j, src, static_cast<uint16_t>(rank)});
            write_order[local_j].push_back(src);
        }

        auto csr = build_rank_ordered_csr(edges, j_lo, chunk_size);
        CHECK(csr.row_offsets.size() == chunk_size + 1);

        for (uint32_t j = 0; j < chunk_size; ++j) {
            uint32_t lo = csr.row_offsets[j], hi = csr.row_offsets[j + 1];
            CHECK(hi - lo == write_order[j].size());

            std::vector<uint32_t> row_new = base_rows[j];
            std::vector<uint32_t> row_ref = base_rows[j];

            merge_rank_ordered_into_row(row_new.data(), csr.src.data() + lo, hi - lo, K_out);
            reference_merge_write_order(row_ref.data(), write_order[j].data(),
                                        static_cast<uint32_t>(write_order[j].size()), K_out);

            for (uint32_t p = 0; p < K_out; ++p) CHECK(row_new[p] == row_ref[p]);
            (void)num_protected;
        }
    }
    printf("test_csr_matches_reference_write_order: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_truncation_keeps_lowest_rank() {
    int before = g_failures;
    const uint32_t K_out = 6;
    const uint32_t num_protected = 3;  // slots 0,1,2 protected; 3,4,5 competable

    uint32_t row[K_out] = {0, 1, 2, 3, 4, 5};

    // 5 candidates for 3 competable slots. rank 0 = strongest.
    // A=10(rank4) B=11(rank1) C=12(rank3) D=13(rank0) E=14(rank2)
    std::vector<EdgeRecord> edges = {
        EdgeRecord{100, 10, 4},
        EdgeRecord{100, 11, 1},
        EdgeRecord{100, 12, 3},
        EdgeRecord{100, 13, 0},
        EdgeRecord{100, 14, 2},
    };
    auto csr = build_rank_ordered_csr(edges, /*j_lo=*/100, /*chunk_size=*/1);
    CHECK(csr.row_offsets[1] - csr.row_offsets[0] == 5);
    // rank-ascending order should be D(13,r0), B(11,r1), E(14,r2), C(12,r3), A(10,r4)
    std::vector<uint32_t> expected_order = {13, 11, 14, 12, 10};
    for (size_t i = 0; i < expected_order.size(); ++i)
        CHECK(csr.src[i] == expected_order[i]);

    merge_rank_ordered_into_row(row, csr.src.data(), 5, K_out);

    std::set<uint32_t> survivors(row + num_protected, row + K_out);
    std::set<uint32_t> expected_survivors = {13, 11, 14};  // ranks 0,1,2 -> ids D,B,E
    CHECK(survivors == expected_survivors);
    // rank 3 (C=12) and rank 4 (A=10) must have been evicted
    CHECK(survivors.count(12) == 0);
    CHECK(survivors.count(10) == 0);
    // protected front untouched
    CHECK(row[0] == 0 && row[1] == 1 && row[2] == 2);

    printf("test_truncation_keeps_lowest_rank: %s\n", g_failures == before ? "ok" : "FAILED");
}

int main() {
    test_chunk_of();
    test_accumulator_roundtrip();
    test_csr_matches_reference_write_order();
    test_truncation_keeps_lowest_rank();

    if (g_failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    } else {
        printf("\n%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
