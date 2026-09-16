// Standalone g++ unit test for prune_windowed_core.hpp (no CUDA/raft/boost):
//
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_pwc test_prune_windowed_core.cpp
//   /tmp/test_pwc
//
// Covers:
//  1. sort_row_by_l2 produces ascending-distance order on a hand-checkable
//     1D layout.
//  2. select_pruned_row picks the lowest-detour-count candidates, ties
//     broken by existing rank order.
//  3. compute_detour_counts fed a WindowLookup covering *every* point
//     matches a fully independent, direct-array-indexed reference
//     implementation (this is the real correctness check of the windowing
//     mechanism itself, not a tautology: the reference never touches
//     WindowLookup at all).
//  4. Monotonicity / conservatism: for randomly restricted windows (some
//     points not resident), the windowed detour count for every candidate
//     is <= the naive/global detour count -- i.e. the approximation can
//     only under-prune (keep an edge it shouldn't have), never over-prune
//     (drop an edge the exact algorithm would have kept).

#include <cassert>
#include <cstdio>
#include <random>
#include <vector>

#include "../prune_windowed_core.hpp"

using prune_windowed::WindowLookup;
using prune_windowed::sort_row_by_l2;
using prune_windowed::compute_detour_counts;
using prune_windowed::select_pruned_row;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Direct array-indexed reference, deliberately NOT using WindowLookup at
// all, so agreement with compute_detour_counts(window=full) is a real test
// of the windowing mechanism rather than circular.
static void naive_detour_counts(
    const std::vector<int32_t>& knn_sorted, int32_t M_in, int32_t a_id,
    std::vector<uint32_t>& out)
{
    out.assign(static_cast<size_t>(M_in), 0);
    const int32_t* a_row = knn_sorted.data() + static_cast<size_t>(a_id) * M_in;
    for (int32_t kAD = 0; kAD < M_in - 1; ++kAD) {
        int32_t d_id = a_row[kAD];
        if (d_id < 0) continue;
        const int32_t* d_row = knn_sorted.data() + static_cast<size_t>(d_id) * M_in;
        for (int32_t kAB = kAD + 1; kAB < M_in; ++kAB) {
            int32_t b_id = a_row[kAB];
            if (b_id < 0) continue;
            for (int32_t kDB = 0; kDB < M_in; ++kDB) {
                if (d_row[kDB] == b_id) { out[static_cast<size_t>(kAB)]++; break; }
            }
        }
    }
}

static void test_sort_row_by_l2() {
    int before = g_failures;
    // 5 points on a 1D line (D=1): ids 0..4 at positions 0,10,20,30,40.
    const int32_t D = 1;
    std::vector<std::vector<float>> vecs = {{0}, {10}, {20}, {30}, {40}};
    WindowLookup w;
    std::vector<int32_t> dummy_cand(4, -1);  // not used for this test
    for (uint32_t i = 0; i < 5; ++i) w.add(i, dummy_cand.data(), vecs[i].data());

    // Point 2 (pos=20)'s raw (unsorted) candidate list: [4, 0, 3, 1]
    // distances: to 4(40)=400, to 0(0)=400, to 3(30)=100, to 1(10)=100
    // expect ascending: two at dist 100 first (3,1 in original relative
    // order since std::sort is not required stable here, but our impl uses
    // std::sort which is not stable -- so just check the *set* + grouping).
    std::vector<int32_t> raw = {4, 0, 3, 1};
    std::vector<int32_t> sorted;
    sort_row_by_l2(2, raw.data(), 4, D, w, sorted);
    CHECK(sorted.size() == 4);
    // first two entries (closest) must be {3,1} as a set, last two {4,0}.
    CHECK((sorted[0] == 3 || sorted[0] == 1));
    CHECK((sorted[1] == 3 || sorted[1] == 1));
    CHECK(sorted[0] != sorted[1]);
    CHECK((sorted[2] == 4 || sorted[2] == 0));
    CHECK((sorted[3] == 4 || sorted[3] == 0));

    printf("test_sort_row_by_l2: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_select_pruned_row() {
    int before = g_failures;
    std::vector<int32_t> cand = {10, 11, 12, 13, 14};
    std::vector<uint32_t> detour = {0, 0, 1, 1, 2};
    uint32_t out[3];
    select_pruned_row(cand.data(), 5, detour, 3, out);
    // lowest detour (0,0) at ranks 0,1 -> ids 10,11; next threshold (1) at
    // rank 2 -> id 12. Order within a threshold follows rank order.
    CHECK(out[0] == 10);
    CHECK(out[1] == 11);
    CHECK(out[2] == 12);
    printf("test_select_pruned_row: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void build_random_graph(int32_t N, int32_t M_in, uint32_t seed,
                               std::vector<int32_t>& knn_sorted) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> id_dist(0, N - 1);
    knn_sorted.assign(static_cast<size_t>(N) * M_in, -1);
    for (int32_t i = 0; i < N; ++i) {
        // M_in distinct candidates != i (simple rejection sampling, N small in tests)
        std::vector<int32_t> picked;
        while (static_cast<int32_t>(picked.size()) < M_in) {
            int32_t c = id_dist(rng);
            if (c == i) continue;
            bool dup = false;
            for (int32_t p : picked) if (p == c) { dup = true; break; }
            if (!dup) picked.push_back(c);
        }
        for (int32_t k = 0; k < M_in; ++k)
            knn_sorted[static_cast<size_t>(i) * M_in + k] = picked[static_cast<size_t>(k)];
    }
}

static void test_windowed_matches_naive_when_full() {
    int before = g_failures;
    const int32_t N = 40, M_in = 6;
    std::vector<int32_t> knn_sorted;
    build_random_graph(N, M_in, /*seed=*/1, knn_sorted);

    WindowLookup full;
    std::vector<float> dummy_vec(1, 0.f);
    for (int32_t i = 0; i < N; ++i)
        full.add(static_cast<uint32_t>(i), knn_sorted.data() + static_cast<size_t>(i) * M_in,
                 dummy_vec.data());

    for (int32_t a = 0; a < N; ++a) {
        std::vector<uint32_t> naive, windowed;
        naive_detour_counts(knn_sorted, M_in, a, naive);
        compute_detour_counts(knn_sorted.data() + static_cast<size_t>(a) * M_in, M_in, full, windowed);
        CHECK(naive.size() == windowed.size());
        for (size_t k = 0; k < naive.size(); ++k)
            CHECK(naive[k] == windowed[k]);
    }
    printf("test_windowed_matches_naive_when_full: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_restricted_window_is_conservative() {
    int before = g_failures;
    const int32_t N = 60, M_in = 8;
    std::vector<int32_t> knn_sorted;
    build_random_graph(N, M_in, /*seed=*/2, knn_sorted);

    std::vector<float> dummy_vec(1, 0.f);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    for (int trial = 0; trial < 30; ++trial) {
        // Randomly restrict the window to a subset of points (always
        // including point 0's own row set isn't required here since we
        // only call compute_detour_counts, not sort/select, and it doesn't
        // require A itself to be resident).
        WindowLookup restricted;
        double keep_prob = 0.3 + 0.6 * coin(rng);  // vary how restrictive
        for (int32_t i = 0; i < N; ++i) {
            if (coin(rng) < keep_prob)
                restricted.add(static_cast<uint32_t>(i),
                               knn_sorted.data() + static_cast<size_t>(i) * M_in, dummy_vec.data());
        }

        for (int32_t a = 0; a < N; a += 3) {  // sample every 3rd point to keep this fast
            std::vector<uint32_t> naive, windowed;
            naive_detour_counts(knn_sorted, M_in, a, naive);
            compute_detour_counts(knn_sorted.data() + static_cast<size_t>(a) * M_in, M_in, restricted, windowed);
            for (size_t k = 0; k < naive.size(); ++k)
                CHECK(windowed[k] <= naive[k]);
        }
    }
    printf("test_restricted_window_is_conservative: %s\n", g_failures == before ? "ok" : "FAILED");
}

int main() {
    test_sort_row_by_l2();
    test_select_pruned_row();
    test_windowed_matches_naive_when_full();
    test_restricted_window_is_conservative();

    if (g_failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    } else {
        printf("\n%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
