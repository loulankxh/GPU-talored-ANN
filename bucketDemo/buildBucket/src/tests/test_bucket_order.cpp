// Standalone g++ unit test for bucket_order.hpp (no CUDA dependency):
//
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_bucket_order test_bucket_order.cpp
//   /tmp/test_bucket_order
//
// Covers:
//  1. Differential correctness: compute_bucket_processing_order's new
//     IndexedMaxHeap-based implementation must produce the EXACT same
//     `order` as the old lazy-deletion std::priority_queue implementation
//     (kept here, renamed, as a reference) across random small/medium
//     graphs plus edge cases (empty, single node, isolated nodes,
//     disconnected components, duplicate edges).
//  2. compute_future_access_lists sanity check against the new order's
//     output (every bucket's own processing position must appear in its
//     own future-access list).
//  3. Benchmark: a synthetic B=10000/avg-degree~32 graph (with a handful
//     of in-degree hot nodes, mimicking a real KNN graph's skew) timing
//     old vs new -- informational only (hardware-dependent), but the new
//     implementation should never be slower.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "../bucket_order.hpp"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------
// Reference implementation: the OLD lazy-deletion std::priority_queue
// version of compute_bucket_processing_order, verbatim (pre-IndexedMaxHeap).
// Kept here only for differential testing against the new implementation.
// ---------------------------------------------------------------------
static std::vector<int32_t> compute_bucket_processing_order_reference(
    const bucket_order::Adjacency& adj, int32_t window) {
    const int32_t B = static_cast<int32_t>(adj.size());
    std::vector<int32_t> order;
    order.reserve(static_cast<size_t>(B));
    if (B == 0) return order;
    window = std::max<int32_t>(1, window);

    std::vector<std::vector<int32_t>> rev(static_cast<size_t>(B));
    for (int32_t v = 0; v < B; ++v)
        for (int32_t u : adj[static_cast<size_t>(v)])
            if (u >= 0 && u < B) rev[static_cast<size_t>(u)].push_back(v);

    int32_t start = 0;
    for (int32_t v = 1; v < B; ++v)
        if (adj[static_cast<size_t>(v)].size() > adj[static_cast<size_t>(start)].size()) start = v;

    std::vector<char> visited(static_cast<size_t>(B), 0);
    std::vector<int64_t> score(static_cast<size_t>(B), 0);

    using Entry = std::pair<int64_t, int32_t>;
    std::priority_queue<Entry> pq;

    auto bump = [&](int32_t entering_or_leaving, int64_t delta) {
        for (int32_t u : adj[static_cast<size_t>(entering_or_leaving)]) {
            if (u < 0 || u >= B) continue;
            for (int32_t v : rev[static_cast<size_t>(u)]) {
                if (visited[static_cast<size_t>(v)]) continue;
                score[static_cast<size_t>(v)] += delta;
                pq.push({score[static_cast<size_t>(v)], v});
            }
        }
    };

    order.push_back(start);
    visited[static_cast<size_t>(start)] = 1;
    bump(start, +1);

    for (int32_t i = 1; i < B; ++i) {
        if (i > window) bump(order[static_cast<size_t>(i - window - 1)], -1);

        int32_t pick = -1;
        while (!pq.empty()) {
            auto [s, v] = pq.top();
            pq.pop();
            if (visited[static_cast<size_t>(v)] || s != score[static_cast<size_t>(v)]) continue;
            pick = v;
            break;
        }
        if (pick < 0) {
            for (int32_t v = 0; v < B; ++v) {
                if (!visited[static_cast<size_t>(v)]) { pick = v; break; }
            }
        }

        order.push_back(pick);
        visited[static_cast<size_t>(pick)] = 1;
        bump(pick, +1);
    }

    return order;
}

static bucket_order::Adjacency make_random_adjacency(
    int32_t B, int32_t max_degree, std::mt19937& rng, double isolate_frac = 0.0) {
    bucket_order::Adjacency adj(static_cast<size_t>(B));
    std::uniform_int_distribution<int32_t> deg_dist(0, max_degree);
    std::uniform_int_distribution<int32_t> node_dist(0, std::max(0, B - 1));
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (int32_t v = 0; v < B; ++v) {
        if (B > 1 && coin(rng) < isolate_frac) continue;  // isolated node
        int32_t deg = deg_dist(rng);
        for (int32_t k = 0; k < deg; ++k) {
            int32_t u = node_dist(rng);
            if (u != v) adj[static_cast<size_t>(v)].push_back(u);  // dup edges allowed, matches real graphs
        }
    }
    return adj;
}

static void test_differential_random_graphs() {
    int before = g_failures;
    std::mt19937 rng(12345);
    struct Case { int32_t B, max_degree; double isolate_frac; };
    std::vector<Case> cases = {
        {1, 4, 0.0}, {2, 4, 0.0}, {5, 0, 0.0}, {50, 4, 0.0}, {50, 8, 0.2},
        {200, 16, 0.0}, {200, 16, 0.3}, {500, 32, 0.0}, {2000, 32, 0.05},
    };
    for (const auto& c : cases) {
        for (int32_t window : {1, 4, 16, 64}) {
            auto adj = make_random_adjacency(c.B, c.max_degree, rng, c.isolate_frac);
            auto expected = compute_bucket_processing_order_reference(adj, window);
            auto actual = bucket_order::compute_bucket_processing_order(adj, window);
            CHECK(actual == expected);
            if (actual != expected) {
                std::fprintf(stderr, "  mismatch at B=%d max_degree=%d isolate=%.2f window=%d\n",
                             c.B, c.max_degree, c.isolate_frac, window);
            }
        }
    }
    printf("test_differential_random_graphs: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_empty_graph() {
    int before = g_failures;
    bucket_order::Adjacency adj;
    auto order = bucket_order::compute_bucket_processing_order(adj, 16);
    CHECK(order.empty());
    printf("test_empty_graph: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_future_access_sanity() {
    int before = g_failures;
    std::mt19937 rng(999);
    auto adj = make_random_adjacency(300, 16, rng, 0.1);
    auto order = bucket_order::compute_bucket_processing_order(adj, 16);
    CHECK(order.size() == adj.size());

    auto future = bucket_order::compute_future_access_lists(order, adj);
    CHECK(future.size() == adj.size());
    for (int32_t pos = 0; pos < static_cast<int32_t>(order.size()); ++pos) {
        int32_t b = order[static_cast<size_t>(pos)];
        const auto& fb = future[static_cast<size_t>(b)];
        bool found = std::find(fb.begin(), fb.end(), pos) != fb.end();
        CHECK(found);
    }
    printf("test_future_access_sanity: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_benchmark_large_skewed_graph() {
    int before = g_failures;
    const int32_t B = 10000;
    const int32_t K = 32;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> node_dist(0, B - 1);

    // Mimic a real KNN graph: most nodes have out-degree K, but a handful
    // of "hot" nodes (picked disproportionately often as a neighbor) give
    // some nodes higher in-degree than K -- the skew this project's real
    // centroid-KNN graphs show. 50 hot nodes / 5% redirect chance lands at
    // max in-degree ~400 (vs baseline ~32), which empirically reproduces
    // the ~3s old-impl / ~0.2s new-impl timing seen on the real B=10000
    // K=32 centroid graph from an actual run -- a much milder, more
    // representative skew than an earlier draft of this benchmark (20 hot
    // nodes / 30% chance, max in-degree ~5000), which was so pathological
    // the old implementation took minutes rather than seconds.
    std::vector<int32_t> hot_nodes;
    for (int32_t i = 0; i < 50; ++i) hot_nodes.push_back(node_dist(rng));
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    bucket_order::Adjacency adj(static_cast<size_t>(B));
    for (int32_t v = 0; v < B; ++v) {
        adj[static_cast<size_t>(v)].reserve(static_cast<size_t>(K));
        for (int32_t k = 0; k < K; ++k) {
            int32_t u;
            if (coin(rng) < 0.05) {
                u = hot_nodes[static_cast<size_t>(node_dist(rng) % static_cast<int32_t>(hot_nodes.size()))];
            } else {
                u = node_dist(rng);
            }
            if (u != v) adj[static_cast<size_t>(v)].push_back(u);
        }
    }
    int32_t window = 128;

    auto t0 = std::chrono::steady_clock::now();
    auto expected = compute_bucket_processing_order_reference(adj, window);
    auto t1 = std::chrono::steady_clock::now();
    auto actual = bucket_order::compute_bucket_processing_order(adj, window);
    auto t2 = std::chrono::steady_clock::now();

    double old_s = std::chrono::duration<double>(t1 - t0).count();
    double new_s = std::chrono::duration<double>(t2 - t1).count();

    CHECK(actual == expected);
    CHECK(new_s <= old_s);
    printf("test_benchmark_large_skewed_graph: %s  (B=%d K=%d: old=%.3fs new=%.3fs, %.1fx)\n",
           g_failures == before ? "ok" : "FAILED", B, K, old_s, new_s,
           new_s > 0 ? old_s / new_s : 0.0);
}

int main() {
    test_differential_random_graphs();
    test_empty_graph();
    test_future_access_sanity();
    test_benchmark_large_skewed_graph();

    if (g_failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    } else {
        printf("\n%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
