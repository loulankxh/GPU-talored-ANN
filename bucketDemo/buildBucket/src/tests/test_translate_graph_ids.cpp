// Standalone g++ unit test for translate_graph_ids.hpp (no CUDA dependency):
//
//   g++ -std=c++17 -O2 -fopenmp -Wall -Wextra -o /tmp/test_tgi test_translate_graph_ids.cpp
//   /tmp/test_tgi
//
// Covers: a synthetic graph in "new id" space + a random permutation,
// translated back to "old id" space, matches a naive in-memory reference
// (row destination and every value both go through inverse_perm), and the
// header (N, K_out) round-trips correctly.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "../translate_graph_ids.hpp"

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static void write_graph(const std::string& path, const std::vector<uint32_t>& graph,
                        int64_t N, int32_t K_out) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&N), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&K_out), sizeof(int32_t));
    out.write(reinterpret_cast<const char*>(graph.data()),
              static_cast<std::streamsize>(graph.size() * sizeof(uint32_t)));
}

static void write_inverse_perm(const std::string& path, const std::vector<uint32_t>& inv) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(inv.data()),
              static_cast<std::streamsize>(inv.size() * sizeof(uint32_t)));
}

static void test_translate_matches_naive_reference() {
    int before = g_failures;
    const int64_t N = 200;
    const int32_t K_out = 5;

    std::mt19937 rng(42);

    // random permutation: inverse_perm[new_id] = old_id
    std::vector<uint32_t> inverse_perm(static_cast<size_t>(N));
    std::iota(inverse_perm.begin(), inverse_perm.end(), 0u);
    std::shuffle(inverse_perm.begin(), inverse_perm.end(), rng);

    // random graph in new-id space (values are valid new ids, no self check needed for this test)
    std::uniform_int_distribution<uint32_t> id_dist(0, static_cast<uint32_t>(N - 1));
    std::vector<uint32_t> graph(static_cast<size_t>(N) * K_out);
    for (auto& v : graph) v = id_dist(rng);

    std::string graph_path = "/tmp/test_tgi_graph.bin";
    std::string perm_path = "/tmp/test_tgi_inv_perm.bin";
    std::string out_path = "/tmp/test_tgi_out.bin";
    write_graph(graph_path, graph, N, K_out);
    write_inverse_perm(perm_path, inverse_perm);

    translate_ids::translate_graph_to_original_order(graph_path, perm_path, out_path);

    // naive reference
    std::vector<uint32_t> expected(static_cast<size_t>(N) * K_out);
    for (int64_t new_id = 0; new_id < N; ++new_id) {
        uint32_t old_id = inverse_perm[static_cast<size_t>(new_id)];
        for (int32_t k = 0; k < K_out; ++k) {
            uint32_t new_neighbor = graph[static_cast<size_t>(new_id) * K_out + k];
            expected[static_cast<size_t>(old_id) * K_out + k] = inverse_perm[new_neighbor];
        }
    }

    std::ifstream in(out_path, std::ios::binary);
    int64_t hdrN = 0; int32_t hdrK = 0;
    in.read(reinterpret_cast<char*>(&hdrN), sizeof(int64_t));
    in.read(reinterpret_cast<char*>(&hdrK), sizeof(int32_t));
    CHECK(hdrN == N);
    CHECK(hdrK == K_out);
    std::vector<uint32_t> got(static_cast<size_t>(N) * K_out);
    in.read(reinterpret_cast<char*>(got.data()),
            static_cast<std::streamsize>(got.size() * sizeof(uint32_t)));
    CHECK(in.good());
    CHECK(got == expected);

    std::remove(graph_path.c_str());
    std::remove(perm_path.c_str());
    std::remove(out_path.c_str());

    printf("test_translate_matches_naive_reference: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_size_mismatch_throws() {
    int before = g_failures;
    std::vector<uint32_t> graph = {0, 1, 2, 3};  // N=2, K=2
    std::vector<uint32_t> bad_inv_perm = {0, 1, 2};  // wrong size (3, should be 2)

    std::string graph_path = "/tmp/test_tgi_bad_graph.bin";
    std::string perm_path = "/tmp/test_tgi_bad_perm.bin";
    std::string out_path = "/tmp/test_tgi_bad_out.bin";
    write_graph(graph_path, graph, 2, 2);
    write_inverse_perm(perm_path, bad_inv_perm);

    bool threw = false;
    try {
        translate_ids::translate_graph_to_original_order(graph_path, perm_path, out_path);
    } catch (const std::exception&) { threw = true; }
    CHECK(threw);

    std::remove(graph_path.c_str());
    std::remove(perm_path.c_str());
    std::remove(out_path.c_str());

    printf("test_size_mismatch_throws: %s\n", g_failures == before ? "ok" : "FAILED");
}

int main() {
    test_translate_matches_naive_reference();
    test_size_mismatch_throws();

    if (g_failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    } else {
        printf("\n%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
