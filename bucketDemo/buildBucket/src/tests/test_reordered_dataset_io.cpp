// Standalone g++ unit test for reordered_dataset_io.hpp (no CUDA dependency):
//
//   g++ -std=c++17 -O2 -Wall -Wextra -o /tmp/test_rdio test_reordered_dataset_io.cpp
//   /tmp/test_rdio
//
// Covers:
//  1. ReorderedDatasetWriter, fed points in a shuffled (non-bucket-sorted)
//     scan order with a tiny buffer capacity (forcing many flushes),
//     produces a file byte-identical to a naive reference that holds two
//     full copies in memory and scatters directly (the "plan A" approach
//     from the design discussion) -- proving the buffered-pwrite path is a
//     correct alternative to it, not just a faster-looking one.
//  2. ReorderedDatasetBucketReader reads back exactly the records each
//     bucket should own, with ids_out = iota over that bucket's region.
//  3. Both under- and over-filling a bucket's region (a deliberately wrong
//     assignments[]/offsets pairing) are caught with an exception rather
//     than silently corrupting neighboring buckets' data.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

#include "../reordered_dataset_io.hpp"

using reordered_io::ReorderedDatasetWriter;
using reordered_io::ReorderedDatasetBucketReader;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

// Minimal stand-in for bucket.cu's compute_bucket_reorder: given
// assignments[old_id] = raw bucket id and a bucket processing order, derive
// offsets_new (indexed by position) and perm[old_id] = new_id.
static void compute_reorder_reference(
    const std::vector<int64_t>& assignments, int64_t N, int64_t n_buckets,
    const std::vector<int32_t>& bucket_order,
    std::vector<uint32_t>& offsets_new, std::vector<uint32_t>& perm,
    std::vector<int32_t>& pos_of)
{
    std::vector<int64_t> counts(static_cast<size_t>(n_buckets), 0);
    for (int64_t i = 0; i < N; ++i) counts[static_cast<size_t>(assignments[static_cast<size_t>(i)])]++;

    offsets_new.assign(static_cast<size_t>(n_buckets) + 1, 0);
    pos_of.assign(static_cast<size_t>(n_buckets), 0);
    std::vector<uint32_t> region_start(static_cast<size_t>(n_buckets), 0);
    uint32_t running = 0;
    for (int64_t pos = 0; pos < n_buckets; ++pos) {
        int32_t b = bucket_order[static_cast<size_t>(pos)];
        pos_of[static_cast<size_t>(b)] = static_cast<int32_t>(pos);
        region_start[static_cast<size_t>(b)] = running;
        running += static_cast<uint32_t>(counts[static_cast<size_t>(b)]);
        offsets_new[static_cast<size_t>(pos) + 1] = running;
    }

    perm.assign(static_cast<size_t>(N), 0);
    std::vector<uint32_t> cursor = region_start;
    for (int64_t old_id = 0; old_id < N; ++old_id) {
        int64_t b = assignments[static_cast<size_t>(old_id)];
        perm[static_cast<size_t>(old_id)] = cursor[static_cast<size_t>(b)]++;
    }
}

static void test_writer_matches_naive_reference() {
    int before = g_failures;
    // N large enough (relative to the buffer budget below, which floors at
    // 4096 bytes/bucket = 256 records for D=4 floats) that most buckets
    // exceed one buffer's worth and must flush multiple times before
    // finish(), not just once.
    const int64_t N = 3000;
    const int64_t D = 4;
    const int64_t n_buckets = 7;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int64_t> bucket_dist(0, n_buckets - 1);

    std::vector<int64_t> assignments(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) assignments[static_cast<size_t>(i)] = bucket_dist(rng);

    std::vector<int32_t> bucket_order = {3, 0, 5, 1, 6, 2, 4};  // arbitrary, not identity

    std::vector<uint32_t> offsets_new, perm;
    std::vector<int32_t> pos_of;
    compute_reorder_reference(assignments, N, n_buckets, bucket_order, offsets_new, perm, pos_of);

    // Synthetic vectors: point i's vector is [i, i+1000, i+2000, i+3000] so
    // reading it back identifies which original point landed where.
    std::vector<std::vector<float>> vecs(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i)
        vecs[static_cast<size_t>(i)] = {float(i), float(i + 1000), float(i + 2000), float(i + 3000)};

    // ---- naive reference: two full copies, direct scatter (plan A) ----
    std::vector<float> raw_out(static_cast<size_t>(N) * D);
    for (int64_t old_id = 0; old_id < N; ++old_id) {
        uint32_t new_id = perm[static_cast<size_t>(old_id)];
        std::memcpy(raw_out.data() + static_cast<size_t>(new_id) * D,
                    vecs[static_cast<size_t>(old_id)].data(), D * sizeof(float));
    }

    // ---- new writer: ascending old_id scan order (the required contract --
    // see reordered_dataset_io.hpp's header comment) with a small buffer to
    // force many flushes per bucket, not just one at finish(). ----
    std::string path = "/tmp/test_rdio_data.fbin";
    auto writer = ReorderedDatasetWriter<float>::create(
        path, D, n_buckets, offsets_new, pos_of, /*total_buffer_budget_bytes=*/n_buckets * D * sizeof(float) * 3);

    for (int64_t old_id = 0; old_id < N; ++old_id)
        writer.add(assignments[static_cast<size_t>(old_id)], vecs[static_cast<size_t>(old_id)].data());
    writer.finish();

    // ---- compare byte-for-byte ----
    std::ifstream in(path, std::ios::binary);
    int32_t hdrN = 0, hdrD = 0;
    in.read(reinterpret_cast<char*>(&hdrN), 4);
    in.read(reinterpret_cast<char*>(&hdrD), 4);
    CHECK(hdrN == N);
    CHECK(hdrD == D);
    std::vector<float> got(static_cast<size_t>(N) * D);
    in.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(got.size() * sizeof(float)));
    CHECK(in.good());
    CHECK(got == raw_out);

    std::remove(path.c_str());
    printf("test_writer_matches_naive_reference: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_reader_matches_offsets() {
    int before = g_failures;
    const int64_t N = 60, D = 3, n_buckets = 4;
    std::vector<int64_t> assignments(static_cast<size_t>(N));
    std::mt19937 rng(7);
    std::uniform_int_distribution<int64_t> bd(0, n_buckets - 1);
    for (int64_t i = 0; i < N; ++i) assignments[static_cast<size_t>(i)] = bd(rng);
    std::vector<int32_t> bucket_order = {2, 0, 3, 1};

    std::vector<uint32_t> offsets_new, perm;
    std::vector<int32_t> pos_of;
    compute_reorder_reference(assignments, N, n_buckets, bucket_order, offsets_new, perm, pos_of);

    std::vector<std::vector<float>> vecs(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i)
        vecs[static_cast<size_t>(i)] = {float(i), float(i * 2), float(i * 3)};

    std::string path = "/tmp/test_rdio_data2.fbin";
    auto writer = ReorderedDatasetWriter<float>::create(path, D, n_buckets, offsets_new, pos_of, 4096);
    for (int64_t i = 0; i < N; ++i) writer.add(assignments[static_cast<size_t>(i)], vecs[static_cast<size_t>(i)].data());
    writer.finish();

    ReorderedDatasetBucketReader<float> reader(path, D, offsets_new, pos_of);
    for (int64_t b = 0; b < n_buckets; ++b) {
        std::vector<int64_t> ids_out;
        std::vector<float> vecs_out;
        reader.read_bucket(b, ids_out, vecs_out);
        int32_t pos = pos_of[static_cast<size_t>(b)];
        uint32_t lo = offsets_new[static_cast<size_t>(pos)], hi = offsets_new[static_cast<size_t>(pos) + 1];
        CHECK(reader.count(b) == static_cast<int64_t>(hi - lo));
        CHECK(ids_out.size() == hi - lo);
        for (size_t k = 0; k < ids_out.size(); ++k)
            CHECK(ids_out[k] == static_cast<int64_t>(lo) + static_cast<int64_t>(k));
        // find which original point each returned new_id corresponds to and
        // verify the vector content matches.
        for (size_t k = 0; k < ids_out.size(); ++k) {
            uint32_t new_id = static_cast<uint32_t>(ids_out[k]);
            int64_t old_id = -1;
            for (int64_t i = 0; i < N; ++i)
                if (perm[static_cast<size_t>(i)] == new_id) { old_id = i; break; }
            CHECK(old_id >= 0);
            for (int64_t d = 0; d < D; ++d)
                CHECK(vecs_out[k * D + static_cast<size_t>(d)] == vecs[static_cast<size_t>(old_id)][static_cast<size_t>(d)]);
        }
    }

    std::remove(path.c_str());
    printf("test_reader_matches_offsets: %s\n", g_failures == before ? "ok" : "FAILED");
}

static void test_region_mismatch_throws() {
    int before = g_failures;
    const int64_t D = 2, n_buckets = 2;
    std::vector<uint32_t> offsets_new = {0, 3, 5};  // bucket at pos0 has 3 slots, pos1 has 2
    std::vector<int32_t> pos_of = {0, 1};           // raw bucket 0 -> pos0, raw bucket 1 -> pos1

    // Under-fill: only add 2 of the 3 records bucket 0 is supposed to get.
    {
        std::string path = "/tmp/test_rdio_bad1.fbin";
        auto writer = ReorderedDatasetWriter<float>::create(path, D, n_buckets, offsets_new, pos_of, 4096);
        float v[2] = {1, 2};
        writer.add(0, v);
        writer.add(0, v);
        writer.add(1, v);
        writer.add(1, v);
        bool threw = false;
        try { writer.finish(); } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::remove(path.c_str());
    }
    // Over-fill: add one more record to bucket 0 than its region allows.
    {
        std::string path = "/tmp/test_rdio_bad2.fbin";
        auto writer = ReorderedDatasetWriter<float>::create(path, D, n_buckets, offsets_new, pos_of, 4096);
        float v[2] = {1, 2};
        bool threw = false;
        try {
            for (int i = 0; i < 4; ++i) writer.add(0, v);  // region only has 3 slots
        } catch (const std::exception&) { threw = true; }
        CHECK(threw);
        std::remove(path.c_str());
    }

    printf("test_region_mismatch_throws: %s\n", g_failures == before ? "ok" : "FAILED");
}

int main() {
    test_writer_matches_naive_reference();
    test_reader_matches_offsets();
    test_region_mismatch_throws();

    if (g_failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    } else {
        printf("\n%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
