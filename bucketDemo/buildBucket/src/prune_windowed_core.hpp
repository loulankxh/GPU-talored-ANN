// prune_windowed_core.hpp
// CPU-only core of prune_windowed: an approximate, memory-bounded stand-in
// for raft's exact CAGRA prune (raft/.../graph_core.cuh, kern_prune +
// "Create pruned kNN graph"). No CUDA dependency, so it's exercised by a
// standalone g++ test (tests/test_prune_windowed_core.cpp) and can also be
// run directly (multi-threaded via OpenMP in the caller) as prune_windowed's
// actual execution path -- see the note in prune_windowed.cu about why this
// ships CPU-only for now rather than as an untested CUDA kernel.
//
// Why "windowed" at all: raft's kern_prune decides whether edge A->B is
// prunable by checking, for each of A's own closer candidates D, whether D
// also connects to B (a 2-hop detour A->D->B). That requires reading D's
// *entire candidate row*, and D can be anywhere in the dataset -- raft's
// implementation copies the whole graph to device to make that access
// pattern cheap. At scale where the whole graph doesn't fit in host/GPU
// memory, this file instead resolves D's row through a bounded "window"
// (the caller populates it with the current bucket + its centroid-knn
// neighbor buckets, via bucket_order.hpp's BeladyBucketCache). A candidate D
// outside the window is treated as contributing zero detours -- this can
// only make a node's final graph *denser* than the exact algorithm would
// (an edge that should have been pruned survives), never incorrect: no edge
// gets dropped for a reason the exact algorithm wouldn't also drop it for.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace prune_windowed {

// Everything the detour/select computation needs for one point: its own
// M_in candidate ids (raw, -1 = padding, NOT necessarily distance-sorted --
// sort_row_by_l2 below produces the sorted copy) and its D-dim vector.
struct PointData {
    const int32_t* candidates;
    const float* vector;
};

// The visibility window for one processing position: global point id ->
// PointData, backed by whatever buckets the caller currently has resident
// (typically center bucket + its centroid-knn neighbors). O(1) average
// lookup; this is the CPU-side counterpart of the device kernel's sorted
// array + binary search (a hash map isn't available in device code).
class WindowLookup {
public:
    void clear() { map_.clear(); }
    void add(uint32_t global_id, const int32_t* candidates, const float* vector) {
        map_[global_id] = PointData{candidates, vector};
    }
    const PointData* find(uint32_t global_id) const {
        auto it = map_.find(global_id);
        return it == map_.end() ? nullptr : &it->second;
    }
    size_t size() const { return map_.size(); }
private:
    std::unordered_map<uint32_t, PointData> map_;
};

// Row-local L2 sort: point a_id's own M_in raw candidates, sorted ascending
// by squared distance to a_id. No cross-row dependency beyond needing each
// candidate's *vector* (not its candidate list) -- unlike the detour step,
// this never needs anything outside {a_id} union {a_id's own candidates},
// which by construction (bucket2's nprobe-bounded search) should almost
// always already be in the window. A candidate whose vector isn't resident
// is conservatively sorted to the back (never wrongly treated as "close").
inline void sort_row_by_l2(
    uint32_t a_id, const int32_t* a_candidates_raw, int32_t M_in, int32_t D,
    const WindowLookup& window,
    std::vector<int32_t>& out_sorted)   // resized to M_in, -1 = padding, ascending distance
{
    const PointData* a = window.find(a_id);
    if (!a)
        throw std::runtime_error("sort_row_by_l2: point " + std::to_string(a_id) +
                                 " itself not resident (bug: center bucket must always be loaded)");
    const float* a_vec = a->vector;

    struct Cand { float dist; int32_t id; };
    std::vector<Cand> cands;
    cands.reserve(static_cast<size_t>(M_in));
    for (int32_t k = 0; k < M_in; ++k) {
        int32_t id = a_candidates_raw[k];
        if (id < 0) continue;
        const PointData* c = window.find(static_cast<uint32_t>(id));
        float dist;
        if (c) {
            float acc = 0.f;
            for (int32_t x = 0; x < D; ++x) {
                float diff = a_vec[x] - c->vector[x];
                acc += diff * diff;
            }
            dist = acc;
        } else {
            dist = std::numeric_limits<float>::infinity();
        }
        cands.push_back(Cand{dist, id});
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& x, const Cand& y) { return x.dist < y.dist; });

    out_sorted.assign(static_cast<size_t>(M_in), -1);
    for (size_t k = 0; k < cands.size() && k < static_cast<size_t>(M_in); ++k)
        out_sorted[k] = cands[k].id;
}

// Mirrors graph_core.cuh's kern_prune: for point A's (already distance-
// sorted) candidates, count for each candidate B at rank kAB how many of
// A's closer candidates D (rank kAD < kAB) also have B in their own
// candidate row. D resolved through `window`; D outside it contributes 0
// (see file header for why that's conservative, not incorrect).
inline void compute_detour_counts(
    const int32_t* a_sorted_candidates, int32_t M_in,
    const WindowLookup& window,
    std::vector<uint32_t>& detour_count_out)   // resized to M_in
{
    detour_count_out.assign(static_cast<size_t>(M_in), 0);
    for (int32_t kAD = 0; kAD < M_in - 1; ++kAD) {
        int32_t d_id = a_sorted_candidates[kAD];
        if (d_id < 0) continue;
        const PointData* d = window.find(static_cast<uint32_t>(d_id));
        if (!d) continue;   // D outside window: no detour info, conservative skip

        for (int32_t kAB = kAD + 1; kAB < M_in; ++kAB) {
            int32_t b_id = a_sorted_candidates[kAB];
            if (b_id < 0) continue;
            bool found = false;
            for (int32_t kDB = 0; kDB < M_in; ++kDB) {
                if (d->candidates[kDB] == b_id) { found = true; break; }
            }
            if (found) detour_count_out[static_cast<size_t>(kAB)]++;
        }
    }
}

// Mirrors graph_core.cuh:425-458's selection: pick output_degree candidates
// with the smallest detour count, scanning in ascending-detour-count
// threshold order and, within one threshold, in existing (distance-sorted)
// rank order.
inline void select_pruned_row(
    const int32_t* a_sorted_candidates, int32_t M_in,
    const std::vector<uint32_t>& detour_count,
    int32_t output_degree,
    uint32_t* out_row)   // [output_degree]
{
    int32_t pk = 0;
    uint32_t num_detour = 0;
    while (pk < output_degree) {
        uint32_t next_num_detour = std::numeric_limits<uint32_t>::max();
        for (int32_t k = 0; k < M_in; ++k) {
            if (a_sorted_candidates[k] < 0) continue;
            uint32_t dk = detour_count[static_cast<size_t>(k)];
            if (dk > num_detour) next_num_detour = std::min(dk, next_num_detour);
            if (dk != num_detour) continue;
            out_row[pk++] = static_cast<uint32_t>(a_sorted_candidates[k]);
            if (pk >= output_degree) break;
        }
        if (pk >= output_degree) break;
        if (next_num_detour == std::numeric_limits<uint32_t>::max())
            throw std::runtime_error(
                "select_pruned_row: fewer than output_degree valid candidates "
                "(check --output-degree <= M_in and that padding was sanitized)");
        num_detour = next_num_detour;
    }
}

// Convenience: the three steps above, composed for one point.
inline void prune_one_point(
    uint32_t a_id, const int32_t* a_candidates_raw, int32_t M_in, int32_t D,
    const WindowLookup& window, int32_t output_degree,
    uint32_t* out_row)
{
    std::vector<int32_t> sorted;
    sort_row_by_l2(a_id, a_candidates_raw, M_in, D, window, sorted);
    std::vector<uint32_t> detour;
    compute_detour_counts(sorted.data(), M_in, window, detour);
    select_pruned_row(sorted.data(), M_in, detour, output_degree, out_row);
}

}  // namespace prune_windowed
