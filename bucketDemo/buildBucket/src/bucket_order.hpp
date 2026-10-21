// bucket_order.hpp
// DiskJoin-style bucket processing order + optimal offline (Belady) bucket
// cache, shared by:
//   - bucket_build.cu Step 4  (load order for computing per-vector KNN from
//     each bucket + its k nearest buckets)
//   - bucket.cu    Phase 7    (compute_bucket_reorder: which bucket gets
//     which contiguous new-ID range)
//   - reorder.cpp             (the standalone, post-hoc version of Phase 7)
//
// Reference: DiskJoin: Large-scale Vector Similarity Join with SSD (SIGMOD
// 2026), Sec 4.2 (Algorithm 1, Belady cache) and Sec 4.3 (Algorithm 2, task
// ordering via graph reordering). https://arxiv.org/abs/2508.18494
//
// The "bucket dependency graph" DiskJoin builds from triangle-inequality +
// probabilistic pruning is, in this project, already exactly the
// k-nearest-bucket graph computed in Step 3 (k_nearest_buckets /
// build_centroid_knn_on_gpu): adj[v] = bucket v's out-neighbors = its k
// nearest centroid buckets. No separate graph needs to be built - callers
// just pass that graph in.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace bucket_order {

// adj[v] = out-neighbor bucket ids of bucket v (self-loops must already be
// removed by the caller, as k_nearest_buckets already does).
using Adjacency = std::vector<std::vector<int32_t>>;

// ---------------------------------------------------------------------
// Algorithm 2 (DiskJoin Sec 4.3): Task Ordering via Graph Reordering
// ---------------------------------------------------------------------
//
// Greedily builds a permutation `order` of all B buckets: at each step, the
// next bucket is whichever unplaced bucket maximizes the overlap between its
// neighbor set and the neighbor sets of the last `window` placed buckets.
// Buckets placed close together in `order` therefore need largely the same
// neighbor buckets resident - exactly what makes a sliding cache (or the
// Belady cache below) effective, and exactly what makes bucket-aligned ID
// ranges in this order keep most graph edges inside one contiguous chunk.
//
// `window` corresponds to the paper's w = C / d_avg (cache capacity in
// bucket units, divided by average out-degree); callers pick it directly
// since here it doubles as "how many buckets a downstream consumer expects
// to keep resident together" (a GPU/CPU cache budget for Step 4, or a
// chunk size for optimize_chunked's method C/D).
//
// Complexity: O(B * K^2 * log(B)) for a K-regular graph, matching the
// paper's O(sum_v outdeg(v)^2) with an added log factor for the priority
// queue.
//
// ---------------------------------------------------------------------
// IndexedMaxHeap: a max-heap over an externally-owned score[] array where
// every bucket has at most one resident entry at a time, updated in place
// (sift up/down) instead of being blindly re-pushed on every score change.
// A naive std::priority_queue with lazy deletion (push a new entry every
// time a score changes, discard stale entries only when popped) lets the
// heap grow to O(B*K^2) total pushes over the whole run instead of staying
// bounded at O(B) resident entries -- at B~10^4, K~32 scale that's ~2*10^7
// pushes into an ever-growing heap (and often several stale pops before
// finding a live entry), which is the dominant real cost of
// compute_bucket_processing_order at that scale. This class keeps the same
// O(B*K^2*log(B)) asymptotic shape but with a much smaller constant
// factor, by capping the heap at the number of live (unvisited, touched)
// buckets.
// ---------------------------------------------------------------------
class IndexedMaxHeap {
public:
    // score must outlive this heap; score[v] is read on every comparison,
    // so the caller updates score[v] in place and then calls push/update.
    explicit IndexedMaxHeap(const std::vector<int64_t>& score)
        : score_(score), pos_(score.size(), -1) {}

    bool contains(int32_t v) const { return pos_[static_cast<size_t>(v)] >= 0; }
    bool empty() const { return heap_.empty(); }

    void push(int32_t v) {
        heap_.push_back(v);
        int32_t i = static_cast<int32_t>(heap_.size()) - 1;
        pos_[static_cast<size_t>(v)] = i;
        sift_up(i);
    }

    // score_[v] must already reflect the new value; sifts in whichever
    // direction is needed (only one of sift_up/sift_down actually moves
    // anything).
    void update(int32_t v) {
        int32_t i = pos_[static_cast<size_t>(v)];
        sift_up(i);
        sift_down(pos_[static_cast<size_t>(v)]);
    }

    void push_or_update(int32_t v) {
        if (contains(v)) update(v); else push(v);
    }

    // Pre: !empty().
    int32_t pop_max() {
        int32_t top = heap_[0];
        pos_[static_cast<size_t>(top)] = -1;
        int32_t last = heap_.back();
        heap_.pop_back();
        if (!heap_.empty()) {
            heap_[0] = last;
            pos_[static_cast<size_t>(last)] = 0;
            sift_down(0);
        }
        return top;
    }

private:
    // Same tie-break as the old std::pair<score,bucket> max-heap: higher
    // score wins; among equal scores, higher bucket id wins. Keeping this
    // identical makes the heap swap a pure performance change -- the
    // chosen `order` doesn't change.
    bool less(int32_t a, int32_t b) const {
        int64_t sa = score_[static_cast<size_t>(a)], sb = score_[static_cast<size_t>(b)];
        if (sa != sb) return sa < sb;
        return a < b;
    }

    void swap_at(int32_t i, int32_t j) {
        std::swap(heap_[static_cast<size_t>(i)], heap_[static_cast<size_t>(j)]);
        pos_[static_cast<size_t>(heap_[static_cast<size_t>(i)])] = i;
        pos_[static_cast<size_t>(heap_[static_cast<size_t>(j)])] = j;
    }

    void sift_up(int32_t i) {
        while (i > 0) {
            int32_t parent = (i - 1) / 2;
            if (!less(heap_[static_cast<size_t>(parent)], heap_[static_cast<size_t>(i)])) break;
            swap_at(parent, i);
            i = parent;
        }
    }

    void sift_down(int32_t i) {
        int32_t n = static_cast<int32_t>(heap_.size());
        for (;;) {
            int32_t l = 2 * i + 1, r = 2 * i + 2, largest = i;
            if (l < n && less(heap_[static_cast<size_t>(largest)], heap_[static_cast<size_t>(l)])) largest = l;
            if (r < n && less(heap_[static_cast<size_t>(largest)], heap_[static_cast<size_t>(r)])) largest = r;
            if (largest == i) break;
            swap_at(i, largest);
            i = largest;
        }
    }

    const std::vector<int64_t>& score_;
    std::vector<int32_t> pos_;    // bucket id -> index in heap_, -1 if absent
    std::vector<int32_t> heap_;   // bucket ids, heap-ordered by score_[]
};

inline std::vector<int32_t> compute_bucket_processing_order(
    const Adjacency& adj, int32_t window) {
    const int32_t B = static_cast<int32_t>(adj.size());
    std::vector<int32_t> order;
    order.reserve(B);
    if (B == 0) return order;
    window = std::max<int32_t>(1, window);

    // Reverse index: rev[u] = { v : u in adj[v] }. Used to find, when a
    // bucket enters/leaves the window, exactly which candidates' scores are
    // affected (only those sharing a neighbor with it) instead of rescanning
    // all B candidates every step.
    std::vector<std::vector<int32_t>> rev(B);
    for (int32_t v = 0; v < B; ++v)
        for (int32_t u : adj[v])
            if (u >= 0 && u < B) rev[u].push_back(v);

    // Start node = largest out-degree (ties -> smallest id), as in the paper.
    int32_t start = 0;
    for (int32_t v = 1; v < B; ++v)
        if (adj[v].size() > adj[start].size()) start = v;

    std::vector<char> visited(B, 0);
    std::vector<int64_t> score(B, 0);  // sum_{j in window} |N(P[j]) ^ N(v)|

    // Indexed max-heap over `score`: every bucket has at most one resident
    // entry (push_or_update sifts it in place instead of re-pushing), so
    // pop_max() always returns a live, up-to-date candidate directly - no
    // stale-entry filtering loop needed (see IndexedMaxHeap's comment).
    IndexedMaxHeap heap(score);

    auto bump = [&](int32_t entering_or_leaving, int64_t delta) {
        for (int32_t u : adj[entering_or_leaving]) {
            if (u < 0 || u >= B) continue;
            for (int32_t v : rev[u]) {
                if (visited[v]) continue;
                score[v] += delta;
                heap.push_or_update(v);
            }
        }
    };

    order.push_back(start);
    visited[start] = 1;
    bump(start, +1);

    for (int32_t i = 1; i < B; ++i) {
        // Slide the window forward: once it would exceed `window` buckets,
        // drop the oldest one's contribution.
        if (i > window) bump(order[i - window - 1], -1);

        // A visited bucket can never be in the heap (bump() never touches
        // one, and pop_max() below removes it from the heap the moment
        // it's picked), so whatever pop_max() returns is always live -
        // unlike the old lazy-deletion queue, no filtering loop is needed.
        int32_t pick = heap.empty() ? -1 : heap.pop_max();
        if (pick < 0) {
            // No scored candidate left (e.g. a disconnected component, or
            // score 0 everywhere and the heap ran dry of live entries) -
            // fall back to the smallest remaining id so `order` stays total.
            for (int32_t v = 0; v < B; ++v) {
                if (!visited[v]) {
                    pick = v;
                    break;
                }
            }
        }

        order.push_back(pick);
        visited[pick] = 1;
        bump(pick, +1);
    }

    return order;
}

// Convenience overload for graphs stored as flat row-major (B x K) arrays
// with -1/invalid padding, e.g. bucket.cu's `centroid_knn_graph_host`
// (uint32_t, no padding but may contain self at index; self is filtered).
inline Adjacency adjacency_from_flat_graph(
    const uint32_t* graph, int64_t B, int32_t K) {
    Adjacency adj(static_cast<size_t>(B));
    for (int64_t v = 0; v < B; ++v) {
        auto& row = adj[static_cast<size_t>(v)];
        row.reserve(K);
        for (int32_t k = 0; k < K; ++k) {
            uint32_t nb = graph[static_cast<size_t>(v) * K + k];
            if (static_cast<int64_t>(nb) != v && static_cast<int64_t>(nb) < B) {
                row.push_back(static_cast<int32_t>(nb));
            }
        }
    }
    return adj;
}

// Convenience overload for the vector<vector<int64_t>> shape returned by
// k_nearest_buckets in bucket_build.cu.
inline Adjacency adjacency_from_nested(
    const std::vector<std::vector<int64_t>>& knb) {
    Adjacency adj(knb.size());
    for (size_t v = 0; v < knb.size(); ++v) {
        adj[v].assign(knb[v].begin(), knb[v].end());
    }
    return adj;
}

// ---------------------------------------------------------------------
// Future access lists + Algorithm 1 (DiskJoin Sec 4.2): Belady's optimal
// offline cache.
// ---------------------------------------------------------------------

// For a fixed processing `order` (order[pos] = bucket id processed at step
// pos) and its adjacency (adj[b] = the extra buckets that must be resident
// while processing bucket b), returns, for every bucket id, the ascending
// list of future positions at which it will be needed - either as the
// bucket being processed, or as a neighbor of the bucket being processed.
inline std::vector<std::vector<int32_t>> compute_future_access_lists(
    const std::vector<int32_t>& order, const Adjacency& adj) {
    const int32_t B = static_cast<int32_t>(order.size());
    std::vector<std::vector<int32_t>> future(B);
    for (int32_t pos = 0; pos < B; ++pos) {
        int32_t b = order[pos];
        future[b].push_back(pos);
        for (int32_t nb : adj[b]) {
            if (nb >= 0 && nb < B) future[nb].push_back(pos);
        }
    }
    for (auto& v : future) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    return future;
}

// Byte-budgeted cache keyed by bucket id, evicting via Belady's algorithm:
// when room is needed, evict whichever resident bucket's next use (per
// `future_access`, relative to the position last accessed) is farthest in
// the future - or never used again, if any. This is optimal given the
// whole future access sequence is known ahead of time, which holds here
// because `order` (and hence every bucket's future access positions) is
// fixed before Step 4 starts.
//
// Usage contract: call get(position, bucket_id, loader) with `position`
// strictly non-decreasing across calls, and with (position, bucket_id)
// pairs matching what compute_future_access_lists was built from. Within
// one position, copy the returned value out before the next get() call for
// a *different* bucket id at that same position - the reference is only
// guaranteed valid until the next get() call, since a later miss can evict
// it once its next use has been consumed.
//
// Storage is indexed directly by bucket_id (0..B-1, B = future_access.size()),
// not hashed: bucket ids here are always a dense small-integer range, so a
// flat array lookup replaces what used to be several std::unordered_map
// accesses per get() call (on *every* call, hit or miss - book-keeping for
// the eviction-order tracking, not just the cache lookup itself). At small
// scale, where the whole working set fits under capacity_bytes_ and nothing
// ever gets evicted, that per-call hashing was pure overhead; at the actual
// target scale, where eviction is genuinely happening, the flat arrays are
// still strictly cheaper than hashing for the same book-keeping - this
// isn't a fast path that only helps the easy case, it helps both.
//
// Picking "the resident bucket needed farthest in the future" needs an
// ordered structure, but it doesn't need a node-based tree: eviction order
// is tracked with the same IndexedMaxHeap used by
// compute_bucket_processing_order (max-heap over next_use_, by bucket id),
// so advance_pointer()'s per-call update - run on *every* get(), hit or
// miss - is an in-place sift instead of a std::multimap erase+insert (two
// node allocations per call). Same fix as that function's heap, same
// reason: the structure itself was never O(B^2), but a tree that reallocates
// on every touch has a much worse constant factor than an array-backed
// heap at this call volume.
template <typename BucketT>
class BeladyBucketCache {
public:
    using SizeFn = std::function<size_t(const BucketT&)>;

    BeladyBucketCache(std::vector<std::vector<int32_t>> future_access,
                       size_t capacity_bytes, SizeFn size_fn)
        : future_access_(std::move(future_access)),
          capacity_bytes_(capacity_bytes),
          size_fn_(std::move(size_fn)),
          B_(static_cast<int32_t>(future_access_.size())),
          next_ptr_(static_cast<size_t>(B_), 0),
          resident_(static_cast<size_t>(B_), 0),
          slot_(static_cast<size_t>(B_)),
          byte_size_(static_cast<size_t>(B_), 0),
          next_use_(static_cast<size_t>(B_), 0),
          next_use_heap_(next_use_) {}

    template <typename Loader>
    const BucketT& get(int32_t position, int32_t bucket_id, Loader&& loader) {
        if (resident_[bucket_id]) {
            advance_pointer(bucket_id, position);
            return slot_[bucket_id];
        }

        BucketT loaded = loader(bucket_id);
        size_t bytes = size_fn_(loaded);
        make_room(bytes);

        slot_[bucket_id] = std::move(loaded);
        resident_[bucket_id] = 1;
        ++resident_count_;
        used_bytes_ += bytes;
        byte_size_[bucket_id] = bytes;
        advance_pointer(bucket_id, position);
        return slot_[bucket_id];
    }

    size_t used_bytes() const { return used_bytes_; }
    size_t resident_count() const { return resident_count_; }
    // Monotonic count of evictions so far. A caller holding references from
    // get() can check it is unchanged to know none of them was invalidated
    // (only an eviction ever releases a slot).
    uint64_t evictions() const { return evictions_; }

private:
    void advance_pointer(int32_t bucket_id, int32_t position) {
        auto& ptr = next_ptr_[bucket_id];
        const auto& fa = future_access_[bucket_id];
        while (ptr < fa.size() && fa[ptr] <= position) ++ptr;
        int32_t next_use = (ptr < fa.size())
            ? fa[ptr]
            : std::numeric_limits<int32_t>::max();

        // Every resident bucket has at most one entry in the heap; this is
        // called on EVERY get() (hit or miss), so it must stay allocation-
        // free - push_or_update sifts the existing entry in place instead
        // of erase+insert into a node-based tree (see IndexedMaxHeap).
        next_use_[bucket_id] = next_use;
        next_use_heap_.push_or_update(bucket_id);
    }

    void make_room(size_t incoming_bytes) {
        while (resident_count_ > 0 && used_bytes_ + incoming_bytes > capacity_bytes_) {
            // Evict whichever resident bucket is needed farthest in the
            // future (or never again, sentinel = INT32_MAX) - i.e. Belady's
            // optimal offline policy. Ties broken by higher bucket id;
            // eviction choice among ties only affects which bucket gets
            // reloaded later, never the correctness of what get() returns.
            int32_t victim = next_use_heap_.pop_max();
            used_bytes_ -= byte_size_[victim];
            byte_size_[victim] = 0;
            slot_[victim] = BucketT();  // release its heap memory, not just the accounting
            resident_[victim] = 0;
            --resident_count_;
            ++evictions_;
        }
        // If a single bucket's bytes alone exceed capacity_bytes_, the loop
        // above empties the cache and still can't make room - we insert it
        // anyway (used_bytes_ temporarily over budget for that one entry)
        // rather than refusing to cache; the next miss will evict it first.
    }

    std::vector<std::vector<int32_t>> future_access_;
    size_t capacity_bytes_;
    size_t used_bytes_ = 0;
    SizeFn size_fn_;

    int32_t B_;
    std::vector<size_t> next_ptr_;
    std::vector<char> resident_;
    std::vector<BucketT> slot_;
    std::vector<size_t> byte_size_;
    std::vector<int64_t> next_use_;   // bucket id -> current next-use position
    IndexedMaxHeap next_use_heap_;    // max-heap over next_use_, by bucket id
    size_t resident_count_ = 0;
    uint64_t evictions_ = 0;
};

}  // namespace bucket_order
