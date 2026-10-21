# bucket2 pipeline — optimization log

Pipeline: `bucket2` (bucketing → bucket order → bucket-pair per-vector KNN, degree 64)
→ prune + reverse edges to degree 32 (`optimize` resident / `prune_windowed` + `optimize_chunked -m E`)
→ `translate_ids` back to original order.

Environment: cuVS / raft / rmm **26.10** (conda `cuvs-bench`), gcc 14 (conda `cuvs-build`),
Tesla V100-SXM2-16GB, **GPU 0 only**. Reference source for ideas: cuVS 26.12 (`/home/lanlu/cuvs`).
Driver: `/optimize-bucket2` skill (`.claude/skills/optimize-bucket2/`), judged by `scripts/judge.py`.

Measured per iteration:
- **build time**: wall clock of every stage (process start, all I/O included), summed.
  SIFT1M-auto (GPU-resident path) and SIFT1M-large (large-dataset path forced on SIFT1M),
  median of 3 each. SIFT10M is not run in the loop for now (user decision 2026-10-21).
- **(a) pre-prune index quality**: bucket2's degree-64 `vector_knn` vs the exact KNN graph
  (recall@k on 10K sampled base points).
- **(b) final search quality**: the final degree-32 graph (original ids) searched with cuVS
  CAGRA search (same searcher as for CAGRA's own graph), recall@10 + QPS at itopk 32…512,
  10K queries, SIFT1M.

Quality gate (user, 2026-10-21): recall@10 at every itopk within **1%** of CAGRA for the
GPU-resident path and within **5%** of CAGRA for the large-dataset path — compared with
CAGRA only (not with the previous best). The pre-prune KNN similarity (a) is recorded, not gated.

CAGRA reference (cuVS 26.10, NN-Descent, intermediate degree 64 → graph degree 32):

| dataset | build (s) | recall@10 itopk 32 / 64 / 128 / 256 / 512 |
|---|---|---|
| SIFT1M | 11.54 (median of 3) | 0.9305 / 0.9782 / 0.9948 / 0.9988 / 0.9998 |
| SIFT10M | 122.59 (1 run) | — |

**Goal:** faster SIFT1M-auto and SIFT1M-large at unchanged search quality; long-term target SIFT10M large-path build (all stages) ≤ 360 s (not measured in the loop for now).

Note: builds before 2026-10-21 02:00 had no `-O3` (CMake 4 left `CMAKE_BUILD_TYPE` empty); fixed, all numbers below are `-O3`.

## Current best

After **#4 (bucket2 `--skip-npy`: don't write the unused neighbors.npy)**, 2026-10-21:

| config | build total (s) | recall@10 itopk 32 / 64 / 128 | QPS itopk 32 / 64 |
|---|---|---|---|
| SIFT1M auto (GPU-resident) | **19.18** | 0.9303 / 0.9781 / 0.9948 | 1943K / 1035K |
| SIFT1M large (out-of-core path) | **55.44** | 0.8971 / 0.9619 / 0.9891 | 1996K / 949K |
| CAGRA native (reference) | 11.54 | 0.9305 / 0.9782 / 0.9948 | 1942K / 1036K |

## Ideas backlog

Priority (from the user, 2026-10-21):
- **Step 4 (resident path):** replace the per-round CAGRA centroid search (~0.63s/round) with an exact fp16 GEMM + top-2 kernel (the queries are already resident as fp16). Expected ≈ −2s on SIFT1M. Must stay exact and keep `--out-of-core` on the old path.
- **Step 6:** fuse GEMM + top-64 selection into one kernel so the bucket × pool distance matrix (~100 GB of device memory traffic per round) is never materialized; top-64 selection is still the biggest GPU cost. Reference: cuVS 26.12 `cpp/src/neighbors/detail/knn_brute_force.cuh` / fused L2 kNN (`fused_l2_knn`) and `cpp/src/neighbors/detail/faiss_select/` block select.

From the IVF-PQ CAGRA build (cuVS 26.12 `neighbors/detail/cagra/cagra_build.cuh::build_knn_graph`, `ivf_pq/ivf_pq_build.cuh`):
- Train centroids on a **fixed-size sample** (IVF-PQ trains balanced k-means on `kmeans_trainset_fraction` of the data) instead of kmeans|| over all 10M points (SIFT10M Step 2 = 265 s over 4 rounds).
- Batch assignment like IVF-PQ `extend`: GEMM vs centroids + select, with data streamed in large batches through **pinned host buffers and overlapping H2D / compute / D2H** (Step 4 on SIFT10M = 129 s).
- IVF-PQ's kNN-graph build searches the dataset against itself in big batches, using `refine` for exact re-ranking of candidates: an analogue for Step 6 is approximate candidate scoring (e.g. fp16 / lower precision) followed by an exact re-rank of the top candidates only.
- Overlap of search and host-side post-processing via a background thread (`build_knn_graph` uses a pinned double buffer): the out-of-core Step 6 `scatter_pending` + `pool_build` (≈100 s / round on SIFT10M) are host-bound and serial with the GPU.
- ACE (`detail/cagra/cagra_ace.cuh`): cuVS's own partitioned out-of-core CAGRA build — check its partition I/O and per-partition graph merge for ideas for `prune_windowed` / `optimize_chunked`.

Other:
- SIFT10M Step 2 (centroid selection on the full 10M working set) is far slower than on 1M — sample before kmeans|| (cuVS kmeans uses a sampled init); compare `cpp/src/cluster/detail/kmeans_balanced.cuh` / `kmeans.cuh` init.
- Step 4.5 inline reorder and Step 6 `pool_build` / `scatter_pending` dominate the SIFT10M out-of-core loop — overlap host gather with GPU work (pinned staging, double buffering).
- Large path quality gap (prune_windowed vs cuVS optimize): SIFT1M recall@10 at itopk 32 is ~3 points below CAGRA — study `cpp/src/neighbors/detail/cagra/graph_core.cuh` (`optimize`: detour counting / rank-based pruning / reverse-edge merge) and align `prune_windowed_core.hpp` + `optimize_chunked` method E with it.
- `optimize` (resident) spends most of its time in file load + sanitize, not in the cuVS call.
- `prune_windowed` after #2 (3.9s): compute 2.6s forks an OMP region per bucket of ~100 rows on 80 threads — process several positions per parallel region, or move the detour count to GPU (cuVS `kern_fused_prune`). Store cached vectors as u8 (4x less cache memory → fewer evictions on SIFT10M); keep the dataset handle open in the loader.
- Resident `optimize` (2.1s) — check load + sanitize vs the cuVS call.
- bucket2 still writes `vector_dists.bin` (256 MB on SIFT1M) that no later stage reads — same treatment as #4 (`--skip-dists`), check `knn_similarity.py` / eval don't need it first.
- bucket2 Step 6 (large path, 22–26s) is the largest single step of the large path.

## Tried & rejected

| # | idea | result | reason |
|---|---|---|---|

## Iterations

### #0 — baseline (2026-10-21, commit 1715d7d)
**Idea / rationale:** starting point after porting to cuVS 26.10 and fixing the missing `-O3`. Pipeline as committed; params as in `params.sh` (centroid ratio 0.01, knn-k 32, nprobe 250, M 64, T 4, output degree 32).
**Files:** —  **Params changed:** —

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 21.75 | — | — | 1.88x | bucket2 17.46, optimize 2.10, translate 2.15 | 4216 | 2.06 |
| sift1M_large | large | 75.10 | — | — | 6.51x | bucket2 49.75, prune 17.98, optimize_chunked 5.30, translate 2.20 | 1584 | 4.19 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.21 | 1.41 | 0.00 | 3.32 | 1.12 | 0.02 | 5.65 | 0.00 | 1.41 |
| sift1M_large | 0.56 | 4.81 | 1.91 | 0.00 | 8.53 | 2.15 | 0.03 | 26.10 | 2.18 | 2.26 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9929 | 0.9889 | 0.9863 | 0.9837 |
| sift1M_large | 0.9927 | 0.9892 | 0.9865 | 0.9838 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9293 / 1945K | 0.9780 / 1036K | 0.9946 / 507K | 0.9987 / 187K | 0.9998 / 68K |
| ours sift1M_large | 0.8970 / 1996K | 0.9622 / 1064K | 0.9888 / 529K | 0.9973 / 188K | 0.9994 / 67K |

**Verdict:** baseline. Both paths pass the quality gate (resident within 1% of CAGRA, large within 5%). Large path search quality is ~3.3 / 1.6 points below CAGRA at itopk 32 / 64 (pre-prune KNN quality is the same as the resident path, so the gap comes from `prune_windowed` + `optimize_chunked`).

### #1 — prune_windowed hot path: bucket-slot window + O(M²) detour count  (2026-10-21)
**Idea / rationale:** `prune_windowed` (18s on SIFT1M-large) spent 4.4s building a per-position `unordered_map` over every window row (serial), ~4.5s tearing it down (unaccounted wall time), and its detour counting was the O(M³) "for each (D, B) scan D's row" loop. Now: (1) `BucketWindow` — the reordered id space is contiguous per bucket, so the window is one slot per bucket, resolved by binary search over `bucket_offsets` (memory O(n_buckets), independent of N; out-of-range ids rejected); (2) detour counting puts A's sorted row in a small open-addressing table (id → rank) and streams each D's row once against it — O(M²) probes, like cuVS 26.12 `detail/cagra/graph_core.cuh::kern_fused_prune` which iterates D's row (kDB) and matches against A's ranks; a per-rank stamp keeps our exact "count each (D, B) once" semantics, so counts are identical (unit test `test_prune_windowed_core` vs the naive reference passes); (3) thread-local scratch instead of per-point vectors. Selection / window semantics unchanged.
**Files:** `src/prune_windowed_core.hpp`, `src/prune_windowed.cpp`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 21.36 | +1.8% | +1.8% | 1.85x | bucket2 17.06, optimize 2.09, translate 2.09 | 3984 | 2.05 |
| sift1M_large | large | 59.83 | +20.3% | +20.3% | 5.18x | bucket2 45.54, prune 6.81, optimize_chunked 5.15, translate 2.14 | 1670 | 4.15 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.73 | 3.23 | 1.12 | 0.00 | 3.21 | 1.14 | 0.02 | 5.65 | 0.00 | 1.43 |
| sift1M_large | 0.55 | 5.19 | 1.91 | 0.00 | 8.74 | 1.77 | 0.03 | 22.45 | 2.13 | 2.26 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9929 | 0.9897 | 0.9870 | 0.9844 |
| sift1M_large | 0.9925 | 0.9895 | 0.9866 | 0.9839 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9297 / 1945K | 0.9779 / 1035K | 0.9947 / 533K | 0.9987 / 189K | 0.9998 / 67K |
| ours sift1M_large | 0.8971 / 1994K | 0.9626 / 1065K | 0.9888 / 512K | 0.9974 / 182K | 0.9995 / 68K |

[exited with code 0]

prune_windowed breakdown (rep1): fetch 3.74s, window_build 0.04s (was 4.36), compute 2.24s (was 4.46), write 0.22s, wall 6.27s (was 18.0).

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +1.8% (untouched path, noise), 1M-large +20.3% (75.10 → 59.83 s).

### #2 — prune_windowed: optimistic pointer handoff from the Belady cache  (2026-10-21)
**Idea / rationale:** after #1 the prune fetch phase was 3.7–4.0s even with warm files. Cause: `prune_windowed` only handed out cache references once *every* bucket was resident (`safe_ref`), which happens near the last position, so nearly every position deep-copied all K+1 window blocks (~25 GB memcpy on SIFT1M although the cache never evicted anything). A reference from `BeladyBucketCache::get` is only invalidated by an eviction, so: the cache now exposes a monotonic `evictions()` counter; the window is fetched as pointers and re-fetched with defensive copies only if the counter moved during that position. Eviction policy unchanged. Same "don't copy, reference the resident buffer unless it can move" idea as cuVS 26.12 `detail/cagra/cagra_build.cuh` (host batches staged by view, copies only into pinned staging). Output verified byte-identical to the previous binary (cache 16000 MB: 0 evictions / 0 copy positions; cache 100 MB: 73645 evictions, 9166 copy positions, still byte-identical).
**Files:** `src/bucket_order.hpp`, `src/prune_windowed.cpp`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 21.30 | +0.3% | +2.1% | 1.85x | bucket2 17.14, optimize 2.09, translate 2.07 | 5544 | 2.06 |
| sift1M_large | large | 57.14 | +4.5% | +23.9% | 4.95x | bucket2 45.85, prune 3.95, optimize_chunked 5.23, translate 2.16 | 1482 | 4.29 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.73 | 3.21 | 1.14 | 0.00 | 3.35 | 1.13 | 0.02 | 5.64 | 0.00 | 1.44 |
| sift1M_large | 0.56 | 5.03 | 1.85 | 0.00 | 8.80 | 1.73 | 0.03 | 22.67 | 2.12 | 2.22 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9921 | 0.9895 | 0.9865 | 0.9841 |
| sift1M_large | 0.9933 | 0.9897 | 0.9867 | 0.9840 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9298 / 1946K | 0.9785 / 1036K | 0.9947 / 524K | 0.9989 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.8984 / 1995K | 0.9628 / 1066K | 0.9893 / 507K | 0.9974 / 188K | 0.9994 / 68K |

[exited with code 0]

prune_windowed breakdown: fetch 0.98s (was 3.74), compute 2.60s, wall 3.88s (was 6.27). Borderline gain → large config rerun (`runs/iter2/sift1M_large_b`): 56.79 s, +5.1% → also ACCEPT.

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +0.3% (untouched path), 1M-large +4.5% / +5.1% on rerun (59.83 → 57.14 s).

### #3 — translate_ids: bounded in-memory output window instead of per-row pwrite  (2026-10-21)
**Idea / rationale:** `translate_ids` took 2.1s on 1M × 32 because every row was one random 128-byte `pwrite` syscall (1M syscalls). Now destination rows [lo, hi) are scattered into an in-memory window (`--out-buffer-mb`, default 1024) while streaming the input graph, then written with one sequential write; graphs larger than the budget take ceil(bytes / budget) sequential passes over the input, so host memory stays bounded for the large path. Added bounds checks: neighbor id ≥ N → error (previously unchecked `inverse_perm[row[k]]`), and every window must receive exactly its row count (inverse_perm is a permutation). Same staging pattern as cuVS 26.12 `detail/cagra/cagra_build.cuh::build_knn_graph` (results gathered into a host buffer, flushed in large contiguous copies). Verified byte-identical to the previous output on SIFT1M at 1024 MB (1 pass, 0.47s, 138 MB RSS) and 7 MB (19 passes, 1.34s, 20 MB RSS); unit test `test_translate_graph_ids` passes.
**Files:** `src/translate_graph_ids.hpp`, `src/translate_ids.cpp`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 20.01 | +6.1% | +8.0% | 1.73x | bucket2 17.51, optimize 2.09, translate 0.41 | 3912 | 2.06 |
| sift1M_large | large | 56.36 | +1.4% | +25.0% | 4.88x | bucket2 46.37, prune 3.89, optimize_chunked 5.30, translate 0.44 | 1506 | 4.19 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.74 | 3.40 | 1.45 | 0.00 | 3.30 | 1.08 | 0.02 | 5.64 | 0.00 | 1.41 |
| sift1M_large | 0.56 | 4.56 | 1.86 | 0.00 | 8.66 | 2.15 | 0.03 | 23.20 | 2.04 | 2.22 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9924 | 0.9898 | 0.9865 | 0.9836 |
| sift1M_large | 0.9934 | 0.9889 | 0.9868 | 0.9842 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9300 / 1948K | 0.9779 / 1035K | 0.9947 / 524K | 0.9988 / 188K | 0.9998 / 68K |
| ours sift1M_large | 0.8988 / 2000K | 0.9620 / 1033K | 0.9885 / 512K | 0.9972 / 188K | 0.9994 / 68K |

[exited with code 0]

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +6.1% (21.30 → 20.01 s), 1M-large +1.4% (57.14 → 56.36 s; translate 2.16 → 0.44 s, offset by bucket2 noise).

### #4 — bucket2 `--skip-npy`: don't write the unused neighbors.npy  (2026-10-21)
**Idea / rationale:** bucket2's "Write" step converted `vector_knn.bin` into `neighbors.npy` (int64, 512 MB on SIFT1M) for Python-side evaluation; no pipeline stage reads it (optimize / prune_windowed / knn_similarity all read `vector_knn.bin`). New flag `--skip-npy` (default off, so standalone behaviour is unchanged) skips it on both the GPU-resident (`GpuResidentKnn::write_outputs`) and out-of-core (`convert_vector_knn_to_npy`) paths; `params.sh` sets `BUCKET2_EXTRA="--skip-npy"`. cuVS's CAGRA build likewise only emits the graph it needs. Write step: 1.41 → 0.76 s (auto), 2.22 → 1.09 s (large); bucket2 peak RSS on auto 2.06 → 1.58 GB (no int64 copy).
**Files:** `src/bucket.cu`  **Params changed:** `BUCKET2_EXTRA="--skip-npy"`

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 19.18 | +4.1% | +11.8% | 1.66x | bucket2 16.71, optimize 2.08, translate 0.35 | 4120 | 1.58 |
| sift1M_large | large | 55.44 | +1.6% | +26.2% | 4.80x | bucket2 46.03, prune 3.95, optimize_chunked 5.26, translate 0.45 | 1910 | 4.24 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.73 | 3.38 | 1.27 | 0.00 | 3.25 | 1.12 | 0.02 | 5.65 | 0.00 | 0.76 |
| sift1M_large | 0.55 | 5.12 | 1.79 | 0.00 | 9.30 | 1.76 | 0.03 | 23.21 | 2.61 | 1.09 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9934 | 0.9891 | 0.9864 | 0.9839 |
| sift1M_large | 0.9922 | 0.9894 | 0.9866 | 0.9840 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9303 / 1943K | 0.9781 / 1035K | 0.9948 / 495K | 0.9988 / 187K | 0.9997 / 68K |
| ours sift1M_large | 0.8971 / 1996K | 0.9619 / 949K | 0.9891 / 485K | 0.9974 / 189K | 0.9994 / 67K |

[exited with code 0]

Borderline auto gain → rerun (`runs/iter4/sift1M_auto_b`): 19.40 s, +3.05% → also ACCEPT.

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +4.1% / +3.05% on rerun (20.01 → 19.18 s), 1M-large +1.6% (56.36 → 55.44 s).

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
