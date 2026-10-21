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

After **#1 (prune_windowed hot path: bucket-slot window + O(M²) detour count)**, 2026-10-21:

| config | build total (s) | recall@10 itopk 32 / 64 / 128 | QPS itopk 32 / 64 |
|---|---|---|---|
| SIFT1M auto (GPU-resident) | **21.36** | 0.9297 / 0.9779 / 0.9947 | 1945K / 1035K |
| SIFT1M large (out-of-core path) | **59.83** | 0.8971 / 0.9626 / 0.9888 | 1994K / 1065K |
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
- Skip the `neighbors.npy` conversion inside bucket2's "Write" step (the pipeline never reads it) or make it optional.
- SIFT10M Step 2 (centroid selection on the full 10M working set) is far slower than on 1M — sample before kmeans|| (cuVS kmeans uses a sampled init); compare `cpp/src/cluster/detail/kmeans_balanced.cuh` / `kmeans.cuh` init.
- Step 4.5 inline reorder and Step 6 `pool_build` / `scatter_pending` dominate the SIFT10M out-of-core loop — overlap host gather with GPU work (pinned staging, double buffering).
- Large path quality gap (prune_windowed vs cuVS optimize): SIFT1M recall@10 at itopk 32 is ~3 points below CAGRA — study `cpp/src/neighbors/detail/cagra/graph_core.cuh` (`optimize`: detour counting / rank-based pruning / reverse-edge merge) and align `prune_windowed_core.hpp` + `optimize_chunked` method E with it.
- `optimize` (resident) spends most of its time in file load + sanitize, not in the cuVS call.
- `prune_windowed` after #1: fetch (Belady cache load: per-bucket `read_rows` + u8→f32 dataset chunk, file reopened per call) = 3.7s of 6.3s, now the biggest part — keep handles open / read the whole bucket range in big sequential chunks, store vectors as u8 (4x less memory, also lets the cache hold 4x more on SIFT10M). Compute (2.2s) still forks an OMP region per bucket of ~100 rows on 80 threads.
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

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
