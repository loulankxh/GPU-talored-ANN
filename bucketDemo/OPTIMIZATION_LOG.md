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

CAGRA reference (cuVS 26.10, NN-Descent, intermediate degree 64 → graph degree 32):

| dataset | build (s) | recall@10 itopk 32 / 64 / 128 / 256 / 512 |
|---|---|---|
| SIFT1M | 11.54 (median of 3) | 0.9305 / 0.9782 / 0.9948 / 0.9988 / 0.9998 |
| SIFT10M | 122.59 (1 run) | — |

**Goal:** faster SIFT1M-auto and SIFT1M-large at unchanged search quality; long-term target SIFT10M large-path build (all stages) ≤ 360 s (not measured in the loop for now).

Note: builds before 2026-10-21 02:00 had no `-O3` (CMake 4 left `CMAKE_BUILD_TYPE` empty); fixed, all numbers below are `-O3`.

## Current best

After **#0 (baseline)**, 2026-10-21:

| config | build total (s) | recall@10 itopk 32 / 64 / 128 | QPS itopk 32 / 64 |
|---|---|---|---|
| SIFT1M auto (GPU-resident) | **21.75** | 0.9293 / 0.9780 / 0.9946 | 1945K / 1036K |
| SIFT1M large (out-of-core path) | **75.10** | 0.8970 / 0.9622 / 0.9888 | 1996K / 1064K |
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

**Verdict:** baseline. Large path search quality is ~3.3 / 1.6 points below CAGRA at itopk 32 / 64 (pre-prune KNN quality is the same as the resident path, so the gap comes from `prune_windowed` + `optimize_chunked`).

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
