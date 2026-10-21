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
  1 run each (user decision 2026-10-21; borderline 3–5% gains are re-run once). SIFT10M is not run in the loop for now (user decision 2026-10-21).
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

_(filled by iteration 0 — baseline)_

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

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
