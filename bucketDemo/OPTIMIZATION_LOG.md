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

After **#13 (out-of-core: last merge writes vector_knn/dists directly)**, 2026-10-21:

| config | build total (s) | recall@10 itopk 32 / 64 / 128 | QPS itopk 32 / 64 |
|---|---|---|---|
| SIFT1M auto (GPU-resident) | **16.17** | 0.9295 / 0.9772 / 0.9945 | 1943K / 1033K |
| SIFT1M large (out-of-core path) | **36.36** | 0.8961 / 0.9618 / 0.9882 | 2002K / 958K |
| CAGRA native (reference) | 11.54 | 0.9305 / 0.9782 / 0.9948 | 1942K / 1036K |

## Ideas backlog

Priority (from the user, 2026-10-21):
- **Step 4 (resident path):** done in #7 with fp32 `brute_force` (search 0.70 → 0.42 s/round). Remaining: an fp16/tensor-core GEMM with fp32 accumulation (cuVS half brute force — check it accumulates in fp32: uint8² × 128 overflows fp16) or a fused GEMM + top-2 kernel that never writes the 990K × 10K distance tile.
- **Step 6:** fuse GEMM + top-64 selection into one kernel so the bucket × pool distance matrix (~100 GB of device memory traffic per round) is never materialized; top-64 selection is still the biggest GPU cost. Reference: cuVS 26.12 `cpp/src/neighbors/detail/knn_brute_force.cuh` / fused L2 kNN (`fused_l2_knn`) and `cpp/src/neighbors/detail/faiss_select/` block select.

From the IVF-PQ CAGRA build (cuVS 26.12 `neighbors/detail/cagra/cagra_build.cuh::build_knn_graph`, `ivf_pq/ivf_pq_build.cuh`):
- Train centroids on a **fixed-size sample** (IVF-PQ trains balanced k-means on `kmeans_trainset_fraction` of the data) instead of kmeans|| over all 10M points (SIFT10M Step 2 = 265 s over 4 rounds).
- Batch assignment like IVF-PQ `extend`: GEMM vs centroids + select (done in #7/#8), still to do: stream data in batches through **pinned host buffers overlapping read / H2D / compute** (Step 4 out-of-core after #8: read 0.13 s + search 0.42 s per round on SIFT1M, serial; SIFT10M Step 4 was 129 s before #8).
- IVF-PQ's kNN-graph build searches the dataset against itself in big batches, using `refine` for exact re-ranking of candidates: an analogue for Step 6 is approximate candidate scoring (e.g. fp16 / lower precision) followed by an exact re-rank of the top candidates only.
- Overlap of search and host-side post-processing via a background thread (`build_knn_graph` uses a pinned double buffer): the out-of-core Step 6 `scatter_pending` + `pool_build` (≈100 s / round on SIFT10M) are host-bound and serial with the GPU.
- ACE (`detail/cagra/cagra_ace.cuh`): cuVS's own partitioned out-of-core CAGRA build — check its partition I/O and per-partition graph merge for ideas for `prune_windowed` / `optimize_chunked`.

Other:
- SIFT10M Step 2 (centroid selection on the full 10M working set) is far slower than on 1M — sample before kmeans|| (cuVS kmeans uses a sampled init); compare `cpp/src/cluster/detail/kmeans_balanced.cuh` / `kmeans.cuh` init.
- Step 4.5 inline reorder and Step 6 `pool_build` / `scatter_pending` dominate the SIFT10M out-of-core loop — overlap host gather with GPU work (pinned staging, double buffering).
- Large path quality gap (prune_windowed vs cuVS optimize): SIFT1M recall@10 at itopk 32 is ~3 points below CAGRA — study `cpp/src/neighbors/detail/cagra/graph_core.cuh` (`optimize`: detour counting / rank-based pruning / reverse-edge merge) and align `prune_windowed_core.hpp` + `optimize_chunked` method E with it.
- `optimize` (resident) spends most of its time in file load + sanitize, not in the cuVS call.
- `prune_windowed` after #2 (3.9s): compute 2.6s forks an OMP region per bucket of ~100 rows on 80 threads — process several positions per parallel region, or move the detour count to GPU (cuVS `kern_fused_prune`). Store cached vectors as u8 (4x less cache memory → fewer evictions on SIFT10M); keep the dataset handle open in the loader.
- Resident `optimize` after #5 (1.3s): ~0.25s is raft/CUDA context start-up + 256 MB graph read; the cuVS call itself is 0.44s. Further gains need fusing optimize into bucket2 (graph already on the GPU) — a structural change, keep the separate-stage fallback for exit code 2.
- bucket2 still writes `vector_dists.bin` (256 MB on SIFT1M) that no later stage reads — same treatment as #4 (`--skip-dists`), check `knn_similarity.py` / eval don't need it first.
- bucket2 Step 6 out-of-core after #10 (~3.75 s/round loop): pool build now overlapped on a producer thread (wait_producer ~0.1 s); main thread = stream_sync ~1.7 s (GPU-bound now: gather+norms+GEMM+select_k+D2H per ~100-row bucket, 3 slots) + scatter_pending ~0.7 s + gpu_submit ~1.2 s. Next: batch several buckets per GPU submission / fewer kernel launches (cuVS 26.12 `all_neighbors` batches clusters), FP16/INT8 GEMM is already used; select_k on tiny matrices is launch-bound.
- bucket2 Step 4 out-of-core (8.8s vs 3.3s resident): per round convert_upload ~0.6s + search 0.7s + bucket_write + finish_iteration; ChunkedKNN merge_iteration write_running ~1.1s/chunk in rounds ≥2.

- **Bucket-size outliers break the GPU-memory bound (found in #9, hit again in #13 — the memory gate is flaky until fixed; NEXT):** the balanced assignment (`batch_assign_with_cagra_anns`, CPU step 4d) reads `cluster_sizes` with relaxed atomics from all threads, so bucket sizes differ run to run despite the fixed seed; occasionally a round produces a bucket ~30x the average (SIFT1M large: max 3201 vs avg 100; resident run in #6: 1980). Out-of-core Step 6 sizes its per-slot `dots` buffers from max_pool (0.72 GB/slot × 3 instead of ~0.2 GB) → GPU peak 2708 MiB instead of ~1450. Fix: cap bucket size (spill to the 2nd-nearest centroid when a bucket exceeds e.g. 4–8× avg), or tile the oversized bucket's GEMM; also makes the build deterministic if done per batch in a fixed order.

- **NPROBE 100** (measured in #11, 1 run each): auto 15.70 s, large 36.79 s (vs 16.79 / 39.32 at 150), search recall@10 itopk 32 auto 0.9269 (CAGRA 0.9305, gate 0.9205), pre-prune KNN recall@64 0.938. Passes the gate but spends 0.3 pt of resident recall headroom — user decision.
- **Resident Step 6 is GPU-bound, not launch-bound** (#11 probe): launching from 4 host threads changed nothing (launches block on a full queue). Kernel elimination per round (1.19 s loop, NPROBE 250): gather ~0.15 s, GEMM ~0.4 s, block top-M ~0.5 s; without GEMM+top-M 0.23 s. top-M block shape G=4/8/16 all ≈ same. Real fix = never materialize the (bucket × pool) dots matrix: fused GEMM + top-M (cuVS 26.12 `detail/fused_l2_knn.cuh` / `knn_brute_force.cuh`), or shrink the pool (#11).

## Tried & rejected

| # | idea | result | reason |
|---|---|---|---|
| 15 | Retry of #12 (incremental k-means|| d(x,C) update) after #13 removed most of the final-merge wait | auto 15.56 s (+3.8%) then rerun 16.05 s (+0.7%); large 36.63 s (−0.7%: Step 2 4.34 → 3.61 s, but Step 4 out-of-core 5.79 → 7.08 s from write-throttled `finish_iteration`) | gain (~0.15 s/round) is within run-to-run noise (±0.5 s on auto). Only worth it bundled with other Step 2 savings; large path is now dominated by disk-throttled writes (bucket_vecs + ChunkedKNN arrival/running, ~1.5–2 GB per round) |
| 14 | **Bounded bucket size** (robustness, not speed): balanced assignment made serial (no relaxed-atomic race), brute-force top-4 with (distance, id) ordering, cap = 8× avg bucket size — when the chosen bucket is full take the nearest of the 4 that is not, else the smallest. Patch kept: `bucketDemo/patches/iter14_bucket_cap.patch` | auto 16.61 s (−2.7%), large 37.19 s (−2.3%); max bucket ≤ 1005 in all 6 reps (was up to 3201), large-path GPU peak a steady 1252 MiB in all reps; search recall unchanged within 0.0006 | judge: no ≥3% speed gain (by design — it fixes the flaky large-path GPU-memory gate, see #9/#13). **Needs a user decision** to apply outside the speed rule. Note: the build is still not bit-reproducible with a fixed seed — Step 3 (centroid graph) / bucket order also vary run to run |
| 12 | k-means|| (Step 2): update d(x, C) incrementally with only the newly sampled candidates + elementwise min (cuVS 26.12 `cluster/detail/kmeans.cuh::initScalableKMeansPlusPlus`), instead of recomputing against all candidates each round; same sampled candidates (counts identical) | auto 15.42 s (+4.5%) then rerun 15.74 s (+2.5%); large 39.51 s (−1.4%): Step 2 4.67 → 3.43 s but the final background-merge wait grew 2.02 → 2.92 s (Step 2 was hiding the previous round's ChunkedKNN merge) | borderline gain not confirmed on rerun. Retry together with a faster out-of-core merge (`merge_iteration` write_running ~1.1 s/chunk), where the Step 2 saving would show up on the large path too |

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

### #5 — resident optimize: no dataset load under --skip-sort, in-place sanitize  (2026-10-21)
**Idea / rationale:** `optimize` (2.08s) loaded the whole dataset as float32 (0.5 GB, 0.55s) although with `--skip-sort` (as the pipeline runs it) the dataset is never used — cuVS 26.12 `cagra::helpers::optimize` / `detail/cagra/graph_core.cuh::optimize` takes only the kNN graph. Now only the header is read for N/D when `--skip-sort`. Sanitize (int32 → uint32, padding/out-of-range/self-loop repair) now works in place on the loaded buffer instead of allocating a second N×M array (256 MB + page faults). GPU-fit check unchanged (still counts the dataset, conservative). Standalone on SIFT1M: 2.15 → 1.28 s, peak RSS 1.37 → 0.76 GB. Output not byte-comparable: cuVS optimize itself is nondeterministic (reverse-edge atomics; two runs of the old binary differ in ~445K rows, same as old vs new); search recall unchanged.
**Files:** `src/optimize.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 18.08 | +5.7% | +16.9% | 1.57x | bucket2 16.39, optimize 1.30, translate 0.40 | 4144 | 1.58 |
| sift1M_large | large | 53.72 | +3.1% | +28.5% | 4.66x | bucket2 44.16, prune 3.95, optimize_chunked 5.27, translate 0.34 | 1512 | 4.19 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.73 | 3.20 | 1.20 | 0.00 | 3.20 | 1.12 | 0.02 | 5.64 | 0.00 | 0.75 |
| sift1M_large | 0.55 | 5.00 | 1.89 | 0.00 | 8.99 | 1.75 | 0.03 | 22.02 | 2.12 | 1.08 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9935 | 0.9892 | 0.9862 | 0.9836 |
| sift1M_large | 0.9933 | 0.9898 | 0.9866 | 0.9838 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9306 / 1943K | 0.9778 / 1033K | 0.9946 / 510K | 0.9988 / 187K | 0.9997 / 68K |
| ours sift1M_large | 0.8979 / 1996K | 0.9624 / 1065K | 0.9891 / 544K | 0.9977 / 189K | 0.9995 / 68K |

[exited with code 0]

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +5.7% (19.18 → 18.08 s), 1M-large +3.1% (55.44 → 53.72 s; path untouched by this change, bucket2 run-to-run noise).

### #6 — out-of-core Step 6: cap pool memcpy at 8 OpenMP threads  (2026-10-21)
**Idea / rationale:** in the out-of-core bucket loop, `pool_build` (3.0–3.4 s/round) was dominated by `pool_memcpy` (1.7–2.5 s/round): for each bucket the ~K+1 cached neighbour buckets (each ~100 rows ≈ 13 KB vectors + ids) are copied into the pinned upload buffer with `#pragma omp parallel for` over all 80 threads — a fork/join per bucket for ~430 KB of copying, plus cross-NUMA bandwidth contention (2-socket Xeon Gold 6133). Measured per-round pool_memcpy on SIFT1M --out-of-core: 80 thr 2.1–2.9 s, 32: 1.6, 16: 1.2–1.6, 8: 1.3–1.6, 4: 1.7, 1: 5.3. Now `num_threads(min(8, omp_get_max_threads()))`. Same principle as cuVS 26.12 `cagra_build.cuh::build_knn_graph`, which stages host batches with a bounded thread count rather than all cores. No algorithmic change.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 18.53 | -2.5% | +14.8% | 1.61x | bucket2 16.82, optimize 1.27, translate 0.45 | 4702 | 1.58 |
| sift1M_large | large | 51.92 | +3.4% | +30.9% | 4.50x | bucket2 42.21, prune 3.84, optimize_chunked 5.17, translate 0.36 | 1598 | 4.29 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.74 | 3.31 | 1.39 | 0.00 | 3.28 | 1.11 | 0.02 | 5.62 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 5.28 | 1.98 | 0.00 | 8.77 | 1.74 | 0.03 | 19.92 | 2.00 | 1.07 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9920 | 0.9889 | 0.9860 | 0.9837 |
| sift1M_large | 0.9936 | 0.9895 | 0.9869 | 0.9843 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9303 / 1945K | 0.9778 / 1035K | 0.9948 / 492K | 0.9988 / 186K | 0.9998 / 67K |
| ours sift1M_large | 0.8977 / 1943K | 0.9618 / 1032K | 0.9886 / 477K | 0.9971 / 187K | 0.9994 / 67K |

[exited with code 0]

Borderline large gain → rerun (`runs/iter6/sift1M_large_b`): +4.19% → also ACCEPT. Step 6 (large) 22.0 → 19.9 s.

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +3.35% / +4.19% on rerun (53.72 → 51.92 s); 1M-auto −2.5% (path untouched, within tolerance).

### #7 — resident Step 4: exact brute-force top-2 instead of CAGRA search  (2026-10-21)
**Idea / rationale:** (user-priority backlog item) Step 4 on the GPU-resident path searched 990K non-centroid points against the 10K centroids with `cagra::search` (k=2, itopk 64, single-CTA), ~0.70 s/round. With queries already on the GPU, exact `cuvs::neighbors::brute_force::search` (L2Expanded; tiled GEMM + select_k — cuVS 26.12 `detail/knn_brute_force.cuh::tiled_brute_force_knn`) is faster: 0.42 s/round, and the top-2 is exact instead of approximate. Index built once per round over `d_centroids` (device view, no copy); int64 ids downloaded and bounds-checked (`[0, n_centroids)`, else error) before the unchanged balanced-assignment logic. Only the resident branch (`gpu_query_loader` set, non-PQ) switches; out-of-core and PQ keep the CAGRA search. fp32 kept on purpose (fp16 accumulation would overflow on uint8 data).
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 17.19 | +7.2% | +21.0% | 1.49x | bucket2 15.52, optimize 1.27, translate 0.37 | 4184 | 1.61 |
| sift1M_large | large | 51.93 | -0.0% | +30.9% | 4.50x | bucket2 42.47, prune 3.76, optimize_chunked 5.16, translate 0.49 | 1550 | 4.19 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.74 | 3.31 | 1.31 | 0.00 | 2.18 | 1.11 | 0.02 | 5.63 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 5.08 | 1.34 | 0.00 | 8.66 | 2.18 | 0.03 | 20.67 | 2.05 | 1.06 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9936 | 0.9895 | 0.9866 | 0.9838 |
| sift1M_large | 0.9912 | 0.9889 | 0.9860 | 0.9834 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9295 / 1946K | 0.9779 / 1035K | 0.9947 / 533K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.8975 / 2000K | 0.9619 / 945K | 0.9888 / 560K | 0.9976 / 189K | 0.9995 / 68K |

[exited with code 0]

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +7.2% (18.53 → 17.19 s; Step 4 3.28 → 2.18 s), 1M-large −0.0% (path untouched).

### #8 — out-of-core Step 4: brute-force top-2 + GPU-side gather/convert  (2026-10-21)
**Idea / rationale:** extend #7 to the out-of-core batch assignment. (1) Exact `brute_force::search` top-2 for every non-PQ run (centroids are small and already on the GPU; queries are still batched by `Nv` under the `Mbatch_bytes` budget, so nothing depends on the full dataset fitting in GPU memory). (2) The batch's raw `uint8` row range is uploaded as is (128 MB instead of 512 MB float, no pageable CPU conversion buffer) together with int32 row offsets of the non-centroid rows (bounds-checked against the read range), and gathered + converted to float on the GPU with the existing `gather_rows_int32` kernel — like cuVS 26.12 IVF-PQ `extend` / `cagra_build.cuh`, which upload raw batches and convert on device. Per round on SIFT1M: convert_upload 0.51–0.62 → 0.02 s, search 0.69–0.75 → 0.42 s. GPU peak 1374–1536 MiB (gate 2116). PQ path unchanged.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 17.42 | -1.3% | +19.9% | 1.51x | bucket2 15.72, optimize 1.30, translate 0.39 | 4160 | 1.61 |
| sift1M_large | large | 48.92 | +5.8% | +34.9% | 4.24x | bucket2 39.31, prune 3.89, optimize_chunked 5.30, translate 0.41 | 1536 | 3.95 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.24 | 1.48 | 0.00 | 2.11 | 1.12 | 0.02 | 5.64 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 5.13 | 1.38 | 0.00 | 5.32 | 2.16 | 0.03 | 20.68 | 2.68 | 1.07 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9933 | 0.9900 | 0.9870 | 0.9842 |
| sift1M_large | 0.9926 | 0.9890 | 0.9865 | 0.9838 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9301 / 1945K | 0.9778 / 1036K | 0.9947 / 525K | 0.9988 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.8996 / 1943K | 0.9629 / 1033K | 0.9889 / 493K | 0.9973 / 186K | 0.9995 / 68K |

[exited with code 0]

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +5.8% (51.93 → 48.92 s; Step 4 8.66 → 5.32 s), 1M-auto −1.3% (noise; resident Step 4 2.18 → 2.11 s).

### #9 — optimize_chunked method E: parallel rank-ordered CSR build  (2026-10-21)
**Idea / rationale:** method E's Phase 2 (3.07 s) was dominated by `edge_chunk::build_rank_ordered_csr`, fully serial over 32M reverse-edge records: count per destination, scatter, sort each row's segment by rank (1M small sorts), copy. Now OpenMP: atomic per-row counts, atomic-cursor scatter, per-row sort in a dynamic-scheduled parallel loop with copy-out fused. Rows are sorted by (rank, src): the src tie-break makes the output independent of scatter order (previously equal ranks kept the GPU scatter's arrival order, so two runs could differ; now two runs are byte-identical). Added a bounds check (edge dst must lie inside the chunk). The CSR + per-row merge is the CPU analogue of cuVS 26.12 `detail/cagra/graph_core.cuh` reverse-graph build + `kern_merge_graphs`. `test_edge_chunk_accumulator` passes (serial and -fopenmp). Standalone: Phase 2 3.07 → 1.25 s, optimize_chunked 5.3 → 3.4 s, RSS unchanged.
**Files:** `src/edge_chunk_accumulator.hpp`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 17.19 | +1.3% | +21.0% | 1.49x | bucket2 15.43, optimize 1.27, translate 0.47 | 4080 | 1.61 |
| sift1M_large | large | 46.90 | +4.1% | +37.5% | 4.06x | bucket2 39.09, prune 3.89, optimize_chunked 3.47, translate 0.39 | 1530 | 3.95 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.25 | 1.18 | 0.00 | 2.14 | 1.12 | 0.03 | 5.64 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 4.95 | 1.25 | 0.00 | 5.54 | 2.15 | 0.03 | 20.65 | 2.08 | 1.08 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9932 | 0.9897 | 0.9869 | 0.9840 |
| sift1M_large | 0.9918 | 0.9892 | 0.9869 | 0.9841 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9304 / 1942K | 0.9784 / 1035K | 0.9947 / 507K | 0.9988 / 187K | 0.9997 / 68K |
| ours sift1M_large | 0.8982 / 1999K | 0.9628 / 1008K | 0.9891 / 557K | 0.9976 / 189K | 0.9994 / 68K |

[exited with code 0]

Borderline large gain → reruns: `sift1M_large_b` +3.23% but **REJECT on GPU peak 2708 MiB** (one rep; spike in bucket2 Step 6 of round 3, caused by a 3201-point bucket from the nondeterministic balanced assignment — this change touches only optimize_chunked's CPU code; see backlog); `sift1M_large_c` +3.56%, GPU peak 1274–1422 MiB → ACCEPT. Speed passed in all three runs (+4.1 / +3.2 / +3.6%).

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +4.1% (48.92 → 46.90 s; optimize_chunked 5.30 → 3.47 s), 1M-auto +1.3% (path untouched).

### #10 — out-of-core Step 6: pool build on a producer thread  (2026-10-21)
**Idea / rationale:** the out-of-core bucket loop was serial per position: stream sync → scatter previous results → pool build (Belady `cache.get` + memcpy into pinned staging, ~2.4–3.0 s/round) → GPU submit. Pool building depends only on the bucket order, so it now runs on a producer thread that owns `bucket_cache` exclusively (cache still single-threaded, same Belady access sequence), staging into the existing per-slot pinned buffers at most `num_slots` positions ahead. Hand-off per slot via mutex/condvar (`slot_ready_pos`); before refilling a slot the producer waits on a CUDA event recorded after that slot's H2D copies; the bucket's own gids are staged separately and copied into `h_ids_bucket[slot]` only after the old results were scattered. Producer exceptions are rethrown on the main thread; an RAII guard aborts + joins the producer if the main loop throws. Added a pool-capacity bounds check. Same pattern as cuVS 26.12 `detail/cagra/cagra_build.cuh::build_knn_graph` (background thread prepares the next batch into a pinned double buffer while the current one is processed). Loop per round 5.0 → 3.75 s; pre-prune KNN recall unchanged.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 17.22 | -0.2% | +20.8% | 1.49x | bucket2 15.60, optimize 1.27, translate 0.38 | 4184 | 1.61 |
| sift1M_large | large | 43.39 | +7.5% | +42.2% | 3.76x | bucket2 35.63, prune 3.92, optimize_chunked 3.41, translate 0.43 | 1610 | 3.94 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.19 | 1.40 | 0.00 | 2.23 | 1.12 | 0.02 | 5.65 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 4.29 | 1.78 | 0.00 | 5.66 | 2.21 | 0.03 | 16.76 | 2.00 | 1.06 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9934 | 0.9898 | 0.9866 | 0.9840 |
| sift1M_large | 0.9935 | 0.9894 | 0.9864 | 0.9839 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9299 / 1946K | 0.9780 / 1035K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_large | 0.8964 / 1944K | 0.9621 / 1034K | 0.9886 / 490K | 0.9973 / 186K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +7.5% (46.90 → 43.39 s; Step 6 20.65 → 16.76 s), 1M-auto −0.2% (path untouched). GPU peak 1500–1610 MiB.

### #11 — NPROBE 250 → 150  (2026-10-21)
**Idea / rationale:** Step 6 cost on both paths is proportional to the pool size per bucket (own bucket + `nprobe` nearest buckets ≈ nprobe × avg bucket size candidates per query): the resident loop is GPU-bound on the materialized (bucket × pool) dots matrix (GEMM write + top-M read), the out-of-core loop on pool building and the same GPU work. Parameter sweep (1 run each): NPROBE 150 → auto 16.79 s / large 39.32 s, search recall@10 within 0.001 of NPROBE 250; NPROBE 100 → 15.70 / 36.79 s but −0.3 pt resident recall at itopk 32 (kept in backlog as a user decision). Chose 150. Analogous to cuVS IVF-based kNN-graph builds, which search only `n_probes` lists per query and rely on refinement + the CAGRA prune to recover quality. **Trade-off recorded:** pre-prune degree-64 KNN recall@64 drops 0.984 → 0.963 (not gated); final graph search recall unchanged within 0.001 (resident 0.9296 vs CAGRA 0.9305 at itopk 32).
**Files:** —  **Params changed:** `NPROBE=150` (was 250)

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 16.15 | +6.2% | +25.7% | 1.40x | bucket2 14.51, optimize 1.32, translate 0.37 | 3710 | 1.60 |
| sift1M_large | large | 38.98 | +10.2% | +48.1% | 3.38x | bucket2 31.21, prune 3.95, optimize_chunked 3.44, translate 0.48 | 1220 | 4.08 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.74 | 3.18 | 1.37 | 0.00 | 2.17 | 1.09 | 0.02 | 4.58 | 0.00 | 0.75 |
| sift1M_large | 0.56 | 4.67 | 1.62 | 0.00 | 5.79 | 2.16 | 0.03 | 12.36 | 2.02 | 1.06 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9786 | 0.9746 | 0.9679 | 0.9628 |
| sift1M_large | 0.9813 | 0.9750 | 0.9686 | 0.9640 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9296 / 1935K | 0.9774 / 1019K | 0.9942 / 518K | 0.9987 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.8973 / 1944K | 0.9619 / 954K | 0.9886 / 561K | 0.9972 / 189K | 0.9994 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +6.2% (17.22 → 16.15 s; Step 6 5.65 → 4.58 s), 1M-large +10.2% (43.39 → 38.98 s; Step 6 16.76 → 12.36 s).

### #13 — out-of-core: last merge writes vector_knn/dists directly  (2026-10-21)
**Idea / rationale:** the final iteration's background ChunkedKNN merge wrote every merged chunk to `running_chunk_*` files, then `finalize_to_file` read them all back and rewrote them as `vector_knn.bin` / `vector_dists.bin` — a full extra N·M·8 B write + read + write at the end of bucket2 (all disk-bandwidth-bound: by then the page cache is full of dirty pages, round ≥1 `write_running` is 1.1–2.0 s/chunk). Now the last merge (`set_final_output` before launching it) pre-sizes the two output files (header + ftruncate) and `pwrite`s each merged chunk at its row offset from the merge threads; `finalize_to_file` becomes a check. Same "write each partition's result straight to its slot of the output" as cuVS 26.12 `detail/cagra/cagra_ace.cuh` (partitioned build writes per-partition graph rows into the final graph). Scales with N (removes one full I/O pass). Also tried first: alternating running-file names by iteration parity — round 1 write became fast but rounds 2–3 stayed slow (dirty-page throttling, not truncate-on-writeback) → dropped.
Final-merge wait 2.0–2.8 → 0.97 s, Write step 1.06 → 0.00 s.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 16.17 | -0.1% | +25.7% | 1.40x | bucket2 14.46, optimize 1.28, translate 0.43 | 4414 | 1.60 |
| sift1M_large | large | 36.36 | +6.7% | +51.6% | 3.15x | bucket2 28.48, prune 4.06, optimize_chunked 3.44, translate 0.48 | 1224 | 3.88 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.17 | 1.33 | 0.00 | 2.14 | 1.14 | 0.02 | 4.57 | 0.00 | 0.76 |
| sift1M_large | 0.56 | 4.34 | 1.53 | 0.00 | 5.79 | 2.15 | 0.03 | 12.31 | 0.97 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9811 | 0.9744 | 0.9683 | 0.9629 |
| sift1M_large | 0.9812 | 0.9744 | 0.9681 | 0.9631 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9289 / 1947K | 0.9769 / 1035K | 0.9944 / 533K | 0.9987 / 189K | 0.9998 / 68K |
| ours sift1M_large | 0.8961 / 2002K | 0.9618 / 958K | 0.9882 / 484K | 0.9970 / 189K | 0.9993 / 68K |

First run (`runs/iter13/sift1M_large`): 36.94 s (+5.2%) but **REJECT on GPU peak 1826 MiB** (rep1 had a 2789-point bucket in round 3 → oversized Step 6 buffers; nondeterministic balanced assignment, see backlog; this change touches no GPU code). Rerun `sift1M_large_b`: 36.36 s (+6.7%), GPU peak 1220–1224 MiB → ACCEPT (recorded as best).

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +6.7% (38.98 → 36.36 s), 1M-auto −0.1% (path untouched).

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
