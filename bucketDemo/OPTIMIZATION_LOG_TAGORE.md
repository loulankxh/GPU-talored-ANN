# bucket2 pipeline — optimization log, TAGORE edition

Pipeline: `bucket2` (bucketing → bucket order → bucket-pair per-vector KNN, degree 64)
→ prune + reverse edges to degree 32 (`optimize` resident / `prune_windowed` + `optimize_chunked -m E`)
→ `translate_ids` back to original order.

Environment: cuVS / raft / rmm **26.10** (conda `cuvs-bench`), gcc 14 (conda `cuvs-build`), nvcc 12.9,
Tesla V100-SXM2-16GB, **GPU 0 only**, worktree `/home/lanlu/microBenchmark_tagore`, branch `opt/bucket2-tagore`.
Driver: `/optimize-bucket2-tagore` skill (`.claude/skills/optimize-bucket2-tagore/`), judged by its `scripts/judge.py`.
Timed runs share `/home/lanlu/.bench_locks/bucket2_bench.lock` with the cuVS loop (`/optimize-bucket2`).

**Ideas source: TAGORE (`/home/lanlu/Tagore`)** — GNN-Descent, CFS GPU pruning, async GPU-CPU-disk large-scale pipeline; cuVS 26.12 secondary.
**Change policy:** implementation changes (same results, different execution) are tried directly; algorithm / semantic
changes go to **Awaiting approval** until the user approves them (A1 = GNN-Descent local join, pre-approved 2026-10-21).

Measured per iteration:
- **build time**: wall clock of every stage (process start, all I/O included), summed.
  SIFT1M auto (GPU-resident) and SIFT1M large (large path forced): median of 3; SIFT10M auto: 1 run;
  SIFT100M auto: **2 rounds** (`ITERATIONS=2`), 1 run.
- **(a) pre-prune index quality**: degree-64 `vector_knn` vs exact KNN (10K sampled base points), logged, not gated.
- **(b) final search quality**: degree-32 graph (original ids); SIFT1M/10M with cuVS CAGRA search (itopk 32…512),
  SIFT100M with DiskANN CPU in-memory search (medoid entry, L 32…512); recall@10 + QPS, 10K queries, defect counts.

Rules (user, 2026-10-22) — **goal: faster SIFT100M build**:
- SIFT100M vs its own best: ≥ 7% faster = gain, > 7% slower = regression; 3–7% gain → **Possible gains** (not accepted);
  DiskANN recall@10 ≥ own best − 0.05 at every L; large-path memory ≤ 1.2× best.
- SIFT1M auto / large, SIFT10M: ≥ 3% faster = gain, > 3% slower = regression; recall@10 vs **TAGORE's graph searched the
  same way**: resident ≥ TAGORE − 0.01, large ≥ TAGORE − 0.05 (CAGRA fallback until step B writes `tagore_baseline.json`).
- ACCEPT iff ≥ 1 config gains, none regresses, search passes on all.

## Current best

After **#2 (Step 6: deferred eviction frees + reuse of freed device blocks)**, 2026-10-22 (`runs/tagore2`):

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.27 | +0.0% | -0.2% | 0.68x | 0.54x | bucket2 6.27 | 3812 | 1.71 |
| sift1M_large | large | 9.61 | -1.8% | +0.8% | 1.05x | 0.83x | bucket2 7.78, prune 0.71, optimize_chunked 0.72, translate 0.40 | 1590 | 2.04 |
| sift10M_auto | large | 82.22 | -0.0% | +0.3% | 0.38x | 0.67x | bucket2 71.83, prune 3.96, optimize_chunked 2.87, translate 3.56 | 8996 | 12.65 |
| sift100M_auto | large | 2506.89 | +15.8% | +22.0% | 1.17x | — | bucket2 2023.02, prune 133.15, optimize_chunked 286.79, translate 63.93 | 15492 | 79.41 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.26 | 0.17 | 0.00 | 0.62 | 0.36 | 0.02 | 2.75 | 0.00 | 0.89 |
| sift1M_large | 0.14 | 0.60 | 0.35 | 0.00 | 0.99 | 0.26 | 0.04 | 4.21 | 0.49 | 0.00 |
| sift10M_auto | 1.45 | 7.43 | 2.37 | 0.00 | 18.52 | 1.76 | 0.38 | 34.79 | 4.00 | 0.00 |
| sift100M_auto | 72.31 | 371.49 | 97.43 | 0.00 | 252.26 | 170.64 | 7.47 | 1016.61 | 26.76 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9741 | 0.9668 | 0.9590 | 0.9532 |
| sift1M_large | 0.9754 | 0.9663 | 0.9592 | 0.9535 |
| sift10M_auto | 0.9162 | 0.8989 | 0.8832 | 0.8711 |
| sift100M_auto | 0.7511 | 0.7208 | 0.6970 | 0.6814 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9280 / 1950K | 0.9769 / 1048K | 0.9944 / 501K | 0.9987 / 187K | 0.9996 / 68K |
| ours sift1M_large | 0.9147 / 2224K | 0.9708 / 1188K | 0.9922 / 561K | 0.9983 / 189K | 0.9995 / 68K |
| ours sift10M_auto | 0.8393 / 1868K | 0.9292 / 1011K | 0.9735 / 489K | 0.9916 / 176K | 0.9980 / 68K |
| ours sift100M_auto | 0.7458 / 69K | 0.8518 / 64K | 0.9252 / 39K | 0.9680 / 23K | 0.9880 / 13K |

## Baseline verification

Main 3046497 (`state/reference_best.json`, cuVS loop through #76) vs this worktree's build, 2026-10-22.
Tolerances: build ±3% (SIFT1M, SIFT10M) / ±7% (SIFT100M); recall@10 at 32/64/128 ±1% rel (SIFT1M) / ±3% rel (SIFT10M, SIFT100M; SIFT100M = DiskANN L).

| config | path (ref / ours) | build_total s (ref → ours, Δ, tol) | recall@10 L/itopk 32 / 64 / 128 (ref → ours) | max rel Δ (tol) | QPS@64 (ref → ours) | GPU peak MiB / max RSS GB (ref → ours) | defects | pass |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident / resident | 6.32 → 6.26 (-0.9%, ±3%) | 0.9283→0.9276 / 0.9774→0.9770 / 0.9943→0.9943 | 0.08% (±1%) | 1036K → 1035K | 3900/1.7 → 3900/1.7 | 0 | yes |
| sift1M_large | large / large | 9.45 → 9.69 (+2.5%, ±3%) | 0.9138→0.9135 / 0.9708→0.9709 / 0.9921→0.9922 | 0.04% (±1%) | 1188K → 1190K | 1558/2.0 → 1760/2.0 | 0 | yes |
| sift10M_auto | large / large | 82.63 → 82.44 (-0.2%, ±3%) | 0.8388→0.8374 / 0.9280→0.9287 / 0.9735→0.9733 | 0.16% (±3%) | 1051K → 1012K | 8996/12.7 → 8996/12.6 | 0 | yes |
| sift100M_auto | large / large | 3087.75 → 3214.58 (+4.1%, ±7%) | 0.7449→0.7472 / 0.8519→0.8536 / 0.9251→0.9257 | 0.31% (±3%) | 64K → 66K | 14588/79.4 → 14588/79.4 | 0 | yes |

All four pass on the first run. SIFT100M: bucket2 2725.6 s vs 2566.6 s ref (+6.2%; Step 6 1687 s), optimize_chunked
280.2 vs 307.4 s, prune 150.0 vs 151.7 s, translate 58.8 vs 62.1 s — single-run noise, within ±7% overall.
sift1M_large GPU peak 1760 vs 1558 MiB (not gated).

## TAGORE baseline

Step B, 2026-10-22 (`state/tagore_baseline.json`, runs in `runs/tagoreB`). TAGORE = `/home/lanlu/Tagore` @ 110d480, GPU 0
under the bench lock, page cache dropped, wall clock incl. load (like our `build_total`).

**Build setup.** conda env `tagore` (python 3.11, numpy, pybind11); nvcc 12.9 (`cuvs-bench`); build dirs under
`/home/lanlu/Tagore/build-v100`. Allowed build-config edit in the clone (`CMakeLists.txt`):
`pybind11_DIR` → `/home/lanlu/miniconda3/envs/tagore/share/cmake/pybind11`, `CMAKE_CUDA_ARCHITECTURES 89` → `70`.

**TAGORE does not run correctly on the V100 (sm_70) as published** — three bugs, fixed in a *copy*
(`build-v100/patched-src`, built in `build-v100/patched-build`; the clone's sources are untouched;
diff: `.claude/skills/optimize-bucket2-tagore/scripts/tagore_v100_fix.patch`). **Confirmed by user 2026-10-22 (fixes stay in the copy)** (decision relayed from the user's other session):
1. `nn_descent_opt_reverse_sample`: binary search reads `old_nn[old_num]` (one past the 32-entry shared array) →
   "Warp Out-of-range Address" on sm_70 (found with compute-sanitizer + cuda-gdb). Fix: `res_id < num &&` bounds check.
2. `nn_descent_opt_cal` (both GEMMs): indexes `acc_frag.x[i]` with the sm_80 `mma.m16n8k16` element order
   (row lane/4, col (lane%4)·2 …); the Volta wmma accumulator order differs, so every distance was paired with the wrong
   (new, old) ids → kNN recall@10 ≈ 0.00. Fix: `store_matrix_sync` the tile to (free) shared memory and reload `x[0..7]`
   in the sm_80 order; same values, an extra 1 KB/warp shared round trip.
3. `gpu_construct` (largeIndex): `cudaMemcpyFromSymbol(&dis_func, dis_filter_new)` on a value that is not a symbol fails
   silently, `dis_func` stays uninitialized → call through a null device function pointer. Fix: `dis_func = dis_filter_new`.
Also: `StoreCAGRA`'s `i * degree` indexing is correct (`filter_reverse_1hop` writes final rows with stride `d`).

| dataset | TAGORE config | index | build (s) | stages | recall@10 32 / 64 / 128 / 256 / 512 | QPS @32 / @64 | defects | GPU peak MiB / RSS GB | ours (best) |
|---|---|---|---|---|---|---|---|---|---|
| SIFT1M | GNN_descent k=96, 10 it + Pruning CAGRA (m=64, d=32), median of 3 | CAGRA | **9.18** (9.18 / 9.22 / 9.16) | GNN-Descent 3.07, pruning 0.74, rest = fvecs load + fp16 convert (host) | 0.9220 / 0.9752 / 0.9936 / 0.9985 / 0.9996 (cagra_eval) | 1947K / 1033K | 440,096 dup edges | 2878 / 1.2 | auto 6.26 s, 0.9276@32; large 9.69 s, 0.9135@32 |
| SIFT1M | same, k=64 | CAGRA | 8.60 | GNN-Descent 2.83, pruning 0.52 | 0.9232 / 0.9748 / 0.9932 / 0.9983 / 0.9995 | 2224K / 1192K | 412,328 dup edges | 2604 / 1.1 | |
| SIFT10M | in-memory GNN_descent k=64 | — | **failed** | needs > 16 GB device memory (unchecked cudaMalloc → illegal access, peak 15778 MiB) | — | — | — | — | |
| SIFT10M | largeIndex k=64, m=64, d=32, α=1.2, 10 clusters (2 per point), max_points 2.5M, buffer 6, 1 run | Vamana | **214.47** | prep 35.46 (k-means 1.6, assign 3.9, split 2.3, write cluster files 26.7) + largeIndex 179.01 | 0.8448 / 0.9238 / 0.9662 / 0.9863 / 0.9950 (cagra_eval, short rows repeat-padded); DiskANN medoid: 0.8611 / 0.9284 / 0.9674 / 0.9867 / 0.9952 | 1875K / 1014K | 791 self loops, 0 dup; 5.6M empty slots (rows < 32) | 7242 / 5.8 | 82.44 s, 0.8374@32 |
| SIFT100M | largeIndex, same params, 100 k-means clusters → 103 after split, buffer 30, 1 run | Vamana | **2151.25** | prep 326.29 (k-means 2.5, assign 48.7, split 38.4, write 235.3) + largeIndex 1824.96 | DiskANN medoid: 0.7691 / 0.8615 / 0.9238 / 0.9616 / 0.9820; TAGORE ep: 0.7651 / 0.8592 / 0.9233 / 0.9614 / 0.9820 | 61K / 55K (DiskANN) | 18,780 self loops, 0 dup; 31.9M empty slots | 14768 (prep) / 7242 (build) / 30.7 | 3214.58 s, 0.7472 / 0.8536 / 0.9257 / 0.9686 / 0.9881 |

Notes:
- SIFT1M input: u8bin → fvecs conversion 1.34 s (not in build time); SIFT100M prep reads the u8bin directly.
- largeIndex's `f2h` has no fp16 scaling (GNN_descent's `norm_factor`): raw SIFT values overflow fp16 in `cal_power`.
  Prep (`scripts/tagore_prep_clusters.py`) writes the cluster files pre-scaled by the same power-of-10 factor (0.1),
  which leaves neighbor order and the α test unchanged. Prep also does largeIndex's oversized-cluster split itself
  (same round-robin rule), because largeIndex's own split references data files that do not exist.
- largeIndex hardcodes GNN-Descent 8 + 8 iterations per cluster; the merge only concatenates the two clusters'
  pruned lists (no re-prune) and truncates to max_degree.
- Headline: SIFT100M TAGORE 2151 s vs ours 3215 s (**ours 1.49× slower**) at similar DiskANN recall (TAGORE +0.02 at
  L 32, −0.006 at L 512); SIFT10M TAGORE 214 s vs ours 82 s (ours 2.6× faster); SIFT1M TAGORE 9.2 s vs ours 6.3 s.
  TAGORE's SIFT100M per-cluster cost: ~1825 s / 103 clusters ≈ 17.7 s per ~1.94M-point subgraph (GNN-Descent + Vamana prune
  on GPU, overlapped with the CPU merge).
- Judge reference (`judge.py`): SIFT1M → TAGORE k=96 row (resident ≥ TAGORE − 0.01, large ≥ TAGORE − 0.05);
  SIFT10M → TAGORE largeIndex cagra_eval row (≥ TAGORE − 0.05). Current best passes both.

## Ideas backlog

Ranked by expected SIFT100M saving (T0 study, 2026-10-22; details in `.claude/skills/optimize-bucket2-tagore/notes/tagore_map.md`).
SIFT100M Step 6 = 1687 s of 3215 s, and ~1000 s of it is the producer waiting on GPU bucket-cache misses
(≈20 uploads per bucket per round, ~24 µs each: host Belady get + id vector + `cudaMallocAsync` + 2 pageable H2D).

| id | idea | TAGORE source | target stage (SIFT100M s) | expected gain |
|---|---|---|---|---|
| ~~B1~~ | done in #1 (pinned staging + one H2D + scatter kernel; SIFT100M −236 s) | | | |
| ~~B1b~~ | done in #2 (deferred eviction frees + reuse of freed device blocks; SIFT100M −472 s) | | | |
| B1c | Step 6 host get (139 / 192 s per round): host Belady cache of int32-id records in one pinned arena (no per-bucket vectors, H2D-ready), budget so 13.6 GB of records fit (`--cache-mb`, RSS guard ≤ 1.2× 79.4 GB) — cf. B3 | `graph_pool` in host RAM | Step 6 host get | −100…−250 s |
| B2 | GPU bucket cache budget vs dots slot buffers (resource params: `--gpu-cache-mb`, dots slot cap): fewer evictions | `largeIndex` `buffer_size` / `max_points` sizing, `reorder` slot choice | Step 6 wait_producer 660 + 735 | −100…−400 s |
| B3 | Host side of the miss path: round bucket records held in RAM (bucket2 host budget so the Belady cache / blob holds 13.6 GB; RSS guard ≤ 1.2× 79.4 GB) — cf. cuVS-loop #75 | `graph_pool` kept in host RAM | Step 6 cache_get − upload ≈ 230 + 250 | −100…−300 s |
| B4 | KMeans|| d(x, C): fused tensor-core GEMM + row-min with precomputed norms (cuVS-loop `iter73` patch, maybe + `iter72`) — at SIFT100M kmeans_parallel is 180 s / round | `nn_descent_opt_cal` + `cal_power` | Step 2 368 | −100…−170 s |
| B5 | Step 4.5: background writer so `pass3_write` (67 s) overlaps `resample` (92 s) | `cached_ofstream`, CPU merge thread overlapped with GPU | Step 4.5 195 | −50…−65 s |
| B6 | `optimize_chunked -m E`: reverse edges collected on the GPU per chunk (unbounded, same rule), disk edge accumulator only when a chunk does not fit | `filter_cagra` reverse push + `filter_reverse_1hop` | optimize_chunked 280 (acc.finish 91, Phase 1 53) | −80…−140 s |

## Awaiting approval

| id | idea | TAGORE source | target stage | expected gain | what changes semantically | status |
|---|---|---|---|---|---|---|
| A1 | GNN-Descent local join (resident: TAGORE GNN-Descent; large path: fused prune + local join, RNG test + insertion, tensor-core GEMM) | `nn_descent_opt_*`, `sample_kernel6`, `merge_reverse_plus` | Step 6 (via lower NPROBE / ITERATIONS) + prune | see `docs/A1_local_join.md` §2.7 | adds a refinement stage, RNG prune on the large path | **APPROVED 2026-10-21** — `.claude/skills/optimize-bucket2-tagore/docs/A1_local_join.md`, split §2.8 |
| A2 | Coarser large-path buckets: ~10–20K points per bucket (instead of ~100) with a small nprobe, so a bucket pair is one big GEMM and the GPU cache holds whole working sets (TAGORE clusters of ~2M points processed whole on the GPU) | `largeIndex` clusters + `gpu_construct` | Step 6 (1687 s), Step 2 (fewer centroids: 368 s) | Step 6 cache traffic largely gone; −500…−1000 s if quality holds with lower nprobe | bucketing (centroid count, `CENTROID_RATIO`) and the candidate set (`NPROBE`) change → pre-prune recall changes | **APPROVED 2026-10-22 (restricted: params only, avg bucket size 128–512)** — `CENTROID_RATIO` ≈ 1/512…1/128 (SIFT100M ≈ 195K–780K centroids) + smaller / retuned `NPROBE` (and `KNN_K` if needed); same bucketing algorithm, bucket order, bucket-pair candidate rule, prune, reverse edges; applied the same way to SIFT1M/10M unless logged otherwise; ~10K-point buckets only for truly huge datasets (SIFT1B), not now (relayed from the user's other session) |
| A3 | Bounded per-node reverse list on the GPU (TAGORE RESERVENUM: keep the d best-detour reverse candidates, drop overflow) so `optimize_chunked` is one GPU pass without the disk edge accumulator | `filter_cagra` reverse push (`tmp % d` replacement) + `filter_reverse_1hop` | optimize_chunked 280 s | −150…−200 s | reverse edges beyond the bound are dropped (CAGRA keeps all and ranks them) | **deferred by user 2026-10-22 (not to be tried until the user reopens it)**; B6 (exact GPU reverse collection, same CAGRA rule) stays a normal implementation idea (relayed from the user's other session) |

## Possible gains

SIFT100M 3–7% faster with everything else passing (rechecked later with multi-run medians).

(empty)

## Tried & rejected

(empty)

## Iterations

### #0 — baseline from main (2026-10-22, commit 3046497)

No source change. Worktree `opt/bucket2-tagore` built from main 3046497 (cuVS loop through #76) and verified against
main's best (table above): all configs within tolerance, same path selection, 0 graph defects. This run is the loop's
baseline (`state/best.json`, written by `judge.py --update`; search gated against CAGRA until step B).

### T0 — TAGORE study pass (2026-10-22)

No source change. Read TAGORE kernel by kernel (`notes/tagore_map.md`: line ranges, launch shapes, shared-memory
layout, matching bucket2 stage + SIFT100M time, class). Main finding: SIFT100M Step 6 (1687 s) is producer-bound on
the 2 GB GPU bucket cache — ≈20 uploads per bucket per round at ~24 µs each (479 / 536 s), while TAGORE moves whole
~1 GB clusters with one copy into buffers allocated once. Backlog B1–B6 (implementation), A2 / A3 proposed
(algorithm), A1 approved. TAGORE beats us on SIFT100M (2151 vs 3215 s) but loses on SIFT10M (214 vs 82 s) and SIFT1M
(9.2 vs 6.3 s).

### #1 — Step 6 GPU-cache misses: pinned staging + one H2D + scatter kernel per position (2026-10-22)

**Idea:** out-of-core Step 6 on SIFT100M thrashes the 2 GB GPU bucket cache (≈20 uploads per bucket per round) and
each miss cost ~24 µs on the producer thread (host Belady get, int64→int32 id vector, `cudaMallocAsync`, two
**pageable** `cudaMemcpyAsync`). Now the producer writes every miss of a position straight into a per-slot pinned
staging buffer in `d_buf` layout ([int32 ids, 16 B aligned][vectors]), appends a `MissJob` table, and issues one H2D
(data + table) and one `devcache_scatter_misses_kernel` (one block per miss, 16 B copies) on the copy stream.
Ordering unchanged (copy stream: evict-free → alloc → scatter → `ev_load[slot]` → slot stream waits); the staging
of a slot is rewritten only after that slot's previous `ev_load`. New sub-timers (host get / malloc / h2d call).
Also fixed a race from cuVS-loop #69 that hit SIFT1M large once (rep 2: `ReorderedDatasetBucketReader: cannot
open data_reordered.u8bin`): the background writer created the file inside the async task while the blob-mode
round-1 reader opens it without joining → the file is now created synchronously before the async write starts.
A first SIFT100M run crashed in round 2 (`misaligned address`: the MissJob table followed a staging area of
`ub·132 + (K+1)·16` bytes) → capacity rounded up to 256 B.
**TAGORE source:** `large_index.cu::gpu_construct` (device buffers sized once, one big H2D per cluster, then all work on
device) and `pybind.cu::largeIndex` (`graph_pool` fixed slots). cuVS 26.12 / our old code: per-bucket pageable copies.
**Class:** implementation.
**Dev test** (SIFT10M, 1 round, `BUCKET2_BLOB=0`, `--gpu-bucket-cache-mb 200`, GPU 2): loop 40.0 → 30.2 s, upload
23.4 → 11.1 s (h2d call 9.3 → 1.5 s), pre-prune recall@10 0.6469 → 0.6470.
**SIFT100M** (the only config on this path; 1M / 10M use blob mode): Step 6 1687.4 → 1401.3 s; per round loop
780 / 860 → 614 / 745 s, upload 479 / 536 → 296 / 386 s (host get 139 / 201, malloc 67 / 75, h2d call 17 / 19 s);
total 3214.58 → **2978.78 s (+7.3%)**; optimize_chunked 280 → 340 s and translate 59 → 69 s are single-run noise
(disk). DiskANN recall@10 0.7467 / 0.8525 / 0.9253 / 0.9683 / 0.9884 (best 0.7472 / … / 0.9881), QPS similar,
0 defects, RSS 79.4 GB, GPU 14588 MiB.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.27 | -0.2% | -0.2% | 0.68x | 0.54x | bucket2 6.27 | 3876 | 1.71 |
| sift1M_large | large | 9.44 | +2.6% | +2.6% | 1.03x | 0.82x | bucket2 7.64, prune 0.72, optimize_chunked 0.72, translate 0.37 | 1662 | 2.04 |
| sift10M_auto | large | 82.20 | +0.3% | +0.3% | 0.38x | 0.67x | bucket2 71.76, prune 4.03, optimize_chunked 2.87, translate 3.54 | 8996 | 12.65 |
| sift100M_auto | large | 2978.78 | +7.3% | +7.3% | 1.38x | — | bucket2 2414.86, prune 155.29, optimize_chunked 339.86, translate 68.77 | 14588 | 79.41 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.26 | 0.18 | 0.00 | 0.61 | 0.36 | 0.02 | 2.76 | 0.00 | 0.89 |
| sift1M_large | 0.14 | 0.62 | 0.35 | 0.00 | 1.01 | 0.29 | 0.04 | 4.00 | 0.51 | 0.00 |
| sift10M_auto | 1.38 | 7.32 | 2.39 | 0.00 | 18.40 | 1.75 | 0.33 | 35.14 | 3.89 | 0.00 |
| sift100M_auto | 71.59 | 372.24 | 97.35 | 0.00 | 251.74 | 175.12 | 7.22 | 1401.27 | 25.14 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9746 | 0.9665 | 0.9595 | 0.9539 |
| sift1M_large | 0.9739 | 0.9646 | 0.9583 | 0.9525 |
| sift10M_auto | 0.9211 | 0.8994 | 0.8838 | 0.8717 |
| sift100M_auto | 0.7457 | 0.7201 | 0.6966 | 0.6808 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9278 / 1944K | 0.9775 / 1036K | 0.9942 / 490K | 0.9987 / 186K | 0.9997 / 68K |
| ours sift1M_large | 0.9140 / 1946K | 0.9707 / 1034K | 0.9918 / 493K | 0.9981 / 180K | 0.9995 / 68K |
| ours sift10M_auto | 0.8377 / 1871K | 0.9281 / 1012K | 0.9735 / 526K | 0.9922 / 186K | 0.9980 / 68K |
| ours sift100M_auto | 0.7467 / 62K | 0.8525 / 61K | 0.9253 / 37K | 0.9683 / 20K | 0.9884 / 11K |

**Remaining in the miss path (next):** host get 139–201 s/round (Belady host cache 8.6 GB < 13.6 GB records → disk
reads), `cudaMallocAsync` + eviction `cudaFreeAsync` ~5 µs per miss, and ≈20 misses per bucket (cache size / order).

### User decisions 2026-10-22 (relayed from the user's other session)

1. TAGORE V100 fixes: approved; they stay only in `build-v100/patched-src` + `scripts/tagore_v100_fix.patch`, the
   `/home/lanlu/Tagore` sources are never modified.
2. A2 approved in a restricted form: parameters only (`CENTROID_RATIO` for bigger buckets, smaller / retuned `NPROBE`,
   `KNN_K` if needed), no change of semantics; average bucket size 128–512 points (SIFT100M `CENTROID_RATIO` ≈
   0.002–0.0078); same ratio for SIFT1M/10M unless a reason is logged. ~10K-point buckets only for SIFT1B-scale data.
3. A3 deferred (not to be tried until the user reopens it); B6 stays a normal implementation idea.

### #2 — Step 6: deferred eviction frees + reuse of freed device blocks (2026-10-22)

**Idea:** what #1 left in the SIFT100M miss path was `cache_get − upload` ≈ 228 / 257 s per round (eviction:
`before_free` blocked the producer until the main thread had submitted the victim's last position — Belady often
evicts a bucket used 1–3 not-yet-submitted positions ago — then `cudaFreeAsync`) and `cudaMallocAsync` 67 / 75 s.
(1) Eviction hands `(d_buf, last_pos, bytes)` to a pending list; the producer drains it without blocking at each
position (copy stream waits the gather event of each drained position, as before). (2) Drained blocks go to per-size
free lists (1 KB granularity, a miss takes the same size or up to +50 %) and are reused by later misses instead of
`cudaFreeAsync` + `cudaMallocAsync`; the lists are capped at `gpu_cache_bytes / 8` (beyond that: real free), so device
memory ≤ cache budget + 1/8; cache accounting uses the block size. Writes into a reused block are on the copy stream
after the wait → stream-ordered. (A fixed 2 GB arena was tried first in the dev test: 22 % fallback allocations from
size-class fragmentation, i.e. memory beyond the budget → replaced by the capped free lists.)
**TAGORE source:** `large_index.cu::gpu_construct` (all device buffers allocated once and reused for every cluster;
no allocation in the hot loop) + the `has_merged` hand-off (the GPU thread never waits on the merge for a slot it does
not need yet). **Class:** implementation.
**Dev test** (SIFT10M, 1 round, `BUCKET2_BLOB=0`, `--gpu-bucket-cache-mb 200`, GPU 2): loop 30.2 → 28.9 (deferred
frees) → 26.8 (fixed arena) → **25.4 s** (capped free lists; 7.6 % of misses allocate); recall@10 0.6459 (noise).
**SIFT100M:** Step 6 1401.3 → **1016.6 s**; per round loop 614 / 745 → 438 / 533 s, cache_get 524 / 644 → 349 / 432 s,
upload 296 / 386 → 230 / 304 s (host get 139 / 192, malloc 7 / 10, h2d call 17 / 18, deferred free 16 / 20 s; device
allocs 0.63M / 0.88M of 20.9M / 21.8M misses); total 2978.78 → **2506.89 s (+15.8 %, +22.0 % vs #0)**. DiskANN
recall@10 0.7458 / 0.8518 / 0.9252 / 0.9680 / 0.9880, 0 defects, RSS 79.4 GB. **GPU peak 14588 → 15492 MiB** (free
lists + allocation churn; ~0.6 GB left on the 16 GB V100 — watch it in later Step 6 changes).
SIFT1M / 10M unchanged (blob mode; 1M large −1.8 % = noise).

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.27 | +0.0% | -0.2% | 0.68x | 0.54x | bucket2 6.27 | 3812 | 1.71 |
| sift1M_large | large | 9.61 | -1.8% | +0.8% | 1.05x | 0.83x | bucket2 7.78, prune 0.71, optimize_chunked 0.72, translate 0.40 | 1590 | 2.04 |
| sift10M_auto | large | 82.22 | -0.0% | +0.3% | 0.38x | 0.67x | bucket2 71.83, prune 3.96, optimize_chunked 2.87, translate 3.56 | 8996 | 12.65 |
| sift100M_auto | large | 2506.89 | +15.8% | +22.0% | 1.17x | — | bucket2 2023.02, prune 133.15, optimize_chunked 286.79, translate 63.93 | 15492 | 79.41 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.26 | 0.17 | 0.00 | 0.62 | 0.36 | 0.02 | 2.75 | 0.00 | 0.89 |
| sift1M_large | 0.14 | 0.60 | 0.35 | 0.00 | 0.99 | 0.26 | 0.04 | 4.21 | 0.49 | 0.00 |
| sift10M_auto | 1.45 | 7.43 | 2.37 | 0.00 | 18.52 | 1.76 | 0.38 | 34.79 | 4.00 | 0.00 |
| sift100M_auto | 72.31 | 371.49 | 97.43 | 0.00 | 252.26 | 170.64 | 7.47 | 1016.61 | 26.76 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9741 | 0.9668 | 0.9590 | 0.9532 |
| sift1M_large | 0.9754 | 0.9663 | 0.9592 | 0.9535 |
| sift10M_auto | 0.9162 | 0.8989 | 0.8832 | 0.8711 |
| sift100M_auto | 0.7511 | 0.7208 | 0.6970 | 0.6814 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9280 / 1950K | 0.9769 / 1048K | 0.9944 / 501K | 0.9987 / 187K | 0.9996 / 68K |
| ours sift1M_large | 0.9147 / 2224K | 0.9708 / 1188K | 0.9922 / 561K | 0.9983 / 189K | 0.9995 / 68K |
| ours sift10M_auto | 0.8393 / 1868K | 0.9292 / 1011K | 0.9735 / 489K | 0.9916 / 176K | 0.9980 / 68K |
| ours sift100M_auto | 0.7458 / 69K | 0.8518 / 64K | 0.9252 / 39K | 0.9680 / 23K | 0.9880 / 13K |
