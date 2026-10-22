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

After **#5 (optimize_chunked disk path without the edge merge copy; accepted by user decision)**, 2026-10-22 (`runs/tagore5`):

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 5.17 | -0.8% | +17.4% | 0.56x | 0.45x | bucket2 5.17 | 4784 | 1.68 |
| sift1M_large | large | 8.61 | +3.0% | +11.1% | 0.94x | 0.75x | bucket2 6.78, prune 0.72, optimize_chunked 0.73, translate 0.38 | 1286 | 1.98 |
| sift10M_auto | large | 74.22 | +0.1% | +10.0% | 0.35x | 0.61x | bucket2 63.80, prune 3.88, optimize_chunked 2.97, translate 3.57 | 8996 | 11.74 |
| sift100M_auto | large | 1336.36 | +6.8% | +58.4% | 0.62x | — | bucket2 941.72, prune 144.98, optimize_chunked 191.88, translate 57.78 | 14588 | 79.13 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.23 | 0.10 | 0.00 | 0.40 | 0.21 | 0.02 | 2.06 | 0.00 | 0.97 |
| sift1M_large | 0.14 | 0.57 | 0.22 | 0.00 | 0.87 | 0.17 | 0.02 | 3.56 | 0.61 | 0.00 |
| sift10M_auto | 1.41 | 3.71 | 0.76 | 0.00 | 18.14 | 1.70 | 0.22 | 31.45 | 5.38 | 0.00 |
| sift100M_auto | 71.99 | 115.03 | 14.38 | 0.00 | 217.27 | 134.34 | 5.09 | 346.53 | 28.92 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9555 | 0.9436 | 0.9338 | 0.9264 |
| sift1M_large | 0.9548 | 0.9448 | 0.9344 | 0.9269 |
| sift10M_auto | 0.8859 | 0.8579 | 0.8389 | 0.8258 |
| sift100M_auto | 0.6914 | 0.6585 | 0.6326 | 0.6161 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9269 / 1946K | 0.9764 / 1035K | 0.9941 / 500K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9138 / 2219K | 0.9704 / 1189K | 0.9920 / 561K | 0.9982 / 189K | 0.9994 / 68K |
| ours sift10M_auto | 0.8328 / 1870K | 0.9247 / 1010K | 0.9714 / 467K | 0.9910 / 185K | 0.9977 / 68K |
| ours sift100M_auto | 0.7331 / 67K | 0.8423 / 63K | 0.9171 / 38K | 0.9631 / 23K | 0.9855 / 13K |

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
| B7 | (user idea, relayed 2026-10-22) **Measure the change ratio first:** per round ≥ 2 log the fraction of points with cnt > 0 in the arrival and the fraction whose merged top-M actually changed (+ arrival bytes vs fixed width, printed since #4: SIFT100M round 2 44.1 / 52.0 GB, SIFT10M rounds 2–3 4.52 / 5.20 GB) — decides whether B8 is worth it | `order_merge_new` | merge stats | 0 (measurement) |
| B8 | (user idea, relayed 2026-10-22) **Sparse running rewrite** — "只重写这一轮真的变了的那些行。每个 chunk 里先按 id 排序，再成段写回，而不是把整个 running 重写一遍": rewrite only rows whose top-M changed, sorted by id per chunk, written back as contiguous runs (pwrite) | `order_merge_new` (only the touched lists are rewritten) | merge iter ≥ 1 (matters when ITERATIONS ≥ 3) | depends on B7 |
| B9 | (user idea, relayed 2026-10-22) **Generalize #4's deferral:** after each round ≥ 2 keep all not-yet-merged arrival files while their total ≤ one full first-round arrival (≈ running size); merge (running + all pending → running, or → vector_knn at the end) only when the bound would be exceeded or at the final round. Open points: next-round threshold = read-only pass over running + pending (like #4) vs keep the older looser kth (exact too, more arrival bytes) — measure; final merge reads pending files one at a time into the same buffer (per-thread memory unchanged, RSS guard) | `order_merge_new` (merge in one final pass) | merges for ITERATIONS ≥ 3 | SIFT10M merges / disk I/O |
| B9a | (sub-item of B9, user decision relayed 2026-10-22, **low priority** — only ITERATIONS ≥ 3) next-round threshold while running is not rewritten — **implement C, or measure A / B / C and keep the fastest**; log the measured cost of each and the choice. A: read-only pass per round over running + pending arrivals per chunk (exact kth, no write; reads ~running size). B: keep the older, looser kth (exact result; later arrivals grow and hit the pending bound sooner). C: keep in host memory a tail of up to r of the last distances of each point's top-M (+ its current length t; r = 8 → N·r·4 B = 3.2 GB at SIFT100M, counted in the RSS guard); with j new candidates (all < old kth) the new kth is exact only if j < t (value at old rank M − j), tail length becomes t − j; else fall back to B for that point; refill the tail to r whenever running is actually rewritten; log fall-backs per round. New candidates' distances come from the arrival records (update at write time or at round end) | `order_merge_new` | B9 thresholds | all keep vector_knn identical |
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

| # | idea | TAGORE source | SIFT100M | other configs | patch |
|---|---|---|---|---|---|
| — | (none open; #5 was accepted by the user on its own) | | | | |

## Tried & rejected

| # | idea | result | reason / patch |
|---|---|---|---|
| 6 | `optimize_chunked -m E` Phase 2: cache-friendly CSR (per-thread block histograms, 64-bit keys (row<<5|rank)<<32|src, per-block sort; byte-identical, fallback to the old build) + no-zero-fill allocator for the edge / key / CSR buffers + Phase 2 sub-timers (`EdgeChunkAccumulator`, `build_rank_ordered_csr_fast`; TAGORE `reorder` / `order_merge_new` block-wise processing) | SIFT100M 1463.29 s vs best 1336.36 (**−9.5 %**): Phase 2 116 → 203 s with breakdown read_fwd 49.4 / read_edges 123.6 / csr 11.3 / merge 2.7 / write 12.1 s — the CSR part improved (~53 → 11 s estimated from SIFT10M ×10) but both reads ran at ~260 MB/s = raw disk speed (files not in page cache in this run; 207 GB free afterwards, cause unknown: environment or the untouched buffers). SIFT10M forced disk path: Phase 2 14.6–17.8 → 10.1–10.6 s, byte-identical. 1M auto −0.2 %, 1M large −1.0 % (rerun; first run 8.96 s was −4.1 % vs the optimistic #5 best, difference only in untouched bucket2), 10M +1.6 %; recall unchanged | judge: SIFT100M regression > 7 % | `bucketDemo/patches/tagore6_csr_radix_noinit.patch` — recheck: CSR part alone (keep zero-fill) and a multi-run SIFT100M |

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

### #3 — A2 (approved, params only): CENTROID_RATIO 0.01 → 0.004, NPROBE 200 → 80 + Step 6 dots row tiling (2026-10-22)

**Idea:** A2 as restricted by the user: bigger buckets by parameters only (avg ~100 → ~250 points, inside 128–512),
`NPROBE` lowered so the pool ((NPROBE+1) × bucket ≈ 20K rows) stays about the same; same bucketing algorithm, order,
candidate rule, prune, reverse edges; applied to all datasets. Fewer centroids make Step 2 (KMeans||), Step 3
(centroid KNN), Step 4 and every per-bucket overhead of Step 6 (cache entries, misses, launches) cheaper.
`params.sh` now takes env overrides (`: "${X:=default}"`) for `CENTROID_RATIO`, `KNN_K`, `NPROBE` (exploration only;
benchmark runs set none).
**Exploration** (SIFT10M, 1 run each, under the bench lock, `runs/tagore3/explore10M_*`):

| `CENTROID_RATIO` / `NPROBE` | avg bucket | SIFT10M build (s) | CAGRA recall@10 itopk 32 / 64 / 128 | pre-prune recall@10 |
|---|---|---|---|---|
| 0.01 / 200 (best #2) | 100 | 82.22 | 0.8377 / 0.9281 / 0.9735 | 0.8994 |
| **0.004 / 80** | 250 | 73.38 | 0.8333 / 0.9255 / 0.9720 | 0.8577 |
| 0.004 / 64 | 250 | 70.58 | 0.8302 / 0.9223 / 0.9698 | 0.8335 |
| 0.0025 / 50 | 400 | 77.43 | 0.8304 / 0.9227 / 0.9710 | 0.8319 |
| 0.0025 / 40 | 400 | 65.80 | 0.8265 / 0.9186 / 0.9686 | 0.8079 |
| 0.002 / 32 | 500 | 69.18 | 0.8231 / 0.9174 / 0.9680 | 0.7950 |

0.004 / 80 kept the pool size and lost the least quality → benchmarked.
**First benchmark (`runs/tagore3/try1_untiled`): rejected by the memory guard** — sift1M_large GPU peak 2230 MiB >
1.2 × 1590: the out-of-core Step 6 `dots` buffer is max_bucket × max_pool per slot (0.19 → 0.77 GB), and at SIFT100M
(15.5 GB peak already) it would not have fit. (Speed was 1M auto +18.0 %, 1M large +8.4 %, 10M +5.0 %.)
**Fix bundled into this iteration (implementation, exact):** Step 6 computes each bucket in row tiles of ≤ 256 rows
(one GEMM with A = pool rows [r0, r0+rows) + one block top-M per tile; the kernel takes `row0`, q stays the bucket
row; rows are independent, so results are identical). `dots` = min(max_bucket, 256) × max_pool → 0.06 GB per slot on
SIFT1M large (GPU peak 1286 MiB), 0.14 / 0.17 GB on SIFT100M (peak 14588 MiB). Raft fallback (M > 256) only for buckets
≤ 256 rows (checked). TAGORE source for the tiling: `gpu_construct` (fixed-size device buffers).
**Class:** A2 (approved 2026-10-22, restricted) + implementation (tiling).
**Results:** SIFT1M auto 5.16 s (+17.7 %), 1M large 9.02 s (+6.1 %), 10M 78.14 s (+5.0 %), **SIFT100M 2506.89 →
1666.75 s (+33.5 %, +48.2 % vs #0; now 0.77× TAGORE's 2151 s)**: Step 2 371 → 122, Step 3 97 → 15, Step 4 252 → 232,
Step 4.5 171 → 133, Step 6 1017 → 477 s (400K buckets; per round 4.17M uploads, loop 163 / 302 s).
**Quality cost (for the user):** DiskANN recall@10 SIFT100M 0.7458 / 0.8518 / 0.9252 / 0.9680 / 0.9880 →
0.7331 / 0.8401 / 0.9157 / 0.9627 / 0.9856 (−0.013 … −0.002, gate −0.05); pre-prune recall@10 0.720 → 0.656
(100M), 0.899 → 0.859 (10M); CAGRA-search recall@10 @32 1M auto 0.9278 → 0.9268, 1M large 0.9140 → 0.9136, 10M 0.8377 →
0.8335 (TAGORE floors 0.912 / 0.872 / 0.795). GPU peak 1M auto 3812 → 5244 MiB (resident path, not gated; bigger
resident buckets), others ≤ best. optimize_chunked 287 → 355 s on SIFT100M (single-run noise; investigate if it stays).

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 5.16 | +17.7% | +17.6% | 0.56x | 0.45x | bucket2 5.16 | 5244 | 1.68 |
| sift1M_large | large | 9.02 | +6.1% | +6.9% | 0.98x | 0.78x | bucket2 7.21, prune 0.69, optimize_chunked 0.71, translate 0.48 | 1286 | 1.96 |
| sift10M_auto | large | 78.14 | +5.0% | +5.2% | 0.36x | 0.64x | bucket2 67.95, prune 3.72, optimize_chunked 2.89, translate 3.58 | 8996 | 12.25 |
| sift100M_auto | large | 1666.75 | +33.5% | +48.2% | 0.77x | — | bucket2 1107.95, prune 145.90, optimize_chunked 354.85, translate 58.05 | 14588 | 79.13 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.23 | 0.09 | 0.00 | 0.39 | 0.23 | 0.02 | 2.06 | 0.00 | 0.93 |
| sift1M_large | 0.14 | 0.48 | 0.26 | 0.00 | 0.96 | 0.17 | 0.04 | 3.71 | 0.51 | 0.00 |
| sift10M_auto | 1.40 | 3.63 | 0.52 | 0.00 | 17.96 | 1.72 | 0.23 | 37.52 | 4.03 | 0.00 |
| sift100M_auto | 72.62 | 121.79 | 14.59 | 0.00 | 231.78 | 133.20 | 5.74 | 477.38 | 42.70 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9547 | 0.9436 | 0.9341 | 0.9268 |
| sift1M_large | 0.9557 | 0.9448 | 0.9336 | 0.9261 |
| sift10M_auto | 0.8827 | 0.8587 | 0.8394 | 0.8257 |
| sift100M_auto | 0.6833 | 0.6560 | 0.6311 | 0.6155 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9268 / 1947K | 0.9770 / 1034K | 0.9943 / 533K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9136 / 2219K | 0.9703 / 1190K | 0.9924 / 561K | 0.9982 / 189K | 0.9996 / 68K |
| ours sift10M_auto | 0.8335 / 1872K | 0.9247 / 1010K | 0.9713 / 501K | 0.9911 / 185K | 0.9977 / 68K |
| ours sift100M_auto | 0.7331 / 61K | 0.8401 / 63K | 0.9157 / 38K | 0.9627 / 22K | 0.9856 / 13K |

### #4 — penultimate merge does not write running + variable-length arrival records (2026-10-22)

**Idea:** after #3, SIFT100M round 2 Step 6 was 302 s wall but only 166 s inside the loop timers: the scatter worker
(`ChunkedKnnAccumulator::add`, 954 chunk files, 255 KB `ofstream` writes) took 297 s and drained 136 s after the
loop. Direct writes on this volume run at **258 MB/s** (`dd oflag=direct`), and round 2's 52 GB of arrival records
competed with the writeback of the round-1 merge's 51 GB running files (`merge_iteration(iter=0)`: 103 GB moved,
145 s, during round 2 Steps 2–4). (1) When the next merge is the final one (`planned_final_iter`), the merge only
builds the merged lists in memory to export `kth` (the next round's top-M threshold), does **not** write running
files and keeps its arrival file; the final merge rebuilds running from the older running files (if any) + the kept
arrival, then merges the last round's arrival. For ITERATIONS = 2 (SIFT100M) the running files are never written.
(2) Arrival records are variable length: `[int64 gid][uint32 cnt][cnt ids][cnt dists]`, cnt = last valid + 1; the
merge pads back to M with −1/+inf → identical merge input (saves ~15 % in rounds ≥ 2: SIFT100M 44.1 vs 52.0 GB,
SIFT10M 4.52 vs 5.20 GB). Final `vector_knn` unchanged.
**TAGORE source:** `large_index.cu::order_merge_new` (local graphs stay in the buffer / on disk and are merged once in
a final streaming pass) + `cached_ofstream` (big writes, fewer passes over the disk). **Class:** implementation.
**Checks:** SIFT10M smoke (3 rounds: iter 1 deferred) pre-prune recall@10 0.8559 vs 0.8587 (#3 run), SIFT1M large
`-t 2` vs a HEAD build 0.8815 vs 0.8811 (run-to-run noise); no leftover chunk files.
**SIFT100M:** round 2 scatter 297 → 85 s (tail drain 136 → 0 s), merge iter 0 145 → 6 s, final merge 43 → 28 s;
Step 6 477 → 335 s, bucket2 1108 → 943 s; total 1666.75 → **1434.63 s (+13.9 %, +55.4 % vs #0; 0.67× TAGORE)**;
optimize_chunked 355 → 293 s (less disk contention). DiskANN recall@10 0.7353 / 0.8402 / 0.9166 / 0.9631 / 0.9859
(best 0.7331 / … / 0.9856), RSS 79.1 GB, GPU 14588 MiB. SIFT10M +4.9 % (third round's penultimate merge), 1M ±1 %.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 5.13 | +0.6% | +18.1% | 0.56x | 0.44x | bucket2 5.13 | 5332 | 1.68 |
| sift1M_large | large | 8.88 | +1.6% | +8.4% | 0.97x | 0.77x | bucket2 6.96, prune 0.72, optimize_chunked 0.72, translate 0.46 | 1286 | 2.00 |
| sift10M_auto | large | 74.30 | +4.9% | +9.9% | 0.35x | 0.61x | bucket2 64.12, prune 3.85, optimize_chunked 2.90, translate 3.43 | 8996 | 11.72 |
| sift100M_auto | large | 1434.63 | +13.9% | +55.4% | 0.67x | — | bucket2 943.15, prune 140.80, optimize_chunked 292.89, translate 57.79 | 14588 | 79.13 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.23 | 0.09 | 0.00 | 0.40 | 0.23 | 0.02 | 2.05 | 0.00 | 0.94 |
| sift1M_large | 0.14 | 0.60 | 0.26 | 0.00 | 0.80 | 0.22 | 0.02 | 3.60 | 0.63 | 0.00 |
| sift10M_auto | 1.42 | 4.04 | 0.57 | 0.00 | 18.59 | 1.71 | 0.21 | 31.84 | 4.82 | 0.00 |
| sift100M_auto | 72.03 | 128.15 | 14.31 | 0.00 | 220.92 | 132.18 | 5.12 | 335.10 | 27.81 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9603 | 0.9441 | 0.9345 | 0.9268 |
| sift1M_large | 0.9532 | 0.9433 | 0.9324 | 0.9248 |
| sift10M_auto | 0.8811 | 0.8589 | 0.8392 | 0.8255 |
| sift100M_auto | 0.6885 | 0.6568 | 0.6325 | 0.6158 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9267 / 1947K | 0.9765 / 1033K | 0.9942 / 532K | 0.9987 / 177K | 0.9997 / 68K |
| ours sift1M_large | 0.9140 / 2221K | 0.9707 / 1188K | 0.9920 / 562K | 0.9980 / 182K | 0.9996 / 68K |
| ours sift10M_auto | 0.8337 / 1871K | 0.9247 / 1011K | 0.9718 / 481K | 0.9910 / 184K | 0.9978 / 68K |
| ours sift100M_auto | 0.7353 / 68K | 0.8402 / 63K | 0.9166 / 39K | 0.9631 / 22K | 0.9859 / 13K |

### #5 — optimize_chunked disk path without the merge copy (2026-10-22) — **ACCEPTED by user decision** (first judged as a possible gain)

`acc.finish()` copied the per-chunk edge files into one merged file before Phase 2 (SIFT100M 107 s); now Phase 2 reads
each chunk file directly. The stage gain is deterministic (optimize_chunked 293 → 192 s) but the SIFT100M total is
+6.85 % (1434.63 → 1336.36 s) — below the 7 % single-run gate — and the only ≥ 3 % config (1M large +3.04 %) did not
hold on the rerun (+1.01 %; this change does not run there). Rejected per the rules, listed under **Possible gains**
for the multi-run recheck; patch kept. Good candidate to bundle with the next optimize_chunked / merge change.
**User decision (2026-10-22, in chat): accept #5 on its own.** `best.json` updated from the first judge set
(`runs/tagore5/{sift1M_auto,sift1M_large,sift10M_auto,sift100M_auto}`, verdict ACCEPT); note that its sift1M_large
entry (+3.0 %) is the faster of the two runs (rerun `sift1M_large_b` +1.0 %), so later 1M-large comparisons start from a
slightly optimistic best. SIFT100M best now **1336.36 s** (+58.4 % vs #0, 0.62× TAGORE).

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 5.17 | -0.8% | +17.4% | 0.56x | 0.45x | bucket2 5.17 | 4784 | 1.68 |
| sift1M_large | large | 8.61 | +3.0% | +11.1% | 0.94x | 0.75x | bucket2 6.78, prune 0.72, optimize_chunked 0.73, translate 0.38 | 1286 | 1.98 |
| sift10M_auto | large | 74.22 | +0.1% | +10.0% | 0.35x | 0.61x | bucket2 63.80, prune 3.88, optimize_chunked 2.97, translate 3.57 | 8996 | 11.74 |
| sift100M_auto | large | 1336.36 | +6.8% | +58.4% | 0.62x | — | bucket2 941.72, prune 144.98, optimize_chunked 191.88, translate 57.78 | 14588 | 79.13 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.23 | 0.10 | 0.00 | 0.40 | 0.21 | 0.02 | 2.06 | 0.00 | 0.97 |
| sift1M_large | 0.14 | 0.57 | 0.22 | 0.00 | 0.87 | 0.17 | 0.02 | 3.56 | 0.61 | 0.00 |
| sift10M_auto | 1.41 | 3.71 | 0.76 | 0.00 | 18.14 | 1.70 | 0.22 | 31.45 | 5.38 | 0.00 |
| sift100M_auto | 71.99 | 115.03 | 14.38 | 0.00 | 217.27 | 134.34 | 5.09 | 346.53 | 28.92 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9555 | 0.9436 | 0.9338 | 0.9264 |
| sift1M_large | 0.9548 | 0.9448 | 0.9344 | 0.9269 |
| sift10M_auto | 0.8859 | 0.8579 | 0.8389 | 0.8258 |
| sift100M_auto | 0.6914 | 0.6585 | 0.6326 | 0.6161 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| TAGORE GNN-Descent k=96 + CAGRA prune, sift1M | 0.9220 / 1947K | 0.9752 / 1033K | 0.9936 / 497K | 0.9985 / 186K | 0.9996 / 68K |
| TAGORE largeIndex Vamana, sift10M | 0.8448 / 1875K | 0.9238 / 1014K | 0.9662 / 482K | 0.9863 / 185K | 0.9950 / 68K |
| TAGORE largeIndex Vamana, sift100M | 0.7691 / 61K | 0.8615 / 55K | 0.9238 / 34K | 0.9616 / 20K | 0.9820 / 11K |
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9269 / 1946K | 0.9764 / 1035K | 0.9941 / 500K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9138 / 2219K | 0.9704 / 1189K | 0.9920 / 561K | 0.9982 / 189K | 0.9994 / 68K |
| ours sift10M_auto | 0.8328 / 1870K | 0.9247 / 1010K | 0.9714 / 467K | 0.9910 / 185K | 0.9977 / 68K |
| ours sift100M_auto | 0.7331 / 67K | 0.8423 / 63K | 0.9171 / 38K | 0.9631 / 23K | 0.9855 / 13K |

### #6 — REJECTED: optimize_chunked Phase 2 cache-friendly CSR + no-zero-fill buffers (2026-10-22)

See **Tried & rejected** (#6). The CSR build is clearly faster, but the single SIFT100M run had Phase 2's file reads at
disk speed (read_edges 124 s, read_fwd 49 s), so the total regressed 9.5 %. Next attempt: the CSR part only
(keep the zero-filled buffers, which may matter for read speed), and/or overlap the next chunk's read with the CSR /
merge / write of the current one.
