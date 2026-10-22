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

After **#0 (baseline from main 3046497)**, 2026-10-22 (`runs/tagore0`):

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.26 | — | — | — | 0.54x | bucket2 6.26 | 3900 | 1.71 |
| sift1M_large | large | 9.69 | — | — | — | 0.84x | bucket2 7.78, prune 0.74, optimize_chunked 0.73, translate 0.45 | 1760 | 2.04 |
| sift10M_auto | large | 82.44 | — | — | — | 0.67x | bucket2 72.01, prune 4.06, optimize_chunked 2.93, translate 3.44 | 8996 | 12.65 |
| sift100M_auto | large | 3214.58 | — | — | — | — | bucket2 2725.61, prune 149.99, optimize_chunked 280.23, translate 58.75 | 14588 | 79.41 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.27 | 0.17 | 0.00 | 0.62 | 0.39 | 0.02 | 2.76 | 0.00 | 0.91 |
| sift1M_large | 0.13 | 0.57 | 0.43 | 0.00 | 1.07 | 0.27 | 0.04 | 4.19 | 0.48 | 0.00 |
| sift10M_auto | 1.37 | 7.46 | 2.38 | 0.00 | 18.61 | 1.74 | 0.34 | 34.70 | 4.25 | 0.00 |
| sift100M_auto | 71.60 | 368.55 | 96.33 | 0.00 | 254.03 | 194.89 | 7.36 | 1687.35 | 32.33 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9725 | 0.9659 | 0.9591 | 0.9532 |
| sift1M_large | 0.9708 | 0.9647 | 0.9584 | 0.9526 |
| sift10M_auto | 0.9225 | 0.8982 | 0.8825 | 0.8716 |
| sift100M_auto | 0.7491 | 0.7204 | 0.6977 | 0.6816 |

**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9276 / 1945K | 0.9770 / 1035K | 0.9943 / 496K | 0.9987 / 187K | 0.9997 / 68K |
| ours sift1M_large | 0.9135 / 2213K | 0.9709 / 1190K | 0.9922 / 561K | 0.9983 / 189K | 0.9996 / 68K |
| ours sift10M_auto | 0.8374 / 1865K | 0.9287 / 1012K | 0.9733 / 525K | 0.9913 / 186K | 0.9980 / 68K |
| ours sift100M_auto | 0.7472 / 69K | 0.8536 / 66K | 0.9257 / 39K | 0.9686 / 23K | 0.9881 / 13K |

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

(filled by step B)

## Ideas backlog

(empty — filled by step T0)

## Awaiting approval

(empty — A1 is pre-approved, see `.claude/skills/optimize-bucket2-tagore/docs/A1_local_join.md`; entered in T0)

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
