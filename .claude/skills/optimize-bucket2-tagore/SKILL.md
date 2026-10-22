---
name: optimize-bucket2-tagore
description: One iteration of the bucket2 build-pipeline optimization loop (bucket2 -> optimize | prune_windowed+optimize_chunked -> translate_ids) in bucketDemo/buildBucket, focused on the SIFT100M build (2 rounds, DiskANN CPU search) and learning from TAGORE (GNN-Descent, CFS GPU pruning, async GPU-CPU-disk large-scale pipeline; source /home/lanlu/Tagore). Own branch opt/bucket2-tagore (from main), own state/best.json, own log bucketDemo/OPTIMIZATION_LOG_TAGORE.md. First invocation sets up + verifies the main baseline (SIFT1M/10M/100M), second runs TAGORE itself on SIFT1M/10M/100M as the build + search reference, third is a TAGORE study pass, later ones implement one TAGORE-derived idea (implementation ideas, or approved algorithm ideas such as A1 = GNN-Descent local join / fused prune+local join, docs/A1_local_join.md), benchmark SIFT1M/10M (quality checks, vs TAGORE) and SIFT100M (vs own best) on GPU 0, and accept/reject with judge.py. Algorithm/semantic changes are only proposed in the log until the user approves them. Run repeatedly with `/loop /optimize-bucket2-tagore`.
---

# bucket2 optimization loop, TAGORE edition: one step per invocation

Sister skill of `optimize-bucket2` (which learned from cuVS/raft 26.12). Same pipeline and
invariants, but **separate everything**: own branch, own `state/best.json`, own scripts copy,
own log, so the two studies can be summarized separately. Runs **fully automatically** (no
confirmation per round). Each invocation does exactly one step, then returns;
`/loop /optimize-bucket2-tagore` keeps it going until the user stops it. Never stop the loop
on your own: the only stop condition is a failed baseline verification in step P. In the
optimization iterations a failing idea (build error, crash, judge reject) is logged with its
exact failure, reverted, and the loop continues with the next idea.

**Own worktree + shared bench lock (user decision 2026-10-22).** This loop lives in the git
worktree **`/home/lanlu/microBenchmark_tagore`** (branch `opt/bucket2-tagore`, own `src/`, own
`build/`, own `runs/`); run it from a Claude session started in that directory. The cuVS loop
(`/home/lanlu/microBenchmark`, branch `opt/bucket2-loop`) may be alive at the same time — editing,
building and smoke tests can overlap — but **timed runs queue on one lock**:
`run_pipeline.sh` / `resume_large_tail.sh` take `/home/lanlu/.bench_locks/bucket2_bench.lock`
(flock) and print `[bench lock] busy, waiting: <owner>` while the other loop holds it
(`cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`). Every other timed run (TAGORE in step B,
cagra_eval / DiskANN baselines) goes through `flock /home/lanlu/.bench_locks/bucket2_bench.lock <cmd>`.
While the other loop holds the lock: build with `nice -n 19 taskset -c 1-79:2 ninja -j 8 -C build`,
smoke-test with `SMOKE=1 bash $S/run_pipeline.sh -d sift1M ...` (no lock, GPU 2, NUMA node 1 cores,
nice 19, 8 threads; judge.py rejects SMOKE dirs). Never run anything unlocked on GPU 0. Never edit
files in `/home/lanlu/microBenchmark` (the cuVS loop's tree) except reading it.

## Goal (user decision 2026-10-22)

**Make the SIFT100M build faster.** SIFT1M / SIFT10M are quality checks that keep their own
speed rules and are compared with **TAGORE** (not CAGRA).
- `sift100M_auto`: built with **2 rounds** (`ITERATIONS=2`, run_pipeline sets it for sift100M),
  **1 run** (no median for now), the merged 2-round index searched with **DiskANN's CPU in-memory
  searcher** (`scripts/diskann_search.py`: converts final_graph.bin to DiskANN's format, medoid
  entry point, L = 32…512, recall@10 + QPS, defect counts). Compared with **its own best**
  (this loop's large-path 2-round build + search):
  - build time ≥ **7%** faster = gain, > **7%** slower = regression (one run is noisy);
  - a 3–7% SIFT100M gain is **not accepted**, but logged under **Possible gains** with its
    numbers: the user will later switch the skill to multi-run medians and recheck those ideas;
  - DiskANN recall@10 ≥ own best − **0.05** at every L;
  - large-path memory ≤ 1.2× best.
- `sift1M_auto` / `sift1M_large` (median of 3) and `sift10M_auto` (1 run): ≥ 3% faster = gain,
  > 3% slower = regression; recall@10 at every itopk (cuVS CAGRA searcher) vs **TAGORE's graph
  searched the same way** (`state/tagore_baseline.json`, written in step B): resident path
  ≥ TAGORE − 0.01, large path ≥ TAGORE − 0.05. If TAGORE has no result for a dataset, judge.py
  falls back to CAGRA and says so.
- **ACCEPT** iff at least one of the four configs gains, none regresses, and search passes on
  all of them (judge.py 4-arg form, `$SK/scripts/judge.py`).
- Pick ideas by SIFT100M stage times first (`state/best.json` → `sift100M_auto.stages_s`,
  `bucket2_steps_s`); ideas must scale with N (the large path stays out-of-core).
- TAGORE's SIFT100M build + search (step B) is the external reference reported next to ours
  in every log entry (`vs TAGORE` column); it is not a gate for SIFT100M.

## Fixed context

| what | where |
|---|---|
| worktree | **`WT=/home/lanlu/microBenchmark_tagore`** (all repo paths below are relative to it). Shared, untracked, read-only from here: datasets `/home/lanlu/microBenchmark/dataset`, eval tool `/home/lanlu/microBenchmark/bucketDemo/evalCuvs/build/cagra_eval` (scripts point there) |
| this skill | `SK=$WT/.claude/skills/optimize-bucket2-tagore` — `docs/A1_local_join.md` (approved A1 design), `scripts/` (copied from main's `optimize-bucket2/scripts` at 3046497; adapted: `judge.py` TAGORE reference, SIFT100M ±7% and own-best −0.05, SMOKE guard; `run_pipeline.sh` / `resume_large_tail.sh` worktree-relative `ROOT`, shared dataset / eval paths, bench lock + SMOKE mode), `state/best.json` (this loop's best, written only by `$S/judge.py --update`), `state/reference_best.json` (cuVS loop's best as merged to main; used only by step P), `state/cagra_baseline.json`, `state/tagore_baseline.json` (step B), `scripts/tagore_*` (step B helpers), `notes/tagore_map.md` |
| branch | **`opt/bucket2-tagore`** (created from `main` 3046497 together with the worktree on 2026-10-22). Commit every accepted iteration there; never push; never touch `opt/bucket2-loop` or `main` |
| log | **`bucketDemo/OPTIMIZATION_LOG_TAGORE.md`** (created in step P). Never write to `OPTIMIZATION_LOG.md` (the cuVS loop's log); read it for its rejected list |
| runs | `bucketDemo/buildBucket/runs/tagore<N>/` (prefix keeps them apart from the cuVS loop's `iter<N>`) |
| code under optimization | `bucketDemo/buildBucket/src/`: `bucket.cu` (bucket2), `optimize.cu` (resident prune+reverse), `prune_windowed.cpp` + `prune_windowed_core.hpp` + GPU chunked prune, `optimize_chunked.cu` (method E), `translate_ids.cpp` + `translate_graph_ids.hpp`, shared `*.hpp` |
| TAGORE source | `/home/lanlu/Tagore` (user's clone; map below). Never change its sources; the only allowed edits are build config for step B (`CMakeLists.txt`: CUDA arch 70 for the V100, pybind11 path) and a separate build dir `/home/lanlu/Tagore/build-v100`. No LICENSE file: **reimplement ideas in our code style, don't paste kernels verbatim** |
| cuVS (secondary reference) | `/home/lanlu/cuvs` = cuVS 26.12, to compare TAGORE's way with cuVS's |
| link-time dependency | conda env `cuvs-bench` = libcuvs / libraft / librmm **26.10**, nvcc 12.9. Public cuVS/raft API only |
| toolchain | conda env `cuvs-build` (gcc 14, cmake 4, ninja). Build (in the worktree): `cd $WT/bucketDemo/buildBucket && /home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -C build` (reconfigure: `CONDA_PREFIX=/home/lanlu/miniconda3/envs/cuvs-bench /home/lanlu/miniconda3/envs/cuvs-build/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/home/lanlu/miniconda3/envs/cuvs-build/bin/ninja`) |
| eval tools | `bucketDemo/evalCuvs/build/cagra_eval` (SIFT1M/10M search), `/home/lanlu/DiskANN/build/apps/search_memory_index` via `$S/diskann_search.py` (SIFT100M search) |
| datasets | `dataset/sift1M/`, `dataset/sift10M/`, `dataset/sift100M/` (u8bin base, 10K queries, groundtruth k=100, exact KNN of 10K sampled base points) |
| machine | 4× V100-SXM2-16GB (GPU 0/1 on NUMA node 0, GPU 2/3 on node 1), 2× Xeon Gold 6133 (80 threads), 251 GB RAM, one LVM volume over two SATA SSDs (`/`, ~400 GB free on 2026-10-22) |
| GPU | **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`, scripts set it) |

## Change policy: implementation vs. algorithm (user rule, 2026-10-21)

- **Implementation changes: try directly.** Same algorithm, same semantics, different
  execution: kernel fusion, tiling / shared-memory layout, tensor cores, launch shapes,
  overlapping GPU / host / disk work, buffering and caching, I/O format of intermediates,
  threading, removing redundant work, exact-result precision tricks (e.g. low-precision
  scoring followed by an exact re-rank that yields the same candidates), resource parameters
  in `params.sh` (cache / budget sizes, batch sizes).
- **Algorithm / semantic changes: propose only.** Anything that changes *which* candidates
  are considered or *how* the graph is defined: bucketing (centroid selection, assignment,
  multi-assignment), bucket order, the bucket-pair candidate set, adding a refinement stage,
  prune or reverse-edge rules, algorithm parameters in `params.sh` (`CENTROID_RATIO`, `KNN_K`,
  `NPROBE`, `ITERATIONS`, `SEED`, sampling fractions), approximate shortcuts that change
  results. **If unsure, treat it as algorithm.**
  - Record it in the log's **Awaiting approval** section as `A<k>: <idea> — TAGORE source — target stage — expected gain — what changes semantically`, then pick an implementation idea instead.
  - Implement an `A<k>` only after the user marked it **APPROVED** (in the log or in chat; when the approval comes in chat, write it into the log with the date). Rejected proposals move to **Tried & rejected** as "user declined".
  - Note: duplication across buckets is already achieved by the multi-round build iterations (`ITERATIONS`), so TAGORE's "each point in its 2 nearest clusters" is not needed as a new idea.
- **Pre-approved: A1 — GNN-Descent local join** (user, 2026-10-21). Resident path: follow TAGORE's GNN-Descent; large path (the one that matters for SIFT100M): the user's fused prune + local join (RNG test + insertion while scanning, tensor-core GEMM), which fits on top of the chunked GPU prune in `prune_windowed`. Full design, the user's original description, open details and defaults: **`docs/A1_local_join.md`** — read it before any A1 iteration, keep it updated with what each A1 iteration learned, and never drop the user's original text. Retuning `NPROBE` / `ITERATIONS` together with A1 is part of A1 (for SIFT100M the run_pipeline override `ITERATIONS=2` stays unless the user changes it). Defaults agreed with the user: detour counting against all closer candidates and fill to 32 by (count, rank), as cuVS `kern_prune` does; expand only `new` forward neighbors, at most SAMPLE = 32 per u, only inside the bucket window; only-u and +reverse are separate variants. Every A1 run must print the `A1_stats:` line (expansion counts, assumption-violated ratio, … — `docs/A1_local_join.md` §2.6); extend `$S/summarize.py` to put it into `metrics.json` as `a1_stats`, and paste it into every A1 log entry (accepted or rejected) — the user needs these numbers for experiments.

## TAGORE map: what it has and where it can land in our pipeline

TAGORE = GPU graph-index construction library (Python/pybind). The `class` column says which
side of the change policy an idea on that row usually falls on.

| TAGORE component | source | what it does | where it can help us | class |
|---|---|---|---|---|
| **GNN-Descent phase 1** | `src/Tagore_src.cu`: `nn_descent_opt_sample`, `nn_descent_opt_reverse_sample`, `nn_descent_opt_cal`, `nn_descent_opt_merge`; driver `pybind.cu::GNN_descent` | per node: split neighbors into new/old by visit flag, sample (SAMPLE=32), add reverse samples via `atomicAdd` into `reverse_graph`; new×old distances as **wmma tensor-core GEMM** on fp16 with precomputed norms (`cal_power`: ‖a‖²+‖b‖²−2a·b); merge with a **k-th-distance threshold (`min_ele`) filter before bitonic sort + binary insertion** | the *kernel techniques* for Step 6 bucket × pool distance + top-64 (SIFT100M: ~1560 s of 2567 s bucket2): shared-memory GEMM tiles + early reject against the current k-th distance = "fused GEMM + top-M, never materialize dots"; Step 4 GEMM + top-2 | implementation |
| **GNN-Descent as a refinement stage** | phase 1 + phase 2 (`sample_kernel6`, `merge_reverse_plus`, `do_reverse_graph`) | local join: neighbors of neighbors become candidates | refine the degree-64 `vector_knn` after bucket-pair KNN so NPROBE can drop; on the large path fused with prune | **algorithm, APPROVED as A1** (`docs/A1_local_join.md`) |
| **CFS pruning, CAGRA mode** (Collect-Filter-Store) | `select_1hop_cagra` (`collect_1hop` + `filter_cagra`), `filter_reverse_1hop`; `Pruning(...,'CAGRA')` in `pybind.cu` | rank-based detour count fully on GPU, one block (32×4) per node; reverse edges collected with `atomicAdd` into a bounded `reverse_graph` (RESERVENUM=128) and merged in the same pass, no host round trip | `optimize_chunked -m E` (reverse edges, SIFT100M ~307 s) and the GPU prune: fused prune + reverse on the large path | implementation if the prune / reverse result rule stays CAGRA's; TAGORE's bounded reverse list that drops overflow = algorithm. NSG / Vamana / NSSG / DPG filters (`select_path`, `select_2hop`, `filter_reverse`, `filter_dpg`) change the index type: out of scope |
| **Large-scale pipeline** | `src/large_index.cu`: `order_merge_main` + `reorder` (Algorithm 4), `gpu_construct`, `order_merge_new` (Algorithm 3); driver `pybind.cu::largeIndex` | clusters > `max_points` split round-robin; greedy cluster order maximizing overlap with the buffered clusters + eviction choice; one GPU thread builds subgraphs while a CPU thread merges, handing off through a fixed `graph_pool` with `has_built` / `has_merged` flags; cached writer (1 GB) for the final index | **overlap GPU compute with host merge / scatter / disk writes** in out-of-core Step 6 (GPU-cache thrash, host gather) and Step 4.5; cache eviction / slot assignment; large-buffer writes | overlap / buffering / eviction = implementation; splitting oversized buckets or changing bucket order = algorithm. Multi-GPU out (GPU 0 only) |
| small tricks | `pybind.cu::GNN_descent`, `include/common.cuh` | power-of-10 `norm_factor` before fp16; norms precomputed once; compile-time shapes (`K_SIZE`, `DIM_SIZE`, `TOPM_SIZE`) | compile-time specialization of hot kernels for D=128 / K=64 / degree 32 (keep runtime fallback) | implementation |

Caveats: research code with hardcoded shapes (K=96, DIM=128), fp16 data, almost no bounds
checks, `atomicAdd` into fixed-size reverse lists silently drops overflow. Port the idea,
accumulate in fp32, add our own bounds checks.

## Design invariants: never violate

The pipeline is meant for datasets far larger than GPU memory; change *how* a stage runs,
never remove or bypass one:

1. **Bucketing**: data is partitioned into buckets (centroid selection + assignment).
2. **Bucket order**: buckets are processed in the DiskJoin-style order derived from the centroid KNN graph.
3. **Bucket-pair computation**: per-vector KNN comes from bucket × neighbor-bucket computation, not from a global index.
4. **Index prune** (CAGRA-style rank-based pruning; exception: the approved A1 large-path fused prune + local join uses the RNG rule with detour-count-style fill, see `docs/A1_local_join.md` §2.4).
5. **Reverse-edge adding** (and merge).
6. **ID translation** back to the original order (`translate_ids`).

Also:
- Build degree **64** (`NEIGHBORS_M`), final degree **32** (`OUTPUT_DEGREE`). Fixed.
- **Both paths keep working**: GPU-resident (bucket2 resident + `optimize`) when data fits; otherwise bucket2 out-of-core + `prune_windowed` + `optimize_chunked -m E`. Auto-selection (bucket2 fit check, `optimize` exit code 2) stays correct. The large path never holds the full dataset or full graph in GPU memory; host memory stays within budget (judge rejects >1.2× peak growth).
- Never replace the pipeline with `cagra::build`, cuVS nn_descent, or TAGORE `GNN_descent` / `largeIndex` on the whole dataset.
- Final graph: no out-of-range ids, self loops or duplicate edges. Add bounds checks with new indexing code.

## Test commands (used by steps P and 3)
```bash
WT=/home/lanlu/microBenchmark_tagore
SK=$WT/.claude/skills/optimize-bucket2-tagore
S=$SK/scripts
R=$WT/bucketDemo/buildBucket/runs/tagore<N>
bash $S/run_pipeline.sh -d sift1M   -m auto  -r 3 -o $R/sift1M_auto     # GPU-resident path, median of 3
bash $S/run_pipeline.sh -d sift1M   -m large -r 3 -o $R/sift1M_large    # large path forced on SIFT1M, median of 3
bash $S/run_pipeline.sh -d sift10M  -m auto  -r 1 -o $R/sift10M_auto    # SIFT10M (→ large path), 1 run (~3 min)
bash $S/run_pipeline.sh -d sift100M -m auto  -r 1 -o $R/sift100M_auto   # SIFT100M (large path, 2 rounds), 1 run (~50 min): timing + DiskANN search + memory
```
- Run in this order. **Quick checks first**: after the three SIFT1M/SIFT10M runs, run the judge on
  them alone (`$S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto`) and look only at
  correctness / search / regression failures (a missing speed gain there is fine — the gain may
  come from SIFT100M). If they already fail, skip SIFT100M and reject.
- Launch every run detached (`setsid nohup bash $S/run_pipeline.sh … > $R/<cfg>.out 2>&1 &`) so a
  harness restart cannot kill it, and wait with Monitor / an until-loop on the process (no
  sleep-polling). The run blocks on the bench lock by itself while the cuVS loop is timing; the
  wait is not part of the measured time. If GPU 0 is used by a process that does not hold the
  lock, wait for it.
- Non-zero exit / "FAILED" = failure (rule out environment first: busy GPU, `df -h /home/lanlu` must keep > 50 GB free; SIFT100M work dirs are large).
- If a SIFT100M run died in optimize_chunked / translate / search with the work dir kept,
  `$S/resume_large_tail.sh -d sift100M -o $R/sift100M_auto` reruns only the tail (the build time is then the sum of the kept stage times; say so in the log).
- Quality per run: (a) pre-prune `knn_similarity` (recall@1/10/32/64, logged, not gated);
  (b) final graph search: cuVS CAGRA searcher (SIFT1M/10M, itopk 32…512) or DiskANN CPU searcher
  (SIFT100M, L 32…512), recall@10 + QPS + defect counts. Always log recall and QPS.

## Step selection

- Always work in `$WT` (`git -C $WT branch --show-current` must be `opt/bucket2-tagore`; if the session was started elsewhere, use absolute `$WT` paths for everything).
- `$SK/state/best.json` does not exist → **step P** (setup + baseline verification).
- else `state/tagore_baseline.json` does not exist → **step B** (TAGORE baseline).
- else `notes/tagore_map.md` does not exist → **step T0** (study).
- else → **one iteration** (steps 0–6).

### P. Setup + verify the main baseline (first invocation, no optimization)
1. Preconditions: worktree `$WT` exists on `opt/bucket2-tagore` (created with the skill, from main 3046497 = cuVS loop through #76); `git -C $WT status --short` shows only the untracked skill dir. If `main` has moved on since, ask the user whether to rebase the branch first.
2. `$SK/scripts` and `$SK/state/{cagra_baseline,reference_best}.json` come from main 3046497; check `diff` of every script against `main:.claude/skills/optimize-bucket2/scripts/` — expected differences are only the adaptations listed in Fixed context. From now on only `$S` (incl. its `params.sh`) is used and edited.
3. Commit the skill on `opt/bucket2-tagore` right away (`bucket2 tagore: add optimize-bucket2-tagore skill` + attribution line), so the worktree is clean before any source change.
4. Configure + build the worktree's own `build/` (reconfigure command above, run in `$WT/bucketDemo/buildBucket`).
5. Run the four test configs into `runs/tagore0/` (SIFT100M last).
6. **Verify the build is the optimized one**: compare each config with `state/reference_best.json`:
   - `build_total` within **±3%** for SIFT1M auto, SIFT1M large and SIFT10M, **±7%** for SIFT100M (user, 2026-10-21 / 2026-10-22);
   - recall@10 at every itopk / L 32 / 64 / 128 within **±1% relative** (|measured − ref| / ref ≤ 0.01) for SIFT1M auto / large, **±3% relative** for SIFT10M and SIFT100M;
   - same path selection (auto → resident on SIFT1M, large on SIFT10M / SIFT100M), 0 graph defects;
   - QPS and large-path memory: recorded in the verification table, not gated.
   If something is off: rerun that config once; still off → **stop the loop** (the only place this skill stops), report the table (reference vs measured) and likely causes (main not updated, `params.sh` differs, stale build, GPU / disk busy) to the user. Do not create `best.json`.
7. Passed: `python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/sift100M_auto --update --md` (no `best.json` yet → becomes this loop's baseline; with no `tagore_baseline.json` yet the 1M/10M search check falls back to CAGRA, which is expected here).
8. Create `bucketDemo/OPTIMIZATION_LOG_TAGORE.md` with: header (pipeline, environment, measurement, the Goal's rules, "ideas source: TAGORE `/home/lanlu/Tagore`" and the change policy in two lines), **Current best** (baseline table), **Baseline verification** (reference vs measured table from 6), **TAGORE baseline** (filled by step B), **Ideas backlog** (empty), **Awaiting approval** (empty), **Possible gains** (SIFT100M 3–7%, empty), **Tried & rejected** (empty), **Iterations** with `### #0 — baseline from main (<date>, commit <main hash>)`.
9. Commit `$SK/state` and the log on `opt/bucket2-tagore`: `bucket2 tagore #0: baseline from main` + attribution line. Return with the verification table in 3–5 lines.

### B. TAGORE baseline: build + search on SIFT1M, SIFT10M, SIFT100M (second invocation, no source change)
Goal: TAGORE's own build time and search quality on our datasets and GPU. SIFT1M / SIFT10M
become the judge's search reference (`state/tagore_baseline.json`); SIFT100M is the reference
reported next to ours. Same benchmarking rules as every test (GPU 0, every timed TAGORE
run wrapped in `flock /home/lanlu/.bench_locks/bucket2_bench.lock …`, detached). If a part turns out impossible (e.g. OOM), record why in the json (`status`) and the
log and move on; never stop the loop for step B.

1. **Build TAGORE for the V100** in `/home/lanlu/Tagore/build-v100`: `CMAKE_CUDA_ARCHITECTURES 70`,
   nvcc 12.9 from `cuvs-bench` (`-DCMAKE_CUDA_COMPILER=/home/lanlu/miniconda3/envs/cuvs-bench/bin/nvcc`),
   pybind11: no conda env has it → create a dedicated env `tagore` (python 3.11, numpy, pybind11)
   rather than installing into `cuvs-bench` / `cuvs-build`. `CMakeLists.txt` hardcodes
   `pybind11_DIR` and the arch with plain `set()`, so edit those two lines; record the diff in the log.
2. **Inputs**: TAGORE `load_data` reads float `.fvecs`. Convert `dataset/sift1M/base.1M.u8bin` and
   `dataset/sift10M/base.10M.u8bin` into the scratchpad (never into `dataset/`) with a small
   helper `$SK/scripts/tagore_u8bin_to_fvecs.py`. Conversion time is not part of TAGORE's build
   time (our pipeline reads u8bin directly); report it separately.
3. **SIFT1M** (`$SK/scripts/tagore_run.py`, run with the `tagore` env's python, `CUDA_VISIBLE_DEVICES=0`):
   `Tagore.GNN_descent(k, 1000000, 128, iters, fvecs)` then `Tagore.Pruning(k, N, 128, 32, 64, ptrs, 'CAGRA', index_path=...)`.
   Main config = the README's (k=96, 10 iterations, m=64, final degree 32; `K_SIZE`=96 is the
   compiled default). Also run k=64 (our `NEIGHBORS_M`) if it works without changing kernels.
   Time: wall clock of the whole process (load included, like our `build_total`) plus TAGORE's
   printed `time of GNN-Descent` / `time of pruning`; median of 3.
4. **Convert + search (SIFT1M / SIFT10M)**: `StoreCAGRA` writes per node `[degree][ids…]`; note it
   indexes the device graph with `i * degree` although rows have stride K — check that the ids are
   each node's real neighbors (spot-check against exact KNN) before trusting it; if the layout is
   wrong, take the graph from the returned device pointers in the wrapper instead of patching TAGORE.
   Convert to `cagra_eval`'s format (int64 N, int32 degree, uint32 ids, `$SK/scripts/tagore_graph_to_cagra.py`)
   and run the same search as `run_pipeline.sh` (`cagra_eval search`, k=10, itopk 32…512, 10K queries):
   recall@10, QPS, defect counts. If the pre-prune kNN graph is accessible, also run
   `knn_similarity.py` on it.
5. **SIFT10M**: whole-dataset `GNN_descent` needs ~20+ GB of device memory (graph, reverse graph,
   distances, `hybrid_list`), more than the 16 GB V100 — try it once (k=64 first), and on OOM use
   `largeIndex` as in step 6 (1 run). Search as in step 4 (a Vamana graph from `largeIndex` has
   variable degree ≤ 32: pad to 32 for `cagra_eval`, report padding / defect counts separately).
6. **SIFT100M** — only TAGORE's large-scale path fits: `Tagore.largeIndex` with one GPU
   (`devicelist=[0]`). It builds a **Vamana** graph (alpha prune), not a CAGRA graph — say so
   next to every number. Defaults: k=64, m=64, final_degree=32, max_degree=32, threshold
   (alpha)=1.2, `max_points` ≈ 2.5M (gpu_construct needs ~2.9 KB of device memory per point →
   ≤ ~5M per cluster on 16 GB; keep headroom), cluster_num ≈ 80–100 (each point is in 2 clusters),
   `buffer_size` chosen so the host `graph_pool` (buffer_size × max_points × k × 4 B) stays well
   below the free RAM (251 GB machine, our SIFT100M build peaks at ~80 GB).
   - **Prep** (`$SK/scripts/tagore_prep_clusters.py`, `cuvs-bench` python with cupy): k-means on a
     sample, then GPU assignment of every point to its 2 nearest centers → `centers_path`
     (N pairs of uint32) and one float file per cluster (`data_path_prefix<i>`: uint32 count +
     float vectors), written straight from the u8bin (no 51 GB fvecs). Needs ~102 GB of disk:
     check `df` first (keep > 50 GB free after it), delete the cluster files and local indexes
     when step B is done.
   - **Time**: prep, build, and total (prep + build, comparable to our build which includes
     bucketing); 1 run; also GPU peak and host peak RSS.
   - **Search**: the output is DiskANN's in-memory format → run the same DiskANN
     `search_memory_index` as `diskann_search.py` (same L list 32…512, K=10, all threads, same
     query / GT files). Report it with the **medoid entry point** (as our graphs are searched;
     rewrite the header's start id with the medoid from `diskann_search.py`) and also with
     TAGORE's own entry point. Variable degree is fine for DiskANN (no padding needed); report
     bad ids / self loops / duplicate edges.
7. Write `state/tagore_baseline.json` in `cagra_baseline.json`'s format so judge.py can use it:
   per dataset `{"label", "index_type" (CAGRA / Vamana), "config", "status", "build_s": [...],
   "build_median_s", "prep_s", "search": {itopk_or_L: recall}, "qps": {...}, "defects",
   "gpu_peak_mib", "peak_rss_gb", "note"}` (SIFT100M's `search` keys are DiskANN L, medoid entry).
   Add a **TAGORE baseline** table to the log next to CAGRA and our current best. Commit
   `$SK/scripts/tagore_*`, the state file and the log (`bucket2 tagore: TAGORE baseline`).
   Return with the table in 3–5 lines.

### T0. Study pass (third invocation, no source change)
1. Read the TAGORE sources listed in the map, kernel by kernel. For each component write in `notes/tagore_map.md`: function + line range, launch shape, memory layout (shared / global, sizes), the algorithmic trick, the **matching bucket2 stage** (file + function in `bucketDemo/buildBucket/src/`) with its current SIFT100M time from `state/best.json` (`stages_s`, `bucket2_steps_s`), and implementation vs. algorithm classification.
2. Rank ideas by expected saving on **SIFT100M**. Implementation ideas → **Ideas backlog** (one line each: idea, TAGORE source, target stage, expected gain). Algorithm ideas → **Awaiting approval** as `A<k>` entries (A1 is entered already marked `APPROVED 2026-10-21`, pointing to `docs/A1_local_join.md`; new ones start at A2). Skip anything already in `OPTIMIZATION_LOG.md`'s **Tried & rejected** unless you write down what TAGORE does differently.
3. Use the step-B profile (`state/tagore_baseline.json`: GNN-Descent vs pruning time, largeIndex prep vs build vs merge) to say which TAGORE stage is fast for which reason, and where TAGORE's SIFT100M build beats or loses to ours stage by stage.
4. Commit notes + log (`bucket2 tagore: T0 study notes`), return with a 3-line summary of the top ideas and the list of `A<k>` waiting for the user.

## One iteration

### 0. Orient (cheap)
- In `$WT` on `opt/bucket2-tagore`, source tree clean (if interrupted: `git -C $WT checkout -- bucketDemo/buildBucket/src $S/params.sh`, log it as "interrupted"; check for orphaned detached runs of this loop with `pgrep -af microBenchmark_tagore` first).
- Read `OPTIMIZATION_LOG_TAGORE.md`: **Current best**, **Ideas backlog**, **Awaiting approval** (any newly APPROVED entries?), **Possible gains**, **Tried & rejected**; and `notes/tagore_map.md`.
- Check the bench lock owner (`cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`): if the cuVS loop is timing, do steps 1–2 gently (niced build, SMOKE tests) and let the timed runs queue.

### 1. Pick ONE idea
- Candidates: implementation ideas from the backlog, plus `A<k>` entries the user APPROVED (A1: follow the split in `docs/A1_local_join.md` §2.8). Start from where SIFT100M time goes (largest stage first).
- Name the TAGORE function(s) the idea comes from and, in one line, how cuVS 26.12 / our current code does the same step.
- If the idea you want turns out to be an algorithm change and isn't approved: add it to **Awaiting approval**, pick the next implementation idea.
- One coherent change so the result is attributable. Write the plan in one or two sentences before editing (it becomes the log entry's "idea").
- Smoke-test new large-path code first (`SMOKE=1 … -d sift1M -m large`, or SIFT10M) before spending an hour on SIFT100M.

### 2. Implement
- Edit sources and/or `$S/params.sh`. Match the surrounding style (Chinese/English comment mix, naming). Credit the source in a short comment, e.g. `// 参考 TAGORE nn_descent_opt_merge: k-th 距离阈值预过滤`.
- Build with ninja. Compile error → fix; infeasible → revert and log as rejected ("infeasible: ..."), then continue with the next idea in the next invocation.

### 3. Test
Run the four test configs into `runs/tagore<N>/` (quick checks first, SIFT100M only if they pass).

### 4. Judge
```bash
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/sift100M_auto --md
```
Rules: see Goal and judge.py's docstring. SIFT1M/10M borderline gain (3–5%): rerun that config into `<dir>_b`, accept only if both pass. SIFT100M is never rerun in an iteration for now: a 3–7% SIFT100M gain (judge note "possible gain") is rejected and goes to **Possible gains**.

### 5a. ACCEPT
1. `python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/sift100M_auto --update --md`.
2. Append `### #<N> — <idea>` to `OPTIMIZATION_LOG_TAGORE.md` with: idea, **TAGORE source** (file + function), class (implementation / approved `A<k>`), the `--md` tables (incl. `vs TAGORE`), `a1_stats` for A1 runs; update **Current best**.
3. Commit on `opt/bucket2-tagore`: sources, `$S/params.sh`, `$SK/state/best.json`, the log, notes / docs if changed. Message `bucket2 tagore #<N>: <idea>` + attribution line. Don't commit `runs/`.

`<N>` = this log's last iteration + 1 (own numbering, starting after #0).

### 5b. REJECT
Applies to every failure in an optimization iteration (judge reject, build error, crash, timeout, wrong path selection, graph defects, quick checks failed). Never stop the loop for it.
1. Keep the patch for later: `git -C $WT diff > $WT/bucketDemo/patches/tagore<N>_<short-name>.patch` (the user may recheck possible gains with multi-run medians).
2. `git -C $WT checkout -- bucketDemo/buildBucket/src $S/params.sh` and rebuild.
3. One line under **Tried & rejected** with the concrete failure: idea (TAGORE function), which config / stage failed, the measured numbers vs best (time per config, recall@10, QPS, memory) or the error message, the judge reason, the patch path. A 3–7% SIFT100M gain with everything else passing goes under **Possible gains** instead (same fields). For an A1 step also paste `a1_stats` and update `docs/A1_local_join.md` with what was learned. Commit the log and the patch (and notes / docs if changed).

### 6. Housekeeping & return
- Delete `runs/tagore<N>` dirs older than the last ~5 iterations except those referenced by `state/best.json` and `runs/tagore0`. SIFT100M work dirs are deleted by run_pipeline; check `df -h /home/lanlu` (> 50 GB free). Never delete the cuVS loop's `runs/iter*`.
- Update `notes/tagore_map.md` with what this iteration taught (worked / didn't and why); re-rank the backlog.
- If no implementation idea is left and nothing is approved: say so in the status line, list the pending `A<k>`, and keep going with the best remaining general idea (from the backlog or profiling).
- End with a 2–3 line status: iteration N, idea + TAGORE source, verdict, new best totals (SIFT100M first), number of `A<k>` awaiting approval.
