---
name: optimize-bucket2-tagore
description: One iteration of the bucket2 build-pipeline optimization loop (bucket2 -> optimize | prune_windowed+optimize_chunked -> translate_ids) in bucketDemo/buildBucket, dataset-generic (datasets.json registry: uint8 / int8 / float32 vectors, size-adaptive parameters for 1M…1B via dataset_env.py), focused on the 100M-scale build (SIFT100M: 2 rounds, median of 3, DiskANN CPU search) and learning from TAGORE (GNN-Descent, CFS GPU pruning, async GPU-CPU-disk large-scale pipeline; source /home/lanlu/Tagore). Own branch opt/bucket2-tagore (from main), own state/best.json, own log bucketDemo/OPTIMIZATION_LOG_TAGORE.md. First invocation sets up + verifies the main baseline (SIFT1M/10M/100M), second runs TAGORE itself on SIFT1M/10M/100M as the build + search reference, third is a TAGORE study pass, later ones implement one TAGORE-derived idea (implementation ideas, or approved algorithm ideas such as A1 = GNN-Descent local join / fused prune+local join, docs/A1_local_join.md), benchmark the bench set (SIFT1M auto/large, SIFT10M, msTuring10M float32, SIFT100M ×3) on GPU 0, and accept/reject with judge.py (±3% for every config). Algorithm/semantic changes are only proposed in the log until the user approves them. Run repeatedly with `/loop /optimize-bucket2-tagore`.
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

## Goal (user decisions 2026-10-22 / 2026-10-23)

**Make the large-scale build faster on any dataset**, with SIFT100M as the main target. Since
2026-10-23 the skill is dataset-generic (see **Datasets, data types and adaptive parameters**).

**Bench set** (every iteration; `datasets.json` fields `bench` and `reps`; user decision 2026-10-23: every bench config ×3, msTuring100M ×3 only at milestones):

| config | reps | search | quality reference (judge.py) |
|---|---|---|---|
| `sift1M_auto` (resident) | 3 (median) | cuVS CAGRA (`cagra_eval`), itopk 32…512 | TAGORE − 0.01 |
| `sift1M_large` (large path forced) | 3 | cuVS CAGRA | TAGORE − 0.05 |
| `sift10M_auto` (→ large path) | 3 (median) | cuVS CAGRA | TAGORE − 0.05 |
| `msTuring1M_auto` (float32, D=100, resident) | 3 (median) | DiskANN CPU, L 32…512; cuVS CAGRA once P1 adds float32 to cagra_eval | own best − 0.01 until P1 / P4 give it a reference |
| `msTuring10M_auto` (float32, D=100) | 3 (median) | DiskANN CPU (`diskann_search.py`), L 32…512; cuVS CAGRA once P1 adds float32 to cagra_eval | own best − 0.01 until P1 / P4 give it a reference (then −0.01 resident / −0.05 large) |
| `sift100M_auto` (2 rounds) | **3 (median)** | DiskANN CPU, L 32…512 | own best − 0.05 |

**Rules (judge.py):**
- **Every config: build_total ≥ 3% faster than best = gain, > 3% slower = regression** (2026-10-23:
  every bench config is the median of 3 runs — SIFT1M, SIFT10M, msTuring1M, msTuring10M and SIFT100M; the old 7% single-run gate and the "Possible gains"
  bundling are retired — one coherent change per iteration again, don't bundle to pass the gate).
- **ACCEPT** iff the full bench set was run, at least one config gains, none regresses, search
  quality passes everywhere, 0 graph defects, path selection unchanged, large-path memory ≤ 1.2× best.
- Pick ideas by SIFT100M stage times first (`state/best.json` → `sift100M_auto.stages_s`,
  `bucket2_steps_s`), then by msTuring10M (float path); ideas must scale with N (the large path
  stays out-of-core).
- TAGORE's build + search (step B) is the external reference reported next to ours (`vs TAGORE`
  column); for SIFT100M it is not a gate.

## Fixed context

| what | where |
|---|---|
| worktree | **`WT=/home/lanlu/microBenchmark_tagore`** (all repo paths below are relative to it). Shared, untracked, read-only from here: datasets `/home/lanlu/microBenchmark/dataset`, eval tool `/home/lanlu/microBenchmark/bucketDemo/evalCuvs/build/cagra_eval` (scripts point there) |
| this skill | `SK=$WT/.claude/skills/optimize-bucket2-tagore` — `docs/A1_local_join.md` (approved A1 design), `scripts/` (copied from main's `optimize-bucket2/scripts` at 3046497; since then made dataset-generic (2026-10-23): `judge.py` keys configs as `<dataset>_<mode>`, ±3% for all, per-dataset quality reference from `datasets.json`, `--quick` / `--rebaseline`, SMOKE guard; `run_pipeline.sh -d <any registered dataset>` with adaptive parameters from `dataset_env.py`, `data_reordered.<ext>`, search tool per dataset, bench lock + SMOKE mode; `diskann_search.py` uint8 / int8 / float32), `state/best.json` (this loop's best, written only by `$S/judge.py --update`), `state/reference_best.json` (cuVS loop's best as merged to main; used only by step P), `state/cagra_baseline.json`, `state/tagore_baseline.json` (step B), `scripts/tagore_*` (step B helpers), `notes/tagore_map.md`; dataset tooling: `scripts/datasets.json` (registry), `scripts/dataset_env.py` (lookup + size-adaptive parameters), `scripts/prep_dataset.py` + `scripts/gen_gt.py` (prepare a dataset dir: base / query links or subsets, exact groundtruth + 10K self-sample KNN) |
| branch | **`opt/bucket2-tagore`** (created from `main` 3046497 together with the worktree on 2026-10-22). Commit every accepted iteration there; never push; never touch `opt/bucket2-loop` or `main` |
| log | **`bucketDemo/OPTIMIZATION_LOG_TAGORE.md`** (created in step P). Never write to `OPTIMIZATION_LOG.md` (the cuVS loop's log); read it for its rejected list |
| runs | `bucketDemo/buildBucket/runs/tagore<N>/` (prefix keeps them apart from the cuVS loop's `iter<N>`) |
| code under optimization | `bucketDemo/buildBucket/src/`: `bucket.cu` (bucket2), `optimize.cu` (resident prune+reverse), `prune_windowed.cpp` + `prune_windowed_core.hpp` + GPU chunked prune, `optimize_chunked.cu` (method E), `translate_ids.cpp` + `translate_graph_ids.hpp`, shared `*.hpp` |
| TAGORE source | `/home/lanlu/Tagore` (user's clone; map below). Never change its sources; the only allowed edits are build config for step B (`CMakeLists.txt`: CUDA arch 70 for the V100, pybind11 path) and a separate build dir `/home/lanlu/Tagore/build-v100`. No LICENSE file: **reimplement ideas in our code style, don't paste kernels verbatim** |
| cuVS (secondary reference) | `/home/lanlu/cuvs` = cuVS 26.12, to compare TAGORE's way with cuVS's |
| link-time dependency | conda env `cuvs-bench` = libcuvs / libraft / librmm **26.10**, nvcc 12.9. Public cuVS/raft API only |
| toolchain | conda env `cuvs-build` (gcc 14, cmake 4, ninja). Build (in the worktree): `cd $WT/bucketDemo/buildBucket && /home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -C build` (reconfigure: `CONDA_PREFIX=/home/lanlu/miniconda3/envs/cuvs-bench /home/lanlu/miniconda3/envs/cuvs-build/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/home/lanlu/miniconda3/envs/cuvs-build/bin/ninja`) |
| eval tools | `cagra_eval` (cuVS CAGRA search; uint8 only, data must fit the GPU) for `search: cagra` datasets, `/home/lanlu/DiskANN/build/apps/search_memory_index` via `$S/diskann_search.py` (uint8 / int8 / float32) for `search: diskann` datasets |
| datasets | `/home/lanlu/microBenchmark/dataset/<name>/` as registered in `datasets.json`: sift1M / sift10M / sift100M (uint8, D=128), msTuring1M / msTuring10M / msTuring100M (float32, D=100; 1M / 10M = first N vectors of `/home/lanlu/SCALER/dataset/msTuring100M`, base / query hard-linked). Each dir: `base.<N>.<ext>`, `query.public.10K.<ext>`, `groundtruth.ibin` (k=100, ids + sq. distances), `base_knn_sample10K_k64.ibin`. No 1B base exists on this machine yet (SCALER `sift1B` has only queries) |
| machine | 4× V100-SXM2-16GB (GPU 0/1 on NUMA node 0, GPU 2/3 on node 1), 2× Xeon Gold 6133 (80 threads), 251 GB RAM, one LVM volume over two SATA SSDs (`/`, ~400 GB free on 2026-10-22) |
| GPU | **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`, scripts set it) |

## Datasets, data types and adaptive parameters (user decision 2026-10-23)

- **Registry** `scripts/datasets.json`: per dataset `dir`, `base` (extension = element type:
  `.u8bin` uint8, `.i8bin` int8, `.fbin` float32 — uint32 vectors are not supported, user
  2026-10-23), `reps`, `iterations` (null = params.sh), `search` (cagra | diskann), `ref` (tagore |
  cagra | own) + `own_tol`, `bench` (modes in the per-iteration bench set). `run_pipeline.sh -d <name>`
  works for any registered dataset.
- **Size-adaptive parameters** `scripts/dataset_env.py <name>` (called by run_pipeline.sh; an
  exported variable overrides it, e.g. for sweeps). Rules, reproducing the hand-tuned SIFT100M values
  (0.004 / 80 / 0.10 / 15000 / 16000 / 8192):
  - average bucket B = 250 points, within the user's 128–512 bound; B grows (×2) beyond 512 only
    when the centroid matrix would exceed 1 GB of GPU memory (huge sets such as SIFT1B) →
    `CENTROID_RATIO` = 1/B; `NPROBE` = ⌈20000 / B⌉ (pool ≈ 20K rows); `--kmeans-train-fraction` =
    min(0.10, 10M / N);
  - host caches from the round records N × (D × itemsize + 8 B): `--cache-mb` = max(15000,
    ⌈1.15 × records⌉₁₀₀₀), `--bucket-mem-mb` = cache + 1000, capped at a quarter of the 200 GB host
    budget each; `--gpu-bucket-cache-mb` = GPU / 2; `optimize_chunked --host-budget-mb` = 200 GB
    (a ceiling, not a target).
  These rules are part of the code under optimization: changing a resource rule is an
  implementation change, changing the B / pool targets or bounds is A2 (params only, approved within
  128–512).
- **Data-type support status** (2026-10-23):

  | stage | uint8 / int8 | float32 |
  |---|---|---|
  | bucket2 read / bucketing / Step 6 GEMM | yes (fp16 tensor-core GEMM) | reads `.fbin`; Step 6 uses fp32 SGEMM (no tensor cores on V100) |
  | Step 6 out-of-core GPU bucket cache (#1/#2/#21) | yes | **no** (`use_dev_cache = kIsInt8Path && …`) |
  | prune (GPU chunked kernel, #76 / #16 / #21) | yes | **no**: `prune_windowed_gpu` is uint8-only → falls back to the slower path (verify) |
| out-of-core Step 6 bucket reader | yes | **crashes** (P2 item 1: `ReorderedDatasetBucketReader: read failed`) |
  | optimize / optimize_chunked / translate | yes | yes (graph only) |
  | search | cagra_eval (uint8) or DiskANN | DiskANN only (cagra_eval reads uint8) |

- **Adding a new dataset** (no code change needed):
  1. `prep_dataset.py <name> <src_base> <src_query> [--n N]` under the bench lock
     (`flock /home/lanlu/.bench_locks/bucket2_bench.lock …`; GPU 2) → dataset dir with GT files;
  2. register it in `datasets.json` (`bench: []` unless the user adds it to the bench set);
  3. smoke run `SMOKE=1 bash $S/run_pipeline.sh -d <name> -m auto -r 1 -o …` (correctness, path,
     memory; check the printed adaptive parameters);
  4. timed baseline run, record with `judge.py <dir> --update --add-new`; optionally TAGORE / CAGRA
     references (step B procedure) → set `ref` with the user's OK;
  5. if its quality / speed trade-off looks off, sweep `CENTROID_RATIO` (1/512…1/128) × `NPROBE`
     on a subset (like #3) and report — changing the defaults is A2 territory.

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
- **Approved (restricted): A2 — bigger buckets by parameters only** (user, 2026-10-22, relayed from the user's other session). Allowed: `CENTROID_RATIO` (bigger buckets), smaller / retuned `NPROBE`, `KNN_K` if needed — no change of semantics (same bucketing algorithm, bucket order, bucket-pair candidate rule, prune, reverse edges). **Bound: average bucket size 128–512 points** (today ~100; SIFT100M `CENTROID_RATIO` ≈ 1/512 … 1/128 ≈ 0.002–0.0078, i.e. ~195K–780K centroids), applied the same way to SIFT1M/10M unless a reason is logged. ~10K-point buckets are only for truly huge datasets (e.g. SIFT1B) — not now.
- **Deferred: A3 — bounded per-node reverse list** (user, 2026-10-22): do not try until the user reopens it. B6 (exact GPU reverse-edge collection, same CAGRA rule) stays a normal implementation idea.
- **TAGORE V100 fixes** (user, 2026-10-22): approved; keep them only in `/home/lanlu/Tagore/build-v100/patched-src` + `scripts/tagore_v100_fix.patch`, never modify `/home/lanlu/Tagore` sources.

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

## Test commands (used by phases P2 / P3 and step 3)
```bash
WT=/home/lanlu/microBenchmark_tagore
SK=$WT/.claude/skills/optimize-bucket2-tagore
S=$SK/scripts
R=$WT/bucketDemo/buildBucket/runs/tagore<N>
bash $S/run_pipeline.sh -d sift1M      -m auto  -o $R/sift1M_auto       # resident path, 3 reps (datasets.json)
bash $S/run_pipeline.sh -d sift1M      -m large -o $R/sift1M_large      # large path forced, 3 reps
bash $S/run_pipeline.sh -d sift10M     -m auto  -o $R/sift10M_auto      # → large path, 3 reps (~4 min)
bash $S/run_pipeline.sh -d msTuring1M  -m auto  -o $R/msTuring1M_auto   # float32 D=100, resident, 3 reps, DiskANN search
bash $S/run_pipeline.sh -d msTuring10M -m auto  -o $R/msTuring10M_auto  # float32 D=100, 3 reps, DiskANN search
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/msTuring1M_auto $R/msTuring10M_auto --quick   # pre-check
bash $S/run_pipeline.sh -d sift100M    -m auto  -o $R/sift100M_auto     # 2 rounds, 3 reps (~35 min), DiskANN search
```
- Reps come from `datasets.json` (`-r` overrides). Run in this order; if the `--quick` pre-check
  fails (correctness, regression, quality, memory), skip SIFT100M and reject.
- Launch every run detached (`setsid nohup bash $S/run_pipeline.sh … > $R/<cfg>.out 2>&1 &`) so a
  harness restart cannot kill it, and wait with Monitor / an until-loop on the process (no
  sleep-polling). The run blocks on the bench lock by itself while another loop is timing; the
  wait is not part of the measured time. If GPU 0 is used by a process that does not hold the
  lock, wait for it.
- Non-zero exit / "FAILED" = failure (rule out environment first: busy GPU, `df -h /home/lanlu` must keep > 50 GB free; 100M work dirs are large).
- If a 100M run died in optimize_chunked / translate / search with the work dir kept,
  `$S/resume_large_tail.sh -d sift100M -o $R/sift100M_auto` reruns only the tail (the build time is then the sum of the kept stage times; say so in the log).
- Quality per run: (a) pre-prune `knn_similarity` (recall@1/10/32/64, logged, not gated);
  (b) final graph search per `datasets.json` `search` (cagra: itopk 32…512, diskann: L 32…512),
  recall@10 + QPS + defect counts. Always log recall and QPS.

## Step selection

- Always work in `$WT` (`git -C $WT branch --show-current` must be `opt/bucket2-tagore`; if the session was started elsewhere, use absolute `$WT` paths for everything).
- `$SK/state/best.json` does not exist → **step P** (setup + baseline verification).
- else `state/tagore_baseline.json` does not exist → **step B** (TAGORE baseline).
- else `notes/tagore_map.md` does not exist → **step T0** (study).
- else the first phase of the **Restart plan** not listed as done in `$SK/state/phase.json` (`{"done": ["P1", …]}`; missing file = nothing done) → work on that phase (P1–P4); one coherent step per invocation, phases P2 / P5 use the iteration steps 0–6.
- else (P1–P4 done) → **one iteration** (steps 0–6) = phase P5.

### P. Setup + verify the main baseline (first invocation, no optimization)
(Done 2026-10-22; kept for reference. Its ±7% SIFT100M tolerance was the rule of that time.)
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

## Restart plan (user decision 2026-10-23, 2nd) — phases P1 → P5

Executed in this order after the loop restarts (step selection uses `state/phase.json`). Mark a phase
done by adding it to `phase.json`, a `### Phase P<k> done — <summary>` log entry and a commit.

### P1. CAGRA + TAGORE build and search baselines for every registered dataset
For sift1M, sift10M, sift100M, msTuring1M, msTuring10M, msTuring100M (no source change to our pipeline):
- **CAGRA** (cuVS 26.10 native): add float32 (and int8) support to `cagra_eval` in the **worktree's own**
  `bucketDemo/evalCuvs` (build it there; never rebuild the shared `/home/lanlu/microBenchmark/...` binary),
  build NN-Descent + optimize. Datasets that do not fit the 16 GB V100 in-memory build (100M): try cuVS's
  host-memory / batched build paths available in 26.10 public API; if impossible, record
  `status: infeasible` + reason. Search: `cagra_eval search` when index + data fit the GPU, else DiskANN
  (`diskann_search.py`, medoid entry) on the CAGRA graph.
- **TAGORE**: as in step B (patched copy only). For float32 / D=100 build a separate TAGORE copy with
  `DIM_SIZE` = 100 (`include/common.cuh`; patched-src copy, never the clone), fvecs input converted from
  `.fbin` into the scratchpad; 1M in-memory `GNN_descent` + CAGRA prune, 10M / 100M `largeIndex`
  (Vamana; prep timed separately).
- **Quality target (user): recall@10 ≈ 0.95 or above.** Interpretation (tell the user if it is wrong):
  the reference must reach recall@10 ≥ 0.95 within the standard sweep at itopk / L = 64 (the mid
  operating point); if it does not, rebuild with larger degrees (graph 32 → 64, intermediate 64 → 128,
  for TAGORE k / final_degree likewise) and record both. Keep the **degree-32 configuration as the gate
  reference** (same final degree as ours — apples to apples): `state/<ref>_baseline.json[dataset]` top-level
  = gate config; higher-degree configs go into `[dataset]["configs"]` for reporting only.
- Every timed run under the bench lock (GPU 0), detached; record build time (+ prep), search recall@10 / QPS
  per itopk or L, defects, GPU / host peak, config (degrees, iterations, search tool).
- Set each dataset's `ref` in `datasets.json`: SIFT stays `tagore`; msTuring → `cagra` (user-approved
  2026-10-23) once its CAGRA gate config exists. Summarize all references in one log table. Done when every
  dataset has a CAGRA and a TAGORE entry (or a recorded infeasibility).

### P2. Port every uint8-only optimization to float32 (judged iterations)
Start by grepping for uint8-only paths (`kIsInt8Path`, `uint8_t` kernels, `.u8bin` assumptions,
`use_dev_cache`) and list them in the log; then one item per iteration (steps 0–6), uint8 configs must not
regress, float configs gain or at least become runnable:
1. **float32 large-path crash** (first): `SMOKE=1 … -d msTuring1M -m large` → bucket2 Step 6
   `FATAL: ReorderedDatasetBucketReader: read failed for bucket 537` (resident path works: msTuring1M auto
   6.96 s, 0 defects). Fix + bounds check; verify msTuring1M large and msTuring10M end to end; record the
   msTuring configs with `judge.py … --update --add-new`.
2. **prune GPU kernel for every element type** (`prune_windowed_gpu` templated on uint8 / int8 / float,
   fp32 distances).
3. **Step 6 out-of-core GPU bucket cache + batched misses + deferred frees + block reuse** (#1 / #2 / #21)
   for float32. fp32 GEMM keeps results exact (implementation); fp16 GEMM for float data changes results
   unless the top candidates are re-ranked exactly in fp32 → fp16 without exact re-rank = `A<k>` for the user.
4. every other uint8-gated path found by the grep (Step 4 / KMeans‖ GEMM precision, host-blob loaders,
   resident Step 6, …).
Done when the grep list is empty or each remaining item has a recorded reason.

### P3. End-to-end check + full summary
- Run every registered dataset with HEAD (bench set ×3 + msTuring100M ×3 + `-m large` on msTuring1M) and
  confirm uint8 and float32 both run cleanly (0 defects, expected paths).
- Re-record our bests as 3-run medians: `judge.py <dirs> --update --rebaseline --add-new --md` (replaces the
  single-run SIFT100M entry; adds the msTuring configs).
- Write `docs/summary_<date>.md` and a log entry: for every dataset × {ours (auto / large), CAGRA, TAGORE
  (gate + higher-degree configs)} the build time (with stages / prep), recall@10 + QPS per itopk / L,
  defects, GPU / host peak memory, search tool used.

### P4. Document the dataset-specific tunables + calibrate msTuring
- Write **`docs/dataset_tuning_params.md`**: every parameter that was tuned for SIFT100M / this machine or
  should be re-tuned for a new dataset or size, one row each: name, where (file:line or flag / script),
  current value per dataset, what it controls, how it is derived now (`dataset_env.py` rule vs hard-coded),
  symptoms when it is wrong, how to tune it (sweep range, which config to watch), class (resource =
  implementation vs algorithm = A2 / user approval). Grep the sources for hard-coded sizes. Seed list (at
  least): `CENTROID_RATIO` / avg bucket, `NPROBE` / pool rows, `KNN_K`, `ITERATIONS`, `SEED`,
  `--kmeans-train-fraction`, `--cache-mb`, `--bucket-mem-mb`, `--gpu-bucket-cache-mb`, `PRUNE_CACHE_MB` and
  the prune chunk budget (4 GB), `CHUNKED_GPU_BUDGET_MB`, `CHUNKED_EDGE_BUFFER_MB`, `--host-budget-mb`,
  Step 6 slot count (6) and the dots row tile (256), `order_window` (128), the CSR row-block size (8192 rows),
  fp16 vs fp32 GEMM choice per dtype, the resident / out-of-core fit check, DiskANN / CAGRA search sweep
  (itopk / L list), `NEIGHBORS_M` / `OUTPUT_DEGREE` (fixed by the user; listed for completeness).
- Then apply it to msTuring (former step C): sweep `CENTROID_RATIO` (avg bucket 128–512) × `NPROBE` (report
  ITERATIONS) on msTuring1M / 10M until they pass the CAGRA gate reference (−0.01 resident / −0.05 large);
  with SIFT parameters msTuring1M reaches only pre-prune recall@10 0.687 (SIFT1M 0.943). Store the chosen
  values as `datasets.json` `"params"` (A2, approved), re-record the msTuring bests (`--rebaseline`), log the
  sweep table, and add the result to the doc as the worked example.

### P5. Continue the normal loop
Steps 0–6 with the backlog: B10's open steps (incl. 1B readiness: per-stage RSS, Phase 1 buffers from the
host budget, 64-bit N×K audit, `dataset_env.py` sanity at N = 1e9 — ask the user before
downloading / generating a ~128 GB 1B base), A1, B7–B9a, and new ideas for the largest SIFT100M / msTuring
stages.

## One iteration

### 0. Orient (cheap)
- In `$WT` on `opt/bucket2-tagore`, source tree clean (if interrupted: `git -C $WT checkout -- bucketDemo/buildBucket/src $S/params.sh`, log it as "interrupted"; check for orphaned detached runs of this loop with `pgrep -af microBenchmark_tagore` first).
- Read `OPTIMIZATION_LOG_TAGORE.md`: **Current best**, **Ideas backlog**, **Awaiting approval** (any newly APPROVED entries?), **Tried & rejected** (the **Possible gains** section is history since 2026-10-23); and `notes/tagore_map.md`.
- Check the bench lock owner (`cat /home/lanlu/.bench_locks/bucket2_bench.lock.owner`): if the cuVS loop is timing, do steps 1–2 gently (niced build, SMOKE tests) and let the timed runs queue.

### 1. Pick ONE idea
- During phase P2 the candidate is the next open float-port item of P2 (see **Restart plan**); in phase P5 pick normally:
- Candidates otherwise: implementation ideas from the backlog, plus `A<k>` entries the user APPROVED (A1: follow the split in `docs/A1_local_join.md` §2.8). Start from where SIFT100M time goes (largest stage first).
- Name the TAGORE function(s) the idea comes from and, in one line, how cuVS 26.12 / our current code does the same step.
- If the idea you want turns out to be an algorithm change and isn't approved: add it to **Awaiting approval**, pick the next implementation idea.
- One coherent change so the result is attributable. Write the plan in one or two sentences before editing (it becomes the log entry's "idea").
- Smoke-test new large-path code first (`SMOKE=1 … -d sift1M -m large`, `-d msTuring1M` for float code, or SIFT10M) before spending ~35 min on SIFT100M.

### 2. Implement
- Edit sources and/or `$S/params.sh`. Match the surrounding style (Chinese/English comment mix, naming). Credit the source in a short comment, e.g. `// 参考 TAGORE nn_descent_opt_merge: k-th 距离阈值预过滤`.
- Build with ninja. Compile error → fix; infeasible → revert and log as rejected ("infeasible: ..."), then continue with the next idea in the next invocation.

### 3. Test
Run the bench set into `runs/tagore<N>/` (quick pre-check first, SIFT100M only if it passes).

### 4. Judge
```bash
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/msTuring1M_auto $R/msTuring10M_auto $R/sift100M_auto --md
```
Rules: see Goal and judge.py's docstring (keys come from each run's metrics.json, so the order of the dirs does not matter).

### 5a. ACCEPT
1. `python3 $S/judge.py $R/sift1M_auto $R/sift1M_large $R/sift10M_auto $R/msTuring1M_auto $R/msTuring10M_auto $R/sift100M_auto --update --md`.
2. Append `### #<N> — <idea>` to `OPTIMIZATION_LOG_TAGORE.md` with: idea, **TAGORE source** (file + function), class (implementation / approved `A<k>`), the `--md` tables (incl. `vs TAGORE`), `a1_stats` for A1 runs; update **Current best**.
3. Commit on `opt/bucket2-tagore`: sources, `$S/params.sh`, `$SK/state/best.json`, the log, notes / docs if changed. Message `bucket2 tagore #<N>: <idea>` + attribution line. Don't commit `runs/`.

`<N>` = this log's last iteration + 1 (own numbering, starting after #0).

### 5b. REJECT
Applies to every failure in an optimization iteration (judge reject, build error, crash, timeout, wrong path selection, graph defects, quick checks failed). Never stop the loop for it.
1. Keep the patch for later: `git -C $WT diff > $WT/bucketDemo/patches/tagore<N>_<short-name>.patch`.
2. `git -C $WT checkout -- bucketDemo/buildBucket/src $S/params.sh` and rebuild.
3. One line under **Tried & rejected** with the concrete failure: idea (TAGORE function), which config / stage failed, the measured numbers vs best (time per config, recall@10, QPS, memory) or the error message, the judge reason, the patch path. For an A1 step also paste `a1_stats` and update `docs/A1_local_join.md` with what was learned. Commit the log and the patch (and notes / docs if changed).

### 6. Housekeeping & return
- Delete `runs/tagore<N>` dirs older than the last ~5 iterations except those referenced by `state/best.json` and `runs/tagore0`. SIFT100M work dirs are deleted by run_pipeline; check `df -h /home/lanlu` (> 50 GB free). Never delete the cuVS loop's `runs/iter*`.
- Update `notes/tagore_map.md` with what this iteration taught (worked / didn't and why); re-rank the backlog.
- If no implementation idea is left and nothing is approved: say so in the status line, list the pending `A<k>`, and keep going with the best remaining general idea (from the backlog or profiling).
- End with a 2–3 line status: iteration N, idea + TAGORE source, verdict, new best totals (SIFT100M first, then msTuring10M), number of `A<k>` awaiting approval.
