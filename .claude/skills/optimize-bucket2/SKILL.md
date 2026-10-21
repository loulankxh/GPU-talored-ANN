---
name: optimize-bucket2
description: One iteration of the bucket2 build-pipeline optimization loop (bucket2 -> optimize | prune_windowed+optimize_chunked -> translate_ids) in bucketDemo/buildBucket — pick one idea (cross-checked against cuVS 26.12 source), implement, build against cuVS 26.10, benchmark SIFT1M (GPU-resident path + large-dataset path) on GPU 0, accept/reject with judge.py, and log accepted results to bucketDemo/OPTIMIZATION_LOG.md. Run repeatedly with `/loop /optimize-bucket2`.
---

# bucket2 optimization loop — one iteration per invocation

Runs **fully automatically** (no confirmation per round). Each invocation does exactly
one idea end-to-end, then returns; `/loop /optimize-bucket2` keeps it going until the
user stops it. Never stop the loop on your own.

## Fixed context

| what | where |
|---|---|
| code under optimization | `bucketDemo/buildBucket/src/` — `bucket.cu` (bucket2), `optimize.cu` (resident prune+reverse), `prune_windowed.cpp` + `prune_windowed_core.hpp`, `optimize_chunked.cu` (method E), `translate_ids.cpp` + `translate_graph_ids.hpp`, shared `*.hpp` |
| pipeline parameters | `.claude/skills/optimize-bucket2/scripts/params.sh` |
| reference source (read-only) | `/home/lanlu/cuvs` = cuVS **26.12**, under `cpp/src/neighbors/`. **Two CAGRA build flavours to learn from:** (1) NN-Descent — `detail/nn_descent.cuh`, `detail/nn_descent_gnnd.hpp`; (2) **IVF-PQ** — `detail/cagra/cagra_build.cuh` (`build_knn_graph` with ivf_pq: batched queries, pinned/managed staging, overlap of search + refine, `refine_rate`), `ivf_pq/ivf_pq_build.cuh` (sampled balanced-kmeans training, `process_and_fill_codes`), `ivf_pq/ivf_pq_search.cuh`, `detail/refine*`. Also `detail/cagra/graph_core.cuh` (sort / prune / reverse edges / merge), `detail/cagra/cagra_ace.cuh` (ACE: partitioned out-of-core CAGRA build — closest to our large path), `all_neighbors/` (batched / clustered kNN — closest to our bucket design), `detail/knn_brute_force.cuh`, `detail/fused_l2_knn.cuh`, `detail/faiss_select/`, `../cluster/detail/kmeans*`. raft headers: `/home/lanlu/miniconda3/envs/cuvs-bench/include/raft` |
| link-time dependency | conda env `cuvs-bench` = libcuvs / libraft / librmm **26.10**, nvcc 12.9. Only the **public** cuVS/raft API is available (no `detail::` from 26.10 headers). Porting a 26.12 kernel = copy it into our source, don't include `/home/lanlu/cuvs/cpp/src` |
| toolchain | conda env `cuvs-build` (gcc 14, cmake 4, ninja, boost 1.87). Build: `cd bucketDemo/buildBucket && /home/lanlu/miniconda3/envs/cuvs-build/bin/ninja -C build` (reconfigure: `CONDA_PREFIX=/home/lanlu/miniconda3/envs/cuvs-bench /home/lanlu/miniconda3/envs/cuvs-build/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=/home/lanlu/miniconda3/envs/cuvs-build/bin/ninja`) |
| eval tool | `bucketDemo/evalCuvs/build/cagra_eval` (cuVS 26.10 CAGRA build / search sweep) |
| datasets | `dataset/sift1M/` and `dataset/sift10M/` (`base.*.u8bin`, `query.public.10K.u8bin`, `groundtruth.ibin` (k=100), `base_knn_sample10K_k64.ibin` (exact KNN of 10K sampled base points)) |
| CAGRA reference | `state/cagra_baseline.json` (cuVS 26.10 NN-Descent 64→32: SIFT1M 11.54s, SIFT10M 122.6s, recall table) |
| current best | `state/best.json` (written only by `judge.py --update`) |
| log | `bucketDemo/OPTIMIZATION_LOG.md` |
| branch | `opt/bucket2-loop` (commit every accepted iteration there; never push) |
| GPU | **GPU 0 only** (`CUDA_VISIBLE_DEVICES=0`, scripts set it) |

## Goal

Make both SIFT1M configs faster — `sift1M_auto` (GPU-resident path) and `sift1M_large`
(large-dataset path forced on SIFT1M) — without hurting search quality. **SIFT10M is not
tested in the loop for now** (user decision 2026-10-21); the long-term target is still a
SIFT10M large-path build ≤ 360 s, so prefer large-path ideas that scale with N (and never
ones that only win because SIFT1M is small). Keep improving until the user stops the loop.

## Design invariants — never violate

The pipeline is designed for datasets far larger than GPU memory. An optimization may
change *how* a stage is done, never remove or bypass a stage:

1. **Bucketing** — data is partitioned into buckets (centroid selection + assignment).
2. **Bucket order** — buckets are processed in the DiskJoin-style order derived from the centroid KNN graph.
3. **Bucket-pair computation** — per-vector KNN comes from bucket × neighbor-bucket computation, not from a global index.
4. **Index prune** (CAGRA-style rank-based pruning).
5. **Reverse-edge adding** (and merge).
6. **ID translation** back to the original order (`translate_ids`).

Also:
- Build index degree **64** (`NEIGHBORS_M`), final degree after prune + reverse edges **32** (`OUTPUT_DEGREE`). Fixed.
- **Two paths must both keep working:** if the data fits on the GPU (SIFT1M), the fully GPU-resident path (bucket2 resident + `optimize` = cuVS `helpers::optimize`); otherwise bucket2 out-of-core + `prune_windowed` + `optimize_chunked -m E`. The auto-selection (bucket2's fit check, `optimize`'s exit code 2) must stay correct. Never make the large path depend on holding the full dataset or full graph in GPU memory, and host memory must stay bounded by the configured budgets — judge.py rejects >1.2× peak growth on the large path.
- Never replace the pipeline with `cagra::build` / nn_descent on the whole dataset.
- Algorithm parameters in `params.sh` (centroid ratio, nprobe, knn-k, iterations, cache sizes, ...) **may** be tuned; quality is enforced by judge.py.
- Keep boundary checks: final graph must have no out-of-range ids, self loops, or duplicate edges (cagra_eval reports them; judge rejects). Add new bounds checks when you add new indexing code.

## One iteration

### 0. Orient (cheap)
- If `state/best.json` does not exist, this invocation is **iteration 0 (baseline)**: skip steps 1–2, run the two tests of step 3 into `runs/iter0/`, then `judge.py $R/sift1M_auto $R/sift1M_large --update --md`, fill the log's **Current best** block, commit, and return.
- `git -C /home/lanlu/microBenchmark status --short` on branch `opt/bucket2-loop`; the source tree must be clean (if a previous iteration was interrupted, `git checkout -- bucketDemo/buildBucket/src .claude/skills/optimize-bucket2/scripts/params.sh` first and log it as "interrupted").
- Read `OPTIMIZATION_LOG.md`: the **Current best** block, the **Ideas backlog**, and the **Tried & rejected** list. Never retry a rejected idea unless you write down what is different this time.
- `nvidia-smi -i 0`: if another process is using GPU 0, wait for it (use Monitor/until-loop, don't sleep-poll) — timings with a shared GPU are invalid.

### 1. Pick ONE idea
- Start from where time goes: the latest `stages_s` and `bucket2_steps_s` in `state/best.json` (largest stage first, both SIFT1M-auto and SIFT1M-large).
- Ground the idea in the cuVS 26.12 source: find how CAGRA / NN-Descent / all_neighbors / brute-force / select_k / kmeans do the equivalent step and note the file + function you took the idea from. If profiling is needed, `nsys profile` on SIFT1M is fine (write output under the scratchpad, not the repo).
- Keep it to one coherent change so the result is attributable.
- Write the plan in one or two sentences before editing (it becomes the log entry's "idea").

### 2. Implement
- Edit sources and/or `params.sh`. Match the surrounding code style (Chinese/English comment mix, naming).
- Build with ninja (command above). Compile error → fix; if the idea turns out infeasible, revert and log it as rejected ("infeasible: ...").

### 3. Test (both, sequentially, GPU 0 idle)
```bash
S=/home/lanlu/microBenchmark/.claude/skills/optimize-bucket2/scripts
R=/home/lanlu/microBenchmark/bucketDemo/buildBucket/runs/iter<N>
bash $S/run_pipeline.sh -d sift1M -m auto  -r 3 -o $R/sift1M_auto     # GPU-resident path: timing median of 3 + quality
bash $S/run_pipeline.sh -d sift1M -m large -r 3 -o $R/sift1M_large    # large-dataset path (out-of-core + prune_windowed + optimize_chunked -m E): median of 3 + quality + memory
```
Run them with `run_in_background` and wait for the completion notification. Any non-zero exit / "FAILED" = reject (after checking it isn't an environment issue such as a busy GPU or full disk; `df -h /home/lanlu` must keep > 50 GB free).
Quality measured per run: (a) `knn_similarity` — pre-prune degree-64 `vector_knn` vs exact KNN (recall@1/10/32/64 on 10K sampled points); (b) `search` — final translated degree-32 graph through the cuVS CAGRA searcher, **recall@10 and QPS** at itopk 32…512 (+ graph defect counts). Always keep both recall and QPS in the log.

### 4. Judge
```bash
python3 $S/judge.py $R/sift1M_auto $R/sift1M_large --md
```
Rules are in judge.py's docstring (≥3% faster on SIFT1M-auto or SIFT1M-large, neither >3% slower; recall@10 at every itopk ≥ best−1%, and vs CAGRA: GPU-resident path ≥ CAGRA−1%, large-dataset path ≥ CAGRA−5%; pre-prune recall@64 ≥ best−1%; large-path memory ≤ 1.2× best; no graph defects).
- If a gain is borderline (3–5%), rerun that config once more into `<dir>_b` and judge with it too; accept only if both pass.

### 5a. ACCEPT
1. `python3 $S/judge.py $R/sift1M_auto $R/sift1M_large --update --md` (writes `state/best.json`).
2. Append an entry to `OPTIMIZATION_LOG.md` (template in the log), pasting the `--md` tables, and update its **Current best** block.
3. Commit on `opt/bucket2-loop`: sources, `params.sh`, `state/best.json`, the log. Message: `bucket2 opt #<N>: <idea>` + the attribution line from the system prompt. Don't commit `runs/`.

### 5b. REJECT
1. Revert code + params: `git checkout -- bucketDemo/buildBucket/src .claude/skills/optimize-bucket2/scripts/params.sh` and rebuild (so the next iteration starts from best).
2. Add one line to **Tried & rejected** in the log: idea, numbers that failed, judge reason. Commit the log only.

### 6. Housekeeping & return
- `run_pipeline.sh` deletes the large intermediates; delete `runs/iter<N>` dirs older than the last ~5 iterations except those referenced by `state/best.json`.
- Refill the **Ideas backlog** in the log with anything learned.
- End with a 2–3 line status: iteration N, idea, verdict, new best totals.
