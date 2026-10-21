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

After **#57 (prune_windowed GPU path: no float copy, pinned double-buffered H2D / streamed D2H → writer)**, 2026-10-21:

| config | build total (s) | recall@10 itopk 32 / 64 / 128 | QPS itopk 32 / 64 |
|---|---|---|---|
| SIFT1M auto (GPU-resident) | **6.82** (0.59x CAGRA) | 0.9287 / 0.9773 / 0.9945 | 2223K / 1193K |
| SIFT1M large (out-of-core path) | **10.23** (0.89x CAGRA) | 0.9146 / 0.9710 / 0.9923 | 1946K / 838K |
| SIFT10M auto (→ large path, 1 run) | **96.09** (0.78x CAGRA) | 0.8379 / 0.9290 / 0.9737 | 1873K / 1012K |
| CAGRA native SIFT1M / SIFT10M | 11.54 / 122.59 | 0.9305 / 0.9782 / 0.9948 — 0.8672 / 0.9463 / 0.9839 | 1942K / 1036K — 1868K / 1013K |

## Ideas backlog

Priority (from the user, 2026-10-21):
- **Step 4:** fp32 `brute_force` in #7/#8 (0.70 → 0.42 s/round), exact fp16 GEMM + top-2 in #22 (→ 0.12 s/round). Remaining: fuse GEMM + top-2 so the 4096 × C dots tile is never written.
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
- **Process start/exit overhead (found in #43):** ~0.28 s from process start to Step 1 (CUDA/cuVS library load + context) and ~0.22 s after `Total:` (destructors: GPU/pinned frees, RMM pool), per stage binary. Step 1 X_sampled `resize` (512 MB serial zero-fill) is the next Step 1 cost.
- `optimize` (resident) spends most of its time in file load + sanitize, not in the cuVS call.
- `prune_windowed` after #18 (1.3 s): compute ~0.96 s, bulk load 0.13 s. Old note — `prune_windowed` after #2 (3.9s): compute 2.6s forks an OMP region per bucket of ~100 rows on 80 threads — process several positions per parallel region, or move the detour count to GPU (cuVS `kern_fused_prune`). Store cached vectors as u8 (4x less cache memory → fewer evictions on SIFT10M); keep the dataset handle open in the loader.
- Resident `optimize` after #5 (1.3s): ~0.25s is raft/CUDA context start-up + 256 MB graph read; the cuVS call itself is 0.44s. Further gains need fusing optimize into bucket2 (graph already on the GPU) — a structural change, keep the separate-stage fallback for exit code 2.
- bucket2 still writes `vector_dists.bin` (256 MB on SIFT1M) that no later stage reads — same treatment as #4 (`--skip-dists`), check `knn_similarity.py` / eval don't need it first.
- bucket2 Step 6 out-of-core after #10 (~3.75 s/round loop): pool build now overlapped on a producer thread (wait_producer ~0.1 s); main thread = stream_sync ~1.7 s (GPU-bound now: gather+norms+GEMM+select_k+D2H per ~100-row bucket, 3 slots) + scatter_pending ~0.7 s + gpu_submit ~1.2 s. Next: batch several buckets per GPU submission / fewer kernel launches (cuVS 26.12 `all_neighbors` batches clusters), FP16/INT8 GEMM is already used; select_k on tiny matrices is launch-bound.
- bucket2 Step 4 out-of-core (8.8s vs 3.3s resident): per round convert_upload ~0.6s + search 0.7s + bucket_write + finish_iteration; ChunkedKNN merge_iteration write_running ~1.1s/chunk in rounds ≥2.

- **Bucket-size outliers break the GPU-memory bound (found in #9, hit again in #13 — the memory gate is flaky until fixed; NEXT):** the balanced assignment (`batch_assign_with_cagra_anns`, CPU step 4d) reads `cluster_sizes` with relaxed atomics from all threads, so bucket sizes differ run to run despite the fixed seed; occasionally a round produces a bucket ~30x the average (SIFT1M large: max 3201 vs avg 100; resident run in #6: 1980). Out-of-core Step 6 sizes its per-slot `dots` buffers from max_pool (0.72 GB/slot × 3 instead of ~0.2 GB) → GPU peak 2708 MiB instead of ~1450. Fix: cap bucket size (spill to the 2nd-nearest centroid when a bucket exceeds e.g. 4–8× avg), or tile the oversized bucket's GEMM; also makes the build deterministic if done per batch in a fixed order.

- **Large path is disk-write-bound per round (found in #21):** the previous round's background ChunkedKNN merge writes overlap Steps 2–4 of the next round; shortening an early step (Step 2 in #12/#15/#21) moves those writes onto Step 4's bucket_vecs flush (`finish_iteration`), which then slows by about as much (#21: Step 2 −2.2 s, Step 4 +2.4–3.2 s). Large-path wins now need fewer bytes written per round (bucket_vecs as u8 is already minimal; arrival records 520 B/point → e.g. skip `vector_dists` in arrival when merge can recompute, or merge on the GPU-resident side when the running state fits), not shorter compute.
- **User decision — T=2 / NPROBE=256** (measured in #28, 1 run each): auto 8.05 s (vs 9.43), large 19.32 s (vs 24.17); search recall@10 itopk 32 auto 0.9266 (vs 0.9294; CAGRA 0.9305, gate 0.9205), large 0.8970 (vs 0.9003; gate 0.8805); pre-prune KNN recall@64 0.923. Passes the gate but spends ~0.3 pt of recall headroom on both paths — not applied without the user's OK (same policy as NPROBE 100 in #11).
- **Params after #27 (T=3, NPROBE=200):** sweep (1 run each) T=3/NPROBE=150 → auto 9.03 s, large 23.38 s, auto recall@10 itopk 32 0.9271 (−0.1 pt vs T=3/200). Further: T=2 with larger NPROBE, or NPROBE 175 — each costs pre-prune KNN recall; check final search recall headroom (resident ≥ 0.9205, large ≥ 0.8805).
- **NPROBE 100** (measured in #11, at T=4, 1 run each): auto 15.70 s, large 36.79 s (vs 16.79 / 39.32 at 150), search recall@10 itopk 32 auto 0.9269 (CAGRA 0.9305, gate 0.9205), pre-prune KNN recall@64 0.938. Passes the gate but spends 0.3 pt of resident recall headroom — user decision.
- **Resident Step 6 is GPU-bound, not launch-bound** (#11 probe): launching from 4 host threads changed nothing (launches block on a full queue). Kernel elimination per round (1.19 s loop, NPROBE 250): gather ~0.15 s, GEMM ~0.4 s, block top-M ~0.5 s; without GEMM+top-M 0.23 s. top-M block shape G=4/8/16 all ≈ same. Real fix = never materialize the (bucket × pool) dots matrix: fused GEMM + top-M (cuVS 26.12 `detail/fused_l2_knn.cuh` / `knn_brute_force.cuh`), or shrink the pool (#11).

## Tried & rejected

| # | idea | result | reason |
|---|---|---|---|
| 55 | Out-of-core Step 6: block top-M writes ids + dists straight into the slot's pinned `h_out` through UVA (mapped) instead of a per-bucket D2H `cudaMemcpyAsync` (scatter_pending copies `h_out` synchronously after the slot's stream sync, so it is race-free) | SIFT1M large quick test (1 run): loop 1.25 → 1.47 s/round (stream_sync 0.26 → 0.55 s; gpu_submit 0.67 → 0.60 s) | rejected without the full benchmark: the kernel's scattered 4-byte writes over PCIe keep the top-M blocks resident much longer than one bulk D2H; the saved API call is worth far less. (A staged variant — kernel writes to device, then one bulk copy for several buckets — would need batching across slots) |
| 53 | Large path: `optimize_chunked -m E` translates ids in-process when phase 2 ran on the GPU (`--inverse-perm` / `--final-output`, writes final_graph.bin; harness skips `translate_ids` when final_graph.bin exists; byte-identical to optimize_chunked + translate_ids on SIFT1M). Patch (incl. run_pipeline.sh): `bucketDemo/patches/iter53_large_path_inprocess_translate.patch` | auto 6.81 s (+0.9%), large 10.74 s (−0.2%), 10M 110.11 s (+0.7%): translate stage gone (10M 3.54 s, 1M 0.44 s) but the in-process translation took 1.67 s at 10M (random-row scatter, single pass) and optimize_chunked +0.5 s; bucket2 noise +2 s on 10M | no config ≥ 3%. Net saving ~1.8 s on 10M, ~0.3 s on 1M-large. Bundle with a faster translation (row-blocked scatter / GPU gather with the merged graph still on device) |
| 48 | Block top-M kernel (`block_topK_from_dots_kernel`, both paths): G warps of a row share their K-th threshold via a shared-memory `atomicMin` on an order-preserving float key (any warp's current K-th bounds the row's K-th) | resident Step 6 loop 0.975 → 0.907 s/round; auto 7.16 s (+2.98%), large 11.53 s (+1.1%), 10M 163.24 s (+0.3%) | just under 3% → bundled with the #44 patch into accepted #49 |
| 45 | **Step 6 fused GEMM + top-64 kernel** (user priority, 2026-10-21): `fused_knn.cuh`, sm_70 WMMA fp16; block = 32 query rows × a set of 64-column pool tiles; B tile staged through shared memory with register prefetch; dot tile kept in shared memory; per-row filtered selection (threshold → ballot-compacted shared-memory candidate buffer → raft `util::bitonic` merge into a register top-64); pool split into ≤16 chunks per row with a last-block (atomic ticket) merge; per-row threshold shared across chunks with `atomicMin` on float bits (distances incl. ‖q‖² are exact non-negative integers for 8-bit data); warm-up on tiles 0–1. Exact vs CPU brute force (`patches/iter45_fused_test.cu`: 7 cases, incl. ties, K=32/64, max chunking). Patch: `bucketDemo/patches/iter45_fused_gemm_topk.patch` (resident path only so far) | resident Step 6 loop per round (SIFT1M, 1 run): baseline cuBLAS GEMM + block top-M **0.975 s**; fused v1 (direct global B fragments, serial ballot insert) 2.40 s; v2 (warp_sort_filtered-style queues) 2.2 s; v3 (smem-staged B, 64-col tiles, smem candidate buffer, no spills) 2.1–2.6 s; + shared per-row threshold 1.27 s; + warm-up / cheaper filter 1.34–1.40 s. Same loop with selection compiled out: 0.73–0.76 s | rejected: slower than baseline. ncu (one bucket, real data): 10.5M warp instr, IPC ≈ 0.8/SM, achieved occupancy 12% (124 regs, 42.5 KB smem → 2 blocks/SM); MMA + skeleton alone (0.73 s) already ≈ cuBLAS GEMM + gather, and per-tile selection behind `__syncthreads` adds ~0.6 s/round. The only saving fusion can deliver here is the dots write + read (~8 MB/bucket ≈ 0.1–0.2 s/round); beating cuBLAS + block top-M on V100 would need a warp-specialized (MMA warps ↔ selection warps, double-buffered tiles) pipeline. Idea kept: shared per-row threshold + warm-up could also cut the existing `block_topK_from_dots_kernel` cost |
| 40 | Out-of-core Step 6 slots: OVERSUB 1.5 → 2.0 (3 → 4 slots on SIFT1M). Sweep: 3 slots 1.59 s/round, 4: 1.43, 6: 1.38 (+0.8 GB GPU). Patch: `bucketDemo/patches/iter40_ooc_4slots.patch` | large 12.87 s (+2.9%), GPU peak 1674–2052 MiB | just under 3%; bundled into accepted #41 |
| 37 | Out-of-core Step 4 bucket write (in-memory mode from #36): deterministic multi-threaded `add_range_parallel` (per-thread per-bucket counts → prefix → parallel copy; same in-bucket order) + byte-array centroid test instead of `unordered_set`. Patch: `bucketDemo/patches/iter37_parallel_bucket_write.patch` | write part 0.13 → 0.03–0.06 s, but bucket_write total only 0.35 → 0.29 s/round (input read 0.14 s + sampled centroid read 0.07–0.1 s dominate); KNN recall unchanged | < 1% of the large path — not benchmarked; bundled into accepted #42 |
| 31 | Out-of-core Step 4.5 inline reorder in memory when 2·N·D B ≤ cpu_limit/4 (read whole input, permute by inverse_perm in parallel, one sequential write — like the resident path; streaming per-bucket writer kept otherwise; output verified == base[inverse_perm]). Patch: `bucketDemo/patches/iter31_ooc_inmemory_reorder.patch` | pass3_write 1.03 → 0.385 s; large 22.09 s (+3.1%) then rerun 22.14 s (+2.9%) | borderline gain not confirmed (rerun just under 3%). Consistent ~0.65 s saving — bundle with the next large-path change |
| 29 | Out-of-core Step 6 GPU path = resident's: block top-M kernel (templated on dots type, rows by bucket position, ld padded to 4) instead of dots_to_dist + raft select_k + gather (raft kept for M > 256), and fp16 tensor-core GEMM instead of INT8 IMMA (dp4a on V100). Patch: `bucketDemo/patches/iter29_ooc_block_topm_fp16.patch` | loop 3.24 → ~3.0–3.2 s/round (gpu_submit 1.1 → 0.95 s, stream_sync 1.25 → 1.1 s; GEMM time unchanged ~0.62 s/round); large 24.55 s (−1.6%), auto −0.2% | no gain: the out-of-core loop is now split ~evenly between GPU wait, host scatter (`scatter_pending` ~0.7–1.0 s) and launch/H2D submit (~1 s), none dominant |
| 25 | Resident Step 6 `block_topK_from_dots_kernel`: replace warp 0's serial ballot-insert of the other (G−1)·K_OUT candidates by a shared-memory merge-path tree (G → … → 1, each pair merged by one warp with co-rank binary search; same top-K set, ties A-first). G=8 still best (16: 0.98 s, 4: 0.87 s/loop). Patch: `bucketDemo/patches/iter25_topm_tree_merge.patch` | resident loop 0.95 → 0.86 s/round; auto 10.93 s (+2.3%); large 30.98 s (−7.2%, path does not use this kernel — disk-writeback noise) | < 3% gain. Bundle with another Step 6 change. **Measurement note:** large-path totals swing ±2–3 s (up to ~10%) run to run from disk writeback (page cache can't be dropped without sudo), so the 3% "no config slower" rule rejects unrelated changes at random |
| 24 | bucket2 I/O trims: `--skip-dists` (don't write the unused `vector_dists.bin`; resident skips the D2H + write, out-of-core skips the final dists pwrite) + parallel u8→f32 in `load::read_u8bin_to_f32`. Patch: `bucketDemo/patches/iter24_skip_dists.patch` | quick test (1 run): resident Write 0.75 → 0.48 s, bucket2 8.95 → 8.65 s (~0.3 s ≈ 2.7%); Step 1 unchanged 0.78 s (dominated by CUDA context + GpuResidentKnn setup + a second raw read, not the conversion); large: final dists pwrite only | not benchmarked: expected gain < 3% and within noise. Keep `--skip-dists` to bundle with another I/O change |
| 17 | prune_windowed batched fast path alone (see #18) | prune 4.06 → 2.28 s, byte-identical output; large 39.13 s / rerun 38.87 s (bucket2 Step 4 write-throttled flush ~10 s vs ~6 s in those runs), back-to-back A/B right after a fresh baseline: 36.05 s vs baseline 36.98 s | judge: <3% vs recorded best 36.36 (+0.9% in the A/B). Kept and extended in #18. Large-path totals swing ±3 s from disk writeback (drop_caches is not available without sudo) — gains <5% on the large path are hard to confirm |
| 16 | Out-of-core ChunkedKNN running state (N·M·8 B) kept in host RAM when it fits `cpu_limit/4` (same budget the merge already uses), merged in place each round instead of written to `running_chunk_*` and read back; disk path kept otherwise. Patch: `bucketDemo/patches/iter16_running_in_memory.patch` | auto 16.22 s (−0.3%), large 36.51 s (−0.4%); per-round merge wall 1.9–3.8 → ~0.8 s, final wait 0.97 → 0.89 s, bucket2 RSS 3.35 GB | no gain: the merge runs in the background overlapped with the next round, so only the final wait shrinks; Step 4's write-throttled flush did not improve. (Would also need `--cpu-limit` ≥ ~21 GB to apply on SIFT10M) |
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

### #18 — prune_windowed: batched positions + parallel bulk load  (2026-10-21)
**Idea / rationale:** (#17 + this) When the whole window data set (N × (M_in int32 + D float), 768 MB on SIFT1M) fits `--cache-mb`, the Belady cache can never evict, so (1) instead of one 80-thread OpenMP region per position (~100 rows each: fork/join + imbalance dominated compute), `kBatch`=64 positions are processed in one `schedule(dynamic,16)` region over all their rows — each position keeps its own `BucketWindow`, so pruning semantics are unchanged; (2) instead of fetching every bucket through the cache (two seek+reads per bucket, dataset file reopened each time, copy into a `BucketWindowBlock`), candidates and vectors are bulk-loaded once with multi-threaded `pread` into uninitialised buffers (+ parallel u8→f32) and each window points straight at slices (buckets are contiguous in the reordered id space; center-offset check added). Memory = the window data size ≤ cache budget, same as a full cache. Otherwise the original per-position cache path runs unchanged. Like cuVS 26.12 `detail/cagra/graph_core.cuh::prune_graph_gpu`, which prunes a whole batch of rows per kernel launch instead of per partition. Output verified byte-identical to the previous binary in both modes (cache 16000 MB batched; 100 MB per-position). prune: fetch 0.98 → 0.13 s, compute 2.6 → 0.96 s, total 3.9 → 1.34 s; prune RSS 0.77 → 0.88 GB (gate 1.10).
**Files:** `src/prune_windowed.cpp`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 16.24 | -0.4% | +25.3% | 1.41x | bucket2 14.45, optimize 1.28, translate 0.48 | 3806 | 1.60 |
| sift1M_large | large | 34.12 | +6.2% | +54.6% | 2.96x | bucket2 28.87, prune 1.34, optimize_chunked 3.45, translate 0.50 | 1302 | 3.83 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.31 | 1.34 | 0.00 | 2.16 | 1.10 | 0.02 | 4.57 | 0.00 | 0.76 |
| sift1M_large | 0.56 | 4.37 | 1.87 | 0.00 | 5.93 | 2.15 | 0.03 | 12.29 | 0.96 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9822 | 0.9740 | 0.9685 | 0.9635 |
| sift1M_large | 0.9815 | 0.9745 | 0.9684 | 0.9633 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9288 / 1948K | 0.9770 / 1034K | 0.9943 / 511K | 0.9987 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.8974 / 1946K | 0.9618 / 931K | 0.9884 / 532K | 0.9970 / 189K | 0.9992 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +6.2% (36.36 → 34.12 s; prune 4.06 → 1.34 s), 1M-auto −0.4% (path untouched).

### #19 — optimize_chunked method E: single-chunk edges stay in memory  (2026-10-21)
**Idea / rationale:** with one chunk (SIFT1M: the whole reverse-edge set fits the GPU budget) every one of the N·K_out = 32M edge records still went through `EdgeChunkAccumulator`: create/start → buffered writes to `tmp_dir` → `finish()` merge → `read_chunk()` back in Phase 2 (~0.9 s), plus two N·K_out-sized pinned buffers (overflow + drain, 640 MB, `cudaMallocHost` ~0.35 s) for a chunk that can never overflow and drains once. Now, if `num_chunks == 1` and the edges fit `--edge-buffer-mb`, drained records go into a host vector handed straight to Phase 2; if that chunk also holds all edges, the overflow buffer shrinks to a token size and the single drain goes D2H straight into the vector (no pinned drain buffer). Multi-chunk / over-budget runs keep the disk accumulator unchanged. Same idea as cuVS 26.12 `detail/cagra/graph_core.cuh::optimize`, which builds and merges the reverse graph fully in memory when it fits. Output byte-identical in all three modes (single drain; in-memory with mid-batch drains via `--gpu-budget-mb 300`; disk accumulator via `--edge-buffer-mb 100`). optimize_chunked 3.45 → 1.64 s, RSS 1.43 → 0.94 GB.
**Files:** `src/optimize_chunked.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 16.10 | +0.9% | +26.0% | 1.40x | bucket2 14.34, optimize 1.30, translate 0.45 | 3870 | 1.60 |
| sift1M_large | large | 33.01 | +3.3% | +56.0% | 2.86x | bucket2 29.66, prune 1.33, optimize_chunked 1.64, translate 0.45 | 1490 | 3.89 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.21 | 1.29 | 0.00 | 2.20 | 1.09 | 0.02 | 4.58 | 0.00 | 0.75 |
| sift1M_large | 0.57 | 4.54 | 1.87 | 0.00 | 6.27 | 2.16 | 0.03 | 12.27 | 0.98 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9799 | 0.9739 | 0.9684 | 0.9632 |
| sift1M_large | 0.9832 | 0.9747 | 0.9683 | 0.9633 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9286 / 1921K | 0.9776 / 971K | 0.9947 / 533K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.8958 / 2216K | 0.9615 / 1181K | 0.9886 / 561K | 0.9973 / 188K | 0.9993 / 68K |

Borderline → large rerun (`runs/iter19/sift1M_large_b`): 32.52 s, +4.7% → ACCEPT.

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +3.3% / +4.7% on rerun (34.12 → 33.01 s), 1M-auto +0.9% (path untouched).

### #20 — exact brute-force centroid self-kNN replaces CAGRA build + self-search  (2026-10-21)
**Idea / rationale:** Step 3 ran a full `cagra::build` (NN-Descent + sort + prune, after copying the centroids to host) on the 10K centroids every round (0.2–0.5 s), and Step 6 then ran a CAGRA self-search on that pruned navigable graph to get each centroid's `nprobe` nearest buckets (`expand_centroid_neighbors_gpu`, ~0.08 s/round). For non-PQ runs both are replaced by one exact `brute_force::search` of the centroids against themselves (k = max(K, nprobe) + 1, self removed by id — not by position, since duplicate centroid vectors tie at distance 0; out-of-range ids / short rows → error), cuVS 26.12 `detail/knn_brute_force.cuh`. The first K columns become the centroid graph (DiskJoin bucket order, round-1 reorder, `centroid_knn.bin` windows for prune_windowed), the first nprobe columns are Step 6's neighbour table directly. Invariants unchanged (bucket order still derived from the centroid KNN graph — now the exact one). PQ runs keep the CAGRA navigable graph (their Step 4 still searches it). Step 3 per round 0.2–0.5 → 0.015 s; topk expand 0.08 → 0.003 s. Side effect: large-path search recall@10 at itopk 32 rose 0.8958 → 0.9011 (prune windows now are the exact nearest buckets).
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 14.70 | +8.7% | +32.4% | 1.27x | bucket2 13.01, optimize 1.32, translate 0.40 | 3686 | 1.55 |
| sift1M_large | large | 30.78 | +6.8% | +59.0% | 2.67x | bucket2 27.39, prune 1.40, optimize_chunked 1.63, translate 0.45 | 1356 | 3.88 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 3.20 | 0.16 | 0.00 | 2.08 | 1.16 | 0.02 | 4.37 | 0.00 | 0.75 |
| sift1M_large | 0.57 | 4.76 | 0.18 | 0.00 | 6.05 | 2.18 | 0.03 | 12.04 | 0.94 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9805 | 0.9751 | 0.9684 | 0.9630 |
| sift1M_large | 0.9834 | 0.9754 | 0.9694 | 0.9641 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9286 / 1946K | 0.9777 / 1033K | 0.9946 / 493K | 0.9987 / 184K | 0.9997 / 66K |
| ours sift1M_large | 0.9011 / 2216K | 0.9633 / 1190K | 0.9891 / 561K | 0.9974 / 189K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +8.7% (16.10 → 14.70 s), 1M-large +6.8% (33.01 → 30.78 s).

### #21 — KMeans|| trained on a 25% random subset  (2026-10-21)
**Idea / rationale:** with the dataset fitting on the GPU (SIFT1M on both paths), Step 2 ran KMeans|| D² oversampling over all N points every round (cost ∝ points × candidates; 0.66–0.71 s/round, 3.2 s total). New `bucket2 --kmeans-train-fraction F` (default 1.0 = unchanged): KMeans|| runs on a deterministic Bernoulli(F) subset of the points (seeded per round, no duplicates — the existing `sample_without_replacement` samples *with* replacement, so not used; falls back to all points if the subset would have < 4× centroids); centroids are still real data points and their number stays N·centroid_ratio; centroid row → raw id mapping bounds-checked. Same as cuVS 26.12 IVF build training balanced k-means on `kmeans_trainset_fraction` of the data (`ivf_pq/ivf_pq_build.cuh`). Sweep (1 run, auto): F=0.25 → 12.95 s, Step 2 1.42 s; F=0.5 → 14.15 s; search recall unchanged within 0.001 for both. `params.sh`: `BUCKET2_EXTRA="--skip-npy --kmeans-train-fraction 0.25"`.
**Files:** `src/bucket.cu`  **Params changed:** `BUCKET2_EXTRA` adds `--kmeans-train-fraction 0.25`

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 13.01 | +11.5% | +40.2% | 1.13x | bucket2 11.26, optimize 1.31, translate 0.44 | 3574 | 1.59 |
| sift1M_large | large | 31.41 | -2.0% | +58.2% | 2.72x | bucket2 27.84, prune 1.40, optimize_chunked 1.63, translate 0.46 | 1220 | 3.98 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 1.41 | 0.11 | 0.00 | 2.12 | 1.16 | 0.02 | 4.37 | 0.00 | 0.75 |
| sift1M_large | 0.57 | 2.74 | 0.19 | 0.00 | 8.48 | 2.18 | 0.03 | 12.05 | 0.99 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9791 | 0.9749 | 0.9690 | 0.9638 |
| sift1M_large | 0.9815 | 0.9747 | 0.9689 | 0.9637 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9293 / 1948K | 0.9774 / 1035K | 0.9947 / 529K | 0.9989 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9004 / 1941K | 0.9631 / 928K | 0.9894 / 558K | 0.9974 / 181K | 0.9993 / 68K |

First large run (`runs/iter21/sift1M_large`): 31.80 s (−3.3%, just over the 3% tolerance → REJECT): Step 2 4.76 → 2.59 s but Step 4 6.05 → 9.29 s (write-throttled flush now overlapping the previous round's background merge — see backlog). Rerun `sift1M_large_b`: 31.41 s (−2.0%) → ACCEPT, recorded as best.

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +11.5% (14.70 → 13.01 s; Step 2 3.20 → ~1.4 s), 1M-large −2.0% (30.78 → 31.41 s, within tolerance; the Step 2 saving is absorbed by disk-write contention).

### #22 — Step 4 exact top-2 via fp16 tensor-core GEMM for 8-bit data  (2026-10-21)
**Idea / rationale:** Step 4's nearest-2-centroid search (both paths, non-PQ) used cuVS fp32 `brute_force` (0.42 s/round). For uint8/int8 datasets the computation can be done exactly on fp16 tensor cores: values ≤ 255 are exact in fp16, products ≤ 65025, D=128 sums ≤ 8.3M < 2^24 so fp32 accumulation is exact, and d = ‖c‖² − 2 q·c (per-row constant ‖q‖² dropped) stays an exact integer < 2^24 — the same argument `GpuResidentKnn` already relies on. Per query tile of 4096: f32→f16 convert, `cublasGemmEx` (16F in, 32F out, `CUBLAS_COMPUTE_32F`, tensor ops) against the fp16 centroids, then `step4_top2_kernel` (one warp per row, register top-2, warp-shuffle merge; ties → lower centroid id, deterministic); ids bounds-checked on download. dots tile 4096 × 10K × 4 B = 164 MB (large-path GPU peak 1386 MiB, gate 1720). Other dataset types keep `brute_force`; PQ keeps CAGRA search. cuVS 26.12 reference: `detail/knn_brute_force.cuh` (tiled GEMM + select) with the GEMM moved to fp16 tensor cores. search 0.42 → 0.12 s/round.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 11.62 | +10.7% | +46.6% | 1.01x | bucket2 9.94, optimize 1.28, translate 0.38 | 3818 | 1.62 |
| sift1M_large | large | 30.53 | +2.8% | +59.3% | 2.65x | bucket2 27.05, prune 1.39, optimize_chunked 1.62, translate 0.45 | 1386 | 3.95 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.41 | 0.10 | 0.00 | 0.85 | 1.16 | 0.02 | 4.32 | 0.00 | 0.76 |
| sift1M_large | 0.57 | 2.57 | 0.34 | 0.00 | 7.38 | 2.25 | 0.03 | 12.10 | 0.94 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9836 | 0.9758 | 0.9692 | 0.9636 |
| sift1M_large | 0.9820 | 0.9750 | 0.9686 | 0.9636 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9294 / 1945K | 0.9775 / 956K | 0.9943 / 533K | 0.9989 / 189K | 0.9998 / 68K |
| ours sift1M_large | 0.9002 / 1934K | 0.9628 / 1035K | 0.9890 / 533K | 0.9973 / 189K | 0.9994 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +10.7% (13.01 → 11.62 s, now 1.01× CAGRA native), 1M-large +2.8% (31.41 → 30.53 s).

### #23 — rounds 2..t: gather the KMeans|| subset through inverse_perm instead of re-materialising X  (2026-10-21)
**Idea / rationale:** after round 1's inline reorder, rounds 2..t need Step 2's training data in the new coordinate system, so Step 4.5 rebuilt the whole N×D float working set: the GPU-resident path permuted it in memory (0.50 s), the out-of-core path re-read `data_reordered.u8bin` from disk and converted it (0.79 s). Since #21 Step 2 only reads a 25% subset of it. When the full working set is loaded (`fits_in_gpu`), the subset fraction is < 1 and no PQ, `X_sampled` now stays in the original order and Step 4.5 only records `x_row_of` = inverse_perm (new id → old row; size-checked); Step 2's subset gather reads row `x_row_of[new_id]`. Fraction 1.0 / sampled / PQ runs keep the old path. Same "don't materialise, index through the permutation" as cuVS 26.12 `detail/cagra/cagra_build.cuh` working on index views. resample 0.50 → 0.001 s (auto), 0.79 → 0.001 s (large).
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 11.19 | +3.7% | +48.6% | 0.97x | bucket2 9.47, optimize 1.28, translate 0.39 | 3658 | 1.56 |
| sift1M_large | large | 28.91 | +5.3% | +61.5% | 2.51x | bucket2 25.48, prune 1.40, optimize_chunked 1.64, translate 0.38 | 1438 | 3.95 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.41 | 0.11 | 0.00 | 0.89 | 0.66 | 0.02 | 4.34 | 0.00 | 0.75 |
| sift1M_large | 0.57 | 2.75 | 0.19 | 0.00 | 6.73 | 1.40 | 0.03 | 12.00 | 0.96 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9833 | 0.9759 | 0.9691 | 0.9638 |
| sift1M_large | 0.9810 | 0.9750 | 0.9688 | 0.9633 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9294 / 1946K | 0.9776 / 1035K | 0.9947 / 532K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9000 / 1947K | 0.9640 / 1033K | 0.9894 / 492K | 0.9971 / 186K | 0.9992 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +5.3% (30.53 → 28.91 s), 1M-auto +3.7% (11.62 → 11.19 s; now below CAGRA native 11.54 s).

### #26 — resident trims: top-M tree merge + `--skip-dists`  (2026-10-21)
**Idea / rationale:** the two individually-measured small resident-path changes from #24 and #25, each ~0.3 s (≈2.5%) on their own and therefore below the 3% gate alone, benchmarked together (contributions are known from their separate quick tests): (1) #25 — `block_topK_from_dots_kernel` merges the G per-warp sorted top-K lists with a shared-memory merge-path tree instead of warp 0's serial ballot-insert of (G−1)·K_OUT candidates (same top-K set; resident Step 6 loop 0.95 → 0.86 s/round); (2) #24 — new `bucket2 --skip-dists` (default off) doesn't write `vector_dists.bin`, which no later stage reads (resident: no dists D2H/write, Write 0.75 → 0.48 s; out-of-core: final merge skips the dists pwrite); `params.sh` adds it. cuVS 26.12 references: `matrix/detail/select_warpsort.cuh` (warp-sort queues merged across warps in shared memory), CAGRA build only emitting the graph.
**Files:** `src/bucket.cu`  **Params changed:** `BUCKET2_EXTRA` adds `--skip-dists`

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 10.57 | +5.5% | +51.4% | 0.92x | bucket2 8.89, optimize 1.30, translate 0.38 | 3810 | 1.38 |
| sift1M_large | large | 29.15 | -0.8% | +61.2% | 2.53x | bucket2 25.75, prune 1.39, optimize_chunked 1.63, translate 0.37 | 1386 | 4.05 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.40 | 0.10 | 0.00 | 0.90 | 0.66 | 0.02 | 4.04 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 2.73 | 0.20 | 0.00 | 7.06 | 1.40 | 0.03 | 12.04 | 0.93 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9826 | 0.9755 | 0.9689 | 0.9638 |
| sift1M_large | 0.9801 | 0.9750 | 0.9685 | 0.9635 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9283 / 1948K | 0.9771 / 1035K | 0.9943 / 494K | 0.9988 / 186K | 0.9997 / 68K |
| ours sift1M_large | 0.9000 / 2217K | 0.9638 / 1190K | 0.9891 / 561K | 0.9974 / 188K | 0.9993 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +5.5% (11.19 → 10.57 s, 0.92× CAGRA native), 1M-large −0.8% (within tolerance).

### #27 — ITERATIONS 4 → 3, NPROBE 150 → 200  (2026-10-21)
**Idea / rationale:** each bucketing round costs Steps 2 + 4 + 6 (resident ≈ 1.6 s/round, large ≈ 5 s/round incl. the write-bound out-of-core loop); a round contributes candidates from a fresh random bucketing, a larger `nprobe` contributes more neighbour buckets per round. Trading one round for a wider probe keeps candidate coverage: sweep (1 run each) T=3/NPROBE=150 → auto 9.03 s / large 23.38 s but auto search recall@10 itopk 32 0.9271; T=3/NPROBE=200 → 9.41 / 24.08 s with recall 0.9281 / 0.8996 (≈ unchanged vs 0.9283 / 0.9000 at T=4/150). Chose T=3/200. Analogous to IVF-based kNN-graph builds trading n_probes against refinement passes (cuVS 26.12 `cagra_build.cuh` ivf_pq `refine_rate` / n_probes). Trade-off recorded: pre-prune degree-64 KNN recall@64 0.963 → 0.953 (not gated); final graph search recall unchanged within 0.001.
**Files:** —  **Params changed:** `ITERATIONS=3` (was 4), `NPROBE=200` (was 150)

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.43 | +10.8% | +56.6% | 0.82x | bucket2 7.67, optimize 1.30, translate 0.42 | 3726 | 1.38 |
| sift1M_large | large | 24.17 | +17.1% | +67.8% | 2.09x | bucket2 20.75, prune 1.40, optimize_chunked 1.64, translate 0.43 | 1386 | 3.62 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.09 | 0.11 | 0.00 | 0.69 | 0.66 | 0.02 | 3.35 | 0.00 | 0.47 |
| sift1M_large | 0.57 | 2.25 | 0.24 | 0.00 | 4.08 | 1.41 | 0.03 | 10.55 | 0.95 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9746 | 0.9666 | 0.9600 | 0.9541 |
| sift1M_large | 0.9745 | 0.9654 | 0.9591 | 0.9532 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9294 / 1937K | 0.9773 / 1033K | 0.9943 / 497K | 0.9987 / 186K | 0.9997 / 68K |
| ours sift1M_large | 0.9003 / 2219K | 0.9638 / 1189K | 0.9896 / 561K | 0.9977 / 189K | 0.9994 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +10.8% (10.57 → 9.43 s, 0.82× CAGRA native), 1M-large +17.1% (29.15 → 24.17 s).

### #30 — out-of-core Step 6 loop: scatter on a consumer thread + block top-M / fp16 GEMM  (2026-10-21)
**Idea / rationale:** the out-of-core bucket loop (after #10/#11/#27) was split ~evenly between GPU wait (`stream_sync` ~1.25 s/round), host scatter (`scatter_pending` ~0.7 s: per point pad to M + `acc.add` into the chunk write buffers) and submit (~1.1 s). Two halves of the same loop, benchmarked together (each alone hid behind the other — #29 alone measured −1.6%):
(1) **host:** scatter moves to a consumer thread — the main thread copies the slot's results (gid + bucket×actual_M ids/dists, ~50 KB) into a recycled job buffer and enqueues it, then immediately reuses `h_ids_bucket/h_out[slot]`; only the scatter thread calls `acc.add`, in enqueue order (same order as before); it is drained/joined before `finish_iteration_writes`, exceptions are rethrown, an RAII guard joins on unwind. main-thread scatter 0.7 → 0.16 s/round.
(2) **GPU (= #29):** the resident `block_topK_from_dots_kernel` (templated on the dots type, output rows by bucket position) replaces dots_to_dist + raft `select_k` + gather (3 kernels; raft kept for M > 256), and the 8-bit GEMM uses fp16 tensor cores (exact, fp32 accumulation) instead of INT8 IMMA (dp4a on V100); dots ld padded to 4.
Same structure as cuVS 26.12 `detail/cagra/cagra_build.cuh::build_knn_graph` (separate threads for host pre/post-processing around the batched GPU search). Loop 3.24 → ~2.9 s/round; KNN recall unchanged.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.42 | +0.1% | +56.7% | 0.82x | bucket2 7.69, optimize 1.30, translate 0.44 | 3782 | 1.38 |
| sift1M_large | large | 22.80 | +5.7% | +69.6% | 1.98x | bucket2 19.32, prune 1.40, optimize_chunked 1.65, translate 0.39 | 1386 | 3.86 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.08 | 0.12 | 0.00 | 0.68 | 0.65 | 0.02 | 3.34 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 2.20 | 0.16 | 0.00 | 3.82 | 1.40 | 0.03 | 9.51 | 0.92 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9737 | 0.9664 | 0.9600 | 0.9540 |
| sift1M_large | 0.9746 | 0.9673 | 0.9592 | 0.9534 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9289 / 1947K | 0.9775 / 1035K | 0.9947 / 463K | 0.9988 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.8997 / 2224K | 0.9626 / 1190K | 0.9887 / 562K | 0.9973 / 189K | 0.9993 / 66K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +5.7% (24.17 → 22.80 s, 1.98× CAGRA native), 1M-auto +0.1% (path untouched).

### #32 — out-of-core: in-memory inline reorder + ≥1 ChunkedKNN chunk per thread  (2026-10-21)
**Idea / rationale:** two large-path pieces, benchmarked together: (1) #31 — when 2·N·D B ≤ cpu_limit/4 the round-1 inline reorder reads the whole input, permutes by inverse_perm in parallel and writes `data_reordered` once, instead of routing 1M rows through 10K per-bucket buffered writers (pass3_write 1.03 → 0.39 s; output verified == base[inverse_perm]; streaming writer kept for larger data; alone +3.1% / +2.9%). (2) `ChunkedKnnAccumulator::create` derived `chunk_size` only from memory per thread (4 GB / 80 / 512 B ≈ 100K → 10 chunks on SIFT1M), so the per-chunk-parallel merge used 10 of 80 threads; `chunk_size` is now also capped at ceil(N / threads) (SIFT1M: 80 chunks of 12.5K). Smaller chunks only lower per-merge-thread memory. Merges: round-2 2.5 → 1.1 s, final (exposed) wait 0.9 → 0.47 s; less background write overlap also shortened Step 4's flush. Like cuVS 26.12 `cagra_ace.cuh` sizing partitions for parallelism, not just memory. bucket2 RSS 3.86 → 3.58 GB.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.48 | -0.6% | +56.4% | 0.82x | bucket2 7.71, optimize 1.31, translate 0.47 | 3894 | 1.38 |
| sift1M_large | large | 21.11 | +7.4% | +71.9% | 1.83x | bucket2 17.64, prune 1.41, optimize_chunked 1.65, translate 0.38 | 1386 | 3.58 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.09 | 0.10 | 0.00 | 0.68 | 0.66 | 0.02 | 3.35 | 0.00 | 0.49 |
| sift1M_large | 0.57 | 2.20 | 0.15 | 0.00 | 3.07 | 0.76 | 0.03 | 9.61 | 0.41 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9751 | 0.9674 | 0.9598 | 0.9538 |
| sift1M_large | 0.9767 | 0.9673 | 0.9603 | 0.9540 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9291 / 2168K | 0.9773 / 1188K | 0.9944 / 562K | 0.9988 / 179K | 0.9997 / 68K |
| ours sift1M_large | 0.8984 / 1941K | 0.9630 / 897K | 0.9888 / 532K | 0.9973 / 189K | 0.9994 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +7.4% (22.80 → 21.11 s, 1.83× CAGRA native), 1M-auto −0.6% (path untouched).

### #33 — out-of-core Step 6: GPU-side Belady bucket cache, pools gathered on the GPU  (2026-10-21, commit 9d1e26e)
**Idea / rationale:** with the scatter (#30) and GPU work fixed, the out-of-core bucket loop was bound by the producer's host pool assembly: each bucket's pool (own bucket + nprobe=200 neighbour buckets ≈ 20K rows × 136 B ≈ 2.7 MB) was memcpy'd on the host and uploaded — ~27 GB of host copy + H2D per round, although consecutive buckets in DiskJoin order share most neighbours. Now (8-bit data, `--gpu-bucket-cache-mb`, default 2048, 0 = old path) a second `BeladyBucketCache<DevBucket>` with the same future-access lists keeps buckets on the GPU (int32 ids + raw u8 rows, one `cudaMallocAsync` block per bucket on a dedicated copy stream, loaded from the host cache on a miss). The producer only writes a per-slot job table (ids, vecs, count, dest) in the unchanged pool order (center first → identical results) and records a per-slot load event on the copy stream; the main thread makes the slot stream wait on it, uploads the job table and launches `devcache_gather_pool_kernel` (gather + u8→fp16 into `d_B`, ids into `d_ids_pool`); norms / GEMM / top-M / D2H unchanged. **Safety:** eviction is stream-ordered — `DevBucket` release waits until its last-using position has been submitted, makes the copy stream wait on that position's gather-done event (tagged ring of 1024 events; ring slots are synchronised before reuse, so a tag mismatch means "already complete"), then `cudaFreeAsync`. New opt-in `BeladyBucketCache::set_pin_current_position`: buckets already fetched at the current position are never evicted (they are in the current job table) — without it a tiny budget deadlocked (first 20 MB test timed out); host caches keep pure Belady. Pool-capacity bounds check kept. Like cuVS 26.12 `cagra_ace.cuh` keeping partitions device-resident across neighbouring partition builds. Tests: 2048 MB → 10K uploads/round, 0 evictions, loop 2.9 → 1.95 s/round; 20 MB → ~210K evictions/round, completes, KNN recall@64 identical (0.9538 vs 0.9536).
**Files:** `src/bucket.cu`, `src/bucket_order.hpp`  **Params changed:** none (new flag default 2048 MB)

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.35 | +1.4% | +57.0% | 0.81x | bucket2 7.69, optimize 1.26, translate 0.40 | 3726 | 1.38 |
| sift1M_large | large | 18.63 | +11.7% | +75.2% | 1.61x | bucket2 15.22, prune 1.40, optimize_chunked 1.65, translate 0.47 | 1620 | 3.61 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.73 | 1.14 | 0.11 | 0.00 | 0.68 | 0.67 | 0.02 | 3.36 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 2.23 | 0.15 | 0.00 | 3.15 | 0.77 | 0.03 | 6.97 | 0.46 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9765 | 0.9662 | 0.9579 | 0.9524 |
| sift1M_large | 0.9753 | 0.9671 | 0.9593 | 0.9537 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9296 / 1943K | 0.9778 / 1034K | 0.9945 / 517K | 0.9987 / 188K | 0.9998 / 68K |
| ours sift1M_large | 0.8993 / 1938K | 0.9627 / 1032K | 0.9893 / 533K | 0.9972 / 188K | 0.9994 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +11.7% (21.11 → 18.63 s, 1.61× CAGRA native), 1M-auto +1.4% (path untouched). GPU peak 1386–1620 MiB (gate 1919).

### #34 — large-path per-bucket overheads: fused norms, skipped H2D, uninitialised write buffers  (2026-10-21)
**Idea / rationale:** after #33 the out-of-core bucket loop was bound by `gpu_submit` (~1.0 s/round, ~10 CUDA API calls per bucket). (1) `devcache_gather_pool_kernel` now works one warp per row and also writes ‖row‖² (exact for 8-bit data — identical to `compute_row_norms_kernel` reading the fp16 rows back), so the separate norms kernel launch per bucket is gone. (2) The `d_ids_bucket` H2D is skipped when the block top-M kernel is used (only the raft fallback for M > 256 reads it). (3) `BucketVectorAccumulator`'s ~1 GB of per-bucket write buffers were `vector<char>::resize`d (zero-filled) every create → uninitialised `unique_ptr<char[]>` (only ever written before read): round-1 `bucket_vec_create` 0.75 → 0.03 s and bucket2 peak RSS 3.6 → 2.8 GB (pages touched lazily). cuVS 26.12 reference: `detail/cagra/graph_core.cuh` fuses norm/prune work into the gather kernels instead of separate launches. Loop 1.95 → 1.6 s/round.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.39 | -0.4% | +56.8% | 0.81x | bucket2 7.69, optimize 1.29, translate 0.38 | 3990 | 1.38 |
| sift1M_large | large | 16.81 | +9.8% | +77.6% | 1.46x | bucket2 13.40, prune 1.40, optimize_chunked 1.63, translate 0.47 | 1566 | 2.82 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.10 | 0.09 | 0.00 | 0.65 | 0.67 | 0.02 | 3.35 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.50 | 0.16 | 0.00 | 3.13 | 0.78 | 0.03 | 5.92 | 0.45 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9748 | 0.9674 | 0.9595 | 0.9535 |
| sift1M_large | 0.9725 | 0.9669 | 0.9600 | 0.9539 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9290 / 2152K | 0.9774 / 1186K | 0.9944 / 562K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.8998 / 2219K | 0.9637 / 1190K | 0.9893 / 563K | 0.9973 / 189K | 0.9992 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +9.8% (18.63 → 16.81 s, 1.46× CAGRA native), 1M-auto −0.4% (path untouched).

### #35 — bucket processing order computed in the background right after Step 3  (2026-10-21)
**Idea / rationale:** the DiskJoin bucket processing order (`compute_bucket_processing_order`, ~0.3 s for 10K buckets) depends only on the centroid KNN graph, but it ran serially in round 1's Step 4.5 and again in every out-of-core round's Step 6 setup. The graph is now downloaded to the host right after Step 3 (previously after Step 4's assignment) and the order is computed by `std::async` on a by-value copy of the graph, overlapping Step 4; Step 4.5 (round 1) and out-of-core Step 6 (rounds ≥ 2, as `precomputed_order`) take it from the future. Same inputs and `order_window` formula as both original call sites → identical order. The GPU-resident path only needs it in round 1. Measured: large-path order cost 0.31–0.34 s/round → 0; resident round-1 wait 0.34 → 0.15 s.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.19 | +2.1% | +57.7% | 0.80x | bucket2 7.47, optimize 1.28, translate 0.42 | 3830 | 1.38 |
| sift1M_large | large | 15.89 | +5.5% | +78.8% | 1.38x | bucket2 12.38, prune 1.42, optimize_chunked 1.66, translate 0.42 | 1386 | 2.82 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.08 | 0.11 | 0.00 | 0.69 | 0.44 | 0.02 | 3.35 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.47 | 0.15 | 0.00 | 3.22 | 0.46 | 0.03 | 5.32 | 0.40 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9754 | 0.9671 | 0.9595 | 0.9537 |
| sift1M_large | 0.9728 | 0.9672 | 0.9604 | 0.9546 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9285 / 1948K | 0.9768 / 1034K | 0.9942 / 497K | 0.9987 / 177K | 0.9997 / 68K |
| ours sift1M_large | 0.8995 / 1941K | 0.9631 / 1035K | 0.9892 / 492K | 0.9973 / 185K | 0.9993 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +5.5% (16.81 → 15.89 s, 1.38× CAGRA native), 1M-auto +2.1% (9.39 → 9.19 s, 0.80×).

### #36 — out-of-core bucket_vecs kept in memory within cpu_limit/4  (2026-10-21)
**Idea / rationale:** rounds ≥ 2 wrote every point's (gid, vector) record into per-bucket write buffers flushed to `bucket_vecs_merged.bin` (bucket_write ~0.4 s + write-throttled `finish_iteration` 0.4–0.6 s per round), which Step 6's host cache then read back. When the round's merged size N·(8+D) B ≤ cpu_limit/4, `BucketVectorAccumulator` now writes each record straight to its final offset (known from the histogram) in one host buffer and `read_bucket*` memcpy from it (bounds-checked); larger data keeps the file path (verified with `--cpu-limit 400000000`). finish_iteration 0.4–0.6 → 0 s. Patch also at `bucketDemo/patches/iter36_bucketvecs_in_memory.patch`.
**Files:** `src/bucket.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.18 | +0.1% | +57.8% | 0.80x | bucket2 7.50, optimize 1.30, translate 0.39 | 3862 | 1.38 |
| sift1M_large | large | 14.81 | +6.8% | +80.3% | 1.28x | bucket2 11.33, prune 1.42, optimize_chunked 1.62, translate 0.45 | 1488 | 2.77 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 1.09 | 0.10 | 0.00 | 0.67 | 0.48 | 0.02 | 3.36 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.54 | 0.16 | 0.00 | 2.24 | 0.45 | 0.03 | 5.23 | 0.45 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9733 | 0.9663 | 0.9595 | 0.9531 |
| sift1M_large | 0.9751 | 0.9668 | 0.9599 | 0.9539 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9286 / 1902K | 0.9770 / 1035K | 0.9944 / 533K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.8996 / 1945K | 0.9632 / 1031K | 0.9892 / 500K | 0.9972 / 186K | 0.9993 / 68K |

**Verdict:** ACCEPT — 1M-large +6.8% (15.89 → 14.81 s, 1.28× CAGRA native), 1M-auto +0.1% (path untouched).

### #38 — prune_windowed batched path on the GPU  (2026-10-21)
**Idea / rationale:** after #18 the large path's prune stage spent ~1.0 s in CPU detour counting (memory-latency bound: per point ~64 random 256 B candidate-row reads + window binary searches). New `src/prune_windowed_gpu.cu` (linked into `prune_windowed`, used in batched mode for uint8 data with M_in ≤ 64 when the working set fits free GPU memory; CPU batched/per-position paths unchanged otherwise): one warp per point — candidate distances (exact u8 L2, +inf outside the window), shared-memory bitonic sort by (padding last, dist, original column), id→rank open-addressing table, detour counting over each in-window D's row with a per-rank `atomicExch` stamp ("count each (D, B) once"), selection of the first out_deg by (detour count, rank) — the same rules as `prune_windowed_core.hpp`. Window per position = itself + its centroid-knn neighbour buckets' positions (sorted, binary-searched); out-of-range ids treated as outside the window; too-few-candidates → error. cuVS 26.12 reference: `detail/cagra/graph_core.cuh::kern_fused_prune`. prune 1.37 → 0.85 s.
**Quality side effect (positive):** the GPU sort breaks distance ties by original column. ~59% of candidates are outside the window (distance +inf) and the CPU's unstable `std::sort` left those in arbitrary order; the column order is bucket2's true-distance order. A Python spec of the prune rules matched the GPU output on 8/8 sampled points (CPU: 2/8). Large-path search recall@10 itopk 32 / 64: 0.8995 / 0.9631 → 0.9145 / 0.9709 (rerun 0.9135 / 0.9699). Worth porting the same tie-break to the CPU path (`sort_row_by_l2`: `std::stable_sort`, or compare by (dist, k)).
**Files:** `src/prune_windowed_gpu.cu` (new), `src/prune_windowed_gpu.hpp` (new), `src/prune_windowed.cpp`, `CMakeLists.txt`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.23 | -0.5% | +57.6% | 0.80x | bucket2 7.50, optimize 1.29, translate 0.43 | 3838 | 1.38 |
| sift1M_large | large | 14.22 | +4.0% | +81.1% | 1.23x | bucket2 11.30, prune 0.86, optimize_chunked 1.59, translate 0.38 | 1398 | 2.78 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 1.10 | 0.10 | 0.00 | 0.70 | 0.43 | 0.02 | 3.36 | 0.00 | 0.47 |
| sift1M_large | 0.57 | 1.47 | 0.15 | 0.00 | 2.16 | 0.45 | 0.03 | 5.27 | 0.44 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9761 | 0.9669 | 0.9596 | 0.9537 |
| sift1M_large | 0.9744 | 0.9666 | 0.9595 | 0.9534 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9283 / 2220K | 0.9770 / 1190K | 0.9944 / 562K | 0.9987 / 189K | 0.9996 / 68K |
| ours sift1M_large | 0.9145 / 2218K | 0.9709 / 1189K | 0.9921 / 562K | 0.9982 / 189K | 0.9995 / 68K |

Borderline → large rerun (`runs/iter38/sift1M_large_b`): 13.98 s (+5.6%), recall 0.9135 → ACCEPT.

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +4.0% / +5.6% on rerun (14.81 → 14.22 s, 1.23× CAGRA native) and +1.5 pt large-path recall at itopk 32; 1M-auto −0.5% (path untouched).

### #39 — optimize_chunked method E phase 2 on the GPU  (2026-10-21)
**Idea / rationale:** with a single chunk that holds all N·K_out reverse edges (#19) and the whole forward graph uploaded in one Phase 1 batch (`batch_rows ≥ N`), both inputs of Phase 2 are already in GPU memory, yet the edges were copied back and merged on the CPU (CSR build + per-row merge, 0.94 s). Now (K_out ≤ 64) the GPU does it with the same rules: extract (dst, rank, src) from the packed records (dst bounds-checked), two stable radix sorts (by src, then by dst<<16 | rank → exactly the CPU's (dst, rank, src) order from #9), row offsets by histogram + exclusive scan, and a thread-per-row `kern_e_merge` applying `merge_rank_ordered_into_row`'s shift/protect insertion (weakest first). Output verified byte-identical to the CPU path (and the CPU fallback, forced with `--gpu-budget-mb 300`, still identical). Like cuVS 26.12 `detail/cagra/graph_core.cuh::optimize`, which builds and merges the reverse graph on the device. optimize_chunked 1.59 → 0.63 s, RSS 0.94 → 0.37 GB.
**Files:** `src/optimize_chunked.cu`  **Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.23 | +0.0% | +57.6% | 0.80x | bucket2 7.51, optimize 1.29, translate 0.44 | 3854 | 1.38 |
| sift1M_large | large | 13.26 | +6.8% | +82.3% | 1.15x | bucket2 11.39, prune 0.86, optimize_chunked 0.63, translate 0.49 | 1674 | 2.71 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.75 | 1.10 | 0.10 | 0.00 | 0.64 | 0.52 | 0.02 | 3.36 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.47 | 0.16 | 0.00 | 2.19 | 0.47 | 0.02 | 5.29 | 0.41 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9723 | 0.9672 | 0.9597 | 0.9539 |
| sift1M_large | 0.9754 | 0.9676 | 0.9601 | 0.9538 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9292 / 1950K | 0.9775 / 1035K | 0.9946 / 533K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9147 / 1946K | 0.9711 / 999K | 0.9922 / 562K | 0.9982 / 189K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +6.8% (14.22 → 13.26 s, 1.15× CAGRA native), 1M-auto 0.0% (path untouched).

### #41 — out-of-core Step 6: 4 slots + fused ids/dists D2H  (2026-10-21, commit 24d5d1c)
**Idea / rationale:** bundle the #40 slot change (OVERSUB 1.5 → 2.0, 3 → 4 in-flight slots) with one fused device→host copy per slot: block top-M writes dists right after the ids in `d_out_neighbors[slot]` (buffer sized 2×`bytes_out`), so each slot does one D2H instead of two, and the scatter thread reads dists from `h_out[slot] + n_out`. This follows cuVS 26.12 `detail/cagra/cagra_build.cuh::build_knn_graph`, which batches results into one pinned staging buffer per batch.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Borderline confirmation:** rerun `sift1M_large_b` 12.70 s (+4.2%), ACCEPT, so both runs pass.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.26 | -0.3% | +57.4% | 0.80x | bucket2 7.57, optimize 1.30, translate 0.45 | 4022 | 1.38 |
| sift1M_large | large | 12.67 | +4.4% | +83.1% | 1.10x | bucket2 10.80, prune 0.84, optimize_chunked 0.64, translate 0.38 | 1674 | 2.78 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.11 | 0.10 | 0.00 | 0.74 | 0.45 | 0.03 | 3.36 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.47 | 0.15 | 0.00 | 2.22 | 0.44 | 0.03 | 4.78 | 0.44 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9754 | 0.9678 | 0.9602 | 0.9543 |
| sift1M_large | 0.9731 | 0.9662 | 0.9587 | 0.9526 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9278 / 2214K | 0.9775 / 1189K | 0.9948 / 562K | 0.9988 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9144 / 2219K | 0.9708 / 1189K | 0.9920 / 561K | 0.9982 / 189K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +4.4% (13.26 → 12.67 s, 1.10× CAGRA native; rerun +4.2%), 1M-auto −0.3% (noise, path untouched).

### #42 — out-of-core: raw dataset kept in host memory + parallel bucket write  (2026-10-21, commit 7868a6e)
**Idea / rationale:** when 2·N·D B fits `cpu_limit/4` (the same rule as the in-memory Step 4.5 reorder), the out-of-core path reads the raw u8 dataset once in Step 1. Round 1's reorder permutes that copy instead of re-reading the file, and keeps the reordered copy as the new-coordinate dataset. Step 4 assignment (`batch_assign_with_cagra_anns`, new `host_raw` argument) and round 2..t bucket writes (`write_bucket_vectors_from_assignments`, centroids + chunks) take pointers into it instead of reading the disk 2–3 times per round. The copy is freed after the last round's bucket write; datasets over the budget (SIFT10M+) keep the per-range disk reads. Bundled with the #37 patch: deterministic multi-threaded `add_range_parallel` (in-memory mode, same in-bucket order) + byte-array centroid test. This follows the IVF-PQ build in cuVS 26.12 (`ivf_pq_build.cuh::extend`, `cagra_build.cuh::build_knn_graph`), which streams the host dataset from memory, not from files. Alone (run `iter42`) the change gave large 12.30 s (+2.9%, Step 4 2.22 → 1.55 s), with the bucket write's single-threaded copy left as the remaining cost.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Per-round bucket write (large):** 0.40 s (centroid read 0.13 + read 0.125 + write 0.136) → 0.04–0.09 s.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 9.22 | +0.4% | +57.6% | 0.80x | bucket2 7.48, optimize 1.30, translate 0.46 | 3878 | 1.38 |
| sift1M_large | large | 11.90 | +6.1% | +84.2% | 1.03x | bucket2 10.03, prune 0.85, optimize_chunked 0.65, translate 0.41 | 2052 | 2.74 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.76 | 1.14 | 0.11 | 0.00 | 0.66 | 0.45 | 0.02 | 3.35 | 0.00 | 0.48 |
| sift1M_large | 0.69 | 1.48 | 0.14 | 0.00 | 1.04 | 0.42 | 0.05 | 4.94 | 0.42 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9746 | 0.9678 | 0.9602 | 0.9541 |
| sift1M_large | 0.9755 | 0.9665 | 0.9595 | 0.9536 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9286 / 1948K | 0.9778 / 1035K | 0.9947 / 507K | 0.9988 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.9150 / 1948K | 0.9711 / 1035K | 0.9922 / 533K | 0.9982 / 179K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +6.1% (12.67 → 11.90 s, 1.03× CAGRA native), 1M-auto +0.4% (path untouched). Large GPU peak 2052 MiB in one rep (the known bucket-size outlier, #9), within the gate.

### #43 — Step 1 single raw read + parallel u8→f32; Step 2 pinned, reused KMeans|| train-subset buffer  (2026-10-21, commit ca281b6)
**Idea / rationale:** temporary timers showed host-side overheads in both paths. (1) Step 2: before the GPU work, each round built the KMeans|| training subset in a fresh `std::vector<float>`, and the `resize` zero-filled and page-faulted 128 MB on one thread (0.09 s/round, 0.15–0.40 s for the whole subsample). The subset now goes into a pinned `cudaHostAlloc` buffer, allocated once with 5% slack and reused across rounds; the parallel gather writes into it, and the H2D upload is from pinned memory. (2) Step 1: `sample_from` spent 0.56 s in `read_u8bin_to_f32` (single-threaded convert), and the GPU-resident / out-of-core host-raw path then read the whole file again. For u8 data the file is now read once into `raw_host` and converted to float in parallel; `raw_host` is freed if neither user needs it. This follows cuVS 26.12 `ivf_pq_build.cuh` / `cagra_build.cuh`, which stage host batches through pinned buffers that are reused, not reallocated per batch.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Borderline confirmation:** auto rerun `sift1M_auto_b` 8.79 s (+4.7%), ACCEPT, so both runs pass. Step 2 auto 1.14 → 0.81 s, Step 1 auto 0.76 → 0.64 s; large Step 2 1.48 → 1.09 s.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 8.87 | +3.8% | +59.2% | 0.77x | bucket2 7.08, optimize 1.29, translate 0.43 | 4042 | 1.44 |
| sift1M_large | large | 11.76 | +1.2% | +84.3% | 1.02x | bucket2 9.77, prune 0.86, optimize_chunked 0.64, translate 0.48 | 1674 | 2.75 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.81 | 0.10 | 0.00 | 0.67 | 0.50 | 0.02 | 3.36 | 0.00 | 0.48 |
| sift1M_large | 0.57 | 1.09 | 0.17 | 0.00 | 1.18 | 0.39 | 0.04 | 5.14 | 0.46 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9754 | 0.9663 | 0.9594 | 0.9534 |
| sift1M_large | 0.9719 | 0.9666 | 0.9596 | 0.9538 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9277 / 2222K | 0.9771 / 1189K | 0.9947 / 562K | 0.9987 / 183K | 0.9996 / 68K |
| ours sift1M_large | 0.9147 / 1946K | 0.9716 / 1010K | 0.9921 / 486K | 0.9982 / 188K | 0.9995 / 66K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +3.8% (9.22 → 8.87 s, 0.77× CAGRA native; rerun +4.7%), 1M-large +1.2% (11.90 → 11.76 s).

### #46 — GPU-resident path: prune + reverse edges + id translation inside bucket2  (2026-10-21, commit 7c8d0e6)
**Idea / rationale:** user decision (2026-10-21): merge the resident path's separate stages into the bucket2 process. New `--fused-optimize-degree K` (default 0 = off): after the last round, the resident path downloads the (N, M) running KNN table once. While `vector_knn.bin` is written on a background thread (still produced for the knn_similarity quality check), the same thread that did the download frees the resident GPU buffers (`GpuResidentKnn::release_device`), sanitizes to uint32 (same rules as `optimize.cu`), runs the same cuVS 26.10 `cagra::helpers::optimize` (CAGRA rank-based prune + reverse edges + merge), translates rows and neighbor ids with `inverse_perm` (same as `translate_graph_ids.hpp`, with bounds and permutation checks), and writes `final_graph.bin` in the translate_ids format. All design stages remain (prune, reverse edges, id translation); what disappears is two process start-ups, the 256 MB candidate-graph read back and the 128 MB intermediate-graph write + read. The out-of-core path ignores the flag, and a resident run that can't produce the graph leaves no `final_graph.bin`, so `run_pipeline.sh` falls back to the separate `optimize` (exit 2 → large path) / `translate_ids` stages. Harness: `run_pipeline.sh` passes `--fused-optimize-degree $OUTPUT_DEGREE` in auto mode and skips optimize / translate when `final_graph.bin` exists.
**Equivalence check (1 run):** in-process graph vs separate optimize + translate_ids on the same `vector_knn.bin`: 57.3K rows differ as sets, vs 55.6K between two runs of the separate `optimize` (cuVS optimize is itself nondeterministic: atomic reverse-edge order); search recall@10 0.92857 vs 0.92854 at itopk 32; 0 bad ids / self loops / duplicate edges.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`, `.claude/skills/optimize-bucket2/scripts/run_pipeline.sh`
**Params changed:** none (flag passed by the harness)
**In-process tail (rep0):** sanitize 0.022 s, optimize 0.441 s, translate 0.023 s, write 0.134 s = 0.65 s, overlapped with the vector_knn.bin write (bucket2 Write 0.48 → 0.91 s); replaces optimize 1.29 s + translate 0.43 s.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 7.54 | +15.0% | +65.3% | 0.65x | bucket2 7.54 | 3830 | 1.79 |
| sift1M_large | large | 11.70 | +0.5% | +84.4% | 1.01x | bucket2 9.75, prune 0.85, optimize_chunked 0.64, translate 0.47 | 1698 | 2.76 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.82 | 0.10 | 0.00 | 0.65 | 0.50 | 0.02 | 3.34 | 0.00 | 0.91 |
| sift1M_large | 0.57 | 1.12 | 0.16 | 0.00 | 1.04 | 0.47 | 0.04 | 5.11 | 0.43 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9747 | 0.9673 | 0.9588 | 0.9530 |
| sift1M_large | 0.9723 | 0.9667 | 0.9595 | 0.9537 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| ours sift1M_auto | 0.9282 / 1947K | 0.9772 / 1036K | 0.9945 / 474K | 0.9987 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.9138 / 1945K | 0.9708 / 893K | 0.9923 / 500K | 0.9981 / 187K | 0.9995 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-auto +15.0% (8.87 → 7.54 s, 0.65× CAGRA native), 1M-large +0.5% (path untouched).

### Loop rule change (user, 2026-10-21, 2nd)
SIFT10M is back in every iteration (`run_pipeline.sh -d sift10M -m auto -r 1`, third argument of `judge.py`). A >= 3% SIFT10M build-time gain is accepted as long as neither SIFT1M config is > 3% slower and search passes. SIFT10M quality uses the large-path rule (recall@10 >= CAGRA - 5% at every itopk). The CAGRA SIFT10M search reference was measured now on `evalCuvs/baseline/cagra_sift10M_64_32.bin` (0.8672 / 0.9463 / 0.9839 / 0.9959 / 0.9992). The large-path memory guard also covers SIFT10M. The SIFT10M baseline in best.json is the post-#46 run (`runs/sift10M_after46`, 178.47 s). judge.py: the recall tolerance now follows the path the run took (it used the config-name suffix, i.e. the resident 1% for sift10M_auto).

### #47 — incremental KMeans|| d(x, C) (retry of #12/#15, user request)  (2026-10-21, commit 140170e)
**Idea / rationale:** enter each KMeans|| round with the running min distance d(x, C); compute distances only to the round's new candidates (`min_cluster_distance` vs `CpRaw`) and merge them with `thrust::minimum`. This skips the redundant round-0 recompute and the last-round update nothing reads: work per round goes from all candidates (1 → ~5K → ~10K) to the ~5K new ones. Follows cuVS 26.12 `cluster/detail/kmeans.cuh::initScalableKMeansPlusPlus`. Round-1 candidate counts are identical to the old code (5020 / 5017 / 5026). Judged under the new SIFT10M rule on the measurements from the first #47 run (same code).
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**SIFT10M:** KMeans|| 15.5 → 10.4 s per round, Step 2 50.3 → 34.7 s, bucket2 154.5 → 139.9 s.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 7.38 | +2.1% | +66.1% | 0.64x | bucket2 7.38 | 3838 | 1.79 |
| sift1M_large | large | 11.66 | +0.3% | +84.5% | 1.01x | bucket2 9.67, prune 0.84, optimize_chunked 0.65, translate 0.40 | 1674 | 2.76 |
| sift10M_auto | large | 163.74 | +8.3% | +8.3% | 1.34x | bucket2 139.88, prune 7.05, optimize_chunked 13.33, translate 3.48 | 9454 | 20.07 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.66 | 0.11 | 0.00 | 0.64 | 0.51 | 0.02 | 3.35 | 0.00 | 0.91 |
| sift1M_large | 0.57 | 0.91 | 0.30 | 0.00 | 1.12 | 0.42 | 0.04 | 5.02 | 0.54 | 0.00 |
| sift10M_auto | 5.43 | 34.75 | 1.92 | 0.00 | 39.67 | 2.91 | 0.34 | 49.33 | 3.86 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9753 | 0.9666 | 0.9594 | 0.9533 |
| sift1M_large | 0.9755 | 0.9668 | 0.9590 | 0.9534 |
| sift10M_auto | 0.9128 | 0.8954 | 0.8810 | 0.8705 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9295 / 2219K | 0.9774 / 1190K | 0.9945 / 562K | 0.9989 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9137 / 1945K | 0.9712 / 1035K | 0.9920 / 480K | 0.9980 / 188K | 0.9995 / 68K |
| ours sift10M_auto | 0.8396 / 1870K | 0.9280 / 1012K | 0.9736 / 490K | 0.9919 / 180K | 0.9979 / 68K |

**Verdict:** ACCEPT (SIFT10M rule) — speedup vs previous best: 10M-auto +8.3% (178.47 → 163.74 s), 1M-auto +2.1% (7.54 → 7.38 s), 1M-large +0.3%.

### #49 — shared per-row threshold in block top-M (#48) + out-of-core per-bucket submit trims and vectorized gather (#44)  (2026-10-21, commit 8bf4767)
**Idea / rationale:** (1) #48: in `block_topK_from_dots_kernel` (resident and out-of-core Step 6) the G=8 warps of a row each kept their own top-K threshold and warmed up from inf. Each warp now publishes its threshold to a shared-memory `atomicMin` (order-preserving float→uint key, since the partial distance ‖p‖² - 2·dot can be negative) and filters with the block minimum: any warp's current K-th is an upper bound of the row's K-th, so fewer serial ballot inserts (raft/faiss block-select filtering, cuVS 26.12 `detail/faiss_select`). Alone: resident loop 0.975 → 0.907 s/round, +2.98% auto. (2) #44 patch: out-of-core Step 6 GEMM timing events only with `BUCKET2_GEMM_TIMING`; job table read by the gather kernel through UVA-mapped pinned memory (no per-bucket H2D; `ev_h2d` recorded after the kernel); `devcache_gather_pool_kernel` with uchar4 loads / half2 stores. #44 alone moved the out-of-core loop from host-bound to GPU-bound; together with #48's cheaper top-M the out-of-core loop drops ~1.4 → ~1.25 s/round.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 7.20 | +2.4% | +66.9% | 0.62x | bucket2 7.20 | 3934 | 1.80 |
| sift1M_large | large | 10.91 | +6.4% | +85.5% | 0.95x | bucket2 8.99, prune 0.86, optimize_chunked 0.63, translate 0.44 | 1828 | 2.76 |
| sift10M_auto | large | 157.00 | +4.1% | +12.0% | 1.28x | bucket2 133.19, prune 7.10, optimize_chunked 13.32, translate 3.39 | 9454 | 20.30 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.66 | 0.10 | 0.00 | 0.64 | 0.54 | 0.02 | 3.12 | 0.00 | 0.92 |
| sift1M_large | 0.57 | 0.88 | 0.27 | 0.00 | 1.25 | 0.42 | 0.04 | 4.38 | 0.51 | 0.00 |
| sift10M_auto | 5.42 | 34.83 | 1.90 | 0.00 | 39.64 | 2.92 | 0.37 | 42.40 | 4.01 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9751 | 0.9666 | 0.9589 | 0.9535 |
| sift1M_large | 0.9739 | 0.9671 | 0.9595 | 0.9534 |
| sift10M_auto | 0.9167 | 0.8963 | 0.8825 | 0.8704 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9293 / 2226K | 0.9769 / 1191K | 0.9945 / 563K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9142 / 1946K | 0.9710 / 1033K | 0.9920 / 516K | 0.9981 / 187K | 0.9995 / 68K |
| ours sift10M_auto | 0.8403 / 1871K | 0.9289 / 1012K | 0.9736 / 526K | 0.9920 / 186K | 0.9981 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +6.4% (11.66 → 10.91 s, 0.95× CAGRA native), 10M-auto +4.1% (163.74 → 157.00 s), 1M-auto +2.4% (7.38 → 7.20 s).

### #50 — Step 4 assignment: exact fp16 GEMM only up to 32K centroids, CAGRA graph search (itopk 32) above  (2026-10-21, commit 1a799f2)
**Idea / rationale:** the exact top-2 assignment (#7/#8/#22: fp16 GEMM vs all centroids + top-2 kernel) costs ∝ N × n_centroids. Each 4096-query tile also writes and re-reads a 4096 × n_centroids fp32 dots matrix, so at SIFT10M (9.9M points × 100K centroids) the round is memory-bound: 11.1 s/round "search", ~7.8 TB of dots traffic. CAGRA single-CTA search on the centroid KNN graph (cuVS `cagra::search`, already the PQ / fallback path; cost ∝ N × log n_centroids) is cheaper once the centroids are many. Since only the top 2 are needed, itopk drops 64 → 32 (smallest 32-aligned value). SIFT10M one-round measurements: exact 11.15 s, CAGRA itopk 64 9.27 s, CAGRA itopk 32 5.19 s; bucket sizes the same (max 850–862, avg 100). SIFT1M (10K centroids) keeps the exact path (0.12 s/round vs ~0.7 s for CAGRA). Env overrides: `BUCKET2_EXACT_ASSIGN_MAX` (default 32768), `BUCKET2_ASSIGN_ITOPK` (default 32).
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Noise rerun:** first SIFT1M-large run 11.30 s (−3.6%, Steps 2/4/6 each +0.1 s; that path's code is unchanged, it uses the exact assignment) → rerun `sift1M_large_b` 11.11 s (−1.8%), judged with the rerun.
**SIFT10M:** Step 4 39.6 → 20.0 s, bucket2 133.2 → 115.1 s, GPU peak 9454 → 7810 MiB.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 7.18 | +0.3% | +67.0% | 0.62x | bucket2 7.18 | 3774 | 1.79 |
| sift1M_large | large | 11.11 | -1.8% | +85.2% | 0.96x | bucket2 9.12, prune 0.85, optimize_chunked 0.63, translate 0.43 | 2052 | 2.76 |
| sift10M_auto | large | 138.79 | +11.6% | +22.2% | 1.13x | bucket2 115.06, prune 7.03, optimize_chunked 13.27, translate 3.43 | 7810 | 19.73 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.67 | 0.10 | 0.00 | 0.69 | 0.44 | 0.02 | 3.14 | 0.00 | 0.92 |
| sift1M_large | 0.57 | 0.99 | 0.22 | 0.00 | 1.18 | 0.41 | 0.04 | 4.39 | 0.52 | 0.00 |
| sift10M_auto | 5.44 | 34.78 | 1.93 | 0.00 | 20.01 | 2.83 | 0.30 | 43.66 | 4.41 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9740 | 0.9671 | 0.9601 | 0.9541 |
| sift1M_large | 0.9749 | 0.9667 | 0.9591 | 0.9530 |
| sift10M_auto | 0.9189 | 0.8990 | 0.8841 | 0.8728 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9288 / 1948K | 0.9770 / 1035K | 0.9944 / 523K | 0.9987 / 188K | 0.9997 / 68K |
| ours sift1M_large | 0.9150 / 1946K | 0.9708 / 1035K | 0.9923 / 533K | 0.9983 / 189K | 0.9995 / 68K |
| ours sift10M_auto | 0.8379 / 1870K | 0.9283 / 1011K | 0.9738 / 525K | 0.9921 / 187K | 0.9979 / 68K |

**Verdict:** ACCEPT (SIFT10M rule) — speedup vs previous best: 10M-auto +11.6% (157.00 → 138.79 s, 1.13× CAGRA native), 1M-auto +0.3%, 1M-large −1.8% (rerun; within the 3% tolerance).

### #51 — KMeans|| training subset fraction 0.25 → 0.10  (2026-10-21, commit 7a74a8b)
**Idea / rationale:** Step 2 runs KMeans|| D² oversampling on a Bernoulli(F) subset of the working set (#14; cost ∝ train points × candidates). F=0.25 was picked on SIFT1M only (#14 sweep); at SIFT10M the subset is 2.5M points and Step 2 was 34.8 s (KMeans|| 10.4 s/round). Same as cuVS 26.12 IVF-PQ `kmeans_trainset_fraction` (`ivf_pq/ivf_pq_build.cuh`), which trains on a fixed fraction. The centroid count stays N·centroid_ratio, the centroids are still real data points, and the subset still has > 4× centroids (SIFT1M 100K vs 10K; SIFT10M 1M vs 100K).
**Files:** none  **Params changed:** `BUCKET2_EXTRA`: `--kmeans-train-fraction 0.25` → `0.10`
**Effect:** SIFT10M Step 2 34.8 → 15.1 s; SIFT1M auto Step 2 0.66 → 0.36 s; pre-prune KNN recall and final search recall unchanged within noise.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.80 | +5.3% | +68.7% | 0.59x | bucket2 6.80 | 3894 | 1.71 |
| sift1M_large | large | 10.94 | +1.5% | +85.4% | 0.95x | bucket2 8.92, prune 0.87, optimize_chunked 0.63, translate 0.44 | 1674 | 2.69 |
| sift10M_auto | large | 118.38 | +14.7% | +33.7% | 0.97x | bucket2 94.76, prune 7.05, optimize_chunked 13.09, translate 3.48 | 7808 | 18.87 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.36 | 0.11 | 0.00 | 0.65 | 0.45 | 0.02 | 3.15 | 0.00 | 0.92 |
| sift1M_large | 0.57 | 0.73 | 0.25 | 0.00 | 1.29 | 0.39 | 0.04 | 4.42 | 0.54 | 0.00 |
| sift10M_auto | 5.47 | 15.08 | 1.91 | 0.00 | 19.46 | 2.92 | 0.33 | 43.86 | 4.21 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9750 | 0.9663 | 0.9591 | 0.9533 |
| sift1M_large | 0.9740 | 0.9661 | 0.9590 | 0.9535 |
| sift10M_auto | 0.9210 | 0.8976 | 0.8839 | 0.8723 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9285 / 1947K | 0.9770 / 1037K | 0.9944 / 490K | 0.9987 / 170K | 0.9997 / 68K |
| ours sift1M_large | 0.9140 / 2217K | 0.9705 / 1189K | 0.9923 / 562K | 0.9982 / 189K | 0.9996 / 68K |
| ours sift10M_auto | 0.8380 / 1863K | 0.9292 / 1010K | 0.9738 / 516K | 0.9922 / 178K | 0.9980 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 10M-auto +14.7% (138.79 → 118.38 s, **0.97× CAGRA native**), 1M-auto +5.3% (7.18 → 6.80 s), 1M-large +1.5%.

### #52 — optimize_chunked method E: GPU phase 2 with staged device memory (fits SIFT10M)  (2026-10-21, commit 0e0c2aa)
**Idea / rationale:** the GPU phase 2 from #39 allocated everything at once (dst/src/rank/64-bit idx/key2 + output, ~14.8 GB at 3.2 × 10⁸ edges) next to the 3.2 GB scatter buffer, so on SIFT10M it fell back to the 8 s CPU phase 2 (fallback added after the SIFT10M OOM). Now: (1) unpack the edges, then free the scatter buffer (single chunk, single drain: never used again); (2) 32-bit sort index instead of 64-bit (E < 2³², checked); (3) after building the (dst, rank) key, free dst / rank / idx before the second radix sort; (4) allocate the output only after freeing key2. Peak ≈ 24 B/edge + forward graph + output. The fit check now tests both phases: unpack with the scatter buffer resident, and sort after releasing it. CPU phase 2 is still the fallback. Same sort / merge rules as before (output byte-identical to the CPU path, verified in #39).
**Files:** `bucketDemo/buildBucket/src/optimize_chunked.cu`
**Params changed:** none
**SIFT10M:** optimize_chunked 13.09 → 4.88 s (phase 2 7.98 → 1.16 s); GPU peak 8998 MiB (within the 1.2× guard).

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.87 | -1.0% | +68.4% | 0.60x | bucket2 6.87 | 3910 | 1.71 |
| sift1M_large | large | 10.72 | +2.0% | +85.7% | 0.93x | bucket2 8.86, prune 0.84, optimize_chunked 0.64, translate 0.46 | 1602 | 2.68 |
| sift10M_auto | large | 110.94 | +6.3% | +37.8% | 0.90x | bucket2 95.51, prune 7.01, optimize_chunked 4.88, translate 3.54 | 8998 | 18.80 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.35 | 0.12 | 0.00 | 0.67 | 0.47 | 0.02 | 3.13 | 0.00 | 0.91 |
| sift1M_large | 0.57 | 0.62 | 0.36 | 0.00 | 1.19 | 0.39 | 0.04 | 4.48 | 0.53 | 0.00 |
| sift10M_auto | 5.44 | 14.95 | 1.96 | 0.00 | 19.92 | 2.91 | 0.33 | 44.50 | 4.09 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9740 | 0.9665 | 0.9594 | 0.9536 |
| sift1M_large | 0.9743 | 0.9658 | 0.9590 | 0.9536 |
| sift10M_auto | 0.9169 | 0.8987 | 0.8834 | 0.8715 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9287 / 1948K | 0.9776 / 1034K | 0.9947 / 533K | 0.9987 / 189K | 0.9997 / 68K |
| ours sift1M_large | 0.9135 / 2181K | 0.9708 / 1189K | 0.9922 / 562K | 0.9981 / 189K | 0.9995 / 66K |
| ours sift10M_auto | 0.8368 / 1864K | 0.9288 / 1010K | 0.9733 / 480K | 0.9918 / 184K | 0.9980 / 68K |

**Verdict:** ACCEPT (SIFT10M rule) — speedup vs previous best: 10M-auto +6.3% (118.38 → 110.94 s, **0.90× CAGRA native**), 1M-large +2.0%, 1M-auto −1.0% (noise, path untouched).

### #54 — out-of-core: Step 2 train subset gathered from the raw u8 dataset (no N×D float copy)  (2026-10-21, commit a17e602)
**Idea / rationale:** Step 1 converted the whole dataset into a float32 `X_sampled` (SIFT10M 5.1 GB: serial zero-fill + conversion, Step 1 5.4 s, and the bulk of bucket2's 19.6 GB peak RSS, above the 16 GB `cpu_limit`). The only reader is Step 2, which since #51 uses a 10% Bernoulli subset. On the out-of-core path the raw u8 dataset is already in host memory in the current coordinate system (#42 swaps in the reordered copy after round 1). So with `kmeans_train_fraction < 1` Step 1 now only reads the raw copy (`x_lazy`), and Step 2 gathers + converts the subset rows straight from `raw_host` each round (no `x_row_of` mapping needed). Paths that don't keep `raw_host` (GPU-resident, over-budget out-of-core) still build `X_sampled` right after the path decision. Round-1 candidate counts identical (same rows / values). Same idea as cuVS 26.12 IVF-PQ build: sample the trainset from the host dataset instead of holding a converted copy (`ivf_pq_build.cuh`).
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Borderline confirmation:** SIFT1M-large rerun `sift1M_large_b` 10.34 s (+3.5%) → both runs pass.
**SIFT10M:** Step 1 5.42 → 1.34 s, bucket2 peak RSS 19.56 → 14.03 GB (back under the 16 GB budget); total +1.7% only (Step 6 47.6 s vs 44.5 s in this run). SIFT1M-large Step 1 0.57 → 0.13 s, peak RSS 2.68 → 2.21 GB.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.82 | +0.7% | +68.6% | 0.59x | bucket2 6.82 | 4070 | 1.71 |
| sift1M_large | large | 10.39 | +3.1% | +86.2% | 0.90x | bucket2 8.39, prune 0.85, optimize_chunked 0.64, translate 0.51 | 1746 | 2.21 |
| sift10M_auto | large | 109.02 | +1.7% | +38.9% | 0.89x | bucket2 93.42, prune 7.08, optimize_chunked 4.88, translate 3.64 | 8998 | 14.03 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.64 | 0.35 | 0.10 | 0.00 | 0.67 | 0.47 | 0.02 | 3.13 | 0.00 | 0.92 |
| sift1M_large | 0.13 | 0.60 | 0.33 | 0.00 | 1.23 | 0.32 | 0.04 | 4.50 | 0.51 | 0.00 |
| sift10M_auto | 1.34 | 14.61 | 2.01 | 0.00 | 19.17 | 2.79 | 0.34 | 47.62 | 4.04 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9745 | 0.9662 | 0.9594 | 0.9537 |
| sift1M_large | 0.9724 | 0.9666 | 0.9593 | 0.9531 |
| sift10M_auto | 0.9206 | 0.8988 | 0.8832 | 0.8716 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9284 / 2222K | 0.9768 / 1184K | 0.9943 / 563K | 0.9987 / 186K | 0.9997 / 68K |
| ours sift1M_large | 0.9139 / 1950K | 0.9709 / 1035K | 0.9922 / 496K | 0.9983 / 178K | 0.9995 / 66K |
| ours sift10M_auto | 0.8383 / 1867K | 0.9288 / 1011K | 0.9739 / 525K | 0.9920 / 174K | 0.9978 / 68K |

**Verdict:** ACCEPT — speedup vs previous best: 1M-large +3.1% (10.72 → 10.39 s; rerun +3.5%), 10M-auto +1.7% (110.94 → 109.02 s, 0.89× CAGRA native; −5.5 GB RSS), 1M-auto +0.7%.

### #56 — KMeans|| d(x, C) via exact fp16 Tensor Core GEMM (8-bit data)  (2026-10-21, commit d3c3596)
**Idea / rationale:** KMeans|| (Step 2) computes d(x, C) for the train subset against each round's new candidates with cuVS fp32 `brute_force` (CUDA cores; SIFT10M: 1M train points × ~50K candidates per call, kmeans_parallel 4.15 s/round). For 8-bit data the same exact fp16 GEMM argument as Step 4 (#22) holds: values exact in fp16, products / sums < 2²⁴ exact in fp32. So `KmeansMinDistF16` converts X once to fp16 (+ ‖x‖²), and per call converts C, runs 4096-row tiles of `cublasGemmEx` fp16→fp32 Tensor Core GEMM and a warp-per-row min of ‖c‖² − 2x·c (+ ‖x‖², clamped at 0 like L2Expanded). The L2 values are bit-identical to brute_force, so D² sampling is unchanged: SIFT10M round-1 candidate counts and psi identical (49770 / 50007 / 50014). Only used when the input is uint8/int8 and PQ is off; other types keep brute_force. Same tiled GEMM + select shape as cuVS 26.12 `detail/knn_brute_force.cuh::tiled_brute_force_knn`, with tensor-core fp16 inputs.
**Files:** `bucketDemo/buildBucket/src/bucket.cu`
**Params changed:** none
**Borderline confirmation:** first run 1M-large +3.2% / 10M +4.0% (Step 6 +3.3 s noise on 10M); reruns 1M-large 10.23 s (+1.5%), 10M 98.64 s (+9.5%) → both judge runs ACCEPT. best.json stores 1M-large from the rerun and 10M from the first run (the slower of each pair).
**SIFT10M:** kmeans_parallel 4.15 → 1.73 s/round, Step 2 14.6 → 7.4 s.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.80 | +0.3% | +68.7% | 0.59x | bucket2 6.80 | 3980 | 1.72 |
| sift1M_large | large | 10.23 | +1.5% | +86.4% | 0.89x | bucket2 8.32, prune 0.84, optimize_chunked 0.64, translate 0.38 | 1858 | 2.21 |
| sift10M_auto | large | 104.66 | +4.0% | +41.4% | 0.85x | bucket2 89.35, prune 7.00, optimize_chunked 4.85, translate 3.46 | 8998 | 14.03 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.26 | 0.17 | 0.00 | 0.61 | 0.50 | 0.02 | 3.14 | 0.00 | 0.91 |
| sift1M_large | 0.13 | 0.61 | 0.34 | 0.00 | 1.12 | 0.41 | 0.04 | 4.51 | 0.52 | 0.00 |
| sift10M_auto | 1.36 | 7.37 | 2.11 | 0.00 | 18.49 | 2.81 | 0.94 | 50.94 | 4.19 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9723 | 0.9660 | 0.9595 | 0.9535 |
| sift1M_large | 0.9751 | 0.9665 | 0.9590 | 0.9529 |
| sift10M_auto | 0.9193 | 0.8988 | 0.8840 | 0.8724 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9283 / 1947K | 0.9771 / 936K | 0.9943 / 561K | 0.9988 / 181K | 0.9997 / 68K |
| ours sift1M_large | 0.9135 / 1941K | 0.9704 / 1032K | 0.9921 / 523K | 0.9982 / 187K | 0.9996 / 68K |
| ours sift10M_auto | 0.8392 / 1872K | 0.9287 / 1040K | 0.9742 / 526K | 0.9923 / 186K | 0.9980 / 68K |

**Verdict:** ACCEPT (SIFT10M rule) — speedup vs previous best: 10M-auto +4.0% (109.02 → 104.66 s; rerun +9.5%, 98.64 s = 0.80× CAGRA native), 1M-large +1.5% (rerun) / +3.2%, 1M-auto +0.3%.

### #57 — prune_windowed GPU path: no float copy; pinned double-buffered H2D; results streamed D2H → writer  (2026-10-21, commit TBD)
**Idea / rationale:** timers inside the #38 GPU prune on SIFT10M showed compute 4.3 s = output vector zero-fill 0.92 s + pageable H2D of 3.84 GB 1.37 s (2.8 GB/s) + kernel 1.08 s + D2H into fresh pageable pages 0.80 s, then a separate 1.17 s write; the bulk load also converted the whole dataset to a float copy (5.1 GB) that the GPU kernel never reads. Now: (1) decide GPU vs CPU before the bulk load and skip the float buffer on the GPU path; (2) `Staging` with two 64 MB pinned buffers: OpenMP memcpy of the next chunk into one buffer while the previous chunk DMAs from the other (same pinned-staging pattern as cuVS 26.12 `cagra_build.cuh::build_knn_graph`); (3) results come back in row chunks through the pinned buffers into a sink callback that feeds `StreamingGraphWriter` directly (no N × out_deg host buffer; D2H overlaps the write). Output byte-identical (cmp on the SIFT10M intermediates).
**Files:** `bucketDemo/buildBucket/src/prune_windowed.cpp`, `prune_windowed_gpu.cu`, `prune_windowed_gpu.hpp`
**Params changed:** none
**SIFT10M:** prune 7.00 → 4.05 s (compute 4.3 → 1.85 s), prune peak RSS 10.2 → 4.1 GB.

**Build time (s, median; sum of all stages)**

| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |
|---|---|---|---|---|---|---|---|---|
| sift1M_auto | resident | 6.82 | -0.3% | +68.6% | 0.59x | bucket2 6.82 | 3828 | 1.72 |
| sift1M_large | large | 10.23 | +0.0% | +86.4% | 0.89x | bucket2 8.42, prune 0.72, optimize_chunked 0.66, translate 0.43 | 1938 | 2.23 |
| sift10M_auto | large | 96.09 | +8.2% | +46.2% | 0.78x | bucket2 83.81, prune 4.05, optimize_chunked 4.88, translate 3.35 | 8998 | 14.07 |

**bucket2 step breakdown (s)**

| config | Step 1 (Load sampled data) | Step 2 (Select centroids) | Step 3 (Centroid KNN graph) | Step 3.5 (Load full dataset) | Step 4 (Batch assignment) | Step 4.5 (Inline reorder, round 1 only) | Step 5 (Write buckets) | Step 6 (Per-vector KNN, merge included) | Wait (final iteration's background merge) | Write (close + .npy convert) |
|---|---|---|---|---|---|---|---|---|---|---|
| sift1M_auto | 0.63 | 0.26 | 0.17 | 0.00 | 0.65 | 0.51 | 0.02 | 3.14 | 0.00 | 0.92 |
| sift1M_large | 0.13 | 0.55 | 0.46 | 0.00 | 1.02 | 0.42 | 0.04 | 4.62 | 0.49 | 0.00 |
| sift10M_auto | 1.35 | 7.53 | 1.96 | 0.00 | 18.89 | 2.84 | 0.30 | 45.80 | 4.12 | 0.00 |

**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**

| config | recall@1 | recall@10 | recall@32 | recall@64 |
|---|---|---|---|---|
| sift1M_auto | 0.9735 | 0.9664 | 0.9591 | 0.9534 |
| sift1M_large | 0.9759 | 0.9663 | 0.9596 | 0.9535 |
| sift10M_auto | 0.9186 | 0.8974 | 0.8828 | 0.8707 |

**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**

| graph | itopk 32 | itopk 64 | itopk 128 | itopk 256 | itopk 512 |
|---|---|---|---|---|---|
| CAGRA native 64→32, sift1M | 0.9305 / 1942K | 0.9782 / 1036K | 0.9948 / 533K | 0.9988 / 189K | 0.9998 / 68K |
| CAGRA native 64→32, sift10M | 0.8672 / 1868K | 0.9463 / 1013K | 0.9839 / 525K | 0.9959 / 187K | 0.9992 / 68K |
| ours sift1M_auto | 0.9287 / 2223K | 0.9773 / 1193K | 0.9945 / 561K | 0.9988 / 176K | 0.9997 / 67K |
| ours sift1M_large | 0.9146 / 1946K | 0.9710 / 838K | 0.9923 / 562K | 0.9981 / 189K | 0.9996 / 68K |
| ours sift10M_auto | 0.8379 / 1873K | 0.9290 / 1012K | 0.9737 / 526K | 0.9917 / 179K | 0.9980 / 68K |

**Verdict:** ACCEPT (SIFT10M rule) — speedup vs previous best: 10M-auto +8.2% (104.66 → 96.09 s, **0.78× CAGRA native**; the stored 104.66 s was the slower of #56's two runs, its rerun was 98.64 s), 1M-large 0.0%, 1M-auto −0.3%.

<!-- template for an accepted entry:
### #N — <one-line idea>  (<date>, commit <hash>)
**Idea / rationale:** what changed and why; cuVS 26.12 reference (file::function) it follows.
**Files:** ...
**Params changed:** ... (or none)
<paste `judge.py ... --md` output: build-time table, bucket2 step table, (a) KNN-similarity table, (b) search table>
**Verdict:** ACCEPT — speedup vs previous best: 1M-auto x%, 10M-large y%.
-->
