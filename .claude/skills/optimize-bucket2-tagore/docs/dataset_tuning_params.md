# Dataset-specific tunables of the bucket2 pipeline

Every parameter that was tuned for SIFT100M on this machine (4× V100-16GB, 251 GB RAM, SATA SSD) or
that must be re-checked for a new dataset / size. Paths are relative to `bucketDemo/buildBucket/` (sources)
or the skill dir (`scripts/`). Precedence for the run_pipeline parameters: exported env variable (sweeps)
> `datasets.json` `"params"` (per-dataset calibration) > `dataset_env.py` size rules > `params.sh`.

**Class**: *resource* = changes only speed / memory (implementation, try directly); *algorithm* = changes
which candidates are considered or how the graph is defined (A2 approved for bucket size / NPROBE within
avg bucket 128–512; anything else needs the user); *fixed* = set by the user, listed for completeness.

## 1. Bucketing and candidate set (algorithm)

| name | where | value per dataset | controls | derivation now | symptoms when wrong | how to tune | class |
|---|---|---|---|---|---|---|---|
| `CENTROID_RATIO` (avg bucket B = 1/ratio) | `scripts/dataset_env.py` → `bucket2 --centroid-ratio` | 0.004 (B 250) for all six; msTuring after P4: see §5 | number of buckets = N·ratio; bucket size drives Step 6 GEMM shape, Step 4 cost (∝ N·C for exact assignment), centroid KNN and Step 2 cost | rule: B = 250, kept in [128, 512]; ×2 beyond 512 only when the centroid matrix > 1 GB GPU | pre-prune recall@10 low at the same pool size (too small B → neighbors split over many buckets); Step 4 / Step 2 time explodes (too many centroids: msTuring100M 400K centroids → Step 4 2451 s) | sweep B ∈ {128, 256, 512} × pool rows on the 1M subset, confirm on 10M; watch knn_similarity recall@10 + recall at itopk 64 vs CAGRA floor | algorithm (A2) |
| `NPROBE` (pool rows ≈ (NPROBE+1)·B) | `dataset_env.py` → `bucket2 --nprobe`; hard cap `MAX_NPROBE = 256` (`src/bucket.cu:8035`) | 80 (pool 20K) all datasets; > 256 fails (P4 sweep B=128 pool 40K = 313: rc 3) | buckets scanned per bucket in Step 6 = candidate pool per point | rule: ⌈20000 / B⌉ | recall saturates low (too small); Step 6 time ∝ pool rows (too large) | sweep with B (pool 20K / 30K / 40K); recall vs Step 6 time; must stay ≤ 256 | algorithm (A2) |
| `KNN_K` | `scripts/params.sh` (32) → `--knn-k`; also the prune window (center + K centroid-knn buckets) and `order_window` | 32 | centroid KNN graph degree: bucket order, Belady look-ahead, prune window | fixed default | prune finds ~28 / 64 candidates out of window (A4); order / cache misses | only with a logged reason (A2 allows "if needed") | algorithm (A2) |
| `ITERATIONS` | `params.sh` (3), `datasets.json` `"iterations"` (sift100M / msTuring100M: 2) → `-t` | 3 (1M / 10M), 2 (100M) | number of re-bucketing rounds (different seeds): duplicates points across buckets | per dataset by hand | recall plateaus (too few); bucket2 time ∝ rounds | report with every sweep; 100M stays 2 unless the user changes it | algorithm |
| `SEED` | `params.sh` (42) | 42 | KMeans‖ sampling, round seeds | fixed | — | keep fixed (results comparable) | algorithm |
| `--kmeans-train-fraction` | `dataset_env.py` → bucket2 | min(0.10, 10M / N): 0.1 (1M / 10M), 0.1 (100M → 10M rows) | rows sampled for KMeans‖ (Step 1 / 2) | rule | Step 2 slow at large N; centroids poorly placed at small N | change the 10M-row cap only with a quality check | algorithm |
| KMeans‖ rounds / oversampling | `src/bucket.cu:2316` `FIXED_NITER = 3`, `TOTAL_OVERSHOOT = 1.5` | hard-coded | centroid quality vs Step 2 time | hard-coded | Step 2 dominant (msTuring100M 316 s) | — (algorithm) | algorithm |
| Step 4 assignment method | `src/bucket.cu:1436` `BUCKET2_EXACT_ASSIGN_MAX` = 32768 (8-bit only); float: exact fp32 SGEMM always (#26) | 8-bit: exact if C ≤ 32768 else CAGRA search; float: exact | exact top-2 nearest centroid vs approximate CAGRA search | hard-coded threshold, measured on SIFT1M / 10M | Step 4 quadratic (float 100M: 2451 s); approximate assignment changes buckets | **A5** (apply the 32768 threshold to float) awaits the user | algorithm |
| `NEIGHBORS_M` / `OUTPUT_DEGREE` | `params.sh` 64 / 32 | 64 / 32 | pre-prune KNN degree / final graph degree | user | — | not tuned | fixed |

## 2. Memory and I/O budgets (resource)

| name | where | value per dataset | controls | derivation now | symptoms when wrong | how to tune | class |
|---|---|---|---|---|---|---|---|
| `--cache-mb` | `dataset_env.py` → bucket2 | `dataset_env.py` output: 15000 for sift1M…100M and msTuring1M / 10M, 45000 for msTuring100M (bucket-mem 16000 / 46000) | host bucket cache: holds a whole round of records (#9) | rule | out-of-core Step 6 re-reads buckets from disk (disk-bound Step 6) | raise until Step 6 stops reading; watch RSS | resource |
| `--bucket-mem-mb` | `dataset_env.py` | cache + 1000 | rounds 2..t records kept in RAM (#14) | rule | spills to disk between rounds | same as cache | resource |
| `--gpu-bucket-cache-mb` | `dataset_env.py` | GPU / 2 = 8192 | Step 6 GPU Belady bucket cache (#1 / #2 / #21; float since #25) | rule | thrashing (cache-miss uploads dominate Step 6) or OOM for the slots | sweep 2–12 GB on 10M | resource |
| Step 6 slot count | `src/bucket.cu:5306` `MAX_SLOTS = 8`, `OVERSUB = 2.0`, `kMinSlotsCacheMiss = 6`; limited by `usable_bytes = 0.85·free` (`:5219`) | dynamic (6 with GPU cache misses) | pipelined bucket kernels / producer look-ahead | measured on SIFT1M / 10M (#40, #3) | GPU idle waiting on producer (too few), OOM (too many) | log `[VectorKNN] num_slots`; 4–8 | resource |
| dots row tile | `src/bucket.cu:5270` `kDotsRowTile = 256` | 256 | rows of the bucket×pool dots matrix per GEMM tile → per-slot GPU memory | hard-coded | per-slot memory too big for large pools | change with slot count | resource |
| Step 4 query tile | `src/bucket.cu:1428` `kF16Tile = 4096`; float `f32_tile = max(32, min(4096, 512 MB / (4·C)))` | 4096 / 3277 (msTuring100M 400K C → 327) | dots buffer of the exact assignment | rule | GPU OOM with many centroids; tiny tiles → launch bound | — | resource |
| `PRUNE_CACHE_MB` + prune chunk budget | `params.sh` 16000 → `prune_windowed --cache-mb` (default 4000); chunk budget `min(cache/2, 4 GB)` (`src/prune_windowed.cpp:524`) | 16000 / 4 GB | bucket window cache of the windowed prune; GPU prune chunk size | hard-coded | prune re-reads windows (slow), GPU OOM (D > 128 floats) | raise cache for large D | resource |
| `order_window` | `src/bucket.cu:5132`, `src/prune_windowed.cpp:197`: default max(4·K, 16) = 128 | 128 | Belady look-ahead window of the bucket order | rule (from KNN_K) | cache misses | `--order-window` | resource (order itself is algorithm) |
| `CHUNKED_GPU_BUDGET_MB` | `params.sh` 16000 → `optimize_chunked --gpu-budget-mb` (default 8000) | 16000 | GPU chunk size of method E reverse-edge build | hard-coded | many chunks (slow) / OOM | — | resource |
| `CHUNKED_EDGE_BUFFER_MB` | `params.sh` 16000 → `--edge-buffer-mb` (default 1024) | 16000 | host edge buffer before spilling to tmp | hard-coded | disk spill (slow) | raise if RSS allows | resource |
| `--host-budget-mb` | `dataset_env.py` → optimize_chunked | 204800 (all) | host memory ceiling of optimize_chunked (a ceiling, not a target) | fixed by the user | OOM-kill on smaller machines | lower on a smaller host | resource |
| CSR / graph row-block writes | `src/prune_windowed.cpp:409`, `src/optimize_chunked.cu:1849` (per bucket / chunk) | per bucket / chunk | write granularity of the graph files | code | small-write I/O overhead | — | resource |
| bucket2 load config | `src/bucket.cu:151` `cpu_limit_bytes 16 GB`, `bucket_vec_buffer_bytes 1 GB`, `gpu_limit_bytes` auto | defaults | resident vs out-of-core decision of Step 1 / 2 | hard-coded | wrong path choice | — | resource |
| resident / out-of-core fit check | `src/bucket.cu:252` (`fits_in_gpu`), `src/bucket.cu:6780`; `optimize` returns 2 when it does not fit (`src/optimize.cu:307`) → run_pipeline falls back to the large path | sift1M / msTuring1M resident; ≥ 10M large | which path runs | estimate vs free GPU memory | path change = judge reject | `-m large` forces the large path | resource (path must not change in a bench) |
| GEMM precision per dtype | `src/bucket.cu` `kExactF16` (8-bit: fp16 tensor cores, exact), float: fp32 SGEMM (V100 has no TF32) | uint8 / int8: fp16; float32: fp32 | Step 4 / Step 6 / KMeans‖ distances | dtype | fp16 for float would change results | fp16 for float only with exact fp32 re-rank (else A<k>) | resource / algorithm |

## 3. Evaluation (not part of the build)

| name | where | value | notes |
|---|---|---|---|
| CAGRA search itopk list | `cagra_eval search --itopk` (default 32,64,128,256,512) | 32…512 | `search: cagra` datasets (1M / 10M) |
| DiskANN L list | `scripts/diskann_search.py --L` | 32…512 | `search: diskann` (100M); medoid entry |
| quality reference | `datasets.json` `ref` / `own_tol`; judge floors: −0.01 resident, −0.05 large | sift: tagore; msTuring: cagra | gate configs = degree 32 references |

## 4. Procedure for a new dataset

1. `prep_dataset.py` + register in `datasets.json` (SKILL.md "Adding a new dataset").
2. Smoke run with the size rules; check the printed `[ds] … ratio= nprobe=` line and the path.
3. Compare the pre-prune recall@10 (`knn_similarity`) with SIFT at the same size (SIFT1M 0.94, 10M 0.86).
   If it is clearly lower, sweep §1 (`CENTROID_RATIO` × `NPROBE`) on the 1M subset, confirm the best 2–3
   points on 10M, store the result as `datasets.json` `"params"`.
4. Resource rows (§2) only if a stage is memory- or disk-bound (RSS / GPU peak in `metrics.json`).

## 5. Worked example: msTuring (float32, D=100) calibration (P4, 2026-10-24)

Symptom: with the SIFT-derived rules (B 250, pool 20K) msTuring1M reaches pre-prune recall@10 0.687 (SIFT1M 0.943)
and is below the CAGRA gate floor (resident −0.01, large −0.05) at every itopk. Sweep: `runs/tagoreP4/` (1 rep each,
GPU 0 under the bench lock, `CENTROID_RATIO` / `NPROBE` exported; ITERATIONS = 3 throughout).

**msTuring1M auto (resident; floor itopk 32 / 64 / 128 = 0.6528 / 0.7824 / 0.8760)**

| B | NPROBE | pool rows | build s | pre@10 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 157 | 20K | 8.34 | 0.739 | 0.627 | 0.763 | 0.863 | 0.929 | 0.968 |
| 128 | 235 | 30K | 9.40 | 0.785 | 0.642 | 0.776 | 0.872 | 0.935 | 0.971 |
| 128 | 256 | 33K | 9.80 | 0.796 | 0.641 | 0.775 | 0.872 | 0.935 | 0.972 |
| 256 | 79 | 20K | 7.17 | 0.686 | 0.619 | 0.757 | 0.860 | 0.927 | 0.965 |
| 256 | 118 | 30K | 8.26 | 0.735 | 0.633 | 0.769 | 0.869 | 0.933 | 0.970 |
| 256 | 157 | 40K | 9.28 | 0.772 | 0.641 | 0.775 | 0.873 | 0.936 | 0.972 |
| 256 | 196 | 50K | 10.11 | 0.807 | 0.644 | 0.776 | 0.877 | 0.938 | 0.973 |
| 256 | 256 | 65K | 12.16 | 0.842 | 0.648 | 0.780 | 0.878 | 0.941 | 0.975 |
| 512 | 40 | 20K | 6.94 | 0.644 | 0.614 | 0.754 | 0.858 | 0.924 | 0.965 |
| 512 | 59 | 30K | 7.71 | 0.692 | 0.629 | 0.764 | 0.865 | 0.930 | 0.969 |
| 512 | 79 | 40K | 8.95 | 0.733 | 0.637 | 0.771 | 0.872 | 0.935 | 0.972 |
| 512 | 118 | 60K | 10.52 | 0.787 | 0.644 | **0.782** | 0.879 | 0.939 | 0.973 |
| 512 | 157 / 256 | 80K / 131K | — | — | GPU OOM in round 2 (resident Step 6 buffers ∝ pool; `src/bucket.cu:1475` / `:3972`) | | | | |
| 128 | 313 | 40K | — | — | rejected: NPROBE > `MAX_NPROBE` 256 | | | | |
| 512 (large path) | 256 | 131K | 23.37 | 0.899 | 0.634 | 0.771 | 0.872 | 0.934 | 0.969 |
| **exact 64-NN + `optimize`** (ceiling) | | all | — | 1.000 | 0.665 | 0.794 | 0.887 | 0.946 | 0.978 |

Final recall saturates far slower than the pre-prune recall (pre@10 0.69 → 0.84 lifts itopk 64 only 0.757 → 0.780);
even the exact KNN graph only gives 0.794 ≈ CAGRA's 0.792, so the resident floor needs pre@10 ≈ 0.9+, which the
resident Step 6 cannot hold in GPU memory today. **msTuring1M: not calibrated** (defaults kept; no A2 point passes;
needs the resident Step 6 to tile large pools (implementation, backlog) or a quality idea such as A1).

**msTuring10M auto (large; floor 32 / 64 / 128 / 256 / 512 = 0.6121 / 0.7495 / 0.8422 / 0.8951 / 0.9241)**

| B | NPROBE | pool rows | build s | pre@10 | 32 | 64 | 128 | 256 | 512 | GPU MiB | RSS GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 (default) | 80 | 20K | 128.30 (3×) | 0.713 | 0.573 | 0.717 | 0.826 | 0.896 | 0.940 | 8996 | 20.4 |
| 128 | 235 | 30K | 183.39 | 0.788 | 0.589 | 0.736 | 0.843 | 0.911 | 0.951 | 9896 | 18.0 |
| 256 | 157 | 40K | 160.12 | 0.759 | 0.594 | 0.736 | 0.844 | 0.910 | 0.951 | 8996 | 18.1 |
| 256 | 256 | 65K | 189.77 | 0.795 | 0.604 | 0.747 | 0.853 | 0.918 | 0.957 | 8996 | 20.6 |
| 512 | 118 | 60K | 161.62 | 0.737 | 0.597 | 0.746 | 0.850 | 0.914 | 0.954 | 8996 | 19.7 |
| 512 | 196 | 100K | 207.01 | 0.780 | 0.612− | 0.757 | 0.860 | 0.924 | 0.959 | 9628 | 20.8 |
| **512** | **256** | **131K** | 244.28 | 0.801 | **0.615** | **0.756** | **0.862** | **0.926** | **0.962** | 10334 | 20.0 |

Chosen: **B 512 (`CENTROID_RATIO` 0.001953125), `NPROBE` 256** — the only point passing every floor (B 512 /
NPROBE 196 misses itopk 32 by 0.0006). Stored as `datasets.json` `msTuring10M.params`. Price: build ≈ 1.9× the
default (Step 6 pool 6.5×); bigger buckets help more than more probes at the same pool (fewer, larger GEMMs).
msTuring100M keeps the size rules: it already passes its gate (L 64 0.722 vs floor 0.701) and its Step 4 / Step 6
cost at B 512 / pool 131K is unmeasured (3 × ~1.2 h; left for a milestone run).
