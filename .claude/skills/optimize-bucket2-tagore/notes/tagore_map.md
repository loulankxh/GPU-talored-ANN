# TAGORE → bucket2 map (step T0, 2026-10-22)

TAGORE = `/home/lanlu/Tagore` @ 110d480 (line numbers refer to the unpatched clone; the V100 build is
`build-v100/patched-src`, see `scripts/tagore_v100_fix.patch`). Our times = `state/best.json` (#0, `runs/tagore0`),
SIFT100M, 2 rounds, total **3214.6 s**:

| stage | s | inside |
|---|---|---|
| bucket2 | 2725.6 | Step 1 load 71.6, Step 2 select centroids 368.6 (kmeans_parallel 179.5 + 180.4), Step 3 centroid KNN 96.3, Step 4 assignment 254.0 (search 64.4 + 44.5, round-2 BucketWrite 80.9), Step 4.5 inline reorder 194.9 (order 23.4, reorder_info 11.9, pass3_write 67.4, resample 91.6), Step 5 7.4, **Step 6 per-vector KNN 1687.4**, final merge wait 32.3 |
| Step 6 per round | 801.9 / 879.4 | bucket loop 779.9 / 859.8 = **wait_producer 659.7 / 735.3** + gpu_submit 88.1 / 91.0 + stream_sync 20.9 / 20.6 + scatter_pending 10.9 / 12.6; producer `cache_get` 710.4 / 784.4 s; **GPU bucket cache (2 GB): 20.2M / 20.9M uploads for 1M buckets (≈20 per bucket), upload 479.0 / 536.2 s (~24 µs per 12.8 KB bucket)**, evictions ≈ uploads; setup 20.8 / 17.3 (future_access 19.2 / 16.0) |
| prune (prune_windowed, chunked GPU, #76) | 150.0 | compute 132.1, write 13.0 |
| optimize_chunked -m E | 280.2 | pinned alloc 6.0, acc create 12.0, Phase 1 scatter 52.8, **acc.finish 90.7**, Phase 2 merge + write 109.9 |
| translate | 58.8 | |

TAGORE SIFT100M (step B, `largeIndex`, Vamana, 1 GPU): **2151 s** = prep 326 (k-means 2.5, 2-NN assignment 48.7,
split 38.4, cluster-file write 235.3) + `largeIndex` 1825 (103 clusters × ~1.94M points: ≈ 17.7 s per cluster for
upload + GNN-Descent 8+8 it + Vamana prune + D2H, CPU merge overlapped). DiskANN recall@10 L32/128/512 =
0.769 / 0.924 / 0.982 vs ours 0.747 / 0.926 / 0.988.

## Where TAGORE is fast and why (vs us, SIFT100M)

- **Data movement granularity.** TAGORE moves a whole ~1 GB cluster to the GPU with one `cudaMemcpy`
  (`gpu_construct` 130–285) and computes everything for it on-device. We move 1M buckets of ~100 rows (12.8 KB) through
  a 2 GB Belady GPU cache, and each bucket is pulled ~20 times per round (nprobe-expanded pools of ~200 neighbour
  buckets). Each miss costs a host Belady `get` + `std::vector` id conversion + `cudaMallocAsync` + two **pageable**
  `cudaMemcpyAsync` (`bucket.cu` dev_loader ≈ 5320–5370): ~24 µs, latency-bound, on the producer thread. This alone is
  ~1000 s of the 1687 s Step 6. → backlog B1, B2, B3.
- **Fixed buffers instead of per-object allocation.** TAGORE allocates all device buffers once per GPU for
  `max_points` and reuses them for every cluster; host side a fixed `graph_pool[buffer_size]` of `max_points × K`
  slots (`pybind.cu::largeIndex`), filled by D2H into a slot chosen by the eviction order (`order_merge_main` +
  `reorder`, `large_index.cu` 9–108). We `cudaMallocAsync` / `cudaFreeAsync` per cached bucket. → B1.
- **Overlap.** GPU thread (`gpu_construct`) and CPU merge thread (`order_merge_new` 286–522) hand off via
  `has_built` / `has_merged` flags (busy-wait 100 ms polls); evicted local graphs written with a 100 MB
  `cached_ofstream`, final index with a 1 GB one. We already overlap producer / GPU / scatter in Step 6, but the
  producer itself is serial per miss.
- **Bucketing cost.** TAGORE: 100 k-means centers on a 1M sample (2.5 s) + 2-NN GEMM assignment (49 s). Ours needs
  ~1M centroids (avg bucket 100): KMeans|| 180 s per round, Step 4 CAGRA/GEMM assignment 64 / 44 s. Different
  algorithm (fine buckets are what bucket-pair KNN needs); only the kernel tricks transfer. → B4.
- **Where TAGORE loses.** Its cluster-file write (235 s, single-threaded fstream of float vectors, 102 GB) and
  `largeIndex`'s 16 GNN-Descent iterations per 2M-point cluster; at SIFT10M ours is 2.6× faster (82 vs 214 s) and at
  SIFT1M 1.5× faster (6.3 vs 9.2 s). Its merge does no re-prune (concatenate + truncate to 32): cheap but lower quality
  at high L (0.982 vs 0.988 @ L512).

## Component notes

### GNN-Descent phase 1 (`pybind.cu::GNN_descent` 19–151; kernels in `src/Tagore_src.cu`)
- `initialize_graph` 158–177: `<<<N, 32>>>`, random K ids per node (curand Philox seeded by clock()·bid; duplicates /
  self possible), `nei_distance = INF`.
- `cal_power` 1164–1186: `<<<N, DIM>>>`, ‖x‖² in **fp16** (per-warp `__hadd` shuffle reduce) → overflows for raw
  SIFT; `GNN_descent` pre-scales by a power-of-10 `norm_factor` from the first vector, `largeIndex`'s `f2h` does not.
- `nn_descent_opt_sample` 531–602: `<<<N, 32>>>`, thread 0 serially splits the K list into ≤ SAMPLE=32 new / 32 old
  by `nei_visit`, marks new as visited; reverse samples pushed with `atomicAdd(reverse_num[v])` into
  `reverse_graph[v*K + slot]` (slot < 32 kept, rest dropped).
- `nn_descent_opt_reverse_sample` 603–683: `<<<N, 32>>>`, binary-search dedup of reverse samples against the forward
  ones (assumes the forward sample sorted by id — it is in graph/distance order, so dedup is partial; OOB bug fixed).
  Lists go to `hybrid_list[N × 128]` (new 0–63, old 64–127).
- `nn_descent_opt_cal` 684–1042: `<<<N, (32,16)>>>`, shared `new_data` / `old_data` = 64 × 128 fp16 each (16 KB +
  16 KB); one block per node, 16 warps = 4×4 tiles of a 64×64 new×old (and new×new) Gram matrix via
  `wmma` 16×16×16 fp16 → fp32; distance = ‖a‖²+‖b‖²−2a·b with precomputed norms; per row / column the minimum
  is reduced with two `__shfl_down` steps inside each 4-lane group and **only the per-row best candidate** is pushed
  (`atomicAdd` into `reverse_graph` / `reverse_distance` with RESERVENUM=128 bound). Relies on the sm_80 accumulator
  element order (fixed for sm_70 by a shared-memory re-layout).
- `nn_descent_opt_merge` 1043–1163: `<<<N, (32,3)>>>`, merges the pushed candidates into the sorted K list:
  **reject against the current k-th distance (`min_ele`) first**, bitonic sort of the survivors, binary insertion.
- Phase 2 `sample_kernel6` 178–404: `<<<N, (32,16)>>>`, per node: neighbours-of-neighbours from graph + reverse
  graph, dedup via `bitonic_sort_id_new2`, fp16 `__hfma2` distances (half4 loads), binary insertion;
  `merge_reverse_plus` 420–513 folds reverse lists in.
- Our counterpart: **Step 6** (`bucket.cu::build_vector_knn_with_tensorcore` 4702–5964 out-of-core, resident loop
  3586–4560): we compute the full bucket × pool distance block (fp16 tensor-core GEMM, fp32 accumulate, 8-bit values
  exact) and select top-64 with `block_topK_from_dots_kernel` (3226) — exact within the pool. TAGORE never materializes
  a block larger than 64×64 per node. Kernel-level tricks that transfer (implementation): precomputed norms (we have),
  k-th threshold reject before insertion (we have since #48/#62), per-row-min push (not applicable: we need top-64).
- As a refinement stage = **A1 (approved)**.

### CFS pruning, CAGRA mode (`Pruning(..., 'CAGRA')`, `pybind.cu` 154–262)
- `select_1hop_cagra` 1787–1798 = `collect_1hop` 1660–1667 + `filter_cagra` 1668–1712: `<<<N, (32,4)>>>`, shared
  `nei0/nei1/nei2/detour_num[K_SIZE=96]`; detour count by rank (for i<j: j is detoured if i is in j's list ahead of …),
  `bitonic_sort_id_by_detour`; first d kept in `new_list`; each kept edge pushed as reverse candidate
  (`atomicAdd(reverse_num)`, `reverse_graph[v*2d + 2*slot] = (u, detour)`, overflow slot `tmp % d` replaced if smaller
  detour → bounded, lossy).
- `filter_reverse_1hop` 1812–1888: per node, sort reverse candidates by detour, drop duplicates, keep forward d/2, fill
  the rest with reverse ids not already present, then forward d/2…d. Final rows written with stride d.
  Produces ~440K duplicate edges on SIFT1M (the "not already present" check only covers part of the list).
- Our counterpart: resident `optimize.cu` (cuVS-style prune + reverse); large path `prune_windowed` (GPU chunked
  prune, 150 s) + **`optimize_chunked -m E` (280 s: disk edge-chunk accumulator for reverse edges, `acc.finish` 90.7 s,
  Phase 2 110 s)**. Reverse edges on the GPU with a bounded per-node list = algorithm (TAGORE drops overflow); an
  unbounded GPU reverse pass with the same rule = implementation (backlog B6).

### Large-scale pipeline (`pybind.cu::largeIndex` 266–388, `src/large_index.cu`)
- Clusters: each point in its 2 nearest centers; clusters > `max_points` split round-robin (but the split ids have no
  data files → we do the split in prep). `order_merge_main` 86–108 + `reorder` 9–85 (Algorithm 4): greedy order that
  maximizes the overlap (shared points) with the clusters in the buffer, and picks buffer slots (`graph_position`).
- `gpu_construct` 130–285: device buffers sized once for `max_points` (data fp32 + fp16, graph K, reverse 128, dists,
  hybrid list: ≈ 2.9 KB/point); per cluster: read file (u32 count + floats), one H2D, centroid on host, GNN-Descent 8+8,
  `cal_ep_gpu` + `select_path` + `filter_reverse` (Vamana α prune), wait for the slot to be free (`has_merged`), one D2H
  into `graph_pool[slot]`, map local → global ids on the host.
- `order_merge_new` 286–522: CPU thread; for a point seen in an earlier buffered cluster: union of both pruned lists
  (dedup by `node_set`), truncate to `max_degree`; evicts a buffered cluster's local graph to disk (100 MB cached
  writer) before its slot is reused; final pass streams all local files once → DiskANN index (1 GB cached writer).
- Our counterpart: out-of-core Step 6 GPU/host bucket caches (Belady, DiskJoin order), Step 4/4.5 writers,
  ChunkedKnnAccumulator merge. Transferable as implementation: big contiguous transfers into fixed slots, overlap of
  GPU compute and host I/O, large-buffer writers. Changing our bucket granularity / order = algorithm (A2).

### Small tricks (`include/common.cuh`, `pybind.cu`)
- Compile-time shapes `K_SIZE 96`, `DIM_SIZE 128`, `TOPM_SIZE 64`, `SAMPLE 32`, `RESERVENUM 128` sized shared arrays;
  runtime K ≤ K_SIZE works (k=64 tested). Power-of-10 `norm_factor` before fp16. We already specialize on D=128 / 8-bit
  (fp16 exact); nothing new except for KMeans|| (B4).

## Ranked ideas (SIFT100M saving)

Implementation → log **Ideas backlog**; algorithm → **Awaiting approval**. Updated after each iteration.

1. B1 — Step 6 GPU-cache misses in batches: fixed device slab + pinned staging, one H2D per position batch, ids
   conversion on the GPU (TAGORE `gpu_construct` fixed buffers + one big copy). Target upload 479 / 536 s per round.
2. B2 — GPU bucket cache budget vs dots slots (resource parameters; TAGORE `buffer_size` / `max_points` sizing).
   Target the ~20 uploads per bucket.
3. B3 — host side of the miss path: hold the round's bucket records in RAM (raise bucket2's host budget so the host
   Belady cache / blob holds 13.6 GB) within the 1.2× RSS guard. Target cache_get − upload ≈ 230 / 250 s per round.
4. B4 — KMeans|| d(x, C): fused tensor-core GEMM + row-min with precomputed norms (TAGORE `nn_descent_opt_cal` +
   `cal_power`; cuVS-loop patch `iter73`), maybe with `iter72`. Target Step 2 368 s.
5. B5 — Step 4.5: overlap `pass3_write` (67 s) with `resample` (92 s) through a background writer (TAGORE
   `cached_ofstream` / CPU merge thread).
6. B6 — `optimize_chunked -m E`: reverse edges collected on the GPU per chunk (TAGORE `filter_cagra` reverse push +
   `filter_reverse_1hop`), unbounded, same rule as now; skip the disk edge accumulator when a chunk fits. Target
   `acc.finish` 90.7 + Phase 1 52.8 s.

## Iteration notes

- **#1 (accepted, +7.3% SIFT100M):** batching the GPU-cache misses of a position (pinned staging in `d_buf` layout,
  one H2D + scatter kernel) cut the per-miss API cost: SIFT100M upload 479 / 536 → 296 / 386 s per round. The copy
  calls are now negligible (17 s); what remains per miss is the host Belady get (139 / 201 s: host cache 8.6 GB <
  13.6 GB of records), `cudaMallocAsync` (67 / 75 s) + eviction `cudaFreeAsync`, and ~20 misses per bucket. TAGORE's
  lesson holds: allocate once, move big blocks. Next: B1b (device arena), then B2 / B3 for the miss count and host get.
- **#2 (accepted, +15.8% SIFT100M → 2506.9 s):** deferred eviction frees (no producer stall on unsubmitted
  positions) + capped per-size free lists for device blocks: Step 6 1401 → 1017 s. A fixed arena fragmented (22%
  fallback allocations) — capped free lists over `cudaMallocAsync` work better. GPU peak 15.5 GB (watch). Left per
  round: host get 139 / 192 s, Belady + drain ≈ 120 s, ~21M misses (≈21 per bucket). Next candidates: B1c / B3 (host
  get), B2 (miss count, but GPU memory is tight now), A2 (approved, params only: bigger buckets → fewer, larger
  transfers and fewer pool entries; directly reduces per-bucket overheads), B4 (Step 2 371 s).
- **#3 (accepted, +33.5% SIFT100M → 1666.8 s, faster than TAGORE 2151 s):** A2 params 0.004 / 80 (avg bucket 250)
  + Step 6 row tiling of the dots buffer (fixed 256-row tile). Biggest wins: Step 2 (KMeans|| scales with the
  centroid count), Step 3, Step 6 per-bucket overheads. Cost: pre-prune recall 0.72 → 0.66 at 100M, final DiskANN
  −0.013 @ L32. Round 2 Step 6 loop is 302 s vs 163 s in round 1 — check why (stream_sync / producer). Remaining big
  stages: Step 6 477, optimize_chunked 355, Step 4 232, prune 146, Step 4.5 133, Step 2 122.
- **#4 (accepted, +13.9% SIFT100M → 1434.6 s):** the disk (258 MB/s direct) is the bottleneck once Step 6 is fast:
  round 2's scatter was blocked by writeback of the round-1 running files. Not writing running before the final merge
  (TAGORE: one final merge pass) removed it. Variable-length arrival records save only ~15% (most rows keep many valid
  candidates under the running threshold). User ideas B7–B9 (change ratio, sparse rewrite, generalized deferral) are
  for ITERATIONS ≥ 3. Remaining SIFT100M: Step 6 335, optimize_chunked 293, Step 4 221, prune 141, Step 4.5 132,
  Step 2 128, Step 1 72, translate 58.
- **#5 (accepted by user decision, +6.8%):** `EdgeChunkAccumulator::finish` copy removed (−100 s in optimize_chunked,
  byte-identical) but single-run SIFT100M noise (Step 6 ±10 s) keeps it under the 7% gate. Bundle the patch with the
  next optimize_chunked change (Phase 2 111 s, Phase 1 48 s, create 12 s) so the pair clears 7%.
- **#6 (rejected, −9.5% SIFT100M):** Phase 2 CSR radix build 53 → 11 s works, but that run read the edge / forward files
  at disk speed (124 + 49 s). Unclear whether the no-zero-fill allocator or the environment caused it; retry the CSR
  part alone. SIFT100M single runs swing by tens of seconds through the page cache — the main noise source now.
- **#7 (rejected, +2.1%):** CSR part alone gives optimize_chunked −33 s (Phase 2 116 → 82 s) with reads at page-cache
  speed → #6's slow reads came from leaving the *read* buffers uninitialized. Next for optimize_chunked: B10 (user plan,
  bounded streaming top-K_out per destination; exact by the protected-set argument), measured first (step 1).
- **#8 (rejected, +2.6%):** 6 slots fix round 1 of Step 6 (stream_sync −34 s) but round 2 is producer-bound (host get
  ~63 s, wait_producer 103 s). Next Step 6 step must be the host side (B1c / B3: records in one pinned arena / RAM),
  then re-apply the slot patch. Single-run gains of 2–3% are below the gates — the remaining easy wins are that size.
- **#9 (possible gain +6.9%):** host cache for a whole round removes most of the host get (47/63 → 20/23 s); with 6 slots
  Step 6 348 → 263 s. Bundle with #7 next.
- **#10 (accepted, +7.7% → 1233.4 s):** #9 + #7 bundled. Remaining SIFT100M: bucket2 875 (Step 6 268, Step 4 218,
  Step 4.5 135, Step 2 128, Step 1 72, wait 27), optimize_chunked 160, prune 141, translate 58. Next candidates: Step 4
  (search 61 / 42 s, BucketWrite 68 s), Step 4.5 (pass3_write 57 + resample 68 s overlap — B5), B10 (memory, SIFT1B).
- **#11 (rejected by rule, flagged):** sampled u8bin reader: bucket2 875 → 725 s, RSS 77 → 42 GB. Masked by optimize_chunked
  Phase 2 reading edges from disk (150 s vs 33 s). The optimize_chunked page-cache dependence is now the dominant noise
  source → fix it next (B10).
- **#12 (accepted, +11.2% → 1094.8 s, 0.51× TAGORE):** reverse edges kept as a bounded exact top-K_out per destination in
  RAM (TAGORE's bounded reverse list, made exact by the protected-set argument) — optimize_chunked 160 → 63 s, RSS 79 →
  38 GB, page-cache noise gone. Next: re-apply #11 (sampled u8bin reader, bucket2 −150 s, RSS −35 GB) now that the noise
  source is gone; then Step 4 (219 s), Step 6 (235 s), prune (140 s), Step 4.5 (132 s).
- **#13 (accepted, +8.8% → 998.6 s, 0.46× TAGORE):** #11 re-applied. SIFT100M now: Step 6 245, Step 4 220, prune 138,
  Step 2 127, optimize_chunked 78, Step 4.5 72, translate 47, wait 30, Step 3 14, Step 1 12. RSS 40 GB.
- **#15 (rejected):** prune reload (10–12 × N rows) is dominated by window overlap between chunks (~19.5M rows per chunk
  union), not the chunk cap; bigger chunks cost memory (guard is per stage). Ideas for prune: process positions in an
  order that reuses the loaded window (sliding window with incremental loads, like DiskJoin / TAGORE `reorder` buffer
  reuse) — i.e. load only the buckets that enter the window. #14 (+6%) still waiting for a bundle partner.
- **#16 (accepted, +10.0% → 898.4 s, 0.42× TAGORE):** #14 + prune device row reuse across chunks. Remaining SIFT100M:
  bucket2 676 (Step 6 241, Step 4 178, Step 2 127, Step 4.5 70, wait 25, Step 3 14, Step 1 11), prune 110 (reload still
  9.8 × N → order chunks / windows for more reuse), optimize_chunked 64, translate 48. RSS 38 GB (guard 45.7 GB).
- **#18 (accepted, +9.7% → 810.9 s, 0.38× TAGORE):** k-means fused kernel (#17) + Step 4.5 windowed permute. Remaining
  SIFT100M: bucket2 578 (Step 6 238, Step 4 176, Step 2 76, Step 4.5 31, wait 26, Step 1 11, Step 3 9), prune 111,
  optimize_chunked 66, translate 55. bucket2 RSS 43.2 GB (guard now 51.9 GB). Next: Step 4 (search 61 / 42 s, reads 12 / 13,
  BucketWrite 28), Step 6 (loop ~116 / 114 s), translate 55 s.
