#!/usr/bin/env bash
# One full build of the bucket pipeline + quality eval, on GPU 0 (SMOKE=1: see the lock block).
#
#   run_pipeline.sh -d DATASET [-m auto|large] [-r REPS] [-o OUTDIR] [-q] [-k]
#     -d DATASET : a name from datasets.json (sift1M, sift10M, sift100M, msTuring10M, ...);
#                  vector type from the base file extension (.u8bin / .i8bin / .fbin),
#                  size-adaptive parameters from dataset_env.py (env values override them)
#     -m auto  : bucket2 picks GPU-resident vs out-of-core itself; optimize (cuVS
#                resident prune+reverse) runs if it fits, else falls back to
#                prune_windowed + optimize_chunked -m E.            (default)
#     -m large : force the large-dataset path end to end
#                (bucket2 --out-of-core, prune_windowed, optimize_chunked -m E).
#     -r REPS  : full build repetitions (median is reported). default: datasets.json "reps"
#     -q       : skip quality eval (timing only)
#     -k       : keep the big intermediate files (default: delete after eval)
#
# Build time = sum of wall-clock of bucket2 + (optimize | prune_windowed +
# optimize_chunked) + translate_ids, each measured with /usr/bin/time
# (process start-up, all I/O included). Writes OUTDIR/metrics.json.
set -uo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$SKILL/../../.." && pwd)"   # this worktree (/home/lanlu/microBenchmark_tagore)
BB=$ROOT/bucketDemo/buildBucket
BIN=$BB/build
EVAL=$ROOT/bucketDemo/evalCuvs/build/cagra_eval   # eval tool (not optimized); worktree build: uint8 / int8 / float32 (P1)
PY=/home/lanlu/miniconda3/envs/cuvs-bench/bin/python
source "$SKILL/scripts/params.sh"

DS=""; MODE=auto; REPS=""; OUT=""; QUALITY=1; KEEP=0
while getopts "d:m:r:o:qk" o; do
  case $o in
    d) DS=$OPTARG ;; m) MODE=$OPTARG ;; r) REPS=$OPTARG ;; o) OUT=$OPTARG ;;
    q) QUALITY=0 ;; k) KEEP=1 ;; *) sed -n 2,17p "$0"; exit 1 ;;
  esac
done
# 数据集查表 + 按规模自适应的参数 (dataset_env.py；环境变量里显式给的值优先，用于参数扫描)
ENV_OUT=$("$PY" -I "$SKILL/scripts/dataset_env.py" "$DS") || { echo "$ENV_OUT"; exit 1; }
eval "$ENV_OUT"
DSDIR=$DS_DIR
: "${REPS:=$DEF_REPS}"
BUCKET2_EXTRA="$BUCKET2_EXTRA --kmeans-train-fraction $KMEANS_TRAIN_FRACTION --cache-mb $CACHE_MB --bucket-mem-mb $BUCKET_MEM_MB --gpu-bucket-cache-mb $GPU_BUCKET_CACHE_MB"
CHUNKED_EXTRA="$CHUNKED_EXTRA --host-budget-mb $CHUNKED_HOST_BUDGET_MB"
export BUCKET2_A1_PASSES=$A1_PASSES   # A1-R local join passes (resident path, #40)
echo "[$DS] $DTYPE N=$N D=$DIM reps=$REPS iters=$ITERATIONS ratio=$CENTROID_RATIO nprobe=$NPROBE search=$SEARCH ($ADAPT_NOTE)"
: "${OUT:=$BB/runs/$(date +%Y%m%d_%H%M%S)_${DS}_${MODE}}"
mkdir -p "$OUT"
# 正式计时排队（user 2026-10-22）：optimize-bucket2 和 optimize-bucket2-tagore 两个 loop 的计时运行
# 共用这把锁，同一时刻只有一个 pipeline 在测时间（CPU / 磁盘 / page cache 是共用的）。
# SMOKE=1：冒烟测试，不拿锁、judge 拒收（OUTDIR/SMOKE），跑在 SMOKE_GPU（默认 2，NUMA node 1）上，
# nice 19、OMP 8 线程、只用 node 1 的核，尽量不干扰另一个 loop 正在计时的运行。
BENCH_LOCK=/home/lanlu/.bench_locks/bucket2_bench.lock
if [[ ${SMOKE:-0} == 1 ]]; then
  GPU=${SMOKE_GPU:-2}
  renice -n 19 $$ >/dev/null; taskset -pc 1-79:2 $$ >/dev/null
  export OMP_NUM_THREADS=${SMOKE_THREADS:-8}
  touch "$OUT/SMOKE"
else
  GPU=0; rm -f "$OUT/SMOKE"
  mkdir -p "$(dirname "$BENCH_LOCK")"
  exec 9>>"$BENCH_LOCK"
  if ! flock -n 9; then
    echo "[bench lock] busy, waiting: $(cat "$BENCH_LOCK.owner" 2>/dev/null)"; flock 9
  fi
  echo "$$ $(date +%F_%T) $SKILL $* -> $OUT" > "$BENCH_LOCK.owner"
fi
export CUDA_VISIBLE_DEVICES=$GPU

TAG="k${KNN_K}p${NPROBE}m${NEIGHBORS_M}t${ITERATIONS}"
timed() {  # timed NAME CMD... -> $OUT/rep$R/NAME.{log,time}; returns cmd exit code
  local name=$1; shift
  # stdbuf -oL: line-buffered logs so long (SIFT100M) runs can be followed live
  /usr/bin/time -f "%e %M" -o "$RD/$name.time" stdbuf -oL -eL "$@" > "$RD/$name.log" 2>&1
}

for ((R = 0; R < REPS; R++)); do
  RD=$OUT/rep$R; W=$RD/work; mkdir -p "$W"
  echo "[$DS/$MODE rep $R] -> $RD"
  # GPU memory sampler (MiB), for the peak-memory guard
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" -lms 200 > "$RD/gpu_mem.txt" 9>&- &
  SMI=$!
  [[ ${SMOKE:-0} == 1 ]] || { sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true; }

  B2_FLAGS=(-i "$BASE" -o "$W" --centroid-ratio "$CENTROID_RATIO" --seed "$SEED"
            --knn-k "$KNN_K" --nprobe "$NPROBE" --neighbors-m "$NEIGHBORS_M" -t "$ITERATIONS")
  [[ $MODE == large ]] && B2_FLAGS+=(--out-of-core)
  # GPU-resident path: bucket2 does prune + reverse edges + id translation in-process
  # and writes final_graph.bin itself (user decision 2026-10-21); out-of-core ignores it
  [[ $MODE == auto ]] && B2_FLAGS+=(--fused-optimize-degree "$OUTPUT_DEGREE")
  # shellcheck disable=SC2086
  timed bucket2 "$BIN/bucket2" "${B2_FLAGS[@]}" $BUCKET2_EXTRA || { echo "bucket2 FAILED"; kill $SMI; exit 3; }
  O=$W/$TAG

  PATH_USED=resident
  FUSED=0
  if [[ $MODE == auto && -f $O/final_graph.bin ]]; then
    FUSED=1   # resident path finished inside bucket2 (optimize + translate_ids folded in)
  elif [[ $MODE == auto ]]; then
    # shellcheck disable=SC2086
    timed optimize "$BIN/optimize" -i "$O/data_reordered$EXT" -g "$O/vector_knn.bin" \
      -o "$O/graph_new_ids.bin" --output-degree "$OUTPUT_DEGREE" $OPTIMIZE_EXTRA
    rc=$?
    if [[ $rc -eq 2 ]]; then PATH_USED=large; rm -f "$RD/optimize.time"
    elif [[ $rc -ne 0 ]]; then echo "optimize FAILED"; kill $SMI; exit 3; fi
  else
    PATH_USED=large
  fi
  if [[ $PATH_USED == large ]]; then
    # shellcheck disable=SC2086
    timed prune "$BIN/prune_windowed" -k "$O/vector_knn.bin" -i "$O/data_reordered$EXT" \
      --bucket-offsets "$O/bucket_offsets.bin" --bucket-process-order "$O/bucket_process_order.bin" \
      -c "$O/centroid_knn.bin" -o "$O/forward_graph_reordered.bin" \
      --output-degree "$OUTPUT_DEGREE" --cache-mb "$PRUNE_CACHE_MB" $PRUNE_EXTRA \
      || { echo "prune FAILED"; kill $SMI; exit 3; }
    # shellcheck disable=SC2086
    timed optimize_chunked "$BIN/optimize_chunked" -g "$O/forward_graph_reordered.bin" \
      -o "$O/graph_new_ids.bin" -m E --bucket-offsets "$O/bucket_offsets.bin" \
      --gpu-budget-mb "$CHUNKED_GPU_BUDGET_MB" --edge-buffer-mb "$CHUNKED_EDGE_BUFFER_MB" \
      --tmp-dir "$W/tmp" $CHUNKED_EXTRA \
      || { echo "optimize_chunked FAILED"; kill $SMI; exit 3; }
  fi
  if [[ $FUSED -eq 0 ]]; then
    timed translate "$BIN/translate_ids" -g "$O/graph_new_ids.bin" -p "$O/inverse_perm.bin" \
      -o "$O/final_graph.bin" || { echo "translate FAILED"; kill $SMI; exit 3; }
  fi
  kill $SMI; wait $SMI 2>/dev/null
  echo "$PATH_USED" > "$RD/path.txt"

  # quality only on the last rep (seeded build -> reps produce the same graph)
  if [[ $QUALITY -eq 1 && $R -eq $((REPS - 1)) ]]; then
    "$PY" -I "$SKILL/scripts/knn_similarity.py" "$O/vector_knn.bin" "$O/inverse_perm.bin" \
      "$DSDIR/base_knn_sample10K_k64.ibin" > "$RD/knn_similarity.json" || echo "knn_similarity FAILED"
    # final-graph structural check + recall/QPS: cuVS CAGRA searcher (SEARCH=cagra, uint8) or
    # DiskANN CPU in-memory search (SEARCH=diskann; same JSON format, points[].itopk = DiskANN L)
    if [[ $SEARCH == diskann ]]; then
      "$PY" -I "$SKILL/scripts/diskann_search.py" "$BASE" "$O/final_graph.bin" \
        "$DSDIR/query.public.10K$EXT" "$DSDIR/groundtruth.ibin" "$W/diskann" \
        2> "$RD/search.err" | grep '^{' > "$RD/search.json" || echo "search FAILED (see search.err)"
    else
    "$EVAL" search --data "$BASE" --graph "$O/final_graph.bin" \
      --query "$DSDIR/query.public.10K$EXT" --gt "$DSDIR/groundtruth.ibin" \
      2> "$RD/search.err" | grep '^{' > "$RD/search.json" || echo "search FAILED (see search.err)"
    fi
  fi
  [[ $KEEP -eq 1 ]] || rm -rf "$W"
done

"$PY" -I "$SKILL/scripts/summarize.py" "$OUT" "$DS" "$MODE" > "$OUT/metrics.json"
cat "$OUT/metrics.json"
