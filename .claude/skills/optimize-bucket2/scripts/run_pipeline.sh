#!/usr/bin/env bash
# One full build of the bucket pipeline + quality eval, on GPU 0.
#
#   run_pipeline.sh -d sift1M|sift10M [-m auto|large] [-r REPS] [-o OUTDIR] [-q] [-k]
#     -m auto  : bucket2 picks GPU-resident vs out-of-core itself; optimize (cuVS
#                resident prune+reverse) runs if it fits, else falls back to
#                prune_windowed + optimize_chunked -m E.            (default)
#     -m large : force the large-dataset path end to end
#                (bucket2 --out-of-core, prune_windowed, optimize_chunked -m E).
#     -r REPS  : full build repetitions (median is reported). default 3 (sift10M: 1)
#     -q       : skip quality eval (timing only)
#     -k       : keep the big intermediate files (default: delete after eval)
#
# Build time = sum of wall-clock of bucket2 + (optimize | prune_windowed +
# optimize_chunked) + translate_ids, each measured with /usr/bin/time
# (process start-up, all I/O included). Writes OUTDIR/metrics.json.
set -uo pipefail

SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=/home/lanlu/microBenchmark
BB=$ROOT/bucketDemo/buildBucket
BIN=$BB/build
EVAL=$ROOT/bucketDemo/evalCuvs/build/cagra_eval
PY=/home/lanlu/miniconda3/envs/cuvs-bench/bin/python
source "$SKILL/scripts/params.sh"

DS=""; MODE=auto; REPS=""; OUT=""; QUALITY=1; KEEP=0
while getopts "d:m:r:o:qk" o; do
  case $o in
    d) DS=$OPTARG ;; m) MODE=$OPTARG ;; r) REPS=$OPTARG ;; o) OUT=$OPTARG ;;
    q) QUALITY=0 ;; k) KEEP=1 ;; *) sed -n 2,17p "$0"; exit 1 ;;
  esac
done
case $DS in
  sift1M)  BASE=$ROOT/dataset/sift1M/base.1M.u8bin;   : "${REPS:=3}" ;;
  sift10M) BASE=$ROOT/dataset/sift10M/base.10M.u8bin; : "${REPS:=1}" ;;
  *) echo "-d sift1M|sift10M required"; exit 1 ;;
esac
DSDIR=$(dirname "$BASE")
: "${OUT:=$BB/runs/$(date +%Y%m%d_%H%M%S)_${DS}_${MODE}}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0

TAG="k${KNN_K}p${NPROBE}m${NEIGHBORS_M}t${ITERATIONS}"
timed() {  # timed NAME CMD... -> $OUT/rep$R/NAME.{log,time}; returns cmd exit code
  local name=$1; shift
  /usr/bin/time -f "%e %M" -o "$RD/$name.time" "$@" > "$RD/$name.log" 2>&1
}

for ((R = 0; R < REPS; R++)); do
  RD=$OUT/rep$R; W=$RD/work; mkdir -p "$W"
  echo "[$DS/$MODE rep $R] -> $RD"
  # GPU-0 memory sampler (MiB), for the peak-memory guard
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 200 > "$RD/gpu_mem.txt" &
  SMI=$!
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

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
    timed optimize "$BIN/optimize" -i "$O/data_reordered.u8bin" -g "$O/vector_knn.bin" \
      -o "$O/graph_new_ids.bin" --output-degree "$OUTPUT_DEGREE" $OPTIMIZE_EXTRA
    rc=$?
    if [[ $rc -eq 2 ]]; then PATH_USED=large; rm -f "$RD/optimize.time"
    elif [[ $rc -ne 0 ]]; then echo "optimize FAILED"; kill $SMI; exit 3; fi
  else
    PATH_USED=large
  fi
  if [[ $PATH_USED == large ]]; then
    # shellcheck disable=SC2086
    timed prune "$BIN/prune_windowed" -k "$O/vector_knn.bin" -i "$O/data_reordered.u8bin" \
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
    # final-graph structural check + recall/QPS through the cuVS CAGRA searcher
    "$EVAL" search --data "$BASE" --graph "$O/final_graph.bin" \
      --query "$DSDIR/query.public.10K.u8bin" --gt "$DSDIR/groundtruth.ibin" \
      2> "$RD/search.err" | grep '^{' > "$RD/search.json" || echo "search FAILED (see search.err)"
  fi
  [[ $KEEP -eq 1 ]] || rm -rf "$W"
done

"$PY" -I "$SKILL/scripts/summarize.py" "$OUT" "$DS" "$MODE" > "$OUT/metrics.json"
cat "$OUT/metrics.json"
