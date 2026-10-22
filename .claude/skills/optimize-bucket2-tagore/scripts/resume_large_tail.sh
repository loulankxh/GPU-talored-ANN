#!/usr/bin/env bash
# Re-run only the tail of a large-path run_pipeline.sh rep whose work dir was kept after a
# failure in optimize_chunked / translate / search (bucket2 + prune .time files are reused).
#   resume_large_tail.sh -d sift100M -o OUTDIR      (rep0 only)
set -uo pipefail
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$SKILL/../../.." && pwd)"; BB=$ROOT/bucketDemo/buildBucket; BIN=$BB/build
PY=/home/lanlu/miniconda3/envs/cuvs-bench/bin/python
source "$SKILL/scripts/params.sh"
while getopts "d:o:" o; do case $o in d) DS=$OPTARG ;; o) OUT=$OPTARG ;; esac; done
[[ $DS == sift100M ]] && ITERATIONS=2
BASE=$(ls /home/lanlu/microBenchmark/dataset/$DS/base.*.u8bin); DSDIR=$(dirname "$BASE")
RD=$OUT/rep0; W=$RD/work; O=$W/k${KNN_K}p${NPROBE}m${NEIGHBORS_M}t${ITERATIONS}
# same bench lock as run_pipeline.sh (timed run)
BENCH_LOCK=/home/lanlu/.bench_locks/bucket2_bench.lock; mkdir -p "$(dirname "$BENCH_LOCK")"
exec 9>>"$BENCH_LOCK"
flock -n 9 || { echo "[bench lock] busy, waiting: $(cat "$BENCH_LOCK.owner" 2>/dev/null)"; flock 9; }
echo "$$ $(date +%F_%T) $SKILL resume $DS -> $OUT" > "$BENCH_LOCK.owner"
export CUDA_VISIBLE_DEVICES=0
timed() { local name=$1; shift; /usr/bin/time -f "%e %M" -o "$RD/$name.time" stdbuf -oL -eL "$@" > "$RD/$name.log" 2>&1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 200 >> "$RD/gpu_mem.txt" 9>&- & SMI=$!
rm -rf "$W/tmp"
timed optimize_chunked "$BIN/optimize_chunked" -g "$O/forward_graph_reordered.bin" \
  -o "$O/graph_new_ids.bin" -m E --bucket-offsets "$O/bucket_offsets.bin" \
  --gpu-budget-mb "$CHUNKED_GPU_BUDGET_MB" --edge-buffer-mb "$CHUNKED_EDGE_BUFFER_MB" \
  --tmp-dir "$W/tmp" $CHUNKED_EXTRA || { echo "optimize_chunked FAILED"; kill $SMI; exit 3; }
timed translate "$BIN/translate_ids" -g "$O/graph_new_ids.bin" -p "$O/inverse_perm.bin" \
  -o "$O/final_graph.bin" || { echo "translate FAILED"; kill $SMI; exit 3; }
kill $SMI; wait $SMI 2>/dev/null
echo large > "$RD/path.txt"
"$PY" -I "$SKILL/scripts/knn_similarity.py" "$O/vector_knn.bin" "$O/inverse_perm.bin" \
  "$DSDIR/base_knn_sample10K_k64.ibin" > "$RD/knn_similarity.json" || echo "knn_similarity FAILED"
"$PY" -I "$SKILL/scripts/diskann_search.py" "$BASE" "$O/final_graph.bin" \
  "$DSDIR/query.public.10K.u8bin" "$DSDIR/groundtruth.ibin" "$W/diskann" \
  2> "$RD/search.err" | grep '^{' > "$RD/search.json" || echo "search FAILED (see search.err)"
rm -rf "$W"
"$PY" -I "$SKILL/scripts/summarize.py" "$OUT" "$DS" auto > "$OUT/metrics.json"
cat "$OUT/metrics.json"
