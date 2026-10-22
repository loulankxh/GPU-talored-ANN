#!/bin/bash
# Step B helper: one timed TAGORE run on GPU 0 under the shared bench lock (same rules as run_pipeline.sh:
# page cache dropped, GPU-0 memory sampled every 200 ms, wall clock + max RSS via /usr/bin/time).
# usage: tagore_timed.sh OUTDIR CMD...   -> OUTDIR/{log,time,gpu_mem.txt}
set -u
OUT=$1; shift; mkdir -p "$OUT"
BENCH_LOCK=/home/lanlu/.bench_locks/bucket2_bench.lock; mkdir -p "$(dirname "$BENCH_LOCK")"
exec 9>>"$BENCH_LOCK"
flock -n 9 || { echo "[bench lock] busy, waiting: $(cat "$BENCH_LOCK.owner" 2>/dev/null)"; flock 9; }
echo "$$ $(date +%F_%T) tagore step B -> $OUT" > "$BENCH_LOCK.owner"
export CUDA_VISIBLE_DEVICES=0
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -lms 200 > "$OUT/gpu_mem.txt" 9>&- & SMI=$!
/usr/bin/time -f "%e %M" -o "$OUT/time" "$@" > "$OUT/log" 2>&1; rc=$?
kill $SMI; wait $SMI 2>/dev/null
echo "rc=$rc wall/maxrss_kb=$(cat "$OUT/time") gpu_peak_mib=$(sort -n "$OUT/gpu_mem.txt" | tail -1)"
exit $rc
