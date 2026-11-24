#!/bin/bash
# exp2 main result: cursor-resume vs segment-restart (BUCKET2_SPOT_RECOVERY), no-event vs replayed-trace.
#
#                          | no events (spot scheduler on) | replay TRACE
#   cursor resume          |               A                |      B
#   segment restart        |               C                |      D
#
# Headline number: D - B (same trace, how much sooner cursor-resume finishes). Also reports B - A and D - C
# (how much the trace itself costs each strategy) -- if the implementation only changes where a revoked lease's
# work gets re-queued, A and C should agree within noise (see spot_sched.hpp Options::Recovery comment).
#   spot_recovery_matrix.sh DATASET G TRACE [REPS] [WORKROOT]
#     DATASET sift100M, G 4 (use G 2 only to debug counters/output before committing to the G=4 main table)
#     TRACE   traces/t1_g${G}.csv for the headline table; traces/d1_g${G}.csv for the stress pair (REPS=1 is
#             enough to diagnose; pass REPS=3 if a d1 speedup number is going in the paper, same as t1)
# Env: SPOT_SCALE, SPOT_NOTICE_REAL_S, BUCKET2_SPOT_MODE pass through to spot_build.sh.
#      CELLS=ABCD (default): which cells to run, e.g. CELLS=BD for a d1 stress pair that reuses the t1 run's
#      A/C (no-event) cells instead of re-measuring them.
# Prints one line per rep (so raw values are kept, not just the median) then the A/B/C/D medians, D-B, B-A,
# D-C, and the base commit SHA. Does not touch state/best_spot.json -- this is an ablation, not a loop candidate.
set -uo pipefail
DS=$1; G=$2; TRACE=$3; REPS=${4:-3}
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$T/../../../.." && pwd)
W=${5:-$ROOT/bucketDemo/buildBucket/runs/spot_matrix_$$}
mkdir -p "$W"
SHA=$(git -C "$ROOT" rev-parse HEAD)
echo "base commit: $SHA"
echo "dataset=$DS G=$G trace=$TRACE reps=$REPS"

GPU_LIST=""   # first successful run pins devices for the rest (fair comparison, same physical GPUs throughout)
run_cell() {   # run_cell CELL RECOVERY RUNTRACE  (RUNTRACE: "none" or $TRACE)  -> appends to $W/$CELL.tsv
  local cell=$1 recov=$2 runtrace=$3
  for ((r = 1; r <= REPS; r++)); do
    local d=$W/${cell}_r$r
    local extra_gpu=(); [[ -n $GPU_LIST ]] && extra_gpu=(GPU_LIST="$GPU_LIST")
    env "${extra_gpu[@]}" BUCKET2_SPOT_RECOVERY=$recov bash $T/spot_build.sh $DS $G "$d" "$runtrace" \
      | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
    if [[ ! -f $d/optimize_chunked.time ]]; then echo "$cell rep$r FAILED" >&2; continue; fi
    [[ -z $GPU_LIST ]] && GPU_LIST=$(cat "$d/gpus.txt")
    local total=0
    for n in bucket2 prune optimize_chunked; do
      [[ -f $d/$n.time ]] && total=$(awk -v a="$total" -v b="$(head -1 "$d/$n.time" | cut -d' ' -f1)" 'BEGIN{print a+b}')
    done
    local pre steals redone ev_pre
    pre=$(grep -h '^\[spot\] stage=' $d/bucket2.log $d/prune.log 2>/dev/null | grep -o 'preempts=[0-9]*' | cut -d= -f2 | awk '{s+=$1} END {print s+0}')
    steals=$(grep -h '^\[spot\] stage=' $d/bucket2.log $d/prune.log 2>/dev/null | grep -o 'steals=[0-9]*' | cut -d= -f2 | awk '{s+=$1} END {print s+0}')
    redone=$(grep -h '^\[spot\] stage=' $d/bucket2.log $d/prune.log 2>/dev/null | grep -o 'redone_units=[0-9]*' | cut -d= -f2 | awk '{s+=$1} END {print s+0}')
    ev_pre=$([[ -f $d/spot_events.log ]] && grep -c ' preempt ' $d/spot_events.log || echo 0)
    echo -e "$cell\t$r\t$total\t$pre\t$steals\t$redone\t$ev_pre" >> "$W/table.tsv"
    echo "  $cell rep$r: total=${total}s scheduler_preempts=$pre steals=$steals redone_units=$redone trace_preempt_events=$ev_pre"
  done
}

CELLS=${CELLS:-ABCD}
echo -e "cell\trep\ttotal_s\tscheduler_preempts\tsteals\tredone_units\ttrace_preempt_events" > "$W/table.tsv"
[[ $CELLS == *A* ]] && { echo "=== A: cursor-resume, no events ==="; run_cell A cursor none; }
[[ $CELLS == *B* ]] && { echo "=== B: cursor-resume, trace $TRACE ==="; run_cell B cursor "$TRACE"; }
[[ $CELLS == *C* ]] && { echo "=== C: segment-restart, no events ==="; run_cell C restart none; }
[[ $CELLS == *D* ]] && { echo "=== D: segment-restart, trace $TRACE ==="; run_cell D restart "$TRACE"; }

echo; echo "=== raw (also at $W/table.tsv -- copy it out before the next run overwrites \$W) ==="
column -t "$W/table.tsv"

python3 - "$W/table.tsv" <<'PYEOF'
import sys, statistics as st, csv
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
by = {c: [float(r["total_s"]) for r in rows if r["cell"] == c] for c in "ABCD"}
med = {c: (round(st.median(v), 3) if v else None) for c, v in by.items()}
print("medians (s):", "  ".join(f"{c}={med[c]}" for c in "ABCD"), f"(n={ {c: len(by[c]) for c in 'ABCD'} })")
a, b, c, d = med["A"], med["B"], med["C"], med["D"]
# D - B > 0 means segment-restart (D) took longer than cursor-resume (B) under the same trace, i.e.
# cursor-resume finished D-B seconds sooner (the headline number from the user's own table definition).
deltas = [("D - B (headline: cursor-resume finishes this much sooner under the same trace)", d, b),
          ("B - A (trace cost under cursor-resume)", b, a),
          ("D - C (trace cost under segment-restart)", d, c),
          ("C - A (no-event sanity check, should be ~0 if only the requeue point changed)", c, a)]
for label, x, y in deltas:
    print(f"{label} = {'n/a (missing cell)' if None in (x, y) else f'{x - y:+.3f} s'}")
PYEOF
echo
echo "raw table kept at: $W/table.tsv"
