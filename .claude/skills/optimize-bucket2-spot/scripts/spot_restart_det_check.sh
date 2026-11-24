#!/bin/bash
# exp2 (segment-restart) byte-level correctness check. Extends spot_det.sh's plain-vs-spot comparison with a
# THIRD build under BUCKET2_SPOT_RECOVERY=restart, and requires the race spot_sched.hpp's segment-restart code
# exists for (steal a suffix off a lease, then preempt the donor while it still owns an unfinished remainder,
# so its own in-flight/late commit must be rejected and its [lease_begin, cursor) prefix gets replayed) to have
# actually been hit at least once, not just "no mismatch because nothing interesting happened".
#   spot_restart_det_check.sh DATASET G TRACE [WORKROOT]     (DATASET sift10M, G 2, TRACE traces/d1_g2.csv recommended:
#                                                              d1 is the dense/correctness trace -- see SPOT_DESIGN.md)
# Env: SPOT_SCALE (default 60), SPOT_NOTICE_REAL_S, BUCKET2_SPOT_MODE pass through (same as spot_det.sh).
# Exit 0 = both spot builds (cursor-resume, segment-restart) are byte-identical to the plain deterministic build
#          AND the segment-restart build actually redid something (redone_units > 0 in its scheduler stats) AND
#          at least one steal happened in that same run (so the redo plausibly followed a steal, not just a
#          preemption of an untouched segment -- the exact race in the user's Q3 example). Deletes its work dirs.
set -uo pipefail
DS=$1; G=$2; TRACE=$3
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$T/../../../.." && pwd)
W=${4:-$ROOT/bucketDemo/buildBucket/runs/spot_restart_det_$$}

export BUCKET2_ASSIGN_DETERMINISTIC=1 BUCKET2_STEP4_CHECKSUM=1

echo "--- plain (no spot) ---"
bash $T/spot_build.sh $DS $G $W/plain | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
[[ -f $W/plain/optimize_chunked.time ]] || { echo "MISMATCH (plain build failed)"; rm -rf $W; exit 1; }
GPUS=$(cat $W/plain/gpus.txt)

echo "--- spot, BUCKET2_SPOT_RECOVERY=cursor (default) ---"
GPU_LIST=$GPUS BUCKET2_SPOT_RECOVERY=cursor bash $T/spot_build.sh $DS $G $W/cursor $TRACE \
  | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
[[ -f $W/cursor/optimize_chunked.time ]] || { echo "MISMATCH (cursor-resume build failed)"; tail -20 $W/cursor/*.log; rm -rf $W; exit 1; }

echo "--- spot, BUCKET2_SPOT_RECOVERY=restart ---"
GPU_LIST=$GPUS BUCKET2_SPOT_RECOVERY=restart bash $T/spot_build.sh $DS $G $W/restart $TRACE \
  | tee >(grep --line-buffered GPU_WAIT >&2) | tail -1
[[ -f $W/restart/optimize_chunked.time ]] || { echo "MISMATCH (segment-restart build failed)"; tail -20 $W/restart/*.log; rm -rf $W; exit 1; }

a=$(ls -d $W/plain/work/k*/ | head -1)
ok=1
for tag in cursor restart; do
  b=$(ls -d $W/$tag/work/k*/ | head -1)
  for f in inverse_perm.bin bucket_offsets.bin vector_knn.bin forward_graph_reordered.bin final_graph.bin; do
    if [[ ! -f $a/$f && ! -f $b/$f ]]; then continue; fi
    if cmp -s $a/$f $b/$f; then echo "$tag $f same"; else echo "$tag $f DIFF"; ok=0; fi
  done
done

stats() {   # stats TAG KEY -> summed int across bucket2.log + prune.log's "[spot] stage=..." lines
  grep -h '^\[spot\] stage=' $W/$1/bucket2.log $W/$1/prune.log 2>/dev/null \
    | grep -o "$2=[0-9]*" | cut -d= -f2 | awk '{s+=$1} END {print s+0}'
}
cursor_pre=$(stats cursor preempts); cursor_steals=$(stats cursor steals); cursor_redone=$(stats cursor redone_units)
restart_pre=$(stats restart preempts); restart_steals=$(stats restart steals); restart_redone=$(stats restart redone_units)
echo "cursor:  preempts=$cursor_pre steals=$cursor_steals redone_units=$cursor_redone (must be 0: cursor-resume never suppresses a commit)"
echo "restart: preempts=$restart_pre steals=$restart_steals redone_units=$restart_redone"
grep -h '^\[spot\] stage=' $W/restart/bucket2.log $W/restart/prune.log 2>/dev/null

for x in plain cursor restart; do echo "$x times: $(for n in bucket2 prune optimize_chunked; do [[ -f $W/$x/$n.time ]] && echo -n "$n=$(head -1 $W/$x/$n.time | cut -d' ' -f1) "; done)"; done
rm -rf $W

reason=""
(( ok )) || reason="byte mismatch"
[[ -z $reason && ${cursor_redone:-0} -ne 0 ]] && reason="cursor-resume redone_units != 0 (should be impossible -- commit() suppression must be gated on recovery mode)"
[[ -z $reason && ${restart_pre:-0} -eq 0 ]] && reason="vacuous: no preemption hit a spot stage in the restart run; use a denser trace / larger SPOT_SCALE"
[[ -z $reason && ${restart_steals:-0} -eq 0 ]] && reason="vacuous: no steal happened in the restart run -- this only exercises plain preemption-of-an-untouched-lease, not the Q3 steal-then-preempt-donor race; use a denser trace or more GPUs"
[[ -z $reason && ${restart_redone:-0} -eq 0 ]] && reason="vacuous: segment-restart never replayed anything (redone_units == 0); the mechanism was not exercised"
if [[ -n $reason ]]; then echo "MISMATCH-VACUOUS ($DS, G=$G, trace=$TRACE): $reason"; exit 2; fi
echo "EQUAL ($DS, G=$G, spot $TRACE, cursor-resume and segment-restart both byte-identical to plain; restart redid $restart_redone units across $restart_pre preemptions / $restart_steals steals)"
exit 0
