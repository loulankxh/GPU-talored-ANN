#!/usr/bin/env bash
# CAGRA 建图时间 benchmark：IVF-PQ vs NN-Descent，多个 graph degree。
#
# 用法:
#   ./run_build_bench.sh [选项]
#     -d DATA      .u8bin 数据集      (默认 dataset/sift1M/base.1M.u8bin)
#     -g "32 64"   graph degree 列表  (默认 "32 64")
#     -a "ivf_pq nn_descent"  算法列表 (默认两个都跑)
#     -r REPS      计时重复次数       (默认 3)
#     -w WARMUP    预热次数           (默认 1)
#     -G GPU       CUDA_VISIBLE_DEVICES (默认 0)
#     -n NITER     NN-Descent 迭代数  (默认 20)
#     -o OUTDIR    结果目录           (默认 ./results)
#     -B           跳过重新编译
# intermediate degree 固定为 2*graph_degree（与 raft-ann-bench 默认一致）。
# 计时 = knn 建图 + optimize + index 构造，不含读数据和写文件。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin
DEGREES="32 64"
ALGOS="ivf_pq nn_descent"
REPS=3
WARMUP=1
GPU=0
NITER=20
OUTDIR="$HERE/results"
SKIP_BUILD=0

while getopts "d:g:a:r:w:G:n:o:B" opt; do
  case $opt in
    d) DATA=$OPTARG ;;
    g) DEGREES=$OPTARG ;;
    a) ALGOS=$OPTARG ;;
    r) REPS=$OPTARG ;;
    w) WARMUP=$OPTARG ;;
    G) GPU=$OPTARG ;;
    n) NITER=$OPTARG ;;
    o) OUTDIR=$OPTARG ;;
    B) SKIP_BUILD=1 ;;
    *) sed -n 2,16p "$0"; exit 1 ;;
  esac
done

ENV=/home/lanlu/miniconda3/envs/rapids_raft
export CONDA_PREFIX=${CONDA_PREFIX:-$ENV}
export PATH="$ENV/bin:$PATH"

if [[ $SKIP_BUILD -eq 0 ]]; then
  cmake -S "$HERE" -B "$HERE/build" >/dev/null
  cmake --build "$HERE/build" --target build_bench -j"$(nproc)"
fi
BIN="$HERE/build/build_bench"

mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
NAME=$(basename "$(dirname "$DATA")")
CSV="$OUTDIR/${NAME}_${TS}.csv"
LOG="$OUTDIR/${NAME}_${TS}.log"
echo "raft source: $(git -C /home/lanlu/raft log -1 --format='%h %s')" | tee "$LOG"

for algo in $ALGOS; do
  for deg in $DEGREES; do
    echo -e "\n########## $algo  graph_degree=$deg ##########" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU "$BIN" --data "$DATA" --algo "$algo" \
      --graph-degree "$deg" --nnd-niter "$NITER" \
      --reps "$REPS" --warmup "$WARMUP" --csv "$CSV" 2>&1 | tee -a "$LOG"
  done
done

echo -e "\n========== 汇总 (秒, $REPS 次平均) ==========" | tee -a "$LOG"
awk -F, 'NR>1 { k=$3" "$4"/"$5; n[k]++; knn[k]+=$9; opt[k]+=$10; idx[k]+=$11; b[k]+=$12
               if (!(k in best) || $12<best[k]) best[k]=$12 }
         END { printf "%-12s %-8s %8s %9s %7s %9s %9s\n","algo","deg","knn","optimize","index","build","best"
               for (k in n) { split(k,p," ")
                 printf "%-12s %-8s %8.3f %9.3f %7.3f %9.3f %9.3f\n",p[1],p[2],knn[k]/n[k],opt[k]/n[k],idx[k]/n[k],b[k]/n[k],best[k] } }' \
  "$CSV" | sort | tee -a "$LOG"
echo "CSV: $CSV"
echo "LOG: $LOG"
