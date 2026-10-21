#!/usr/bin/env bash
# 用 rapids_raft 环境里现成的 RAFT_CAGRA_ANN_BENCH（raft-ann-bench 官方可执行文件，
# 即 /home/lanlu/raft 源码编译后 install 进环境的那份）测 CAGRA 整体建图时间。
# 不拆步骤：计时就是 ann-bench 的 build 时间（kNN 建图 + optimize + index 构造），
# 不含读数据、写 index 文件。IVF-PQ / NN-Descent 只通过 graph_build_algo 参数切换。
#
# 用法:
#   ./run_raft_ann_bench.sh [选项]
#     -d DATA      base .u8bin/.fbin       (默认 dataset/sift1M/base.1M.u8bin)
#     -q QUERY     query 文件（只 build 不用，但 config 必填）(默认同目录 query.public.10K.u8bin)
#     -g "32 64"   graph_degree 列表        (默认 "32 64")
#     -a "IVF_PQ NN_DESCENT"  建图算法列表  (默认两个都跑)
#     -i MULT      intermediate_graph_degree = MULT*graph_degree (默认 2，ann-bench 默认)
#     -n NITER     nn_descent_niter         (默认 20)
#     -r REPS      每个配置重复跑几次       (默认 3)
#     -G GPU       CUDA_VISIBLE_DEVICES     (默认 0)
#     -o OUTDIR    结果目录                 (默认 ./results_annbench)
#     -x 'k=v,...' 额外 build_param，如 'ivf_pq_build_pq_dim=64,ivf_pq_search_refine_ratio=4'
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV=/home/lanlu/miniconda3/envs/rapids_raft
BIN=$ENV/bin/ann/RAFT_CAGRA_ANN_BENCH

DATA=/home/lanlu/microBenchmark/dataset/sift1M/base.1M.u8bin
QUERY=""
DEGREES="32 64"
ALGOS="IVF_PQ NN_DESCENT"
IMULT=2
NITER=20
REPS=3
GPU=0
OUTDIR="$HERE/results_annbench"
EXTRA=""

while getopts "d:q:g:a:i:n:r:G:o:x:" opt; do
  case $opt in
    d) DATA=$OPTARG ;;
    q) QUERY=$OPTARG ;;
    g) DEGREES=$OPTARG ;;
    a) ALGOS=$OPTARG ;;
    i) IMULT=$OPTARG ;;
    n) NITER=$OPTARG ;;
    r) REPS=$OPTARG ;;
    G) GPU=$OPTARG ;;
    o) OUTDIR=$OPTARG ;;
    x) EXTRA=$OPTARG ;;
    *) sed -n 2,20p "$0"; exit 1 ;;
  esac
done

DATA=$(realpath "$DATA")
[[ -z $QUERY ]] && QUERY="$(dirname "$DATA")/query.public.10K.u8bin"
QUERY=$(realpath "$QUERY")
PY=$ENV/bin/python
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

TS=$(date +%Y%m%d_%H%M%S)
NAME=$(basename "$(dirname "$DATA")")
RUN_DIR="$OUTDIR/${NAME}_${TS}"
mkdir -p "$RUN_DIR"
CSV="$RUN_DIR/results.csv"
LOG="$RUN_DIR/run.log"
echo "algo,graph_degree,intermediate_graph_degree,nn_descent_niter,rep,build_s" > "$CSV"
{
  echo "binary     : $BIN"
  echo "libraft.so : $(ldd "$BIN" | awk '/libraft.so/{print $3}')"
  echo "raft source: $(git -C /home/lanlu/raft log -1 --format='%h %s' 2>/dev/null || echo n/a)"
  echo "GPU        : $(CUDA_VISIBLE_DEVICES=$GPU nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
  echo "data       : $DATA"
} | tee "$LOG"

for algo in $ALGOS; do
  for deg in $DEGREES; do
    ideg=$((deg * IMULT))
    tag="${algo}_g${deg}_i${ideg}"
    conf="$RUN_DIR/$tag.json"

    # 生成 ann-bench 配置（数据路径都用绝对路径，combine_path 会直接采用）
    "$PY" - "$conf" <<EOF
import json, sys
bp = {"graph_degree": $deg, "intermediate_graph_degree": $ideg,
      "graph_build_algo": "$algo", "nn_descent_niter": $NITER}
for kv in filter(None, "$EXTRA".split(",")):
    k, v = kv.split("=", 1)
    try: v = json.loads(v)
    except ValueError: pass
    bp[k] = v
if "$algo" != "NN_DESCENT": bp.pop("nn_descent_niter")
conf = {
  "dataset": {"name": "$NAME", "base_file": "$DATA", "query_file": "$QUERY",
              "distance": "euclidean"},
  "search_basic_param": {"batch_size": 10000, "k": 10},
  "index": [{"name": "raft_cagra.$tag", "algo": "raft_cagra",
             "build_param": bp, "file": "$RUN_DIR/index/$tag",
             "search_params": [{"itopk": 64}]}],
}
json.dump(conf, open(sys.argv[1], "w"), indent=2)
EOF

    for ((r = 0; r < REPS; r++)); do
      echo -e "\n########## $algo graph_degree=$deg intermediate=$ideg rep=$r ##########" | tee -a "$LOG"
      out="$RUN_DIR/${tag}_rep$r.json"
      CUDA_VISIBLE_DEVICES=$GPU "$BIN" --build --force \
        --benchmark_out="$out" --benchmark_out_format=json "$conf" 2>&1 | tee -a "$LOG"
      t=$("$PY" -c "
import json,sys
b=json.load(open('$out'))['benchmarks']
if not b or b[0].get('error_occurred'): sys.exit('build failed: '+str(b[0].get('error_message') if b else 'no result'))
print(b[0]['real_time'])")
      echo "$algo,$deg,$ideg,$NITER,$r,$t" >> "$CSV"
    done
  done
done
rm -rf "$RUN_DIR/index"

echo -e "\n========== 汇总：CAGRA build 时间 (秒) ==========" | tee -a "$LOG"
"$PY" - "$CSV" <<'EOF' | tee -a "$LOG"
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
groups = {}
for r in rows:
    groups.setdefault((r["algo"], int(r["graph_degree"]), int(r["intermediate_graph_degree"])), []).append(float(r["build_s"]))
print(f"{'algo':<11} {'degree':>6} {'interm':>6} {'reps':>4} {'mean':>8} {'min':>8} {'max':>8}  all")
for (a, g, i), ts in groups.items():
    print(f"{a:<11} {g:>6} {i:>6} {len(ts):>4} {st.mean(ts):>8.3f} {min(ts):>8.3f} {max(ts):>8.3f}  "
          + " ".join(f"{t:.3f}" for t in ts))
EOF
echo "CSV: $CSV"
echo "LOG: $LOG"
