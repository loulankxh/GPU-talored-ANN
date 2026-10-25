# Pipeline parameters (sourced by run_pipeline.sh). The loop may tune these;
# every change must be recorded in OPTIMIZATION_LOG.md like a code change.
# Hard constraints (do NOT change): NEIGHBORS_M=64 (build index degree),
# OUTPUT_DEGREE=32 (final degree after prune + reverse edges).
NEIGHBORS_M=64
OUTPUT_DEGREE=32

# CENTROID_RATIO / NPROBE / kmeans train fraction / host + GPU cache sizes / ITERATIONS per dataset come from
# dataset_env.py (size-adaptive, reproduces the hand-tuned SIFT100M values; export a value to override it).
SEED=42
: "${KNN_K:=32}"
: "${ITERATIONS:=3}"   # default (exported value wins, for sweeps); datasets.json "iterations" overrides (SIFT100M / msTuring100M: 2)

PRUNE_CACHE_MB=16000
CHUNKED_GPU_BUDGET_MB=16000
CHUNKED_EDGE_BUFFER_MB=16000

# extra flags appended verbatim (keep empty unless an optimization needs them)
BUCKET2_EXTRA="--skip-npy --skip-dists"   # + --kmeans-train-fraction / --cache-mb / --bucket-mem-mb / --gpu-bucket-cache-mb from dataset_env.py (#9 host cache holds a whole round, #14 round 2..t records in RAM, #21 GPU bucket cache = half the GPU)
OPTIMIZE_EXTRA="--skip-sort"
PRUNE_EXTRA=""
CHUNKED_EXTRA=""
