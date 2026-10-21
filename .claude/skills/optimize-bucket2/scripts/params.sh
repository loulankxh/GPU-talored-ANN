# Pipeline parameters (sourced by run_pipeline.sh). The loop may tune these;
# every change must be recorded in OPTIMIZATION_LOG.md like a code change.
# Hard constraints (do NOT change): NEIGHBORS_M=64 (build index degree),
# OUTPUT_DEGREE=32 (final degree after prune + reverse edges).
NEIGHBORS_M=64
OUTPUT_DEGREE=32

CENTROID_RATIO=0.01
SEED=42
KNN_K=32
NPROBE=150
ITERATIONS=4

PRUNE_CACHE_MB=16000
CHUNKED_GPU_BUDGET_MB=16000
CHUNKED_EDGE_BUFFER_MB=16000

# extra flags appended verbatim (keep empty unless an optimization needs them)
BUCKET2_EXTRA="--skip-npy --kmeans-train-fraction 0.25"
OPTIMIZE_EXTRA="--skip-sort"
PRUNE_EXTRA=""
CHUNKED_EXTRA=""
