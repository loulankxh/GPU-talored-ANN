# Pipeline parameters (sourced by run_pipeline.sh). The loop may tune these;
# every change must be recorded in OPTIMIZATION_LOG.md like a code change.
# Hard constraints (do NOT change): NEIGHBORS_M=64 (build index degree),
# OUTPUT_DEGREE=32 (final degree after prune + reverse edges).
NEIGHBORS_M=64
OUTPUT_DEGREE=32

: "${CENTROID_RATIO:=0.004}"   # A2 (user-approved, params only): avg bucket ~250 points
SEED=42
: "${KNN_K:=32}"
: "${NPROBE:=80}"   # A2: pool (NPROBE+1) x bucket kept ~20K rows
ITERATIONS=3

PRUNE_CACHE_MB=16000
CHUNKED_GPU_BUDGET_MB=16000
CHUNKED_EDGE_BUFFER_MB=16000

# extra flags appended verbatim (keep empty unless an optimization needs them)
BUCKET2_EXTRA="--skip-npy --skip-dists --kmeans-train-fraction 0.10 --cache-mb 15000 --bucket-mem-mb 16000"   # #9: Step 6 host Belady cache holds a whole SIFT100M round (13.6 GB of records; default cpu_limit/2 = 8.6 GB); #14: round 2..t records stay in RAM (no BucketWrite to disk), Step 6 copies misses from them
OPTIMIZE_EXTRA="--skip-sort"
PRUNE_EXTRA=""
CHUNKED_EXTRA=""
