#!/usr/bin/env python3
"""Dataset lookup + size-adaptive pipeline parameters, printed as shell assignments for run_pipeline.sh.

usage: dataset_env.py NAME            (NAME from datasets.json)
Prints DS_DIR, BASE, EXT, DTYPE, N, DIM, DEF_REPS, SEARCH, plus the adaptive parameters below. Precedence:
exported environment variable (sweeps) > datasets.json "params" (per-dataset calibration, step C) > rules.

Adaptive rules (user decisions 2026-10-22/23; they reproduce the hand-tuned SIFT100M values of #3/#9/#14/#21):
  CENTROID_RATIO  = 1 / B, B = average bucket size, target 512 points (#33, A2; was 250), kept in [128, 512]; B grows beyond 512
                    only when the centroid matrix (N/B x DIM x 4 B) would exceed CENTROID_GPU_MB (huge sets,
                    e.g. SIFT1B) -- semantics unchanged, only the bucket count
  NPROBE          = ceil(POOL_ROWS / B), POOL_ROWS = 20000 rows per query pool ((NPROBE+1) x B ~ constant)
  ITERATIONS      = datasets.json "iterations" (SIFT100M / msTuring100M: 2), else params.sh
  --kmeans-train-fraction = min(0.10, 10M / N)        (KMeans|| trains on <= 10M points)
  --cache-mb      = max(15000, roundup1000(1.15 x round records MiB)), records = N x (DIM x itemsize + 8 B)
  --bucket-mem-mb = cache-mb + 1000      (floors = the values every dataset used before 2026-10-23, so the
                    SIFT configs run with exactly the same flags; only larger round records scale up)
  --gpu-bucket-cache-mb = GPU_MB / 2
  all host-side sizes capped by HOST_BUDGET_MB / 4 each (machine: 251 GB RAM; optimize_chunked
  --host-budget-mb = HOST_BUDGET_MB is a ceiling, not a target)
"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ITEM = {".u8bin": ("uint8", 1), ".i8bin": ("int8", 1), ".fbin": ("float32", 4)}
GPU_MB = int(os.environ.get("GPU_MB", 16384))            # V100-SXM2-16GB
HOST_BUDGET_MB = int(os.environ.get("HOST_BUDGET_MB", 204800))
CENTROID_GPU_MB = int(os.environ.get("CENTROID_GPU_MB", 1024))
B_TARGET, B_MIN, B_MAX, POOL_ROWS = 512, 128, 512, 20000   # target 250 -> 512: #33 (A2)

reg = json.load(open(os.path.join(HERE, "datasets.json")))
name = sys.argv[1] if len(sys.argv) > 1 else ""
if name not in reg or name.startswith("_"):
    sys.exit(f"unknown dataset '{name}'; registered: {', '.join(k for k in reg if not k.startswith('_'))}")
d = reg[name]
base = os.path.join(d["dir"], d["base"])
ext = os.path.splitext(base)[1]
if ext not in ITEM:
    sys.exit(f"{base}: unsupported vector type {ext}")
dtype, item = ITEM[ext]
with open(base, "rb") as f:
    hdr = f.read(8)
n, dim = int.from_bytes(hdr[:4], "little"), int.from_bytes(hdr[4:], "little")
if os.path.getsize(base) != 8 + n * dim * item:   # bounds check
    sys.exit(f"{base}: size does not match header {n} x {dim}")

B = B_TARGET
while (n / B) * dim * 4 / 2**20 > CENTROID_GPU_MB:   # huge datasets only
    B *= 2
B = max(B_MIN, B) if B <= B_MAX else B
records_mib = n * (dim * item + 8) / 2**20
cap = HOST_BUDGET_MB // 4
cache_mb = min(cap, max(15000, int(math.ceil(records_mib * 1.15 / 1000.0) * 1000)))
adaptive = {
    "CENTROID_RATIO": f"{1.0 / B:.6g}",
    "NPROBE": str(math.ceil(POOL_ROWS / B)),
    "KMEANS_TRAIN_FRACTION": f"{min(0.10, 10_000_000 / n):.6g}",
    "CACHE_MB": str(cache_mb),
    "BUCKET_MEM_MB": str(min(cap, cache_mb + 1000)),
    "GPU_BUCKET_CACHE_MB": str(GPU_MB // 2),
    "CHUNKED_HOST_BUDGET_MB": str(HOST_BUDGET_MB),
    "A1_PASSES": "0",   # A1-R local join passes (resident path; #40, user-approved A1)
    "A7_UNWIN": "0",    # A7: batched GPU prune without the window restriction (#55, user-approved for 10M)
}
if d.get("iterations"):
    adaptive["ITERATIONS"] = str(d["iterations"])

# per-dataset calibration (datasets.json "params", e.g. msTuring after step C) overrides the size rules
for k, v in d.get("params", {}).items():
    if k not in adaptive:
        sys.exit(f"{name}: unknown param override {k}")
    adaptive[k] = str(v)

out = {"DS_DIR": d["dir"], "BASE": base, "EXT": ext, "DTYPE": dtype, "N": str(n), "DIM": str(dim),
       "DEF_REPS": str(d.get("reps", 1)), "SEARCH": d.get("search", "diskann")}
for k, v in adaptive.items():
    out[k] = os.environ.get(k) or v     # explicit environment value wins
for k, v in out.items():
    print(f"{k}={v}")
Bf = 1.0 / float(out["CENTROID_RATIO"])
print(f"ADAPT_NOTE='avg bucket {Bf:.0f}, centroids {int(n / Bf)}, round records {records_mib / 1024:.1f} GiB{" (datasets.json params)" if d.get("params") else ""}'")
