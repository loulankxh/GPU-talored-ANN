#!/usr/bin/env python3
"""Collect one run_pipeline.sh output dir into metrics.json (stdout).
usage: summarize.py OUTDIR DATASET MODE"""
import glob, json, os, re, statistics as st, sys

out, ds, mode = sys.argv[1:4]
STAGES = ["bucket2", "optimize", "prune", "optimize_chunked", "translate"]
reps = []
for rd in sorted(glob.glob(os.path.join(out, "rep*"))):
    r = {"stages": {}, "peak_rss_gb": {}, "bucket2_steps": {}}
    for s in STAGES:
        f = os.path.join(rd, s + ".time")
        if os.path.exists(f):
            last = open(f).read().split("\n")
            wall, rss = [l for l in last if re.match(r"^[\d.]+ \d+$", l)][-1].split()
            r["stages"][s] = float(wall)
            r["peak_rss_gb"][s] = round(int(rss) / 2**20, 3)
    log = os.path.join(rd, "bucket2.log")
    if os.path.exists(log):
        for m in re.finditer(r"^\s+(Step [\d.]+|Wait|Write)\s+\(([^)]*)\).*?:\s+([\d.]+)s", open(log).read(), re.M):
            r["bucket2_steps"][f"{m.group(1)} ({m.group(2)})"] = float(m.group(3))
    r["build_total_s"] = round(sum(r["stages"].values()), 3)
    p = os.path.join(rd, "path.txt")
    r["path"] = open(p).read().strip() if os.path.exists(p) else "?"
    g = os.path.join(rd, "gpu_mem.txt")
    vals = [int(x) for x in open(g).read().split() if x.isdigit()] if os.path.exists(g) else []
    r["gpu_peak_mib"] = max(vals) if vals else None
    for q in ("knn_similarity", "search"):
        f = os.path.join(rd, q + ".json")
        if os.path.exists(f) and os.path.getsize(f):
            r[q] = json.loads(open(f).read().strip().splitlines()[-1])
    reps.append(r)

if not reps:
    sys.exit("no reps found in " + out)
med = lambda xs: round(st.median(xs), 3)
stage_names = [s for s in STAGES if all(s in r["stages"] for r in reps)]
res = {
    "dataset": ds, "mode": mode, "path": reps[-1]["path"], "reps": len(reps),
    "build_total_s": med([r["build_total_s"] for r in reps]),
    "build_total_all_s": [r["build_total_s"] for r in reps],
    "stages_s": {s: med([r["stages"][s] for r in reps]) for s in stage_names},
    "bucket2_steps_s": {k: med([r["bucket2_steps"][k] for r in reps if k in r["bucket2_steps"]])
                        for k in reps[-1]["bucket2_steps"]},
    "peak_rss_gb": {s: max(r["peak_rss_gb"].get(s, 0) for r in reps) for s in stage_names},
    "gpu_peak_mib": max((r["gpu_peak_mib"] or 0) for r in reps),
}
for q in ("knn_similarity", "search"):
    if q in reps[-1]:
        res[q] = reps[-1][q]
print(json.dumps(res, indent=1))
