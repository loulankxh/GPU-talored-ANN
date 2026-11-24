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
        txt = open(log).read()
        for m in re.finditer(r"^\s+(Step [\d.]+|Wait|Write)\s+\(([^)]*)\).*?:\s+([\d.]+)s", txt, re.M):
            r["bucket2_steps"][f"{m.group(1)} ({m.group(2)})"] = float(m.group(3))
        # A1 local join (docs/A1_local_join.md §2.6): every "A1_stats: k=v ..." line, in order
        # 只取 k=v 形式的字段 (行尾的说明文字如 "(fp16 screen)" 跳过)
        a1 = [dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
              for m in re.finditer(r"^A1_stats:\s*(.*)$", txt, re.M)]
        if a1:
            r["a1_stats"] = a1
    r["build_total_s"] = round(sum(r["stages"].values()), 3)
    f = os.path.join(rd, "gpu.txt")
    gl = [int(x) for x in open(f).read().strip().split(",") if x.strip()] if os.path.exists(f) else [0]
    r["gpu"] = gl[0]; r["gpu_list"] = gl   # MGPU: all devices of the rep
    p = os.path.join(rd, "path.txt")
    r["path"] = open(p).read().strip() if os.path.exists(p) else "?"
    g = os.path.join(rd, "gpu_mem.txt")
    vals = [int(x) for x in open(g).read().split() if x.isdigit()] if os.path.exists(g) else []
    r["gpu_peak_mib"] = max(vals) if vals else None
    for q in ("knn_similarity", "search"):
        f = os.path.join(rd, q + ".json")
        if os.path.exists(f) and os.path.getsize(f):
            r[q] = json.loads(open(f).read().strip().splitlines()[-1])
    # spot simulation (optimize-bucket2-spot): scheduler stats lines "[spot] stage=..." of bucket2 / prune and the
    # simulator's event log
    sp = {"stages": []}
    for lg in ("bucket2.log", "prune.log"):
        f = os.path.join(rd, lg)
        if os.path.exists(f):
            for m in re.finditer(r"^\[spot\] stage=(\S+) (.*)$", open(f).read(), re.M):
                kv = dict(x.split("=", 1) for x in m.group(2).split() if "=" in x)
                kv["stage"] = m.group(1)
                sp["stages"].append(kv)
    f = os.path.join(rd, "spot_events.log")
    if os.path.exists(f):
        ev = open(f).read()
        sp["events"] = {e: len(re.findall(r" inst=\d+ " + e + r" ", ev)) for e in ("notice", "preempt", "join")}
    if sp["stages"] or "events" in sp:
        # redone_units: exp2 segment-restart (spot_sched.hpp) -- 0 always under the default cursor-resume
        # recovery, since nothing is ever replayed/suppressed there.
        for key in ("rejected_units", "redone_units", "preempts", "steals", "setups", "reaped", "drains"):
            sp[key] = sum(int(x.get(key, 0)) for x in sp["stages"])
        r["spot"] = sp
    reps.append(r)

if not reps:
    sys.exit("no reps found in " + out)
med = lambda xs: round(st.median(xs), 3)
stage_names = [s for s in STAGES if all(s in r["stages"] for r in reps)]
res = {
    "dataset": ds, "mode": mode, "path": reps[-1]["path"], "reps": len(reps),
    "gpus": sorted({g for r in reps for g in r["gpu_list"]}),
    "num_gpus": int(os.environ.get("MGPU", "1") or 1),   # run_pipeline MGPU=G (multi-GPU build)
    "build_total_s": med([r["build_total_s"] for r in reps]),
    "build_total_all_s": [r["build_total_s"] for r in reps],
    "stages_s": {s: med([r["stages"][s] for r in reps]) for s in stage_names},
    "bucket2_steps_s": {k: med([r["bucket2_steps"][k] for r in reps if k in r["bucket2_steps"]])
                        for k in reps[-1]["bucket2_steps"]},
    "peak_rss_gb": {s: max(r["peak_rss_gb"].get(s, 0) for r in reps) for s in stage_names},
    "gpu_peak_mib": max((r["gpu_peak_mib"] or 0) for r in reps),
}
if os.environ.get("SPOT", "0") == "1":   # spot run: config key gets _spot<TAG> (judge.py)
    res["spot"] = os.environ.get("SPOT_TAG", "") or "none"
    res["spot_reps"] = [r.get("spot", {}) for r in reps]
for q in ("knn_similarity", "search", "a1_stats"):
    if q in reps[-1]:
        res[q] = reps[-1][q]
print(json.dumps(res, indent=1))
