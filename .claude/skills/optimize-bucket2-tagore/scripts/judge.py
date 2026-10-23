#!/usr/bin/env python3
"""Accept / reject one candidate iteration against state/best.json (dataset-generic).

usage:
  judge.py RUN_DIR [RUN_DIR ...] [--update] [--md] [--quick] [--add-new] [--rebaseline]
    RUN_DIR      : run_pipeline.sh output dirs (each holds metrics.json); the config key is
                   <dataset>_<mode> from metrics.json (e.g. sift1M_auto, sift100M_auto, msTuring10M_auto)
    --update     : on ACCEPT (or when best.json does not exist yet) write the candidate into state/best.json
    --md         : also print the markdown result tables for the log
    --quick      : pre-check of a subset (e.g. the SIFT1M / SIFT10M runs before SIFT100M): correctness,
                   regressions, search quality and memory only -- no "needs a gain", no bench-set check
    --add-new    : with --update, record configs best.json does not have yet even if the verdict is REJECT
    --rebaseline : with --update, overwrite the best entries of these configs with the candidate when only
                   speed rules failed (used once after a measurement-rule change, e.g. SIFT100M 1 -> 3 runs)

Rules (agreed with the user; 2026-10-23: every config >= 3% = gain, > 3% slower = regression, SIFT100M now
median of 3 runs like SIFT1M):
  correctness  every run produced a graph; 0 bad ids / self loops / duplicate edges; final degree 32,
               pre-prune degree 64; mode "large" must take the large path; a config must take the same
               path as its best entry (e.g. sift1M auto = resident).
  speed        ACCEPT needs the full bench set (datasets.json "bench"), at least one config >= 3% faster
               than best and no config > 3% slower (build_total = sum of all stages, median of reps).
  quality      recall@10 at every itopk / DiskANN L, reference from datasets.json "ref":
                 tagore / cagra: state/<ref>_baseline.json[dataset] -- resident path >= ref - 0.01,
                                 large path >= ref - 0.05 (falls back to the other table, with a note)
                 own           : >= own best - own_tol (SIFT100M / msTuring100M 0.05, else 0.01)
               pre-prune KNN similarity is recorded, not gated.
  memory       large-path configs: host peak RSS per stage and GPU peak <= 1.2x best (scalability guard).
"""
import json, os, sys

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(SKILL, "state")
BEST = os.path.join(STATE, "best.json")
REG = {k: v for k, v in json.load(open(os.path.join(SKILL, "scripts", "datasets.json"))).items()
       if not k.startswith("_")}
def load_ref(name):
    p = os.path.join(STATE, f"{name}_baseline.json")
    return json.load(open(p)) if os.path.exists(p) else {}
REFS = {"tagore": load_ref("tagore"), "cagra": load_ref("cagra")}
SPEED_GAIN, SPEED_TOL, MEM_TOL = 0.03, 0.03, 1.2
REF_TOL = {"resident": 0.01, "large": 0.05}
BENCH = [f"{ds}_{m}" for ds, v in REG.items() for m in v.get("bench", [])]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
if not args:
    sys.exit(__doc__)
cand, cand_dirs = {}, {}
for d in args:
    m = json.load(open(os.path.join(d, "metrics.json")))
    k = f"{m['dataset']}_{m['mode']}"
    if k in cand:
        sys.exit(f"config {k} given twice")
    cand[k], cand_dirs[k] = m, os.path.abspath(d)
KEYS = list(cand)
best = json.load(open(BEST)) if os.path.exists(BEST) else None
brun = best["runs"] if best else {}

fail, speed_fail, notes = [], [], []
for k, d in cand_dirs.items():   # SMOKE=1 runs (no bench lock, other GPU, niced) are never judged
    if os.path.exists(os.path.join(d, "SMOKE")):
        fail.append(f"{k}: {d} is a SMOKE run (not timed under the bench lock)")
recalls = lambda m: {str(p["itopk"]): p["recall"] for p in m.get("search", {}).get("points", [])}

# ---- correctness
for k, m in cand.items():
    if m["dataset"] not in REG:
        fail.append(f"{k}: dataset not in datasets.json")
    s = m.get("search")
    if not s:
        fail.append(f"{k}: no search result")
    else:
        if s["bad_ids"] or s["self_loops"] or s["dup_edges"]:
            fail.append(f"{k}: graph defects bad={s['bad_ids']} self={s['self_loops']} dup={s['dup_edges']}")
        if s["degree"] != 32:
            fail.append(f"{k}: final degree {s['degree']} != 32")
    kn = m.get("knn_similarity")
    if not kn:
        fail.append(f"{k}: no knn_similarity")
    elif kn["degree"] != 64:
        fail.append(f"{k}: pre-prune degree {kn['degree']} != 64")
    if m["mode"] == "large" and m["path"] != "large":
        fail.append(f"{k}: did not take the large path")
    if k in brun and brun[k]["path"] != m["path"]:
        fail.append(f"{k}: took the {m['path']} path, best took {brun[k]['path']}")

# ---- speed
speed = {}
for k in KEYS:
    if k not in brun:
        notes.append(f"{k}: no best yet (speed not compared; recorded with --update / --add-new)")
        continue
    b, c = brun[k]["build_total_s"], cand[k]["build_total_s"]
    speed[k] = (b - c) / b
    if speed[k] < -SPEED_TOL:
        speed_fail.append(f"{k}: {(-speed[k]) * 100:.1f}% slower than best ({c:.2f}s vs {b:.2f}s)")
if "--quick" not in flags and best:
    missing = [k for k in BENCH if k not in cand]
    if missing:
        speed_fail.append("incomplete bench set, missing " + ", ".join(missing))
    if speed and not any(v >= SPEED_GAIN for v in speed.values()):
        speed_fail.append("no config is >= 3% faster than best ("
                          + ", ".join(f"{k} {v * 100:+.1f}%" for k, v in speed.items()) + ")")

# ---- quality
for k in KEYS:
    m = cand[k]; ds = m["dataset"]; reg = REG.get(ds, {})
    cr = recalls(m)
    ref_name = reg.get("ref", "own")
    tab = None
    if ref_name in REFS:
        for name in (ref_name, "cagra" if ref_name == "tagore" else "tagore"):
            if "search" in REFS[name].get(ds, {}):
                tab = REFS[name][ds]["search"]
                if name != ref_name:
                    notes.append(f"{k}: no {ref_name} reference for {ds}, gated against {name}")
                ref_name = name
                break
        if tab is None:
            notes.append(f"{k}: no external reference for {ds}, gated against own best")
            ref_name = "own"
    if ref_name == "own":
        own = brun.get(k)
        if not own:
            notes.append(f"{k}: no own best yet, search quality recorded only")
            continue
        tab, tol = recalls(own), reg.get("own_tol", 0.01)
        for it, r in tab.items():
            if it not in cr:
                fail.append(f"{k}: missing itopk/L {it}")
            elif cr[it] < r - tol:
                fail.append(f"{k} itopk/L {it}: recall {cr[it]:.4f} < own best {r:.4f} - {tol}")
    else:
        tol = REF_TOL["large" if m["path"] == "large" else "resident"]
        for it, r in tab.items():
            if it not in cr:
                fail.append(f"{k}: missing itopk {it}")
            elif cr[it] < r - tol:
                fail.append(f"{k} itopk {it}: recall {cr[it]:.4f} < {ref_name} floor {r - tol:.4f}")

# ---- memory (large path)
for k in KEYS:
    if cand[k]["path"] != "large" or k not in brun:
        continue
    c, b = cand[k], brun[k]
    for st, v in c["peak_rss_gb"].items():
        if st in b["peak_rss_gb"] and v > MEM_TOL * b["peak_rss_gb"][st] + 0.2:
            fail.append(f"{k} {st}: peak RSS {v:.2f}GB > 1.2x best {b['peak_rss_gb'][st]:.2f}GB")
    if c["gpu_peak_mib"] and b["gpu_peak_mib"] and c["gpu_peak_mib"] > MEM_TOL * b["gpu_peak_mib"] + 256:
        fail.append(f"{k}: GPU peak {c['gpu_peak_mib']}MiB > 1.2x best {b['gpu_peak_mib']}MiB")
if not best:
    notes.append("no best.json yet: this run becomes the baseline")

verdict = "ACCEPT" if not (fail or speed_fail) else "REJECT"
print(json.dumps({"verdict": verdict, "fail": fail + speed_fail, "notes": notes,
                  "speedup_vs_best": {k: round(v * 100, 2) for k, v in speed.items()},
                  "mode": "quick" if "--quick" in flags else "full"}, indent=1))

def write(runs_update):
    new = dict(best) if best else {"runs": {}, "dirs": {}}
    new["runs"] = dict(new.get("runs", {})); new["dirs"] = dict(new.get("dirs", {}))
    new["baseline_runs"] = dict(new.get("baseline_runs", new["runs"]))
    for k in runs_update:
        new["runs"][k] = cand[k]; new["dirs"][k] = cand_dirs[k]
        new["baseline_runs"].setdefault(k, cand[k])   # first time a config is recorded it is its own baseline
    json.dump(new, open(BEST, "w"), indent=1)
    print(f"state/best.json updated: {runs_update}", file=sys.stderr)

if "--update" in flags:
    if "--quick" in flags:
        print("--update ignored with --quick", file=sys.stderr)
    elif verdict == "ACCEPT" or best is None:
        write(KEYS)
    elif "--rebaseline" in flags and not fail:
        write(KEYS); print("rebaseline: only speed rules failed; entries replaced", file=sys.stderr)
    elif "--add-new" in flags and best:
        new_keys = [k for k in KEYS if k not in brun and not any(f.startswith(k) for f in fail)]
        if new_keys:
            write(new_keys)

if "--md" in flags:
    base = (best or {}).get("baseline_runs", brun)
    print("\n**Build time (s, median of reps; sum of all stages)**\n")
    print("| config | path | reps | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in KEYS:
        m = cand[k]; ds = m["dataset"]
        vb = f"{speed[k] * 100:+.1f}%" if k in speed else "—"
        vbase = f"{(base[k]['build_total_s'] - m['build_total_s']) / base[k]['build_total_s'] * 100:+.1f}%" if k in base else "—"
        vt = f"{m['build_total_s'] / REFS['tagore'][ds]['build_median_s']:.2f}x" if REFS["tagore"].get(ds, {}).get("build_median_s") else "—"
        vc = f"{m['build_total_s'] / REFS['cagra'][ds]['build_median_s']:.2f}x" if REFS["cagra"].get(ds, {}).get("build_median_s") else "—"
        stg = ", ".join(f"{s} {v:.2f}" for s, v in m["stages_s"].items())
        print(f"| {k} | {m['path']} | {m.get('reps', 1)} | {m['build_total_s']:.2f} | {vb} | {vbase} | {vt} | {vc} | {stg} | "
              f"{m['gpu_peak_mib']} | {max(m['peak_rss_gb'].values()):.2f} |")
    steps = []
    for k in KEYS:
        steps += [s for s in cand[k]["bucket2_steps_s"] if s not in steps]
    print("\n**bucket2 step breakdown (s)**\n")
    print("| config | " + " | ".join(steps) + " |")
    print("|---|" + "---|" * len(steps))
    for k in KEYS:
        print(f"| {k} | " + " | ".join(f"{cand[k]['bucket2_steps_s'].get(s, 0):.2f}" for s in steps) + " |")
    print("\n**(a) pre-prune index (degree 64) vs exact KNN, 10K sampled points**\n")
    print("| config | recall@1 | recall@10 | recall@32 | recall@64 |")
    print("|---|---|---|---|---|")
    for k in KEYS:
        kn = cand[k].get("knn_similarity", {})
        print(f"| {k} | " + " | ".join(f"{kn.get(f'recall@{x}', float('nan')):.4f}" for x in (1, 10, 32, 64)) + " |")
    print("\n**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); search per datasets.json (cagra: cuVS CAGRA itopk, diskann: DiskANN CPU L)**\n")
    its = ["32", "64", "128", "256", "512"]
    print("| graph | " + " | ".join(its) + " |")
    print("|---|" + "---|" * len(its))
    for ds in dict.fromkeys(cand[k]["dataset"] for k in KEYS):
        for name in ("tagore", "cagra"):
            r = REFS[name].get(ds, {})
            if "search" in r:
                print(f"| {name.upper()} {r.get('label', '')}, {ds} | " + " | ".join(
                    f"{r['search'][i]:.4f} / {r['qps'][i] / 1e3:.0f}K" if i in r["search"] else "—" for i in its) + " |")
    for k in KEYS:
        pts = {str(p["itopk"]): p for p in cand[k].get("search", {}).get("points", [])}
        if pts:
            print(f"| ours {k} | " + " | ".join(
                f"{pts[i]['recall']:.4f} / {pts[i]['qps'] / 1e3:.0f}K" if i in pts else "—" for i in its) + " |")
