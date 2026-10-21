#!/usr/bin/env python3
"""Accept / reject one candidate iteration against state/best.json.

usage:
  judge.py RUN_1M_AUTO RUN_1M_LARGE [--update] [--md]
    (SIFT10M is not part of the loop for now — user decision 2026-10-21)
    RUN_*    : run_pipeline.sh output dirs (each holds metrics.json)
    --update : on ACCEPT (or when best.json does not exist yet) write the
               candidate as the new state/best.json
    --md     : also print the markdown result tables for OPTIMIZATION_LOG.md

Rules (agreed with the user):
  correctness  every run produced a graph; final graph has 0 bad ids / self loops /
               duplicate edges; final degree 32, pre-prune degree 64;
               sift1M auto must take the resident path, sift1M large the large path.
  speed        build_total (sum of all stages) of sift1M-auto or sift1M-large is
               >= 3% faster than best, and no config is > 3% slower than best.
  quality      CAGRA-search recall@10 at every itopk, sift1M auto + large:
                 resident path: >= best - 0.01  and  >= CAGRA - 0.01
                 large path   : >= best - 0.01  and  >= CAGRA - 0.05   (user decision 2026-10-21)
               pre-prune KNN recall@64 (sampled exact KNN) >= best - 0.01 on every run.
  memory       large path (sift1M large): host peak RSS per stage and GPU peak <= 1.2x best
               (scalability guard: nothing may start growing with N).
"""
import json, os, sys

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(SKILL, "state")
BEST = os.path.join(STATE, "best.json")
CAGRA = json.load(open(os.path.join(STATE, "cagra_baseline.json")))
KEYS = ["sift1M_auto", "sift1M_large"]
SPEED_GAIN, SPEED_TOL, RECALL_TOL, MEM_TOL = 0.03, 0.03, 0.01, 1.2
CAGRA_TOL = {"auto": 0.01, "large": 0.05}  # allowed recall@10 gap to CAGRA per path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
if len(args) != len(KEYS):
    sys.exit(__doc__)
cand = {k: json.load(open(os.path.join(d, "metrics.json"))) for k, d in zip(KEYS, args)}
cand_dirs = dict(zip(KEYS, map(os.path.abspath, args)))
best = json.load(open(BEST)) if os.path.exists(BEST) else None

fail, notes = [], []

def initial_floor(b, k):
    """recall table of the large path at the first baseline (it starts below CAGRA-1%)."""
    init = b.get("initial_large_recall", {})
    if init and all(isinstance(v, float) for v in init.values()):  # old format: sift1M only
        init = {"sift1M_large": init}
    return init.get(k)
recalls = lambda m: {str(p["itopk"]): p["recall"] for p in m.get("search", {}).get("points", [])}

# ---- correctness
for k, m in cand.items():
    s = m.get("search")
    if not s:
        fail.append(f"{k}: no search result")
    if s:
        if s["bad_ids"] or s["self_loops"] or s["dup_edges"]:
            fail.append(f"{k}: graph defects bad={s['bad_ids']} self={s['self_loops']} dup={s['dup_edges']}")
        if s["degree"] != 32:
            fail.append(f"{k}: final degree {s['degree']} != 32")
    kn = m.get("knn_similarity")
    if not kn:
        fail.append(f"{k}: no knn_similarity")
    elif kn["degree"] != 64:
        fail.append(f"{k}: pre-prune degree {kn['degree']} != 64")
if cand["sift1M_auto"]["path"] != "resident":
    fail.append("sift1M auto did not take the resident path")
for k in ("sift1M_large",):
    if cand[k]["path"] != "large":
        fail.append(f"{k} did not take the large path")

# ---- speed / quality / memory vs best
speed = {}
if best:
    for k in KEYS:
        b, c = best["runs"][k]["build_total_s"], cand[k]["build_total_s"]
        speed[k] = (b - c) / b
        if speed[k] < -SPEED_TOL:
            fail.append(f"{k}: {(-speed[k]) * 100:.1f}% slower than best ({c:.2f}s vs {b:.2f}s)")
    if max(speed.values()) < SPEED_GAIN:
        fail.append("no config is >= 3% faster than best ("
                    + ", ".join(f"{k} {v * 100:+.1f}%" for k, v in speed.items()) + ")")

    for k in KEYS:
        cr, br = recalls(cand[k]), recalls(best["runs"][k])
        cag_tab = CAGRA[k.split("_")[0]].get("search")
        for it in (cag_tab or CAGRA["sift1M"]["search"]):
            if it not in cr:
                fail.append(f"{k}: missing itopk {it}"); continue
            if it in br and cr[it] < br[it] - RECALL_TOL:
                fail.append(f"{k} itopk {it}: recall {cr[it]:.4f} < best {br[it]:.4f} - 0.01")
            if cag_tab is None:
                continue
            cag_floor = cag_tab[it] - CAGRA_TOL[k.split("_")[1]]
            if cr[it] < cag_floor:
                fail.append(f"{k} itopk {it}: recall {cr[it]:.4f} < floor {cag_floor:.4f}")
    for k in KEYS:
        c = cand[k].get("knn_similarity", {}).get("recall@64", 0)
        b = best["runs"][k].get("knn_similarity", {}).get("recall@64", 0)
        if c < b - RECALL_TOL:
            fail.append(f"{k}: pre-prune recall@64 {c:.4f} < best {b:.4f} - 0.01")

    c, b = cand["sift1M_large"], best["runs"]["sift1M_large"]
    for st, v in c["peak_rss_gb"].items():
        if st in b["peak_rss_gb"] and v > MEM_TOL * b["peak_rss_gb"][st] + 0.2:
            fail.append(f"sift1M large {st}: peak RSS {v:.2f}GB > 1.2x best {b['peak_rss_gb'][st]:.2f}GB")
    if c["gpu_peak_mib"] and b["gpu_peak_mib"] and c["gpu_peak_mib"] > MEM_TOL * b["gpu_peak_mib"] + 256:
        fail.append(f"sift1M large: GPU peak {c['gpu_peak_mib']}MiB > 1.2x best {b['gpu_peak_mib']}MiB")
else:
    notes.append("no best.json yet: this run becomes the baseline")

verdict = "ACCEPT" if not fail else "REJECT"
print(json.dumps({"verdict": verdict, "fail": fail, "notes": notes,
                  "speedup_vs_best": {k: round(v * 100, 2) for k, v in speed.items()}}, indent=1))

if "--update" in flags and (verdict == "ACCEPT" or best is None):
    new = {"runs": cand, "dirs": cand_dirs,
           "initial_large_recall": ({k: initial_floor(best, k) or recalls(cand[k])
                                     for k in ("sift1M_large",)} if best else
                                    {k: recalls(cand[k]) for k in ("sift1M_large",)})}
    if best:
        new["baseline_runs"] = best.get("baseline_runs", best["runs"])
    json.dump(new, open(BEST, "w"), indent=1)
    print("state/best.json updated", file=sys.stderr)

if "--md" in flags:
    base = (best or {}).get("baseline_runs", (best or {}).get("runs"))
    print("\n**Build time (s, median; sum of all stages)**\n")
    print("| config | path | total | vs best | vs baseline | vs CAGRA | stages | GPU peak MiB | max RSS GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in KEYS:
        m = cand[k]; ds = k.split("_")[0]
        vb = f"{speed[k] * 100:+.1f}%" if k in speed else "—"
        vbase = f"{(base[k]['build_total_s'] - m['build_total_s']) / base[k]['build_total_s'] * 100:+.1f}%" if base else "—"
        vc = f"{m['build_total_s'] / CAGRA[ds]['build_median_s']:.2f}x"
        stg = ", ".join(f"{s} {v:.2f}" for s, v in m["stages_s"].items())
        print(f"| {k} | {m['path']} | {m['build_total_s']:.2f} | {vb} | {vbase} | {vc} | {stg} | "
              f"{m['gpu_peak_mib']} | {max(m['peak_rss_gb'].values()):.2f} |")
    print("\n**bucket2 step breakdown (s)**\n")
    steps = list(cand["sift1M_auto"]["bucket2_steps_s"])
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
    print("\n**(b) final graph (degree 32, original ids), cuVS CAGRA search, recall@10 / QPS (10K queries)**\n")
    its = list(CAGRA["sift1M"]["search"])
    print("| graph | " + " | ".join(f"itopk {i}" for i in its) + " |")
    print("|---|" + "---|" * len(its))
    for ds in ("sift1M", "sift10M"):
        if "search" in CAGRA[ds]:
            print(f"| CAGRA native 64→32, {ds} | " + " | ".join(
                f"{CAGRA[ds]['search'][i]:.4f} / {CAGRA[ds]['qps'][i] / 1e3:.0f}K" for i in its) + " |")
    for k in KEYS:
        pts = {str(p["itopk"]): p for p in cand[k].get("search", {}).get("points", [])}
        if pts:
            print(f"| ours {k} | " + " | ".join(
                f"{pts[i]['recall']:.4f} / {pts[i]['qps'] / 1e3:.0f}K" if i in pts else "—" for i in its) + " |")
