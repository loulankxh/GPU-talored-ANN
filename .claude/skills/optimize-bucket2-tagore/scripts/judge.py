#!/usr/bin/env python3
"""Accept / reject one candidate iteration against state/best.json.

usage:
  judge.py RUN_1M_AUTO RUN_1M_LARGE [RUN_10M_AUTO [RUN_100M_AUTO]] [--update] [--md]
    RUN_10M_AUTO: sift10M auto (1 rep; auto-selects the large path). Part of the loop
                  again since user decision 2026-10-21 (2nd).
    RUN_100M_AUTO: sift100M auto (1 rep, large path). User decision 2026-10-22: the loop now
                  optimizes SIFT100M; SIFT1M / SIFT10M are kept as quality checks.
    RUN_*    : run_pipeline.sh output dirs (each holds metrics.json)
    --update : on ACCEPT (or when best.json does not exist yet) write the
               candidate as the new state/best.json
    --md     : also print the markdown result tables for OPTIMIZATION_LOG.md
    --add-new: with --update, record configs that best.json does not have yet (e.g. the first
               sift100M baseline) even when the verdict is REJECT; existing entries stay untouched

Rules (agreed with the user):
  correctness  every run produced a graph; final graph has 0 bad ids / self loops /
               duplicate edges; final degree 32, pre-prune degree 64;
               sift1M auto must take the resident path, sift1M large the large path.
  speed        build_total (sum of all stages) of sift1M-auto, sift1M-large or
               sift10M-auto is >= 3% faster than best, and no config is > 3% slower than
               best (user 2026-10-21: a >= 3% SIFT10M gain is accepted as long as neither
               SIFT1M config is > 3% slower and search passes).
  sift100M     (TAGORE loop, user decision 2026-10-22): SIFT100M is built with 2 rounds, 1 rep, and
               searched with DiskANN's CPU in-memory searcher (points[].itopk = DiskANN L); it is
               compared with its own best only: >= 7% faster = gain, > 7% slower = regression
               (1 run is noisy; a 3-7% gain is reported as "possible gain" for a later rerun);
               search recall@10 >= own best - 0.05 at every L; large-path 1.2x memory guard.
               sift1M / sift10M: >= 3% = gain, > 3% slower = regression, search vs TAGORE.
               ACCEPT iff at least one config gains, none regresses, and search passes on all.
  quality      (TAGORE loop, user decision 2026-10-22) sift1M / sift10M search recall@10 at every
               itopk vs TAGORE's own graph searched the same way (state/tagore_baseline.json):
                 resident path: >= TAGORE - 0.01
                 large path   : >= TAGORE - 0.05
               falls back to state/cagra_baseline.json (with a note) for a dataset TAGORE has no
               search result for. Pre-prune KNN similarity (a) is recorded, not gated.
  memory       large path (sift1M large, sift10M auto): host peak RSS per stage and GPU peak <= 1.2x best
               (scalability guard: nothing may start growing with N).
"""
import json, os, sys

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(SKILL, "state")
BEST = os.path.join(STATE, "best.json")
CAGRA = json.load(open(os.path.join(STATE, "cagra_baseline.json")))
# TAGORE loop: sift1M / sift10M quality reference = TAGORE (step B writes tagore_baseline.json in
# cagra_baseline.json's format: {ds: {build_median_s, search: {itopk: recall}, qps: {itopk: qps}}})
TB = os.path.join(STATE, "tagore_baseline.json")
TAGORE = json.load(open(TB)) if os.path.exists(TB) else {}
def ref_for(ds):
    """(name, table) of the search reference for a dataset: TAGORE if it has one, else CAGRA."""
    if "search" in TAGORE.get(ds, {}):
        return "TAGORE", TAGORE[ds]
    if "search" in CAGRA.get(ds, {}):
        return "CAGRA", CAGRA[ds]
    return None, None
KEYS = ["sift1M_auto", "sift1M_large", "sift10M_auto", "sift100M_auto"]
LARGE_KEYS = ["sift1M_large", "sift10M_auto", "sift100M_auto"]   # configs that must take the large path
SPEED_GAIN, SPEED_TOL, RECALL_TOL, MEM_TOL = 0.03, 0.03, 0.01, 1.2
CAGRA_TOL = {"auto": 0.01, "large": 0.05}  # allowed recall@10 gap to CAGRA per path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
if len(args) not in (2, 3, 4):
    sys.exit(__doc__)
KEYS = KEYS[:len(args)]
SPEED_KEYS = KEYS
SLOW_TOL = {"sift100M_auto": 0.07}   # per-config regression threshold (default SPEED_TOL)
GAIN_TOL = {"sift100M_auto": 0.07}   # per-config gain threshold (default SPEED_GAIN); TAGORE loop 2026-10-22
OWN_RECALL_TOL = {"sift100M_auto": 0.05}   # recall vs own best (configs without a reference)
LARGE_KEYS = [k for k in LARGE_KEYS if k in KEYS]
cand = {k: json.load(open(os.path.join(d, "metrics.json"))) for k, d in zip(KEYS, args)}
cand_dirs = dict(zip(KEYS, map(os.path.abspath, args)))
best = json.load(open(BEST)) if os.path.exists(BEST) else None

fail, notes = [], []
for k, d in cand_dirs.items():   # SMOKE=1 runs (no bench lock, other GPU, niced) are never judged
    if os.path.exists(os.path.join(d, "SMOKE")):
        fail.append(f"{k}: {d} is a SMOKE run (not timed under the bench lock)")

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
for k in LARGE_KEYS:
    if cand[k]["path"] != "large":
        fail.append(f"{k} did not take the large path")

# ---- speed / quality / memory vs best
speed = {}
if best:
    for k in KEYS:
        if k not in best["runs"]:
            notes.append(f"{k}: no best yet (speed not compared; recorded on --update)")
            continue
        b, c = best["runs"][k]["build_total_s"], cand[k]["build_total_s"]
        speed[k] = (b - c) / b
        if k in SPEED_KEYS and speed[k] < -SLOW_TOL.get(k, SPEED_TOL):
            fail.append(f"{k}: {(-speed[k]) * 100:.1f}% slower than best ({c:.2f}s vs {b:.2f}s)")
    gated = {k: v for k, v in speed.items() if k in SPEED_KEYS}
    if gated and not any(v >= GAIN_TOL.get(k, SPEED_GAIN) for k, v in gated.items()):
        fail.append("no gated config is faster than best by its gain threshold ("
                    + ", ".join(f"{k} {v * 100:+.1f}% (need {GAIN_TOL.get(k, SPEED_GAIN) * 100:.0f}%)"
                                for k, v in speed.items()) + ")")
    for k, v in gated.items():
        if SPEED_GAIN <= v < GAIN_TOL.get(k, SPEED_GAIN):
            notes.append(f"{k}: possible gain {v * 100:+.1f}% (below the {GAIN_TOL[k] * 100:.0f}% gate; "
                         "log it under 'Possible gains' for a multi-run recheck)")

    for k in KEYS:
        cr = recalls(cand[k])
        ds = k.split("_")[0]
        ref_name, ref = ref_for(ds) if ds != "sift100M" else (None, None)
        if ref_name == "CAGRA":
            notes.append(f"{k}: no TAGORE search reference for {ds}, gated against CAGRA")
        cag_tab = ref["search"] if ref else None
        own = best["runs"].get(k) if cag_tab is None else None   # sift100M: compare with own best
        own_tab = recalls(own) if own else None
        otol = OWN_RECALL_TOL.get(k, RECALL_TOL)
        for it in (cag_tab or CAGRA["sift1M"]["search"]):
            if it not in cr:
                fail.append(f"{k}: missing itopk {it}"); continue
            if cag_tab is None:
                if own_tab and it in own_tab and cr[it] < own_tab[it] - otol:
                    fail.append(f"{k} itopk {it}: recall {cr[it]:.4f} < own best {own_tab[it]:.4f} - {otol}")
                continue
            # tolerance follows the path the run took (sift10M auto → large path)
            cag_floor = cag_tab[it] - CAGRA_TOL["large" if cand[k]["path"] == "large" else "auto"]
            if cr[it] < cag_floor:
                fail.append(f"{k} itopk {it}: recall {cr[it]:.4f} < {ref_name} floor {cag_floor:.4f}")

    for k in LARGE_KEYS:
        if k not in best["runs"]:
            continue
        c, b = cand[k], best["runs"][k]
        for st, v in c["peak_rss_gb"].items():
            if st in b["peak_rss_gb"] and v > MEM_TOL * b["peak_rss_gb"][st] + 0.2:
                fail.append(f"{k} {st}: peak RSS {v:.2f}GB > 1.2x best {b['peak_rss_gb'][st]:.2f}GB")
        if c["gpu_peak_mib"] and b["gpu_peak_mib"] and c["gpu_peak_mib"] > MEM_TOL * b["gpu_peak_mib"] + 256:
            fail.append(f"{k}: GPU peak {c['gpu_peak_mib']}MiB > 1.2x best {b['gpu_peak_mib']}MiB")
else:
    notes.append("no best.json yet: this run becomes the baseline")

verdict = "ACCEPT" if not fail else "REJECT"
print(json.dumps({"verdict": verdict, "fail": fail, "notes": notes,
                  "speedup_vs_best": {k: round(v * 100, 2) for k, v in speed.items()}}, indent=1))

if "--update" in flags and "--add-new" in flags and best and verdict != "ACCEPT":
    new_keys = [k for k in cand if k not in best["runs"]]
    for k in new_keys:
        best["runs"][k] = cand[k]
        best.setdefault("dirs", {})[k] = cand_dirs[k]
        best.setdefault("baseline_runs", dict(best["runs"])).setdefault(k, cand[k])
    json.dump(best, open(BEST, "w"), indent=1)
    print(f"state/best.json: added {new_keys}", file=sys.stderr)
elif "--update" in flags and (verdict == "ACCEPT" or best is None):
    runs = dict(best["runs"]) if best else {}
    runs.update(cand)   # a 2-config judge keeps the stored sift10M entry
    dirs = dict(best.get("dirs", {})) if best else {}
    dirs.update(cand_dirs)
    new = {"runs": runs, "dirs": dirs,
           "initial_large_recall": ({k: initial_floor(best, k) or recalls(cand[k])
                                     for k in ("sift1M_large",)} if best else
                                    {k: recalls(cand[k]) for k in ("sift1M_large",)})}
    if best:
        new["baseline_runs"] = dict(best.get("baseline_runs", best["runs"]))
        for k in cand:   # first time a config is recorded it is its own baseline
            new["baseline_runs"].setdefault(k, cand[k])
    json.dump(new, open(BEST, "w"), indent=1)
    print("state/best.json updated", file=sys.stderr)

if "--md" in flags:
    base = (best or {}).get("baseline_runs", (best or {}).get("runs"))
    print("\n**Build time (s, median; sum of all stages)**\n")
    print("| config | path | total | vs best | vs baseline | vs TAGORE | vs CAGRA | stages | GPU peak MiB | max RSS GB |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in KEYS:
        m = cand[k]; ds = k.split("_")[0]
        vb = f"{speed[k] * 100:+.1f}%" if k in speed else "—"
        vbase = f"{(base[k]['build_total_s'] - m['build_total_s']) / base[k]['build_total_s'] * 100:+.1f}%" if base and k in base else "—"
        vc = f"{m['build_total_s'] / CAGRA[ds]['build_median_s']:.2f}x" if ds in CAGRA else "—"
        vt = (f"{m['build_total_s'] / TAGORE[ds]['build_median_s']:.2f}x"
              if TAGORE.get(ds, {}).get("build_median_s") else "—")
        stg = ", ".join(f"{s} {v:.2f}" for s, v in m["stages_s"].items())
        print(f"| {k} | {m['path']} | {m['build_total_s']:.2f} | {vb} | {vbase} | {vt} | {vc} | {stg} | "
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
    print("\n**(b) final graph (degree 32, original ids), recall@10 / QPS (10K queries); sift1M/10M: cuVS CAGRA search, sift100M: DiskANN CPU search (itopk = L)**\n")
    its = list(CAGRA["sift1M"]["search"])
    print("| graph | " + " | ".join(f"itopk {i}" for i in its) + " |")
    print("|---|" + "---|" * len(its))
    for ds in ("sift1M", "sift10M", "sift100M"):
        if "search" in TAGORE.get(ds, {}):
            print(f"| TAGORE {TAGORE[ds].get('label', '')}, {ds} | " + " | ".join(
                f"{TAGORE[ds]['search'][i]:.4f} / {TAGORE[ds]['qps'][i] / 1e3:.0f}K"
                if i in TAGORE[ds]['search'] else "—" for i in its) + " |")
    for ds in ("sift1M", "sift10M"):
        if "search" in CAGRA.get(ds, {}):
            print(f"| CAGRA native 64→32, {ds} | " + " | ".join(
                f"{CAGRA[ds]['search'][i]:.4f} / {CAGRA[ds]['qps'][i] / 1e3:.0f}K" for i in its) + " |")
    for k in KEYS:
        pts = {str(p["itopk"]): p for p in cand[k].get("search", {}).get("points", [])}
        if pts:
            print(f"| ours {k} | " + " | ".join(
                f"{pts[i]['recall']:.4f} / {pts[i]['qps'] / 1e3:.0f}K" if i in pts else "—" for i in its) + " |")
