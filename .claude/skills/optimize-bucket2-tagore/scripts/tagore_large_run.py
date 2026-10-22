#!/usr/bin/env python3
"""Step B helper: Tagore.largeIndex (Vamana, one GPU) on the files written by tagore_prep_clusters.py.

usage: tagore_large_run.py PREPDIR N CLUSTERS INDEX_OUT [--k 64] [--m 64] [--final-degree 32]
                           [--max-degree 32] [--alpha 1.2] [--max-points 2500000] [--buffer 30]
CLUSTERS = cluster count after the prep split (its JSON "clusters"). Local indexes go to PREPDIR/local<i>.
"""
import argparse, os, sys, time
sys.path.insert(0, os.environ.get("TAGORE_BUILD", "/home/lanlu/Tagore/build-v100/patched-build"))
import Tagore

ap = argparse.ArgumentParser()
ap.add_argument("prep"); ap.add_argument("n", type=int); ap.add_argument("clusters", type=int); ap.add_argument("index_out")
ap.add_argument("--k", type=int, default=64); ap.add_argument("--m", type=int, default=64)
ap.add_argument("--final-degree", type=int, default=32); ap.add_argument("--max-degree", type=int, default=32)
ap.add_argument("--alpha", type=float, default=1.2); ap.add_argument("--max-points", type=int, default=2_500_000)
ap.add_argument("--buffer", type=int, default=30)
a = ap.parse_args()
t0 = time.time()
Tagore.largeIndex([0], 128, a.k, a.clusters, a.max_points, os.path.join(a.prep, "cluster"),
                  os.path.join(a.prep, "local"), a.index_out, a.max_degree, a.buffer, a.n,
                  os.path.join(a.prep, "centers.bin"), a.alpha, a.final_degree, a.m)
print(f"PY_TIMERS largeIndex_call {time.time() - t0:.3f}", flush=True)
