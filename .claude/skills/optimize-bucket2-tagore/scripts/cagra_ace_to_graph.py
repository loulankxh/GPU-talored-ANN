#!/usr/bin/env python3
"""P1 helper: cuVS CAGRA ACE disk-mode artifacts -> our forward-graph format (int64 N, int32 K, uint32 ids).

usage: cagra_ace_to_graph.py ACE_DIR OUT
ACE disk mode leaves ACE_DIR/cagra_graph.npy (rows and neighbor ids in the reordered id space) and
ACE_DIR/dataset_mapping.npy (reordered row -> original id, as cuVS's HNSW export reads it):
out[map[r]] = map[graph[r]]. Prints JSON: n, degree, seconds, defects (bad ids, self loops, dup edges).
"""
import json, os, sys, time
import numpy as np

d, out = sys.argv[1:3]
t0 = time.time()
g = np.load(os.path.join(d, "cagra_graph.npy"), mmap_mode="r")
m = np.load(os.path.join(d, "dataset_mapping.npy")).astype(np.int64)
n, k = g.shape
assert m.shape == (n,) and m.min() >= 0 and m.max() < n
res = np.empty((n, k), dtype=np.uint32)
bad = self_l = dup = 0
CH = 4_000_000
for s in range(0, n, CH):
    rows = np.asarray(g[s:s + CH]).astype(np.int64)
    ok = rows < n
    bad += int((~ok).sum())
    mapped = np.where(ok, m[np.where(ok, rows, 0)], 0xFFFFFFFF).astype(np.uint32)
    orig = m[s:s + CH]
    self_l += int((mapped == orig[:, None].astype(np.uint32)).sum())
    srt = np.sort(mapped, axis=1)
    dup += int((srt[:, 1:] == srt[:, :-1]).sum())
    res[orig] = mapped
with open(out, "wb") as f:
    f.write(np.int64(n).tobytes()); f.write(np.int32(k).tobytes()); res.tofile(f)
print(json.dumps({"n": int(n), "degree": int(k), "convert_s": round(time.time() - t0, 2),
                  "bad_ids": bad, "self_loops": self_l, "dup_edges": dup}))
