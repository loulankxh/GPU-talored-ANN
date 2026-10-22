#!/usr/bin/env python3
"""Step B helper: TAGORE graph -> cagra_eval / diskann_search.py graph format
(int64 N, int32 degree, uint32 ids[N*degree]).

usage: tagore_graph_to_cagra.py IN OUT {cagra|vamana} [--degree 32]
  cagra : StoreCAGRA (per node u32 degree + degree ids, fixed degree)
  vamana: DiskANN in-memory index (u64 size, u32 max_degree, u32 start, u64 frozen, per node u32 k + k ids);
          rows shorter than --degree are padded: --pad invalid (0xFFFFFFFF, dropped by diskann_search.py)
          or --pad repeat (the row's last neighbor again, safe for cagra_eval; adds duplicate edges).
prints JSON: n, degree, padded slots, start (vamana), and defects of the real edges (padding excluded):
bad_ids, self_loops, dup_edges.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("fmt", choices=["cagra", "vamana"])
ap.add_argument("--degree", type=int, default=32); ap.add_argument("--pad", choices=["invalid", "repeat"], default="invalid")
a = ap.parse_args()
raw = np.fromfile(a.inp, dtype=np.uint32)
info = {}
if a.fmt == "cagra":
    d = int(raw[0]); n = raw.size // (d + 1)
    assert raw.size == n * (d + 1), "StoreCAGRA size mismatch"
    rows = raw.reshape(n, d + 1)
    assert (rows[:, 0] == d).all(), "variable degree in a CAGRA file"
    G = rows[:, 1:]; pad = 0; padm = np.zeros(G.shape, dtype=bool)
else:
    hdr = np.fromfile(a.inp, dtype=np.uint8, count=24)
    maxd, start = (int(x) for x in hdr[8:16].view(np.uint32))
    info["start"] = start; info["max_degree"] = maxd
    body = raw[6:]
    d = a.degree; offs = []; p = 0
    while p < body.size:
        offs.append(p); p += 1 + int(body[p])
    n = len(offs)
    G = np.full((n, d), 0xFFFFFFFF, dtype=np.uint32)
    for i, o in enumerate(offs):
        k = min(int(body[o]), d); G[i, :k] = body[o + 1:o + 1 + k]
    padm = G == 0xFFFFFFFF; pad = int(padm.sum())
# defects of the real edges
real = np.where(padm, np.uint32(0xFFFFFFFF), G)
bad = int(((real >= n) & ~padm).sum()); selfl = int((real == np.arange(n, dtype=np.uint32)[:, None]).sum())
srt = np.sort(real, axis=1); dup = int(((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] != 0xFFFFFFFF)).sum())
info.update(bad_ids=bad, self_loops=selfl, dup_edges=dup)
if a.fmt == "vamana" and a.pad == "repeat":
    deg = (~padm).sum(1); assert (deg > 0).all(), "empty row cannot be repeat-padded"
    last = G[np.arange(n), deg - 1]
    G = np.where(padm, last[:, None], G)
with open(a.out, "wb") as f:
    f.write(np.int64(n).tobytes()); f.write(np.int32(d).tobytes()); f.write(np.ascontiguousarray(G).tobytes())
info.update(n=int(n), degree=int(d), padded_slots=pad)
print(json.dumps(info))
