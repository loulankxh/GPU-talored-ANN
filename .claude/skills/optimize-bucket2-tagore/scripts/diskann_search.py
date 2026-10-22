#!/usr/bin/env python3
"""SIFT100M search for our final graph with DiskANN's CPU in-memory searcher
(cuVS CAGRA search does not fit 100M on a 16 GB GPU).

usage: diskann_search.py BASE.u8bin final_graph.bin QUERY.u8bin GT.ibin WORKDIR [--L 32,64,128,256,512] [--K 10]

1. reads our graph (int64 N, int32 K, uint32 graph[N*K]; graph_io.hpp write_graph) and checks it:
   out-of-range ids, self loops, duplicate edges per row (same counters as cagra_eval);
2. entry point = medoid (base point nearest to the mean, exact over all N);
3. writes DiskANN's in-memory graph format (in_mem_graph_store.cpp load_impl:
   u64 file_size, u32 max_degree, u32 start, u64 frozen_pts=0, then per node u32 k + k ids)
   to WORKDIR/diskann_index, symlinks WORKDIR/diskann_index.data -> BASE;
4. runs search_memory_index (uint8, l2, all threads) for every L, recall@K vs GT;
prints one JSON line in cagra_eval's search format:
   {"degree", "bad_ids", "self_loops", "dup_edges", "points": [{"itopk": L, "recall", "qps"}]}
(points[].itopk holds DiskANN's L so judge.py / summarize.py can read it unchanged).
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np

SEARCH = "/home/lanlu/DiskANN/build/apps/search_memory_index"
ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("graph"); ap.add_argument("query"); ap.add_argument("gt"); ap.add_argument("work")
ap.add_argument("--L", default="32,64,128,256,512"); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--entry", type=int, default=None)   # entry point override (TAGORE step B: its own ep); default medoid
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
log = lambda *x: print(*x, file=sys.stderr, flush=True)

N = int(np.fromfile(a.graph, dtype=np.int64, count=1)[0])
K = int(np.fromfile(a.graph, dtype=np.int32, count=1, offset=8)[0])
assert os.path.getsize(a.graph) == 12 + 4 * N * K, "graph file size mismatch"
G = np.memmap(a.graph, dtype=np.uint32, mode="r", offset=12, shape=(N, K))
bn, bd = (int(x) for x in np.fromfile(a.base, dtype=np.int32, count=2))
assert bn == N, f"graph N {N} != base N {bn}"
B = np.memmap(a.base, dtype=np.uint8, mode="r", offset=8, shape=(bn, bd))

# ---- checks + DiskANN graph file (one pass over the graph)
t0 = time.time()
CH = 2_000_000
bad = selfl = dup = 0
idx_path = os.path.join(a.work, "diskann_index")
with open(idx_path, "wb") as f:
    f.write(np.zeros(24, dtype=np.uint8).tobytes())   # header, filled in below
    size = 24
    for s in range(0, N, CH):
        g = np.asarray(G[s:s + CH])
        rows = np.arange(s, s + g.shape[0], dtype=np.uint32)[:, None]
        oob = g >= N
        bad += int(oob.sum())
        selfl += int((g == rows).sum())
        srt = np.sort(g, axis=1)
        dup += int((srt[:, 1:] == srt[:, :-1]).sum())
        if oob.any():   # drop out-of-range ids (variable degree rows)
            for r in range(g.shape[0]):
                v = g[r][g[r] < N]
                f.write(np.uint32(v.size).tobytes()); f.write(v.astype(np.uint32).tobytes())
                size += 4 + 4 * v.size
        else:
            out = np.empty((g.shape[0], K + 1), dtype=np.uint32)
            out[:, 0] = K; out[:, 1:] = g
            f.write(out.tobytes()); size += out.nbytes
log(f"[diskann_search] graph N={N} K={K} bad={bad} self={selfl} dup={dup} convert {time.time() - t0:.1f}s")

# ---- medoid
t1 = time.time()
if a.entry is not None:
    start = a.entry
else:
  mean = np.zeros(bd, dtype=np.float64)
  for s in range(0, N, CH):
    mean += np.asarray(B[s:s + CH], dtype=np.float64).sum(0)
  mean /= N
  best_d, start = np.inf, 0
  for s in range(0, N, CH):
    x = np.asarray(B[s:s + CH], dtype=np.float32)
    d = ((x - mean.astype(np.float32)) ** 2).sum(1)
    i = int(d.argmin())
    if d[i] < best_d: best_d, start = float(d[i]), s + i
log(f"[diskann_search] entry {start} ({'given' if a.entry is not None else 'medoid'}) ({time.time() - t1:.1f}s)")
with open(idx_path, "r+b") as f:
    f.write(np.uint64(size).tobytes()); f.write(np.uint32(K).tobytes())
    f.write(np.uint32(start).tobytes()); f.write(np.uint64(0).tobytes())
data_link = idx_path + ".data"
if os.path.lexists(data_link): os.remove(data_link)
os.symlink(os.path.abspath(a.base), data_link)

# ---- search
Ls = [int(x) for x in a.L.split(",")]
cmd = [SEARCH, "--data_type", "uint8", "--dist_fn", "l2", "--index_path_prefix", idx_path,
       "--result_path", os.path.join(a.work, "diskann_res"), "--query_file", a.query, "--gt_file", a.gt,
       "-K", str(a.K), "-L", *map(str, Ls)]
log("[diskann_search] " + " ".join(cmd))
t2 = time.time()
p = subprocess.run(cmd, capture_output=True, text=True)
sys.stderr.write(p.stdout[-4000:] + p.stderr[-2000:])
if p.returncode != 0:
    sys.exit(f"search_memory_index failed rc={p.returncode}")
# table rows: L  QPS  avg_cmps  mean_lat  p999_lat  recall
pts = []
for line in p.stdout.splitlines():
    f = line.split()
    if len(f) >= 6 and re.fullmatch(r"\d+", f[0]) and int(f[0]) in Ls:
        try:
            pts.append({"itopk": int(f[0]), "recall": float(f[-1]) / 100.0, "qps": float(f[1])})
        except ValueError:
            pass
log(f"[diskann_search] search {time.time() - t2:.1f}s")
for fn in (idx_path, data_link):
    os.remove(fn)
for fn in os.listdir(a.work):
    if fn.startswith("diskann_res"): os.remove(os.path.join(a.work, fn))
print(json.dumps({"searcher": "diskann_memory", "degree": K, "bad_ids": bad, "self_loops": selfl,
                  "dup_edges": dup, "entry": start, "points": pts}))
