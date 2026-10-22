#!/usr/bin/env python3
"""Step B helper: input files for Tagore.largeIndex from our u8bin base, written straight from the u8bin
(no full fvecs copy). Runs on the GPU (cupy), timed as TAGORE's "prep".

usage: tagore_prep_clusters.py BASE.u8bin OUTDIR [--clusters 100] [--max-points 2500000] [--sample 1000000]
                               [--iters 20] [--seed 0]
1. k-means (Lloyd, fp32) on a uniform sample -> centers;
2. every point -> its 2 nearest distinct centers (GPU GEMM, chunks of 2M);
3. clusters larger than max_points are split exactly as largeIndex would (round-robin over the members in id
   order, new ids cluster_num, cluster_num+1, ...), so largeIndex finds no oversized cluster and every cluster
   id has a data file (largeIndex's own split would point at files that do not exist);
4. OUTDIR/centers.bin: N x (u32 first, u32 second); OUTDIR/cluster<i>: u32 count + count x D float32,
   members in increasing id order (= largeIndex's clusters[i] order). Vectors are multiplied by the same
   power-of-10 norm_factor GNN_descent applies (largeIndex's f2h has none; raw SIFT values overflow fp16 in
   cal_power), which leaves neighbor order and the alpha test unchanged.
prints JSON: cluster count after split, sizes, norm_factor, per-phase seconds.
"""
import argparse, json, os, sys, time
import numpy as np, cupy as cp

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--clusters", type=int, default=100); ap.add_argument("--max-points", type=int, default=2_500_000)
ap.add_argument("--sample", type=int, default=1_000_000); ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
T = {}; t0 = time.time()
n, d = (int(x) for x in np.fromfile(a.base, dtype=np.int32, count=2))
B = np.memmap(a.base, dtype=np.uint8, mode="r", offset=8, shape=(n, d))

# norm_factor as in pybind.cu GNN_descent (from the first vector's mean |x|)
avg = float(np.abs(B[0].astype(np.float32)).mean()); nf = 1.0
while avg < 0.5: avg *= 10; nf *= 10
while avg > 5: avg /= 10; nf /= 10

# 1. k-means on a sample
rng = np.random.default_rng(a.seed)
sid = np.sort(rng.choice(n, min(a.sample, n), replace=False))
X = cp.asarray(B[sid].astype(np.float32))
C = X[cp.asarray(rng.choice(X.shape[0], a.clusters, replace=False))].copy()
for _ in range(a.iters):
    dist = (C * C).sum(1)[None, :] - 2.0 * X @ C.T
    lab = dist.argmin(1)
    cnt = cp.bincount(lab, minlength=a.clusters).astype(cp.float32)
    oh = cp.zeros((a.clusters, X.shape[0]), dtype=cp.float32); oh[lab, cp.arange(X.shape[0])] = 1
    S = oh @ X; del oh
    nz = cnt > 0; C[nz] = S[nz] / cnt[nz, None]
del X
T["kmeans_s"] = round(time.time() - t0, 2); t1 = time.time()

# 2. two nearest centers for every point
first = np.empty(n, dtype=np.uint32); second = np.empty(n, dtype=np.uint32)
cn = (C * C).sum(1)[None, :]
CH = 2_000_000
for s in range(0, n, CH):
    x = cp.asarray(np.asarray(B[s:s + CH]), dtype=cp.float32)
    dist = cn - 2.0 * x @ C.T
    top = cp.argpartition(dist, 1, axis=1)[:, :2]
    dd = cp.take_along_axis(dist, top, 1)
    sw = dd[:, 1] < dd[:, 0]
    f = cp.where(sw, top[:, 1], top[:, 0]); sc = cp.where(sw, top[:, 0], top[:, 1])
    first[s:s + CH] = cp.asnumpy(f); second[s:s + CH] = cp.asnumpy(sc)
T["assign_s"] = round(time.time() - t1, 2); t2 = time.time()

# 3. split oversized clusters (largeIndex's rule)
K0 = a.clusters
members = [[] for _ in range(K0)]
order = np.argsort(np.concatenate([first, second]), kind="stable")   # stable -> id order inside a cluster
labs = np.concatenate([first, second])[order]; ids = (order % n).astype(np.int64); isfirst = order < n
bounds = np.searchsorted(labs, np.arange(K0 + 1))
extra = 0
for c in range(K0):
    lo, hi = bounds[c], bounds[c + 1]; m = hi - lo
    if m <= a.max_points: continue
    div = 2
    while m // div > a.max_points: div += 1
    cand = np.array([c] + [K0 + extra + j for j in range(div - 1)], dtype=np.uint32)
    # members of c in increasing id order (a point is in c at most once: its two centers are distinct)
    mi = ids[lo:hi]; mf = isfirst[lo:hi]
    srt = np.argsort(mi, kind="stable"); mi, mf = mi[srt], mf[srt]
    newc = cand[np.arange(m) % div]
    first[mi[mf]] = newc[mf]; second[mi[~mf]] = newc[~mf]
    extra += div - 1
KC = K0 + extra
assert (first != second).all()
np.stack([first, second], 1).astype(np.uint32).tofile(os.path.join(a.out, "centers.bin"))
sizes = np.bincount(first, minlength=KC) + np.bincount(second, minlength=KC)
assert sizes.max() <= a.max_points, sizes.max()
T["split_s"] = round(time.time() - t2, 2); t3 = time.time()

# 4. per-cluster float files, streamed in id order
fs = []
for c in range(KC):
    f = open(os.path.join(a.out, f"cluster{c}"), "wb", buffering=8 << 20)
    f.write(np.uint32(sizes[c]).tobytes()); fs.append(f)
for s in range(0, n, CH):
    x = np.asarray(B[s:s + CH], dtype=np.float32) * np.float32(nf)
    lab2 = np.concatenate([first[s:s + CH], second[s:s + CH]]); rows = np.concatenate([np.arange(x.shape[0])] * 2)
    o = np.lexsort((rows, lab2)); lab2, rows = lab2[o], rows[o]
    cb = np.searchsorted(lab2, np.arange(KC + 1))
    for c in np.nonzero(np.diff(cb))[0]:
        fs[c].write(x[rows[cb[c]:cb[c + 1]]].tobytes())
for f in fs: f.close()
T["write_s"] = round(time.time() - t3, 2)
T["total_s"] = round(time.time() - t0, 2)
print(json.dumps({"n": n, "dim": d, "clusters_kmeans": K0, "clusters": int(KC), "max_points": a.max_points,
                  "size_min": int(sizes.min()), "size_max": int(sizes.max()), "size_mean": float(sizes.mean()),
                  "norm_factor": nf, **T}))
