#!/usr/bin/env python3
"""Exact L2 ground truth on GPU (torch), big-ann format:
   int32 nq, int32 k, int32 ids[nq*k], float32 sq_dists[nq*k].
u8 SIFT squared distances are integers < 2^24, so fp32 is exact.

usage: gen_gt.py BASE.u8bin QUERY.u8bin OUT.ibin [--k 100] [--chunk 500000]
       gen_gt.py BASE.u8bin - OUT.ibin --self-sample 10000 --k 64
  --self-sample S: queries = S random base points (seed 0), self excluded;
  output then also carries the sample ids:
   int32 S, int32 k, int32 sample_ids[S], int32 ids[S*k], float32 sq_dists[S*k].
"""
import argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("query"); ap.add_argument("out")
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--chunk", type=int, default=500_000)
ap.add_argument("--self-sample", type=int, default=0)
a = ap.parse_args()

def load(p):
    n, d = np.fromfile(p, dtype=np.int32, count=2)
    return np.memmap(p, dtype=np.uint8, mode="r", offset=8, shape=(int(n), int(d)))

B = load(a.base)
sample = None
if a.self_sample:
    sample = np.sort(np.random.default_rng(0).choice(B.shape[0], a.self_sample, replace=False))
    Q = B[sample]
    a.k += 1  # one slot for the point itself, dropped below
else:
    Q = load(a.query)
dev = torch.device("cuda:0")
q = torch.from_numpy(np.asarray(Q, dtype=np.float32)).to(dev)
qn = (q * q).sum(1, keepdim=True)
best_d = torch.full((q.shape[0], a.k), float("inf"), device=dev)
best_i = torch.zeros((q.shape[0], a.k), dtype=torch.int64, device=dev)
for s in range(0, B.shape[0], a.chunk):
    b = torch.from_numpy(np.asarray(B[s:s + a.chunk], dtype=np.float32)).to(dev)
    bn = (b * b).sum(1)
    for qs in range(0, q.shape[0], 500):
        d = qn[qs:qs+500] + bn[None, :] - 2.0 * (q[qs:qs+500] @ b.T)
        dv, di = torch.topk(d, a.k, dim=1, largest=False)
        cd = torch.cat([best_d[qs:qs+500], dv], 1)
        ci = torch.cat([best_i[qs:qs+500], di + s], 1)
        o = torch.argsort(cd, dim=1)[:, :a.k]
        best_d[qs:qs+500] = torch.gather(cd, 1, o)
        best_i[qs:qs+500] = torch.gather(ci, 1, o)
    print(f"  {min(s + a.chunk, B.shape[0])}/{B.shape[0]}", flush=True)

ids = best_i.cpu().numpy()
dist = best_d.clamp_min(0).cpu().numpy().astype(np.float32)
if sample is not None:  # drop self (ties at distance 0 may put it anywhere in the row)
    keep = ids != sample[:, None]
    keep[keep.sum(1) > a.k - 1, -1] = False  # self not found (exact duplicate): drop last
    a.k -= 1
    ids = ids[keep].reshape(-1, a.k); dist = dist[keep].reshape(-1, a.k)
with open(a.out, "wb") as f:
    np.array([q.shape[0], a.k], dtype=np.int32).tofile(f)
    if sample is not None:
        sample.astype(np.int32).tofile(f)
    ids.astype(np.int32).tofile(f)
    dist.tofile(f)
print("wrote", a.out)
