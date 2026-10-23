#!/usr/bin/env python3
"""Exact L2 ground truth on GPU (torch), big-ann format, for uint8 / int8 / float32 base vectors.

  int32 nq, int32 k, int32 ids[nq*k], float32 sq_dists[nq*k]
  self-sample file: int32 S, int32 k, int32 sample_ids[S], int32 ids[S*k], float32 sq_dists[S*k]

Element type comes from the file extension: .u8bin uint8, .i8bin int8, .fbin float32.
Distances in fp32 (u8 / i8 SIFT-like squared distances are integers < 2^24, so exact).

usage: gen_gt.py BASE [--n N] [--query Q --out GT.ibin] [--self-sample S --self-out OUT.ibin]
                 [--k 100] [--self-k 64] [--chunk 500000] [--device cuda:0]
  --n N: use only the first N base vectors (subset datasets; ids are 0..N-1)
  both outputs are produced in ONE pass over the base file (the expensive part at 100M+).
"""
import argparse, os, time
import numpy as np, torch

DTYPES = {".u8bin": np.uint8, ".i8bin": np.int8, ".fbin": np.float32}

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--query"); ap.add_argument("--out")
ap.add_argument("--self-sample", type=int, default=0); ap.add_argument("--self-out")
ap.add_argument("--k", type=int, default=100); ap.add_argument("--self-k", type=int, default=64)
ap.add_argument("--chunk", type=int, default=500_000)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
if not a.query and not a.self_sample:
    ap.error("nothing to do: give --query/--out and/or --self-sample/--self-out")
if bool(a.query) != bool(a.out) or bool(a.self_sample) != bool(a.self_out):
    ap.error("--query needs --out and --self-sample needs --self-out")

def load(p):
    ext = os.path.splitext(p)[1]
    if ext not in DTYPES:
        raise SystemExit(f"unknown vector file type {ext} ({p})")
    n, d = (int(x) for x in np.fromfile(p, dtype=np.int32, count=2))
    dt = DTYPES[ext]
    if os.path.getsize(p) != 8 + n * d * np.dtype(dt).itemsize:   # bounds check
        raise SystemExit(f"{p}: size does not match header {n} x {d} {np.dtype(dt).name}")
    return np.memmap(p, dtype=dt, mode="r", offset=8, shape=(n, d))

B = load(a.base)
N = a.n or B.shape[0]
if N > B.shape[0]:
    raise SystemExit(f"--n {N} > base size {B.shape[0]}")
dev = torch.device(a.device)

jobs = []   # [name, queries (host fp32), k, sample ids or None, out path]
if a.query:
    Q = load(a.query)
    if Q.shape[1] != B.shape[1]:
        raise SystemExit(f"query dim {Q.shape[1]} != base dim {B.shape[1]}")
    jobs.append(["query", np.asarray(Q, dtype=np.float32), a.k, None, a.out])
if a.self_sample:
    sample = np.sort(np.random.default_rng(0).choice(N, a.self_sample, replace=False))
    jobs.append(["self", np.asarray(B[sample], dtype=np.float32), a.self_k + 1, sample, a.self_out])  # +1: self

state = []
for _, Qh, k, _, _ in jobs:
    q = torch.from_numpy(Qh).to(dev)
    state.append(dict(q=q, qn=(q * q).sum(1, keepdim=True), k=k,
                      d=torch.full((q.shape[0], k), float("inf"), device=dev),
                      i=torch.zeros((q.shape[0], k), dtype=torch.int64, device=dev)))

t0 = time.time()
for s in range(0, N, a.chunk):
    e = min(s + a.chunk, N)
    b = torch.from_numpy(np.asarray(B[s:e], dtype=np.float32)).to(dev)
    bn = (b * b).sum(1)
    for st in state:
        q, qn, k = st["q"], st["qn"], st["k"]
        for qs in range(0, q.shape[0], 500):
            d = qn[qs:qs + 500] + bn[None, :] - 2.0 * (q[qs:qs + 500] @ b.T)
            dv, di = torch.topk(d, min(k, e - s), dim=1, largest=False)
            cd = torch.cat([st["d"][qs:qs + 500], dv], 1)
            ci = torch.cat([st["i"][qs:qs + 500], di + s], 1)
            o = torch.argsort(cd, dim=1)[:, :k]
            st["d"][qs:qs + 500] = torch.gather(cd, 1, o)
            st["i"][qs:qs + 500] = torch.gather(ci, 1, o)
    if (s // a.chunk) % 20 == 0 or e == N:
        print(f"  {e}/{N}  {time.time() - t0:.0f}s", flush=True)

for (name, _, k, sample, out), st in zip(jobs, state):
    ids = st["i"].cpu().numpy()
    dist = st["d"].clamp_min(0).cpu().numpy().astype(np.float32)
    if sample is not None:   # drop self (ties at distance 0 may put it anywhere in the row)
        keep = ids != sample[:, None]
        keep[keep.sum(1) > k - 1, -1] = False   # self not found (exact duplicate): drop last
        k -= 1
        ids = ids[keep].reshape(-1, k); dist = dist[keep].reshape(-1, k)
    if ids.max() >= N or ids.min() < 0:   # bounds check
        raise SystemExit(f"{name}: id out of range")
    with open(out, "wb") as f:
        np.array([ids.shape[0], k], dtype=np.int32).tofile(f)
        if sample is not None:
            sample.astype(np.int32).tofile(f)
        ids.astype(np.int32).tofile(f)
        dist.tofile(f)
    print("wrote", out, ids.shape, flush=True)
