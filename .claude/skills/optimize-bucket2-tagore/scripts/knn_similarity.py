#!/usr/bin/env python3
"""Metric (a): how close bucket2's pre-prune index (vector_knn.bin, degree 64,
reordered id space) is to the exact KNN graph, on 10K sampled base points.

usage: knn_similarity.py vector_knn.bin inverse_perm.bin base_knn_sample10K_k64.ibin
prints JSON: recall@{1,10,32,64} = |approx top-k ∩ exact top-k| / k, plus padding rate.
"""
import json, sys, numpy as np

knn_p, inv_p, gt_p = sys.argv[1:4]
N, M = int(np.fromfile(knn_p, dtype=np.int64, count=1)[0]), int(np.fromfile(knn_p, dtype=np.int32, offset=8, count=1)[0])
knn = np.memmap(knn_p, dtype=np.int32, mode="r", offset=12, shape=(N, M))
inv = np.fromfile(inv_p, dtype=np.uint32)  # inverse_perm[new_id] = orig_id (raw, no header)
if inv.size != N:  # tolerate an int64 N header
    inv = np.fromfile(inv_p, dtype=np.uint32, offset=8)
assert inv.size == N, (inv.size, N)
perm = np.empty(N, dtype=np.int64); perm[inv] = np.arange(N)  # orig -> new

S, k = np.fromfile(gt_p, dtype=np.int32, count=2)
sid = np.fromfile(gt_p, dtype=np.int32, offset=8, count=S)
gt = np.fromfile(gt_p, dtype=np.int32, offset=8 + 4 * S, count=S * k).reshape(S, k)

rows = np.asarray(knn[perm[sid]])                     # new-id neighbor lists of sampled points
pad = rows < 0
cand = np.where(pad, -1, inv[np.clip(rows, 0, N - 1)].astype(np.int64))  # -> original ids
out = {"n": int(N), "degree": int(M), "samples": int(S), "padding_rate": float(pad.mean())}
for kk in (1, 10, 32, 64):
    if kk > min(M, k): continue
    hit = sum(len(np.intersect1d(cand[i, :kk], gt[i, :kk], assume_unique=False)) for i in range(S))
    out[f"recall@{kk}"] = round(hit / (S * kk), 5)
print(json.dumps(out))
