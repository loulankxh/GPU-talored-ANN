#!/usr/bin/env python3
"""Prepare one benchmark dataset under /home/lanlu/microBenchmark/dataset/<name>/ from a source base / query.

usage: prep_dataset.py NAME SRC_BASE SRC_QUERY [--n N] [--k 100] [--self-k 64] [--device cuda:2]
  base.<N>.<ext>               hard link of SRC_BASE when N is the full size, else its first N vectors
  query.public.10K.<ext>       hard link of SRC_QUERY
  groundtruth.ibin             exact top-k (k=100) of the queries   (gen_gt.py, big-ann format)
  base_knn_sample10K_k64.ibin  exact top-64 of 10K sampled base points (knn_similarity.py input)
Both GT files come from one pass of gen_gt.py. Run it under the bench lock (it reads the whole base).
Ext / element type: .u8bin uint8, .i8bin int8, .fbin float32.
"""
import argparse, os, subprocess, sys
import numpy as np

ROOT = "/home/lanlu/microBenchmark/dataset"
TORCH_PY = "/home/lanlu/miniconda3/envs/torch/bin/python"
ITEM = {".u8bin": 1, ".i8bin": 1, ".fbin": 4}

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("src_base"); ap.add_argument("src_query")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--k", type=int, default=100); ap.add_argument("--self-k", type=int, default=64)
ap.add_argument("--device", default="cuda:2")
a = ap.parse_args()

ext = os.path.splitext(a.src_base)[1]
if ext not in ITEM or os.path.splitext(a.src_query)[1] != ext:
    sys.exit(f"base / query must both be one of {list(ITEM)}")
n_src, d = (int(x) for x in np.fromfile(a.src_base, dtype=np.int32, count=2))
n = a.n or n_src
if n > n_src:
    sys.exit(f"--n {n} > source size {n_src}")
tag = f"{n // 1_000_000_000}B" if n % 1_000_000_000 == 0 else f"{n // 1_000_000}M" if n % 1_000_000 == 0 else str(n)
out = os.path.join(ROOT, a.name); os.makedirs(out, exist_ok=True)
base = os.path.join(out, f"base.{tag}{ext}")
query = os.path.join(out, f"query.public.10K{ext}")

if not os.path.exists(base):
    if n == n_src:
        os.link(a.src_base, base)
    else:   # first n vectors, streamed in 256 MB pieces
        row = d * ITEM[ext]
        with open(a.src_base, "rb") as fi, open(base + ".tmp", "wb") as fo:
            np.array([n, d], dtype=np.int32).tofile(fo)
            fi.seek(8); left = n * row
            while left:
                buf = fi.read(min(left, 256 << 20)); fo.write(buf); left -= len(buf)
        os.replace(base + ".tmp", base)
    print("base", base, flush=True)
if not os.path.exists(query):
    os.link(a.src_query, query)

gt = os.path.join(out, "groundtruth.ibin")
knn = os.path.join(out, f"base_knn_sample10K_k{a.self_k}.ibin")
if not (os.path.exists(gt) and os.path.exists(knn)):
    subprocess.run([TORCH_PY, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_gt.py"), base,
                    "--query", query, "--out", gt, "--k", str(a.k),
                    "--self-sample", "10000", "--self-out", knn, "--self-k", str(a.self_k),
                    "--device", a.device], check=True)
print("done", out, flush=True)
