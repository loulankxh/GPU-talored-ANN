#!/usr/bin/env python3
"""Step B helper: TAGORE in-memory build (GNN-Descent + CAGRA-mode pruning) on one GPU.

usage: tagore_run.py DATA.fvecs N K ITERS INDEX_OUT [--final-degree 32] [--m 64] [--dump-knn KNN.bin]

Calls Tagore.GNN_descent(K, N, 128, ITERS, fvecs) then Tagore.Pruning(K, N, 128, final_degree, m, ptrs,
'CAGRA', index_path=INDEX_OUT) (StoreCAGRA: per node u32 degree + ids; filter_reverse_1hop writes the
final rows with stride final_degree, so StoreCAGRA's i*degree indexing is right).
--dump-knn copies the pre-prune kNN graph (device graph, stride K) before pruning and writes it in
knn_similarity.py's format (int64 N, int32 K, int32 ids) + an identity inverse_perm next to it;
it costs a device->host copy, so timed runs leave it off.
Prints one JSON line with TAGORE's own stage timers (parsed from its stdout by the caller).
"""
import argparse, ctypes, os, sys, time
# patched copy (tagore_v100_fix.patch: shared-array bounds check in nn_descent_opt_reverse_sample, faults on sm_70)
sys.path.insert(0, os.environ.get("TAGORE_BUILD", "/home/lanlu/Tagore/build-v100/patched-build"))
import numpy as np
import Tagore

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("n", type=int); ap.add_argument("k", type=int)
ap.add_argument("iters", type=int); ap.add_argument("index_out")
ap.add_argument("--final-degree", type=int, default=32); ap.add_argument("--m", type=int, default=64)
ap.add_argument("--dump-knn", default=None)
a = ap.parse_args()

t0 = time.time()
ptrs = Tagore.GNN_descent(a.k, a.n, 128, a.iters, a.data)
t1 = time.time()
if a.dump_knn:
    cudart = ctypes.CDLL("/home/lanlu/miniconda3/envs/cuvs-bench/lib/libcudart.so.12")
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    g = np.empty((a.n, a.k), dtype=np.uint32)
    rc = cudart.cudaMemcpy(g.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(get(ptrs[0], None)),
                           ctypes.c_size_t(g.nbytes), 2)
    assert rc == 0, f"cudaMemcpy failed: {rc}"
    with open(a.dump_knn, "wb") as f:
        f.write(np.int64(a.n).tobytes()); f.write(np.int32(a.k).tobytes()); f.write(g.view(np.int32).tobytes())
    np.arange(a.n, dtype=np.uint32).tofile(a.dump_knn + ".inverse_perm")
t2 = time.time()
Tagore.Pruning(a.k, a.n, 128, a.final_degree, a.m, ptrs, "CAGRA", index_path=a.index_out)
t3 = time.time()
print(f"PY_TIMERS gnn_descent_call {t1 - t0:.3f} dump {t2 - t1:.3f} pruning_call {t3 - t2:.3f}", flush=True)
