#!/usr/bin/env python3
"""Step B helper: our u8bin base (int32 N, int32 D, uint8 data) -> float .fvecs for TAGORE's load_data
(per row: int32 D, then D float32). Written in chunks; the time is reported separately from TAGORE's build.

usage: tagore_u8bin_to_fvecs.py BASE.{u8bin,i8bin,fbin} OUT.fvecs [--pad-dim 128]
--pad-dim appends zero columns (L2 distances unchanged): TAGORE's kernels assume D = DIM_SIZE (128)
(cal_power reduces one warp per 32 dims, the GEMM tiles D in SS = 128 chunks), so D = 100 data is padded.
"""
import sys, time
import numpy as np

src, dst = sys.argv[1:3]
pad = int(sys.argv[sys.argv.index("--pad-dim") + 1]) if "--pad-dim" in sys.argv else 0
dt = np.float32 if src.endswith(".fbin") else np.int8 if src.endswith(".i8bin") else np.uint8
t0 = time.time()
n, d0 = (int(x) for x in np.fromfile(src, dtype=np.int32, count=2))
B = np.memmap(src, dtype=dt, mode="r", offset=8, shape=(n, d0))
d = max(d0, pad)
CH = 1_000_000
with open(dst, "wb") as f:
    for s in range(0, n, CH):
        b = np.asarray(B[s:s + CH])
        out = np.zeros((b.shape[0], d + 1), dtype=np.float32)
        out[:, 1:d0 + 1] = b
        out.view(np.int32)[:, 0] = d
        f.write(out.tobytes())
print(f"converted {n} x {d} in {time.time() - t0:.2f} s", file=sys.stderr)
