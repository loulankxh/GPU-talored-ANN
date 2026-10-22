#!/usr/bin/env python3
"""Step B helper: our u8bin base (int32 N, int32 D, uint8 data) -> float .fvecs for TAGORE's load_data
(per row: int32 D, then D float32). Written in chunks; the time is reported separately from TAGORE's build.

usage: tagore_u8bin_to_fvecs.py BASE.u8bin OUT.fvecs
"""
import sys, time
import numpy as np

src, dst = sys.argv[1:3]
t0 = time.time()
n, d = (int(x) for x in np.fromfile(src, dtype=np.int32, count=2))
B = np.memmap(src, dtype=np.uint8, mode="r", offset=8, shape=(n, d))
CH = 1_000_000
with open(dst, "wb") as f:
    for s in range(0, n, CH):
        b = np.asarray(B[s:s + CH])
        out = np.empty((b.shape[0], d + 1), dtype=np.float32)
        out[:, 1:] = b
        out.view(np.int32)[:, 0] = d
        f.write(out.tobytes())
print(f"converted {n} x {d} in {time.time() - t0:.2f} s", file=sys.stderr)
