#!/usr/bin/env python3
"""Square fp16 GEMM sweep on the ANE: min/median latency and Frobenius relerr per N.

Run from the repo root: PYTHONPATH=. python3 notes/scripts/ane_gemm_sweep.py [N ...]
Results and interpretation: notes/ane-gemm-roofline.md
"""
from __future__ import annotations

import sys
import time

import numpy as np

import aneforge as af

REPS, WARMUP = 20, 5


def relerr(got, ref):
  got = np.asarray(got, dtype=np.float64).ravel()
  ref = np.asarray(ref, dtype=np.float64).ravel()
  return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def bench(N, reps=REPS, warmup=WARMUP):
  rng = np.random.default_rng(0)
  x32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))
  W32 = (rng.standard_normal((N, N)).astype(np.float32) / np.float32(np.sqrt(N)))
  net = af.compile(af.input((N, N)).linear(W32.astype(np.float16)))
  xf = x32.astype(np.float16)
  for _ in range(warmup): net(xf)
  ts = []
  for _ in range(reps):
    t = time.perf_counter(); out = net(xf); ts.append(time.perf_counter() - t)
  ts = np.array(ts)
  flops = 2.0 * N * N * N
  rblk = min(N, 128)
  ref = x32[:rblk].astype(np.float64) @ W32.astype(np.float64).T
  err = relerr(out[:rblk], ref)
  print(f"N={N:5d}  min {ts.min()*1e3:9.3f} ms  med {np.median(ts)*1e3:9.3f} ms  "
        f"max {ts.max()*1e3:9.3f} ms  |  {flops/ts.min()/1e12:6.3f} TF/s (min)  "
        f"{flops/np.median(ts)/1e12:6.3f} TF/s (med)  |  relerr(128 rows) {err:.2e}", flush=True)


if __name__ == "__main__":
  for n in [int(a) for a in sys.argv[1:]] or [512, 1024, 2048, 3072, 4096]:
    bench(n)
