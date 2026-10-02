#!/usr/bin/env python3
"""Monolithic vs split-K GEMM on the ANE, with an fp64 correctness check.

Slices the activation along K into chunks, runs one linear per chunk and adds the
partial sums. Recovers ~5x on K > 4096 shapes with numerically identical output.

Run from the repo root:
  PYTHONPATH=. python3 notes/scripts/ane_split_k_bench.py [M K N]
  PYTHONPATH=. python3 notes/scripts/ane_split_k_bench.py --k-sweep [M N]
Results and interpretation: notes/large-k-cliff.md
"""
from __future__ import annotations

import sys
import time

import numpy as np

import aneforge as af

CHUNKS = (512, 1024, 2048, 4096)
REPS, WARMUP = 8, 3


def relerr(got, ref):
  got = np.asarray(got, dtype=np.float64).ravel()
  ref = np.asarray(ref, dtype=np.float64).ravel()
  return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def build(M, K, N, chunk, seed=0):
  """Return (net, x, W). chunk=None -> one monolithic linear."""
  rng = np.random.default_rng(seed)
  W = (rng.standard_normal((N, K)).astype(np.float32) / np.float32(np.sqrt(K))).astype(np.float16)
  x = (rng.standard_normal((M, K)).astype(np.float32) / np.float32(np.sqrt(K))).astype(np.float16)
  if chunk is None or chunk >= K:
    return af.compile(af.input((M, K)).linear(W)), x, W
  xs = af.input((M, K)); parts = []
  for i in range(0, K, chunk):
    w = min(chunk, K - i)
    parts.append(xs.slice_by_size([0, i], [M, w]).linear(W[:, i:i + w]))
  y = parts[0]
  for p in parts[1:]: y = y + p
  return af.compile(y), x, W


def run(M, K, N, chunk, reps=REPS):
  net, x, W = build(M, K, N, chunk)
  for _ in range(WARMUP): net(x)
  best = float("inf")
  for _ in range(reps):
    t = time.perf_counter(); net(x); best = min(best, time.perf_counter() - t)
  ref = x.astype(np.float64) @ W.astype(np.float64).T
  return best, relerr(net(x), ref), 2.0 * M * N * K


def compare(M, K, N):
  flops = 2.0 * M * N * K
  tm, rm, _ = run(M, K, N, None)
  print(f"M={M} K={K} N={N}:  mono {tm*1e3:8.2f} ms {flops/tm/1e12:6.2f} TF/s  relerr {rm:.1e}")
  best = (tm, None)
  for ch in CHUNKS:
    t, r, _ = run(M, K, N, ch)
    print(f"    chunk {ch:5d}: {t*1e3:8.2f} ms {flops/t/1e12:6.2f} TF/s  relerr {r:.1e}"
          f"  ({tm/t:.2f}x)")
    if t < best[0]: best = (t, ch)
  print(f"    best: chunk={best[1]}  {tm/best[0]:.2f}x\n")


def k_sweep(M, N, ks=(1024, 2048, 4096, 8192, 12288, 16384)):
  print(f"K sweep at M={M} N={N}, monolithic:")
  for K in ks:
    t, r, flops = run(M, K, N, None)
    print(f"  K={K:6d}  {t*1e3:9.3f} ms  {flops/t/1e12:5.2f} TF/s  relerr {r:.1e}")
  print()


if __name__ == "__main__":
  args = [a for a in sys.argv[1:] if not a.startswith("--")]
  if "--k-sweep" in sys.argv:
    M, N = (int(a) for a in (args or [2048, 3072]))
    k_sweep(M, N)
  else:
    M, K, N = (int(a) for a in (args or [2048, 12288, 3072]))
    compare(M, K, N)
