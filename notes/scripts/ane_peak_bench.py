#!/usr/bin/env python3
"""ANE fp16 GEMM peak: dispatch vs pure execute, chunk sweep, chain overlap.

Two timings per shape:

  dispatch - `net(x)`: host writes the input into the port buffer and reads the
             output back. This is what a caller pays, and it is what the earlier
             notes measured; on wide-N shapes it is host-path-bound (~7-10 GB/s),
             not engine-bound.
  execute  - the input is written once through `input_view`, then `execute()` is
             repeated. This is the pure engine rate.

Shape of the result (M4 Pro, see notes/ane-gemm-roofline.md): the engine holds
~13.4-15.1 TF/s of fp16 GEMM; the same programs measure 7.7-10.1 TF/s through
`net(x)`.

Run from the repo root (needs the built dylib and an ANE):
  PYTHONPATH=. python3 notes/scripts/ane_peak_bench.py
  PYTHONPATH=. python3 notes/scripts/ane_peak_bench.py --chains
"""
from __future__ import annotations

import time

import numpy as np

import aneforge as af

REPS, WARMUP = 8, 3

SHAPES = [
  (2048, 3072, 3072),
  (2048, 3072, 9216),
  (2048, 3072, 27648),
  (2048, 12288, 3072),
  (1024, 3072, 27648),
  (4096, 3072, 9216),
  (8192, 3072, 3072),
]
CHUNK_SWEEP = [(2048, 3072, 27648), (512, 1024, 2048, 4096)]
CHAIN_SHAPE = (2048, 3072, 13824)


def rand(*shape: int, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return (rng.standard_normal(shape).astype(np.float32) / np.float32(np.sqrt(shape[-1]))).astype(np.float16)


def build(M: int, K: int, N: int, chunk: int = 1024, chains: int = 1):
  """Compile one program holding `chains` independent [M,K]x[K,N] GEMMs, summed."""
  weights = [rand(N, K, seed=i) for i in range(chains)]
  x = rand(M, K)
  xs = af.input((M, K))
  outs = []
  for W in weights:
    if chunk >= K:
      outs.append(xs.linear(W))
    else:
      parts = []
      for i in range(0, K, chunk):
        w = min(chunk, K - i)
        parts.append(xs.slice_by_size([0, i], [M, w]).linear(W[:, i:i + w]))
      y = parts[0]
      for p in parts[1:]:
        y = y + p
      outs.append(y)
  out = outs[0]
  for o in outs[1:]:
    out = out + o
  return af.compile(out), x, weights[0]


def time_dispatch(net, x: np.ndarray) -> float:
  for _ in range(WARMUP):
    net(x)
  best = float("inf")
  for _ in range(REPS):
    t = time.perf_counter()
    net(x)
    best = min(best, time.perf_counter() - t)
  return best


def time_execute(net, x: np.ndarray) -> float:
  name, shape = net._inputs[0]  # private, like the other notes scripts' internals
  net.input_view(name)[...] = x.reshape(shape)
  for _ in range(WARMUP):
    net.execute()
  best = float("inf")
  for _ in range(REPS):
    t = time.perf_counter()
    net.execute()
    best = min(best, time.perf_counter() - t)
  return best


def row(M: int, K: int, N: int, chunk: int = 1024, chains: int = 1) -> tuple[float, float]:
  net, x, _ = build(M, K, N, chunk, chains)
  flops = chains * 2.0 * M * N * K
  td = time_dispatch(net, x)
  te = time_execute(net, x)
  print(
    f"  M={M:5d} K={K:5d} N={N:6d} ch={chunk:4d} chains={chains}"
    f"  dispatch {td * 1e3:8.2f} ms {flops / td / 1e12:6.2f} TF/s"
    f"  execute {te * 1e3:8.2f} ms {flops / te / 1e12:6.2f} TF/s",
    flush=True,
  )
  net.release()
  return flops / td / 1e12, flops / te / 1e12


def check_numerics() -> None:
  """One fp64 cross-check (smallest shape only - the fp64 reference is expensive)."""
  M, K, N = SHAPES[0]
  net, x, W = build(M, K, N)
  got = np.asarray(net(x), dtype=np.float64)
  ref = x.astype(np.float64) @ W.astype(np.float64).T
  relerr = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
  print(f"  数值自检 M={M} K={K} N={N}: relerr {relerr:.1e} (fp16 累积, ~1e-02 量级)\n")
  net.release()


def main() -> None:
  print(f"负载 {__import__('os').getloadavg()[0]:.2f};REPS={REPS} 取最小,先预热 {WARMUP} 次\n")
  check_numerics()
  best_d = best_e = 0.0
  print("[形状扫描,split-K 1024]")
  for M, K, N in SHAPES:
    d, e = row(M, K, N)
    best_d, best_e = max(best_d, d), max(best_e, e)
  print("\n[split-K 块大小]")
  M, K, N = CHUNK_SWEEP[0]
  for chunk in CHUNK_SWEEP[1]:
    d, e = row(M, K, N, chunk)
    best_d, best_e = max(best_d, d), max(best_e, e)
  print("\n[同程序里的独立 GEMM 链(是否并行;chains>1 时 I/O 被摊薄,"
        "dispatch TF/s 会虚高,只看 execute)]")
  M, K, N = CHAIN_SHAPE
  for chains in (1, 2, 4):
    row(M, K, N, 1024, chains)
  print(f"\n最高(单 GEMM 形状):纯 execute {best_e:.2f} TF/s;含下发 {best_d:.2f} TF/s")


if __name__ == "__main__":
  main()
