#!/usr/bin/env python3
"""FLUX.2-klein-4B: per-layer ANE GEMM cost at a given image resolution.

Pure shape simulation - does not read the checkpoint, so it runs anywhere. The
layer shapes below are the model's real weight dimensions (derived from
transformer/config.json; see notes/flux2-klein-4b-ane.md) with M set to the token
count for the requested resolution.

Run from the repo root:
  PYTHONPATH=. python3 notes/scripts/flux2_klein_layer_bench.py [WIDTH HEIGHT]
"""
from __future__ import annotations

import sys
import time

import numpy as np

import aneforge as af

CHUNK = 1024           # split-K chunk; see notes/large-k-cliff.md
REPS, WARMUP = 8, 3
TEXT_TOKENS = 512      # mflux pads to max_sequence_length=512


def tokens(width, height):
  """Image tokens after the VAE 8x downsample and the 2x2 latent pack (16x total)."""
  return (width // 16) * (height // 16)


def families(img, txt):
  """(label, M, K=in, N=out, count-per-step) for the real FLUX.2-klein-4B weights."""
  joint = img + txt
  return [
    ("single.qkv_mlp_proj  ", joint, 3072, 27648, 20),
    ("single.to_out        ", joint, 12288, 3072, 20),
    ("double.ff.linear_in  ", img, 3072, 18432, 5),
    ("double.ff.linear_out ", img, 9216, 3072, 5),
    ("double.ff_ctx.in     ", txt, 3072, 18432, 5),
    ("double.ff_ctx.out    ", txt, 9216, 3072, 5),
    ("double.attn proj x4  ", img, 3072, 3072, 20),
    ("double.attn_add x4   ", txt, 3072, 3072, 20),
  ]


def measure(M, K, N, chunk=CHUNK, seed=0):
  rng = np.random.default_rng(seed)
  W = (rng.standard_normal((N, K)).astype(np.float32) / np.float32(np.sqrt(K))).astype(np.float16)
  x = (rng.standard_normal((M, K)).astype(np.float32) / np.float32(np.sqrt(K))).astype(np.float16)
  if chunk is None or chunk >= K:
    net = af.compile(af.input((M, K)).linear(W))
  else:
    xs = af.input((M, K)); parts = []
    for i in range(0, K, chunk):
      w = min(chunk, K - i)
      parts.append(xs.slice_by_size([0, i], [M, w]).linear(W[:, i:i + w]))
    y = parts[0]
    for p in parts[1:]: y = y + p
    net = af.compile(y)
  for _ in range(WARMUP): net(x)
  best = float("inf")
  for _ in range(REPS):
    t = time.perf_counter(); net(x); best = min(best, time.perf_counter() - t)
  return best


def main(width, height):
  img = tokens(width, height)
  txt = TEXT_TOKENS
  print(f"{width}x{height} -> {img} image tokens + {txt} text = {img + txt} joint\n")
  print(f"{'layer':22s} {'M':>6s} {'K':>6s} {'N':>6s} {'#':>3s} | {'mono ms':>8s} {'TF/s':>5s} |"
        f" {'split ms':>8s} {'TF/s':>5s} | {'gain':>5s}")
  tot = 0.0
  for name, M, K, N, cnt in families(img, txt):
    flops = 2.0 * M * N * K
    tm = measure(M, K, N, None)
    ts = measure(M, K, N, CHUNK)
    tot += ts * cnt
    print(f"{name:22s} {M:6d} {K:6d} {N:6d} {cnt:3d} | {tm*1e3:8.2f} {flops/tm/1e12:5.2f} |"
          f" {ts*1e3:8.2f} {flops/ts/1e12:5.2f} | {tm/ts:4.2f}x")
  attn = 2 * (2 * (img + txt) ** 2 * 3072) * 25      # QK^T + AV, 25 blocks
  print(f"\nweight GEMMs, split-K chunk={CHUNK}: {tot*1e3:7.1f} ms/step"
        f"  ({tot*4:5.2f} s for 4 distilled steps)")
  print(f"attention core (est @6.2 TF/s, not measured): {attn/6.2e12*1e3:6.1f} ms/step")
  print("excludes: RoPE/norms/modulation/SwiGLU, inter-op overhead, text encoder, VAE")


if __name__ == "__main__":
  a = [int(v) for v in sys.argv[1:]] or [512, 768]
  main(a[0], a[1])
