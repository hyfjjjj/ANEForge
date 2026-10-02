#!/usr/bin/env python3
"""FLUX.2 single-stream block, ported to aneforge: ANE vs mflux/MLX, stage by stage.

The block (mflux `Flux2ParallelSelfAttention`):

  proj = to_qkv_mlp_proj(x)        [S, 27648] = qkv 9216 | mlp gate+up 18432
  q,k,v = split(qkv)               [S, 3072] each, 24 heads x 128
  q,k   = rms_norm(head_dim=128), then 4-axis interleaved RoPE
  attn  = sdpa(q, k, v, scale=1/sqrt(128))
  mlp   = silu(gate) * up          [S, 9216]
  out   = to_out(concat([attn, mlp]))   [S, 12288] -> [S, 3072]

Reference is mflux's own module: fp32 weights = ground truth, bfloat16 = what the
MLX backend actually runs (ModelConfig.precision). Weights come from the int8
checkpoint via mx.dequantize, so both engines see identical numbers.

Run (needs aneforge + mlx + mflux; e.g. fluxlab's venv):
  python3 notes/scripts/flux2_single_block_poc.py --seq 256
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

import aneforge as af

CKPT = "/Volumes/fxdisk/AIModels/flux2-klein-4b-int8-mlx/transformer/0.safetensors"
PREFIX = "single_transformer_blocks.0.attn"
DIM, HEADS, HEAD_DIM = 3072, 24, 128
MLP_INNER = 9216
QKV_DIM = HEADS * HEAD_DIM
AXES_DIM = (32, 32, 32, 32)
ROPE_THETA = 2000


def load_weights(path: str, prefix: str) -> dict[str, np.ndarray]:
  """Dequantize this block's weights with MLX itself (the authoritative layout)."""
  raw = mx.load(path)
  out: dict[str, np.ndarray] = {}
  for name in ("to_qkv_mlp_proj", "to_out", "norm_q", "norm_k"):
    w = raw[f"{prefix}.{name}.weight"]
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, raw[f"{prefix}.{name}.scales"], raw[f"{prefix}.{name}.biases"],
                        group_size=64, bits=8)
    out[name] = np.array(w.astype(mx.float32))
  return out


def rope_tables(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """4-axis RoPE cos/sin, [S, sum(axes)/2] -- FLUX.2 `Flux2PosEmbed` in numpy."""
  cos_parts, sin_parts = [], []
  for axis, dim in enumerate(AXES_DIM):
    scale = np.arange(0, dim, 2, dtype=np.float32) / dim
    omega = 1.0 / (ROPE_THETA ** scale)
    ang = ids[:, axis].astype(np.float32)[:, None] * omega[None, :]
    cos_parts.append(np.cos(ang))
    sin_parts.append(np.sin(ang))
  return np.concatenate(cos_parts, -1), np.concatenate(sin_parts, -1)


def make_ids(seq: int) -> np.ndarray:
  """Real id layout: image `[t=0, h, w, layer=0]`, text `[t=0, 0, 0, token]`."""
  n_txt = seq // 4
  n_img = seq - n_txt
  w_grid = 8
  h = np.arange(n_img) // w_grid
  w = np.arange(n_img) % w_grid
  img = np.stack([np.zeros(n_img), h, w, np.zeros(n_img)], 1)
  txt = np.stack([np.zeros(n_txt), np.zeros(n_txt), np.zeros(n_txt), np.arange(n_txt)], 1)
  return np.concatenate([txt, img], 0).astype(np.int32)


def interleaved_rope(x: af.Tensor, cos: np.ndarray, sin: np.ndarray) -> af.Tensor:
  """Rotate adjacent pairs of the last axis of a [1, heads, S, head_dim] tensor.

  mflux's `apply_rope_bshd` uses the interleaved pair layout (x[0::2]/x[1::2]),
  unlike aneforge's half-split `rope()` helper. Doing this BEFORE the head
  transpose (rather than on a flattened [S, heads*head_dim]) matters a lot: a
  transpose whose producer is the rope's concat costs ~62 ms at S=2048, while
  the same transpose fed by a matmul is free -- see notes/flux2-klein-4b-ane.md.
  """
  _, heads, S, d = x.shape
  pairs = d // 2
  c = cos.astype(np.float16)[None, None]
  s = sin.astype(np.float16)[None, None]
  x2 = x.reshape(1, heads, S, pairs, 2)
  even = x2.slice_by_size([0, 0, 0, 0, 0], [1, heads, S, pairs, 1]).reshape(1, heads, S, pairs)
  odd = x2.slice_by_size([0, 0, 0, 0, 1], [1, heads, S, pairs, 1]).reshape(1, heads, S, pairs)
  out_even = (even * c - odd * s).reshape(1, heads, S, pairs, 1)
  out_odd = (odd * c + even * s).reshape(1, heads, S, pairs, 1)
  return af.concat([out_even, out_odd], axis=4).reshape(1, heads, S, d)


def build_block(seq: int, weights: dict[str, np.ndarray], cos: np.ndarray,
                sin: np.ndarray, split_k: int = 0,
                attn_tile: int = 0, split_proj: bool = False) -> dict[str, af.Tensor]:
  """The whole block as one aneforge graph; returns each stage's output tensor.

  `split_k` > 0 chunks the to_out K dimension (K=12288 is past the engine's
  cliff at 4096; see notes/large-k-cliff.md).
  """
  S = seq
  x = af.input((S, DIM))

  w_full = weights["to_qkv_mlp_proj"].astype(np.float16)
  if split_proj:
    # five separate linears instead of one fused GEMM + five slices: same FLOPs,
    # but nothing has to stream the [S, 27648] result back in pieces
    q = x.linear(w_full[0:QKV_DIM])
    k = x.linear(w_full[QKV_DIM:2 * QKV_DIM])
    v = x.linear(w_full[2 * QKV_DIM:3 * QKV_DIM])
    gate = x.linear(w_full[3 * QKV_DIM:3 * QKV_DIM + MLP_INNER])
    up = x.linear(w_full[3 * QKV_DIM + MLP_INNER:])
    proj = mlp = None
  else:
    proj = x.linear(w_full)
    qkv = proj.slice_by_size([0, 0], [S, 3 * QKV_DIM])
    mlp = proj.slice_by_size([0, 3 * QKV_DIM], [S, 2 * MLP_INNER])
    q = qkv.slice_by_size([0, 0], [S, QKV_DIM])
    k = qkv.slice_by_size([0, QKV_DIM], [S, QKV_DIM])
    v = qkv.slice_by_size([0, 2 * QKV_DIM], [S, QKV_DIM])

  def to_bhsd(t: af.Tensor) -> af.Tensor:
    # transpose first: right after a matmul the compiler folds it into the
    # producer's output layout (free); after the rope's concat it does not
    return t.reshape(S, HEADS, HEAD_DIM).transpose([1, 0, 2]).reshape(1, HEADS, S, HEAD_DIM)

  def head_norm(t: af.Tensor, gamma: np.ndarray) -> af.Tensor:
    # rms_norm is 2D-only and normalizes the last dim: fold heads into the row axis
    return (t.reshape(HEADS * S, HEAD_DIM)
             .rms_norm(gamma.astype(np.float16), eps=1e-5)
             .reshape(1, HEADS, S, HEAD_DIM))

  scale = 1.0 / np.sqrt(HEAD_DIM)
  q4 = interleaved_rope(head_norm(to_bhsd(q), weights["norm_q"]), cos, sin)
  k4 = interleaved_rope(head_norm(to_bhsd(k), weights["norm_k"]), cos, sin)
  v4 = to_bhsd(v)
  if attn_tile and attn_tile < S:
    # query-tiled native sdpa: the layer's regime check is min(q,k seq) < 512,
    # so a full 2048x2048 falls back to the (slower) decomposed path
    tiles = [af.sdpa(q4.slice_by_size([0, 0, i, 0], [1, HEADS, min(attn_tile, S - i), HEAD_DIM]),
                     k4, v4, scale=scale) for i in range(0, S, attn_tile)]
    attn4 = af.concat(tiles, axis=2)
  else:
    attn4 = af.sdpa(q4, k4, v4, scale=scale)
  attn = attn4.reshape(HEADS, S, HEAD_DIM).transpose([1, 0, 2]).reshape(S, QKV_DIM)

  if not split_proj:
    gate = mlp.slice_by_size([0, 0], [S, MLP_INNER])
    up = mlp.slice_by_size([0, MLP_INNER], [S, MLP_INNER])
  mlp_out = gate.silu() * up

  joined = af.concat([attn, mlp_out], axis=1)
  w_out = weights["to_out"].astype(np.float16)
  if split_k:
    parts = [joined.slice_by_size([0, i], [S, min(split_k, joined.shape[1] - i)])
               .linear(w_out[:, i:i + split_k])
             for i in range(0, joined.shape[1], split_k)]
    out = parts[0]
    for part in parts[1:]:
      out = out + part
  else:
    out = joined.linear(w_out)
  return {"proj": proj if proj is not None else mlp_out,
          "q_rope": q, "attn": attn, "mlp": mlp_out, "out": out}


def reference(x32: np.ndarray, weights: dict[str, np.ndarray], cos: np.ndarray,
              sin: np.ndarray, dtype) -> np.ndarray:
  """mflux's own module, weights and activations cast to `dtype`."""
  from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import (
    Flux2ParallelSelfAttention,
  )
  attn = Flux2ParallelSelfAttention(dim=DIM, heads=HEADS, dim_head=HEAD_DIM, mlp_ratio=3.0)
  attn.load_weights([(f"{n}.weight", mx.array(weights[n]).astype(dtype))
                     for n in ("to_qkv_mlp_proj", "to_out", "norm_q", "norm_k")])
  # the module is 3D [B, S, D]; the graph works on 2D [S, D]
  out = attn(mx.array(x32[None]).astype(dtype),
             (mx.array(cos).astype(dtype), mx.array(sin).astype(dtype)))
  mx.eval(out)
  return np.array(out.astype(mx.float32))[0]


def relerr(got: np.ndarray, ref: np.ndarray) -> float:
  g = got.astype(np.float64).ravel()
  r = ref.astype(np.float64).ravel()
  return float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-30))


def timed(fn, reps: int) -> float:
  best = float("inf")
  for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    best = min(best, time.perf_counter() - t0)
  return best


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--seq", type=int, default=256)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--split-k", type=int, default=0,
                  help="chunk width for the K=12288 to_out linear (0 = monolithic)")
  ap.add_argument("--attn-tile", type=int, default=0,
                  help="query-tile width for native sdpa (0 = whole sequence)")
  ap.add_argument("--split-proj", action="store_true",
                  help="five separate linears instead of one fused projection + slices")
  args = ap.parse_args()

  rng = np.random.default_rng(0)
  x32 = rng.standard_normal((args.seq, DIM)).astype(np.float32)
  weights = load_weights(CKPT, PREFIX)
  ids = make_ids(args.seq)
  cos, sin = rope_tables(ids)

  from mflux.models.flux2.model.flux2_transformer.pos_embed import Flux2PosEmbed
  c_ref, s_ref = Flux2PosEmbed(theta=ROPE_THETA, axes_dim=AXES_DIM)(mx.array(ids))
  print(f"RoPE tables vs mflux: cos maxdiff {np.abs(np.array(c_ref) - cos).max():.2e}  "
        f"sin maxdiff {np.abs(np.array(s_ref) - sin).max():.2e}")

  truth = reference(x32, weights, cos, sin, mx.float32)
  ref_bf16 = reference(x32, weights, cos, sin, mx.bfloat16)
  print(f"mflux bf16 (what the MLX backend runs) vs fp32: relerr {relerr(ref_bf16, truth):.3e}")

  stages = build_block(args.seq, weights, cos, sin, args.split_k, args.attn_tile, args.split_proj)
  xf = x32.astype(np.float16)
  print(f"\n{'stage':10s} {'ms':>9s} {'delta':>9s}")
  prev = 0.0
  for name in ("proj", "q_rope", "attn", "mlp", "out"):
    net = af.compile(stages[name])
    net(xf)
    lat = timed(lambda: net(xf), args.reps) * 1e3
    print(f"{name:10s} {lat:9.2f} {lat - prev:9.2f}")
    prev = lat

  out = stages["out"]
  t0 = time.perf_counter()
  net = af.compile(out)
  compile_s = time.perf_counter() - t0

  xf = x32.astype(np.float16)
  got = net(xf)
  lat = timed(lambda: net(xf), args.reps)
  print(f"compile {compile_s:.1f}s; ANE single dispatch {lat * 1e3:.2f} ms")
  print(f"ANE fp16 vs fp32: relerr {relerr(got, truth):.3e}")


if __name__ == "__main__":
  main()
