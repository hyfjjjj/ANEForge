#!/usr/bin/env python3
"""FLUX.2-klein-4B transformer, end to end on the ANE vs mflux/MLX.

Assembles the real model on top of the two block POCs: x/context embedders, the
5 double-stream blocks, the stream concat, the 20 single-stream blocks, norm_out
and proj_out -- including the layer norms, modulation and residual gates the
block-level POCs left out.

Each block is its own compiled program (the 25 weight sets are ~7.75 GB in fp16,
past what one e5rt program should hold) and they are dispatched in sequence,
which is also the granularity a hybrid ANE/GPU split would work at. The
time/guidance embedding is a handful of scalars per step and stays host-side.

Run (needs aneforge + mlx + mflux):
  python3 notes/scripts/flux2_transformer_poc.py --txt 512 --img 1536
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import aneforge as af

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from flux2_double_block_poc import build_block as build_double  # noqa: E402
from flux2_double_block_poc import modulation  # noqa: E402
from flux2_single_block_poc import CKPT, DIM, build_block as build_single  # noqa: E402
from flux2_single_block_poc import make_ids, relerr, rope_tables  # noqa: E402

SHARDS = [str(Path(CKPT).parent / f"{i}.safetensors") for i in (0, 1)]


def load_shards() -> dict:
  """Merge both checkpoint shards (blocks are split across them)."""
  raw: dict = {}
  for shard in SHARDS:
    raw.update(mx.load(shard))
  return raw


def block_weights(raw: dict, prefix: str) -> dict[str, np.ndarray]:
  """Every `.weight` under `prefix`, dequantized, keyed without the prefix."""
  out: dict[str, np.ndarray] = {}
  for key, w in raw.items():
    if not key.startswith(prefix + ".") or not key.endswith(".weight"):
      continue
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, raw[key.replace(".weight", ".scales")],
                        raw[key.replace(".weight", ".biases")], group_size=64, bits=8)
    out[key[len(prefix) + 1:-len(".weight")]] = np.array(w.astype(mx.float32))
  return out

N_DOUBLE, N_SINGLE = 5, 20
TXT_DIM = 7680
GUIDANCE_CH = 256
ONES = np.ones(DIM, np.float32)
ZEROS = np.zeros(DIM, np.float32)


def timestep_embedding(t: float, dim: int = GUIDANCE_CH) -> np.ndarray:
  """`Flux2TimestepGuidanceEmbeddings._timestep_embedding` (flip_sin_to_cos)."""
  half = dim // 2
  freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
  args = np.float32(t) * freqs
  emb = np.concatenate([np.sin(args), np.cos(args)])
  return np.concatenate([emb[half:], emb[:half]])[None]        # [1, 256]


def silu(x: np.ndarray) -> np.ndarray:
  return x / (1.0 + np.exp(-x))


def modulate(x: af.Tensor, shift: np.ndarray, scale: np.ndarray) -> af.Tensor:
  """`(1 + scale) * LayerNorm(x) + shift` (the blocks' entry modulation)."""
  n = x.layer_norm(ONES, ZEROS, eps=1e-6)
  return n * (1.0 + scale.astype(np.float16)) + shift.astype(np.float16)


class AneTransformer:
  """The 25 blocks plus embeddings/output head, each a compiled program."""

  def __init__(self, txt: int, img: int, k_chunk: int = 1024, verbose: bool = True,
               fuse: int = 1):
    self.txt, self.img, self.k_chunk, self.verbose = txt, img, k_chunk, verbose
    self.fuse = max(1, fuse)
    self.S = txt + img
    self.cos, self.sin = rope_tables(make_ids(self.S))
    self.raw = load_shards()
    self.nets: dict[str, object] = {}
    self.order: list[tuple[str, int]] = []

  def _log(self, msg: str) -> None:
    if self.verbose:
      print(msg, flush=True)

  def _w(self, name: str) -> np.ndarray:
    w = self.raw[name]
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, self.raw[name.replace(".weight", ".scales")],
                        self.raw[name.replace(".weight", ".biases")],
                        group_size=64, bits=8)
    return np.array(w.astype(mx.float32))

  def build(self, temb32: np.ndarray) -> None:
    t_all = time.perf_counter()
    mg = modulation(temb32, self._w("double_stream_modulation_img.linear.weight"))
    mt = modulation(temb32, self._w("double_stream_modulation_txt.linear.weight"))
    ms = modulation(temb32, self._w("single_stream_modulation.linear.weight"))[0]

    # --- input embedders: two inputs (different widths), one joint output ---
    x_txt = af.input((self.txt, TXT_DIM))
    x_img = af.input((self.img, 128))
    emb = af.concat([x_txt.linear(self._w("context_embedder.weight").astype(np.float16)),
                     x_img.linear(self._w("x_embedder.weight").astype(np.float16))], axis=0)
    self.nets["embed"] = af.compile(emb)

    # --- 5 double-stream blocks, joint [S, DIM] in and out ---
    for i in range(N_DOUBLE):
      t0 = time.perf_counter()
      xj = af.input((self.S, DIM))
      st = build_double(block_weights(self.raw, f"transformer_blocks.{i}"), self.txt, self.img,
                        self.cos, self.sin, mg, mt, self.k_chunk,
                        x_img=xj.slice_by_size([self.txt, 0], [self.img, DIM]),
                        x_txt=xj.slice_by_size([0, 0], [self.txt, DIM]))
      self.nets[f"d{i}"] = af.compile(af.concat([st["txt"], st["img"]], axis=0))
      self._log(f"  double {i} compiled ({time.perf_counter() - t0:4.1f}s)")

    # --- 20 single-stream blocks: norm + modulation + attention + gate ---
    # `fuse` blocks share one program (fewer cross-program activation round-trips)
    for i in range(0, N_SINGLE, self.fuse):
      t0 = time.perf_counter()
      x = af.input((self.S, DIM))
      n = self.fuse if self.fuse > 1 else 1
      for j in range(i, min(i + n, N_SINGLE)):
        attn = build_single(self.S, block_weights(self.raw, f"single_transformer_blocks.{j}.attn"),
                            self.cos, self.sin, self.k_chunk,
                            x=modulate(x, ms[0], ms[1]))["out"]
        x = x + attn * ms[2].astype(np.float16)
      self.nets[f"s{i}"] = af.compile(x)
      self._log(f"  single {i}-{min(i + n, N_SINGLE) - 1} compiled ({time.perf_counter() - t0:4.1f}s)")

    # --- norm_out (Ada) + proj_out on the image stream only ---
    # AdaLayerNormContinuous: linear(silu(temb)), then scale = [:DIM], shift = [DIM:]
    temb = (silu(temb32) @ self._w("norm_out.linear.weight").T).astype(np.float32)
    self.scale, self.shift = temb[:, :DIM], temb[:, DIM:]
    x = af.input((self.img, DIM))
    n = x.layer_norm(ONES, ZEROS, eps=1e-6)
    n = n * (1.0 + self.scale.astype(np.float16)) + self.shift.astype(np.float16)
    self.nets["head"] = af.compile(n.linear(self._w("proj_out.weight").astype(np.float16)))
    # the raw shards are only needed while compiling: 4 GB that the inference
    # path should not keep resident (compiled programs hold their own weights)
    self.raw = {}
    gc.collect()
    self._log(f"build done in {time.perf_counter() - t_all:.1f}s")

  def run(self, ctx32: np.ndarray, lat32: np.ndarray, reps: int = 1) -> tuple[np.ndarray, dict]:
    """Full forward; returns the output and per-program times (ms)."""
    ctx = ctx32.astype(np.float16)
    lat = lat32.astype(np.float16)
    times: dict[str, float] = {}
    out = None
    for rep in range(reps):
      last = rep == reps - 1
      t0 = time.perf_counter()
      x = self.nets["embed"](ctx, lat)
      if last:
        times["embed"] = (time.perf_counter() - t0) * 1e3
      for i in range(N_DOUBLE):
        t0 = time.perf_counter()
        x = self.nets[f"d{i}"](x)
        if last:
          times[f"d{i}"] = (time.perf_counter() - t0) * 1e3
      for i in range(0, N_SINGLE, self.fuse):
        t0 = time.perf_counter()
        x = self.nets[f"s{i}"](x)
        if last:
          times[f"s{i}"] = (time.perf_counter() - t0) * 1e3
      t0 = time.perf_counter()
      out = self.nets["head"](x[self.txt:])
      if last:
        times["head"] = (time.perf_counter() - t0) * 1e3
    return out, times


def reference_mlx(ctx32, lat32, cos, sin, temb32, dtype):
  """mflux's full Flux2Transformer with the checkpoint's own weights."""
  from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
  tf = Flux2Transformer(patch_size=1, in_channels=128, out_channels=None,
                        num_layers=N_DOUBLE, num_single_layers=N_SINGLE,
                        attention_head_dim=128, num_attention_heads=24,
                        joint_attention_dim=TXT_DIM, timestep_guidance_channels=GUIDANCE_CH,
                        mlp_ratio=3.0, axes_dims_rope=(32, 32, 32, 32),
                        rope_theta=2000, guidance_embeds=False)
  raw = load_shards()
  weights = []
  for name, w in raw.items():
    if not name.endswith(".weight"):
      continue
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, raw[name.replace(".weight", ".scales")],
                        raw[name.replace(".weight", ".biases")], group_size=64, bits=8)
    weights.append((name, w.astype(dtype)))
  tf.load_weights(weights)
  ids = make_ids(ctx32.shape[0] + lat32.shape[0])
  out = tf(hidden_states=mx.array(lat32[None]).astype(dtype),
           encoder_hidden_states=mx.array(ctx32[None]).astype(dtype),
           timestep=mx.array([1000.0]).astype(dtype),
           img_ids=mx.array(ids[ctx32.shape[0]:])[None],
           txt_ids=mx.array(ids[:ctx32.shape[0]])[None],
           guidance=None)
  mx.eval(out)
  return np.array(out.astype(mx.float32))[0]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--txt", type=int, default=512)
  ap.add_argument("--img", type=int, default=1536)
  ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--k-chunk", type=int, default=1024)
  ap.add_argument("--reference", action="store_true", help="also run mflux's transformer (bf16)")
  ap.add_argument("--fuse", type=int, default=1, help="single-stream blocks per program")
  args = ap.parse_args()

  rng = np.random.default_rng(0)
  ctx32 = rng.standard_normal((args.txt, TXT_DIM)).astype(np.float32) * 0.5
  lat32 = rng.standard_normal((args.img, 128)).astype(np.float32)
  # mflux scales a timestep <= 1 by 1000 before embedding (transformer.py)
  t_emb = timestep_embedding(1000.0)
  print(f"txt={args.txt} img={args.img} joint={args.txt + args.img}")

  tr = AneTransformer(args.txt, args.img, args.k_chunk, fuse=args.fuse)
  # time/guidance embedding is tiny and per-step: keep it host-side
  w1, w2 = tr._w("time_guidance_embed.linear_1.weight"), tr._w("time_guidance_embed.linear_2.weight")
  temb32 = (silu(t_emb @ w1.T) @ w2.T).astype(np.float32)
  tr.build(temb32)

  out, times = tr.run(ctx32, lat32, reps=args.reps)
  total = sum(times.values())
  print(f"\n{'program':10s} {'ms':>8s}")
  for key in ("embed", *[f"d{i}" for i in range(N_DOUBLE)],
              *[f"s{i}" for i in range(0, N_SINGLE, tr.fuse)], "head"):
    print(f"{key:10s} {times[key]:8.2f}")
  print(f"{'TOTAL':10s} {total:8.2f}   ({total / 1e3:.2f} s/step, {total * 4 / 1e3:.2f} s for 4 steps)")

  if args.reference:
    t0 = time.perf_counter()
    ref = reference_mlx(ctx32, lat32, tr.cos, tr.sin, temb32, mx.bfloat16)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ref = reference_mlx(ctx32, lat32, tr.cos, tr.sin, temb32, mx.bfloat16)
    fwd_s = time.perf_counter() - t0
    print(f"\nmflux bf16: load {build_s:.1f}s, forward {fwd_s:.2f}s "
          f"({fwd_s * 1e3:.0f} ms/step, {fwd_s * 4:.1f} s for 4 steps)")
    print(f"ANE fp16 vs mflux bf16 (same weights): relerr {relerr(out, ref):.3e}")


if __name__ == "__main__":
  main()
