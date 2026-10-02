#!/usr/bin/env python3
"""FLUX.2 double-stream block on the ANE vs mflux/MLX.

Two streams (text 512 + image 1536 at 512x768): each projects its own q/k/v,
the sequences are concatenated for one joint attention, then split back for
separate projections and separate feed-forwards.

  n_img = (1+scale_msa) * LN(img) + shift_msa        (same for txt with c_*)
  q/k/v = concat([txt, img]) per stream projections, joint RoPE, joint sdpa
  img += gate_msa * to_out(attn[img]); txt += c_gate_msa * to_add_out(attn[txt])
  img += gate_mlp * ff((1+scale_mlp) * LN(img) + shift_mlp)     (and ff_context)

Modulation params come from the real `double_stream_modulation_{img,txt}` weights
applied to a synthetic temb, so their magnitudes are realistic.

Run (needs aneforge + mlx + mflux):
  python3 notes/scripts/flux2_double_block_poc.py --txt 512 --img 1536
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import aneforge as af

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flux2_single_block_poc import (  # noqa: E402
  CKPT, DIM, HEADS, HEAD_DIM, interleaved_rope, make_ids, relerr, rope_tables,
)

PREFIX = "transformer_blocks.0"
QKV = HEADS * HEAD_DIM
FF_IN = 18432
FF_MID = 9216


def load_weights(path: str, prefix: str) -> dict[str, np.ndarray]:
  """All double-block weights, dequantized with MLX itself."""
  raw = mx.load(path)
  names = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
           "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
           "attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k",
           "ff.linear_in", "ff.linear_out",
           "ff_context.linear_in", "ff_context.linear_out"]
  out: dict[str, np.ndarray] = {}
  for name in names:
    key = f"{prefix}.{name}.weight"
    w = raw[key]
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, raw[f"{prefix}.{name}.scales"], raw[f"{prefix}.{name}.biases"],
                        group_size=64, bits=8)
    out[name] = np.array(w.astype(mx.float32))
  for name in ("double_stream_modulation_img", "double_stream_modulation_txt"):
    key = f"{name}.linear.weight"
    w = raw[key]
    if w.dtype == mx.uint32:
      w = mx.dequantize(w, raw[f"{name}.linear.scales"], raw[f"{name}.linear.biases"],
                        group_size=64, bits=8)
    out[name] = np.array(w.astype(mx.float32))
  return out


def modulation(temb: np.ndarray, w: np.ndarray) -> tuple:
  """`Flux2Modulation`: silu -> linear -> `mod_param_sets` x (shift, scale, gate).

  The set count comes from the weight shape: 1 for the single-stream modulation
  (9216 = 3 x 3072), 2 for the double-stream ones (18432 = 6 x 3072).
  """
  mod = (temb / (1.0 + np.exp(-temb))) @ w.T
  sets = w.shape[0] // (DIM * 3)
  chunks = np.split(mod, 3 * sets, axis=-1)
  return tuple(tuple(chunks[3 * i:3 * i + 3]) for i in range(sets))


def ff(x: af.Tensor, w_in: np.ndarray, w_out: np.ndarray, S: int) -> af.Tensor:
  h = x.linear(w_in.astype(np.float16))
  gate = h.slice_by_size([0, 0], [S, FF_MID])
  up = h.slice_by_size([0, FF_MID], [S, FF_MID])
  return (gate.silu() * up).linear(w_out.astype(np.float16))


def linear_k(x: af.Tensor, w: np.ndarray, chunk: int) -> af.Tensor:
  """`x @ w.T`, optionally with K chunked (the engine's K > 4096 cliff)."""
  w16 = w.astype(np.float16)
  if not chunk or chunk >= w.shape[1]:
    return x.linear(w16)
  parts = [x.slice_by_size([0, i], [x.shape[0], min(chunk, w.shape[1] - i)])
            .linear(w16[:, i:i + chunk])
           for i in range(0, w.shape[1], chunk)]
  out = parts[0]
  for part in parts[1:]:
    out = out + part
  return out


def build_block(weights: dict[str, np.ndarray], txt: int, img: int,
                cos: np.ndarray, sin: np.ndarray,
                mod_img, mod_txt, k_chunk: int = 0,
                x_img: "af.Tensor | None" = None,
                x_txt: "af.Tensor | None" = None) -> dict[str, af.Tensor]:
  St, Si = txt, img
  x_img = af.input((Si, DIM)) if x_img is None else x_img
  x_txt = af.input((St, DIM)) if x_txt is None else x_txt
  (sh_msa, sc_msa, g_msa), (sh_mlp, sc_mlp, g_mlp) = mod_img
  (c_sh_msa, c_sc_msa, c_g_msa), (c_sh_mlp, c_sc_mlp, c_g_mlp) = mod_txt

  def modulate(x: af.Tensor, shift, scale, S: int) -> af.Tensor:
    n = x.layer_norm(np.ones(DIM, np.float32), np.zeros(DIM, np.float32), eps=1e-6)
    return n * (1.0 + scale.astype(np.float16)) + shift.astype(np.float16)

  n_img = modulate(x_img, sh_msa, sc_msa, Si)
  n_txt = modulate(x_txt, c_sh_msa, c_sc_msa, St)

  def heads(t: af.Tensor, w: np.ndarray, g: np.ndarray, S: int) -> af.Tensor:
    # transpose right after the matmul (free) then norm on [1,H,S,D] --
    # see notes/flux2-block-poc.md for why this order matters.
    # RoPE is applied after the two streams are concatenated (joint sequence).
    return (t.linear(w.astype(np.float16)).reshape(S, HEADS, HEAD_DIM)
             .transpose([1, 0, 2]).reshape(1, HEADS, S, HEAD_DIM)
             .reshape(HEADS * S, HEAD_DIM).rms_norm(g.astype(np.float16), eps=1e-5)
             .reshape(1, HEADS, S, HEAD_DIM))

  def heads_v(t: af.Tensor, w: np.ndarray, S: int) -> af.Tensor:
    return (t.linear(w.astype(np.float16)).reshape(S, HEADS, HEAD_DIM)
             .transpose([1, 0, 2]).reshape(1, HEADS, S, HEAD_DIM))

  q = interleaved_rope(af.concat(
    [heads(n_txt, weights["attn.add_q_proj"], weights["attn.norm_added_q"], St),
     heads(n_img, weights["attn.to_q"], weights["attn.norm_q"], Si)], axis=2), cos, sin)
  k = interleaved_rope(af.concat(
    [heads(n_txt, weights["attn.add_k_proj"], weights["attn.norm_added_k"], St),
     heads(n_img, weights["attn.to_k"], weights["attn.norm_k"], Si)], axis=2), cos, sin)
  v = af.concat([heads_v(n_txt, weights["attn.add_v_proj"], St),
                 heads_v(n_img, weights["attn.to_v"], Si)], axis=2)

  attn = af.sdpa(q, k, v, scale=1.0 / np.sqrt(HEAD_DIM))       # [1,H,St+Si,D]

  def unheads(t4: af.Tensor, S: int, off: int) -> af.Tensor:
    return (t4.slice_by_size([0, 0, off, 0], [1, HEADS, S, HEAD_DIM])
              .reshape(HEADS, S, HEAD_DIM).transpose([1, 0, 2]).reshape(S, QKV))

  txt_attn = unheads(attn, St, 0)
  img_attn = unheads(attn, Si, St)
  x_img = x_img + linear_k(img_attn, weights["attn.to_out"], k_chunk) * g_msa.astype(np.float16)
  x_txt = x_txt + linear_k(txt_attn, weights["attn.to_add_out"], k_chunk) * c_g_msa.astype(np.float16)

  x_img = x_img + ff(
    modulate(x_img, sh_mlp, sc_mlp, Si), weights["ff.linear_in"], weights["ff.linear_out"], Si
  ) * g_mlp.astype(np.float16)
  x_txt = x_txt + ff(
    modulate(x_txt, c_sh_mlp, c_sc_mlp, St), weights["ff_context.linear_in"],
    weights["ff_context.linear_out"], St
  ) * c_g_mlp.astype(np.float16)
  return {"img": x_img, "txt": x_txt}


def reference(x_img32, x_txt32, weights, cos, sin, mod_img, mod_txt, dtype):
  """mflux's own Flux2TransformerBlock, weights/activations cast to `dtype`."""
  from mflux.models.flux2.model.flux2_transformer.transformer_block import Flux2TransformerBlock
  blk = Flux2TransformerBlock(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, mlp_ratio=3.0)
  pairs = [(f"attn.{n}.weight", f"attn.{n}") for n in
           ("to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj",
            "to_add_out", "norm_q", "norm_k", "norm_added_q", "norm_added_k")]
  pairs += [(f"{n}.weight", n) for n in
            ("ff.linear_in", "ff.linear_out", "ff_context.linear_in", "ff_context.linear_out")]
  blk.load_weights([(dst, mx.array(weights[src]).astype(dtype)) for dst, src in pairs])
  cast = lambda a: mx.array(a).astype(dtype)  # noqa: E731
  # the module is 3D [B, S, D]; the graph works on 2D [S, D]
  out = blk(hidden_states=cast(x_img32[None]), encoder_hidden_states=cast(x_txt32[None]),
            temb_mod_params_img=tuple(tuple(cast(p) for p in group) for group in mod_img),
            temb_mod_params_txt=tuple(tuple(cast(p) for p in group) for group in mod_txt),
            image_rotary_emb=(cast(cos), cast(sin)))
  mx.eval(*out)
  return [np.array(o.astype(mx.float32))[0] for o in out]    # (txt, img), drop batch


def timed(fn, reps: int) -> float:
  best = float("inf")
  for _ in range(reps):
    t0 = time.perf_counter()
    fn()
    best = min(best, time.perf_counter() - t0)
  return best


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--txt", type=int, default=512)
  ap.add_argument("--img", type=int, default=1536)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--k-chunk", type=int, default=0)
  args = ap.parse_args()

  rng = np.random.default_rng(0)
  x_img32 = rng.standard_normal((args.img, DIM)).astype(np.float32)
  x_txt32 = rng.standard_normal((args.txt, DIM)).astype(np.float32)
  temb = rng.standard_normal((1, DIM)).astype(np.float32)
  weights = load_weights(CKPT, PREFIX)
  ids = make_ids(args.txt + args.img)
  cos, sin = rope_tables(ids)
  mod_img = modulation(temb, weights["double_stream_modulation_img"])
  mod_txt = modulation(temb, weights["double_stream_modulation_txt"])

  truth_txt, truth_img = reference(x_img32, x_txt32, weights, cos, sin, mod_img, mod_txt, mx.float32)
  ref_txt, ref_img = reference(x_img32, x_txt32, weights, cos, sin, mod_img, mod_txt, mx.bfloat16)
  print(f"mflux bf16 vs fp32: txt {relerr(ref_txt, truth_txt):.3e}  img {relerr(ref_img, truth_img):.3e}")

  stages = build_block(weights, args.txt, args.img, cos, sin, mod_img, mod_txt, args.k_chunk)
  xf_img, xf_txt = x_img32.astype(np.float16), x_txt32.astype(np.float16)
  for name in ("img", "txt"):
    net = af.compile(stages[name])
    inp = (xf_img, xf_txt) if name == "img" else (xf_img, xf_txt)
    got = net(*inp)
    lat = timed(lambda: net(*inp), args.reps)
    truth = truth_img if name == "img" else truth_txt
    print(f"{name}: {lat * 1e3:8.2f} ms   ANE fp16 vs fp32 relerr {relerr(got, truth):.3e}")


if __name__ == "__main__":
  main()
