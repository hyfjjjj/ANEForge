# FLUX.2-klein-4B on the ANE: inventory and port feasibility

Question this answers: could `black-forest-labs/FLUX.2-klein-4B` run on aneforge?
Short answer: **feasible in principle, not usable today** - aneforge has no DiT
code at all, the transformer forward would have to be written. What follows is
the evidence for both halves of that sentence.

Text encoder and VAE are explicitly out of scope here; this is the transformer
only. Sources are the on-disk checkpoints and configs listed at the bottom.

## 1. Architecture, as read from the configs

`Flux2Transformer2DModel` (`transformer/config.json`):

| Field | Value | Consequence |
| --- | --- | --- |
| `attention_head_dim` / `num_attention_heads` | 128 / 24 | inner dim 3072, qkv 9216 |
| `num_layers` / `num_single_layers` | 5 / 20 | 5 double-stream + 20 single-stream blocks |
| `mlp_ratio` | 3.0 | MLP inner 9216, gated (SwiGLU) |
| `joint_attention_dim` | 7680 | = 3 x 2560, three Qwen3-4B hidden layers concatenated |
| `axes_dims_rope` / `rope_theta` | (32,32,32,32) / 2000 | **4-axis** RoPE over 2D position ids |
| `in_channels` / `patch_size` | 128 / 1 | packing happens before the transformer |
| `timestep_guidance_channels` | 256 | time/guidance MLP |

The shapes are self-consistent once you see the gated MLP: the single-stream
block's `to_qkv_mlp_proj` emits 27648 = 9216 (qkv) + 2 x 9216 (MLP gate and up,
pre-activation), and `to_out` consumes 12288 = 3072 (attention out) + 9216
(post-SwiGLU MLP). The double-stream blocks mirror this at MLP scale 18432 -> 9216.

## 2. Token math

`vae/config.json`: `latent_channels` 32, `patch_size` [2,2], `block_out_channels`
[128,256,512,512] with 3 downsamplers -> 8x spatial. The latent is then packed
2x2, so **16x total**, and the transformer sees 128 channels at latent/2.

For a 512 x 768 image:

```
512 x 768 pixels
  -> VAE 8x            -> 64 x 96 latent, 32 channels
  -> 2x2 latent pack   -> 32 x 48 = 1536 image tokens, 128 channels
  -> text (mflux pads to max_sequence_length=512)
  -> joint sequence = 2048 tokens for the single-stream blocks
```

Double-stream blocks keep the streams separate: 1536 image, 512 text.

FLUX.2-klein-4B is distilled (`model_index.json`: `is_distilled: true`) and
mflux's README states 4 steps, which is the multiplier used below.

## 3. Weight inventory

The `int8-mlx` checkpoint is MLX quantization (`quantization_level: 8`,
mflux 0.18.1): weights are U32 with **4 int8 per u32** (logical columns =
stored columns x 4), plus BF16 scales and biases at `[out, in/64]`
(group_size 64). 4.118 GB / 387 tensors = 3.876 G int8 params + 0.121 G bf16,
i.e. ~4.0 B params, matching the "4B" name.

Logical shapes, `[out, in]`:

| Family | Count | Shape | Each | Total |
| --- | --- | --- | --- | --- |
| `single_*.attn.to_qkv_mlp_proj` | 20 | 27648 x 3072 | 84.9 MB | 1.70 GB |
| `single_*.attn.to_out` | 20 | 3072 x 12288 | 37.7 MB | 0.76 GB |
| `transformer_blocks.*.ff{,_context}.linear_in` | 10 | 18432 x 3072 | 56.6 MB | 0.57 GB |
| `transformer_blocks.*.ff{,_context}.linear_out` | 10 | 3072 x 9216 | 28.3 MB | 0.28 GB |
| double-stream attn projections (8 per block) | 40 | 3072 x 3072 | 9.4 MB | 0.38 GB |
| `double_stream_modulation_{img,txt}` | 2 | 18432 x 3072 | 56.6 MB | 0.11 GB |
| `single_stream_modulation` | 1 | 9216 x 3072 | 28.3 MB | 0.03 GB |
| `context_embedder` / `norm_out` | 2 | 3072x7680 / 6144x3072 | - | 0.04 GB |
| embedders, time-guidance, `proj_out` | - | all <= 9.4 MB | - | ~0.01 GB |

Largest single matrix: **27648 x 3072** (the fused qkv+MLP projection). Largest
single dimension: 27648. That is past the 16384 matmul dimension the repo
mentions for the older families, but it **dispatches fine on this M4 Pro**:
`[128, 3072] @ [3072, 27648]` compiled in 0.25 s and matched an fp64 reference
at relerr 1.9e-02.

## 4. Measured per-layer ANE cost at 512 x 768

`scripts/flux2_klein_layer_bench.py`. Each row is one fused program on the
engine; "split" is the K-chunking from [`large-k-cliff.md`](large-k-cliff.md)
with chunk 1024.

| Layer | M | K | N | #/step | mono ms | mono TF/s | split ms | split TF/s | gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single.qkv_mlp_proj` | 2048 | 3072 | 27648 | 20 | 49.39 | 7.04 | 38.29 | 9.09 | 1.29x |
| `single.to_out` | 2048 | 12288 | 3072 | 20 | 80.48 | 1.92 | 15.97 | 9.68 | 5.04x |
| `double.ff.linear_in` | 1536 | 3072 | 18432 | 5 | 25.02 | 6.95 | 19.15 | 9.08 | 1.31x |
| `double.ff.linear_out` | 1536 | 9216 | 3072 | 5 | 44.83 | 1.94 | 8.70 | 10.00 | 5.15x |
| `double.ff_ctx.linear_in` | 512 | 3072 | 18432 | 5 | 9.52 | 6.09 | 4.78 | 12.12 | 1.99x |
| `double.ff_ctx.linear_out` | 512 | 9216 | 3072 | 5 | 15.20 | 1.91 | 2.84 | 10.22 | 5.36x |
| double attn projections x4 | 1536 | 3072 | 3072 | 20 | 5.10 | 5.69 | 3.89 | 7.46 | 1.31x |
| double attn add-projections x4 | 512 | 3072 | 3072 | 20 | 1.48 | 6.55 | 0.96 | 10.03 | 1.53x |

Totals, weights only:

| | weight GEMMs | + attention core (est) | 1 step | 4 steps |
| --- | --- | --- | --- | --- |
| monolithic | 3.20 s (3.92 TF/s) | 0.21 s | 3.41 s | **13.6 s** |
| split-K | 1.36-1.52 s (9.2-8.5 TF/s) | 0.21 s | 1.57-1.73 s | **6.3-6.9 s** |

Run-to-run variance: the table above and a second run of the same script on the
same machine agreed on the monolithic total to 0.1% (3.20 s), but the split-K
total moved from 1.36 s to 1.52 s. The mover was mostly `qkv_mlp_proj`
(38.3 -> 45.9 ms) rather than the large-K layers, which were stable. Power and
thermal state on this machine are worth +/-10% on the chunked path; quote the
split number as a range, not a point.

The two `to_out` families are the whole story of the monolithic number: at
K = 12288/9216 they sit right past the cliff and eat 1.8 s of the 3.2 s.

Scaling to other resolutions: weight-GEMM time is linear in token count (M), so
1024 x 1024 (4096 image + 512 text = 4608 joint) is roughly
1.36 s x 4608/2048 = **~3.1 s/step**; attention grows with L^2 on top of that.

## 5. What the number does not include

- **Attention core** (QK^T, softmax, AV) - estimated at 1.29 TFLOP/step ~ 0.21 s
  at the measured square-GEMM rate, not measured. `af.sdpa` exists and
  `examples/sdpa.py` covers it; the 2048-token case (a `[24, 2048, 2048]` fp16
  score tensor, ~201 MB before tiling) is untested.
- **RoPE, norms, modulation, SwiGLU, all elementwise** - small FLOPs, but each
  is a program boundary with an activation round-trip.
- **Inter-op overhead.** The 1.36 s is pieces measured in isolation. A real
  end-to-end step adds ~300 dispatches and boundary DMAs, which is plausibly
  another 0.5-1 s/step.
- **Text encoder, VAE** - out of scope by request.

## 6. Could it be ported? Op coverage

Checked the needed ops against `docs/op-catalog.md`, `aneforge/graph.py` and
`aneforge/llm.py`:

| Needed | aneforge | Note |
| --- | --- | --- |
| `linear` | yes | all shapes above measured |
| `rms_norm` (over head_dim 128) | yes | `Tensor.rms_norm(gamma, eps)` |
| SwiGLU (silu + gate + mul) | yes | `silu`, `mul` |
| joint attention (concat text+image KV) | yes | `sdpa`, `concat`, `split` |
| softmax, layer_norm, tanh, cos, sin | yes | in catalog |
| modulation (6x3072 scale/shift/gate) | yes | `linear` + elementwise |
| `slice_by_size` back to the image stream | yes | exact on this machine |
| 4-axis RoPE | **build it yourself** | see below |

The one real gap is RoPE. mflux's `AttentionUtils.apply_rope_bshd` uses the
**interleaved pair layout** (`x[0::2]` / `x[1::2]` as real/imag, cos/sin of shape
`[seq, dh/2]`), and FLUX.2 needs four axes of 32 dims each with per-axis position
ids. aneforge's `rope()` helper is Llama "neox" half-split over `[seq, dh]`, so it
does not drop in. The primitives are all present (`reshape`, `slice_by_size`,
`mul`, `add`, `concat`) and cos/sin tables are host-side constants
(`rope_tables` has an `interleaved=True` variant worth checking first), so this
is constructible - but it is hand-written code that needs numeric validation,
not a parameter change.

Also: no FLUX/DiT code exists in the repo (`grep -ri flux` is empty). aneforge
ships loaders for Llama/Qwen/GPT-2 (`llm.py`), MoE, and sentence-transformers;
DiT is a new frontend. `examples/sd_unet.py` and `examples/sd15.py` show the
pattern for writing a diffusion model forward from real diffusers weights, which
is the template to follow.

## 7. Risks, in order

1. **fp16 numerics.** Measured single-GEMM Frobenius error is 1.9e-02 at K=3072
   and 7-10% at K=9216/12288, and the `to_out` / `linear_out` layers are exactly
   those large-K ones. What 25 blocks x 4 steps does to image quality is
   unverified. The project's own `examples/sd15.py` already flags
   "end-to-end fp16-degraded by CFG cancellation", so this class of problem is
   real here. mflux computes the norms in fp32; the ANE path is fp16 throughout.
2. **Weight residency.** 25 blocks is ~7.7 GB of fp16 weights, so no single
   e5rt program. Needs a streamed/layer-resident compile - the repo has patterns
   for this (`examples/train_charlm_deep.py` for compile-size-bounded streaming,
   `examples/gpt_multilayer_resident.py` for `share_buffer` residency), but
   neither has been tried at this depth.
3. **Weight format.** The MLX int8 layout (U32-packed, group-64 scales/biases)
   is not what aneforge consumes (fp16 numpy, or its own per-channel int8
   streaming). A bf16 copy of the same model exists on disk, which sidesteps the
   dequantization work entirely.
4. **Attention at 2048 tokens.** Untested on this engine at that scale.

## 8. Next step if this is pursued

A single-block proof of concept: build one single-stream block (fused
qkv+MLP projection, 4-axis RoPE, joint `sdpa`, SwiGLU, `to_out`), compile it to
the ANE, and compare against mflux's MLX output layer by layer. That one
experiment exercises risks 1 and 4 and validates the RoPE convention, which
together are the parts most likely to sink the port.

## Sources

- `/Volumes/fxdisk/AIModels/flux2-klein-4b-bf16/{model_index.json, transformer/config.json, vae/config.json, text_encoder/config.json,scheduler/scheduler_config.json}`
- `/Volumes/fxdisk/AIModels/flux2-klein-4b-int8-mlx/transformer/*.safetensors` (headers only)
- `mflux` 0.18.1 in the `mlx` conda env: `flux2_transformer/{transformer,attention,parallel_self_attention}.py`, `attention_utils.py`, `README.md`
