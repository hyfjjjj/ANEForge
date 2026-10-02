# int4 (LUT) weight compression: what it cost, and what it took to use it

`compress="int4"` is the one weight encoding that matches a 4-bit source
checkpoint's size - 0.5 B/param against the 0.5625 B/param an MLX
`quantization_level: 4` package uses - so it is the natural choice when the
checkpoint is already 4-bit. Two things stood between it and being usable at
model scale: the codebook trainer's cost, and an accuracy gate no real weight
could pass. Both are fixed; the accuracy price is now measurable.

## What the trainer cost

`_blob.palettize_lut4` trained its 16 centroids with 20 element-wise Lloyd
iterations, each materializing an `[N, 16]` fp32 distance matrix (plus its
`abs`), so both time and memory were linear in the element count with a
constant ~64x the tensor itself:

| weight | params | `palettize_lut4` | peak RSS |
| --- | --- | --- | --- |
| 3072x3072 | 9.4 M | 7.24 s | 1.65 GB |
| 9216x3072 | 28.3 M | 22.17 s | 4.81 GB |
| 27648x3072 | 84.9 M | 69.78 s | 12.96 GB |

FLUX.2-klein-4B's transformer is ~3.7 G parameters and its largest single
linear is the 27648x3072 fused qkv+MLP projection, so one int4 pass over the
model cost **~45-50 minutes**, with ~13 GB of transient host memory for that one
tensor.

## What replaced it

A per-tensor 16-level codebook is a *1-D* quantizer, so neither the assignment
nor the update needs the `[N, 16]` matrix:

- assign by midpoint lookup - `searchsorted` over the 15 midpoints of sorted
  levels is exact for 1-D, and breaks ties the way `argmin` did;
- update with two weighted `bincount`s (mass and mass*value) over the assigned
  bins instead of 16 masked gathers.

And the k-means objective depends only on the *multiset* of values, not their
positions, so the codebook can be trained on a histogram: one chunked pass
builds a 2^20-bin histogram (over a range taken from a fixed subsample's
1e-5/1-1e-5 quantiles, so a single outlier cannot coarsen the bulk), and Lloyd
then runs on the non-empty bins - a few 100k points instead of 84.9 M.

| weight | params | new | peak RSS | relerr (old -> new) |
| --- | --- | --- | --- | --- |
| 27648x3072 | 84.9 M | **1.89 s** | **1.83 GB** | 0.1121 -> 0.1140 |

38x faster, 7x smaller peak, and the quality is the same in kind: the 1.7%
relerr difference is the clipped range plus the histogram init, and the bin
count does not matter - 2^20, 2^22 and 2^24 bins all give 0.1140 on that tensor.
A whole-model int4 pass is now minutes, not ~50.

## The gate was the other half

`compress_atol` defaulted to 0.05 for every mode, and a per-tensor 16-level
codebook cannot reach that. Measured against the **bf16 original** of the same
checkpoint (all three packages exist on this machine, so this is a real
comparison and not an estimate):

| tensor | source int4 | source int8 | our LUT4 (on bf16) | int4 source + LUT4 |
| --- | --- | --- | --- | --- |
| `single...to_qkv_mlp_proj` (27648x3072) | 0.0940 | 0.0072 | 0.1143 | 0.1439 |
| `single...to_out` (3072x12288) | 0.0926 | 0.0071 | 0.1192 | 0.1463 |
| `transformer...to_q` (3072x3072) | 0.0964 | 0.0075 | 0.1457 | 0.1709 |

So the *source's own* 4-bit package carries a 0.093-0.096 relative error, and
the 0.05 default rejected every one of these weights - int4 fell back to int8
silently (the fallback is a warning, easy to miss in a build log). The default
is now per-mode (`_compile._DEFAULT_ATOL`): **0.2 for int4**, 0.05 for
blockwise, whose per-block int8 sits far under it. The gate still catches
pathological tensors.

Two things fall out of that table:

- **Deploying a 4-bit source as int4 programs is roughly as faithful as the
  source itself** - 0.144-0.171 against the bf16 original, i.e. ~1.5x the
  source's own error, since it is the same weights quantized a second time.
- **int8 programs are ~20x more faithful (0.007) at 2x the size.** Both are
  legitimate points on the curve; the change is that int4 is a choice now
  instead of a 50-minute compile that lands on int8 anyway.

Why a per-tensor codebook cannot do better: 16 levels on gaussian-like weights
floor at ~0.097 (Lloyd-Max), which is exactly where the source's 0.093 sits.
MLX's group-64 affine adaptation buys ~20% over one codebook for the whole
tensor on the worst shapes (0.094 vs 0.114 on the fused projection, 0.096 vs
0.146 on `to_q`); per-row codebooks would close most of the rest for
+0.0104 B/param, since a 3072-wide row costs 32 B to describe.

## The A/B relative error saturates at this error level

fluxlab's `bench_opt/ane_ab.py` verifies a program set by Frobenius relerr
against mflux running the same weights - it is how the int8 path was checked
(5.6e-02). At int4's weight error the same harness reads **0.747**, and that is
not a broken encoding: perturbing the *reference itself* the same way - every
2-D weight of the bf16 reference replaced by its LUT4 round-trip (mean weight
error 0.1188, every other input identical) - moves its output by **relerr
1.0849**. A weight perturbation of ~0.12 is amplified ~9x through 25 blocks
under the harness's random inputs.

So the metric detects small perturbations and saturates for large ones: at
int4's error level it says "different", not "how different". Codebook quality at
this level has to be judged on generated images (or on the weights themselves,
as the table above does).

## What the error does to an image

Measured in fluxlab on the same int4 checkpoint, same prompt and seed, 4 steps,
against the MLX path on that checkpoint:

| program encoding | vs the MLX reference | |
| --- | --- | --- |
| int8 (weight error 0.007) | mean abs diff 5.4/255 | **PSNR 27.1 dB** |
| int4 (weight error 0.14) | mean abs diff 33.9/255 | **PSNR 14.6 dB** |

Both images are clean and of the same apparent quality - but int8 reproduces
the reference's sample almost pixel for pixel, while int4 lands on a *different*
one. A 4-step distilled sampler is that sensitive to a 0.14 weight perturbation;
int4 is a size/fidelity trade (1.86 GB and 1.60 s/step against int8's 3.71 GB
and 1.64 s/step), not a free win.

## Still open

- Per-row (or per-group) LUTs through the lut shape's leading dims: would take
  the worst tensors from 0.146 to ~0.10 for +1% size. Needs a device probe of
  what `constexpr_lut_to_dense` accepts first - nothing here has tried a lut
  whose leading dims are not 1.
