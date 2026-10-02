# `notes/` - on-device measurement notes

Working notes from a set of ANE measurements done while evaluating whether a
diffusion transformer (FLUX.2-klein-4B) could be ported onto aneforge. Separate
from [`docs/`](../docs/) on purpose: these are raw findings and open questions,
not curated project documentation. Nothing here is wired into the MkDocs build.

Everything was measured on the machine below, with the scripts in
[`scripts/`](scripts/), each in a warm process (warmup dispatches before timing,
min-of-N latency). Numbers are single-machine and power-state dependent - treat
them as indicative, not as a cross-chip roofline. For the curated, CI-aggregated
cross-machine numbers, see [`bench/results/ROOFLINES.md`](../bench/results/ROOFLINES.md).

## Environment

| | |
| --- | --- |
| Chip | Apple M4 Pro (Mac16,7), 14 CPU cores, 20 GPU cores, 48 GB |
| macOS | 27.0.1 (build 26A434) |
| Power | AC attached, battery not charging |
| aneforge | 0.4.1.dev36, `dc266086bb59` (main) |
| Dtype | fp16 activations and weights (aneforge's default path) |

## Contents

| Doc | What it answers |
| --- | --- |
| [`ane-gemm-roofline.md`](ane-gemm-roofline.md) | How many TOPS does this ANE actually deliver on fp16 GEMM? Square-N sweep, dispatch floor, large-N falloff. |
| [`large-k-cliff.md`](large-k-cliff.md) | Throughput collapses below 2 TF/s once K > 4096, and a split-K workaround recovers 5x. The most actionable finding here. |
| [`flux2-klein-4b-ane.md`](flux2-klein-4b-ane.md) | FLUX.2-klein-4B weight inventory, token math for 512x768, measured per-layer ANE cost, and what a port to aneforge would take. |
| [`flux2-block-poc.md`](flux2-block-poc.md) | One single-stream block actually built and running on the ANE vs mflux: RoPE convention, 2048-token attention, fp16 numerics, and the layout trap that made it 2.5x slower. |
| [`flux2-transformer-e2e.md`](flux2-transformer-e2e.md) | The whole 25-block transformer on the ANE, vs the MLX/GPU path it would replace: 3.06 vs 2.15 s/step on an M4 Pro, and what the accuracy and hybrid trade-offs actually look like. |
| [`compile-cache.md`](compile-cache.md) | What `af.compile`'s content-addressed cache actually buys (re-emission, ~0.15 s/program) and what it costs (7.35 GB per build) - and why compiling the same program again is not faster. |

## Scripts

All run from the repo root with `PYTHONPATH=.`, no sudo needed, no files written.

```sh
PYTHONPATH=. python3 notes/scripts/ane_gemm_sweep.py 512 1024 2048 3072 4096
PYTHONPATH=. python3 notes/scripts/ane_split_k_bench.py 2048 12288 3072
PYTHONPATH=. python3 notes/scripts/flux2_klein_layer_bench.py
```

`flux2_klein_layer_bench.py` is pure shape simulation - it does not read the
checkpoint, so it runs anywhere.

The three FLUX.2 POCs (`flux2_single_block_poc.py`, `flux2_double_block_poc.py`,
`flux2_transformer_poc.py`) additionally need `mlx` + `mflux` and the model
checkpoint on disk, and must run in an env that has aneforge installed (e.g.
fluxlab's `.venv`), not with `PYTHONPATH=.`:

```sh
python3 notes/scripts/flux2_transformer_poc.py --txt 512 --img 1536
```

## Headline numbers

- Square fp16 GEMM peaks at **~6.2 TF/s** (N=2048-3072); small N is
  dispatch-bound, N >= 4096 falls off.
- With K split into <= 1024 chunks, the same hardware sustains **~10 TF/s** on
  wide-N GEMMs - so 6.2 TF/s is a shape artifact, not a hardware ceiling.
- FLUX.2-klein-4B at 512x768 is 1536 image + 512 text tokens; its weight GEMMs
  cost **3.20 s/denoise step** monolithic vs **1.4-1.5 s/step** with split-K
  (+/-10% run to run; see the variance note in the FLUX.2 doc).
- One real single-stream block, built as a graph and measured end to end, is
  **113.65 ms** at 2048 tokens, so a whole-file budget is closer to
  **2.5-3 s/step** than the GEMM-only number above.
- The full 25-block transformer runs end to end at **3.06 s/step** (ANE) against
  **2.15 s/step** for the same weights through MLX on the GPU - the port is
  sound but ~40% slower. See [`flux2-transformer-e2e.md`](flux2-transformer-e2e.md).
- Streaming int8 weights (`int8=True`) takes it to **2.54 s/step** and halves
  the program size, at the cost of an extra quantization round
  (3.9e-02 -> 6.0e-02 against the same bf16 reference).
- `af.compile`'s on-disk cache does **not** make a repeat compile faster
  (2.39 s cold vs 2.45 s warm, clean A/B); it only skips re-emission. See
  [`compile-cache.md`](compile-cache.md).
