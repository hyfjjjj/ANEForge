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
| [`ane-gemm-roofline.md`](ane-gemm-roofline.md) | How many TOPS does this ANE actually deliver on fp16 GEMM? Square-N sweep, dispatch floor, large-N falloff - and the revisit that separates the host path from the engine: **~15 TF/s execute-only vs 7.7-10 through `net(x)`**, plus what survives inside a six-GEMM program (~11). |
| [`large-k-cliff.md`](large-k-cliff.md) | Throughput collapses below 2 TF/s once K > 4096, and a split-K workaround recovers 5x. The most actionable finding here. Also: a size sweep that looked like a ~32 MB on-chip capacity cliff was this cliff in disguise - pin K when you sweep. |
| [`flux2-klein-4b-ane.md`](flux2-klein-4b-ane.md) | FLUX.2-klein-4B weight inventory, token math for 512x768, measured per-layer ANE cost, and what a port to aneforge would take. |
| [`flux2-block-poc.md`](flux2-block-poc.md) | One single-stream block actually built and running on the ANE vs mflux: RoPE convention, 2048-token attention, fp16 numerics, the layout traps (the transpose-after-concat, and the rank-5 rope slice worth 35 ms/block), and why a sub-graph measurement - including "attention is slow" and the GPU-offload idea - measures the host path instead. |
| [`flux2-transformer-e2e.md`](flux2-transformer-e2e.md) | The whole 25-block transformer on the ANE, vs the MLX/GPU path it would replace: 3.06 vs 2.15 s/step in the POC on an M4 Pro, and what the accuracy and hybrid trade-offs actually look like. Updated with the production port: 1.63 vs 2.20 s/step, and where a steady step actually goes. |
| [`compile-cache.md`](compile-cache.md) | Why `af.compile` recompiled every program every time (`force_recompilation=1` in the shim), the one-line fix, and the 12x startup it buys (115 s -> 9.7 s for FLUX.2, bit-identical output). Also: where that warm build actually goes (a direct program load is 6 ms against 0.3-0.7 s through `af.compile`), that a compiled program is a netlist with weights streamed at dispatch, and the route-optimizer graph memo that pins ~7 GB of weights if it is not cleared. |
| [`int4-lut-cost.md`](int4-lut-cost.md) | `compress="int4"` at real weight shapes: 7.2 s / 1.6 GB for a 3072x3072, 69.8 s / 13 GB for FLUX.2's largest linear, i.e. ~45-50 min for one pass over the model. Why the tests miss it and what would fix it. |

## Scripts

All run from the repo root with `PYTHONPATH=.`, no sudo needed, no files written.

```sh
PYTHONPATH=. python3 notes/scripts/ane_gemm_sweep.py 512 1024 2048 3072 4096
PYTHONPATH=. python3 notes/scripts/ane_split_k_bench.py 2048 12288 3072
PYTHONPATH=. python3 notes/scripts/ane_peak_bench.py
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
  wide-N GEMMs *through the dispatch path*; with the host path out of the way
  (input written once, `execute()` repeated) the engine itself holds **~15 TF/s**
  - 79% of the fp16 half of Apple's 38 TOPS int8, or 40% if that 38 counts MACs.
  So 6.2 TF/s is a shape artifact *and* the ~10 TF/s wide-N figure was the host
  path, not the ceiling. Inside a six-GEMM program (a FLUX.2 block's shape mix),
  ~11 TF/s survives.
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
- `af.compile` recompiled every program on every process start because the
  dispatch shim set `force_recompilation=1`. Gating that flag (reuse the
  content-addressed cache by default, `ANEFORGE_FORCE_RECOMPILE=1` to opt out)
  takes a FLUX.2 rebuild from 115 s to **9.7 s** with bit-identical output. See
  [`compile-cache.md`](compile-cache.md).
- Almost all of that remaining 9.7 s is the cache *lookup*, not the load: a
  compiled program loads from its directory in **6 ms** and never reads
  `weights.bin` (they stream at dispatch), while `af.compile` spends 0.3-0.7 s
  per program re-deriving the content hash. A 27-program FLUX.2 set: 18.5 s
  through `af.compile` vs **0.143 s** loading directly.
- Isolating an op into its own program measures the **host path**, not the op:
  the ANE's host-facing buffers move ~7-10 GB/s. Attention measures 26 ms alone
  but **8.9 ms** inside the fused block, and handing it to the GPU (7.9 ms fn +
  a 38 MB round trip) makes the block *slower* (69.7 vs 63.7 ms).
- The rope's rank-5 last-axis slice is a layout trap of its own: replacing it
  with a permutation-matrix matmul (`out = x*C + swap(x)*S`) takes a
  single-stream block from 101 to **65 ms**, even though the two forms differ by
  only 0.9 ms/tensor in isolation. Across the model that was **-0.7 s/step**.
- `compile(opt="routes")` memoizes graphs by identity and thereby pins every
  weight: five double-stream blocks compiled without clearing the memo leave
  **2.5 GB** more resident than clearing after each compile (~7 GB extrapolated
  to the full 27-program set).
- `compress="int4"` costs **69.8 s and 13 GB of transient memory** for FLUX.2's
  largest linear alone (it trains a codebook over every element of the tensor),
  which is why 4-bit checkpoints currently land on int8 programs instead. See
  [`int4-lut-cost.md`](int4-lut-cost.md).
- The weights stream from `weights.bin` at dispatch, and the **first execute of
  a freshly loaded program pays for it**: ~5 ms more than its steady state per
  block, ~0.14 s over a 27-program set, mostly a fixed per-program cost (a
  0.4 MB program still pays 0.9 ms). It is charged per release+reload cycle,
  not per process. See [`compile-cache.md`](compile-cache.md).
- A production FLUX.2 step is **97.8% ANE execute, 1.9% host copies between
  programs (~10 GB/s), 0.2% everything else** - and a generation loop's printed
  "per step" is an average whose first step is ~0.29 s slower, which is how a
  phantom 90 ms/step of host overhead appeared. See
  [`flux2-transformer-e2e.md`](flux2-transformer-e2e.md).
