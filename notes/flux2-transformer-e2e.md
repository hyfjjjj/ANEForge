# FLUX.2-klein-4B transformer end to end: the ANE port vs the GPU it would replace

The whole model - 5 double-stream blocks, 20 single-stream blocks, embedders,
modulation, norm_out, proj_out - built on aneforge, run on the ANE at 512x768
(1536 image + 512 text tokens), and compared against mflux's own MLX/GPU path
running the same weights.

`scripts/flux2_transformer_poc.py`. Each block is its own compiled program
(25 weight sets, ~7.75 GB in fp16); the time/guidance embedding stays host-side.

## Verdict

**The port works and is numerically sound. It is also ~40% slower than the GPU
path it would replace.**

| | accuracy vs fp32 (25 blocks) | warm speed |
| --- | --- | --- |
| MLX bf16 (shipped) | 3.572e-02 | **2.15 s/step** |
| ANE fp16 (this port) | 4.822e-02 | 3.06 s/step |

Both are "a few percent"; the ANE is 1.35x further from fp32 than bf16. Per
block the ANE looked *better* than bf16 (4.5e-03 vs 4.3e-03); over 25 blocks the
bf16 format's wider exponent range wins, and fp16's per-op advantage does not
accumulate.

The MLX number is a warm 3-run minimum with the model loaded (reload included:
2.86/2.30/2.15 s). An earlier 4.4 s/step figure came from a harness that reloaded
weights every call - not a fair comparison, and worth stating because it inverted
the conclusion.

**Superseded (2026-10-02): the port moved into fluxlab and now runs 1.63 s/step
against the GPU's 2.20 s/step - 26% faster, not 40% slower.** The update section
below has the production breakdown; the POC numbers that follow stay for the
record.

## Update: the production port (fluxlab), and where a step actually goes

Everything below this section is the POC as first measured. Moved into
fluxlab's ANE backend - int8 streaming everywhere, split-K by default, the rope
rewritten as a permutation matmul, modulation computed on the GPU, zero-copy
program boundaries - the same transformer runs **1.63 s/step** warm at 512x768
against **2.20 s/step** for the MLX/GPU path on the same machine, same
checkpoint, same prompt, measured minutes apart: the ANE is now 26% *faster*
than the GPU it would replace, where the POC was 40% slower. Numerics are at
the int8-streaming level (relerr vs mflux bf16 5.7e-02), and end to end the
image takes 8.3 s (ANE) vs 10.5 s (GPU), text encoder and VAE included.

Where a steady step goes, with timers wrapped around the real generation loop
(4 steps, same process, `bench_opt/ane_step_breakdown.py`):

| segment | ms/step | share |
| --- | --- | --- |
| execute (27 programs, ANE compute + wait) | 1591.5 | 97.8% |
| write inputs (host memcpy between program buffers) | 31.5 | 1.9% |
| everything else | 3.7 | 0.2% |

"Everything else" is 3.7 ms of: the mx->numpy input conversion (0.9), reading
the output view (0.8), the loop's `mx.eval` (1.8), the memoized modulation
(0.2), the RoPE tables (0.1) and mflux's scheduler (0.03).

The 1.9% is the price of one program per block: 27 boundaries carrying up to
12.58 MB each ([2048, 3072] fp16), ~320 MB/step at ~10 GB/s - the same
host-facing buffer bandwidth the sub-graph measurements in
[`flux2-block-poc.md`](flux2-block-poc.md) ran into. `Program.share_buffer`
only rebinds ports *inside* a program, so a cross-program hand-off without the
copy is not available today; fusing blocks is the only way to remove
boundaries (worth 3% at 2 blocks/program, and the compiler rejects 4).

The execute time is 13.86 TFLOP / 1591.5 ms = **8.7 TF/s**, the same rate a
single block sustains on its own, so assembling 25 of them adds no per-step
overhead beyond the copies. The remaining distance to the engine's best
measured GEMM rate (~10 TF/s, [`large-k-cliff.md`](large-k-cliff.md)) lives
inside the programs - layout conversions, softmax, elementwise ops - which is
what the block-level tuning addresses.

One methodology trap, because it cost a detour: the per-step number a
generation loop prints is an **average over its steps**, and step 1 is ~0.29 s
slower than the rest (see [`compile-cache.md`](compile-cache.md)), so a 4-step
run reports 1.72 s/step against a 1.64 s steady state. Comparing that average
with a min-of-N harness number produced a phantom ~90 ms/step of "host
overhead"; the same-process breakdown above is what disproved it. Compare like
with like. (The GPU path has the same shape: 2.20 s/step steady, 2.71 s for
step 1.)

## Where the time goes (per step, 2048 tokens)

| program | ms | | program | ms |
| --- | --- | --- | --- | --- |
| embed | 15.7 | | 20 x single | 114-120 each |
| 5 x double | 137-139 each | | head | 2.9 |

Total 3.06 s. Single blocks dominate (2.34 s of it); a double block costs about
the same as a single block plus the second stream's feed-forward.

## Optimizations, in order of what they were worth

| lever | effect |
| --- | --- |
| RoPE before the head transpose, not after | 280 -> 114 ms per single block ([`flux2-block-poc.md`](flux2-block-poc.md)) |
| Release the raw checkpoint before inference | 5.98 -> 3.06 s/step |
| Split-K on the K=12288 `to_out` | included above |
| Fuse 2 blocks per program | 3.06 -> 2.97 s/step (3%) |
| Fuse 4 blocks per program | **fails**: `ane_e5rt_program_compile failed` |

The second row is the surprising one: holding the 4 GB of dequantized-but-unused
checkpoint shards alive while dispatching inflated every block by ~2x. The
per-op cost of this port is sensitive to machine memory state; benchmarks that
keep weights resident for convenience are measuring the wrong thing.

Block fusion - the obvious way to remove the 25 program boundaries - barely
helps at 2 and is rejected by the ANE compiler at 4, so it is not a path to
closing the gap.

## int8 weight streaming: +17% and half the disk

`af.compile(..., int8=True)` streams linear weights as per-channel int8 and
dequantizes during the tile DMA. It is worth a look here because the checkpoint
is already int8-quantized - but the two schemes are **not the same thing**:

| | scheme | format |
| --- | --- | --- |
| checkpoint (mflux/MLX export) | group-64 affine, zero point folded into the bias | **uint8** 0..255, one scale/bias per 64 elements |
| aneforge `int8=True` | its own per-channel streaming quantization | **signed int8**, one scale per output channel |

So the weights are quantized twice on this path: uint8/group-64 -> (dequantize)
-> fp16 -> (aneforge requantizes) -> int8/per-channel. That is where the
accuracy cost comes from.

| weights | s/step | 4 steps | relerr vs mflux bf16 |
| --- | --- | --- | --- |
| fp16 | 3.06 | 12.3 s | 3.907e-02 |
| **int8** | **2.54** | **10.2 s** | 6.013e-02 |

Per-block that is 112.9 -> 99.6 ms, and the program on disk halves (236 -> 119 MB
of `weights.bin`). Both numbers are against the same bf16 reference, so the
3.9e-02 -> 6.0e-02 move is the extra quantization, and it is the price of the
17%. Feeding aneforge per-group uint8 directly would avoid the round trip, but
its interface is per-channel.

One caution on the MLX side of these comparisons: warm single measurements of
the reference moved between 2.15 and 4.41 s/step across runs on this machine.
The ANE numbers were stable (3.06, 3.08); prefer them for A/B.

## Compile cost

Building the 26 programs took ~115 s per process in every run measured above,
because the dispatch shim forced recompilation and never read the cache back.
Gating that flag (see [`compile-cache.md`](compile-cache.md)) brings a warm
build down to **9.7 s** with bit-identical output - so the first run pays
~86-115 s and every run after that pays ~10 s. The numbers in this doc were
measured before the fix and are unaffected by it (inference timing is the same).

## On the hybrid idea

Nothing measured here is "unoptimizable on the ANE" in a way that moving it to
the GPU would obviously fix. The gap is systemic: the ANE spends its time on
per-op plumbing around medium-sized GEMMs (~4.3 TF/s effective across the block,
against ~7 TF/s in the GEMMs themselves), while the GPU runs the same graph at
~6 TF/s effective with elementwise ops near-free.

The blocks are strictly sequential, so splitting one image's blocks between ANE
and GPU gains nothing - there is no parallelism to exploit. The hybrid that
could pay is **across requests**: the ANE and the GPU are separate engines, so
serving two images concurrently (one per engine) should beat either alone. The
repo's own `bench/device_serving_sweep.py` measures exactly that kind of
crossover on this silicon.

Where the ANE port plausibly still wins is **energy**, not latency: the M4 Pro
roofline submission has the ANE at ~2x the GPU's GFLOP/s per watt (318 vs 158
GF/s/W at N=512, short-window indicative samples). That is the case for this
port to make - and it is untested here.

## Caveats

- One machine (M4 Pro, macOS 27.0.1, AC), one resolution, synthetic inputs.
  Accuracy is measured against a second approximation (mflux), not against
  generated images.
- The ANE time excludes the text encoder and VAE, and excludes compile time
  (~115 s for the 26 programs, once per process - real, but amortized).
- Inputs are random, so activation magnitudes are plausible but not the real
  distribution; the fp16-vs-bf16 comparison should be redone on a real denoise
  trajectory before trusting the accuracy verdict.
- The MLX baseline runs the same transformer only; fluxlab's end-to-end numbers
  include the text encoder and VAE.

## Open questions

- Would the accuracy hold on a real 4-step trajectory (image-level check)?
- Is the ANE's energy advantage over the GPU large enough to matter for a
  batch-1 image generator? Needs a `powermetrics` run.
- The 2-block fusion gave 3%; would fusing the *double* blocks with their
  following single block do better, or also hit the compiler limit?
