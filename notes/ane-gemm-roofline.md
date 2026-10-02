# ANE fp16 GEMM throughput on an M4 Pro

Measured with `scripts/ane_gemm_sweep.py`: square `N x N` fp16 GEMM as one
fused program, `af.compile(af.input((N, N)).linear(W))`, min latency over N reps
after warmup. FLOPs counted as `2*N^3`, so TF/s and TOPS are the same number
here (1 MAC = 2 ops).

## Result

| N | min ms | med ms | TF/s (min) | TF/s (med) | Frobenius relerr |
| --- | --- | --- | --- | --- | --- |
| 512 | 0.162 | - | 1.66 | - | 2.5e-03 |
| 1024 | 0.715 | - | 3.00 | - | 6.4e-03 |
| 2048 | 2.874 | 3.038 | 5.98 | 5.66 | 1.2e-02 |
| **3072** | **9.280** | 9.715 | **6.25** | 5.97 | 1.9e-02 |
| 4096 | 23.480 | 24.997 | 5.85 | 5.50 | 2.7e-02 |
| 5120 | 54.857 | 55.765 | 4.89 | 4.81 | 3.5e-02 |
| 6144 | 127.756 | 128.596 | 3.63 | 3.61 | 4.4e-02 |
| 8192 | 380.300 | 406.058 | 2.89 | 2.71 | 6.2e-02 |

**Peak ~6.2 TF/s at N = 2048-3072.** First-pass numbers taken on a cold process
read ~20% lower at the same sizes (N=2048: 4.81 TF/s), so warm up before timing
anything on this engine.

Cross-check: the committed `bench/results/rooflines/` entry for M4 Pro recorded
`gemm_peak_gflops = 6199.8` at N=2048 (6.20 TF/s) from a separate session on this
same machine - consistent with the table above.

## Reading it

- **Small N is dispatch-bound.** N=512 runs at 1.66 TF/s because the fixed
  per-call cost dominates; `af.compile` emits a `DispatchFloorWarning` for it.
  Roughly 60-100 us of the 162 us is fixed overhead.
- **Peak is reached around N = 2048-3072**, i.e. exactly where the arithmetic
  intensity of a square GEMM outruns the fixed cost but the working set still
  fits the engine's tiling.
- **N >= 4096 falls off** (5.85 -> 4.89 -> 3.63 -> 2.89 TF/s). This matches the
  "large-square-GEMM falloff" the repo's `bench/device_saturation_sweep.py`
  already sweeps for.
- **TOPS context.** Apple rates the M4 Pro ANE at 38 TOPS. The 6.2 TF/s above is
  ~16% of that for dense fp16. That gap is expected - the marketing figure is an
  int8 peak - but note `af.compile(..., int8=True)` is *not* an int8 MAC path: it
  only stores linear weights per-channel int8 and dequantizes during the tile
  DMA, so it cannot be used to probe the int8 ceiling from this API.
- **Precision.** Frobenius relative error grows with K (2.5e-03 at N=512 to
  6.2e-02 at N=8192). This is fp16 accumulation, consistent with the
  `reduce exact-sum <= 2048` cliff in
  [`docs/numeric-cliffs.md`](../docs/numeric-cliffs.md), and it matters when
  composing many large-K GEMMs in a row.

## Revisited: pure execute vs dispatch, and what the ceiling really is

The 6.2 TF/s above - and the ~10 TF/s wide-N figure in
[`large-k-cliff.md`](large-k-cliff.md) - were both measured through `net(x)`,
which includes the host-side write of the activations into the program's input
buffer and the read of the output back. That path moves ~7-10 GB/s, so on
wide-N shapes, where the output is the big tensor, it was setting the number,
not the engine.

Measured with the input written once through `input_view` and only `execute()`
repeated (`notes/scripts/ane_peak_bench.py`, min of 8 after 3 warmups, same
machine):

| M | K | N | dispatch ms | TF/s | execute ms | **TF/s** |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 3072 | 3072 | 3.84 | 10.07 | 2.82 | 13.70 |
| 2048 | 3072 | 9216 | 13.32 | 8.70 | 7.99 | 14.52 |
| **2048** | **3072** | **27648** | 37.95 | 9.17 | **23.46** | **15.07** |
| 2048 | 12288 | 3072 | 15.79 | 9.79 | 11.56 | 13.37 |
| 1024 | 3072 | 27648 | 19.82 | 8.78 | 11.76 | 14.79 |
| 4096 | 3072 | 9216 | 27.23 | 8.52 | 15.65 | 14.82 |
| 8192 | 3072 | 3072 | 20.13 | 7.68 | 11.24 | 13.76 |

**The engine sustains ~15 TF/s of fp16 GEMM, flat at 13.4-15.1 across shapes.**
A second run reproduced the execute column within ~2%; the dispatch column
varies more (the first shape measured in a process can read 20% high on
dispatch, while its execute time is unchanged).

The dispatch column is 1.5-2x lower, and the difference is exactly the host
traffic: M=2048/N=27648 reads back a 113 MB output and writes a 12.6 MB input,
~14.5 ms of the 37.95, i.e. ~8.7 GB/s - the same host-path rate the sub-graph
measurements in [`flux2-block-poc.md`](flux2-block-poc.md) ran into.

TOPS context, restated: 15.1 TF/s is **79% of the fp16 half of Apple's 38 TOPS
int8** (19), or 40% if that 38 is counted in MACs - in which case an fp16 peak
would be 38 TFLOP/s in this note's FLOP=2xMAC convention. Which reading applies
is not something this API can settle: `int8=True` stores weights int8 and
dequantizes during the DMA, it is not an int8 MAC path.

Split-K chunk sizes, execute-only, at M=2048 K=3072 N=27648:

| chunk | ms | TF/s |
| --- | --- | --- |
| 512 | 28.23 | 12.32 |
| 1024 | 23.09 | 15.07 |
| 2048 | 23.14 | 15.03 |
| 4096 | 34.55 | 10.07 |

This refines [`large-k-cliff.md`](large-k-cliff.md)'s "chunk 1024 wins": 1024
and 2048 are the same within noise, and 4096 is already on the cliff.

Two negative results worth keeping:

- **Independent GEMM chains in one program do not overlap.** Two identical
  [2048,3072]x[3072,13824] chains in one program take 24.52 ms; as two separate
  programs, 2 x 11.64 = 23.28 ms. Four chains: 49.05 vs 46.6. The engine runs
  them in order, so batching independent GEMMs buys nothing (it costs ~5%).
- **1x1 conv is not a faster GEMM** (so far). `af.conv` with x `[1,K,M,1]` or
  `[1,K,1,M]` against a `[N,K,1,1]` weight did not reproduce the GEMM at all
  (Frobenius relerr ~1.2 in both layouts, against 1.4e-02 for the same shape
  through `linear`), so those timings cannot be read as a conv-path GEMM rate.
  What did come out (9.5-10 TF/s) is not promising enough to chase the layout
  question further - open for whoever knows the conv lowering.

## And inside a multi-GEMM program: ~11 TF/s

15 TF/s is one GEMM alone in a program. Built the way the production FLUX.2
single-stream block is - the same six linears, same shapes, same chunking and
weight slices, with rope/norm/attention/silu replaced by adds so nothing gets
dead-code-eliminated - those six GEMMs take **45.87 ms of execute time, 10.96
TF/s**, against ~36 ms if the same six ran at their isolated rates (3 x 2.82 +
2 x 7.99 + 11.56), i.e. **+27%**.

Two identical chains in one program cost only ~5% over running them separately,
so the tax is not "any second GEMM" - it is specific to this mix: five different
N, a K-chunked `to_out` whose 12 partials are summed, and the live-tensor
pressure of the [2048, 3072] fp16 intermediates.

That gives the block its first honest budget, all execute-only: **62.4 ms =
45.9 GEMM + ~8.9 attention + ~7.6 everything else** (norm, rope, silu, gate,
residual) - see [`flux2-transformer-e2e.md`](flux2-transformer-e2e.md) for the
step-level version. A production step runs at 8.9 TF/s effective, 81% of the
10.96 its own GEMMs reach in context.

## Caveat

The square shapes above are not the engine's compute ceiling - the wide-N
chunked shapes reach ~15 TF/s execute-only. What none of these numbers settle is
how much of the 15 TF/s survives when a program has to interleave six GEMMs with
attention and elementwise work; the multi-GEMM measurement above says roughly
two thirds of it.
