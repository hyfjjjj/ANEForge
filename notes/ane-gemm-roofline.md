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

## Caveat

These are square shapes. The number is *not* the engine's compute ceiling - see
[`large-k-cliff.md`](large-k-cliff.md), where chunked wide-N GEMMs reach ~10 TF/s
on the same hardware.
