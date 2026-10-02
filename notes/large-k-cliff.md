# The K > 4096 cliff, and the split-K workaround

The single most actionable finding in these notes. Measured with
`scripts/ane_split_k_bench.py`.

## Observation

Square GEMMs never expose this because K = N there. Holding M and N fixed and
sweeping only K (`M=2048, N=3072`, min latency over 6 reps):

| K | ms | TF/s | weight bytes | effective GB/s |
| --- | --- | --- | --- | --- |
| 1024 | 2.237 | 5.76 | 6.3 MB | 4.7 |
| 2048 | 3.752 | 6.87 | 12.6 MB | 5.6 |
| 4096 | 9.167 | 5.62 | 25.2 MB | 4.6 |
| 8192 | 28.891 | 3.57 | 50.3 MB | 2.9 |
| 12288 | 79.337 | **1.95** | 75.5 MB | 1.6 |
| 16384 | 105.941 | **1.95** | 100.7 MB | 1.6 |

Throughput drops by ~3.5x crossing K = 4096, then pins at ~1.95 TF/s: from
K = 12288 to K = 16384 the time grows exactly linearly (79.3 -> 105.9 ms, ratio
1.33 = 16384/12288). So past the cliff the engine is doing *something* at a
fixed, much slower rate rather than degrading gracefully.

It is not DRAM bandwidth: effective GB/s *falls* as throughput falls (4.6 -> 1.6
GB/s), so the memory system is being used less efficiently, not saturated. The
work is big enough that it is not the dispatch floor either (79 ms vs a ~0.1 ms
fixed cost).

## Workaround: split K, add the partials

Slice the activation along K into chunks, run one `linear` per chunk against the
matching weight columns, and add the partial sums. On this engine, chunk sizes of
**1024** win.

Same `M=2048, K=12288, N=3072` GEMM (mono = 80.48 ms, 1.92 TF/s):

| chunk | ms | TF/s |
| --- | --- | --- |
| 1024 | **15.18** | **10.19** |
| 2048 | 16.0 | 9.66 |
| 3072 | 22.2 | 6.96 |
| 4096 | 25.1 | 6.16 |

**5.3x faster, and the output is numerically identical** - the chunked graph and
the monolithic one produce the same Frobenius relative error against an fp64
reference (9.86e-02 both; that large error is the fp16 accumulation over
K=12288, not a split-K artifact). Verified across K=3072/9216/12288 shapes.

The monolithic time here is stable run to run (< 1%), but a repeat of the FLUX.2
battery put the chunked path at 15.9 ms instead of 15.2 ms for this shape -
power/thermal state is worth roughly +/-10% on split-K numbers, so prefer the
ratios over the absolute milliseconds.

Re-measured alone with 15 reps to rule out a timing artifact:

| shape | min ms | med ms | TF/s (min) | TF/s (med) | relerr |
| --- | --- | --- | --- | --- | --- |
| M=2048 K=3072 N=27648, ch=1024 | 38.47 | 39.76 | 9.04 | 8.75 | 1.9e-02 |
| M=2048 K=12288 N=3072, ch=1024 | 14.96 | 15.18 | 10.33 | 10.18 | 9.9e-02 |

**~10 TF/s is above the 6.2 TF/s square-GEMM peak** from
[`ane-gemm-roofline.md`](ane-gemm-roofline.md), which means that peak is a shape
artifact of square GEMMs, not a hardware ceiling. Wide-N GEMMs have more
arithmetic intensity per streamed byte and go faster.

**Update (2026-10-02): the ~10 TF/s is the dispatch rate, not the engine's.**
Every number in this note went through `net(x)`, which pays the host write of
the activation and the read-back of the output (~7-10 GB/s); on wide-N shapes,
where the output is the big tensor, that is what the number was hitting.
Re-measured with the input written once and only `execute()` repeated, the same
chunked shapes run at **15.1 TF/s** (13.4-15.1 across shapes) - table in
[`ane-gemm-roofline.md`](ane-gemm-roofline.md). The cliff itself, the 5.3x
split-K recovery and every ratio here are unaffected (same-harness comparisons).
One refinement: chunk 1024 and 2048 tie at ~15 TF/s execute-only, and 4096 is
already on the cliff (10.07).

## The graph that does it

```python
xs = af.input((M, K))
parts = []
for i in range(0, K, 1024):
  w = min(1024, K - i)
  parts.append(xs.slice_by_size([0, i], [M, w]).linear(W[:, i:i + w]))
y = parts[0]
for p in parts[1:]:
  y = y + p
```

`slice_by_size` with a nonzero last-axis offset has a documented pre-A16 quirk
(`docs/op-catalog.md`); this machine is in the "exact" group, and the numeric
check above confirms it. Verify on older silicon before trusting it there.

## A capacity cliff that is really this cliff: pin K when you sweep

A later sweep looked like an on-chip capacity cliff. It varied an intermediate
tensor from 4 MB to 128 MB with small I/O, and the time jumped 3.0x from 16.8 to
33.6 MB and another 6.3x from 33.6 to 67.1 MB - right around the ~32 MB SRAM
figure people quote for the ANE. It was this cliff in disguise: the sweep used N
as both the intermediate width and the contraction length of the matmul that
consumed it, so every step also lengthened K (K = N, up to 65536).

Re-run with K pinned at 128, I/O at [256,128] in and [N] out, and only the
intermediate `[256, N]` growing, the same matmul scales smoothly:

| intermediate | ms | intermediate traffic |
| --- | --- | --- |
| 4.2 MB | 0.33 | 12.7 GB/s |
| 8.4 MB | 0.46 | 18.3 GB/s |
| 16.8 MB | 0.85 | 19.8 GB/s |
| 25.2 MB | 1.24 | 20.3 GB/s |
| 33.6 MB | 1.63 | 20.6 GB/s |

Throughput *rises* with size and there is no cliff through 33.6 MB; a row
reduction over the intermediate is close to free. A direct test in the
production block agrees: N-chunking the MLP branch so its widest tensor drops
from 37.7 MB to 4.2 MB changes a 63-64 ms block by less than the run-to-run
noise (63.5 / 63.1 / 63.6 / 63.4 ms).

So for these shapes there is no measurable on-chip capacity effect, and the
"restructure to fit ~32 MB" idea is not a lever here. That is not proof the ANE
has no capacity limit - the SRAM size is undocumented and another op pattern
might hit one - but a super-linear jump in a size sweep is more likely to be
this cliff than a capacity effect. Pin K.

## Open questions

- Why 4096? The cliff is sharp enough to look like a hard tile limit in the
  matmul lowering rather than a memory effect. An explanation would say whether
  the compiler *could* tile K itself instead of making the user do it.
- Is the win specific to chunk 1024, or does it track something like
  "K * N bytes per tile below a cache threshold"? The sweep above only had four
  chunk sizes and one (M, N) pair.
- The chunk adds are not free (they stream full `[M, N]` partials), so at small
  M the crossover may move. It was not swept.
