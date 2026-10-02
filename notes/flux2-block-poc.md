# FLUX.2 single-stream block on the ANE: a working POC

`scripts/flux2_single_block_poc.py` builds one FLUX.2 single-stream block as a
single aneforge graph and runs it on the ANE, comparing against mflux's own MLX
module. This is the experiment that decides whether the FLUX.2 port is viable -
it exercises the three risks flagged in [`flux2-klein-4b-ane.md`](flux2-klein-4b-ane.md):
the 4-axis RoPE convention, attention at 2048 tokens, and fp16 numerics.

**Outcome: all three clear.** The block runs, matches mflux within fp16
rounding, and got 2.5x faster once two layout traps were found.

## What the block does

```
proj = to_qkv_mlp_proj(x)      [S, 27648] = qkv 9216 | mlp gate+up 18432
q,k,v = split(qkv)             [S, 3072] each, 24 heads x 128
q,k   = rms_norm(head_dim), then 4-axis interleaved RoPE
attn  = sdpa(q, k, v, scale=1/sqrt(128))
mlp   = silu(gate) * up        [S, 9216]
out   = to_out(concat([attn, mlp]))    [S, 12288] -> [S, 3072]
```

Weights come from the int8 checkpoint via `mx.dequantize`, so both engines see
identical numbers. The reference is mflux's `Flux2ParallelSelfAttention` with
weights cast to fp32 (ground truth) and bf16 (what the MLX backend actually
runs: `ModelConfig.precision = bfloat16`).

## Correctness

| S | ANE fp16 vs fp32 | mflux bf16 vs fp32 | RoPE tables vs mflux |
| --- | --- | --- | --- |
| 256 | 4.43e-03 | 4.45e-03 | 1.9e-06 |
| 2048 | 4.48e-03 | 4.34e-03 | 1.5e-05 |

**The ANE's fp16 path is as accurate as the bf16 the MLX backend already ships**
(both ~4.4e-03 Frobenius relative error against fp32). RoPE tables match mflux's
`Flux2PosEmbed` to fp32 rounding, which pins the 4-axis convention: the tables
are `[S, 64]` (4 axes x 16 pairs) and the rotation is the **interleaved pair**
layout (`x[0::2]`/`x[1::2]`), not aneforge's half-split `rope()` helper - the
block constructs it explicitly, and a wrong pairing would show up as O(1) error,
not 4e-3.

Both sdpa regimes are covered: S=256 takes the **native** fused layer
(`min(seq) < 512`), S=2048 takes the **decomposed tiled** path (the native layer
requires `min(q,k seq) < 512`, so a full 2048x2048 never uses it).

## Performance at S=2048 (one block, M4 Pro)

| configuration | ms | note |
| --- | --- | --- |
| initial | 280.74 | monolithic to_out (K=12288), rope on the flattened [S, 3072] |
| + `--split-k 1024` | 225.23 | chunks the K=12288 `to_out` (see [`large-k-cliff.md`](large-k-cliff.md)) |
| + `--split-proj` | 220.59 | five separate linears instead of one fused GEMM + five slices |
| + rope after the head transpose | **113.65** | see below - the big one |

Numerics are unchanged across all four (4.44-4.48e-03).

## The layout trap: a transpose after a concat costs 150x more

The single largest win was reordering the graph, not removing work.

| op sequence | ms at S=2048 |
| --- | --- |
| `linear -> reshape -> transpose` | 7.27 |
| `linear -> reshape -> transpose` (isolated, 12.6 MB tensor) | 2.65 |
| `rope -> transpose` | 89.53 |
| `transpose -> rope` (same math, transpose moved before the rope) | **27.40** |

Marginally, the transpose costs ~0.4 ms when its producer is a matmul and ~62 ms
when its producer is the rope's `concat`. The transpose itself is not slow - the
compiler folds it into a matmul's output layout, but cannot fold it into a
`concat`. Reshapes are free in both cases (verified: `rope -> reshape(S,H,D) ->
reshape back` = 25.71 ms, same as `rope` alone at 26.06 ms).

So the fix is ordering: transpose to `[1,H,S,D]` immediately after the q/k
projections, do the rms_norm and the RoPE there (rank-5 pair slicing on the last
axis), and feed the rope's concat straight into sdpa. The block graph is 2.5x
faster for it.

**This looks like a compiler issue worth reporting**: a layout-only consumer
(the transpose) should not cost 150x more depending on its producer's op kind.

**Follow-up, measured later in the full production port: the rank-5 last-axis
slice is itself a layout trap** - see the next section.

## Follow-up: the rank-5 slice costs more than the rope

The fix above still views the tensor as rank-5 `[1,H,S,64,2]` and slices the
LAST axis for the real/imaginary parts. In the full model that slice turns out
to be expensive: it keeps the compiler from holding the matmul layout of the
`rms_norm` output feeding it, so every block pays whole-tensor layout
conversions. Replacing it with a permutation-matrix matmul plus a bit-expanded
cos/sin table

    out = x*C + swap(x)*S     # C[2i]=C[2i+1]=cos_i, S[2i]=-sin_i, S[2i+1]=+sin_i
    swap(x) = x @ P           # 128x128 permutation matrix, P[i, i^1] = 1

is element-wise identical (negating sin is exact) and takes a single-stream
block from **101 to 65 ms** at S=2048.

| rope form | isolated | in the block |
| --- | --- | --- |
| rank-5 view, last-axis slice, 4 arith, concat | 4.76 ms/tensor | 101 ms/block |
| permutation-matmul swap + expanded tables | 3.87 ms/tensor | **65 ms/block** |

The 0.9 ms/tensor gap in isolation is not where the 35 ms/block comes from - the
surrounding ops decide what layout the slice lands in, and in the real graph
they do not have one to spare. Same lesson as the transpose, one level down:
prefer ops that keep the matmul layout (a matmul swap) over ops that force a
repack (a last-axis slice of a rank-5 view), and measure in the full graph.

Two smaller measured items from the same pass, for completeness: splitting the
fused `to_qkv_mlp_proj` into five linears and summing the `to_out` K blocks
instead of concatenating is worth 2.1 ms/block (it removes five slices of a
[S,27648] tensor and one [S,12288] concat); and the block's weight bytes are on
the critical path - the same block compiles to 114.7 ms with fp16 weights and
101.2 ms with int8, so every halving of the weight encoding buys ~12%.

## Other measured pieces (S=2048, isolated)

| piece | ms |
| --- | --- |
| `sdpa` alone, q/k/v already resident | 26.27 |
| q linear [2048,3072]x[3072,3072] | 6.79 |
| rms_norm + rope (flat layout) | ~18.6 |
| `transpose` alone | 2.65 |

Attention is *not* the bottleneck - 26 ms of a 114 ms block - which is the
opposite of the intuition from the FLOP counts (QK^T + AV = 51.6 GFLOP, ~8.6 ms
at the measured GEMM rate). Query-tiling the native sdpa was tried and did not
help (225 ms either way), so the decomposed path is what a port should use.

Caveat added later: that 26 ms is a sub-graph measurement and is inflated by the
host path - see "A sub-graph measurement measures the host path" below. Inside
the fused block the attention is ~8.9 ms.

## A sub-graph measurement measures the host path

Isolating an op into its own program measures the transfer in and out of it, not
the op. The ANE's host-facing buffers move data at **~7-10 GB/s** (measured: a
201 MB tensor round-trips in 58.8 ms, and adding two more elementwise passes
over it changes nothing, so the internal work is not what costs). Any sub-graph
whose I/O is large is therefore dominated by transfer - which is exactly the
case for attention, whose q/k/v are 38 MB in and 12.6 MB out at S=2048:

| attention at S=2048, 24 heads | ms |
| --- | --- |
| compiled alone with q/k/v as inputs | 26.3 |
| the same attention inside the fused block (from splitting the block) | **8.9** |
| the GPU's fused kernel, same shapes, data already resident | 7.9 |

In the fused graph the attention runs at ~5.8 TF/s - in line with a dense matmul
of the same size (5.5) - and the softmax's 2048-wide row reductions and `exp`,
which dominate the isolated number, are largely free once the compiler keeps the
intermediate on-chip. The lesson is the same as the layout traps: measure
whole-graph, or diff two graphs with the same I/O shape, and never conclude from
a sub-graph's absolute milliseconds.

That also settles the obvious idea of handing attention to the GPU:

| single-stream block | ms |
| --- | --- |
| fused, everything on the ANE | 63.7 |
| split: ANE front half | 44.3 |
| q/k/v out (38 MB), GPU attention, result back (12.6 MB) | 14.9 |
| ANE back half | 10.5 |
| split total | **69.7** |

The round trip plus the extra program costs more than the 8.9 ms of attention it
replaces, and even with the transfer set to zero it only breaks even
(44.3 + 7.9 + 10.5 = 62.7). Sharing the buffers instead of copying them is not
an option: the attention's intermediates live inside the compiled program, not
in any addressable memory, so a cross-engine split has to materialize them on
the program's ports first - which is the split.

## What this does to the earlier estimate

`flux2-klein-4b-ane.md` estimated 1.4-1.5 s per denoise step from GEMMs measured
in isolation. A real block is ~114 ms, so 20 single-stream blocks alone are
~2.3 s/step - the plumbing between GEMMs, not the GEMMs, dominates. Add the 5
double-stream blocks (smaller M, two streams) and the embeddings/modulation, and
a realistic budget is **~2.5-3 s/step, ~10-12 s per 4-step image** for the
transformer.

Caveats: this is one block compiled as its own program. A 25-block graph may fuse
better, or may hit memory limits - untested. It also excludes attention-related
work beyond the block (none), the inter-op boundaries of a full model, the text
encoder, and the VAE.

## Open questions

- Why does the transpose cost 150x more after a concat? If a compiler fix lands,
  the reordering workaround may become unnecessary.
- Would fusing more of the block into fewer ops help further? The remaining
  113 ms is spread across ~20 ops on [2048, 3072]-sized tensors.
- Does the reordered graph still behave at other sequence lengths (only 256 and
  2048 were tested)?
- `rms_norm` is 2D-only, which forces reshape gymnastics around it; a 3D/4D form
  would simplify the graph.
