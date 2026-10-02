# int4 (LUT) weight compression is unusable at model scale

`compress="int4"` is the one weight encoding that would match a 4-bit source
checkpoint's size - 0.5 B/param against the 0.5625 B/param an MLX
`quantization_level: 4` package uses - so it is the natural choice when the
checkpoint is already 4-bit. It is also far and away the slowest thing in the
library, because `_blob.palettize_lut4` trains its codebook on the whole tensor:

```python
def palettize_lut4(W):
  flat = W.reshape(-1).astype(np.float32)
  ...
  for _ in range(20):
    idx = np.abs(flat[:, None] - centroids[None, :]).argmin(axis=1)
```

Each of the 20 Lloyd iterations materializes an `[N, 16]` fp32 distance matrix
(plus its `abs`), so both time and memory are linear in the element count with a
constant ~64x the tensor itself.

Measured on an M4 Pro (one process per size, `ru_maxrss` after the call):

| weight | params | `palettize_lut4` | peak RSS |
| --- | --- | --- | --- |
| 3072x3072 | 9.4 M | 7.24 s | 1.65 GB |
| 9216x3072 | 28.3 M | 22.17 s | 4.81 GB |
| 27648x3072 | 84.9 M | 69.78 s | 12.96 GB |

FLUX.2-klein-4B's transformer is ~3.8 G parameters and its largest single
linear is the 27648x3072 fused qkv+MLP projection, so one int4 pass over the
model costs **~45-50 minutes**, and that one tensor alone needs **~13 GB** of
transient host memory. The library's own tests never see this: they train
codebooks on 32x64 and 16x16 tensors (`tests/test_compress.py`).

Two changes, both local to `palettize_lut4`, would make it usable:

- train the 16 centroids on a subsample - a few 100k values are plenty for 1-D
  k-means - instead of every element, and
- assign indices in row blocks, since only the assignment step needs the
  `[N, 16]` temporary.

Whether the surrounding accuracy gate (`compress_atol`, default 0.05, falling
back per tensor to per-channel int8 with a `CompressionFallbackWarning`) would
then accept FLUX.2's weights is a separate, unmeasured question - one 16-entry
codebook per weight matrix is a coarse approximation, and a rejected tensor
lands at int8, twice the size the source checkpoint had.

Context for why this matters: ANE compute is fp16 either way, so the encoding
only decides memory and disk footprint. A caller with a 4-bit checkpoint
currently has to choose between int8 programs (1.7x the source size) and a
~45 minute compile.
