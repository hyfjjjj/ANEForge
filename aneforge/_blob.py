"""BLOBFILE weight container: 64-byte header, then per blob a 64-byte descriptor (magic 0xDEADBEEF, dtype code, length, data offset) and raw bytes."""
from __future__ import annotations

import struct

import numpy as np

_HEADER = 64
_DESCRIPTOR = 64
# BLOBFILE container dtype codes (distinct from the MIL-proto DataType enum).
FP16, INT8 = 1, 4
UINT1 = 9   # 1-bit sparse mask
UINT4 = 11  # packed 4-bit LUT indices


def fp16_bytes(a: np.ndarray) -> bytes:
  return np.ascontiguousarray(a.astype(np.float16)).tobytes()


def quantize_per_row(W: np.ndarray) -> tuple[bytes, bytes]:
  """Per-output-channel symmetric int8 quant of [OUT,IN] -> (int8 bytes, fp16 scale bytes); reconstruct as `int8 * scale[:,None]`."""
  W = W.astype(np.float32)
  scale = np.clip(np.abs(W).max(axis=1, keepdims=True) / 127.0, 1e-8, None)
  q = np.round(W / scale).clip(-127, 127).astype(np.int8)
  return np.ascontiguousarray(q).tobytes(), fp16_bytes(scale[:, 0])


_LUT4_LEVELS = 16
_LUT4_ITERS = 20
_LUT4_MAX_BINS = 1 << 20
_LUT4_CHUNK = 1 << 22
_LUT4_RANGE_Q = 1e-6   # trim this mass off each tail when picking the training range


def _lut4_pack(idx: np.ndarray, shape: tuple[int, ...]) -> bytes:
  """Two 4-bit indices per byte (low nibble first); an odd row length is padded with 0."""
  if idx.shape[1] % 2:
    idx = np.pad(idx, ((0, 0), (0, 1)))
  packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)
  return np.ascontiguousarray(packed).tobytes()


def _lut4_histogram(flat: np.ndarray, lo: float, hi: float, bins: int) -> tuple[np.ndarray, np.ndarray]:
  """(bin centers, counts) of `flat` inside [lo, hi] - one chunked pass, no [N, L] temporary."""
  scale = bins / (hi - lo)
  counts = np.zeros(bins, dtype=np.float64)
  for start in range(0, flat.size, _LUT4_CHUNK):
    chunk = flat[start:start + _LUT4_CHUNK]
    b = np.rint((chunk - lo) * scale).astype(np.int64)
    keep = (b >= 0) & (b < bins)     # values outside the training range fall to the end centroids
    counts += np.bincount(b[keep], minlength=bins)
  centers = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
  return centers, counts


def _lut4_centroids(centers: np.ndarray, counts: np.ndarray) -> np.ndarray:
  """Weighted 1-D Lloyd over the histogram -> 16 centroids (deterministic, no RNG).

  A 1-D codebook needs only each value's nearest level, which for sorted levels is a
  midpoint lookup, so neither the assignment nor the update needs an [N, L] distance
  matrix - it is the same k-means objective the element-wise form computes, at O(N).
  """
  nz = counts > 0
  c, w = centers[nz], counts[nz]
  cum = np.cumsum(w)
  edges = c[np.searchsorted(cum, np.linspace(0.0, cum[-1], _LUT4_LEVELS + 1))]
  centroids = (edges[:-1] + edges[1:]) / 2.0
  for _ in range(_LUT4_ITERS):
    idx = np.searchsorted((centroids[:-1] + centroids[1:]) / 2.0, c)
    num = np.bincount(idx, weights=w, minlength=_LUT4_LEVELS)
    den = np.bincount(idx, weights=w * c, minlength=_LUT4_LEVELS)
    new = np.where(num > 0, den / np.maximum(num, 1e-30), centroids)   # an empty cluster keeps its level
    if np.array_equal(new, centroids): break
    centroids = new
  return centroids.astype(np.float32)


def palettize_lut4(W: np.ndarray) -> tuple[bytes, bytes]:
  """Per-tensor 4-bit LUT palettization of [OUT,IN] -> (packed-index bytes, fp16 codebook bytes). 16 centroids via deterministic weighted Lloyd over a value histogram (the range is a quantile of a fixed subsample, so one outlier cannot coarsen the bulk); the index assignment is exact on the values (sorted-level midpoints). Indices are packed two-per-byte (low nibble first). Reconstruction `codebook[index]` matches constexpr_lut_to_dense."""
  flat = W.reshape(-1).astype(np.float32)
  step = max(1, flat.size // (1 << 20))
  lo, hi = np.quantile(flat[::step], [_LUT4_RANGE_Q, 1.0 - _LUT4_RANGE_Q]).tolist()
  out = W.shape[0]
  if not hi > lo:                                        # constant tensor: one distinct value
    return _lut4_pack(np.zeros((out, flat.size // out), dtype=np.uint8), W.shape), fp16_bytes(np.full(_LUT4_LEVELS, lo))
  bins = int(min(_LUT4_MAX_BINS, max(1 << 12, flat.size)))
  centers, counts = _lut4_histogram(flat, lo, hi, bins)
  centroids = _lut4_centroids(centers, counts)
  mid = (centroids[:-1] + centroids[1:]) / 2.0
  idx = np.empty(flat.size, dtype=np.uint8)
  for start in range(0, flat.size, _LUT4_CHUNK):
    idx[start:start + _LUT4_CHUNK] = np.searchsorted(mid, flat[start:start + _LUT4_CHUNK])
  return _lut4_pack(idx.reshape(out, -1), W.shape), fp16_bytes(centroids)


def quantize_blockwise(W: np.ndarray, block_size: int = 32) -> tuple[bytes, bytes, int]:
  """Per-block symmetric int8 quant of [OUT,IN] -> (int8 data, fp16 scale [OUT,nblocks], nblocks). IN splits into contiguous `block_size`-column blocks; `block_size` clamps to the largest divisor of IN. Reconstructs as `data.reshape(OUT,nblocks,bs)*scale[:,:,None]`."""
  W = W.astype(np.float32)
  OUT, IN = W.shape
  bs = int(block_size)
  while bs > 1 and IN % bs: bs -= 1
  nblocks = IN // bs
  Wb = W.reshape(OUT, nblocks, bs)
  scale = np.clip(np.abs(Wb).max(axis=2) / 127.0, 1e-8, None)        # [OUT, nblocks]
  q = np.round(Wb / scale[:, :, None]).clip(-127, 127).astype(np.int8).reshape(OUT, IN)
  return np.ascontiguousarray(q).tobytes(), fp16_bytes(scale), nblocks


def sparsify(W: np.ndarray) -> tuple[bytes, bytes]:
  """Bitmask sparse encoding of [OUT,IN] -> (packed 1-bit mask, fp16 nonzero values), row-major (mask bit 1 = keep, LSB-first). Matches constexpr_sparse_to_dense; lossless but for fp16 rounding."""
  flat = W.reshape(-1)
  nz = flat != 0.0
  mask = np.packbits(nz.astype(np.uint8), bitorder="little")
  return np.ascontiguousarray(mask).tobytes(), fp16_bytes(flat[nz])


class BlobWriter:
  """Accumulates weight payloads; `add` returns each blob's descriptor offset (the MIL `BLOBFILE(offset=...)` value)."""

  def __init__(self) -> None:
    self._items: list[tuple[bytes, int]] = []

  def add(self, payload: bytes, code: int) -> int:
    offset = _HEADER + sum(_DESCRIPTOR + len(p) for p, _ in self._items)
    self._items.append((payload, code))
    return offset

  def build(self) -> bytes:
    parts = [struct.pack("<II", len(self._items), 2) + b"\0" * 56]
    cursor = _HEADER
    for payload, code in self._items:
      data_offset = cursor + _DESCRIPTOR
      parts.append(struct.pack("<IIQQ", 0xDEADBEEF, code, len(payload), data_offset) + b"\0" * 40)
      parts.append(payload)
      cursor = data_offset + len(payload)
    return b"".join(parts)
