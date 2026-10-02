# Making aneforge's compile cache actually work

`af.compile()` writes a content-addressed program directory (MIL + weights) and
passes a cache location to Apple's e5rt compiler. The docstring says identical
graph+weights is "written and compiled ONCE, reused on every later compile".

Measured on an M4 Pro: the first half was true, the second was not - a repeat
compile of the identical program cost the same 2.4 s as the first. The cause was
one line in the dispatch shim.

## Root cause

`aneforge/_lib/ane_e5rt_dispatch.mm`, `compile_and_build_op()`:

```cpp
e5rt_e5_compiler_options_set_force_recompilation(options, 1);
```

The shim sets the cache bundle location (so compiled bundles are *written*) and
then tells the compiler to **force recompilation**, so a written bundle is never
read back. It dates from the initial public release with no note explaining it -
most likely a leftover from development, where recompiling every time avoids
stale artifacts while iterating.

## The fix

Gate it: reuse the cache by default, keep always-recompile behind an env var.

```cpp
{
  const char *force = getenv("ANEFORGE_FORCE_RECOMPILE");
  e5rt_e5_compiler_options_set_force_recompilation(options, (force && *force == '1') ? 1 : 0);
}
```

This is safe because the cache location is content-addressed by
`sha256(MIL + weights)[:24]` (see `_emit_program_dir`), so a hit can only be for
the same program. Rebuild with `python -m aneforge.build`.

## Verification

One FLUX.2 single-stream block, fresh process each time:

| | compile |
| --- | --- |
| cold (cache miss) | 3.26 s |
| warm (cache hit) | **0.14 s** |
| warm again | **0.15 s** |

Full 25-block transformer build (26 programs), same script:

| | build |
| --- | --- |
| first run (populating) | 86.2 s |
| second run | **9.7 s** |
| inference, both runs | 3.06 s/step (identical) |

And the cached program is not just fast but identical in output:

| | compile | output mean |
| --- | --- | --- |
| cache hit | 0.24 s | -0.01619201 |
| `ANEFORGE_FORCE_RECOMPILE=1` | 2.99 s | -0.01619201 |

Element-wise difference between the two outputs: **0.0** (bit-identical).

## What is left

~10 s of the 26-program build is now emission + hashing + loading the program
directories, down from ~115 s. The disk the cache uses (7.35 GB fp16, 3.7 GB
with int8 streaming, no eviction) now buys a 12x startup, so keeping it is the
right trade - the opposite of the conclusion before the fix.

The next section splits that ~10 s apart, and the one after it is a memory
footgun worth knowing about when compiling many programs in one process.

Reverting if needed: `git checkout aneforge/_lib/ane_e5rt_dispatch.mm` and
rebuild, or set `ANEFORGE_FORCE_RECOMPILE=1` at run time.

## What the warm build is actually spending its time on

Measured with aneforge 0.4.1.dev37 while porting the model into fluxlab, one
FLUX.2 double-stream block (245.8 MB of fp16 weights, program directory already
populated):

| call | time | what it does |
| --- | --- | --- |
| `E5RT.compile(mil, cache_dir=dir/cache, inputs, outputs)` on the existing dir | **6.4 ms** | loads the compiled program; never opens `weights.bin` |
| the same call with `cache_dir` pointed at an empty directory | **5081 ms** | a real Apple compile (so the 6.4 ms is a genuine cache hit) |
| `af.compile(graph)` for the same program | 0.3-0.7 s | rebuilds the graph, reads and dequantizes the block's weights, hashes MIL + weights to find that same dir |

Across a whole 27-program set (512x768, int8), a warm build is 18.5 s through
`af.compile` and **143 ms** loading each program directly (slowest program
41 ms; the first load in a process costs ~25 ms extra). So essentially all of
the warm build is re-deriving a content key whose value the caller could have
recorded the first time.

That is also why the direct load can skip the weights: **a compiled program is
a netlist, not a copy of the weights.** A program directory's `cache/` holds
~8 KB of e5rt bundle (largest file 2.7 KB), and the MIL refers to the weights as
`BLOBFILE(path = string("@model_path/weights.bin"), offset = ...)` - they are
streamed from `weights.bin` at dispatch. Earlier wording in this note ("loading
the bundles") implied the weights were inside the bundle; they are not. This is
also why it is safe to drop the dequantized checkpoint from host memory after
compiling, and why `weights.bin` must stay on disk for dispatch.

What aneforge could expose to make the fast path the default: `_emit_program_dir`
already knows the program directory and the returned `Model` carries the port
names/shapes (`_inputs`, `_out_name`, `_out_shape`), so a `Model.program_dir`
attribute plus a public `af.load_program(dir)` (or a `reuse_dir=` argument to
`af.compile`) would turn repeated-process startup from seconds into
milliseconds. fluxlab does it today by intercepting `_emit_program_dir` for the
directory and calling `E5RT.compile` itself; its output is bit-identical to the
`af.compile` path.

## The first execute after a load is not free

The direct load is 6 ms and never opens `weights.bin` - but the weights still
have to reach the engine, and the *first execute* of a freshly loaded program
pays for it. Measured while porting into fluxlab (512x768, int8 streaming, 27
programs, warm page cache, one process, `bench_opt/ane_step_breakdown.py`):

| program | weights.bin | first execute | steady execute | excess |
| --- | --- | --- | --- | --- |
| 5 x double block | 245.9 MB | 73.4-74.1 ms | 67.4-67.6 ms | +5.8 .. +6.6 |
| 20 x single block | 122.9 MB | 66.6-68.5 ms | 62.3-62.5 ms | +4.2 .. +6.1 |
| embed | 24.0 MB | 3.99 ms | 2.68 ms | +1.31 |
| head | 0.4 MB | 2.77 ms | 1.83 ms | +0.94 |
| total | 3.71 GB | | | **+138 ms** |

The excess is not proportional to the weight bytes: `head` carries 0.4 MB and
still pays 0.94 ms, while doubling a block's weights (123 -> 246 MB) adds less
than 1 ms. So it is mostly a fixed per-program cost (~0.9 ms, the e5rt/ANE side
mapping each program's resources) plus a weak size term - not a plain copy at
some bandwidth (~27 GB/s if it were one, which no measured host path here
reaches).

It is paid per **load cycle**, not per process: `Program.release()` followed by
a reload pays it again (a second image in the same process, after fluxlab's
stage-unload release/reload, still shows a 1.91 s first step against a 1.64 s
steady one), and a fresh process with a warm page cache shows the same 1.94 s,
so it is not disk I/O.

Across the 27-program set that is ~0.14 s; adding the program load (~0.10 s,
above) and the first host-side modulation/RoPE computation, the first denoise
step of a freshly loaded set is **~0.29 s slower** than steady state (1.93 vs
1.64 s at 512x768).

Pre-warming is not a lever: the excess is bound to the execute itself, so a
warm-up dispatch costs a full step (1.6 s) to save 0.14 s of it. The way to
amortize it is to keep the programs loaded (`FLUX2_STAGE_UNLOAD=off` in
fluxlab) when one process serves several images.

## The route optimizer memo keeps every weight alive

`compile(opt="routes")` is the default and is cost-model driven; its bookkeeping
memoizes graphs by identity (`aneforge/_optimize.py`):

```python
_TOPO_MEMO: dict[int, tuple] = {}

def _topo(out):
  hit = _TOPO_MEMO.get(id(out))
  if hit is not None and hit[0] is out: return hit[1]
  order = _raw_topo(out)
  if len(_TOPO_MEMO) >= 256: _TOPO_MEMO.clear()
  _TOPO_MEMO[id(out)] = (out, order)
  return order
```

`out` is the graph root, and every weight is a constant hanging off it, so an
entry pins that program's weights for the life of the process (until the
256-entry cap happens to trip). Compiling five double-stream blocks in one
process without clearing it leaves **2.5 GB** more resident than clearing after
each compile - measured as peak RSS, 5.68 GB vs 3.18 GB for the same five
compiles, i.e. ~0.5 GB per block, about twice the block's fp16 weights (the
transient peak of a single compile is in both numbers). Extrapolated to the
27-program FLUX.2 set that is ~7 GB of weights silently held, which cancels the
point of loading block weights only while compiling them.

`compress=` (and `opt=0`) never enter `_compile_routes`, so those paths are
immune. For the dense-fp16 path, `aneforge._optimize._TOPO_MEMO.clear()` after
each compile is enough - that is what fluxlab does. A weak reference in the
memo, or a byte budget instead of a 256-entry cap, would remove the footgun.

## Cache locations

Four separate paths, with inconsistently named overrides, none documented
outside the code:

| purpose | default | override |
| --- | --- | --- |
| program / compile cache | `~/Models/.aneforge-cache` | `ANEFORGE_CACHE_DIR` |
| measurement (autotune) cache | `<repo>/.aneforge_cache`, else `~/.cache/aneforge` | `ANEFORGE_CACHE_DIR` |
| dylib build cache | `~/.cache/aneforge/<version>/` | `ANEFORGE_CACHE` (no `_DIR`) |
| e5rt default cache | `~/.cache/aneforge/e5rt` | none |

```sh
export ANEFORGE_CACHE_DIR=/Volumes/fxdisk/ane-cache   # relocate
```

## Measured cost model, for reference

Compile time scales with graph size, not just weight bytes - one matmul with
170 MB of weights compiles in 0.44 s, while a 150-op block with 245 MB takes
2.39 s:

| program | weights | compile |
| --- | --- | --- |
| 1 matmul [512,512] | 0.5 MB | 0.06 s |
| 1 matmul [3072,3072] | 18.9 MB | 0.07 s |
| 1 matmul [9216,3072] | 56.6 MB | 0.14 s |
| 1 matmul [27648,3072] | 169.9 MB | 0.44 s |
| FLUX.2 single-stream block | 245 MB | 2.39 s |

`opt=0` (no route search) costs the same as the default `opt="routes"`, so the
optimizer is not part of this; emission alone is 0.15 s.
