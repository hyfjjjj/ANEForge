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

~10 s of the 26-program build is now emission + hashing + loading the bundles,
down from ~115 s. The disk the cache uses (7.35 GB fp16, 3.7 GB with int8
streaming, no eviction) now buys a 12x startup, so keeping it is the right
trade - the opposite of the conclusion before the fix.

Reverting if needed: `git checkout aneforge/_lib/ane_e5rt_dispatch.mm` and
rebuild, or set `ANEFORGE_FORCE_RECOMPILE=1` at run time.

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
