# What aneforge's compile cache actually buys (and what it costs)

`af.compile()` writes a content-addressed program directory and passes a cache
location to Apple's e5rt compiler. The docstring says identical graph+weights is
"written and compiled ONCE, reused on every later compile". Measured on an M4
Pro, the first half holds and the second does not.

## The cache is not reused

Same FLUX.2 single-stream block program (~150 ops, 245 MB of weights), compiled
repeatedly:

| measurement | time |
| --- | --- |
| three fresh processes, same program | 2.49 / 2.45 / 2.44 s |
| same process, compiled twice | 2.44 / 2.44 s |
| cache wiped, then compiled twice | 2.39 (cold) / 2.45 (warm) s |

The last row is the clean A/B: with the program freshly in the cache, the next
compile is not faster. The bundle cache directory
(`cache/com.apple.e5rt.e5bundlecache/<os-build>/<hash>/<hash>.bundle`) is written
- 236 MB of it - but compiling again does not appear to read it back.

It is not aneforge's optimizer either:

| variant | time |
| --- | --- |
| `opt="routes"` (default) | 2.48 s |
| `opt=0` (route search skipped), twice | 2.49 / 2.49 s |
| emission only (`_lower_fused_to_dir`: MIL + weights + hash) | 0.15 s |

So ~2.3 s of the 2.4 s is inside `E5RT.compile(...)`, which calls
`ane_e5rt_program_compile` -> `e5rt_e5_compiler_config_options_set_cache_bundle_location`
(the documented Apple hook) and the ANE compiler.

Cost scales with graph size, not just weight bytes - a single matmul with 170 MB
of weights compiles in 0.44 s, while the 150-op block with 245 MB takes 2.39 s:

| program | weights | compile |
| --- | --- | --- |
| 1 matmul [512,512] | 0.5 MB | 0.06 s |
| 1 matmul [3072,3072] | 18.9 MB | 0.07 s |
| 1 matmul [9216,3072] | 56.6 MB | 0.14 s |
| 1 matmul [27648,3072] | 169.9 MB | 0.44 s |
| FLUX.2 single-stream block | 245 MB | 2.39 s |

## What it does buy

Re-emission. On a cache hit `_emit_program_dir` returns early, so the MIL text
and the weight blob are not rebuilt (0.15 s per FLUX.2 block, ~4 s per full
model build). The files also have to exist on disk regardless: `E5RT.compile`
takes a *path*, not an in-memory program.

Against that, the disk cost is the program's full weights: 236 MB for an fp16
FLUX.2 block (119 MB with int8 streaming), and one full 25-block build is
**7.35 GB**. There is no eviction; the cache only grows.

Net: for this workload the cache trades ~7.5 GB of disk for ~4 s of a ~115 s
build. Deleting it is close to free.

## Cache locations

Four separate paths, with inconsistently named overrides, none of them
documented outside the code:

| purpose | default | override |
| --- | --- | --- |
| program / compile cache | `~/Models/.aneforge-cache` | `ANEFORGE_CACHE_DIR` |
| measurement (autotune) cache | `<repo>/.aneforge_cache`, else `~/.cache/aneforge` | `ANEFORGE_CACHE_DIR` |
| dylib build cache | `~/.cache/aneforge/<version>/` | `ANEFORGE_CACHE` (no `_DIR`) |
| e5rt default cache | `~/.cache/aneforge/e5rt` | none |

`ANEFORGE_CACHE_DIR` covers the first two, which have different defaults;
`ANEFORGE_CACHE` and `ANEFORGE_CACHE_DIR` are distinct variables.

```sh
export ANEFORGE_CACHE_DIR=/Volumes/fxdisk/ane-cache   # relocate
rm -rf ~/Models/.aneforge-cache                       # or drop it
```

## Why this matters for a port

A fixed model at a fixed resolution compiles each program once per *process*.
So a long-lived server pays the build once at startup and then serves; a CLI
invocation pays it every run - ~115 s for FLUX.2-klein-4B's 26 programs, of
which only ~4 s is emission that the cache removes.

## Open question

Is the e5rt bundle cache supposed to make the next compile faster, and aneforge
is missing a step - or does the ANE compiler re-run its pipeline every call by
design? The A/B above says the cache does not help today; which side owns that
is not determinable from outside.
