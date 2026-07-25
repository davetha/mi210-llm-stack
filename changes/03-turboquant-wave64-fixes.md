# Change 03 — TurboQuant Wave64 Fixes

Four categories of fixes to the CUDA/HIP TurboQuant kernels for the 64-lane
wavefronts on gfx90a. **Genuine fixes** (each changed the corruption pattern),
but the GPU path remains partially corrupted — the wave64 gap is pervasive.

See [`docs/02-turboquant-analysis.md`](../docs/02-turboquant-analysis.md) for the full root-cause analysis.

## Fix 1 — Ballot macro (root cause) · `vendors/hip.h:59`

```diff
- #define __ballot_sync(mask, var) ((uint32_t)__ballot(var))
+ #define __ballot_sync(mask, var) ((uint64_t)__ballot(var))
```

On gfx90a (wave64), `__ballot()` returns a **64-bit** value covering all 64
lanes. The original macro truncated to `uint32_t`, silently dropping lanes
32–63. This affected **ALL** `__ballot_sync` calls globally. The single biggest
fix — improved output from total garbage → semi-coherent.

## Fix 2 — `__shfl_sync` width ×5 · `set-rows.cu`

All 5 plain `__shfl_sync(mask, var, src)` calls changed to 4-arg form with
explicit `WARP_SIZE` width:

| Line | Context | Change |
|------|---------|--------|
| 374 | turbo3 qs packing | `+ , WARP_SIZE` |
| 504 | turbo2 qs packing | `+ , WARP_SIZE` |
| 741 | turbo4 qs packing | `+ , WARP_SIZE` |
| 855 | turbo4 qs packing | `+ , WARP_SIZE` |
| 1070 | partner_nibble shuffle | `+ , WARP_SIZE` |

On AMD wave64, the default `warpSize=64` causes cross-block data contamination
when 2 blocks of 32 threads share a wavefront. Adding explicit `WARP_SIZE`
(32) constrains the shuffle to the logical 32-lane subgroup.

> Note: `__shfl_xor_sync` calls are **NOT** affected — XOR semantics preserve
> grouping regardless of width.

## Fix 3 — turbo3 signs ballot · `set-rows.cu:381`

```diff
- const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
- const int local_signs_byte = lane / 8;
+ const uint64_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
+ const int phys_lane_sr = j % warpSize;
+ const int local_signs_byte = phys_lane_sr / 8;
```

Changed `uint32_t` → `uint64_t` (works with Fix 1's macro), and `lane / 8` →
`j % warpSize / 8` to correctly map the physical wave lane to the ballot byte.

## Fix 4 — turbo2 signs ballot · `set-rows.cu:510`

Same pattern as Fix 3, applied to the TURBO2 code path:

```diff
- const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
- const int signs_byte_idx = lane / 8;
+ const uint64_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
+ const int phys_lane_sr2 = j % warpSize;
+ const int signs_byte_idx = phys_lane_sr2 / 8;
```

## Also: dkq64 fattn-mma-turbo instances (7 new files)

Created template instantiations for `head_dim=64` (mimo's dimension), copied
from the dkq128 instances with `128` → `64` substitution:

```
fattn-mma-turbo-instance-dkq64-ncols1_1-ncols2_8.cu
fattn-mma-turbo-instance-dkq64-ncols1_2-ncols2_4.cu
fattn-mma-turbo-instance-dkq64-ncols1_2-ncols2_8.cu
fattn-mma-turbo-instance-dkq64-ncols1_4-ncols2_2.cu
fattn-mma-turbo-instance-dkq64-ncols1_4-ncols2_4.cu
fattn-mma-turbo-instance-dkq64-ncols1_4-ncols2_8.cu
fattn-mma-turbo-instance-dkq64-ncols1_8-ncols2_1.cu
```

## Status: CPU correct, GPU still partially corrupted

- **CPU round-trip:** PASSED (cosine > 0.98) — algorithm is correct.
- **GPU quantization (elements 0–31):** VERIFIED CORRECT via printf.
- **GPU output:** Still semi-corrupted (real words, wrong meaning).

The remaining corruption is **not** in the quantization kernel (verified
correct). It's in an unidentified code path — likely `__reduce_add_sync`,
`warp_reduce_sum`, or MMA fragment handling when 2 blocks share a wavefront.
All code-level debugging exhausted via SSH; `rocprof` GPU tracing is needed.

## Root cause summary

`__GFX9__` macro is **NOT** defined for gfx90a (compiler defines `__gfx90a__`
→ CDNA2, but not `__GFX9__`). So `ggml_cuda_get_physical_warp_size()` returns
32, but hardware wavefronts are 64 lanes. HIP intrinsics (`__shfl`, `__ballot`)
use hardware `warpSize=64`. This mismatch causes subtle data contamination when
2 blocks of 32 threads share a 64-lane wavefront.

## Recommendation

The **Triton** implementation ([`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd)) is the recommended solution — GEMM-based WHT, zero wave64 issues. Or use the **per-layer KV types** feature ([change 01](01-per-layer-kv-types.md)) to pin turbo to CPU layers only (where it's correct).

## Patch

→ [`patches/03-turboquant-wave64-fixes.patch`](https://github.com/davetha/llama.cpp-mi210/blob/main/patches/03-turboquant-wave64-fixes.patch)
