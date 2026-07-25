# Change Log: rdna2 Occupancy Patch Applied

## Date: 2026-07-25

## Summary
Applied the rdna2-patch.diff from llama.cpp upstream (commit 32aead0d) to the TurboQuant build. This patch replaces the `GGML_ASSERT(max_blocks_per_sm > 0)` assertion in `fattn-common.cuh` with a graceful fallback to `max_blocks_per_sm = 1`.

## The Fix
**File**: `ggml/src/ggml-cuda/fattn-common.cuh`
**Line**: ~1483

Before:
```cpp
GGML_ASSERT(max_blocks_per_sm > 0);
```

After:
```cpp
if (max_blocks_per_sm <= 0) {
    GGML_LOG_WARN("cudaOccupancyMaxActiveBlocksPerMultiprocessor returned %d, falling back to 1\n", max_blocks_per_sm);
    max_blocks_per_sm = 1;
}
```

## Why This Matters
On AMD GPUs (gfx90a/CDNA2 and RDNA2), `hipOccupancyMaxActiveBlocksPerMultiprocessor` can return 0 for certain FA kernel configurations. The original assertion crashes the process. This patch allows the FA kernel to run with fallback occupancy.

## Test Results
- Patch applied cleanly to TurboQuant build
- Rebuild succeeded (incremental, HIP)
- **q4_0/q4_0 FA-on**: Loads successfully, output correct ("2+2 is equal to")
- Prefill: 124 tok/s, Decode: 28.8 tok/s
- Note: occupancy fallback was NOT triggered for this config (value was already > 0)
- The patch is a safety net for edge cases (certain architectures/split configs)

## Key Finding
**q4_0/q4_0 FA-on was ALREADY working on gfx90a before the patch.** The earlier comprehensive benchmark showed 492 tok/s for this config. The patch ensures robustness but was not the blocker we initially thought. FA-on is functional but SLOWER than FA-off (the gfx90a FA fallback kernel is less optimized than the standard attention path).
