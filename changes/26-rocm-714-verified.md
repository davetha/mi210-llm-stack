# ROCm 7.14.0 Installed + ATOM Verified

**Date**: 2026-07-27
**Status**: ✅ ROCm 7.14.0, HSA Runtime 1.21, ATOM producing coherent text

## ROCm 7.14 Installation

Installed via runfile installer to `/opt/rocm-7.2.0/core-7.14/` (co-installed
with existing 7.2.0). Symlinked `/opt/rocm -> core-7.14`.

Key version changes:
- ROCm platform: 7.2.0 → 7.14.0
- HSA Runtime: 1.18 → 1.21
- rocminfo Runtime Version: 1.21

## reshape_and_cache Root Cause

The `reshape_and_cache` JIT kernel (`module_cache`) faults on gfx90a with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` regardless of:
- ROCm version (7.2 or 7.14)
- asm_layout (True or False)
- Source (JIT-compiled or wheel pre-compiled cp312 .so)

The kernel template `reshape_and_cache_kernel<..., vllm::Fp8KVCacheDataType, ...>`
uses vLLM headers from the base image (vLLM 0.23.0) which are incompatible
with AITER 0.1.17's expected vLLM API.

This does NOT affect the working path because `unified_attention` uses its
own cache write path (`fused_qk_rope_reshape_and_cache`) which doesn't
trigger the reshape_and_cache kernel.

## Verified Working Config

```bash
ATOM_USE_UNIFIED_ATTN=1 ATOM_LOADER_USE_THREADPOOL=0 \
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B --block-size 64 --enforce-eager --level 0
```

Performance: TTFT=0.532s, TPOT=0.031s (32 tok/s), coherent text.
