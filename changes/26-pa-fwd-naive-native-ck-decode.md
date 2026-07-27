# pa_fwd_naive: Native CK Decode Working on gfx90a

**Date**: 2026-07-27
**Status**: ✅ Coherent output achieved with native JIT-compiled CK decode kernel

## Breakthrough

Discovered `pa_fwd_naive` in AITER's CK (Composable Kernel) path — a HIP C++
attention kernel that is **JIT-compiled natively** for the target GPU. Unlike
the pre-compiled `.co` ASM blobs (which are proprietary gfx942/gfx950 binaries),
`pa_fwd_naive` compiles from source at runtime, producing correct native
instructions for gfx90a.

## Root Cause of Previous Failures

The binary-patched ASM `pa_fwd_asm` kernel produced garbage output because:

1. **No BF16 MFMA on gfx90a**: gfx90a lacks `v_mfma_f32_16x16x16_bf16`. Binary
   patch to F16 MFMA (D3CD) misinterprets BF16 bit patterns.

2. **Block tables granularity mismatch**: Scheduler uses block_size=64, physical
   KV cache uses block_size=16 (ratio=4). Both `pa_fwd_asm` and `pa_fwd_naive`
   received block_tables at scheduler granularity but expected physical granularity.
   This was the hidden root cause that made ALL decode kernels produce the same
   garbage pattern.

## Solution

### 1. Use pa_fwd_naive (CK JIT) instead of pa_fwd_asm (binary-patched ASM)

Patched `attention_mha.py` `paged_attention_asm` method to dispatch to
`aiter.ops.attention.pa_fwd_naive` on gfx90a:

```python
if get_gfx() == "gfx90a":
    from aiter.ops.attention import pa_fwd_naive
    # Convert V from SHUFFLE to NHD (pa_fwd_naive expects NHD V)
    v_nhd = v_cache.permute(0, 1, 3, 2, 4).contiguous().view(nb, nkv, hd, bs)
    o = pa_fwd_naive(query=q, key_cache=k_cache, value_cache=v_nhd, ...)
    return o
```

### 2. Use --block-size 16 to align scheduler and physical blocks

```bash
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --block-size 16 \  ① aligns scheduler block_size with physical block_size
  --max-model-len 256 --max-tokens 10 \
  --enforce-eager --level 0
```

This eliminates the 4× block_tables granularity mismatch.

### 3. Option 2 patch (force reshape_and_cache)

Keeps the separate RoPE + norm + reshape_and_cache path (bypassing the
binary-patched fused kernel).

## Verification

### pa_fwd_naive correctness (standalone)

```
pa_fwd_naive vs PyTorch reference:
  Max diff:  0.003906
  Mean diff: 0.000253
  Match<0.5: 100.0%
  RESULT: pa_fwd_naive CORRECT ✅
```

### End-to-end inference

```
Prompt: "introduce yourself"
Output: "<think>\nOkay, the user asked me to introduce"

Prompt: "list all prime numbers within 100"
Output: "<think>\nOkay, so I need to list all"

Prompt: "1+2+3=?"
Output: "<think>\nOkay, so I need to solve "

Prompt: "如何在一个月内增肌10公斤"
Output: "<think>\n好的，用户问的是如何在一个月"
```

TTFT=0.609s, TPOT=0.093s, EXIT=0. Coherent English and Chinese output.

## Performance Comparison

| Path | TPOT | Output | Notes |
|------|------|--------|-------|
| Triton unified_attention (ATOM_USE_UNIFIED_ATTN=1) | 0.029s | ✅ Coherent | Production-ready |
| pa_fwd_naive CK (this work) | 0.093s | ✅ Coherent | Native CK, 3× slower |
| Binary-patched pa_fwd_asm | 0.029s | ❌ Garbage | ISA incompatibility |

pa_fwd_naive is 3× slower than Triton because:
1. "Naive" reference implementation (not optimized)
2. V cache conversion overhead (SHUFFLE→NHD per step)
3. No tuned CK configs for gfx90a

## Files Modified

- `attention_mha.py`: pa_fwd_naive dispatch + V cache conversion + block_tables expansion
- `configs/patch_option2_reshape_and_cache.py`: force reshape_and_cache on gfx90a
- `simple_inference.py`: added `--dtype` argument

## Future Optimization

1. **Pre-allocate V cache in NHD format** — eliminates per-step conversion
2. **Fix block_tables expansion for --block-size 64** — enables larger blocks
3. **Use optimized CK attention** — AITER has `attention_ck` variants beyond naive
4. **Tune CK configs for gfx90a** — add gfx90a entries to tuned_gemm configs
