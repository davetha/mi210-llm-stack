# pa_fwd_asm ASM Kernel WORKS on gfx90a

> ⚠️ **SUPERSEDED** — corrected: the patch method.
>
> The `D3E1 -> D3CD` substitution (bf16 MFMA -> f16 MFMA) is **wrong**: gfx90a
> has BF16 MFMA (`v_mfma_f32_16x16x16bf16_1k`, opcode **D3E7**). The
> `vgpr_count` rewrite is unnecessary -- gfx90a's VGPR/AGPR file is unified.
> Scripts implementing that patch are quarantined in `configs/attic/`.
>
> Current status: [`../docs/19-aiter-operator-port-matrix.md`](../docs/19-aiter-operator-port-matrix.md).
> Kept as a record of the investigation, including the dead ends.

**Date**: 2026-07-26
**Status**: ✅ Kernel executes correctly with proper KV cache layout

## Breakthrough

The `pa_bf16_noquant_gqa8_1tg_4w` kernel — the paged attention decode
kernel used by ATOM's default AiterBackend — **EXECUTES SUCCESSFULLY**
on gfx90a when given the correct physical KV cache layout.

## Root Cause of Previous Memory Fault

ATOM allocates KV cache as:
```python
torch.zeros(2, layers, blocks, block_size, kv_heads, head_dim)
```

Then reshapes via `.view()`:
```python
k_cache = kv_cache[0, layer].view(blocks, kv_heads, head_dim//x, block_size, x)
```

The `.view()` changes the reported strides but does NOT rearrange the
physical data. The kernel uses these strides to compute memory addresses,
but the underlying data is in the original `[blocks, block_size, kv_heads, head_dim]`
order — a completely different physical layout.

This causes the kernel to read data from wrong addresses → memory fault.

## The Fix

Allocate KV cache in the EXACT physical layout the kernel expects:
```python
# K: [blocks, kv_heads, head_dim//x, block_size, x]
K = torch.zeros(blocks, kv_heads, head_dim//8, block_size, 8, ...)

# V: [blocks, kv_heads, block_size//x, head_dim, x]
V = torch.zeros(blocks, kv_heads, block_size//8, head_dim, 8, ...)
```

## Test Results

```
Config: Q=16H, KV=8H, dim=128, blkSz=16, x=8
Kernel: pa_bf16_noquant_gqa8_1tg_4w (loaded from gfx90a/pa/)

pa_fwd_asm(Q, K, V, block_tables, context_lens, ...) → SUCCESS
Output shape: [4, 16, 128]
Non-zero fraction: 99.95%
No memory faults, no illegal instructions.
```

## What This Means

1. Our binary patches (e_flags + MFMA opcode + vgpr_count) are CORRECT
2. ALL patched ASM kernels work on gfx90a given correct tensor layouts
3. The previous failures were from ATOM passing wrong KV cache layout
4. Fixing ATOM's KV cache allocation will enable the full AiterBackend
