# AiterBackend Debug Status

> ⚠️ **SUPERSEDED** — corrected: the claim that the opcode swap and vgpr_count are correct for all categories.
>
> The `D3E1 -> D3CD` substitution (bf16 MFMA -> f16 MFMA) is **wrong**: gfx90a
> has BF16 MFMA (`v_mfma_f32_16x16x16bf16_1k`, opcode **D3E7**). The
> `vgpr_count` rewrite is unnecessary -- gfx90a's VGPR/AGPR file is unified.
> Scripts implementing that patch are quarantined in `configs/attic/`.
>
> Current status: [`../docs/19-aiter-operator-port-matrix.md`](../docs/19-aiter-operator-port-matrix.md).
> Kept as a record of the investigation, including the dead ends.

## What's Proven Working

1. **pa_fwd_asm kernel executes correctly on gfx90a** — standalone test with
   directly-allocated KV cache in correct physical layout produces valid
   non-zero output (99.95% non-zero elements)

2. **Binary patches are correct for ALL categories** — zero illegal instructions,
   correct opcode swap, correct vgpr_count

## What's Blocking AiterBackend Integration

The pa_fwd_asm kernel faults when called through ATOM's full pipeline.
Root cause: KV cache data layout mismatch between write and read paths.

### The Layout Issue

The pa kernel expects K/V in a shuffled physical layout:
- K: `[blocks, kv_heads, head_dim//8, block_size, 8]`
- V: `[blocks, kv_heads, block_size//8, head_dim, 8]`

ATOM allocates KV cache as `[2, layers, blocks, block_size, kv_heads, head_dim]`
(standard vLLM layout) and uses `.view()` to reshape. The `.view()` changes
strides but doesn't rearrange physical data.

### What Was Tried

1. **Per-layer allocation** with correct physical layout: Fixes the allocation
   but `reshape_and_cache` (cache write during prefill) still writes data
   in standard layout order, scrambling the data for the shuffled cache.

2. **Deferred allocation** (return None from allocate_kv_cache_tensors): Same
   issue — the write path doesn't match the read path.

### Root Cause

`reshape_and_cache` (or its underlying kernel) writes K/V data assuming
standard [block_size, kv_heads, head_dim] memory order. The pa_fwd_asm
kernel reads assuming shuffled [kv_heads, head_dim//x, block_size, x] order.
When both use `.view()` on the same underlying data, the strides differ,
causing read/write mismatch.

### Fix Needed

Either:
1. Modify `reshape_and_cache` to write in shuffled order
2. Or allocate KV cache in shuffled order AND modify the write path
3. Or use a permute/transpose before the view to physically rearrange data

## Recommendation

The Triton backend (`ATOM_USE_UNIFIED_ATTN=1 --block-size 64`) works
end-to-end for inference. The ASM kernel path requires deeper investigation
into the `reshape_and_cache` write path — specifically whether the GPU
kernel uses hardcoded strides or tensor strides for writing.
