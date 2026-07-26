# AITER MLA Operations Discovery

**Date**: 2025-07-25
**Status**: Discovery — Full testing pending
**Impact**: CRITICAL for MiMo-V2.5 optimization

## Executive Summary

AITER contains purpose-built **Multi-head Latent Attention (MLA)** operations designed specifically for DeepSeek-V2/V3 and similar architectures. MiMo-V2.5 uses MLA attention, making these operations directly applicable. This discovery fundamentally changes the integration strategy.

## MLA Operations Found

### Prefill Operations

| Operation | Source File | Description |
|-----------|-------------|-------------|
| `mla_prefill_asm_fwd` | `aiter/ops/attention.py` | MLA prefill with assembly-optimized kernels |
| `mla_prefill_ps_asm_fwd` | `aiter/ops/attention.py` | MLA persistent prefill (fuses kernel launches) |

**What "persistent" means**: The persistent kernel variant (`ps_asm`) keeps the kernel resident on the GPU, eliminating kernel launch overhead for sequential prefill operations. This is especially valuable for MI210 which has higher kernel launch latency than MI300X.

### Decode Operations

| Operation | Source File | Description |
|-----------|-------------|-------------|
| `mla_decode_stage1_asm_fwd` | `aiter/ops/attention.py` | MLA decode stage 1 (metadata + routing) |
| `hk_mla_decode_fwd` | `aiter/ops/attention.py` | "Hummingbird" MLA decode (hk prefix) |

**Two-stage decode**: MLA decode is split into stages because MLA's latent structure requires a metadata computation pass before the actual attention. Stage 1 handles this metadata.

### Metadata Operations

| Operation | Description |
|-----------|-------------|
| `get_mla_metadata_v1` | Compute MLA metadata for attention |
| `get_mla_metadata_v1_no_redundant` | Optimized metadata computation (dedup) |
| `get_mla_metadata_info_v1` | Metadata info query |
| `decode_update_mla_metadata_v1` | Update metadata during decode (incremental) |
| `decode_update_mla_metadata_v1_kernel` | Low-level metadata update kernel |

### Reduction Operations

| Operation | Description |
|-----------|-------------|
| `mla_reduce_v1` | MLA output reduction |

### KV Cache Operations (MLA-Specific)

| Operation | Description |
|-----------|-------------|
| `concat_and_cache_mla` | Concatenate and store MLA KV cache |
| `fused_qk_rope_concat_and_cache_mla` | **SUPER-FUSED**: Q normalization + K normalization + RoPE + cache concat in ONE kernel |

**The fused operation is remarkable**: `fused_qk_rope_concat_and_cache_mla` combines 4 operations into a single GPU kernel:
1. Q normalization
2. K normalization
3. RoPE (Rotary Position Embedding)
4. KV cache concatenation

In the current llama.cpp implementation, these are separate operations with separate kernel launches. Fusing them eliminates 3 kernel launches per layer.

---

## Sparse MLA

In addition to standard MLA, AITER has:

| File | Description |
|------|-------------|
| `unified_attention_sparse_mla.py` | Unified Sparse MLA attention |
| `unified_attention.py` | Unified attention interface (supports MLA) |

Sparse MLA is relevant for models with sparse attention patterns within the MLA framework. MiMo-V2.5 may or may not use sparse MLA — need to check the model config.

---

## MLA Decode with RoPE

File: `aiter/ops/triton/attention/mla_decode_rope.py`

This is a specialized MLA decode variant that fuses RoPE into the decode path. In standard attention, RoPE is applied before KV cache storage. In MLA, the latent structure means RoPE can be applied during decode, reducing cache size.

This is exactly what MiMo-V2.5 does — it stores the compressed latent and applies RoPE during attention computation.

---

## How MLA Works in MiMo-V2.5

For context, here's why these MLA operations matter:

### Standard MHA:
```
Cache stores: K[Layers × Heads × Dim) + V(Layers × Heads × Dim)
For 4096 tokens: 4096 × 32 heads × 128 dim × 2 (K+V) × 2 bytes = 64MB per layer
```

### MLA (MiMo/DeepSeek):
```
Cache stores: latent(Layers × 1 × compressed_dim)
For 4096 tokens: 4096 × 1 × 1024 × 2 bytes = 8MB per layer (8x reduction!)

During attention:
1. Decode latent → K, V (on the fly)
2. Apply RoPE to K
3. Compute attention
4. Reduce output

This is exactly what mla_decode_stage1_asm_fwd + mla_reduce_v1 do.
```

### The MLA Pipeline:
```
Prefill:
  q,k,v → q_norm, k_norm → rope → concat_and_cache_mla → mla_prefill_asm_fwd

Decode:
  latent_cache → decode_update_mla_metadata_v1 → mla_decode_stage1_asm_fwd → mla_reduce_v1 → output
```

AITER's `fused_qk_rope_concat_and_cache_mla` collapses the first three prefill steps into one kernel.

---

## Implications for MiMo Integration

### Current Architecture (llama.cpp):
```
llama.cpp handles everything:
  - Attention (flash_attn or SDPA)
  - MLA decompression
  - KV cache management
  - RoPE
  - RMSNorm
  - MoE routing
```

### Potential Architecture (AITER offload):
```
llama.cpp handles:
  - Tensor operations
  - Weight loading
  - Tokenization
  
AITER handles (offloaded via Python sidecar):
  - MLA prefill: mla_prefill_asm_fwd
  - MLA decode: mla_decode_stage1_asm_fwd
  - MLA KV cache: concat_and_cache_mla
  - Fused QK norm + RoPE + cache: fused_qk_rope_concat_and_cache_mla
  - MoE routing: ck_moe_stage1/2 (if needed)
  - RMSNorm: rmsnorm2d_fwd_with_add_ck (if needed)
```

This would offload the most compute-intensive operations to purpose-built kernels while keeping llama.cpp as the orchestration layer.

---

## Testing Plan

Each MLA operation needs:
1. **Correctness verification**: Compare output against PyTorch SDPA reference
2. **Performance benchmark**: Measure tok/s vs current llama.cpp implementation
3. **Integration feasibility**: Determine API calling conventions, tensor formats

### Priority Order:
1. `flash_attn_func` — ✅ DONE (2.09M tok/s, diff=0.002)
2. `mla_prefill_asm_fwd` — HIGHEST (MiMo prefill)
3. `mla_decode_stage1_asm_fwd` — HIGHEST (MiMo decode)
4. `fused_qk_rope_concat_and_cache_mla` — HIGH (fused prefill prep)
5. `concat_and_cache_mla` — HIGH (KV cache)
6. `mla_reduce_v1` — MEDIUM (output reduction)
7. `hk_mla_decode_fwd` — MEDIUM (alternative decode)
8. `unified_attention_sparse_mla` — LOW (investigate if MiMo uses sparse)

---

## Source File Inventory

All MLA-related Triton kernel sources in AITER:

```
aiter/ops/triton/attention/
├── mla_decode.py              # MLA decode Triton implementation
├── mla_decode_rope.py         # MLA decode with fused RoPE
├── unified_attention.py       # Unified attention (supports MLA mode)
└── unified_attention_sparse_mla.py  # Sparse MLA

aiter/ops/triton/_triton_kernels/attention/
├── mla_decode_rope.py         # Compiled MLA decode RoPE kernel
├── unified_attention.py       # Compiled unified attention
└── unified_attention_sparse_mla.py  # Compiled sparse MLA

aiter/ops/attention.py         # Python API (mla_prefill_asm_fwd, etc.)
```

---

## Next Steps

1. **Benchmark MLA prefill** (`mla_prefill_asm_fwd`) against standard FA on MiMo-shaped tensors
2. **Benchmark MLA decode** (`mla_decode_stage1_asm_fwd`) for single-token decode latency
3. **Test fused operations** (`fused_qk_rope_concat_and_cache_mla`) for correctness and speedup
4. **Investigate unified_attention** as a drop-in replacement for the current attention backend
5. **Design Python sidecar architecture** for llama.cpp ↔ AITER communication

This discovery represents the most promising path to significant performance improvement for MiMo on MI210.
