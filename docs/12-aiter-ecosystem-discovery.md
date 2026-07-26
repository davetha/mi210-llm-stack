# AITER Ecosystem Discovery on MI210 (gfx90a)

**Date**: 2026-07-25
**Platform**: AMD Instinct MI210, gfx90a, 64GB VRAM, ROCm 7.14
**Container**: `fa-build` (PyTorch 2.11.0+rocm7.14.0, Python 3.14)
**AITER Version**: `0.1.13.post2.dev1+gb32deb267`

## Executive Summary

AITER (AMD Inference Tuning and Extension Repository) contains **397 public functions** spanning the entire LLM inference pipeline. On MI210 (gfx90a), AITER JIT-compiles CK (Composable Kernel) templates natively for our architecture. This document is the comprehensive inventory of what AITER offers for MiMo-V2.5 (310B MoE, MLA attention, 256 routed experts).

## Architecture: AITER Stack on MI210

```
┌──────────────────────────────────────────────────────┐
│                   User Code (vLLM)                    │
├──────────────────────────────────────────────────────┤
│                        AITER                          │
│  ┌─────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐ │
│  │ MLA Ops │  │ MoE Ops  │  │Attention│ │  GEMM   │ │
│  └────┬────┘  └────┬─────┘  └───┬───┘  └────┬────┘ │
│       │            │            │            │       │
│  ┌────▼────────────▼────────────▼────────────▼────┐ │
│  │              Triton Kernels (3.7.1)             │ │
│  │    (aiter.ops.triton._triton_kernels)           │ │
│  └────────────────────┬───────────────────────────┘ │
│                       │                              │
│  ┌────────────────────▼───────────────────────────┐ │
│  │            CK Templates (JIT-compiled)          │ │
│  │  (aiter/jit/*.so — built for gfx90a on import)  │ │
│  └────────────────────┬───────────────────────────┘ │
├───────────────────────┼──────────────────────────────┤
│              HIP Runtime (ROCm 7.14)                 │
├──────────────────────────────────────────────────────┤
│              MI210 Hardware (gfx90a)                 │
└──────────────────────────────────────────────────────┘
```

## JIT Compilation on MI210

AITER JIT-compiles CK kernels on first import. On MI210, this produces:

```
/opt/python/lib/python3.14/site-packages/aiter/jit/
├── module_aiter_core.so         # Core CK operations
└── mha_fwd_fp16_nbias_mask_nlse_ndropout_nqscale.so  # Flash attention
```

These are **compiled specifically for gfx90a** and cached. First-run compilation takes ~30 seconds; subsequent imports load from cache instantly.

**Verified**: `ENABLE_CK: True` for gfx90a. CK flash attention achieves **2,086,037 tok/s per layer** (1.96ms for S=4096, D=128, H=32) with diff=0.002 vs PyTorch SDPA.

---

## Complete Operation Inventory (397 functions)

### 1. Attention Operations (22 variants!)

This is the most critical category for MiMo. AITER contains specialized attention implementations for every modern architecture:

#### MLA (Multi-head Latent Attention) — MiMo/DeepSeek Style

| Function | File | Description |
|----------|------|-------------|
| `mla_prefill_asm_fwd` | `attention.py` | MLA prefill with ASM (assembly-optimized) kernels |
| `mla_prefill_ps_asm_fwd` | `attention.py` | MLA persistent prefill (reduces kernel launch overhead) |
| `mla_decode_stage1_asm_fwd` | `attention.py` | MLA decode stage 1 (metadata computation) |
| `hk_mla_decode_fwd` | `attention.py` | Hummingbird MLA decode (hk = Hummingbird Kernels) |
| `mla_reduce_v1` | `attention.py` | MLA reduction operation |
| `concat_and_cache_mla` | `attention.py` | MLA KV cache management |
| `fused_qk_rope_concat_and_cache_mla` | `attention.py` | **FUSED**: QK norm + RoPE + cache MLA (single kernel!) |
| `get_mla_metadata_v1` | `attention.py` | MLA metadata computation |
| `get_mla_metadata_v1_no_redundant` | `attention.py` | Optimized MLA metadata |
| `decode_update_mla_metadata_v1` | `attention.py` | MLA decode metadata update |
| `get_mla_metadata_info_v1` | `attention.py` | MLA metadata info |

**Triton kernel sources** for MLA:
- `aiter/ops/triton/attention/mla_decode.py`
- `aiter/ops/triton/attention/mla_decode_rope.py` (MLA decode with RoPE)
- `aiter/ops/triton/_triton_kernels/attention/unified_attention_sparse_mla.py` (Sparse MLA!)

#### Standard Flash Attention (CK-backed)

| Function | Description |
|----------|-------------|
| `flash_attn_func` | Standard FA2 forward (CK backend) — **VERIFIED 2.09M tok/s** |
| `flash_attn_varlen_func` | Variable-length FA (for batched prefill) |
| `mha_fwd` | Low-level MHA forward |
| `mha_bwd` | Low-level MHA backward |
| `mha_varlen_fwd` | Variable-length MHA forward |
| `mha_varlen_bwd` | Variable-length MHA backward |
| `mha_batch_prefill` | Batched prefill MHA |
| `FlashAttnFunc` | Autograd-aware FA wrapper |
| `FlashAttnVarlenFunc` | Autograd-aware varlen FA wrapper |

#### FA v3 (Flash Attention 3)

| Function | Description |
|----------|-------------|
| `fmha_v3_fwd` | FA v3 forward |
| `fmha_v3_bwd` | FA v3 backward |
| `fmha_v3_varlen_fwd` | FA v3 variable-length forward |
| `fmha_v3_varlen_bwd` | FA v3 variable-length backward |
| `can_impl_fmha_v3_bwd` | Check if FA v3 backward is supported on this GPU |
| `flash_attn_fp8_pertensor_func` | FP8 flash attention (per-tensor quant) |
| `flash_attn_varlen_fp8_pertensor_func` | FP8 variable-length flash attention |

**Triton kernel sources** for FA v3:
- `aiter/ops/triton/attention/fav3_sage_attention.py` (FA v3 with SAGE)
- `aiter/ops/triton/attention/fav3_sage_attention_mxfp4_wrapper.py` (MX FP4 wrapper)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention_mxfp4.py`

#### Paged Attention

| Function | Description |
|----------|-------------|
| `pa_fwd_asm` | Paged attention forward (ASM) |
| `pa_fwd_naive` | Paged attention forward (naive) |
| `pa_persistent_fwd` | Persistent paged attention |
| `pa_ps_fwd_asm` | Persistent speculative paged attention |
| `pa_reduce_v1` | Paged attention reduction |
| `pa_decode_gluon` | Paged attention decode (Gluon framework) |
| `paged_attention_v1` | Paged attention v1 |
| `paged_attention_rocm` | ROCm paged attention |
| `paged_attention_ragged` | Ragged paged attention |

**Triton kernel sources** for paged attention:
- `aiter/ops/triton/attention/pa_prefill.py`
- `aiter/ops/triton/attention/pa_decode.py`
- `aiter/ops/triton/attention/chunked_pa_prefill.py` (chunked variant)
- `aiter/ops/triton/attention/pa_mqa_logits.py` (MQA logits)

#### Specialized Attention

| Function | Description |
|----------|-------------|
| — (lean_atten.py) | Lean Attention |
| — (lean_atten_paged.py) | Lean Paged Attention |
| — (pod_attention.py) | POD Attention |
| — (hstu_attention.py) | HSTU Attention (recommendation systems) |
| — (unified_attention.py) | Unified Attention Interface |
| — (unified_attention_sparse_mla.py) | Unified Sparse MLA |
| — (extend_attention.py) | Extend Attention (context extension) |
| — (prefill_attention.py) | Prefill Attention |
| — (fp8_mqa_logits.py) | FP8 MQA Logits |
| — (mha_fused_bwd.py) | MHA Fused Backward |
| — (mha_onekernel_bwd.py) | MHA Single-Kernel Backward |

---

### 2. MoE Operations (MiMo has 256 experts!)

MiMo-V2.5 uses top-8 routing across 256 experts. AITER has optimized MoE kernels:

| Function | Description |
|----------|-------------|
| `fmoe_g1u1` | Fused MoE G1U1 (gate+up in one, down in another) |
| `fmoe_g1u1_a16` | Fused MoE G1U1 (A16 precision) |
| `fmoe_g1u1_tkw1` | Fused MoE G1U1 (TK W1 variant) |
| `fmoe_fp8_blockscale_g1u1` | FP8 block-scale MoE |
| `fmoe_int8_g1u0` | INT8 MoE G1U0 |
| `fmoe_int8_g1u0_a16` | INT8 MoE G1U0 (A16) |
| `fmoe` | Generic fused MoE |
| `ck_moe_stage1` | CK MoE stage 1 (routing/dispatch) |
| `ck_moe_stage1_fwd` | CK MoE stage 1 forward |
| `ck_moe_stage2` | CK MoE stage 2 (combine) |
| `ck_moe_stage2_fwd` | CK MoE stage 2 forward |
| `moe_cktile2stages_gemm1` | CK-tile 2-stage MoE GEMM 1 |
| `moe_cktile2stages_gemm1_ck` | CK-tile 2-stage MoE GEMM 1 (CK) |
| `moe_cktile2stages_gemm2` | CK-tile 2-stage MoE GEMM 2 |
| `moe_cktile2stages_gemm2_ck` | CK-tile 2-stage MoE GEMM 2 (CK) |
| `moe_stage1_g1u1` | MoE stage 1 G1U1 |
| `moe_sorting_fwd` | MoE sorting forward |
| `moe_sorting_opus_fwd` | MoE sorting OPUS forward |
| `moe_fused_gate` | MoE fused gate (routing + topk fused) |
| `moe_align_block_size` | MoE block alignment |
| `moe_sum` | MoE output summation |
| `moe_smooth_per_token_scaled_quant` | MoE smooth per-token quant |
| `moe_smooth_per_token_scaled_quant_v1` | MoE smooth per-token quant v1 |
| `moe_smooth_per_token_scaled_quant_v2` | MoE smooth per-token quant v2 |
| `moe_smoothquant_fwd` | MoE SmoothQuant forward |
| `fused_dynamic_mxfp4_quant_moe_sort` | Fused MXFP4 quant + MoE sort |
| `fused_dynamic_mxfp4_quant_moe_sort_hip` | Fused MXFP4 quant + MoE sort (HIP) |
| `mxfp4_moe_sort_fwd` | MXFP4 MoE sort forward |
| `mxfp4_moe_sort_hip` | MXFP4 MoE sort (HIP) |

**Triton kernel sources**:
- `aiter/ops/triton/moe/` — Full MoE Triton implementation
- `aiter/ops/triton/_triton_kernels/moe/` — Compiled MoE kernels

---

### 3. GEMM Operations

Extensive GEMM library supporting all precision levels:

| Category | Variants |
|----------|----------|
| A8W8 (INT8) | `gemm_a8w8`, `gemm_a8w8_CK`, `gemm_a8w8_asm`, `gemm_a8w8_ASM`, `gemm_a8w8_tune` |
| A8W8 BlockScale | `gemm_a8w8_blockscale`, `gemm_a8w8_blockscale_ck`, `gemm_a8w8_blockscale_asm`, `gemm_a8w8_blockscale_bpreshuffle`, many more |
| A4W4 (INT4) | `gemm_a4w4`, `gemm_a4w4_asm`, `gemm_a4w4_blockscale` |
| A16W16 | `gemm_a16w16_asm` |
| BF16 Batched | `batched_gemm_bf16`, `batched_gemm_bf16_CK` |
| FlatMM | `flatmm_a8w8_blockscale_ASM`, `flatmm_a8w8_blockscale_asm` |
| DeepGEMM | `deepgemm`, `deepgemm_ck` |
| HipBLASLt | `hipb_mm`, `hipb_findallsols` |
| RocBLAS | `rocb_mm`, `rocb_findallsols` |
| FlyDSL | `gemm_a8w8_bpreshuffle_flydsl`, `is_flydsl_available` |

---

### 4. Normalization Operations

| Function | Description |
|----------|-------------|
| `rms_norm` / `rmsnorm` | RMS Normalization |
| `rms_norm_cu` | RMS Norm (CUDA compat) |
| `rmsnorm2d_fwd` | 2D RMS Norm |
| `rmsnorm2d_fwd_ck` | 2D RMS Norm (CK) |
| `rmsnorm2d_fwd_with_add` | RMS Norm + residual add (fused) |
| `rmsnorm2d_fwd_with_add_ck` | Same, CK backend |
| `rmsnorm2d_fwd_with_add_dynamicquant` | RMS Norm + add + dynamic quant |
| `rmsnorm2d_fwd_with_add_smoothquant` | RMS Norm + add + SmoothQuant |
| `rmsnorm2d_fwd_with_dynamicquant` | RMS Norm + dynamic quant |
| `rmsnorm2d_fwd_with_smoothquant` | RMS Norm + SmoothQuant |
| `add_rmsnorm` | Fused add + RMS norm |
| `add_rmsnorm_quant` | Fused add + RMS norm + quant |
| `rmsnorm_quant` | RMS norm + quant |
| `fused_add_rms_norm_cu` | Fused add + RMS norm (CUDA) |
| `fused_allreduce_rmsnorm` | Fused all-reduce + RMS norm (TP) |
| `fused_allreduce_rmsnorm_quant` | Fused all-reduce + RMS norm + quant |
| `layer_norm` | Layer norm |
| `layernorm2d_fwd` | 2D layer norm |
| `layernorm2d_fwd_with_add` | Layer norm + add |
| `layernorm2d_fwd_with_add_smoothquant` | Layer norm + add + SmoothQuant |
| `layernorm2d_fwd_with_smoothquant` | Layer norm + SmoothQuant |
| `GroupNorm` | Group normalization |

---

### 5. RoPE (Rotary Position Embedding)

| Function | Description |
|----------|-------------|
| `rope_fwd` / `rope_bwd` | Standard RoPE |
| `rope_fwd_inplace` | In-place RoPE |
| `rope_2d_fwd` / `rope_2d_bwd` | 2D RoPE |
| `rope_2c_fwd` / `rope_2c_bwd` | 2-channel RoPE |
| `rope_cached_fwd` / `rope_cached_bwd` | Cached RoPE |
| `rope_cached_2c_fwd` | Cached 2-channel RoPE |
| `rope_cached_positions_fwd` | Cached positions RoPE |
| `rope_cached_positions_offsets_fwd` | Cached positions+offsets RoPE |
| `rope_thd_fwd` / `rope_thd_bwd` | Thread-level RoPE |
| `batched_rotary_embedding` | Batched rotary embedding |
| `rotary_embedding_fwd` | Rotary embedding forward |
| `RoPE` / `RoPE2D` / `RoPECached` / `RoPETHD` | Class-based RoPE interfaces |

---

### 6. TopK Operations (Expert Routing)

| Function | Description |
|----------|-------------|
| `grouped_topk` | Grouped TopK (for MoE routing) |
| `grouped_topk_torch` | Grouped TopK (PyTorch) |
| `biased_grouped_topk` | Biased Grouped TopK |
| `biased_grouped_topk_hip` | Biased Grouped TopK (HIP) |
| `biased_grouped_topk_torch` | Biased Grouped TopK (PyTorch) |
| `topk_softmax` | TopK with softmax |
| `topk_softmax_asm` | TopK with softmax (ASM) |
| `topk_sigmoid` | TopK with sigmoid |
| `topk_plain` | Plain TopK |
| `top_k_per_row_decode` | Per-row TopK (decode) |
| `top_k_per_row_decode_fast` | Fast per-row TopK (decode) |
| `top_k_per_row_prefill` | Per-row TopK (prefill) |
| `top_k_per_row_prefill_fast` | Fast per-row TopK (prefill) |

---

### 7. Quantization Operations

| Function | Description |
|----------|-------------|
| `dynamic_per_tensor_quant` | Dynamic per-tensor quant |
| `dynamic_per_token_scaled_quant` | Dynamic per-token scaled quant |
| `dynamic_per_group_scaled_quant_fp4` | Dynamic per-group FP4 quant |
| `per_tensor_quant` / `per_tensor_quant_hip` / `per_tensor_quant_triton` | Per-tensor quant |
| `per_token_quant_hip` / `per_token_quant_triton` | Per-token quant |
| `per_group_quant_hip` | Per-group quant |
| `pertoken_quant` | Per-token quant (unified) |
| `per_1x32_f4_quant` / `per_1x32_f4_quant_hip` / `per_1x32_f4_quant_triton` | 1×32 FP4 quant |
| `per_1x32_i4_quant` | 1×32 INT4 quant |
| `per_1x32_f8_scale_f8_quant` | 1×32 FP8 quant |
| `static_per_tensor_quant` | Static per-tensor quant |
| `smooth_per_token_scaled_quant` | Smooth per-token quant |
| `smoothquant_fwd` | SmoothQuant forward |
| `moe_smooth_per_token_scaled_quant` | MoE per-token quant |

---

### 8. Communication (Tensor Parallel)

| Function | Description |
|----------|-------------|
| `all_reduce` | All-reduce |
| `all_gather_reg` / `all_gather_unreg` | All-gather (registered/unregistered) |
| `reduce_scatter` | Reduce-scatter |
| `init_custom_ar` | Custom all-reduce init |
| `init_custom_qr` | Custom QR init |
| `qr_all_reduce` | QR all-reduce |
| `fused_allreduce_rmsnorm` | Fused all-reduce + RMS norm |
| `fused_allreduce_rmsnorm_quant` | Fused all-reduce + RMS norm + quant |
| `init_dist_env` / `destroy_dist_env` | Distributed env management |
| `ensure_model_parallel_initialized` | Model parallel init |
| `get_semaphore_workspace` | Semaphore workspace |
| `register_graph_buffers` | Graph buffer registration |
| `get_graph_buffer_ipc_meta` | IPC metadata |

---

### 9. KV Cache Management

| Function | Description |
|----------|-------------|
| `reshape_and_cache` | Reshape and cache KV |
| `reshape_and_cache_flash` | Flash-style reshape and cache |
| `reshape_and_cache_with_block_quant` | Block-quantized cache |
| `reshape_and_cache_with_pertoken_quant` | Per-token quantized cache |
| `reshape_and_cache_with_block_quant_for_asm_pa` | Block-quant for ASM PA |
| `concat_and_cache_mla` | MLA-specific cache concat |
| `copy_blocks` | Copy cache blocks |
| `swap_blocks` | Swap cache blocks |
| `indexer_k_quant_and_cache` | K-quant cache indexer |
| `cp_gather_indexer_k_quant_cache` | CP gather K-quant cache |

---

### 10. Sampling & Activation

| Function | Description |
|----------|-------------|
| `greedy_sample` | Greedy sampling |
| `random_sample` | Random sampling |
| `mixed_sample` | Mixed sampling |
| `random_sample_outer_exponential` | Outer exponential random |
| `mixed_sample_outer_exponential` | Outer exponential mixed |
| `sigmoid` / `exponential` / `tanh` | Element-wise activations |
| `gelu_fast` / `gelu_tanh_and_mul` / `gelu_and_mul` | GELU variants |
| `silu_and_mul` / `scaled_silu_and_mul` | SiLU variants |
| `causal_conv1d_update` | Causal Conv1D update (for Mamba/Hybrid models) |

---

### 11. Utility Operations

| Function | Description |
|----------|-------------|
| `partial_transpose` | Partial transpose |
| `ragged_layout_trans` | Ragged layout transform |
| `wvSplitKQ` / `wvSpltK` | Weight-value split KQ |
| `wv_splitk_small_fp16_bf16` | Small split KQ |
| `get_padded_m` | Get padded M dimension |
| `get_cu_num` | Get CU count |
| `get_gfx` | Get GFX architecture |
| `get_dtype_max` | Get dtype max value |
| `compile_ops` | Operation compilation |
| `direct_register_custom_op` | Custom op registration |
| `torch_compile_guard` | torch.compile guard |

---

## Key Findings for MiMo-V2.5

### What MiMo Needs and AITER Has:

| MiMo Component | AITER Solution | Priority |
|----------------|----------------|----------|
| MLA attention (prefill) | `mla_prefill_asm_fwd`, `mla_prefill_ps_asm_fwd` | **CRITICAL** |
| MLA attention (decode) | `mla_decode_stage1_asm_fwd`, `hk_mla_decode_fwd` | **CRITICAL** |
| MLA KV cache | `concat_and_cache_mla`, `fused_qk_rope_concat_and_cache_mla` | **CRITICAL** |
| Sparse MLA | `unified_attention_sparse_mla` | HIGH |
| 256-expert MoE routing | `ck_moe_stage1/2`, `fmoe_g1u1`, `moe_fused_gate` | HIGH |
| Expert GEMMs | `gemm_a8w8`, `batched_gemm_bf16_CK` | MEDIUM |
| RMSNorm | `rmsnorm2d_fwd_with_add_ck` (fused with residual) | MEDIUM |
| RoPE | `rope_cached_positions_fwd` | MEDIUM |
| Top-8 routing | `topk_softmax`, `grouped_topk` | MEDIUM |

### Integration Paths:

1. **Python sidecar** (recommended): Run AITER as a Python process, communicate with llama.cpp via shared memory or IPC. AITER handles attention+MLA+MoE offload.

2. **Direct CK port**: Port CK HIP kernel templates directly into llama.cpp's ggml-cuda layer. Multi-day effort but native integration.

3. **vLLM migration**: Use AITER through vLLM's native integration (vLLM 0.25.2.dev already installed). Requires fixing TP=2 NCCL crash.

4. **Hybrid**: Use AITER MLA operations as an external attention backend for llama.cpp, replacing the built-in flash attention.

---

## JIT Build System

AITER uses a JIT compilation system that builds CK kernels on-demand:

```python
# Example: Building MLA prefill kernel
from aiter import jit_build
# The build system:
# 1. Detects GPU arch (gfx90a)
# 2. Selects CK template parameters
# 3. Compiles HIP/C++ to .so
# 4. Caches in aiter/jit/
# 5. Loads .so on next import
```

Current JIT cache (gfx90a):
```
aiter/jit/
├── module_aiter_core.so                              # Core ops
└── mha_fwd_fp16_nbias_mask_nlse_ndropout_nqscale.so  # Flash attention
```

Additional kernels will be JIT-compiled on first use. Each takes ~30-120 seconds to compile.

---

## Conclusion

AITER is not just a flash attention replacement — it's a **complete inference kernel library** with operations for every component of modern LLM architectures. For MiMo-V2.5 specifically, AITER has purpose-built MLA and MoE kernels that could replace multiple components of the current llama.cpp pipeline.

The discovery that AITER has `mla_prefill_asm_fwd`, `mla_decode_stage1_asm_fwd`, and `fused_qk_rope_concat_and_cache_mla` fundamentally changes the integration strategy: instead of patching flash attention into llama.cpp, we can potentially offload the entire MLA + MoE pipeline to AITER.
