# Platform Gaps — gfx90a / CDNA2 vs what modern LLM stacks expect

A consolidated gap analysis: which capabilities the vLLM / SGLang / KTransformers stacks assume, which the MI210 (gfx90a) box actually has, and why **llama.cpp is the one engine that sidesteps every gap**. This ties together [`docs/01`](01-gfx90a-architecture-constraints.md), [`docs/06`](06-vllm-poc-results.md), and [`docs/07`](07-ktransformers-poc-results.md).

---

## Hardware gaps

| Capability | MI210 (gfx90a) | What the stacks want | Impact |
|---|---|---|---|
| **UVA zero-copy** (unified virtual addressing, host buffer mapped to GPU) | ❌ requires CDNA3 (MI300) | vLLM `cpu_offload_gb` UVA path | Offload silently no-ops — see [`changes/07`](../changes/07-vllm-cpu-offload-analysis.md) |
| **Matrix Core (MFMA)** | ✅ fp16/bf16 | — | Fine; this is what rocBLAS uses |
| **FP8 / BF8** | ❌ | vLLM fp8 Marlin, `_scaled_mm` | fp8 checkpoints (Qwen3-Coder-Next fp8) unusable |
| **P2P / peer-DMA** | ❌ (PCIe only, no xGMI) | NCCL/RCCL native peer access | TP all-reduce bounces through host RAM (works, slow) |
| **AVX-512** | ❌ (Zen3 = AVX2) | KTransformers CPU expert kernels | KTransformers CPU path unusable — see [`docs/07`](07-ktransformers-poc-results.md) |
| **AMX** | ❌ | KTransformers preferred CPU path | same |

## Software gaps

| Component | Status on gfx90a | Notes |
|---|---|---|
| **sgl-kernel** | ⚠️ patchable for SGLang, ❌ CUDA-only for KTransformers | SGLang: 2 patches make it build (see [`docs/05`](05-sglang-on-gfx90a.md)). KTransformers consumes it as a prebuilt dep with no build hook. |
| **vLLM MoE expert cache** | 📋 surveyed, not deployed | `--moe-expert-cache-size` evaluated in [`docs/04`](04-moe-engine-survey.md) |
| **Ray + ROCm** | ✅ **NOW SOLVED** | Previously a blocker for multi-node vLLM; resolved in this session |
| **AITER** (AMD Instinct Triton Extension for ROCm) | ❌ CDNA3+ only | Gates some SGLang/vLLM attention backends to MI300 |

## Why llama.cpp works where the others don't

llama.cpp sidesteps every gap above by making none of the assumptions the others lean on:

| Assumption the modern stacks make | llama.cpp's answer |
|---|---|
| GPU-resident attention via vendor FlashAttention / sgl-kernel | **Software attention** in CPU fallback — no rocWMMA/CK dependency |
| Vendor GEMM via rocBLAS-only paths | **hipBLAS** + explicit `memcpy` for CPU↔GPU splits |
| UVA zero-copy for offload | **Explicit `memcpy`** for every host↔device transfer (the `PrefetchOffloader` philosophy) |
| AVX-512/AMX CPU kernels | **AVX2-tuned** CPU kernels — runs on any x86-64 server |
| Automatic tensor-parallel sharding | **Manual `-ot` tensor-split** — you say exactly which layers go where |
| FP8 inference | **GGUF quant** (Q4/Q8/KIVI2) — hardware-agnostic |

This is why mimo (230B MoE) runs in production on llama.cpp with a 25-CPU/23-GPU layer split, while every other engine either OOMs, fails to import, or rejects the geometry. llama.cpp's lack of magic is the feature.

---

## CDNA2 vs CDNA3 — the full picture

| Feature | CDNA2 (gfx90a / MI210) | CDNA3 (gfx942 / MI300X) |
|---|---|---|
| Compute units | 64 CU | 304 CU |
| Matrix engine | MFMA (fp16/bf16) | MFMA + WMA |
| fp8 / bf8 | ❌ | ✅ |
| Wavefront | 64 lanes (wave64) | 64 lanes |
| xGMI / P2P | ❌ (MI210 PCIe only) | ✅ |
| **UVA zero-copy offload** | ❌ | ✅ |
| HBM capacity | 64 GB HBM2e | 192 GB HBM3 |
| AITER support | ❌ | ✅ |

The MI210 is the cheapest CDNA2 card. Almost every "modern" feature the LLM serving stacks assume is a CDNA3 addition: FP8, UVA zero-copy, native P2P, AITER. The MI210 has Matrix Core and a working HIP/rocBLAS/Triton toolchain — which is enough for llama.cpp, vLLM single-GPU, vLLM TP=2 (now), and SGLang small models, but not enough for the CPU-offload / fp8 / expert-parallel paths that the newer stacks are built around.

## What this means for the stack

- **Production reasoner (230B MoE):** llama.cpp `-ot` CPU split. Unchanged. Nothing else fits.
- **Smaller / interactive models:** vLLM (single or TP=2) and SGLang are now both viable — see [`docs/05`](05-sglang-on-gfx90a.md) and [`docs/06`](06-vllm-poc-results.md).
- **CPU-offload path on MI210:** must use explicit H2D copy (`PrefetchOffloader`, `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`), never UVA. See [`changes/07`](../changes/07-vllm-cpu-offload-analysis.md).
- **KTransformers:** blocked on this hardware indefinitely (AVX-512/AMX + sgl-kernel + geometry). See [`docs/07`](07-ktransformers-poc-results.md).
