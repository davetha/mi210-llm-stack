# Framework Landscape on MI210 (gfx90a)

**Date**: 2025-07-25
**Platform**: AMD Instinct MI210, gfx90a, ROCm 7.14
**Container**: `fa-build`

## Executive Summary

The MI210 runs a modern AMD inference stack with multiple overlapping kernel frameworks. This document provides the **authoritative inventory** of what is actually installed and working, correcting earlier misconceptions.

## Installed Framework Inventory

### Confirmed Installed and Importable

| Framework | Version | Import Name | Role | MI210 Status |
|-----------|---------|-------------|------|--------------|
| **AITER** | 0.1.13.post2.dev1 | `aiter` | Unified inference ops (397 functions) | ✅ Working |
| **CK (Composable Kernel)** | via AITER JIT | (internal) | C++ kernel template library | ✅ JIT-compiles for gfx90a |
| **Triton** | 3.7.1+git0263a6a6 | `triton` | Python DSL → HIP JIT compiler | ✅ Working |
| **flash_attn** | 2.8.3 | `flash_attn` | Flash Attention (uses CK backend on ROCm) | ✅ Working |
| **PyTorch** | 2.11.0+rocm7.14.0 | `torch` | Tensor framework + SDPA | ✅ Working |
| **vLLM** | 0.25.2.dev0 | `vllm` | Inference engine (editable install) | ✅ TP=1, ❌ TP=2 |
| **tilelang** | 0.1.10 | `tilelang` | Tile-based kernel language (experimental) | ✅ Imports, untested |
| **conch-triton-kernels** | 1.2.1 | `conch` | Triton kernel collection (StackAV) | ✅ Imports, contents TBD |

### NOT Installed (Earlier Documentation Was Wrong)

| Framework | Previous Claim | Reality |
|-----------|----------------|---------|
| **tokenspeed-mla** | "Installed, untested" | ❌ NOT installed |
| **humming-kernels** | "Installed" | ❌ NOT installed |
| **flashinfer** | "Should be available" | ❌ NOT installed |

**Correction note**: Earlier documentation mentioned tokenspeed-mla and humming-kernels as installed. They appear in vLLM 0.26.0 dependency lists but were NOT installed in our `fa-build` container (which uses vLLM 0.25.2.dev0). The `pip list` output confirms their absence.

---

## Framework Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  vLLM 0.25.2.dev │ Custom Python │ llama.cpp (via sidecar) │
├─────────────────────────────────────────────────────────────┤
│              High-Level Inference Library                   │
│                        AITER 0.1.13                         │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────────────┐ │
│  │ MLA Ops  │ │ MoE Ops  │ │ GEMM   │ │ 22 Attention     │ │
│  │ (11 fns) │ │ (28 fns) │ │(40+ fns)│ │ Variants         │ │
│  └──────────┘ └──────────┘ └────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│              Kernel Implementation Layer                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ CK Templates    │  │ Triton 3.7.1 │  │ flash_attn 2.8 │ │
│  │ (C++/HIP, JIT)  │  │ (Python DSL) │  │ (wraps CK)     │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
│  ┌─────────────────┐  ┌──────────────┐                     │
│  │ conch 1.2.1     │  │ tilelang 0.1 │                     │
│  │ (Triton kernels)│  │ (tile DSL)   │                     │
│  └─────────────────┘  └──────────────┘                     │
├─────────────────────────────────────────────────────────────┤
│                 Runtime Layer                               │
│              HIP / ROCm 7.14.0                              │
├─────────────────────────────────────────────────────────────┤
│              MI210 Hardware (gfx90a, CDNA2)                 │
│              64GB HBM2e, 104 CUs, 8MB L2                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Framework Details

### 1. AITER (AMD Inference Tuning and Extension Repository)

**Version**: `0.1.13.post2.dev1+gb32deb267`
**Location**: `/opt/python/lib/python3.14/site-packages/aiter/`
**API Surface**: 397 public functions

AITER is AMD's unified inference operations library. It wraps both CK (C++ templates, JIT-compiled) and Triton (Python DSL) kernel implementations behind a unified Python API.

**Architecture**:
- `aiter/jit/` — CK kernels JIT-compiled to .so on first import
- `aiter/ops/` — Python API wrapping CK + Triton kernels
- `aiter/ops/triton/` — Pure Triton kernel implementations
- `aiter/ops/triton/_triton_kernels/` — Pre-written Triton kernel sources
- `aiter/ops/triton/gluon/` — Gluon framework kernels (experimental)
- `aiter/ops/flydsl/` — FlyDSL kernels (experimental, minimal)

**JIT Cache (gfx90a)**:
```
aiter/jit/
├── module_aiter_core.so         (core CK operations)
└── mha_fwd_fp16_nbias_mask_nlse_ndropout_nqscale.so  (flash attention)
```

Additional kernels JIT-compile on first invocation (~30-120s each).

### 2. CK (Composable Kernel)

CK is AMD's C++ template library for writing high-performance GPU kernels. AITER uses CK as its primary backend.

**MI210 Compatibility**: ✅ Full. AITER JIT system detects `gfx90a` and compiles CK templates with correct architecture flags.

**Verified Performance**: CK flash attention achieves 2,086,037 tok/s (1.96ms for S=4096, D=128, H=32) — the fastest attention kernel on MI210.

### 3. Triton (AMD Fork)

**Version**: `3.7.1+git0263a6a6.rocm7.14.0`
**Location**: `/opt/python/lib/python3.14/site-packages/triton/`

AMD's fork of Triton with ROCm support. JIT-compiles Python DSL to HIP code at runtime.

**MI210 Compatibility**: ✅ Working. AITER includes Triton kernel sources in `aiter/ops/triton/_triton_kernels/` that are designed for AMD GPUs.

**Available Triton attention kernels**:
- `flash_attn_triton_amd/fwd_prefill.py` — Triton flash attention prefill
- `flash_attn_triton_amd/fwd_decode.py` — Triton flash attention decode
- `flash_attn_triton_amd/interface_v2.py` — Interface v2
- `flash_attn_triton_amd/interface_v3.py` — Interface v3

**Also available** (in `aiter/ops/triton/attention/`):
- `mla_decode.py`, `mla_decode_rope.py`
- `unified_attention.py`, `unified_attention_sparse_mla.py`
- `lean_atten.py`, `lean_atten_paged.py`
- `pod_attention.py`, `hstu_attention.py`
- `pa_prefill.py`, `pa_decode.py`, `chunked_pa_prefill.py`
- `fav3_sage_attention.py`, `fav3_sage_attention_mxfp4.py`
- And more

### 4. flash_attn 2.8.3

**Version**: 2.8.3
**Location**: `/opt/python/lib/python3.14/site-packages/flash_attn/`

On ROCm, flash_attn 2.8.3 uses the CK backend (not the NVIDIA cutlass backend). This means it goes through the same CK compilation path as AITER's `flash_attn_func`.

**Performance**: ~1,967,134 tok/s (essentially identical to AITER CK).

### 5. tilelang

**Version**: 0.1.10
**Location**: `/opt/python/lib/python3.14/site-packages/tilelang/`

A newer tile-based kernel programming language, designed as an alternative to Triton with a different abstraction model. Still experimental (0.1.x version).

**MI210 Status**: ✅ Imports successfully. Has its own CK fork in `tilelang/3rdparty/composable_kernel/`. Attention kernel availability TBD — needs investigation.

**Key exports**: `DataType`, `Fragment`, `JITKernel`, `Layout`, `PassConfigKey`, `Path`, `Profiler`, `TensorSupplyType`, `autotune`, `autotuner`, `backend`, `cache`

### 6. conch-triton-kernels

**Version**: 1.2.1
**Author**: StackAV (Jacob Manning, Ryan Hsu)
**Location**: `/opt/python/lib/python3.14/site-packages/conch/`
**Repo**: https://github.com/stackav-oss/conch

A Triton kernel repository. Required by vLLM.

**MI210 Status**: ✅ Imports as `conch`. Contents need investigation — the `triton_kernels` submodule import failed, suggesting a different import path.

### 7. vLLM (Editable Install)

**Version**: 0.25.2.dev0+g752a3a504.d20260722.rocm714
**Location**: `/build/vllm` (editable install)

**MI210 Status**: ✅ TP=1 works (loads models, runs inference). ❌ TP=2 crashes (NCCL-related, not gfx90a fundamental).

vLLM integrates with AITER natively. When running vLLM, it automatically uses AITER for attention, MoE, and other operations on AMD GPUs.

---

## What's NOT Available

### flashinfer
Not installed. flashinfer-python requires gfx942+ (MI300X) for ROCm — it does not support gfx90a. This is a hardware limitation, not a configuration issue.

### tokenspeed-mla / humming-kernels
These are dependencies of vLLM 0.26.0+ but we're running 0.25.2.dev0. They are NOT installed. Upgrading to vLLM 0.26.0 would require a full rebuild.

### NVIDIA-only frameworks
- CuTeDSL (NVIDIA's new Cutlass DSL) — NVIDIA only
- xformers — NVIDIA focused
- FlashInfer — gfx942+ on ROCm

---

## Performance Comparison (Attention Kernels)

Benchmarked on MI210 with B=1, S=4096, H=32, D=128, fp16, causal=True:

| Kernel | Time (ms) | Speed (tok/s) | Correctness (max diff) | Status |
|--------|-----------|---------------|----------------------|--------|
| **AITER CK** (`flash_attn_func`) | 2.06 | 1,985,756 | 0.0020 | ✅ Fastest |
| **flash_attn 2.8.3** (CK) | 2.08 | 1,967,134 | 0.0020 | ✅ Same backend |
| **PyTorch SDPA** | 2.33 | 1,757,190 | 0.0000 | ✅ Reference |

AITER CK is **13% faster** than PyTorch SDPA and is the fastest verified attention kernel on MI210.

---

## Summary

The MI210 has access to a modern, comprehensive kernel ecosystem through AITER. The key realization is that AITER is not just flash attention — it's a complete inference kernel library with MLA-specific, MoE-specific, and many other operations purpose-built for models like MiMo-V2.5.

The main gap is flashinfer (requires MI300X) and vLLM 0.26.0 features (tokenspeed-mla, humming-kernels). Neither is critical — AITER already provides superior coverage for our use case.
