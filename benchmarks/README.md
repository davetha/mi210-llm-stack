# MI210 (gfx90a) Benchmarks

Central index of every measured performance number from the MI210 LLM stack. All results are **real measured** data from this hardware session, not projections.

## Hardware

- 2× AMD MI210 (gfx90a / CDNA2, 64 GB HBM2e each)
- AMD EPYC 74F3 (24c / 48t, Zen3)
- 499 GB DDR4 RAM
- PCIe 4.0 (**no xGMI / NVLink bridge** — cards are PCIe-linked only)
- ROCm 7.14, Ubuntu 26.04

For the full architecture constraints behind these numbers, see [`docs/01-gfx90a-architecture-constraints.md`](../docs/01-gfx90a-architecture-constraints.md).

---

## Summary Table

### vLLM (single MI210)

| Model | Config | TTFT | Decode (tok/s) | Prefill (tok/s) | Notes |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite (16B MoE) | TP=1, BF16, eager | 0.06–0.38 s | 25.0 | ~1,600 | Correct output |
| DeepSeek-V2-Lite + `cpu_offload_gb=20` | TP=1, UVA | 0.06 s | 23.9 | — | ❌ offload is a no-op (KV cache identical) |
| DeepSeek-V2-Lite + `cpu_offload_gb=30` | TP=1, UVA | 0.05 s | 23.3 | — | ❌ offload is a no-op |
| DeepSeek-V2-Lite + prefetch offload | TP=1, explicit H2D | 0.06 s | 23.5 | — | ❌ offload is a no-op |

> **Why the offload numbers look "fine":** `cpu_offload_gb` silently does nothing on CDNA2. The KV-cache token budget is identical at offload=0/10/20/30, and decode speed is unchanged — if 20 GB were actually streaming over PCIe, decode would cap at ~15 tok/s, not stay at ~24. See [`changes/07-vllm-cpu-offload-analysis.md`](../changes/07-vllm-cpu-offload-analysis.md).

### vLLM (dual MI210, TP=2) ✅ WORKS

| Model | Config | Load Time | Decode (tok/s) | Output | Notes |
|---|---|---|---|---|---|
| facebook/opt-1.3b | TP=2, BF16, eager | 29.6 s | 10.6 | Correct | First TP=2 success on MI210 |
| DeepSeek-V2-Lite (16B MoE) | TP=2, BF16, eager | ~58 min | 21.7 | Correct | **MoE works with TP=2!** |

The TP=2 path was unlocked purely by Docker resource limits — see [`changes/06-vllm-tp2-success.md`](../changes/06-vllm-tp2-success.md).

### llama.cpp (production, for comparison)

| Model | Config | Cold Prefill | Cached Prefill | Decode | Notes |
|---|---|---|---|---|---|
| mimo (230B MoE) | 25 CPU / 23 GPU split, `q8_0` weights / `q4_1` KV | ~43 tok/s | 0.8 s (cached) | — | Production via llama-swap |

### TurboQuant (Triton on gfx90a)

| Config | Cosine Similarity | Compression | Status |
|---|---|---|---|
| 3-bit (Triton) | 0.9838 | 4.92× | ✅ PASS |
| 4-bit (Triton) | 0.9955 | 3.76× | ✅ PASS |

The HIP-kernel TurboQuant path is broken on gfx90a (wave64 corruption); the Triton GEMM path is the working alternative. See [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd).

### KIVI 2-bit Quantization (`GGML_TYPE_KIVI2`)

| Test | Result | Notes |
|---|---|---|
| Exact 4-level `{0,1,2,3}` | error=0.000000 | ✅ PASS |
| Endpoints (min / max) | max_error=0.000977 | ✅ PASS |
| Constant group | error=0.000000 | ✅ PASS |
| Random invariant (32,768 elems) | 0 violations | ✅ PASS |

### FlashAttention (ROCm CK backend)

| Metric | Value |
|---|---|
| Version | flash-attn 2.8.3 |
| Backend | Composable Kernel (CK) |
| Build objects | 2,926 |
| Build time | ~59 min (`MAX_JOBS=32`) |
| Max diff vs reference | 0.001804 |
| head_dim=64 causal | ✅ PASS |

Build guide: [`guides/build-flashattention-gfx90a.md`](../guides/build-flashattention-gfx90a.md).

### KTransformers

| Component | Status |
|---|---|
| kt-kernel build | ✅ Compiles (ROCm 7.14, gfx90a) |
| GPU HIP matmul | ✅ Works (~60 iters/s for 4096³ bf16) |
| Server import | ❌ `sgl-kernel` is CUDA-only |
| CPU expert kernels | ❌ AVX2-only (needs AVX-512 / AMX) |
| DeepSeek-V2-Lite | ❌ Geometry rejected (1408 % 256 ≠ 0) |

Three independent blockers — KTransformers cannot serve any model on this hardware today. Details: [`docs/07-ktransformers-poc-results.md`](../docs/07-ktransformers-poc-results.md).

---

## Related docs

- [`docs/06-vllm-poc-results.md`](../docs/06-vllm-poc-results.md) — vLLM single-GPU + multi-GPU POC detail
- [`docs/07-ktransformers-poc-results.md`](../docs/07-ktransformers-poc-results.md) — KTransformers POC detail
- [`docs/08-platform-gaps-gfx90a.md`](../docs/08-platform-gaps-gfx90a.md) — full platform gap analysis (CDNA2 vs CDNA3)
- [`docs/01-gfx90a-architecture-constraints.md`](../docs/01-gfx90a-architecture-constraints.md) — architecture constraints
