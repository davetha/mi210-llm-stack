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

## The optimization stack — what each patch is worth

The headline numbers, all measured on **Qwen3-30B-A3B W8A8, TP=2**, one
variable per A/B, each with an assertion proving the code path actually ran.
Rounds 31–42; full method in `docs/40`–`docs/45`.

| Optimization | Metric | Gain | Round / doc |
|---|---|---:|---|
| **AITER CK int8 GEMM** (`VLLM_ROCM_USE_AITER_LINEAR=1`) | decode | **1.480×** (1.708× at TP=1) | 40 · `docs/43` |
| **AITER ASM flash attention** (`VLLM_PREFER_AITER_FA=1`) | prefill | **1.190×** @16k, **1.332×** @25k | 37 · `docs/28` |
| | TTFT | 0.850× / 0.752× | |
| **PCIe P2P** (`NCCL_P2P_DISABLE=0`) | prefill | **1.112×** | 31 · `docs/40` |
| **Async scheduling** (default-on in ≥0.26) | decode | **1.110×** | 38b · `docs/42` |
| **256k paged attention** (patched gfx9 gate) | decode @205k | **13.08×** | — · `docs/36` |
| ~~vLLM 0.23.1 → 0.26.1rc0~~ | decode | ~~1.035×~~ **inside noise** | 36 · `docs/46` |

**The decode noise floor is 1.036×** (5 identical runs, `docs/46`). Any
decode row above must clear it; the version-climb row does not, and is struck
out. Prefill and TTFT need only ~1.005×.

**Read the gains as independent, not cumulative.** Each row is a single-variable
A/B against a baseline that already had the *other* optimizations on. The
reference arm moved **55.7 → 82.9 tok/s decode**, and that is the CK GEMM row
alone — its 55.7 baseline already included AITER FA, P2P and async scheduling.
Reproduced on two independently built images (82.48 and 82.92). Nobody has
measured a stock-everything-off arm against the full stack, so no total-stack
multiplier is claimed here.

### Measured and rejected

Recorded because a negative result costs the same GPU time as a positive one,
and re-testing these is pure waste.

| Tried | Result |
|---|---|
| GPU clock pinning (1700 MHz) | 0.968× — DPM was never throttling decode |
| CUDA-graph capture geometry (131k vs 32k) | 0.986× — no self-inflicted config tax |
| PCIe ACS redirect removal | null, all arms within 0.4% |
| Tuned fused-MoE config (partial) | **0.786×** — nearest-M match with no fallback |
| AITER fused MoE (`_MOE=1`) | 0.977× — runs (`module_moe_asm` loads), just slower |
| AITER rope / unified-attn / custom-AR / triton-GEMM | cannot engage; four distinct reasons, `docs/45` |
| `enforce_eager` | **3.5× slower**, 82% of decode in gaps between kernels |

### Where the remaining time goes

Decode-window kernel profile at **99.9% GPU-busy** (rocprofv3, round 41) — so
the residual is *in-kernel*, not launch overhead:

| kernel | % of decode |
|---|---:|
| MoE cluster (`fused_moe` + `topkGating` + `moe_sum_vec`) | **32%** |
| `paged_attention_ll4mi` | 13.1% |
| `wvSplitK` | 8.5% |
| `dynamic_scaled_int8_quant` | 6.2% |

---

## Standardized 13K-Token Prefill Benchmarks

Cross-engine cold/hot prefill comparison with a deterministic ~13K token prompt.
See [`benchmarks-13k-prefill.md`](./benchmarks-13k-prefill.md) for full details.

| Setup | Cold Prefill (tok/s) | Hot Prefill (tok/s) | Cache Speedup | Decode (tok/s) |
|---|---:|---:|---:|---:|
| vLLM TP=1 (DSV2-Lite BF16) | 3,743 | 5,753 | 1.5× | 23.9 |
| llama.cpp (DSV2-Lite q8_0) | 194 | 1,509 | 7.8× | 63.8 |
| llama.cpp (mimo 230B) | 63 | 2,389 | 38× | 20.8 |

---

## KV Cache Compression Prefill A/B Benchmark

How KV cache quantization type (`f16`, `q4_0`, `q8_0`, `q8_0/q4_1`) affects
prefill throughput on llama.cpp, across all-GPU and split CPU/GPU configs.
See [`kv-compression-prefill.md`](./kv-compression-prefill.md) for full details.

| Config | GPU layers | KV type | Prefill (tok/s) | vs f16 |
|---|---|---|---:|---:|
| A: all-GPU f16 | 27/27 | f16/f16 | 14.4 ⚠ | 1.0× |
| B: all-GPU q4_0 | 27/27 | q4_0/q4_0 | 492.4 | **34×** |
| C: all-GPU q8_0 | 27/27 | q8_0/q8_0 | 462.1 | 32× |
| D: all-GPU q8_0/q4_1 | 27/27 | q8_0/q4_1 | 461.5 | 32× |
| E: split f16 | 23/27 | f16/f16 | 648.9 | 1.0× |
| F: split q4_0 | 23/27 | q4_0/q4_0 | 445.6 | 0.69× |
| G: split q8_0/q4_1 | 23/27 | q8_0/q4_1 | 417.1 | 0.64× |

> f16 KV on all-GPU is **34× slower** than q4_0 — VRAM starvation under
> production load kills the flash-attention workspace. With adequate VRAM
> (split config), f16 is actually the **fastest** prefill type.

---

## AITER ASM Attention — Kernel and Serving

Two linked results on AMD's hand-written ASM attention kernels, which ship for
gfx942/gfx950 only and were ported to gfx90a here.

**Kernels in isolation** ([`asm-attention-gfx90a.md`](./asm-attention-gfx90a.md)) —
every number produced by a backend that passed a correctness check immediately
before being timed:

| Path | vs alternative | Peak |
|---|---:|---|
| Prefill `fmha_v3_fwd` vs PyTorch SDPA | 1.13–1.86× | 89.9 TFLOP/s (50% of bf16 peak) |
| Decode `pa_fwd_asm` vs HIP kernel | 0.99–1.72× | >1 TB/s (64% of HBM2e peak) |

**End to end under vLLM** ([`vllm-aiter-asm-gfx90a.md`](./vllm-aiter-asm-gfx90a.md)) —
Qwen3-14B bf16, same hardware, attention backend the only variable:

| Prompt | conc 1 | conc 8 | conc 32 |
|---|---:|---:|---:|
| 128 tokens | 1.02× | 1.00× | 1.02× |
| 4096 tokens | 1.01× | **1.23×** | **1.23×** |

> The ASM kernels are worth **1.23× serving throughput on long prompts under
> concurrency, and nothing on short prompts or single streams**. Of that gain,
> the ASM *decode* kernel accounts for ~1% — within run-to-run noise, so a
> 1.72× kernel proved indistinguishable from zero once the GEMMs and scheduling
> around it are included. The gain comes from the prefill path.
>
> vLLM could not reach AITER on gfx90a at all before this: its dispatch gate
> calls `on_mi3xx()` (gfx942/gfx950) while documenting itself as gfx9, and the
> failure is silent. [`configs/enable_vllm_aiter_gfx90a.py`](../configs/enable_vllm_aiter_gfx90a.py)
> opens it for **attention only** — the master gate stays closed so AITER's
> GEMM/MoE/FP8 paths remain unreachable on a chip with no FP8 ALU — and
> separately *narrows* the `torch._scaled_mm` gate, which wrongly admits gfx90a
> because CDNA2 reports compute capability 9.0.

**FP8 weight-only on CDNA2** (same writeup, part 3) — Qwen3-14B-FP8, block-quantized:

| | bf16 | FP8 | |
|---|---:|---:|---|
| Weights in VRAM | 27.52 GiB | 15.71 GiB | 1.75× saving, **survives** |
| KV cache | 172,000 tok | 240,992 tok | 1.40× more |
| Output tok/s (128 tok, conc 1) | 39.7 | 2.7 → **29.2** | 0.07× → **0.74×** |

> FP8 loads, keeps its weights as `float8_e4m3fn` in VRAM (no load-time
> dequant), and **still reaches the bf16 ASM attention kernels** — the
> weight-only hypothesis holds.
>
> ⚠️ **The "10–15× slower" result below was real but its diagnosis was wrong,
> and it is now fixed.** This page originally blamed a missing tuning config,
> because vLLM names the exact missing file. The actual cause is that gfx90a has
> no FP8 *decode* instruction: the kernel spent **7,106 of its 11,997
> instructions** emulating `e4m3 → fp16` in software, against 64 doing matrix
> math. Tuning alone only reaches 4.1× slower; a 3-op bit-reinterpret decode
> reaches **0.67–0.85× of bf16**, and is *faster* than bf16 at decode (M=1),
> where streaming half the weight bytes beats the shared 181 TFLOP/s ceiling.
>
> See [`../docs/21-fp8-block-gemm-gfx90a.md`](../docs/21-fp8-block-gemm-gfx90a.md).

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

**The optimization rounds** (the table at the top of this file):

- [`docs/43-ck-int8-gemm-gfx90a.md`](../docs/43-ck-int8-gemm-gfx90a.md) — the CK int8 GEMM, **1.48× decode**, the three gates that hid it, and the tuner that deadlocks
- [`docs/45-aiter-fast-path-survey.md`](../docs/45-aiter-fast-path-survey.md) — all 17 AITER gates surveyed; one runs and loses, four cannot engage
- [`docs/42-decode-gap-probes.md`](../docs/42-decode-gap-probes.md) — clocks, capture geometry and launch overhead ruled out; the ~3× gap is **in-kernel** at 99.9% coverage
- [`docs/40-the-three-collective-gates.md`](../docs/40-the-three-collective-gates.md) — PCIe P2P works: 26.98 GB/s peer vs 14.16 staged
- [`docs/36-depth-curves-and-256k-validation.md`](../docs/36-depth-curves-and-256k-validation.md) — the 256k paged-attention result
- [`benchmarks/matrix/REPRODUCE.md`](./matrix/REPRODUCE.md) — exact images, flags and patch order to reproduce any of it

**Background:**

- [`docs/06-vllm-poc-results.md`](../docs/06-vllm-poc-results.md) — vLLM single-GPU + multi-GPU POC detail
- [`docs/07-ktransformers-poc-results.md`](../docs/07-ktransformers-poc-results.md) — KTransformers POC detail
- [`docs/08-platform-gaps-gfx90a.md`](../docs/08-platform-gaps-gfx90a.md) — full platform gap analysis (CDNA2 vs CDNA3)
- [`docs/01-gfx90a-architecture-constraints.md`](../docs/01-gfx90a-architecture-constraints.md) — architecture constraints
