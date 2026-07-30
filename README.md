# MI210 LLM Stack — Optimizing 310B MoE Inference on 2× AMD MI210 (gfx90a)

A complete optimization journal for running **large Mixture-of-Experts LLMs** (up to 310B parameters, MiMo-V2.5) on a pair of AMD Instinct MI210 accelerators — the cheapest CDNA2 cards, **PCIe-linked with no xGMI bridge**. This repo is the hub: architecture, deep-dive docs, build guides, and the change log for every patch that shipped. The actual code lives in two companion repos.

> ⚠️ **Retracted (2026-07-26)**: an earlier headline here claimed a binary-patched MLA ASM breakthrough at **3M tok/s prefill / 0.090ms decode**. Both numbers were real measurements but **mis-attributed** — `mla.py` gates its ASM paths on gfx942/gfx950, so on gfx90a they were measuring the Triton/CK fallback, not ASM. The patch method described in [`docs/14-mla-asm-binary-patch.md`](docs/14-mla-asm-binary-patch.md) is also wrong. See [`docs/19`](docs/19-aiter-operator-port-matrix.md).

> **ATOM integration (2026-07-27)**: ATOM generates coherent text on MI210 at 34.5 tok/s. The throughput is real; the "hybrid ASM prefill" attribution was not — no ASM ran. Of the 1,251 `.co` files installed at the time, **1,147 could not execute on CDNA2** and have since been removed. See [`docs/16`](docs/16-complete-technical-reference.md), noting its superseded banner.

> **ASM flash attention works (2026-07-27)**: `fmha_v3_fwd` **does** run on gfx90a — 80/80 configs numerically exact, batched and varlen. It was never a hardware limit; six architecture-string comparisons gated it out, one written in a negated form that a grep for the positive form misses. See [`docs/19-aiter-operator-port-matrix.md`](docs/19-aiter-operator-port-matrix.md).

> **ASM paged-attention decode fixed (2026-07-27)**: `pa_fwd_asm` **does** run on gfx90a — 48/48 configs numerically exact vs a PyTorch reference. The earlier "gfx942 binaries can't run on gfx90a" conclusion was wrong: the real blocker was a **stale JIT module** whose kernarg layout predated the installed `.co` files, so the kernel ran at full speed and silently discarded every store. Also audits all 1,251 patched `.co` files. See [`docs/18-pa-fwd-asm-resolved.md`](docs/18-pa-fwd-asm-resolved.md).

> **Hardware:** 2× AMD MI210 (gfx90a / CDNA2, 64 GB HBM2e each) · AMD EPYC 74F3 (24c / 48t) · 499 GB DDR4 · ROCm 7.14 · Ubuntu 26.04. Everything runs in Docker.

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │            llama-swap  (:8090)               │
                          │   OpenAI-compatible · VRAM lifecycle mgr     │
                          │   -watch-config · evicts on VRAM pressure    │
                          └───────────────┬─────────────────────────────┘
                                          │ starts / stops containers
            ┌─────────────────────────────┼──────────────────────────────────┐
            ▼                             ▼                                  ▼
   ┌─────────────────┐          ┌──────────────────┐              ┌──────────────────┐
   │  coder (GDN)    │          │  mimo (MoE)      │    ...       │  deephat (7B)    │
   │  Qwen3-Next-80B │          │  230B/10B-active │              │  always resident │
   │  RPC → 2 cards  │          │  native 2-card   │              │  RPC pair        │
   │  256K ctx, hot  │          │  -ot CPU split   │              │  32K ctx         │
   └─────────────────┘          └──────────────────┘              └──────────────────┘
        │                            │  25/48 expert                      │
        │                            │  layers on CPU                     │
        ▼                            ▼                                    ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                    2× AMD MI210 (gfx90a, 64GB each)                      │
   │                         PCIe only (no xGMI)                              │
   ├──────────────────────────────────────────────────────────────────────────┤
   │              EPYC 74F3  24c/48t  ·  499 GB DDR4  ·  ROCm 7.14            │
   └──────────────────────────────────────────────────────────────────────────┘
```

**6 models** behind one OpenAI-compatible endpoint (`http://host:8090/v1`, no auth). `mimo` is the primary reasoner — a 230B MoE that exceeds VRAM, so 25 of its 48 expert layers are pinned to CPU.

### The bottleneck

`mimo` is a **full-attention CPU-hybrid MoE**. Prefill pushes every prompt token through all 48 layers — including the 25 whose expert FFNs are CPU-resident. The result: **prefill is CPU-dominated** (~7.4 s per chunk), even though decode is GPU-bound at ~22 t/s. This whole optimization session was about attacking that prefill bottleneck from every angle.

> **Correction.** This used to read "~7.4 s per chunk reading the growing KV cache back from DDR4", which was wrong. `launch-mimo.sh` passes `-ngl 999` and an `-ot` regex matching only `ffn.*exps`, so **every attention block and the whole KV cache live in HBM** (`-ctk q8_0 -ctv q4_1`). There is no DDR4 KV re-read and the O(n²) attention runs on the 1.6 TB/s side. The 7.4 s is 25 layers of Q4_K expert-FFN GEMM on the CPU. The placement was already right; the explanation attached to it was not, and it pointed at sparse-attention work that the FLOP crossover says is aimed at the smaller term below ~800k tokens. Caught by outside review, 2026-07-30.

---

## What was accomplished

| # | Goal | Status | Repo / Doc |
|---|------|--------|------------|
| G1 | **Per-layer KV cache types** (`-ctk-cpu` / `-ctv-cpu`) — compress only the CPU layers, keep GPU layers at full precision | ✅ **Built & verified** | [`changes/01`](changes/01-per-layer-kv-types.md) · [llama.cpp-mi210](https://github.com/davetha/llama.cpp-mi210) |
| G2 | **KIVI 2-bit quant** (`GGML_TYPE_KIVI2`) — hardware-agnostic 3.0 bpw, all correctness tests pass | ✅ **Built & verified** | [`changes/02`](changes/02-kivi2-quant-type.md) · [llama.cpp-mi210](https://github.com/davetha/llama.cpp-mi210) |
| G3 | **TurboQuant wave64 fixes** — 4 categories of fixes, root-caused but GPU path still corrupted | ⚠️ **Partial** (CPU correct, GPU blocked) | [`changes/03`](changes/03-turboquant-wave64-fixes.md) · [llama.cpp-mi210](https://github.com/davetha/llama.cpp-mi210) |
| G4 | **Triton TurboQuant** — GEMM-based WHT, wave64-safe, cosine 0.98 (3-bit) / 0.99 (4-bit) | ✅ **Working** | [turboquant-triton-amd](https://github.com/davetha/turboquant-triton-amd) |
| G5 | **SGLang on gfx90a** — proven viable (2 patches to sgl-kernel), server starts, models load | ✅ **Working** (small models) | [`changes/05`](changes/05-sglang-gfx90a-build.md) · [`docs/05`](docs/05-sglang-on-gfx90a.md) |
| G6 | **Session persistence** — `--slot-save-path` + auto-restore + TTL 24h | ✅ **Deployed** | [`changes/04`](changes/04-session-persistence.md) |
| G7 | **FlashAttention on gfx90a** — CK backend, built from `origin/main` | ✅ **Working** | [`guides/build-flashattention-gfx90a.md`](guides/build-flashattention-gfx90a.md) |
| G8 | **MoE expert cache (vLLM)** — `--moe-expert-cache-size` evaluated | 📋 **Surveyed** | [`docs/04`](docs/04-moe-engine-survey.md) · [`guides/moe-expert-cache-vllm.md`](guides/moe-expert-cache-vllm.md) |

---

## Quick links

### Deep-dive docs
- [`docs/01-gfx90a-architecture-constraints.md`](docs/01-gfx90a-architecture-constraints.md) — CDNA2 vs CDNA3, wave64 vs wave32, why rocWMMA / fp8 / P2P are all dead on gfx90a
- [`docs/02-turboquant-analysis.md`](docs/02-turboquant-analysis.md) — what TurboQuant is, the wave64 root cause, the Triton GEMM solution
- [`docs/03-kivi-and-rotatekv.md`](docs/03-kivi-and-rotatekv.md) — KIVI 2-bit implementation, RotateKV evaluation, KV-compression comparison
- [`docs/04-moe-engine-survey.md`](docs/04-moe-engine-survey.md) — vLLM / SGLang / KTransformers / DeepSpeed-MII / PowerInfer compared
- [`docs/05-sglang-on-gfx90a.md`](docs/05-sglang-on-gfx90a.md) — SGLang proven working, patches, what it unlocks (RadixAttention)
- [`docs/06-vllm-poc-results.md`](docs/06-vllm-poc-results.md) — vLLM single-GPU (25 tok/s) + **TP=2 across both MI210s works** (DeepSeek-V2-Lite 21.7 tok/s)
- [`docs/07-ktransformers-poc-results.md`](docs/07-ktransformers-poc-results.md) — KTransformers POC: 3 independent blockers, cannot serve on this hardware
- [`docs/08-platform-gaps-gfx90a.md`](docs/08-platform-gaps-gfx90a.md) — consolidated CDNA2 vs CDNA3 gap analysis, why llama.cpp sidesteps every gap
- [`docs/09-flashattention-gfx90a-patching.md`](docs/09-flashattention-gfx90a-patching.md) — why FA is a regression on gfx90a, the V-dequant gap, three patching approaches
- [`docs/19-aiter-operator-port-matrix.md`](docs/19-aiter-operator-port-matrix.md) — which of AITER's 1,422 ASM kernels can run on gfx90a, and why the other 1,180 cannot
- [`docs/20-int8-gemm-gfx90a.md`](docs/20-int8-gemm-gfx90a.md) — INT8 GEMM built from source on gfx90a, bit-exact, 4.3x at decode shapes
- [`docs/21-fp8-block-gemm-gfx90a.md`](docs/21-fp8-block-gemm-gfx90a.md) — block-scaled FP8 was never a tuning problem; gfx90a has no FP8 *decoder*, and a 3-instruction bit trick fixes it
- [`docs/22-rocwmma-flash-attention-gfx90a.md`](docs/22-rocwmma-flash-attention-gfx90a.md) — rocWMMA FlashAttention is 18-26% **slower**; the matrix cores were never idle, so the premise was wrong rather than the result merely negative
- [`docs/23-vllm-gfx90a-cudagraph-decode-cliff.md`](docs/23-vllm-gfx90a-cudagraph-decode-cliff.md) — `--max-model-len` above 128k costs **10x decode on every request**, because the gfx9 attention gate is evaluated at CUDA-graph capture against the *configured* max
- [`docs/24-mi210-quantization-matrix.md`](docs/24-mi210-quantization-matrix.md) — **which quantization wins on an MI210**: INT8 W8A8, and it was the one format vLLM refused to run. Six formats, two engines, correctness-checked
- [`docs/25-optimization-backlog.md`](docs/25-optimization-backlog.md) — leads not yet pursued, each with how it could be *wrong*, plus the closed ones so they are not retried
- [`docs/26-choosing-checkpoints-on-mi210.md`](docs/26-choosing-checkpoints-on-mi210.md) — **what to actually download**: two fields in `config.json` predict nearly all the performance, and neither is the repo name. Plus running MoE with experts in system RAM
- [`docs/27-which-formats-reach-the-matrix-cores.md`](docs/27-which-formats-reach-the-matrix-cores.md) — int8 is the **only** sub-16-bit matrix dtype gfx90a has (no int4 MFMA, no fp8, no sparsity), and the AITER ASM kernels are bf16 *attention* — so they help every format, not just W8A8
- [`docs/28-model-weight-matrix.md`](docs/28-model-weight-matrix.md) — **start here for "what should I download"**: best weights per tier, the smaller secondary picks, and what never to use — every row measured on this box

### Benchmarks
- [`benchmarks/matrix/`](benchmarks/matrix/) — the quantization matrix: harness, sweeps, raw JSON, and [`REPRODUCE.md`](benchmarks/matrix/REPRODUCE.md) with exact tags, versions and CLI args
- [`benchmarks/README.md`](benchmarks/README.md) — **all measured performance numbers in one place** (vLLM single/TP=2, llama.cpp, TurboQuant, KIVI, FlashAttention, KTransformers)

### Build & ops guides
- [`guides/build-flashattention-gfx90a.md`](guides/build-flashattention-gfx90a.md) — FlashAttention 2.8.3 (CK backend) on MI210
- [`guides/build-turboquant-triton.md`](guides/build-turboquant-triton.md) — run the Triton TurboQuant test
- [`guides/setup-ccache-docker.md`](guides/setup-ccache-docker.md) — ccache in Docker for fast incremental GPU builds
- [`guides/moe-expert-cache-vllm.md`](guides/moe-expert-cache-vllm.md) — vLLM expert cache on gfx90a

### Change log (what shipped)
- [`changes/01-per-layer-kv-types.md`](changes/01-per-layer-kv-types.md) — `-ctk-cpu` / `-ctv-cpu` flags
- [`changes/02-kivi2-quant-type.md`](changes/02-kivi2-quant-type.md) — `GGML_TYPE_KIVI2`
- [`changes/03-turboquant-wave64-fixes.md`](changes/03-turboquant-wave64-fixes.md) — wave64 patches
- [`changes/04-session-persistence.md`](changes/04-session-persistence.md) — KV session save/restore + TTL
- [`changes/05-sglang-gfx90a-build.md`](changes/05-sglang-gfx90a-build.md) — SGLang Docker + patches
- [`changes/06-vllm-tp2-success.md`](changes/06-vllm-tp2-success.md) — **vLLM TP=2 works on 2× MI210** (Docker #2942 fix: shm-size, ulimits, seccomp, `/dev/dri`, numeric GID)
- [`changes/07-vllm-cpu-offload-analysis.md`](changes/07-vllm-cpu-offload-analysis.md) — `cpu_offload_gb` proven no-op on CDNA2 (UVA needs CDNA3); `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` workaround
- [`changes/09-fa-patch-opportunity.md`](changes/09-fa-patch-opportunity.md) — "V quant requires flash_attn" is a code gap, not hardware; V-dequant patch (~20-50 lines) unlocks compressed V + fast FA-off attention

### Configs (production)
- [`configs/launch-mimo.sh`](configs/launch-mimo.sh) — mimo wrapper (deployed)
- [`configs/llama-swap-config.yaml`](configs/llama-swap-config.yaml) — llama-swap config (mimo TTL = 86400)
- [`configs/Dockerfile.sglang-gfx90a`](configs/Dockerfile.sglang-gfx90a) — SGLang Docker image
- [`configs/warm-mimo-session.sh`](configs/warm-mimo-session.sh) — one-time KV warmup
- [`configs/patch_layernorm.py`](configs/patch_layernorm.py) — SGLang runtime layernorm fix

### Tests
- [`tests/test_turboquant_triton.py`](tests/test_turboquant_triton.py) — Triton TurboQuant round-trip on gfx90a

---

## Companion repos

| Repo | What |
|------|------|
| [`davetha/llama.cpp-mi210`](https://github.com/davetha/llama.cpp-mi210) | The llama.cpp fork: per-layer KV types, KIVI2, TurboQuant wave64 patches (+ `modified-files/`, `patches/`, build guide) |
| [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd) | The wave64-safe Triton TurboQuant (GEMM-based WHT) |

---

## The key insights that tied it together

### 1. TurboQuant CPU/GPU split
TurboQuant's **CPU** path is numerically correct (cosine > 0.98). Its **GPU** path is broken on gfx90a (wave64 shuffle/ballot bugs). So instead of fixing the GPU kernels (a multi-kernel port), we added **per-layer KV types** (`-ctk-cpu turbo3 -ctv-cpu turbo3`) to compress *only* the 25 CPU-pinned layers — getting 5× less DDR4 traffic on exactly the layers that are bandwidth-bound, while keeping the GPU layers at full fp16 quality. The GPU TurboQuant bug becomes irrelevant.

### 2. ASM kernel port (gfx942 → gfx90a)
AMD's AITER ships its ASM kernels as pre-compiled code objects for gfx942/gfx950 only. **242 of the 1,422 gfx942 kernels are portable to gfx90a**, and that is a hard ceiling — verified three independent ways, with no kernel blocked by a merely cosmetic difference.

- Swap `v_mfma_f32_16x16x16_bf16` (D3E1) → `v_mfma_f32_16x16x16bf16_1k` (**D3E7**) and rewrite ELF `e_flags`. That is the whole patch.
- The split is exactly by **data type**: every bf16 attention kernel ports, every FP8/INT8 one does not. CDNA2 has no FP8 ALU, no gfx942-shaped INT8 MFMA, and no packed-bf16 atomic — so the 1,180 blocked kernels are arithmetic this hardware cannot do, not effort not yet spent.
- ⚠️ That INT8 verdict is about the **prebuilt ASM blobs only**. gfx90a *does* have native INT8 MFMA at the full 181 TOPS; what it lacks is gfx942's K=32 encoding, which is why those blobs cannot be binary-patched. INT8 kernels compiled **from source** run fine — see insight 3.
- Result: ASM paged-attention decode (48/48) and ASM flash attention (80/80) numerically exact.

An earlier version of this section described a "3-layer patch" swapping D3E1 → **D3CD** (bf16 → f16) plus a `vgpr_count` rewrite. **Both were wrong.** gfx90a has BF16 MFMA, and its VGPR/AGPR file is unified so the register-count rewrite was unnecessary. Scripts implementing that patch are quarantined in [`configs/attic/`](configs/attic/).

See [`docs/19-aiter-operator-port-matrix.md`](docs/19-aiter-operator-port-matrix.md) for the full matrix and reproduction steps.

### 3. INT8 GEMM works on MI210 — but buys throughput only where it saves bytes
`aiter.gemm_a8w8` reported unavailable on gfx90a for a reason unrelated to INT8: four **FP8** kernel instances hang the LLVM register allocator indefinitely, and ninja stops at the first failure, taking all 36 working INT8 instances down with them. Excluding FP8 from the build (CDNA2 has no FP8 hardware to reach anyway) makes the module compile in ~170 s.

The result is **bit-exact** on every shape tested, verified against an exact fp64 integer reference. But CDNA2 gives INT8 and BF16 the *same* 181 TOPS/TFLOPS peak — unlike CDNA3, where INT8 is 2×:

- **Compute-bound (prefill):** 102 TOP/s at 4096³ = 57% of peak, versus bf16's 95 TFLOP/s = 53% of the *same* ceiling. A wash.
- **Memory-bound (decode):** at M=16, N=K=8192, **4.3×** faster than bf16 — half the weight bytes, moved at 73% of HBM bandwidth.

So quantization on this hardware should be justified by memory traffic and capacity, never by arithmetic throughput. See [`docs/20-int8-gemm-gfx90a.md`](docs/20-int8-gemm-gfx90a.md).

### 4. Block-scaled FP8 was never a tuning problem — gfx90a has no FP8 *decoder*
FP8 serving was 10–15x slower than bf16, and the obvious explanation was the missing tuning config vLLM warns about by name. Disassembly says otherwise: vLLM's block-FP8 kernel is **11,997 instructions, of which 64 are MFMA and 7,106 are `v_cmp_ne_u16`/`v_cndmask_b32`** — a software emulation of `e4m3 -> fp16`, because `v_cvt_pk_fp8_f32` does not assemble for gfx90a. A convert-only kernel pins the cost to the format: **117 VALU ops per 4 e4m3 values, against 11 for e5m2 and 11 for int8**, both of which decode with a shift. AMD's `fnuz` spelling does not help (118).

Reinterpreting the bits instead of converting them is exact and nearly free — `h = ((u & 0x80) << 8) | ((u & 0x7f) << 7)` yields an fp16 holding exactly `2^-8` times the value, for normals and denormals alike, and folding `2^16` into the accumulator restores it. **Bit-exact on all 254 non-NaN byte patterns**; 11,997 instructions become 3,954 with the same 64 MFMA.

- 5.5–6x faster than the stock kernel before any tuning; tuning composes with it.
- Tuning *alone* recovers only part of the gap (12.6x → 4.1x at M=1 `qkv_proj`); decode + tuning reaches **1.24x**.
- And at decode the GEMM is weight-bandwidth-bound, so FP8 does not just catch up — at M=1 `gate_up_proj` it runs at **0.59x of bf16's time**.
- **In serving: `Qwen3-14B-FP8` on one MI210 goes from 2.7 to 29.2 tok/s (10.8x), TPOT 373 ms → 33.7 ms.** Across the full 128/4096 × 1/8/32 grid FP8 runs at **0.67–0.85x of bf16** with 1.75x less weight memory and 1.45x more KV cache — a trade worth making, where 0.07x was not. Unlike the ASM decode kernel (1.72x in isolation, ~1% in serving), this GEMM win translated in full, because TPOT pinned at 373 ms regardless of batch size had already identified it as the bottleneck.

**Would CK be faster than Triton for FP8 here? No — it is wrong, not just slow.** CK's block-scaled FP8 GEMM *does* build for gfx90a and needs no FP8 hardware (12,288 `v_mfma_f32_16x16x4f32`, zero FP8 opcodes), because `ck/utility/amd_xdlops.hpp` guards the FP8 MFMA with `#if defined(__gfx94__)` and falls back to f32. But that fallback feeds one scalar per lane to a K=4 matrix instruction, so it computes **exactly one quarter** of the sum (relerr vs `full/4` = 0.0014, i.e. bf16 rounding) plus NaNs — confirmed with all-ones scales, where layout cannot matter. Even repaired, fp32 MFMA runs at ~45 TFLOP/s on CDNA2 against the 181 Triton reaches through fp16.

Also settles the open question on AITER's FP8 GEMM: it is **dequant-based in every implementation that compiles**, CK included — no FP8 hardware is needed anywhere. `_hip_blockscale_supported()` still routes gfx90a to aiter's **Triton** kernel, so opening the gate would reach Triton rather than ASM, and the CK path behind it is broken. The gate stays closed. The one genuinely hardware-dependent piece is the ASM `fp8gemm_blockscale` family: prebuilt gfx942 blobs with no source fallback. See [`docs/21-fp8-block-gemm-gfx90a.md`](docs/21-fp8-block-gemm-gfx90a.md).

## License

MIT.
