# MI210 LLM Stack — Optimizing 230B MoE Inference on 2× AMD MI210 (gfx90a)

A complete optimization journal for running **large Mixture-of-Experts LLMs** (up to 230B parameters) on a pair of AMD Instinct MI210 accelerators — the cheapest CDNA2 cards, **PCIe-linked with no xGMI bridge**. This repo is the hub: architecture, deep-dive docs, build guides, and the change log for every patch that shipped. The actual code lives in two companion repos.

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

`mimo` is a **full-attention CPU-hybrid MoE**. Prefill pushes every prompt token through all 48 layers — including the 25 CPU-resident ones — and attention is O(n²). The result: **prefill is CPU-dominated** (~7.4 s per chunk reading the growing KV cache back from DDR4), even though decode is GPU-bound at ~22 t/s. This whole optimization session was about attacking that prefill bottleneck from every angle.

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

### Benchmarks
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

## The key insight that tied it together

TurboQuant's **CPU** path is numerically correct (cosine > 0.98). Its **GPU** path is broken on gfx90a (wave64 shuffle/ballot bugs). So instead of fixing the GPU kernels (a multi-kernel port), we added **per-layer KV types** (`-ctk-cpu turbo3 -ctv-cpu turbo3`) to compress *only* the 25 CPU-pinned layers — getting 5× less DDR4 traffic on exactly the layers that are bandwidth-bound, while keeping the GPU layers at full fp16 quality. The GPU TurboQuant bug becomes irrelevant.

## License

MIT.
