# MoE Inference Engine Survey for gfx90a

Comprehensive comparison of every Mixture-of-Experts inference engine evaluated for the 2× MI210 box, for running **230B-class MoE models** that exceed VRAM (64 GB × 2 = 128 GB) and must spill experts to the 499 GB system RAM.

## The core challenge

A 230B MoE (e.g. mimo: 230B params, 10B active) is ~130 GB at Q4 — **bigger than the 128 GB combined VRAM**. So you need an engine that can **selectively offload expert weights to CPU** while keeping attention + active experts on GPU. This is a narrow field.

## Engine comparison

| Engine | Selective expert offload? | GGUF? | gfx90a ROCm? | Cross-session prefix cache? | Verdict |
|--------|:------------------------:|:-----:|:------------:|:---------------------------:|---------|
| **llama.cpp** | ✅ (`-ot` / `-ncmoe`) | ✅ | ✅ | ⚠️ slot save/restore (no prefill skip) | **Current production** |
| **vLLM** (`--moe-expert-cache-size`) | ✅ | ❌ (HF format) | ✅ | ✅ (prefix caching) | **Best alternative** |
| **SGLang** | ❌ (no selective offload) | ❌ | ✅ (2 patches) | ✅ (RadixAttention) | Blocked for 230B |
| **KTransformers** | ✅ | ✅ | ⚠️ beta ROCm | — | Promising, immature |
| **DeepSpeed-MII** | ❌ (EP only) | ❌ | ✅ | — | No CPU offload |
| **PowerInfer** | ✅ | ✅ | ❌ (CUDA) | — | Research prototype |

### llama.cpp — current production ✅

- **Selective offload:** `-ot "blk\.N\.ffn.*exps=CPU"` pins specific expert layers to CPU via regex. Or `-ncmoe N` to offload N expert layers automatically.
- **Per-layer KV types:** our patch (`-ctk-cpu`/`-ctv-cpu`) lets CPU layers use compressed KV.
- **Strengths:** GGUF ecosystem, mature, flexible tensor placement, single-stream fast.
- **Weakness:** weak batching (throughput flat ~185 t/s regardless of concurrency), no true cross-session prefix caching (slot restore works but doesn't skip prefill).

### vLLM with `--moe-expert-cache-size` — best alternative

vLLM 0.25+ runs on gfx90a. The `--moe-expert-cache-size N` flag implements an **expert weight cache**: experts are loaded into VRAM on demand and evicted LRU when the cache is full — effectively selective offload without the manual `-ot` regex.

- **Strengths:** excellent batching/concurrency (scales to thousands of t/s aggregate), prefix caching, continuous batching.
- **Weakness:** no GGUF support (must convert to HuggingFace format), single-GPU testing recommended first (multi-GPU MI210 has edge cases), fp8 dead on gfx90a.

**Setup:** see [`guides/moe-expert-cache-vllm.md`](../guides/moe-expert-cache-vllm.md).

### SGLang — BLOCKED for 230B MoE

SGLang **does work on gfx90a** (proven — see [`docs/05`](05-sglang-on-gfx90a.md)), and it unlocks **RadixAttention** (prefix-tree KV reuse that would eliminate 100% of cached-prefix prefill). But for the 230B MoE specifically it's blocked:

1. **No selective expert offload** — SGLang tries to load all weights into VRAM → OOM on a 130 GB model with 128 GB VRAM.
2. **No GGUF support** — mimo is GGUF-only.
3. KTransformers-style CPU+GPU MoE for SGLang is on the roadmap but not available.

**Viable path:** hybrid — SGLang serves a small model (with RadixAttention prefix caching) while llama.cpp serves the full 230B. See [`docs/05`](05-sglang-on-gfx90a.md).

### KTransformers

- CPU+GPU MoE offload (the exact feature we need), GGUF support.
- **ROCm support is beta** — works but not battle-tested on gfx90a.
- Worth revisiting as it matures.

### DeepSpeed-MII

- **Expert parallelism only** — shards experts across GPUs. No CPU offload. So a 130 GB model can't fit.
- Not viable for frontier-size MoE on this box.

### Expert parallelism on PCIe — NOT recommended

Splitting experts across the 2 MI210s (each card holds half the experts) sounds natural for MoE, but the cards are **PCIe-linked with no xGMI**. Every routed expert activation crosses the PCIe bus. Measured on Qwen3-Next-80B:

| Mode | agg@256 |
|------|--------:|
| Tensor-parallel (TP=2) | **2342 t/s** |
| Expert parallel | 811 t/s |

EP is **2.9× slower** than TP at high concurrency. The PCIe fabric is the bottleneck — each expert dispatch pays PCIe latency with no xGMI shortcut. **Use tensor-parallel, not expert-parallel, on PCIe-linked cards.**

### PowerInfer

- Academic prototype for CPU+GPU MoE offload.
- CUDA-only (no ROCm path). Not viable on gfx90a without a port.

## Recommendation matrix

| Use case | Recommended engine |
|----------|--------------------|
| 230B MoE (exceeds VRAM), single user | **llama.cpp** (`-ot` CPU split + per-layer KV) |
| ≤128 GB MoE, many concurrent users | **vLLM** (TP=2, `--moe-expert-cache-size`) |
| Small model + prefix caching | **SGLang** (RadixAttention) |
| Max single-stream speed (model fits VRAM) | **llama.cpp** (single card) |
| Future: CPU+GPU MoE with batching | Watch **KTransformers** ROCm maturity |
