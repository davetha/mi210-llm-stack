
> Sisyphus - ultraworker · glm-5.2

• Map Qwen3 family quantization repos Librarian Agent
• Find permissive 400B+ LLMs and MI210 compat Librarian Agent
I'll classify intent: **research/exploratory** — gather current data on permissively-licensed LLMs across four tiers, optimize for MI210 hardware constraints. No implementation. Output a structured report.

My approach: heavy parallel fan-out. I'll fire direct MCP searches (web-search-prime, exa) + background librarian agents for deep dives, then synthesize.

Kicking off Round 1 in parallel now.
✓ Map Qwen3 family quantization repos Librarian Agent
✓ Find permissive 400B+ LLMs and MI210 compat Librarian Agent
⚙ websearch_web_search_exa {"query":"Olmo 3 OLMo 32B Apache 2.0 release 2025 quantization AWQ GPTQ GGUF","numResults":10}
⚙ web-search-prime_web_search_prime {"search_query":"DeepSeek V3 R1 MIT license quantization AWQ GPTQ huggingface","content_size":"high","location":"us"}
⚙ websearch_web_search_exa {"query":"Apache 2.0 permissively licensed open weight LLM 2025 400B parameters huggingface","numResults":10}
⚙ web-search-prime_web_search_prime {"search_query":"Qwen3-Next-80B-A3B gDNA gated DeltaNet linear attention hybrid architecture","content_size":"high","location":"us"}
⚙ web-search-prime_web_search_prime {"search_query":"AMD MI210 gfx90a FP8 support ROCm vLLM INT8 W8A8 AWQ GPTQ inference","content_size":"high","location":"us"}
✗ web-search-prime_web_search_prime {"search_query":"Qwen3 model family 2025 Apache 2.0 license MoE dense releases list","content_size":"high","location":"us","search_recency_filter":"oneYear"} failed
Error: MCP error -32001: Request timed out
Excellent first round. Key findings: Arcee Trinity Large (399B/A13B, Apache-2.0) is the 400B-tier answer; Qwen3-Next-80B-A3B has confirmed Gated DeltaNet hybrid attention (perfect for 256k context); Olmo 3 32B is Apache-2.0; MI210 confirmed no native FP8 but INT8/W8A8/AWQ/GPTQ work via vLLM/ROCm.

Round 2: deep dive on specific model quantization repos and current 2026 releases.
⚙ websearch_web_search_exa {"numResults":10,"query":"DeepSeek V3.2 official AWQ INT8 BF16 huggingface repo license"}
⚙ websearch_web_search_exa {"numResults":8,"query":"Qwen3-Next-80B-A3B-Thinking huggingface AWQ GGUF quantization repo"}
⚙ web-reader_webReader {"url":"https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8","return_format":"markdown"}
⚙ web-search-prime_web_search_prime {"content_size":"high","location":"us","search_query":"Qwen3-30B-A3B MoE AWQ GPTQ GGUF huggingface quantization repos"}
⚙ web-reader_webReader {"url":"https://huggingface.co/arcee-ai/Trinity-Large-Thinking","return_format":"markdown"}
⚙ web-search-prime_web_search_prime {"content_size":"high","location":"us","search_recency_filter":"oneYear","search_query":"GLM-4.6 GLM-4.5 Zhipu MIT license huggingface parameters MoE"}
⚙ web-search-prime_web_search_prime {"content_size":"high","location":"us","search_query":"Kimi K2 MiniMax M1 linear attention hybrid Apache MIT license open weights"}
✗ web-search-prime_web_search_prime {"content_size":"high","location":"us","search_query":"Qwen3-235B-A22B huggingface AWQ GPTQ INT8 FP8 GGUF official quantization repo"} failed
Error: MCP error -32001: Request timed out
Strong intel. Critical findings: Qwen3-Next-80B-A3B is 3:1 Gated DeltaNet:Gated Attention hybrid (linear-attention native 256k); Trinity Large is Apache-2.0 sparse MoE 399B/A13B with GQA full-attention (512k via extension); GLM-4.5/4.6 are MIT MoE 355B/A32B; MiniMax-M1 is Apache-2.0 hybrid lightning attention.

Round 3: verify specific quantization repo IDs and fill remaining gaps.
⚙ websearch_web_search_exa {"numResults":8,"query":"Qwen3-Coder-480B-A35B huggingface Apache 2.0 license MoE"}
⚙ websearch_web_search_exa {"numResults":8,"query":"Qwen3-235B-A22B AWQ GPTQ Int4 FP8 GGUF official huggingface repo"}
⚙ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507"}
⚙ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507"}
⚙ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/deepseek-ai/DeepSeek-V3"}
✗ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/zai-org/GLM-4.6"} failed
Error: MCP error -32001: Request timed out
✗ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/MiniMaxAI/MiniMax-M1-80k"} failed
Error: MCP error -32001: Request timed out
Excellent. Confirmed: Qwen3-Coder-480B-A35B (Apache-2.0, 480B/A35B, 256k) is a strong ~400B option. DeepSeek-V3 itself is **Model License NOT MIT** (only code is MIT) - excluded. But DeepSeek-V3.1/V3.2/R1 are properly MIT. Let me check background agents and fill remaining gaps.
⚙ background_output librarian - Map Qwen3 family quantization repos
⚙ background_output librarian - Find permissive 400B+ LLMs and MI210 compat
Librarian agents still running. Let me do non-overlapping targeted work — verify GLM-4.6 quants and AWQ availability for the 235B model while waiting.
⚙ websearch_web_search_exa {"numResults":6,"query":"DeepSeek-V3.2 INT8 W8A8 quantization huggingface vLLM"}
⚙ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/MiniMaxAI/MiniMax-M1-80k"}
⚙ websearch_web_search_exa {"numResults":6,"query":"Qwen3-235B-A22B AWQ huggingface official quantization repo"}
⚙ web-reader_webReader {"return_format":"markdown","url":"https://huggingface.co/zai-org/GLM-4.6"}
I have enough core data. Let me do final targeted checks on GLM-4.6 quants, Qwen3-32B alternatives, and cancel background agents that are still running.
⚙ background_cancel {"taskId":"bg_330a44bd"}
⚙ background_cancel {"taskId":"bg_6c48cf60"}
⚙ websearch_web_search_exa {"numResults":8,"query":"GLM-4.6 AWQ GPTQ INT8 GGUF quantization huggingface zai-org"}
⚙ websearch_web_search_exa {"numResults":8,"query":"Qwen3-32B AWQ GPTQ Int4 FP8 GGUF huggingface dense Apache"}
I have comprehensive data. Compiling the final report now.

---

# Open-Weight LLM Selection Report — 2× AMD MI210 (gfx90a/CDNA2)
**As of July 2026 · Permissive licenses only (Apache-2.0 / MIT)**

## Hardware Reality Check (MI210 / gfx90a)

| Format | MI210 Native Compute? | Practical Verdict |
|---|---|---|
| **BF16 / FP16** | ✅ Yes (MFMA) | Reference. Always works. |
| **INT8 / W8A8** | ✅ Yes (CDNA2 MFMA INT8) | **Best runtime format.** Real speedup + ~50% weight RAM. |
| **AWQ-Int4** | ⚠️ Weight-only; compute in BF16 | **Recommended.** ~25% weight RAM, full BF16 accuracy path via Marlin/ROCm kernels. |
| **GPTQ-Int4** | ⚠️ Weight-only; compute in BF16 | Works; slightly lower accuracy than AWQ on Qwen3-32B per Qwen's own benchmarks. |
| **FP8 (e4m3 / e5m2)** | ❌ **NO native FP8 on CDNA2** | **AVOID on MI210.** vLLM-ROCm upcasts FP8 → BF16 at load time. You store FP8 on disk but pay BF16 VRAM at runtime — no benefit. |
| **GGUF (Q4_K_M, Q8_0, IQ4_XS)** | ✅ via vLLM-ROCm GGUF backend or llama.cpp+HIP | Excellent for split VRAM/CPU-RAM offload (KTransformers / llama.cpp RPC). |

**KV-cache math at 256k context (BF16):** for a 235B-class GQA model with ~94 layers × 4 KV heads × 128 head_dim → ~192 KB/token → **~48 GB just for KV at 256k**. This is why **linear/hybrid attention is a hard requirement at the 235B+ tier** if you want 256k context to fit in 128 GB VRAM.

---

## Tier 1 — ~35B (incl. 32B dense)

| Property | **Qwen3-30B-A3B-Thinking-2507** ⭐ | Olmo 3.1 32B Think | Qwen3-32B (dense) |
|---|---|---|---|
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Total / Active params | 30.5B / **3.3B** (MoE) | ~32B dense / 32B | 32.8B dense / 32.8B |
| Architecture | Sparse MoE (128 experts, 8 active), GQA (32Q/4KV) | Dense Transformer, GQA | Dense Transformer, GQA (64Q/8KV) |
| Attention | **Full attention** | Full attention | Full attention |
| Native context | **262,144** (extensible to 1M via YaRN) | 65,536 | 32,768 (YaRN → 131,072) |
| 256k needs YaRN? | No (native) | N/A (cap 65k) | Yes |
| **BF16 repo** | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `allenai/Olmo-3.1-32B-Think` | `Qwen/Qwen3-32B` |
| **AWQ-Int4** | (use QuantTrio or community analog of 30B-A3B) | `cyankiwi/Olmo-3-32B-Think-AWQ-4bit` | `Qwen/Qwen3-32B-AWQ` (official) |
| **GPTQ-Int4** | `Qwen/Qwen3-30B-A3B-GPTQ-Int4` (orig 30B-A3B) | `kaitchup/Olmo-3-32B-Think-gptq-w4a16-g128` | `JunHowie/Qwen3-32B-GPTQ-Int4` |
| **INT8 / W8A8** | (community: AngelSlim/Qwen3-32B_int4_gptq pattern) | `cyankiwi/Olmo-3-32B-Think-AWQ-8bit` (8-bit AWQ) | n/a (no official) |
| **FP8** | n/a (orig 30B-A3B-FP8 exists) | n/a | `qwen-community/Qwen3-32B-FP8` ⚠️ *useless on MI210* |
| **GGUF Q4_K_M / Q8_0 / IQ4_XS** | `unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF`, `bartowski/Qwen3-30B-A3B-Thinking-2507-GGUF` | `unsloth/Olmo-3-32B-Think-GGUF` (Q4_K_M ~19.5 GB, Q8_0 ~34.3 GB, IQ4_XS ~17.3 GB) | `Qwen/Qwen3-32B-GGUF` (official) |
| **BF16 size** | ~61 GB | ~65 GB | ~66 GB |
| **AWQ-Int4 size** | ~18 GB | ~18 GB | ~19 GB |
| **GPTQ-Int4 size** | ~18 GB | ~18 GB | ~19 GB |
| **INT8 / W8A8 size** | ~32 GB | ~33 GB | ~33 GB |
| **GGUF Q4_K_M / Q8_0 / IQ4_XS** | ~19 / ~33 / ~17 GB | ~19.5 / ~34.3 / ~17.3 GB | ~19.5 / ~34.3 / ~17.3 GB |

### 🏆 Tier-1 Recommendation: **Qwen/Qwen3-30B-A3B-Thinking-2507**

**Why:** Highest-quality reasoning model in this tier (AIME25 85.0, beats Gemini-2.5-Flash-Thinking 72.0 by a wide margin), MoE so only 3.3B active = blazing fast on MI210, native 256k context (no YaRN quality hit), Apache-2.0. BF16 fits comfortably in 1× MI210 (61 GB) leaving ~3 GB for KV — or split across 2× MI210 for full 256k context. The AWQ-Int4 (~18 GB) leaves ~110 GB of headroom for huge batches / multiple parallel users.

**Olmo 3.1 32B Think** is the pick if you need a US-origin, fully-open (data+code) model for auditability reasons; quality is comparable on math/code but context is capped at 65k.

---

## Tier 2 — ~80B

| Property | **Qwen3-Next-80B-A3B-Thinking** ⭐ | (no strong Apache/MIT alternative at this exact size) |
|---|---|---|
| License | Apache-2.0 | — |
| Total / Active params | 80B / **3B** (extreme-sparsity MoE, 512 experts × 10 active + 1 shared) | — |
| Architecture | Sparse MoE + **HYBRID attention** | — |
| Attention | **3:1 mix of Gated DeltaNet (linear) → Gated Attention (full)**. 48 layers = 12 blocks of (3×GDN + 1×GA). **LINEAR**-dominant. | — |
| Native context | **262,144** (extensible to 1,010,000 via YaRN) | — |
| 256k needs YaRN? | **No — native 256k.** | — |
| Why this matters for MI210 | At 256k context, KV cache is **near-constant** (only 12 of 48 layers store KV). Fits in 128 GB easily. A comparable full-attention 80B would blow VRAM at 256k. | — |
| **BF16 repo** | `Qwen/Qwen3-Next-80B-A3B-Thinking` | — |
| **AWQ-Int4** | `cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-4bit` | — |
| **GPTQ-Int4** | (no widely-adopted repo — use AWQ) | — |
| **INT8 / W8A8** | `cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-8bit` (8-bit AWQ ≈ W8A8 weight-only; safe on MI210) | — |
| **FP8** | `Qwen/Qwen3-Next-80B-A3B-Thinking-FP8` ⚠️ *useless on MI210 — upcasts to BF16* | — |
| **GGUF Q4_K_M** | `bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF` (Q4_K_M = **48.7 GB**, single-file, no split) | — |
| **GGUF Q8_0** | `bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF` (Q8_0 = **84.8 GB**, split) | — |
| **GGUF IQ4_XS** | `bartowski/Qwen_Qwen3-Next-80B-A3B-Thinking-GGUF` (IQ4_XS = **42.85 GB**, single-file) — also `unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF` | — |
| **BF16 size** | ~160 GB | — |
| **AWQ-Int4 size** | ~45 GB | — |
| **AWQ-8bit size** | ~83 GB | — |
| **GGUF Q4_K_M / Q8_0 / IQ4_XS** | 48.7 / 84.8 / 42.85 GB | — |

### 🏆 Tier-2 Recommendation: **Qwen/Qwen3-Next-80B-A3B-Thinking**

**Why:** This is the **only** genuinely permissive ~80B model that solves the 256k-on-MI210 problem. The 3:1 Gated-DeltaNet:Gated-Attention hybrid gives O(1) KV growth on 75% of layers — at 256k context KV stays well under 8 GB (vs ~32+ GB for an equivalent full-attention 80B). It also beats Qwen3-235B-A22B on multiple benchmarks while running 10× cheaper. Apache-2.0, native 256k, 80 GB BF16 weight fits across 2× MI210 with room to spare, AWQ-Int4 (~45 GB) fits in 1× MI210 with full 256k context and room for batching.

**Caveat:** vLLM and SGLang both have day-one support for Qwen3-Next (merged into transformers mainline). Make sure you're on vLLM ≥ 0.11.x / SGLang latest main. The linear-attention kernels are ROCm-compatible (no NVIDIA-only paths).

---

## Tier 3 — 120B-250B

| Property | **Qwen3-235B-A22B-Instruct-2507** ⭐ | Qwen3-235B-A22B-Thinking-2507 |
|---|---|---|
| License | Apache-2.0 | Apache-2.0 |
| Total / Active params | 235B / **22B** (MoE 128 experts × 8 active) | 235B / 22B |
| Architecture | Sparse MoE, GQA (64Q / 4KV) | Same |
| Attention | **Full attention** (GQA) | Full attention |
| Native context | **262,144** (extensible to 1,010,000 via YaRN + DualChunkAttention) | 262,144 |
| 256k needs YaRN? | No (native 256k) | No |
| KV cache at 256k | ~48 GB (heavy!) — only fits with INT4 weights + careful batching, or CPU offload | Same |
| **BF16 repo** | `Qwen/Qwen3-235B-A22B-Instruct-2507` | `Qwen/Qwen3-235B-A22B-Thinking-2507` |
| **AWQ-Int4** | (use QuantTrio style; exists for Thinking variant) → `QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ` | `QuantTrio/Qwen3-235B-A22B-Thinking-2507-AWQ` |
| **GPTQ-Int4** | `Qwen/Qwen3-235B-A22B-GPTQ-Int4` (older base; same arch) | (same official repo) |
| **INT8 / W8A8** | (community/RedHat AI produces these on-request — none confirmed for 235B 2507 specifically as of Jul 2026) | — |
| **FP8** | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` ⚠️ *useless on MI210* | (older: `Qwen/Qwen3-235B-A22B-FP8`) |
| **GGUF Q4_K_M / Q8_0 / IQ4_XS** | `Qwen/Qwen3-235B-A22B-GGUF` (official, Q4_K_M ~135 GB / 5 split; also Q5_K_M, Q6_K, Q8_0) ; `unsloth/Qwen3-235B-A22-Instruct-2507-GGUF` (IQ4_XS available) | Same family |
| **BF16 size** | ~470 GB | ~470 GB |
| **AWQ-Int4 size** | ~135 GB | ~135 GB |
| **GPTQ-Int4 size** | ~135 GB | ~135 GB |
| **GGUF Q4_K_M / Q8_0** | ~135 / ~250 GB | Same |

### 🏆 Tier-3 Recommendation: **Qwen/Qwen3-235B-A22B-Instruct-2507**

**Why:** Strongest permissive 200B-class model with the most complete official quantization suite. The Instruct-2507 variant (July 2025) was the biggest jump in quality among Qwen3 235B variants — it beats DeepSeek-V3.1, Kimi K2, and Claude Opus 4 Non-Thinking on most benchmarks and **scores 41.8 on ARC-AGI** (vs 9.0 for DeepSeek-V3-0324). It's Apache-2.0, has official GPTQ-Int4 + FP8 + GGUF (Q4_K_M / Q5_K_M / Q6_K / Q8_0) and the QuantTrio AWQ community port.

**Critical MI210 fit warning:** INT4 weights (~135 GB) **+ KV cache for 256k (~48 GB) = ~183 GB** which exceeds 128 GB VRAM. Practical options:
1. **Drop context to ~128k** (KV ~24 GB) → fits in 128 GB VRAM at INT4 with ~25 GB headroom.
2. **Use IQ3_XS GGUF (~100 GB)** via llama.cpp with CPU-RAM offload for full 256k.
3. **Use the dual-chunk sparse attention** config (`config_1m.json` + `VLLM_ATTENTION_BACKEND=DUAL_CHUNK_FLASH_ATTN`) — Qwen officially supports this and reports 3× speedup at long context.

**Alternative for 256k context on this hardware:** *MiniMax-M1-80k* (Apache-2.0, 456B/A45.9B, **lightning hybrid attention**, native **1M context**) — but at 456B params, even INT4 is ~230 GB and won't fit in 128 GB VRAM without aggressive offload. Recommended only if 256k+ context is non-negotiable.

---

## Tier 4 — ~400B (interpreting flexibly as 300B-700B "frontier" tier)

| Property | **Arcee Trinity-Large-Thinking** ⭐ | Qwen3-Coder-480B-A35B-Instruct | GLM-4.6 | MiniMax-M1-80k |
|---|---|---|---|---|
| License | **Apache-2.0** (OpenMDW-1.1, functionally Apache-2.0) | Apache-2.0 | **MIT** | **Apache-2.0** |
| Origin | 🇺🇸 US (Arcee AI, Miami) | 🇨🇳 China (Alibaba Qwen) | 🇨🇳 China (Zhipu/Z.AI) | 🇨🇳 China (MiniMax) |
| Total / Active | 399B / **13B** (256 experts × 4 active) | 480B / **35B** (160 experts × 8 active) | 357B / **~32B** (160 experts × 8 active) | 456B / **45.9B** (32 experts × 8 active) |
| Architecture | Sparse MoE (`AfmoeForCausalLM`) | Sparse MoE (qwen3_moe) | `glm4_moe` | Sparse MoE + **lightning hybrid attention** |
| Attention | **Full GQA** | Full GQA | Full attention | **Hybrid lightning (linear-dominant) + softmax** |
| Native context | 8,192 (pre-train) → **512,000 extended** | 256,000 native | **200,000** | **1,000,000** |
| 256k needs YaRN/extension? | Yes (extended via post-training, 512k cap) | No (native 256k) | No (native 200k, below 256k) | No (native 1M) |
| **BF16 repo** | `arcee-ai/Trinity-Large-Thinking` | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | `zai-org/GLM-4.6` | `MiniMaxAI/MiniMax-M1-80k` |
| **AWQ-Int4** | (community ports emerging; check `unsloth/` and `QuantTrio/`) | (no official; QuantTrio pattern expected) | `QuantTrio/GLM-4.6-AWQ`, `bullpoint/GLM-4.6-AWQ` (4-bit, 176 GB, Marlin-optimized) | (limited community quants) |
| **GPTQ-Int4** | (community) | (none confirmed) | `QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix` (mixed-precision), `ModelCloud/GLM-4.6-GPTQMODEL-W4A16-v2` | (none confirmed) |
| **INT8 / W8A8** | (none confirmed official) | (none confirmed) | **`RedHatAI/GLM-4.6-quantized.w8a8`** ✅ (official Red Hat INT8) | (none) |
| **FP8** | `arcee-ai/Trinity-Large-Preview-FP8` ⚠️ *useless on MI210* | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` ⚠️ *useless on MI210* | (community FP8 exists) | (FP8 static available from MiniMax) |
| **GGUF Q4_K_M** | `arcee-ai/Trinity-Large-Preview-GGUF` (official) | (community: `unsloth/`, `bartowski/` patterns) | `bartowski/zai-org_GLM-4.6-GGUF` (Q4_K_M = **217.65 GB**), `unsloth/GLM-4.6-GGUF` | (limited) |
| **GGUF Q8_0 / IQ4_XS** | (in official GGUF repo) | (community) | Q8_0 = **379.32 GB** / IQ4_XS = **192.02 GB** | (limited) |
| **BF16 size** | ~800 GB | ~960 GB | ~714 GB | ~912 GB |
| **AWQ-Int4 size** | ~200 GB (est.) | ~240 GB (est.) | **~176 GB** (bullpoint, measured) | ~230 GB (est.) |
| **INT8 / W8A8 size** | ~400 GB (est.) | ~480 GB (est.) | ~360 GB | ~455 GB |
| **GGUF Q4_K_M / Q8_0 / IQ4_XS** | ~200 / 400 / 180 GB (est.) | ~240 / 480 / 215 GB (est.) | **217.65 / 379.32 / 192.02 GB** (bartowski, measured) | ~230 / 460 / 200 GB (est.) |

### 🏆 Tier-4 Recommendation: **GLM-4.6 (zai-org/GLM-4.6)** — with caveats

**Why GLM-4.6 wins for MI210 specifically:**
1. **MIT license** — cleanest license of the four, zero ambiguity for commercial use.
2. **Smallest of the frontier tier (357B vs 399/456/480/671B)** — only one that *might* fit in 128 GB VRAM at INT4 (176 GB AWQ still exceeds 128 GB but is close enough that llama.cpp/llamafile can offload the remainder to your 512 GB system RAM with acceptable throughput).
3. **RedHat AI officially published `RedHatAI/GLM-4.6-quantized.w8a8`** — this is the *only* frontier-tier model with a verified, production-grade W8A8 INT8 quant purpose-built for vLLM ≥ 0.11.0. **W8A8 INT8 is exactly what MI210 accelerates natively** (CDNA2 MFMA INT8). This is the best-case runtime path on your hardware.
4. **Excellent community AWQ port** (`bullpoint/GLM-4.6-AWQ` — 176 GB, Marlin-optimized, vLLM-ready).
5. **Complete GGUF ecosystem** — `bartowski/zai-org_GLM-4.6-GGUF` and `unsloth/GLM-4.6-GGUF` cover every quant from IQ1_S (76 GB) to Q8_0 (379 GB), so you can pick exactly the size that fits your VRAM+RAM budget.
6. **200k native context, full attention** — KV cache at 200k for GQA-32/4 KV heads × 128 head_dim × ~96 layers is significant (~24 GB), but feasible if you use AWQ-Int4 weights (~176 GB → split: 128 GB VRAM + 48 GB offloaded).

**Why not the others:**
- **Arcee Trinity-Large-Thinking**: Best Apache-2.0 option from US origin; reasoning quality is high (PinchBench 91.9, AIME25 96.3). However: native pretrain context is only 8k, "extended" to 512k via post-training (quality degrades beyond 128k-256k in practice). No verified INT8/W8A8 quant exists. Best for agentic workloads under 128k context — use **Trinity-Large-Thinking if your priority is US-origin + tool-calling agent + ≤128k context**.
- **Qwen3-Coder-480B-A35B-Instruct**: Best for **code** specifically (it's the only frontier-tier "Coder" model here). Apache-2.0, 256k native. But 480B is heavier than GLM-4.6 with no accuracy advantage on non-code tasks. No INT8 quant available.
- **MiniMax-M1-80k**: Most KV-efficient (lightning linear attention, native 1M context) but at 456B params it's simply too large for 2× MI210 even at INT4 (~230 GB). Community quants are scarce. Choose this **only if you absolutely need >256k context** and accept heavy CPU offload.
- **DeepSeek-V3.2** (671B / A37B, MIT, MLA+sparse attention): too large; BF16 1.3 TB, AWQ ~300 GB. Even with 512 GB system RAM, throughput would be CPU-bound. Mentioned for completeness.

---

## Summary Table — One Pick Per Tier

| Tier | Model | License | Quant to use on MI210 | Fits 128 GB VRAM @ 256k ctx? |
|---|---|---|---|---|
| **~35B** | **Qwen3-30B-A3B-Thinking-2507** | Apache-2.0 | BF16 (61 GB) or AWQ-Int4 (~18 GB) | ✅ Yes — easily, even in BF16 |
| **~80B** | **Qwen3-Next-80B-A3B-Thinking** | Apache-2.0 | AWQ-Int4 (~45 GB) — linear-attn KV stays tiny | ✅ Yes — at AWQ-Int4 with full 256k |
| **120B–250B** | **Qwen3-235B-A22B-Instruct-2507** | Apache-2.0 | AWQ-Int4 (~135 GB) — *drop to 128k ctx or use IQ3_XS GGUF + offload* | ⚠️ Tight — needs ≤128k ctx or CPU offload for 256k |
| **~400B** | **GLM-4.6** | MIT | `RedHatAI/GLM-4.6-quantized.w8a8` (native INT8!) or `bullpoint/GLM-4.6-AWQ` (4-bit, 176 GB) | ⚠️ No — needs CPU-RAM offload (llama.cpp/KTransformers) |

## Cross-Tier Notes & Warnings

1. **FP8 quants are a trap on MI210.** They occupy disk but provide no runtime benefit — vLLM-ROCm upcasts to BF16. Skip them; you're paying for nothing.
2. **Linear/hybrid attention is the unlock for 256k context on MI210.** Of the recommended models, only **Qwen3-Next-80B-A3B** (Gated DeltaNet) and **MiniMax-M1** (lightning attention) are linear-dominant. Full-attention models at 80B+ will blow your VRAM budget at 256k context due to KV growth.
3. **Olmo 3 / 3.1** is the only model in this report that is *fully open* (training data, code, checkpoints). All others are open-weights only. If auditability/reproducibility matters more than peak quality, swap Olmo 3.1 32B Think into Tier 1.
4. **Arcee Trinity-Large-Thinking** is the only US-origin model in the frontier tier — relevant for procurement / EU AI Act / US Federal use cases. It also has the best agentic / tool-calling benchmark scores of any Apache-2.0 model (PinchBench 91.9, τ²-Bench 94.7).
5. **vLLM version**: Use latest vLLM nightly for ROCm (`pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly`) — Qwen3-Next, GLM-4.6 MoE, and Trinity all require vLLM ≥ 0.11.x.
6. **No tier is "empty"** for permissive licenses as of July 2026 — Apache-2.0 / MIT coverage is now complete from 30B through 671B. The bottleneck on MI210 is purely VRAM, not license.

### Sources
- HuggingFace model cards (Qwen3 family, arcee-ai Trinity, zai-org GLM-4.6, MiniMaxAI MiniMax-M1, allenai Olmo 3, deepseek-ai DeepSeek-V3.2)
- AMD ROCm docs: `rocm.docs.amd.com/.../vllm-optimization` (Quark FP8/W8A8 INT8 on ROCm)
- vLLM docs: `docs.vllm.ai/.../int8_w8a8`, `docs.vllm.ai/.../amd-installation`
- vLLM blog: `vllm.ai/blog/2025-09-11-qwen3-next` (hybrid attention support)
- Sebastian Raschka's "Hybrid Attention" gallery (Gated DeltaNet 3:1 confirmation)
- Community quant repos: QuantTrio, bartowski, unsloth, cyankiwi, RedHatAI, bullpoint
