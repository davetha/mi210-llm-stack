# MI210 LLM Stack — Optimizing 310B MoE Inference on 2× AMD MI210 (gfx90a)

A complete optimization journal for running **large Mixture-of-Experts LLMs** (up to 310B parameters, MiMo-V2.5) on a pair of AMD Instinct MI210 accelerators — the cheapest CDNA2 cards, **PCIe-linked with no xGMI bridge**. This repo is the hub: architecture, deep-dive docs, build guides, and the change log for every patch that shipped. The actual code lives in two companion repos.

> ⚠️ **Retracted (2026-07-26)**: an earlier headline here claimed a binary-patched MLA ASM breakthrough at **3M tok/s prefill / 0.090ms decode**. Both numbers were real measurements but **mis-attributed** — `mla.py` gates its ASM paths on gfx942/gfx950, so on gfx90a they were measuring the Triton/CK fallback, not ASM. The patch method described in [`docs/14-mla-asm-binary-patch.md`](docs/14-mla-asm-binary-patch.md) is also wrong. See [`docs/19`](docs/19-aiter-operator-port-matrix.md).

> **ATOM integration (2026-07-27)**: ATOM generates coherent text on MI210 at 34.5 tok/s. The throughput is real; the "hybrid ASM prefill" attribution was not — no ASM ran. Of the 1,251 `.co` files installed at the time, **1,147 could not execute on CDNA2** and have since been removed. See [`docs/16`](docs/16-complete-technical-reference.md), noting its superseded banner.

> **ASM flash attention works (2026-07-27)**: `fmha_v3_fwd` **does** run on gfx90a — 80/80 configs numerically exact, batched and varlen. It was never a hardware limit; six architecture-string comparisons gated it out, one written in a negated form that a grep for the positive form misses. See [`docs/19-aiter-operator-port-matrix.md`](docs/19-aiter-operator-port-matrix.md).

> **ASM paged-attention decode fixed (2026-07-27)**: `pa_fwd_asm` **does** run on gfx90a — 48/48 configs numerically exact vs a PyTorch reference. The earlier "gfx942 binaries can't run on gfx90a" conclusion was wrong: the real blocker was a **stale JIT module** whose kernarg layout predated the installed `.co` files, so the kernel ran at full speed and silently discarded every store. Also audits all 1,251 patched `.co` files. See [`docs/18-pa-fwd-asm-resolved.md`](docs/18-pa-fwd-asm-resolved.md).

> **Launch-flag matrix campaign (2026-08-16 → 08-20)**: both production models on both stacks, 227 gate-v2 cells. Dense qwen38 ladder: **18.3 → 42.3 (+AITER) → 80.0 (+MTP n2, robust) → 108.5 (n5, bimodal)** tok/s decode on the same w8a8 checkpoint; MoE w8a8+MTP n3+AITER = **139.8**. The MTP×quant cells were blocked by a one-line M-RoPE flatten bug (fixed, upstream #52973). Numerics: vLLM DIVERGE base rate measured **zero**; TP and the compile/JIT cache policy are each independent numerics parameters. Suffix decoding on this stack is a **reproduced correctness blocker** (premature stop at temp 0, 2/2). See [docs/58](docs/58-launch-flag-matrix-campaign-2026-08.md) and [docs/59](docs/59-mtp-unblocked-rope-fix-depth-ladders.md).

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

> **Correction.** This used to read "~7.4 s per chunk reading the growing KV cache back from DDR4", which was wrong. `launch-mimo.sh` passes `-ngl 999` and an `-ot` regex matching only `ffn.*exps`, so **every attention block and the whole KV cache live in HBM** (`-ctk q8_0 -ctv q4_1`). There is no DDR4 KV re-read and the O(n²) attention runs on the 1.6 TB/s side. The 7.4 s is 25 layers of Q4_K expert-FFN GEMM on the CPU. The placement was already right; the explanation attached to it was not, and it pointed at sparse-attention work that the FLOP crossover says is aimed at the smaller term below ~800k tokens. Caught by outside review from [Andrei-Dr](https://github.com/Andrei-Dr), 2026-07-30.

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
- [`docs/29-offload-incidents-and-harness-defects.md`](docs/29-offload-incidents-and-harness-defects.md) — the `HSA_XNACK` unified-memory fault that **damaged the host** (load average 70, wedged SVM workers), the prefetch NCCL crash, and four harness defects that cost real arms
- [`docs/30-instruction-budget-gfx90a.md`](docs/30-instruction-budget-gfx90a.md) — the shipped Triton GEMMs issue **~100 instructions per MFMA**; under 3% of issued instructions are matrix ops. Also the full assembled ISA verification table
- [`docs/31-cpu-prefill-arithmetic.md`](docs/31-cpu-prefill-arithmetic.md) — attention and KV are on GPU, so the 7.4 s/chunk is CPU expert GEMM; the chunk-size model, and the round-13 measurement that **refuted** it
- [`docs/32-published-rules-not-yet-applied.md`](docs/32-published-rules-not-yet-applied.md) — published methods that emit **data, not kernels**: per-layer KV allocation rules, absorbed rotations, low-rank correction. No wave64/FP8/int4 exposure
- [`docs/33-macs-per-instruction.md`](docs/33-macs-per-instruction.md) — the MACs-per-instruction ceiling: AITER ASM 539–1,058 against the shipping `fused_moe` at 195 median / 496 best, which makes `fmoe` the largest remaining lead
- [`docs/34-sharded-state-loader.md`](docs/34-sharded-state-loader.md) — **`--load-format sharded_state` turns a 12,366 s bf16 load into 114.65 s** (108×) with throughput unchanged; verified end to end, and it silently corrupts AWQ
- [`docs/35-asm-support-matrix-and-model-selection.md`](docs/35-asm-support-matrix-and-model-selection.md) — the ASM support matrix enumerated, `on_mi3xx()` as the **single upstream predicate** behind every AITER miss, and the MLA field that caused a 44× KV error
- [`docs/36-depth-curves-and-256k-validation.md`](docs/36-depth-curves-and-256k-validation.md) — first depth curves to **150k** for both engines; the 256k patch validated in production (`npar_loops: 10`); a `llama-bench` multi-GPU fault that `llama-server` does not have
- [`docs/37-the-shareable-build.md`](docs/37-the-shareable-build.md) — **start here to build it**: why a Dockerfile and not a fork, the measured before/after for each patch, and a good/better/best model guide with the four mechanisms behind it
- [`docs/38-serving-layer-and-structural-reduction.md`](docs/38-serving-layer-and-structural-reduction.md) — **`--load-format sharded_state` skips the loader entirely**, making the 7-hour load a one-time cost; why prefix caching is worth more here than on a GPU box; LMCache, REAP
- [`docs/39-backlog-addendum.md`](docs/39-backlog-addendum.md) — backlog item **1c resized** (~3x, not 6.4x), **1b inverted** (W4A16-fp16 over W4A8-int8), item 2 cliff ruled out, plus DP=2 and low-bit allreduce
- [`docs/40-the-three-collective-gates.md`](docs/40-the-three-collective-gates.md) — "no xGMI" never meant "no peer-to-peer": PCIe P2P measures **26.98 GB/s** against 14.16 staged, worth **+11.2% prefill** at TP=2, and the default is now on
- [`docs/41-moe-tuning-mi210.md`](docs/41-moe-tuning-mi210.md) — MoE tuning on MI210: the tuner works after Andrei's fix, but a **partial tuned config is harmful** (0.79× prefill) — nearest-M matching has no fallback
- [`docs/59-mtp-after-the-ck-repair.md`](docs/59-mtp-after-the-ck-repair.md) — **`docs/39`'s "re-run MTP after the decode gap closes" condition was tested, and it was the wrong condition.** With the `docs/57` CK path verified live and identical flags, native MTP on Qwen3.6-27B W8A8 measures **0.838× / 0.748× / 0.679× decode at N=1/2/3** (d0), and 0.813×/0.664×/0.558× at d32768 — at **88.6% position-0 acceptance**, right in the 84.8–89.1% band `docs/25` saw. The draft head is fine. Decomposing step cost as `a + bN` gives **a = 1.11** (verification *is* nearly free at M=2…4, exactly as `docs/57`'s 30.60→33.64 µs M=1→M=8 shape predicted) and **b = 1.16** — one MTP draft forward costs **more than all 64 layers of the target**, so break-even is impossible at any acceptance rate, not merely unprofitable. Mechanism: the draft reads 3.29 GB (12.2% of the target's 26.93 GB, of which 2.54 GB is the bf16 `lm_head` vLLM shares with it) but runs at **102 GB/s — 9% of 1170**, against the target's 973 GB/s (83%). That 102 is round 62's *"113 GB/s of 1170, latency-bound, not enough work in flight"* reproduced on an independent workload: a one-layer forward cannot fill a 104-CU card. **Improving the target made speculation worse, not better.** Separately and unexplained: prefill regresses **23–25% and flat across N** (a decode-side flag should not touch prefill; suspect `gdn_attn.py:112 _init_reorder_batch_threshold(1, use_spec_decode)` on this 48-of-64-GDN-layer model)
- [`docs/57-the-ck-gemm-was-never-applied.md`](docs/57-the-ck-gemm-was-never-applied.md) — **the CK int8 GEMM patch had bit-rotted and was writing nothing**: `enable_aiter_ck_gemm_gfx90a.py` aborts on the `register_ops_once` anchor (vLLM ≥0.26.1 now decorates that method itself with `@if_aiter_attention_supported`), and because the write is atomic the `is_linear_enabled` carve-out that *did* match never reached disk — so the deployed image was silently running `TritonInt8ScaledMMLinearKernel`. Fixed with an equivalence case; `check()` was also returning exit 1 on a correctly-patched image. Worth **2.9–3.5× decode** end-to-end on Qwen3.6-27B W8A8 (9.59 → 28.11 t/s at d32768, +17% prefill) — larger than `docs/43`'s isolated 1.662× because the Triton fallback also cost scheduler overhead, and not in tension with `docs/55`'s 0.982×, which measured a *different* model whose decode is 82% latency. **Tuning `a8w8_tuned_gemm.csv` for gfx90a is a null result**: 12 shapes, 1,032 candidates, all within −1.57%…+0.07%, nothing written. M=1 and M=8 take 30.60 vs 33.64 µs on the same shape — 8× the arithmetic for 10% more time, so these GEMMs sit at the bandwidth roofline and tile configs schedule compute. Grep `Selected .*ScaledMMLinearKernel` before trusting any W8A8 number
- [`docs/56-the-tuning-gap-on-gfx90a.md`](docs/56-the-tuning-gap-on-gfx90a.md) — **upstream ships no tuned kernel configs for this card**: 0 of vLLM's 317 `fused_moe` configs are MI210, and all 21 AITER tuning CSVs have **zero gfx90a rows**. One config was produced locally (`docs/41`) and measured 0.786×, so it is not deployed. A second was tuned for `int4_w4a16` and **round 78 closes the line**: with the tuned M=1 config verifiably loaded, single-stream decode is **1.006× / 0.998×** — flat, inside the noise floor. That corroborates round 75's finding that ~82% of decode is latency, not something a tile config can move, and it cost 45 minutes instead of the ~8 GPU-hours the full sweep would have taken. W4A16 does hit AITER FA and paged attention (why its prefill is only 9.5% behind W8A8) but runs an untuned MoE. Two leads closed: MoE stride padding was already measured at 0.961× (`docs/45` round 44), and every 4-bit format converges on the same `TritonW4A16LinearKernel` on CDNA2 — Marlin/Machete are CUDA-only, the RDNA3 kernel is uncompiled, WMMA-based and wave32-hardcoded, so it is a port not a patch
- [`docs/55-vllm-vs-llamacpp-decode-on-the-80b.md`](docs/55-vllm-vs-llamacpp-decode-on-the-80b.md) — **the two engines on the SAME model** (round 76, Qwen3-Next-80B-A3B-Thinking, both ~4-bit): vLLM prefills **2.70×** faster, llama.cpp decodes **1.27× faster at 8K but only 1.11× at 130K** — the earlier 1.22×/1.35× figures compared two *different* models. vLLM's decode is far flatter with depth (−3.5% vs −15.6% over 8K→130K), so the curves converge. llama.cpp also **faults intermittently at long depth** — roughly half of runs across rounds 69/71/76 faulted or returned degraded numbers, from an idle card. Also: vLLM decode has no lever left — the CK int8 GEMM that gave 1.480× on the 30B measures **0.982×** here (and costs 23% of aggregate), and DP=2 is structurally impossible at 77 GB on 64 GB cards. **W4A16 (`t80-awq`, 46 GB) is the better vLLM checkpoint** — 0.905× prefill for 40% less weight memory, 1.10× decode and **2.43× KV capacity (2,334,585 tokens, so 1M fits)**. And the decode gap is bounded: halving weight bytes bought only 1.10×, so weight traffic is **~18% of decode time** and even free weights top out near 57.3 tok/s — **no vLLM checkpoint closes it by getting smaller**. Also: round 73 measured two Triton arms because it set the image for the bench *client* and not the *server* — the fourth such incident — so round 74 asserts the selected kernel and aborts rather than report an unearned ratio
- [`docs/54-kv-compression-on-vllm-and-llamacpp.md`](docs/54-kv-compression-on-vllm-and-llamacpp.md) — **the 1M-context question, answered**: bf16 holds 902,160 KV tokens and physically cannot fit 1M; `turboquant_4bit_nc` holds 2,790,992 (3.09×) but its decode penalty **grows with context instead of amortizing** — 1.48× TPOT at 8K, 3.45× at 64K, **5.21× at 130K**, where it is down to 8.0 tok/s per stream against bf16's 41.9. The two turboquant variants compress 2.6× and 3.8× yet perform identically, so the cost is the backend, not the bytes; prefill is untouched (TTFT within 4.8%). But the question was framed on the wrong stack: **llama.cpp serves a full 1M-token context uncompressed at 27.5 tok/s** (80,267 MiB of 131,040, 50 GB spare, on an 80B). And one production setting is actively wrong — **`f16` KV beats `q8_0`/`q8_0` at every depth**, 1.076× at 8K rising to 1.566× at 130K, while saving only **509 MiB (1%)** because a hybrid-GDN model keeps most layers on constant-size state. Quantizing K is free; quantizing V is what costs. KIVI2 is **CPU-only** (correcting `docs/03` and `changes/02`), and vLLM has no CPU KV offload at all
- [`docs/53-the-power-cap.md`](docs/53-the-power-cap.md) — **the largest lever in the project, and it was not software**: the cards sat pinned at 199–200 W against a 300 W `power1_cap_max` the entire time. 250 W gives **+2.9% prefill**, 300 W gives +8.4%; decode is flat because it is memory-bound and mclk was already maxed. Deployed persistently as a systemd unit at 250 W
- [`docs/52-data-parallel-plus-asm-attention.md`](docs/52-data-parallel-plus-asm-attention.md) — **the best throughput result measured here: 1.118×** (0.888× TPOT) from DP=2 plus ASM paged attention — and it beat its parts for a reason neither half predicted
- [`docs/51-the-low-level-knobs-measured.md`](docs/51-the-low-level-knobs-measured.md) — eleven low-level leads: **seven closed**, one is a pattern-coverage gap rather than a performance result, and the biggest finding is the power cap. Two harness defects recorded as findings, including a restore path that printed success while the cards sat at the wrong value
- [`docs/50-the-five-leads-measured.md`](docs/50-the-five-leads-measured.md) — five leads run: **four nulls or corrections, one confirmed diagnosis** — decode is *latency*-bound (113 GB/s of 1170, VALU 15.5/cyc of ~104), which explains the null streak. Method note: a gap in a config table is a hypothesis, not a finding
- [`docs/49-the-fmoe-asm-tree-at-instruction-level.md`](docs/49-the-fmoe-asm-tree-at-instruction-level.md) — `docs/48`'s *mechanism* was wrong and this replaces it: `v_mfma_f32_16x16x16bf16_1k` and `v_mfma_f32_16x16x16_bf16` are the same instruction under two names. The conclusion survives, the reasoning did not
- [`docs/48-rounds-48-51-and-two-corrections.md`](docs/48-rounds-48-51-and-two-corrections.md) — four nulls, and two earlier documents corrected (itself since corrected by `docs/49`)
- [`docs/47-production-is-broken-and-how-to-restore-it.md`](docs/47-production-is-broken-and-how-to-restore-it.md) — **production cannot start**: llama-swap launches `llama-rocm714-rpc:latest`, which is not on the host and has no build recipe. Native multi-GPU serves both checkpoints correctly with no rebuild (GPU0 25.3 GB / GPU1 26.9 GB, decode 73.4 / 72.3 t/s). Incidentally: **A3B MoE decode barely notices Q5→Q4**, 1.4% not the predicted 20%
- [`docs/46-deployment-noise-and-the-int8-fusion-gap.md`](docs/46-deployment-noise-and-the-int8-fusion-gap.md) — production runs llama.cpp not vLLM, so the measured wins are deployed nowhere; the **decode noise floor is 1.036×** (which retracted one published result); int8 activation quant has no fusion path in vLLM at all; and the ASM tree is 80% dead weight (48 useful of 242)
- [`docs/45-aiter-fast-path-survey.md`](docs/45-aiter-fast-path-survey.md) — **the AITER surface, surveyed and closed**: all 17 gates queried, five opened and measured. MoE runs but loses (0.977×); unified-attention is a dead flag, custom-AR is bolted at the engine, triton-GEMM is shape-allowlisted to gpt-oss dimensions, rope needs a second switch. No third easy win behind a flag
- [`docs/44-cdna1-mi100-portability.md`](docs/44-cdna1-mi100-portability.md) — what ports to **MI100 / gfx908**, decided by assembler probe rather than spec sheet: INT8 is identical to CDNA2 and worth **2× bf16** there (inverting `docs/20`'s CDNA2 finding), while the ASM work and 256k paged attention don't port at all. No MI100 was involved — a portability map, not a result
- [`docs/43-ck-int8-gemm-gfx90a.md`](docs/43-ck-int8-gemm-gfx90a.md) — **the biggest single win in this repo: 1.48× decode** (1.71× at TP=1). AITER's CK int8 GEMM was three gates from reachable on gfx90a — a missing `GFX_CU_NUM_MAP` entry, our own attention-only carve-out, and unregistered custom ops. Triton was launching 40 workgroups on a 104-CU card at M=1
- [`docs/42-decode-gap-probes.md`](docs/42-decode-gap-probes.md) — rounds 38–39: clocks, capture geometry and launch overhead all **ruled out**; async scheduling is default-on and worth **1.06–1.11× decode** on both versions (a "masked regression" claim was made and **retracted the same day** — sync paths are flat); rocprofv3 shows **99.9% kernel coverage** — the ~3× decode gap is in-kernel

### Benchmarks
- [`benchmarks/matrix/`](benchmarks/matrix/) — the quantization matrix: harness, sweeps, raw JSON, and [`REPRODUCE.md`](benchmarks/matrix/REPRODUCE.md) with exact tags, versions and CLI args
- [`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl) — **check this before proposing any optimization**: one queryable record per measured lead (round, factor, verdict, doc) across all rounds. `grep '<lead>' benchmarks/ledger.jsonl` answers "has this been tried?" — the prose index below did not, and MoE stride padding was re-proposed eight months after round 44 measured it at 0.961×. Companions: [`matrix/moe-configs-manifest.json`](benchmarks/matrix/moe-configs-manifest.json) (which tuned config is which, and why **none** is deployable) and [`matrix/probe_image_patches.sh`](benchmarks/matrix/probe_image_patches.sh) (which gfx90a patches are in which image — round 73 measured a flag on an image that lacked its gate)
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
