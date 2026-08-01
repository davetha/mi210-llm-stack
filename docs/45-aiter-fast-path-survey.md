# Round 42: the remaining AITER fast paths, surveyed and closed

**Date**: 2026-08-01 · Model: Qwen3-30B-A3B W8A8 · Image: `vllm-mi210:aiterops`
· TP=2, P2P on, AITER FA on, `LINEAR=1` (the CK GEMM) held on in every arm ·
Patch: `configs/enable_aiter_ops_gfx90a.py` · Script:
`benchmarks/matrix/round42_aiter_paths.sh` · Results:
`benchmarks/matrix/results/rd42-*.json`

`docs/43` opened the CK int8 GEMM for **1.48× decode** and, incidentally, fixed
op *registration* — before it, `register_ops_once()` was `on_mi3xx()`-gated and
not one AITER custom op existed on gfx90a. That raised an obvious question:
**what else is behind those gates?**

**Answer: nothing worth having.** Of five capabilities opened and measured, one
runs and is a mild regression, and four cannot engage at all — each for a
different, specific reason. This document records those reasons so the next
person does not re-open them.

## The survey

All 17 `is_*_enabled` gates, queried on this card: `linear` and `mha` return
`True` (carved out previously), `shuffle_kv_cache` returns `False` (carved out,
flag off), and **14 return `None`** — the `@if_aiter_supported` → `on_mi3xx()`
short-circuit.

Every underlying AITER symbol imports cleanly on gfx90a:
`aiter.fused_moe.fused_moe`, `fused_topk`, `asm_moe_tkw1`, `topk_softmax`,
`grouped_topk`, `topk_sigmoid`, `dynamic_per_token_scaled_quant`,
`dynamic_per_tensor_quant`, `fused_add_rmsnorm_pad`,
`fused_qk_rope_reshape_and_cache`, `pa_fwd_asm`, `hipb_mm` — 12/12 OK. **The
plumbing exists; the gate is the only blocker.** That is what made this survey
worth running, and also what made it misleading: importing is not running.

Targets, from the round-41 decode profile (7,979 ms busy in an 8 s window):

| kernel | ms | % decode |
|---|---:|---:|
| `fused_moe_kernel` | 1947 | 24.4% |
| `paged_attention_ll4mi` | 1045 | 13.1% |
| `wvSplitK` (2 variants) | 680 | 8.5% |
| `dynamic_scaled_int8_quant` | 498 | 6.2% |
| `topkGating` | 393 | 4.9% |
| `moe_sum_vec` | 220 | 2.8% |

## Result

One image, one flag per arm, each flag defaulting off so a broken kernel is
isolated to its own arm.

| arm | flag | decode | vs base | prefill | engaged? |
|---|---|---:|---:|---:|---|
| baseline | — | 82.92 | 1.000× | 8550 | — |
| moe | `MOE` | 80.99 | **0.977×** | 8604 | **YES** — `module_moe_asm` |
| rope | `TRITON_ROPE` | 82.11 | 0.990× | 8547 | no |
| unifattn | `UNIFIED_ATTENTION` | 82.78 | 0.998× | 8555 | no |
| customar | `CUSTOM_AR` | 81.42 | 0.982× | 8546 | no |
| tritongemm | `TRITON_GEMM` | 83.29 | 1.004× | 8566 | no |

**Baseline reproduces `docs/43` on an independently built image** — 82.92 vs
82.48 decode, 8550 vs 8555 prefill — so the 1.48× CK result replicates.

## Why each one did not engage

The round's own assertion (does this arm load an AITER JIT module baseline did
not?) flagged four arms as "changed nothing". That assertion is **sound for
ASM/CK kernels and blind for Triton ones** — `module_*` lines come from AITER's
C++ JIT, and a Triton kernel compiles through Triton's own JIT and leaves no
such line. So four "no new module" verdicts needed individual attribution
before any of their numbers could be believed. All four resolved:

- **`UNIFIED_ATTENTION` — dead flag.** `is_triton_unified_attn_enabled` has
  **zero consumers** anywhere in vLLM 0.26.1rc0 outside its own definition.
  Nothing reads it, so it cannot change behaviour on any architecture.
- **`CUSTOM_AR` — inert at the engine level.** The arm's own config logs
  `disable_custom_all_reduce=True`, and the serverlog contains zero
  `AiterCustomAllreduce` mentions. The gate opens a door the engine has already
  bolted. Testing it for real needs that engine setting flipped too.
- **`TRITON_GEMM` — shape allowlist, not an arch gate.**
  `use_aiter_triton_gemm()` (`layers/utils.py`) ends in a hardcoded list:
  `(m==5120 and k==2880) or (m==2880 and k==4096) or (m==128 and k==2880) or
  (m==640 and k==2880) or (m==2880 and k==512)`. Those are gpt-oss-shaped
  dimensions. Measured `False` for every Qwen3-30B-A3B shape tried. Also note
  `is_fp8_fnuz()` is **False** on this card, so that clause — which reads like
  the obvious blocker — is *not* what stops it.
- **`TRITON_ROPE` — needs a second, unrelated switch.**
  `base.py:51` sets `self.use_aiter = self.enabled() and
  is_triton_rotary_embed_enabled()`. The gate is `True` and `use_aiter` is
  still **`False`**, because `enabled()` is a `CustomOp` check against the
  compilation config, which defaults to `custom_ops: ['none']`. Engaging it
  would additionally require `-cc.custom_ops='+rotary_embedding'`. Not chased:
  rope does not appear in the decode top-10, so the ceiling is small.

## MoE: it runs, it is just not faster

The one arm that genuinely engaged. `module_moe_asm` loaded, the correctness
probe passed, and decode came out at **0.977×** — a mild regression sitting at
the ±2–3% noise floor, with prefill flat at 1.006×.

This **corrects `docs/37`**, which recorded `fmoe` as "the kernel declines the
device." It does not decline; it runs and serves correct output. The rest of
that entry is the likely explanation for why it does not win: all 8 gfx90a
`fmoe` objects are `noquant{Fp16,Bf16}` and cannot consume quantized weights,
so on a W8A8 checkpoint the ASM path cannot be doing the expert GEMMs in int8.

**The 32% of decode sitting in the MoE cluster** (`fused_moe_kernel` +
`topkGating` + `moe_sum_vec` = 2,560 ms) is therefore *not* reachable by
flipping this flag. It remains the largest single opportunity on the card and
now needs a different idea than "use AITER's version."

## Also closed, without an arm

- **`hipb_mm` cannot replace `wvSplitK`** (8.5% of decode). Its registered op is
  `rocm_aiter_hipb_mm_fp8` — an FP8 path — and `is_linear_hipbmm_enabled`
  carries a **second** `on_mi3xx()` inside its own body, so a decorator swap
  would not be enough even if FP8 existed here.
- **MLA, FP4/FP8 BMM, ASM FP4 dynamic quant** — no MLA model here, and no FP8
  or FP4 compute on CDNA2 (`docs/27`).

## What this leaves

The AITER surface is now **surveyed and closed**: of 17 gates, two are live and
winning (`mha` → 1.19–1.33× prefill, `linear` → 1.48× decode), one runs and
loses (`moe`), four are provably inert, and the rest are FP8/FP4/MLA paths this
architecture cannot use. There is no third easy win hiding behind a flag.

Remaining decode targets, and none of them is an AITER flag:

| target | ms | % | note |
|---|---:|---:|---|
| MoE cluster | 2560 | 32% | AITER's version measured 0.977×; needs a different approach |
| `paged_attention_ll4mi` | 1045 | 13.1% | `pa_fwd_asm` has **no call site** in vLLM (`docs/37` §5); wiring one is real work, and `docs/18` shows the kernels are numerically exact |
| `wvSplitK` | 680 | 8.5% | no AITER int8 equivalent |
| `dynamic_scaled_int8_quant` | 498 | 6.2% | separate kernel per GEMM; `fuse_norm_quant` describes itself as fp8-only, so int8 activation quant may have no fusion path at all — **unexamined, and the most concrete remaining lead** |
