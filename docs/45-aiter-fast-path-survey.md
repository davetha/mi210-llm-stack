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

## Round 43: the `topksoftmax` ASM objects are unreachable, and the reason is not the gate

`docs/37` dismissed the 22 `topksoftmax` ASM objects as *"untested, and not
worth testing: MoE routing is <1% of MoE work."* Two things about that were
worth checking: the estimate is wrong — round 41 measures `topkGating` at
**392.7 ms = 4.9% of decode** — and unlike almost every other family, **22 of
22 objects ported to gfx90a**. Fully-ported kernels against a real 5% target,
dismissed on a bad number, looked like the best remaining ASM lead.

It is not, and the profile says why. First, the dispatch is **not separately
gated**: `fused_topk_router.py:107` reads
`use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()`, so topk rides on
`VLLM_ROCM_USE_AITER_MOE` and round 42's MoE arm had already exercised it —
that 0.977× was the *combined* effect, not `fmoe` alone.

Round 43 profiled `AITER_MOE` off vs on at TP=1 (`round43_moe_profile.sh`),
same warm-then-profile harness, and diffed the decode-window kernel tables:

| kernel | MoE off | MoE on | delta |
|---|---:|---:|---:|
| `vllm::moe::topkGating<8,128,4,16,64,...>` | 388.5 ms | **gone** | −388.5 |
| `vllm::moe::topkGatingSoftmax<bf16,16,128,8,32,...>` | **new** | 386.8 ms | +386.8 |
| `fused_moe_kernel` | 1973.9 | 2188.9 | **+215.0** |
| `__amd_rocclr_copyBuffer` | 197.2 | 34.5 | −162.7 |
| `dynamic_scaled_int8_quant` | 521.6 | 458.0 | −63.6 |
| **total decode window** | **7980.0** | **7982.7** | **+2.7** |

**The topk kernel swaps for a different kernel at the same price — and both are
`vllm::moe::` kernels.** Neither is an AITER ASM kernel. Turning the flag on
selects `topkGatingSoftmax` instead of `topkGating`, 386.8 vs 388.5 ms, a 0.4%
difference. The 22 ported ASM objects are never reached through vLLM's path at
all.

`fused_moe_kernel` also does not go away — it gets **10.9% slower**. So
`module_moe_asm` loading (round 42) does not mean the ASM kernels took over the
expert GEMMs; consistent with `docs/37`'s note that all 8 ported `fmoe` objects
are `noquant{Fp16,Bf16}` and cannot consume int8 weights.

Net: **+2.7 ms across an 8 s window, and TP=1 wall time 11.88 s → 11.87 s.** A
wash at TP=1, the mild 0.977× regression at TP=2.

**Verdict.** `docs/37`'s "<1%" estimate was wrong and its "not worth testing"
conclusion was right, for a reason nobody had established: the flag does not
route topk to AITER ASM. Closing this properly needed the profile, not the
arithmetic. The MoE cluster's 32% of decode is not reachable through any AITER
flag.

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
| MoE cluster | 2560 | 32% | **Closed via AITER** (rounds 42–43): the flag is a wash at TP=1 and 0.977× at TP=2, and does not reach the ASM kernels. One untested idea remains — see below |
| `paged_attention_ll4mi` | 1045 | 13.1% | `pa_fwd_asm` has **no call site** in vLLM (`docs/37` §5); wiring one is real work, and `docs/18` shows the kernels are numerically exact |
| `wvSplitK` | 680 | 8.5% | no AITER int8 equivalent |
| `dynamic_scaled_int8_quant` | 498 | 6.2% | separate kernel per GEMM; `fuse_norm_quant` describes itself as fp8-only, so int8 activation quant may have no fusion path at all — **unexamined, and the most concrete remaining lead** |

## The one untested MoE idea: stride padding is quantized-path-blind

Found while researching whether a load-time "conversion layer" could reshape
models onto fast paths (verdict: no — the shape allowlist that motivated it is
an `if` statement in vLLM we can edit, and padding to satisfy it would add
bytes in exactly the bandwidth-bound decode regime the path targets, on top of
an RMSNorm-denominator hazard for any hidden-dim pad).

The useful thing that fell out of it is a real gap. vLLM has a ROCm MoE
**stride padding** optimization — pad by 256 B then slice back, leaving the
logical shape and values identical while bumping the row stride to de-alias
memory banks:

```python
num_pad = 256 // weight.element_size()
weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
```

`VLLM_ROCM_MOE_PADDING` defaults **True** (`envs.py:146`), and upstream
reported up to 10% on Mixtral. But grepping our tree, `VLLM_ROCM_MOE_PADDING`
appears in exactly three files: `envs.py`,
`fused_moe/oracle/unquantized.py`, and `unquantized_fused_moe_method.py`.
**It is nowhere in the quantized path.** A W8A8 model never gets it — not
because a stray `.contiguous()` undoes it (vLLM guards against that at
`oracle/unquantized.py:353`), but because it is never applied.

The guard requires `weight.stride(-2) * element_size() % 512 == 0`. For this
model at int8, `w13` is `[E, 2*768, 2048]` — row stride 2048 B, divisible by
512, **qualifies**. `w2` is `[E, 2048, 768]` — 768 B, does not.

So roughly half the expert weights are eligible for a change that costs 256 B
of allocation per row, carries no correctness risk, and targets
`fused_moe_kernel` — still the largest single decode kernel at 24.4% even after
everything above. **Untested.** It is the only MoE idea left that is not
"write a kernel", and unlike the AITER flags it has upstream evidence behind
it.
