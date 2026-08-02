# Rounds 48–51: four nulls, and two earlier documents were wrong

> **THIS DOCUMENT IS ITSELF CORRECTED BY `docs/49`.** The round 51 sections below
> get the *mechanism* wrong. "The gfx90a port dropped the entire `fmoe_bf16_*`
> family, so the kernels for the dtype family we need were not ported" is not
> what happened. Disassembly shows all 8 ported gfx90a objects **do** perform
> bf16 matrix multiply — the gfx942 bf16 kernel and the gfx90a fp16 kernel are
> the same 3833 instructions, differing in exactly two substitutions, one of
> which is a pure per-architecture rename of the same MFMA. The real gap is a
> single instruction, `global_atomic_pk_add_bf16`, plus the separate fact that
> the filename the dispatcher constructs (`fmoe_b16.co` for bf16 input,
> `fmoe_int8_g1u0_subGU_*.co` for int8) simply is not on disk. The *conclusion*
> — no `fmoe` ASM reachable on gfx90a — stands. The reasoning does not. See
> `docs/49` for the instruction-level account.
>
> The rounds 48/49/50 sections are unaffected and remain accurate.

Four rounds, no throughput win. That is the honest headline. But three of the
four closed a question that was genuinely open, and the fourth refuted its own
premise before costing a 3.4-hour model load — which is the cheapest way an
experiment can fail.

Two documents in this repo are corrected below. `docs/45` claimed the AITER
surface was "surveyed and closed"; it was not, and the way it was wrong is
instructive. `docs/35` row 4 of the eligibility checklist is imprecise in a way
that would send a reader down exactly the dead end round 51 hit.

## Round 48 — RPC vs native multi-GPU: discarded, twice

Never produced a usable number. Both attempts are recorded because the first
failure was mine and the second is a hazard anyone re-running this will hit.

**Attempt 1** — the RPC arm GPU VM-faulted. Cause: `serve_llamacpp.sh` passes
`--device /dev/kfd --device /dev/dri`, which production's main `llama-server`
container does **not**. So llama.cpp enumerated 2 local ROCm devices *and* 2 RPC
backends — four backends over two physical cards — double-allocated them,
measured prefill at 0.658×, and died in `rocr VMFaultHandler` mid-longctx. That
0.658× is not a cost of RPC; it is a harness misconfiguration. Fixed by adding
`LLAMA_NO_LOCAL_GPU=1` to `serve_llamacpp.sh`.

**Attempt 2** — the *native* arm then VM-faulted too (`amdgpu [mmhub0] no-retry
page fault`), which it had no reason to. Most likely GPU state residue from
attempt 1. **All round 48 data is discarded.** The GPUs were verified healthy
afterwards.

The `LLAMA_NO_LOCAL_GPU=1` fix is kept — it is correct and needed by any future
RPC arm — but the RPC-vs-native question remains **open and unmeasured**. Do not
cite round 48 for anything.

## Round 49 — AITER Triton Gated-DeltaNet: 1.015×, null

`are_gdn_triton_kernels_available` gates an entire forward implementation for
Qwen3-Next GDN layers (`qwen_gdn_linear_attn.py:72` reads it once at import,
`:799` branches on it). On gfx90a the `@if_aiter_supported` decorator returned
None, so every GDN layer took the generic fallback. The kernels themselves are
Triton, not architecture-specific ASM, and all of them import cleanly here —
`on_mi3xx()` was the only obstacle.

Carved out in `configs/enable_aiter_gdn_and_moe_policy_gfx90a.py`, measured on
`t80-awq` (a genuine `Qwen3NextForCausalLM`) against the identical image minus
the carve-out:

| metric | generic | AITER GDN | ratio |
|---|---:|---:|---:|
| longctx decode | — | — | **1.015×** |

The decode noise bar for a single pair of arms is **1.036×** (`docs/46`). 1.015×
does not clear it. **Null**, and the carve-out is kept only because it is inert
and correct, not because it pays.

## Round 50 — MoE dispatch policy: null, and it upgrades round 42

`get_moe_dispatch_policy` is also `@if_aiter_supported`, so on gfx90a it returned
None and `VLLM_ROCM_AITER_MOE_DISPATCH_POLICY` **had no effect at all**. Round 42
therefore measured AITER MoE on policy 0 — the large-batch heuristic — against
the least-batched workload there is, without knowing it. Round 50 is the first
time the setting can do anything on this card.

| arm | longctx decode | vs MoE off |
|---|---:|---:|
| MoE off | 82.87 | — |
| policy 0 | 80.99 | 0.977× |
| policy 1 | 80.66 | 0.973× |
| policy 2 | 80.67 | 0.974× |

All three policies land within 0.4% of each other. The policy does nothing on
this model — see the next section for why, which is more interesting than the
null itself.

**The upgrade:** policy 0 reproduced round 42's **0.977× exactly**. A number that
reproduces to three digits across separate rounds and images is not noise. AITER
MoE is a **real ~2.3% decode regression** on this hardware, not an inconclusive
result, and `docs/45` should be read that way.

## Round 51 — bf16 full-ASM: premise refuted before it ran

Both arms failed at load:

```
sharded_state_loader.py: param_data = state_dict[key].data
KeyError: 'model.layers.0.mlp.experts.runner.gate.weight'
```

The snapshot **does** contain that key. `state_dict` here is the *model's* dict,
so it is the instantiated model that lacks the parameter. `experts.runner` is a
`MoERunner` submodule whose presence depends on `runner_cls` (`layer.py:142`).
So a `sharded_state` snapshot is **backend-configuration-specific**, not merely
engine-version-specific — sharper than `docs/34` currently states, and worth
recording on its own.

Regenerating the snapshot would have cost ~3.4 hours. Before spending it, the
premise was checked. It does not hold.

### The gfx90a port dropped the entire `fmoe_bf16_*` family

Counted directly in `aiter_meta/hsa/`:

| arch | `fmoe_bf16_noquant` csv | `fmoe_fp16_noquant` csv | `fmoe_bf16_*.co` | `fmoe_fp16_*.co` |
|---|---:|---:|---:|---:|
| **gfx90a** | **1 (header only)** | 3 | **0** | 4 |
| gfx942 | 3 | 3 | 202 | 188 |
| gfx950 | 3 | 3 | 224 | 202 |

Zero `fmoe_bf16_*` objects on gfx90a, against 202 and 224 on the MI300 targets.
Not just `noquant` — *every* quantization mode of the bf16 family is gone.

Stronger still: of the **23** fmoe config tables shipped for gfx90a, exactly
**two** contain any rows at all — `fmoe/silu/fmoe_fp16_noquant_g1u0_silu.csv`
and `fmoe/gelu/fmoe_fp16_noquant_g1u0_gelu.csv`, two rows each. The other 21 —
every int8 table, every fp8 table, every blockscale table, every bf16 table —
are header-only.

For comparison, gfx950's bf16 table is populated:

```
$ cat gfx950/fmoe/silu/fmoe_bf16_noquant_g1u0_silu.csv
knl_name,co_name,atm,vskip,smf,tg_num_perCU,ps,subGU_m,subGU_n
_ZN5aiter57fmoe_bf16_noquantBf16_g1u0_vs_atm_inlv_silu_1tg_ps_32x512E,...
_ZN5aiter54fmoe_bf16_noquantBf16_g1u0_vs_atm_inlv_silu_1tg_32x512E,...

$ cat gfx90a/fmoe/silu/fmoe_bf16_noquant_g1u0_silu.csv
knl_name,co_name,atm,vskip,smf,tg_num_perCU,ps,subGU_m,subGU_n
```

`t35-bf16` is `bfloat16` throughout. There is **no checkpoint on this hardware
that can reach `fmoe` ASM**, because the kernels for the dtype family it would
need were not ported. The 3.4-hour load would have bought a guaranteed dead end.

### And this re-explains rounds 42–43 correctly

The prior explanation — "all 8 objects are `noquant`, so int8 experts can't use
them" — was true but not the operative constraint. The W8A8 checkpoint runs
**bf16 activations** with int8 weights, so it wants
`fmoe_bf16_pertokenInt8_g1u0_silu`, whose gfx90a table is header-only like every
other bf16 table. AITER MoE could never have reached an ASM kernel on this card
regardless of expert quantization. That is why round 43's kernel diff still
showed `fused_moe_kernel` running with `module_moe_asm` loaded.

**Naming caveat, stated because it is easy to get wrong.** These objects carry
two dtype fields: `fmoe_{A}_noquant{B}`. They vary independently — gfx950 ships
`bf16_noquantBf16`, `fp16_noquantFp16`, *and* `fp16_noquantBf16`. The config
table is keyed by `A`. So the presence of two `fp16_noquantBf16` objects on
gfx90a does **not** mean bf16 is served here; they are reachable only through
the fp16-family table. Which field is activation vs weight dtype was not traced
to the dispatcher, and the conclusion above does not depend on it — the
asymmetry in the table is decisive either way.

## Correction 1 — `docs/45` was not "closed"

`docs/45` concluded: *"The AITER surface is now surveyed and closed: of 17
gates…"*. That survey enumerated `is_*_enabled` gates. The pattern is
incomplete. Enumerating every `@if_aiter_supported`-decorated method finds
**23**, and the six that do not match `is_*_enabled` were never examined:

| method | status |
|---|---|
| `are_gdn_triton_kernels_available` | **gated real functionality** — round 49 |
| `get_moe_dispatch_policy` | **gated real functionality** — round 50 |
| `is_enabled` | carved out by `enable_aiter_master_gate_gfx90a.py` |
| `register_ops_once` | carved out by `enable_aiter_ck_gemm_gfx90a.py` |
| `fused_moe_supports_gate_mode` | no consumers in vLLM; dead |
| `fuse_sigmoid_in_kernel` | shared-experts router only; no such model here |
| `topk_softmax_supports_fused_sigmoid` | no consumers in vLLM; dead |

Two of the six gated real functionality. Both were measured; both are nulls. So
the *conclusion* of `docs/45` survives — there is no third easy win behind a
flag — but its *basis* did not, and it was stated with more confidence than the
method supported. **A survey is only as complete as its search pattern.**

The surface is now closed by measurement rather than by pattern-matching.

## Correction 2 — `docs/35`'s eligibility checklist, row 4

The checklist row reads:

| MoE expert weights | **unquantized bf16/fp16** | `fmoe` is `noquant` only |

On gfx90a this is wrong in a way that matters: it implies a bf16 MoE checkpoint
qualifies, and `docs/35` goes on to name `t35-bf16` as satisfying "all five".
It does not. The correct row is:

| MoE expert weights | **unquantized fp16-family only** | of 23 gfx90a `fmoe` tables, only the two `fp16_noquant` ones have rows; the entire `fmoe_bf16_*` family is absent from the port |

`t35-bf16` satisfies rows 1–3 (head_dim 128, GQA ratio 8, bf16 attention) and
fails row 4 — for a reason the original row does not state. No model on this
card clears all five.

## Status

| round | question | result |
|---|---|---|
| 48 | does the RPC hop cost anything? | **discarded** — both attempts VM-faulted; still open |
| 49 | AITER Triton GDN on Qwen3-Next | 1.015×, under the 1.036× bar — **null** |
| 50 | is 0.977× a dispatch-policy artifact? | no; all policies within 0.4% — **null**, and 0.977× is confirmed real |
| 51 | can any model reach fmha + pa + fmoe at once? | **no** — premise refuted; `fmoe_bf16_*` absent from the port |

Artifacts: `benchmarks/matrix/round4{8,9}_*.sh`, `round5{0,1}_*.sh`,
`configs/enable_aiter_gdn_and_moe_policy_gfx90a.py`,
`configs/Dockerfile.llama-rpc`.
