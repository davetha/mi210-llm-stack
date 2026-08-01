# What of this ports to MI100 (gfx908 / CDNA1)

**Date**: 2026-08-01 · Method: LLVM assembler probes (`llvm-mc -mcpu=gfx908`)
and source-gate reading, run from the gfx90a image. **No MI100 was involved.**

Written because the question came up and the answers are checkable without the
hardware — every ISA claim below is an assembler verdict, not a spec sheet.
**Nothing here has been executed on a gfx908 card**, here or anywhere in this
repo. Treat it as a portability map, not a result.

## The one-line answer

**INT8 W8A8 ports and is worth *more* there than here. The ASM work and the
256k paged attention do not port at all.** The CK int8 GEMM — this repo's
largest win — is a plausible port and the only one worth real effort.

## ISA: measured, not assumed

`llvm-mc` accepts (`ok`) or rejects each instruction for a given `-mcpu`:

| instruction | gfx908 | gfx90a |
|---|---|---|
| `v_mfma_i32_32x32x8i8` | ok | ok |
| `v_mfma_i32_16x16x16i8` | ok | ok |
| `v_dot4_i32_i8` | ok | ok |
| `v_mfma_i32_32x32x16_i8` (gfx942 wide INT8) | rejected | rejected |
| `v_mfma_f32_32x32x8f16` | ok | ok |
| `v_mfma_f32_32x32x8bf16_1k` | **rejected** | ok |
| `v_mfma_f32_32x32x4bf16` | ok | (older form) |

Two conclusions follow directly.

**INT8 is identical on both.** Same three instructions, same absence of the
gfx942 wide form. W8A8 arithmetic on MI100 is native, not emulated.

**bf16 is half-depth on CDNA1.** gfx908 has bf16 matrix ops, but only the older
forms — `32x32x4bf16`, K=4 — where gfx90a has the `_1k` variants at K=8. fp16
is K=8 on both.

### The consequence that inverts a docs/20 finding

`docs/20` established that **on CDNA2 INT8 and BF16 share the same 181 peak**,
so INT8 buys no arithmetic throughput here and its only advantage is bytes.
That symmetry **does not hold on CDNA1**: int8 stays at K=8 while bf16 drops to
K=4, so INT8 has roughly a 2× per-instruction MAC advantage over BF16 on MI100.

So W8A8 is a *better* proposition on an MI100 than on an MI210 — the opposite
of the conclusion this repo reached for its own hardware. Same reasoning says
**fp16 beats bf16** on that card (K=8 vs K=4), subject to fp16's narrower
dynamic range on bf16-trained checkpoints.

Caveat carried from `docs/24`: on MoE checkpoints the experts run
`int8_w8a16` — weights int8, activations bf16 — so the INT8 matrix cores are
not engaged for expert GEMMs at all. The 2× advantage applies to dense linears
(qkv, o_proj), which makes *dense* W8A8 the stronger case on CDNA1.

## Software gates: gfx908 is excluded even where it qualifies

- **`vllm/platforms/rocm.py`**: `_ON_GFX9 = ["gfx90a", "gfx942", "gfx950"]`.
  gfx908 is a literal gfx9 chip and is **not in it**. So the ROCm custom paged
  attention never dispatches on MI100, and `extend_rocm_pa_256k_gfx9.py` widens
  a gate that card never reaches. The 13× depth result does not port.
- **`configs/enable_int8_moe_rocm.py`** admits
  `is_rocm() and on_gfx9()` — so it would leave MI100 refusing MoE W8A8 exactly
  as before. Its own rationale is that the refusal is not a hardware limit
  because CDNA has INT8 MFMA, and the table above shows that premise holds on
  gfx908. Widening the predicate to include gfx908 is justified, not a hack.
- **AITER** knows the architecture — `GFX_MAP` has `2: "gfx908"` — but
  `GFX_CU_NUM_MAP` omits it, the same defect `docs/43` §2 hit for gfx90a. MI100
  is **120 CUs**.

## The ASM work does not port

`repatch_gfx942_to_gfx90a.py` substitutes gfx942 bf16 MFMA onto gfx90a's `_1k`
forms, which gfx908 **rejects** — and the substitution is not merely a rename,
since the K depths differ (8 vs 4), so the data layout differs too. Separately,
**539 of 1,422** kernels already fail on `global_atomic_pk_add_bf16`, which the
assembler rejects on *gfx90a*; an older core will not have it either. Our yield
was 242/1,422 portable; expect far worse.

The *tool* ports even though its output does not: it re-assembles every
instruction for the target and refuses to emit anything that will not assemble,
so pointing it at gfx908 answers the question empirically instead of by
argument. It currently hardcodes `-mcpu=gfx90a` in two places and would need a
fresh substitution table.

## What a first run should look like

Do **not** build this repo's image: it is `GPU_ARCHS=gfx90a` /
`PYTORCH_ROCM_ARCH=gfx90a` from line 75, its 242 ASM objects are re-assembled
for gfx90a, and `verify-gfx90a` check 1 aborts unless `rocminfo` reports
gfx90a. It fails loudly, which is the correct behaviour, but it fails.

Stock image, dense W8A8, minimal flags, `--tensor-parallel-size 2` (32 GB per
card), no AITER env flags (inert on gfx908, as on gfx90a without our patches),
and **not** `--enforce-eager` (measured 3.5× slower with 82% of decode in gaps,
`docs/42`). Dense matters: the `No Int8 MoE backend` failure is MoE-specific,
and vLLM's dense int8 path (`TritonInt8ScaledMMLinearKernel.is_supported`) gates
only on `current_platform.is_cuda_alike()` — no arch or version check at all.

Unresolved and gating everything: whether current ROCm/PyTorch wheels still ship
gfx908 kernels. `torch.cuda.get_arch_list()` answers it in two minutes.

## If they want the real win

The CK int8 GEMM (`docs/43`, 1.48× decode here) is the port worth attempting:
it is C++ template code rather than hand-written assembly, CK supports gfx908,
and the defect it fixes — Triton emitting 40 workgroups for qkv_proj and 16 for
o_proj at M=1 — is **worse** on MI100's 120 CUs than on the 104 CUs measured
here. It needs the same three carve-outs as `enable_aiter_ck_gemm_gfx90a.py`
with `"gfx908": 120` and gfx908 added to the predicates.

`configs/enable_sharded_state_tp_check.py` is arch-independent and ports as-is.
