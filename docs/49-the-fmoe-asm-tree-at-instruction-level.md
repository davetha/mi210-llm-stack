# The `fmoe` ASM tree at instruction level — and why `docs/48` was wrong

`docs/48` concluded that no checkpoint on this card can reach `fmoe` ASM because
"the kernels for the dtype family it would need were not ported." That reasoning
is **wrong**, and this document replaces it. The conclusion — no `fmoe` ASM on
gfx90a today — survives, but for a completely different and much more specific
reason, and the corrected picture opens one lead the old one closed.

Everything here is from disassembly and the LLVM assembler, not inference.

## 1. The gfx942 bf16 kernel and the gfx90a fp16 kernel are the same kernel

Compare `gfx942/fmoe/silu/fmoe_bf16_noquantBf16_g1u0_atm_inlv_silu_1tg_32x512.co`
against `gfx90a/fmoe/silu/fmoe_fp16_noquantFp16_g1u0_atm_inlv_silu_1tg_32x512.co`:

- both are **exactly 3833 instructions**
- instructions sit at **identical addresses**
- register operands are **identical**
- the complete mnemonic diff is **1072 positions, two substitutions, nothing else**:

```
1024x  v_mfma_f32_16x16x16_bf16    ->  v_mfma_f32_16x16x16bf16_1k    (D3E1 -> D3E7)
  48x  global_atomic_pk_add_bf16   ->  global_atomic_pk_add_f16      (DD48 -> DD38)
```

Operand-level confirmation of the second one — same registers, same address, one
byte of opcode:

```
gfx942:  global_atomic_pk_add_bf16 v34, v10, s[8:9]   // 55C8: DD488000 00080A22
gfx90a:  global_atomic_pk_add_f16  v34, v10, s[8:9]   // 55C8: DD388000 00080A22
```

## 2. The MFMA difference is a rename, not a capability

Fed to `llvm-mc`:

| instruction | gfx90a | gfx942 |
|---|---|---|
| `v_mfma_f32_16x16x16_bf16` | REJECTS — *not supported on this GPU* | ACCEPTS `[...,0xe1,0xd3,...]` |
| `v_mfma_f32_16x16x16bf16_1k` | **ACCEPTS** `[...,0xe7,0xd3,...]` | **ACCEPTS** `[...,0xe1,0xd3,...]` |

The `_1k` spelling assembles on gfx942 to gfx942's own encoding. LLVM treats them
as **the same operation under two per-architecture names**. Same M/N/K (16×16×16),
same dtype (bf16), same operands.

So there is no MFMA shape problem, no K mismatch, and nothing to "shim" from
32×32 to 16×16 — the kernel never uses a 32×32 MFMA. **All 8 ported gfx90a `fmoe`
objects perform bf16 matrix multiply**, including every one whose filename says
`fp16`.

This also decodes the naming. `fmoe_{A}_noquant{B}` is **A = output / atomic
accumulation dtype, B = weight dtype**. `docs/48` flagged this as untraced; it is
now traced, and it is what the whole conclusion hung on.

## 3. The one real silicon gap

```
global_atomic_pk_add_bf16     gfx90a  REJECTS  -- not supported on this GPU
v_mfma_f32_16x16x32_fp8_fp8   gfx90a  REJECTS  -- not supported on this GPU
```

CDNA2 has no packed-bf16 atomic add and no FP8 MFMA. Of the 47 distinct mnemonics
in the gfx942 bf16 MoE kernel, **45 exist on gfx90a**; one is a rename and one is
genuinely absent.

AITER states this itself. `aiter/fused_moe.py` ~line 70:

> *"...accumulate via `global_atomic_pk_add_bf16`, so moe_sorting is a
> pass-through for them."*

and line 716:

```python
assert not metadata.flat or get_gfx() in ("gfx942", "gfx950"), \
    f"FLAT fmoe asm kernels require gfx942/gfx950; got {get_gfx()}. "
```

**That arch check is load-bearing.** It is not conservatism about CDNA2 — the
FLAT kernels are built around an instruction the hardware cannot encode. Stripping
it, in the style of the `configs/enable_aiter_*.py` carve-outs, would move the
failure later rather than remove it.

## 4. Why every MoE round this year failed — one mechanism

The live ASM dispatcher (`csrc/cpp_itfs/moe/asm_moe.py`, the `g1u0` branch) keys
on **input dtype** and constructs a filename:

```python
if   input_dtype == "__half":          -> "fmoe_f16.co"
elif input_dtype == "__hip_bfloat16":  -> "fmoe_b16.co"
elif input_dtype == "uint8_t":         -> f"fmoe/silu/fmoe_int8_g1u0_subGU_{tile}.co"
```

Against what gfx90a actually ships:

| model | object requested | present | round |
|---|---|:---:|---|
| fp16 unquantized | `fmoe_f16.co` | **yes** | never tried |
| bf16 unquantized | `fmoe_b16.co` | **no** | 51 |
| int8 / W8A8 | `fmoe_int8_g1u0_subGU_*.co` | **no** (zero) | 42, 43, 50 |

Rounds 42, 43, 50 and 51 all failed for the same reason: **the filename the
dispatcher builds does not exist on disk.** Not "the objects are noquant" (`docs/45`),
not "the bf16 family wasn't ported" (`docs/48`). Those descriptions were true
statements about the inventory that misidentified the mechanism.

## 5. The stranded asset

The 8 `fmoe_fp16_noquant*` objects are FLAT-family kernels (`atm_inlv` = atomic
interleaved) that accumulate with `global_atomic_pk_add_f16` — the fp16 variant
CDNA2 **does** have. AMD built fp16-atomic FLAT kernels for this card and shipped
them. They are then stranded twice:

1. the `metadata.flat` assert above gates the FLAT path to gfx942/gfx950, and
2. **nothing in the build can load them** — no `.so` contains the string
   `noquant`, `cfg_fmoe_*noquant*` appears nowhere in the C++ tree, and the
   generated `asm_fmoe_configs.hpp` that would define those config maps does not
   exist in this build

Their two CSV tables (`fmoe_fp16_noquant_g1u0_{silu,gelu}.csv`, two rows each)
are the only populated `fmoe` tables on gfx90a, and they index objects no code
path reaches.

Unlocking them would require stripping the assert **and** generating the missing
config maps **and** accepting fp16 output accumulation (fp16 tops out at ~65504;
bf16 carries fp32 range, so expert-output accumulation has a real overflow risk).
That is source and build work, not binary patching — the tractable category — but
it is three things, not one.

## 6. The finding that matters most for performance

> **CORRECTED 2026-08-02 — see `docs/50` §1b.** The call is there; the kernels
> are not. Running the CK 2-stage tuner on gfx90a returns `us1 = us2 = us = -1`
> for all 20 shapes with *"stage1 and stage2 should be valid together"* — **zero
> valid CK 2-stage kernel pairs exist for int8 MoE on this card.** Round 55
> independently shows Triton `fused_moe_kernel_gptq_awq` doing the work. So the
> MoE path here is **Triton**: the ASM tree (this document) and the CK 2-stage
> path are both unavailable to it. The paragraph below infers the backend from a
> call site rather than from what actually executes — the same mistake this
> document criticises elsewhere.

vLLM's AITER MoE path goes through `aiter.fused_moe`, which calls
**`ck_moe_stage1` / `ck_moe_stage2_fwd`** — Composable Kernel, compiled from
source. **Not the ASM tree at all.**

So the 0.977× MoE regression reproduced in rounds 42 and 50 was never a fact
about these ASM objects. If MoE is worth chasing further, the lever is CK
instantiation and tuning for gfx90a — the same category that produced the 1.48×
int8 GEMM win (`docs/43`) — and the entire ASM investigation is orthogonal to it.

## 7. On binary patching, since it was the original question

Mechanically, converting the gfx942 bf16 object to gfx90a is trivial: 1024 MFMA
opcodes `D3E1`→`D3E7`, 48 atomics `DD48`→`DD38`, plus the ELF target metadata.
Every operand and address already matches.

It is also **pointless**: performing it reproduces, byte for byte, a file AMD
already ships as `fmoe_fp16_noquantFp16_...` — which is one of the 8 orphans
nothing can load. The patching path dead-ends on the absence of a consumer, not
on the ISA.

## Corrections this document makes

| document | claim | status |
|---|---|---|
| `docs/45` | "`fmoe` is `noquant` only" as the reason int8 can't reach ASM | true of the inventory, wrong mechanism — the int8 objects are simply absent |
| `docs/48` | "the kernels for the dtype family it would need were not ported" | **wrong** — bf16 compute is present in all 8 ported objects; the gap is the bf16 *atomic*, and separately the missing `fmoe_b16.co` |
| `docs/48` | naming field semantics "not traced to the dispatcher" | now traced: A = output/atomic dtype, B = weight dtype |
| `docs/35` | correction banner added by `docs/48` | its reasoning is superseded by §2 above |

## What is genuinely closed

- No `fmoe` ASM kernel is reachable on gfx90a today, for any checkpoint dtype.
- FP8 compute is CDNA3-only. Confirmed at the assembler.
- Binary patching the ASM tree cannot change either of the above.

## What is genuinely open

- CK MoE instantiation/tuning for gfx90a — the path vLLM actually uses.
- Whether the 8 stranded FLAT objects can be given a config map and an fp16
  output path, and whether that survives numerically.
