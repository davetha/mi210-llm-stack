# 29 — ASM operator port matrix, and ASM flash attention working on gfx90a

**Date**: 2026-07-27
**Status**: done, validated

## What changed

Established the complete, evidence-backed answer to "which AITER ASM operators
can run on an MI210", and used it to bring up ASM flash-attention forward — an
operator the repo previously recorded as impossible on this hardware.

### 1. The port matrix (`configs/classify_gfx942_kernels.py`, new)

Disassembles all 1,422 gfx942 ASM kernels and classifies each by the hardware
capability blocking it, judging assembler-validity per distinct *instruction
text* rather than per mnemonic (a per-mnemonic verdict wrongly condemns
`flat_load_dword` because one use carries gfx942 `sc0 sc1` cache modifiers).

**242 portable, 1,180 blocked.** Blockers: FP8 (647), packed-bf16 atomics (539),
INT8 (482), 64-bit VALU (3). The split falls exactly along data type — every
bf16 attention kernel is portable, every FP8/INT8 one is not.

242 is a ceiling, not a milestone. The tool explicitly hunts for kernels blocked
*only* by cosmetic differences, since those would be recoverable; there are none.
51 candidates initially looked recoverable (including all 22 `bf16gemm`), but 49
need `global_atomic_pk_add_bf16` — absent on CDNA2 — and 2 need 64-bit VALU ops
that expand to multiple instructions and so cannot be patched in place. Three
independent methods agree on the identical 242-file set.

### 2. Cleaned the installed kernel tree

`hsa/gfx90a/` held 1,251 `.co` files, of which **1,147 could not run on the
GPU**. Replaced with the generated 242. `hsa/gfx90a` is now a build artifact —
regenerate it with `repatch_gfx942_to_gfx90a.py`, never hand-edit it.
`pa_fwd_asm` re-validated after the swap: 48/48 pass.

### 3. ASM flash attention enabled (`configs/enable_gfx90a_asm_paths.py`, new)

All 48 bf16 `fmha_v3_fwd` kernels are portable; only the 8 FP8 ones are not.
What blocked them was six architecture-string comparisons, not hardware —
`aiter/ops/mha.py:1674` even calls them *"hand-written gfx9 ASM"*, and gfx90a is
gfx9.

The load-bearing site is an early-out in `cpp_itfs/mha_fwd.cu` written in the
**negated** form `(arch_id != "gfx942") && (arch_id != "gfx950")`, which a grep
for `arch_id == "gfx942"` misses entirely. It returns `-1` before the kernel
lookup runs, surfacing as `RuntimeError: invalid argument for fmha_fwd`.

The patch deliberately does **not** widen `is_fmha_v3_fp8()`, which contains a
byte-identical gate line — widening it would route FP8 tensors at a GPU with no
FP8 ALU. Anchoring is on the preceding comment, and every rewrite asserts on its
expected match count so an upstream change fails loudly instead of silently
leaving the fast path off.

### 4. Validation (`tests/test_fmha_v3_fwd_asm_gfx90a.py`, new)

48 configurations — head dim 128 and 192×128, causal and non-causal, GQA ratios
1/8/8, seqlen 129–1024, batch 1–4. **48/48 pass**, rel_rms 1.1e-4…7.6e-4, 100%
element match. The test captures the loader log at the file-descriptor level to
prove the ASM kernel actually ran; `--require-asm` makes its absence a failure,
because correctness alone cannot distinguish ASM from the CK fallback.

### 5. Repo hygiene

Added `.gitignore`. Notably it excludes `*.co` (`hsa/gfx90a` is generated) and
`gpucore.*.gpu` — ROCm writes a full ~60 GB HBM snapshot per unhandled GPU
exception, and 426 GB of them had accumulated on the host. Set
`HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` to prevent it.

## Why it matters

Doc 17 concluded gfx942 ASM could not run on gfx90a. Doc 18 disproved that for
one operator. This change generalises the result and puts a defensible number on
it: everything CDNA2 can arithmetically do, it can do in ASM.

## Caveat carried forward

`V_MFMA_F32_16X16X16*` is 4-pass on gfx942 and 8-pass on gfx90a, so ported code
can be short on wait states at an MFMA→consumer edge. Static verification cannot
see this — the instructions re-encode perfectly while the *schedule* is unsound.
`pa_fwd` and `fmha_v3_fwd` both pass with no sign of it, but that must not be
generalised: every newly enabled ASM family needs its own numerical validation.

## Also found

The CK flash-attention fallback does not build on this container
(`ck_tile/host/device_prop.hpp` redefinition errors). It is pre-existing and
unrelated to this work, but it means `flash_attn_func` previously had **no**
working path here — neither ASM nor CK. Earlier benchmark rows claiming ASM
flash-attention throughput on gfx90a should be treated as unverified.

See [`docs/19-aiter-operator-port-matrix.md`](../docs/19-aiter-operator-port-matrix.md).
