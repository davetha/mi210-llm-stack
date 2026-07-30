# The instruction budget: these GEMMs issue ~100 instructions per MFMA

**Date**: 2026-07-30 · **Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2)
**Source**: 3,032 compiled Triton kernels in `fa-build:/root/.triton/cache`,
counted from the emitted `.amdgcn`. ISA claims re-verified by assembling for
gfx90a with `llvm-mc` (LLVM 21.1.8).

`docs/25` item 1c established that decode sits far off its bandwidth bound and
left open "what specifically consumes the other 123 ms/token." This answers a
large part of it without a profiler and without touching the GPU: **the shipped
GEMM kernels spend under 3% of their issued instructions on matrix ops.**

## The measurement

Every cached `.amdgcn` on the box, grouped by dominant MFMA opcode:

| dominant MFMA | files | total ins | MFMA | **ins/MFMA** | `v_pk_*` | `v_perm_b32` |
|---|---:|---:|---:|---:|---:|---:|
| `v_mfma_f32_32x32x8f16` | 1757 | 16,852,715 | 174,848 | **96.4** | 414,318 | 623,894 |
| `v_mfma_f32_16x16x16f16` | 1131 | 6,355,342 | 60,656 | **104.8** | 68,203 | 167,228 |
| `v_mfma_f32_32x32x8bf16_1k` | 2 | 11,679 | 448 | 26.1 | 384 | 384 |
| `v_mfma_f32_16x16x16bf16_1k` | 6 | 7,834 | 364 | 21.5 | 159 | 232 |

Per kernel, largest cached variant of each:

| kernel | total | MFMA | `v_cmp`+`v_cndmask` | `v_perm_b32` | `v_pk_*` |
|---|---:|---:|---:|---:|---:|
| `_w8a8_triton_block_scaled_mm` (stock) | 79,060 | 1,024 | **38,371 (48.5%)** | 1,152 | 1,094 |
| `_fast_blockscale_gemm_kernel` (the fix) | 36,293 | 1,024 | **1,507** | 2,061 | 2,030 |
| `_gemm_a8w8_blockscale_kernel` (AITER) | 23,674 | 512 | 14,160 | 448 | 257 |
| `_gfx90a_fast_block_scaled_mm` | 5,808 | 128 | 582 | 544 | 420 |

Reproduce: for each `.amdgcn`, count lines beginning with a tab whose first token
is a mnemonic (skip `.`/`;`/`//`), then bucket by prefix.

## Three things this establishes

### 1. `docs/21` understated itself by 3×

The published figures were 11,997 instructions with 7,106
`v_cmp_ne_u16`/`v_cndmask_b32`. The **largest cached variant is 79,060 with
38,371** — the FP8-decode emulation is **48.5% of the whole kernel**, and it
scales with tile size rather than being a fixed cost. The reported 5.5–6× is a
floor, not a ceiling.

### 2. The fix's signature confirms which idiom this chip wants

`v_cmp`/`v_cndmask` 38,371 → 1,507 (**25×**) while `v_perm_b32` (1,152 → 2,061)
and `v_pk_*` (1,094 → 2,030) roughly **double**. Byte-permute plus packed math is
the right shape here, and that is now measured rather than argued. Worth keeping
as a general rule for any future dequant work on this target.

### 3. Dequant was the biggest term, not the only one

Even after the fix: **36,293 instructions for 1,024 MFMA = 34 non-MFMA
instructions per MFMA.** ~97% of issue is still not matrix work — scale
application, address arithmetic, masking. A GEMM at that ratio is issue-bound,
which is consistent with item 1c's residual and explains why speculative decoding
lost at 100% draft acceptance: if per-token cost is instruction issue, verifying
N+1 tokens costs ~N+1×, exactly as measured.

## What is NOT measured here, and it is the important gap

This cache is dominated by the **FP8 blockscale family** — the already-known
pathological path. **No int8 `w8a16`, `fused_moe`, or `wna16` kernel is cached in
any container on the box.** So the path that actually ships per `docs/24` (W8A8
dense + **W8A16 experts**, which dequantizes int8 → float and never reaches
`v_mfma_i32_16x16x16i8`) has never been counted.

Given that the FP8 path measured 48.5% emulation and the *fixed* path still sits
at 34 ins/MFMA, the prior that the int8 path is clean is weak.

**There is a smoking gun for it already in `docs/24`:** AWQ-Int4 loses to INT8
W8A8 (**5.07 s vs 3.20 s TTFT**) despite moving half the bytes. On a chip where
quantization buys only bytes, a format with fewer bytes losing means the dequant
ate the win — the same signature as FP8.

**UPDATE (2026-07-30): partially answered, and it does not support the
prediction above.** The shipping `fused_moe_kernel` was subsequently found in a
*different* cache (`/mnt/llm-storage/bench-results/vllm-opt/cache/triton`, gfx90a
confirmed) and counted: **median 195 MACs/ins, best 497**, with
`v_cmp`+`v_cndmask` at only **10–14%** against FP8's 48.5%. There is no
software-emulation tax in the shipping MoE path. See the correction section in
`docs/33`. The Lead-A hypothesis below — that the int8 path carries the same
pathology as FP8 — should be read as **not supported**; the remaining question is
tile selection, not dequant. Retained here as written because the reasoning is
the record.

**Next experiment, no GPU required (~1 hour):** get a `fused_moe`/`w8a16` variant
into a Triton cache from any prior W8A8 MoE serving run, then run the same count.
If `v_cmp`/`v_cndmask` is a double-digit percentage, the `docs/21` fix transfers
directly.

## ISA verification (assembled, not assumed)

`llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding`, one instruction per invocation:

| instruction | verdict | encoding |
|---|---|---|
| `v_perm_b32` | **OK** | `[0x00,0x00,0xed,0xd1,...]` |
| `v_pk_add_f16` | **OK** | `[0x00,0x40,0x8f,0xd3,...]` |
| `v_pk_mul_f16` / `v_pk_fma_f16` | **OK** | — |
| `v_pk_add_bf16` | **REJECT** | invalid instruction |
| `v_mfma_f32_16x16x16f16` | **OK** | opcode `0xCD` |
| `v_mfma_f32_16x16x16bf16_1k` | **OK** | opcode `0xE7` |
| `v_mfma_i32_16x16x16i8` | **OK** | opcode `0xD5` |
| `v_mfma_i32_16x16x32_i8` | **REJECT** | not supported on this GPU |
| `v_cvt_pk_fp8_f32` | **REJECT** | not supported on this GPU |
| `global_atomic_pk_add_bf16` | **REJECT** | not supported on this GPU |
| `v_lshl_or_b32` / `v_and_or_b32` / `v_bfi_b32` | **OK** | — |
| `v_dot4c_i32_i8` / `v_dot4_i32_i8` | **OK** | — |

Every ISA claim in this repo reproduces, including the D3E7/D3CD pair from the
`docs/14` retraction — `0xE7` is `bf16_1k`, `0xCD` is `f16`. The three rejections
are the three hardware gaps the repo already documents.

Note `v_pk_add_bf16` **does not exist** while `v_pk_add_f16` does, and both MFMA
flavours run at the same 181 TFLOP/s. That would make fp16 the better dequant
target — except the cache shows **2,888 of 2,896 MFMA kernels already use f16
MFMA** (bf16 appears in 8 files). Triton on ROCm is already targeting fp16, so
the dtype is not the bug. Recorded because it is the obvious next hypothesis and
the machine already refutes it.

## The cheap dequant, for whenever a kernel does get written

Deposit uint8 `v` into the fp16 bit pattern for 1024.0 and you get fp16(1024+v)
**bit-exactly** — fp16's 11-bit significand spans 1024–1279, so every byte value
is representable:

- One `v_perm_b32` builds `[b0, 0x64, b1, 0x64]` from the weight dword and a
  constant — **1 instruction per 2 values**, with the OR folded into the byte
  select.
- The −1024 bias goes in the fp32 epilogue:
  `Σ(1024+w)·x = 1024·Σx + Σw·x`, and `Σx` is one per-token reduction.
- **~0.5 VALU ops per weight, numerically exact.** The identical trick works on
  int4 nibbles (0–15) at the same cost.

This is the same structure as the `docs/21` bit-reinterpretation, and matches what
QServe (arXiv **2405.04532**) calls "subtraction after multiplication" — keeping
the zero-point out of the inner loop. Marlin (arXiv **2408.11743**) uses the same
magic-number trick via `lop3.b32`; gfx9 has no 3-input LUT, so it costs 2 ops
(`v_and_b32` + `v_lshl_or_b32`/`v_bfi_b32`) instead of 1, or 1 `v_perm_b32` if the
constant byte is folded into the permute.

## How this could be wrong

- If the int8 path turns out to be ~10 ins/MFMA, there is nothing to win here and
  the residual gap lives in launch overhead instead — see the backlog addendum.
- The ins/MFMA ratio counts *static* instructions, not dynamic issue. A high
  static ratio in a prologue that executes once is not the same as a hot inner
  loop. The 1,024-MFMA kernels are large tiled bodies so the ratio is likely
  representative, but `rocprofv3 --kernel-trace` is what would settle it.
- fp16 dequant overflows if scales are applied *before* the MFMA. They must go in
  the fp32 epilogue, which is a kernel-structure change rather than a
  substitution.

## Confidence

| Claim | Confidence | Basis |
|---|---|---|
| f16 GEMM kernels are 96–105 ins/MFMA | **High** | 2,888 files counted |
| Stock FP8 kernel is 48.5% `v_cmp`/`v_cndmask` | **High** | direct count, largest variant |
| The fix cuts that 25× and doubles perm/packed | **High** | direct count |
| Every ISA verdict in the table | **High** | assembled with `llvm-mc` |
| int8 `w8a16` path has the same pathology | **UNMEASURED** | not cached anywhere on the box |
| Static ratio ⇒ issue-bound at runtime | **Medium** | needs `rocprofv3` to confirm |
