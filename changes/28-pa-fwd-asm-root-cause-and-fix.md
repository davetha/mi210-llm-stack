# pa_fwd_asm Fixed on gfx90a: Stale JIT Module, Not ISA Incompatibility

**Date**: 2026-07-27
**Status**: ✅ RESOLVED. 48/48 configs numerically exact vs PyTorch reference.

## Summary

Change 27 concluded "gfx942 pa_fwd binaries cannot run on gfx90a. Period." That was
wrong. Two ordinary bugs were responsible, neither architectural:

1. A stale `module_attention_asm.so` whose `KernelArgs` layout predated the installed
   `.co` files → `GQA` read as 0 → output buffer `num_records = 0` → **every
   `buffer_store` silently discarded**.
2. Three of the eight portable pa kernels still carried the wrong `D3E1→D3CD` MFMA
   patch instead of `D3E1→D3E7`.

Full analysis: [`docs/18-pa-fwd-asm-resolved.md`](../docs/18-pa-fwd-asm-resolved.md).

## The Silent Failure

The kernel launched, ran full speed (258 µs, ~1 TB/s of KV reads), reached `s_endpgm`,
and wrote nothing. What change 25 recorded as "incoherent output" was never the kernel's
output — it was uninitialised `torch.empty_like(Q)`.

```asm
s_load_dword s77, s[0:1], 0xf0      ; GQA         -> 0   (host wrote it at 0xe0)
s_mul_i32    s76, 0x100, s77        ; num_records -> 0
s_mov_b32    s10, s76               ; V# word2
buffer_store_dword v128, v8, s[8:11], 0 offen   ; dropped: num_records == 0
```

Per both CDNA2 and CDNA3 ISA guides, `num_records == 0` puts every buffer access out of
range, and out-of-range writes are dropped.

Cause: `module_attention_asm.so` was built at 20:29; `asm_pa.cu` and every `.co` were
replaced at 22:05, and an `mtp` field was added to `KernelArgs` in between.

## Fixes Applied

```bash
# 1. Rebuild the stale JIT module
rm .../aiter/jit/module_attention_asm.so          # aiter rebuilds in ~13s

# 2. Repatch the 3 mis-patched bf16 kernels from the pristine gfx942 originals
python configs/repatch_gfx942_to_gfx90a.py .../hsa/gfx942 ./out pa/
cp out/pa/*.co .../hsa/gfx90a/pa/
```

## Correction to Change 27

Change 27's ISA analysis does not hold. Disassembling the kernel and re-assembling all
3,303 instructions with `llvm-mc -mcpu=gfx90a` produces **zero errors**; the only
byte-level difference in the entire kernel is the MFMA opcode.

- "0 accvgpr instructions ⇒ incompatible register model" — gfx90a has a *unified*
  VGPR/AGPR file; MFMA writing ArchVGPRs is normal on CDNA2.
- "FLAT / VOP3P encoding differences" — none, byte-identical.
- "GPU HANG at block-size 16" — the ASM kernel *requires* block_size 16 (`pa_asm.csv`,
  `blkSz=16`). No hang once the module is rebuilt.
- The kernel descriptor needs no change; the vgpr patches were unnecessary.

Change 27 *is* right about fp8/int8: those use `v_cvt_pk_fp8_f32` and
`v_mfma_i32_16x16x32_i8`, which have no gfx90a equivalent.

## Audit of All 1,251 Patched .co Files

New tool `configs/repatch_gfx942_to_gfx90a.py` re-encodes every instruction through the
assembler and refuses kernels that don't assemble for gfx90a.

```
TALLY (all 1,422 gfx942 kernels): {'OK': 242, 'NOTPORT': 1180}
```

| | Count |
|---|---|
| installed `.co` in `gfx90a/` | 1,251 |
| **installed but NOT portable** | **1,147** |
| installed and portable | 104 |
| — of those **mis-patched** | **74** |

The 74 miss a second opcode: `v_mfma_f32_32x32x8_bf16` → `v_mfma_f32_32x32x8bf16_1k`
(48 `fmha_v3_fwd`, 11 `mla`, 8 `fmoe`, 7 top-level). They are inert on gfx90a because
`mha.py` / `mla.py` gate the v3 ASM paths to `gfx942`/`gfx950` — which also means the
"ASM flash attention 4.79M tok/s" and "ASM MLA" rows in earlier docs measured the
CK/Triton fallback, not ASM.

## Environment

Rebuilt 6 of 7 stale JIT modules (`module_aiter_core`, `module_activation`,
`module_sample`, `module_rmsnorm_quant`, `module_rope_2c_cached_positions_fwd`,
`module_gemm_common`); pa re-validated 48/48 afterwards.

`mha_varlen_fwd_bf16_*` cannot be rebuilt — CK `ck_tile/host/device_prop.hpp` throws 6
redefinition errors. Its pre-upgrade `.so` still works and was restored.

## Files

| File | Purpose |
|---|---|
| `docs/18-pa-fwd-asm-resolved.md` | Full analysis, audit, and corrections to doc 17 |
| `configs/repatch_gfx942_to_gfx90a.py` | Assembler-verified gfx942→gfx90a repatcher |
| `tests/test_pa_fwd_asm_gfx90a.py` | 48-config correctness harness vs PyTorch reference |
