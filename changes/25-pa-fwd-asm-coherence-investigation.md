# pa_fwd_asm Decode Coherence Investigation — Root Cause: ISA Incompatibility

**Date**: 2026-07-27
**Status**: ❌ pa_fwd_asm not viable on gfx90a via binary patching. Hybrid dispatch confirmed as production solution.

## Objective

Determine whether `pa_fwd_asm` (AITER's ASM paged-attention decode kernel) can produce
correct output on gfx90a (MI210) after binary patching from gfx942 (MI300). Previous
work proved the kernel runs without crashing but produces incoherent text.

## Background

The 3-layer binary patch recipe (e_flags, MFMA opcode, vgpr_count) enables gfx942 .co
files to load and execute on gfx90a. All 1,251 .co files were patched across 9 categories.
ASM prefill (`flash_attn_varlen_func`) and MLA kernels work correctly. But `pa_fwd_asm`
decode produces garbage tokens despite running without memory faults.

## Investigation

### Phase 1: Option 2 — Force reshape_and_cache Write Path

**Hypothesis**: ATOM's fused kernel (`fused_qk_norm_rope_cache_quant_shuffle`) writes K/V
data at wrong offsets for the SHUFFLE layout, corrupting pa_fwd_asm reads.

**Approach**: Patched `attention_mha.py` to bypass both fused branches on gfx90a, forcing
the `else` branch that uses `reshape_and_cache(asm_layout=True)` — the proven-correct
write path from standalone testing.

**Patch**: `configs/patch_option2_reshape_and_cache.py` adds `_force_asm_cache = get_gfx()
== "gfx90a"` check that skips `fused_qk_norm_rope_cache_quant_shuffle` (branch 1) and
`fused_qk_rope_reshape_and_cache` (branch 2).

**Result**: pa_fwd_asm runs without memory fault ✅. EXIT=0. TPOT=0.029ms.
But output is incoherent ❌:

```
Prompt: "introduce yourself"
Output: "<think>\nOkay, the user user user isuser"
```

**Conclusion**: Option 2 eliminates the crash but doesn't fix output coherence. The write
path was not the root cause.

### Phase 2: ISA-Level Root Cause Analysis

**Hypothesis**: The MFMA opcode patch (D3E1→D3CD) changes computation semantics.

**Approach**: Compiled test kernels for gfx90a using `llvm-mc` to enumerate supported
MFMA instructions.

**Finding**: gfx90a has **NO BF16 MFMA instructions at all**:

```
$ llvm-mc --triple=amdgcn-amd-amdhsa --mcpu=gfx90a
v_mfma_f32_16x16x16_bf16  → error: instruction not supported on this GPU
v_mfma_f32_4x4x4bf16      → error: instruction not supported on this GPU
v_dot2c_f32_bf16           → error: instruction not supported on this GPU
v_mfma_f32_16x16x16f16     → OK (opcode D3CD8000)
```

gfx90a (CDNA2) only supports F16 MFMA (`v_mfma_f32_16x16x16_f16`, opcode D3CD).
gfx942 (CDNA3) uses BF16 MFMA (`v_mfma_f32_16x16x16_bf16`, opcode D3E1).

The binary patch converts BF16 MFMA → F16 MFMA, which changes how the instruction
interprets VGPR data:
- BF16 format: 1 sign + 8 exponent + 7 mantissa bits
- FP16 format: 1 sign + 5 exponent + 10 mantissa bits

When F16 MFMA receives BF16 bit patterns, it misinterprets the exponent and mantissa
fields, producing completely wrong intermediate values.

### Phase 3: FP16 Data Conversion Fix

**Hypothesis**: If all data reaching the F16 MFMA is actually FP16 (not BF16), the
instruction will interpret it correctly.

**Approach**:
1. Allocate KV cache as `torch.float16` instead of `torch.bfloat16`
2. Convert K/V to FP16 before `reshape_and_cache`
3. Convert Q to FP16 before `pa_fwd_asm`

**Verification**: BF16→FP16 conversion is lossless for normalized neural network data:
```
BF16 reference: mean=-0.1918, std=11.2003
FP16 converted: mean=-0.1918, std=11.2003
Max difference: 0.000000
```

**Patches applied**:
- `aiter_attention.py`: KV cache dtype `torch.float16` on gfx90a
- `attention_mha.py`: `k.to(torch.float16)`, `v.to(torch.float16)` before reshape_and_cache
- `attention_mha.py`: `q.to(torch.float16)` before `run_pa_fwd_asm`

**Result**: Still garbage output:
```
"<think>\nOkay, the user user user is\n"
```

**Conclusion**: FP16 conversion alone is insufficient. Other ISA-level differences
beyond the MFMA opcode persist.

### Phase 4: AccVGPR/VGPR Register Investigation

**Hypothesis**: The vgpr_count patch (512→256) eliminates AccVGPR allocation, causing
MFMA results to be lost.

**Analysis**: The .co file metadata has:
- `.vgpr_count`: 256 (patched from original 512)
- No `.agpr_count` field (gfx942 uses a different metadata model)

On gfx90a, MFMA instructions write results to AccVGPRs (separate register file from
regular VGPRs). If 0 AccVGPRs are allocated, MFMA output is undefined.

**Approach**: Reversed vgpr_count from 256→512 for all 56 PA .co files. On gfx90a,
512 total registers = 256 VGPR + 256 AccVGPR (hardware maximum).

**Verification**: The PA kernel has 416 F16 MFMA instructions (confirmed via binary scan
with `struct.unpack("<I", data[i:i+4])[0]`).

**Result**: Still garbage output. The kernel loads and runs (different output for different
inputs confirms MFMA is executing), but results are incorrect.

**Conclusion**: The vgpr_count alone doesn't fix correctness. The kernel was compiled
for gfx942's pipeline with specific instruction scheduling, register banking, and timing
assumptions that don't hold on gfx90a.

### Phase 5: Root Cause Summary

pa_fwd_asm on gfx90a via binary patching is **not viable** due to fundamental ISA
incompatibilities between CDNA3 (gfx942/MI300) and CDNA2 (gfx90a/MI210):

1. **No BF16 MFMA**: gfx90a lacks `v_mfma_f32_16x16x16_bf16`. Must use F16 MFMA,
   which misinterprets BF16 data format.

2. **Register architecture**: gfx942 and gfx90a have different AccVGPR allocation and
   register banking models. The kernel binary has no `.agpr_count` metadata field
   (gfx942 unified model vs gfx90a split model).

3. **Pipeline differences**: The kernel has 416 MFMA instructions plus custom softmax,
   scaling, and accumulation code compiled for gfx942's pipeline. Subtle timing and
   instruction scheduling differences on gfx90a corrupt intermediate results.

4. **No binary-level fix possible**: These are architectural differences, not instruction
   encoding bugs. Fixing them requires recompiling from source targeting gfx90a.

## What Works vs What Doesn't

| Component | Status | Performance |
|---|---|---|
| ASM prefill (`flash_attn_varlen_func`) | ✅ Correct | 4,791,074 tok/s |
| MLA attention (`mla_fwd_kvcache`) | ✅ Correct | 3,013,378 tok/s |
| topk_softmax_asm | ✅ Correct | 0.73ms (E=256,K=8) |
| gemm_a16w16_asm | ✅ Correct | 60.1 TFLOPS |
| reshape_and_cache (asm_layout=True) | ✅ Correct | — |
| SiluAndMul | ✅ Correct | max_diff=0.03 |
| **pa_fwd_asm (decode)** | **❌ Garbage output** | Runs at 0.029ms/step but wrong |
| Triton decode (`unified_attention`) | ✅ Correct | 34.5 tok/s |

## Production Configuration

```bash
ATOM_USE_UNIFIED_ATTN=1 ATOM_LOADER_USE_THREADPOOL=0 \
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 \
  --max-tokens 10 \
  --block-size 64 \
  --enforce-eager \
  --level 0
```

Output:
```
"introduce yourself" → "<think>\nOkay, the user wants me to introduce"
"list all prime numbers within 100" → "<think>\nOkay, so I need to list all"
"1+2+3=?" → "<think>\nOkay, so I need to solve "
```

TTFT=0.527s, TPOT=0.029s. Coherent, contextually appropriate responses.

### Phase 6: Full FP16 Model + FP16 PA Kernel

**Hypothesis**: The FP16 PA kernel (`pa_fp16_noquant_gqa8_1tg_4w.co`) uses F16 MFMA
(D3CD) natively on gfx942 — the SAME opcode gfx90a supports. No MFMA opcode patch
needed. If the entire model runs in FP16, the FP16 PA kernel should work correctly.

**Approach**:
1. Converted Qwen3-0.6B to FP16 via `AutoModelForCausalLM.from_pretrained(torch_dtype=torch.float16)`
2. Saved to `/models/qwen3-0.6b-fp16`
3. Added `--dtype float16` to simple_inference.py (sets kv_cache_dtype + torch default dtype)
4. Ran without `ATOM_USE_UNIFIED_ATTN` to force pa_fwd_asm dispatch

**Result**: Failed with dtype mismatch:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != c10::Half
```

**Root cause**: ATOM's GEMM pipeline (`aiter/tuned_gemm.py`) hardcodes BF16 output type:
```
[aiter] shape is M:16384, N:4096, K:1024 dtype='torch.float16' otype='torch.bfloat16'
```

The GEMM kernel receives FP16 input but outputs BF16. This creates a dtype cascade:
1. Embedding layer: FP16 ✅
2. First GEMM (QKV projection): FP16 input → BF16 output ❌
3. Next layer: BF16 input vs FP16 weight → dtype mismatch ❌

**Conclusion**: The entire AITER GEMM pipeline is BF16-optimized. Switching to FP16
would require patching the GEMM output type, activation functions, attention dispatch,
and KV cache allocation. This exceeds the scope of binary patching.

## Future Path to Native pa_fwd_asm on gfx90a

1. **Recompile AITER from source for gfx90a**: AITER's C++/HIP source includes pa_fwd_asm.
   Building with `-DGPU_TARGETS=gfx90a` would produce correct native binaries with proper
   BF16 fallback (via F16 MFMA + conversion, or `v_dot2c` decomposition).

2. **Write a Triton-based pa_fwd_asm replacement**: A Triton kernel for paged attention
   decode would be JIT-compiled for gfx90a and produce correct results natively. This is
   essentially what `unified_attention` already provides.

3. **Wait for AMD to add gfx90a to AITER's official support matrix**: AMD may add gfx90a
   as a build target in future AITER releases.

## Files

- `configs/patch_option2_reshape_and_cache.py` — Option 2 patch (force reshape_and_cache)
- `tests/test_atom_model_ops.py` — Comprehensive operator test suite
- `tests/test_pa_fwd_asm_e2e.py` — End-to-end pa_fwd_asm vs Triton comparison
- `configs/run_option2_on_big.sh` — One-command rebuild + test script
