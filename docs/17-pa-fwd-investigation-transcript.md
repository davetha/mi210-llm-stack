# pa_fwd_asm Decode on gfx90a: Full Investigation Transcript

> ⚠️ **SUPERSEDED — the conclusion below is wrong.** `pa_fwd_asm` **does** work on
> gfx90a. The blocker was a stale JIT module whose kernarg layout predated the
> installed `.co` files, which made every `buffer_store` get silently dropped —
> not an ISA incompatibility. See
> [`18-pa-fwd-asm-resolved.md`](18-pa-fwd-asm-resolved.md).
> This document is kept as a record of the investigation, including the dead ends.

**Date**: 2026-07-27
**Goal**: Get AITER ASM paged-attention decode (`pa_fwd_asm`) working on AMD MI210 (gfx90a)
**Outcome**: gfx942 binaries cannot run on gfx90a. Two native alternatives work.
**Correction (2026-07-27)**: Outcome disproved — see banner above.

---

## Background

We binary-patched 1,251 AITER ASM `.co` files from gfx942 (MI300X) to gfx90a (MI210).
All operators work EXCEPT `pa_fwd_asm` (paged-attention decode) — the kernel either
crashes, hangs, or produces garbage text.

**Hardware**: 2× AMD Instinct MI210 (gfx90a/CDNA2), 64GB HBM2e each
**Software**: ROCm 7.14.0, AITER 0.1.17, ATOM v0.1.5, PyTorch 2.11

---

## Phase 1: Initial Problem

`pa_fwd_asm` runs but produces incoherent output:

```
Prompt: "introduce yourself"
Output: "<think>\nOkay, the user user user isuser"
```

No memory fault — EXIT=0, TPOT=0.029s. But text is garbage.

## Phase 2: Option 2 — Force reshape_and_cache

**Hypothesis**: ATOM's fused kernel writes K/V data at wrong offsets.

**Fix**: Patch `attention_mha.py` to bypass fused kernels and use
`reshape_and_cache(asm_layout=True)` directly.

**Result**: No crash, but still garbage. Write path was not the root cause.

## Phase 3: ISA Analysis — "No BF16 MFMA on gfx90a"

Tested with `llvm-mc --mcpu=gfx90a`:

```
v_mfma_f32_16x16x16_bf16  → error: instruction not supported
v_mfma_f32_16x16x16f16     → OK (D3CD8000)
```

**Conclusion** (later proven WRONG): gfx90a has no BF16 MFMA. Our binary patch
(D3E1→D3CD) changes BF16 MFMA to F16 MFMA, which misinterprets BF16 bit patterns.

## Phase 4: FP16 Data Conversion

Converted Q, K, V, and KV cache to `torch.float16`. BF16→FP16 is lossless for
normalized data (max diff 0.000000).

**Result**: Still garbage. Other ISA differences persist beyond MFMA.

## Phase 5: FP16 Model + FP16 Kernel

Converted Qwen3-0.6B weights to FP16. The FP16 PA kernel
(`pa_fp16_noquant_gqa8_1tg_4w.co`) uses D3CD natively on gfx942.

**Result**: `RuntimeError: expected mat1 and mat2 to have the same dtype,
but got: c10::BFloat16 != c10::Half` — ATOM's GEMM pipeline hardcodes BF16 output.

## Phase 6: ASM Source Not Available

AITER repo ships pre-compiled `.co` binary blobs. No assembly source published.
Recompilation from source is not possible.

## Phase 7: pa_fwd_naive — CK JIT Works!

Discovered `pa_fwd_naive` in AITER's Composable Kernel path — JIT-compiled natively.

**Correctness test** (vs PyTorch reference):
```
Max diff:  0.003906
Mean diff: 0.000253
Match:     100.0%
RESULT:    pa_fwd_naive CORRECT ✅
```

### Hidden Root Cause: Block Tables Mismatch

Scheduler uses `--block-size 64`, physical KV cache uses block_size=16 (ratio=4).
Block tables at scheduler granularity → wrong block indices for pa_fwd_naive.

**Fix**: `--block-size 16` aligns scheduler and physical blocks.

**Result**: COHERENT OUTPUT ✅
```
"introduce yourself"     → "<think>\nOkay, the user asked me to introduce"
"list all prime numbers" → "<think>\nOkay, so I need to list all"
"1+2+3=?"                → "<think>\nOkay, so I need to solve "
```

TPOT=0.093s (3× slower than Triton — expected for naive implementation).

## Phase 8: Definitive Binary Patch Investigation

### Discovery: gfx90a HAS BF16 MFMA

Research revealed gfx90a has `v_mfma_f32_16x16x16bf16_1k` (opcode D3E7):

```
$ llvm-mc --mcpu=gfx90a
v_mfma_f32_16x16x16bf16_1k a[0:3], v[0:1], v[2:3], 0
// D3E78000 02020500  ← VALID on gfx90a!
```

Our original D3E1→D3CD patch was WRONG. Should have been D3E1→D3E7.

### Test: Correct BF16 MFMA (D3E1→D3E7) with --block-size 16

Patched original gfx942 `.co` with only:
1. e_flags: 0x4c → 0x3f
2. MFMA: D3E1 → D3E7 (correct gfx90a BF16 MFMA _1k)
3. No vgpr patch, no F16 conversion

**Result**: GPU HANG (`rocdevice.cpp:3678`). Same as all previous attempts.

### Complete Test Matrix

| Patch | block_size | Data | Result |
|-------|-----------|------|--------|
| D3E1→D3CD + vgpr=256 | 64 | BF16 | Garbage output |
| D3E1→D3CD + vgpr=512 | 64 | BF16 | Garbage output |
| D3E1→D3CD + vgpr=256 | 16 | BF16 | GPU HANG |
| D3E1→D3CD + vgpr=512 | 16 | BF16 | GPU HANG |
| D3E1→D3CD + FP16 cache | 64 | FP16 | Garbage output |
| Native FP16 kernel (D3CD) | 16 | FP16 | GPU HANG |
| D3E1→D3E7 (correct BF16_1k) | 16 | BF16 | GPU HANG |
| pa_fwd_naive (CK JIT) | 16 | BF16 | ✅ Coherent (0.093s) |
| Triton unified_attention | 64 | BF16 | ✅ Coherent (0.029s) |

### Binary Analysis

The pa_fwd kernel contains:
- 416 MFMA instructions (D3CD or D3E7 after patching)
- 162 FLAT load/store instructions
- 146+ VOP3P operations
- 0 `v_accvgpr_read` / `v_accvgpr_write` instructions

The zero AccVGPR instructions is significant: gfx942 and gfx90a handle MFMA
output registers differently. The binary was compiled for gfx942's register
model and cannot execute correctly on gfx90a.

### Why It Hangs

Even with the correct MFMA opcode (D3E7), the gfx942 binary hangs on gfx90a
because:

1. **FLAT instruction encoding** may use gfx942-specific addressing modes
2. **VOP3P modifier bits** have different meanings between architectures
3. **No AccVGPR management** — gfx942 binary doesn't transfer MFMA results
   from AccVGPRs to VGPRs (unnecessary on gfx942, required on gfx90a)
4. **Pipeline scheduling** optimized for gfx942's CU design

### AMD Community Guidance

From [NGKore MI210 deployment guide](https://docs.ngkorefoundation.org/ai-ml/amd-mi210-llm-deployment/):
> "No newest FP8/FP4 kernels or AITER fast paths — those target gfx942+ (MI300)"

AMD does not attempt cross-architecture binary compatibility for MFMA kernels.

---

## Working Solutions

### 1. Triton unified_attention (RECOMMENDED — Production)

```bash
ATOM_USE_UNIFIED_ATTN=1 ATOM_LOADER_USE_THREADPOOL=0 \
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 --max-tokens 10 \
  --block-size 64 --enforce-eager --level 0
```

- **Prefill**: ASM flash_attn_varlen_func (4.8M tok/s)
- **Decode**: Triton unified_attention (JIT-compiled for gfx90a)
- **Performance**: TTFT=0.527s, TPOT=0.029s (34.5 tok/s)
- **Output**: Coherent English + Chinese ✅

### 2. CK pa_fwd_naive (Native, Slower)

```bash
ATOM_LOADER_USE_THREADPOOL=0 \
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 --max-tokens 10 \
  --block-size 16 --enforce-eager --level 0
```

- **Decode**: CK pa_fwd_naive (JIT-compiled for gfx90a)
- **Performance**: TPOT=0.093s (10.8 tok/s)
- **Output**: Coherent ✅
- **Caveat**: Requires `--block-size 16` and V cache format conversion per step

---

## What Would Fix pa_fwd_asm on gfx90a

1. **AMD publishes gfx90a .co files** — feature request on AITER GitHub
2. **Custom HIP kernel** using `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`
   compiled natively with `hipcc --offload-arch=gfx90a`
3. **AMD's HotSwap/COMGR** facility for load-time code rewriting (complex)

---

## Performance Summary

| Component | Status | Performance |
|-----------|--------|-------------|
| ASM MLA prefill | ✅ Works | 3,013,378 tok/s |
| ASM MLA decode | ✅ Works | 0.090ms/step |
| ASM topk_softmax | ✅ Works | 0.73ms (E=256,K=8) |
| ASM BF16 GEMM | ✅ Works | 60.1 TFLOPS |
| ASM flash attention | ✅ Works | 4,791,074 tok/s |
| ASM reshape_and_cache | ✅ Works | Correct SHUFFLE writes |
| **ASM pa_fwd decode** | **❌ Cannot work** | gfx942 binary incompatible |
| **Triton unified_attention** | **✅ Works** | **0.029s/step (34.5 tok/s)** |
| **CK pa_fwd_naive** | **✅ Works** | **0.093s/step (10.8 tok/s)** |

---

## Files

| File | Purpose |
|------|---------|
| `changes/25-pa-fwd-asm-coherence-investigation.md` | Phases 1-6: initial investigation |
| `changes/26-pa-fwd-naive-native-ck-decode.md` | Phase 7: CK JIT breakthrough |
| `changes/27-definitive-pa-fwd-investigation.md` | Phase 8: complete test matrix + root cause |
| `configs/patch_option2_reshape_and_cache.py` | Force reshape_and_cache on gfx90a |
| `configs/patch_correct_bf16.py` | D3E1→D3E7 patch (correct BF16 MFMA for gfx90a) |
| `tests/test_atom_model_ops.py` | Comprehensive ATOM operator test suite |
| `tests/test_pa_naive_correctness.py` | pa_fwd_naive vs PyTorch reference |
| `docs/16-complete-technical-reference.md` | Full 9-section technical reference |

**Repository**: https://github.com/davetha/mi210-llm-stack
