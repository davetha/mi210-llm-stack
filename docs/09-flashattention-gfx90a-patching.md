# FlashAttention on gfx90a: Why It's Broken and How to Fix It

## The Problem

FlashAttention (FA) is a REGRESSION on AMD MI210 (gfx90a/CDNA2):

| Mode | Prefill (tok/s) | Decode (tok/s) | Notes |
|---|---|---|---|
| FA-off (standard attention) | **1,984** | **162** | Fast, but no V cache quantization |
| FA-on (broken fallback) | 770 | 24 | 2.6× slower prefill, 6.7× slower decode |

The standard (non-FA) attention path is FAST on gfx90a. But it can't do V cache
quantization — forcing users to either:
- Use FA-on (slow, but compressed V works)  
- Use FA-off (fast, but f16 V only — 4× larger KV cache)

## Root Cause Chain

### Why FA-on Is Slow
1. llama.cpp's built-in FA uses rocWMMA (Matrix Core) instructions
2. The rocWMMA FA path (`GGML_HIP_ROCWMMA_FATTN=ON`) targets CDNA3-specific MFMA variants
3. gfx90a (CDNA2) HAS Matrix Core, but with different MFMA instruction formats
4. When rocWMMA FA is disabled (as on our build), FA falls back to a GENERIC kernel
5. This generic FA fallback is much slower than standard non-FA attention

> **Superseded (2026-07-27).** Points 4 and 5 are wrong for current llama.cpp.
> With rocWMMA disabled, gfx90a does *not* fall back to a generic kernel — it
> routes to `fattn-mma-f16`, which uses `v_mfma_f32_16x16x16f16` on the matrix
> cores, verified by disassembling the shipped `libggml-hip.so`. Enabling
> rocWMMA is measurably *slower*. See
> [`22-rocwmma-flash-attention-gfx90a.md`](22-rocwmma-flash-attention-gfx90a.md).

### Why FA-off Can't Do V Quantization
1. The non-FA attention path has K cache dequantization code (works fine)
2. The non-FA attention path is MISSING V cache dequantization code
3. llama.cpp throws: "V cache quantization requires flash_attn"
4. This is a CODE COMPLETENESS issue, not a hardware limitation
5. The FA path has fused V dequant+attention; the non-FA path never got V dequant added

## gfx90a Matrix Core Capabilities

Contrary to initial assumptions, MI210 (gfx90a/CDNA2) DOES have Matrix Core:
- `v_mfma_f32_16x16x16f16` — 16×16×16 FP16 matrix multiply-accumulate
- `v_mfma_f32_32x32x16f16` — 32×32×16 FP16 (gfx908+, may work on gfx90a)
- `v_mfma_i32_16x16x16i8` — INT8 matrix multiply-accumulate

The rocWMMA failure is because the llama.cpp FA code targets CDNA3 (MI300)
MFMA formats, not because the hardware lacks Matrix Core entirely.

## ROCm Standalone FlashAttention WORKS

We built and verified ROCm's standalone FlashAttention on gfx90a:
- flash-attn 2.8.3 via Composable Kernel (CK) backend
- Build: `GPU_ARCHS="gfx90a" python3 setup.py install` (2,926 kernel objects)
- Correctness: max_diff=0.001804 vs fp32 reference
- head_dim=64 causal attention: ✅ PASS
- Build time: ~59 min with MAX_JOBS=32

The CK backend uses CDNA2-compatible MFMA instructions. This proves FA CAN
work on gfx90a — just not through llama.cpp's rocWMMA path.

## Three Patching Approaches

### Approach A: Fix rocWMMA FA for CDNA2 (Complex)
- Modify the FA kernel to use gfx90a-compatible MFMA instructions
- Map CDNA3 MFMA calls → CDNA2 equivalents
- Risk: instruction format differences may require significant rework
- Benefit: proper FA with fused V dequant

### Approach B: Integrate CK FlashAttention (Medium)
- Patch llama.cpp to dispatch to CK's flash_attn_func when running on gfx90a
- Challenge: tensor format conversion (CK expects different Q/K/V layouts)
- We have flash-attn 2.8.3 built and verified on this hardware
- Benefit: production-tested FA kernel

### Approach C: Add V Dequantization to Non-FA Path (Simplest, Best)
- The non-FA attention path already has K dequantization
- Add V dequantization to the same code path (same pattern, different tensor)
- Estimated: 20-50 lines of code
- Removes the "requires flash_attn" check for V cache
- Benefit: compressed V + fast attention (1984 tok/s) simultaneously

## The V Dequantization Gap

In the non-FA attention path:
```
// K dequantization (ALREADY EXISTS):
K_value = dequantize(K_cache[position], type_k)

// V usage (CURRENT — no dequant, raw access):
output += attention_weight * V_cache[position]    // Assumes f16!

// V dequantization (NEEDS TO BE ADDED):
V_value = dequantize(V_cache[position], type_v)   // Same pattern as K
output += attention_weight * V_value
```

The K dequantization code in the non-FA path provides the exact template.
V dequantization is the same operation applied to the V tensor.

## Impact of the V Dequant Patch

If patched, the optimal config becomes:
```
-ctk q4_0 -ctv q4_0 -fa off
  → 75% smaller KV cache (high context capability)
  → 1,984 tok/s prefill (fast attention)
  → 162 tok/s decode (fast attention)
  → Both speed AND capacity
```

Currently impossible because:
- q4_0 V requires FA-on → FA-on is slow → 24 tok/s decode
- FA-off requires f16 V → 4× larger KV → can't fit high context

The patch breaks this constraint.

## For the 230B MiMo Production Model

With V dequantization + FA-off:
- q4_0 K + q4_0 V: ~75% KV VRAM savings → more layers on GPU
- Fast non-FA attention on GPU layers
- Compressed KV for CPU layers (less DDR4 bandwidth)
- Higher context windows (65K+ tokens)
- Better session persistence (smaller KV files)

This is the single highest-impact optimization remaining.
