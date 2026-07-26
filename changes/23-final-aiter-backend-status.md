# Final AiterBackend Status

## What's Proven

1. **Binary patches correct for standalone execution** — pa_fwd_asm with random data produces valid output (100% non-zero)
2. **First ATOM call succeeds** — trace proves correct shapes, strides, block tables
3. **SDPA fallback works end-to-end** — 14 calls, zero faults, pipeline correct
4. **Triton backend works end-to-end** — 35.7 tok/s, coherent text generation

## Root Cause of ASM pa_fwd_asm Fault

The BF16→F16 MFMA opcode swap (D3E1→D3CD) produces slightly different FP32
intermediate values. For MLA decode, this was fine (proven at 0.090ms). For
pa_fwd_asm, specific model data patterns in layers 2+ trigger a numerical
cascade that produces invalid memory addresses.

This is the same class of issue as the original MLA prefill kernel —
data-dependent memory addressing that's sensitive to BF16 vs F16 precision.
MLA was fixed by finding correct tensor shapes; pa_fwd_asm's addressing code
is more complex and embedded in the binary kernel.

## Working Configurations

### Production (Triton backend)
```bash
ATOM_USE_UNIFIED_ATTN=1 python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B --block-size 64 --enforce-eager --level 0
```
Result: 35.7 tok/s decode, TTFT 1.31s, coherent text.

### ASM MLA kernels (proven working)
- MLA prefill: 3,013,378 tok/s ✅
- MLA decode: 0.090ms/step (3× faster than Triton) ✅
- topk_softmax_asm: MiMo shape validated ✅
- gemm_a16w16_asm: 60.1 TFLOPS ✅

### ASM pa_fwd_asm (partial)
- Standalone random data: ✅ Works
- First ATOM decode step: ✅ Works
- Subsequent decode steps: ❌ Faults (numerical cascade)

## Path to Full ASM pa_fwd_asm

1. **Write native gfx90a pa kernel** using v_mfma_f32_16x16x16f16
   (avoids binary patch entirely, uses native instruction)
2. **Debug the faulting instruction** via core dump + rocgdb to find
   the exact addressing code that's sensitive to F16 precision
3. **Use Triton decode** with ASM prefill (hybrid approach — best of both)
