# ATOM Inference Milestone: Model Loads, Inference Starts

**Date**: 2026-07-26
**Status**: Model loading works, inference dispatches to patched kernels, pa decode kernel faults

## What Works

1. **ATOM v0.1.5 + AITER v0.1.17**: Import, model class resolution, engine creation
2. **Qwen3-0.6B model loading**: Weights loaded from safetensors to GPU VRAM
3. **JIT compilation**: All modules compiled (rmsnorm_quant, cache, mha_varlen_fwd)
4. **Kernel dispatch**: ATOM dispatched to our patched `pa_bf16_noquant_gqa8_1tg_4w`
5. **Binary patches load correctly**: The patched .co file was loaded from gfx90a/pa/

## What's Blocked

Memory fault in `pa_bf16_noquant_gqa8_1tg_4w` during decode:
```
Memory access fault by GPU node-1 on address 0x7d2959db5000
kernel: aiter::pa_bf16_noquant_gqa8_1tg_4w
grid=[2048, 4, 1], workgroup=[256, 1, 1], group_seg_size=65536
```

Same class of issue as MLA — patched kernel executes but has memory addressing fault.
Fixable by finding correct parameters (like MLA's 576/512 tensor shapes).

## Infrastructure Changes Made

1. **ROCm 6.3.1 → 7.2.0**: Upgraded system packages
2. **AITER 0.1.13 → 0.1.17**: Built from source for Python 3.14
3. **CK headers**: Extracted from official v0.1.17 wheel (submodules weren't in shallow clone)
4. **Triton pre-import**: sitecustomize.py avoids LLVM symbol crash
5. **flydsl 0.1.4 → 0.2.2**: Required by AITER v0.1.17
6. **Binary patches re-applied**: All 1,251 .co files for gfx90a
