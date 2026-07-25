# Change Log: Final MiMo Configuration Deployed

## Date: 2026-07-25

## Summary
Deployed winning mimo configuration after exhaustive A/B testing of all viable KV cache types. The winner uses the SAME KV types as before (q8_0/q4_1) but with the TurboQuant binary (3× faster prefill) and CUDA graphs disabled (fixes decode crash).

## Config Changes
| Setting | Before | After |
|---|---|---|
| Binary | Stock llama.cpp | TurboQuant fork binary |
| KV types | q8_0/q4_1 | q8_0/q4_1 (unchanged) |
| FA mode | on | on (unchanged) |
| CUDA graphs | enabled | **disabled** (GGML_CUDA_DISABLE_GRAPHS=1) |
| Prefill speed | ~130 tok/s | **392 tok/s** (3× faster) |
| 15K cold prefill | ~115s | **38.8s** |

## What Was Tested and Failed
- **kivi2**: Custom quant type, no GPU SET_ROWS kernel → crash
- **turbo3**: Custom quant type, decode graph capture bug → crash
- **q4_0/f16 FA-off**: f16 V cache OOM at 65K context
- **turbo3 K**: Pathologically slow (24 tok/s vs q8_0's 392 tok/s)

## Lesson Learned
The biggest prefill speedup came from using the TurboQuant fork's optimized binary, NOT from KV cache compression. Custom quant types (kivi2, turbo3) require GPU-specific HIP kernels that don't exist on gfx90a. Standard GGML types (q8_0, q4_0, q4_1) are the only viable options for GPU KV cache on this hardware.
