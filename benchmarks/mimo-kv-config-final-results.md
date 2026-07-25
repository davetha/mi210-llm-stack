# MiMo 230B KV Configuration Final Results

## Date: 2026-07-25

## Setup
- Model: Huihui-MiMo-V2.5-abliterated Q4_K (230B MoE, 174GB, 21 shards)
- Hardware: 2× AMD MI210 (gfx90a/CDNA2, 64GB each), EPYC 74F3, 499GB RAM
- Binary: TurboQuant llama.cpp fork (with rdna2 occupancy patch)
- Split: -ot blks 0-24 FFN→CPU, 25-36→ROCm0, 37-48→ROCm1
- Prompt: ~15K tokens (realistic opencode system prompt size)

## Results

### 65K Context Tests
| Config | Prefill (tok/s) | Prefill Time (15K) | Decode (tok/s) | Status |
|---|---|---|---|---|
| q8_0/q4_1 + TurboQuant + graphs off | **392** | **38.8s** | 19.9 | ✅ WINNER |
| q8_0/q4_1 + stock binary | ~130 | ~115s | 16 | Old production |
| turbo3/turbo3 | 457* | — | CRASH | ❌ Decode graph bug |
| turbo3 K + q4_1 V | 24 | — | ok | ❌ 16× slower |
| kivi2/kivi2 | CRASH | — | — | ❌ No GPU SET_ROWS kernel |
| q4_0/f16 FA-off | OOM | — | — | ❌ f16 V too large |

*turbo3 prefill measured before decode crash

### Key Discovery: Binary Speedup
The TurboQuant binary is **3× faster at prefill** than stock llama.cpp for the SAME KV config (q8_0/q4_1):
- TurboQuant binary: 392 tok/s
- Stock binary: ~130 tok/s
- Same model, same KV types, same GPU — different binary only

### Why Custom Types Fail on GPU
- **kivi2**: `SET_ROWS` operation not implemented in HIP backend for custom GGML types
- **turbo3**: Crashes during CUDA graph capture (decode path)
- **Both work on CPU only** — the GPU path needs custom HIP kernels that don't exist

### Winning Configuration
```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
/path/to/turboquant/llama-server \
  -m mimo.gguf \
  -ngl 999 -c 65536 -b 2048 -ub 2048 -np 1 \
  -ctk q8_0 -ctv q4_1 -fa on \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,..." \
  --no-warmup
```

Critical flags:
- `GGML_CUDA_DISABLE_GRAPHS=1`: Fixes decode crash with TurboQuant binary
- `-fa on`: Required for quantized V cache (q4_1)
- TurboQuant binary: 3× faster prefill than stock

### 256K Context Feasibility
At 256K context, q8_0/q4_1 KV cache = ~27.7 GB. With ~40 GB free VRAM for KV, this fits but is tight.
turbo3/kivi2 (which would be ~11-14 GB) are NOT viable on GPU. q4_0/q4_1 (~18 GB) is the most compact viable option.

### Root Cause of Original 200+ Second Prefill
1. Crashed zombie container (turbo3 256K test) leaking 101 GB VRAM
2. Stock llama.cpp binary 3× slower than TurboQuant binary
3. No session persistence warm cache on cold start
