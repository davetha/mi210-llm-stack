# Change 11 — Deploy K-only quantization + FA-off to production MiMo 230B

Deploy the proven optimal KV cache configuration (`-ctk q4_0 -ctv f16 -fa off`)
to the production MiMo 230B instance.

## 1. Summary

| Setting | Current (launch-mimo.sh) | Proposed (launch-mimo-optimized.sh) |
|---|---|---|
| `-ctk` (K cache type) | `q8_0` (8.5 bit) | `q4_0` (4.5 bit) |
| `-ctv` (V cache type) | `q4_1` (5 bit) | `f16` (16 bit) |
| `-fa` (FlashAttention) | `on` | `off` |
| Context length | 65536 | 65536 |
| 3-way CPU/GPU split | unchanged | unchanged |
| Warm session restore | unchanged | unchanged |

## 2. Why this change

Benchmarks from the [KV×FA benchmark matrix](../benchmarks/comprehensive-kv-fa-matrix.md)
proved that `-ctk q4_0 -ctv f16 -fa off` is the winning configuration on gfx90a:

1. **FA-off avoids the broken FA fallback** — llama.cpp's FlashAttention on gfx90a
   is a software fallback that is 2.6× slower at prefill and 6.7× slower at decode
   compared to the native attention path (measured on DeepSeek-V2-Lite).
2. **K cache compressed 47%** — q4_0 (4.5 bit) vs q8_0 (8.5 bit) halves the K cache
   size, saving 2.18 GB at 65536 context.
3. **f16 V required for FA-off** — quantized V types (q4_1, q8_0, q4_0) are rejected
   by llama.cpp at context creation unless `-fa on`. f16 is the only standard type
   that loads without FA.
4. **Correctness verified** — the K-only quant config was tested on the same gfx90a
   hardware and produced correct outputs.

## 3. VRAM analysis (CRITICAL)

Model architecture extracted from GGUF metadata:

| Parameter | Value |
|---|---|
| Architecture | `mimo2` (MoE 256×8.2B) |
| Layers | 51 |
| KV head distribution | 9 layers with 4 KV heads, 42 with 8 |
| Key length per head | 192 |
| Value length per head | 128 |
| Context | 65536 |

### KV cache size by type (at 65536 context)

| KV cache component | Current | Proposed | Delta |
|---|---|---|---|
| K cache | q8_0: **4.63 GB** | q4_0: **2.45 GB** | **-2.18 GB** (-47%) |
| V cache | q4_1: **1.82 GB** | f16: **5.81 GB** | **+4.00 GB** (+220%) |
| **Total KV cache** | **6.45 GB** | **8.26 GB** | **+1.81 GB** |

### Free VRAM headroom

Current usage (rocm-smi at session time):

| GPU | Total | Used | Free |
|---|---|---|---|
| GPU0 (ROCm0) | 64.0 GB | ~47 GB | **~17 GB** |
| GPU1 (ROCm1) | 64.0 GB | ~51 GB | **~13 GB** |

The +1.81 GB delta is well within the available headroom on both GPUs. Even if the
full delta lands on a single GPU, neither is at risk of OOM.

### Calculation details

Total KV elements per token (across all 51 layers):

- K cache: Σ(n_kv_head × key_length) = **71,424 elements/token**
  - 9 layers × 4 heads × 192 = 6,912
  - 42 layers × 8 heads × 192 = 64,512
- V cache: Σ(n_kv_head × val_length) = **47,616 elements/token**
  - 9 layers × 4 heads × 128 = 4,608
  - 42 layers × 8 heads × 128 = 43,008

Bytes per element by type:
- q8_0: 34 bytes / 32 elements = 1.0625 B/elem
- q4_0: 18 bytes / 32 elements = 0.5625 B/elem
- q4_1: 20 bytes / 32 elements = 0.625 B/elem
- f16: 2 bytes / element

Total at N=65536:
- K(q8_0) = 71,424 × 65,536 × 1.0625 = 4.63 GB
- K(q4_0) = 71,424 × 65,536 × 0.5625 = 2.45 GB
- V(q4_1) = 47,616 × 65,536 × 0.625 = 1.82 GB
- V(f16) = 47,616 × 65,536 × 2 = 5.81 GB

## 4. Expected performance impact

Based on the DSV2-Lite benchmark (same gfx90a hardware, same llama.cpp build):

| Metric | FA-on (current) | FA-off (proposed) | Expected gain |
|---|---|---|---|
| Prefill (tok/s) | ~146-178 | ~380-460* | **~2.6×** |
| Decode (tok/s) | ~22 | ~147* | **~6.7×** |

\* Caveat: MiMo's 3-way CPU/GPU split may reduce these gains vs. DSV2-Lite's
all-GPU topology. The FFN-offloaded layers (blks 0-24 → CPU) mean prefill is
partially CPU/PCIe-bound. However, decode should still benefit significantly
because attention dominates decode latency and FA-off removes the slow software
FA fallback for every decode token.

## 5. Deployment steps

```bash
# 1. Drain production mimo slot in llama-swap
#    (operator action — mark slot draining, wait for in-flight requests to finish)

# 2. Stop the current container
docker stop llama-main

# 3. Deploy the optimized config
cp /mnt/llm-storage/launch-mimo-optimized.sh /mnt/llm-storage/launch-mimo.sh

# 4. Start with the new config
sh /mnt/llm-storage/launch-mimo.sh <port>

# 5. Verify:
#    - Health check passes
#    - Warm session restores (if applicable)
#    - Correctness: prompt "2+2=" → response "4"
#    - Monitor VRAM: rocm-smi --showmeminfo vram
```

## 6. Rollback

If any issue occurs:

```bash
# Restore from git: the original launch-mimo.sh is on GitHub
GIT_MASTER=1 git checkout origin/main -- configs/launch-mimo.sh
GIT_MASTER=1 git checkout origin/main -- configs/launch-mimo-optimized.sh
```

## 7. VRAM scenarios at reduced contexts (in case of issues)

| Context | K(q4_0) | V(f16) | Total | vs. current (6.45 GB) |
|---|---|---|---|---|
| **65536** | 2.45 GB | 5.81 GB | **8.26 GB** | +1.81 GB |
| 49152 | 1.84 GB | 4.36 GB | **6.20 GB** | -0.25 GB |
| 32768 | 1.23 GB | 2.91 GB | **4.13 GB** | -2.32 GB |
| 16384 | 0.61 GB | 1.45 GB | **2.07 GB** | -4.38 GB |

All scenarios at 65536+ fit within available VRAM. The config file keeps 65536.
