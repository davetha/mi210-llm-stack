# KIVI, RotateKV, and KV-Cache Compression Options on gfx90a

A comparison of every KV-cache compression approach evaluated for the MI210, with the KIVI implementation details.

## The problem

On a full-attention MoE like mimo (230B, 48 layers, 25 CPU-pinned), the KV cache scales with `context × layers`. A 64K KV across 48 layers is tens of GB. When 25 of those layers run on CPU, **attention re-reads the growing KV from DDR4 every token** — ~7.4 s/chunk. Compressing the CPU-layer KV directly cuts DDR4 traffic.

## KIVI (arXiv:2402.02750) — IMPLEMENTED ✅

**Algorithm:** 2-bit **asymmetric** quantization with a key structural insight from the paper — K and V have different outlier patterns:

| Tensor | Granularity | Why |
|--------|-------------|-----|
| **K** (keys) | **per-channel** | outliers cluster in specific channels |
| **V** (values) | **per-token** | outliers cluster in specific tokens |

Plus a **128-token FP16 residual** (the most recent tokens stay full precision to bound the error of the rapidly-changing tail). Group size = 32.

### Our implementation: `GGML_TYPE_KIVI2`

We implemented a simplified, hardware-agnostic scalar version (no per-channel vs per-token split in the block layout — uniform per-group min-max), exposed as a new ggml type:

- **`block_kivi2`**: 2-byte fp16 scale `d` + 2-byte fp16 min `m` + 8 bytes packed 2-bit indices (4 per byte) = **12 bytes / 32 values = 3.0 bits/value** → 5.3× compression vs fp16.
- **Algorithm:** per-group (group_size=32) min-max asymmetric quant, 4 levels `{0,1,2,3}`, `scale = (max-min)/3`, reconstruct as `idx*scale + min`.
- **Pure scalar C** — no wave-level intrinsics, so it is correct on **every** architecture including gfx90a wave64.

**Files (9):** `ggml.h` (enum), `ggml-common.h` (block struct), `ggml-quants.{c,h}` (quantize/dequantize), `ggml.c` (type traits), `ggml-cpu.c` (CPU traits + vec_dot), `quants.{c,h}` (dispatch), `arg.cpp` (registration).

**Correctness:** all 4 unit tests **PASS**:
1. Exact 4-level round-trip on known values.
2. Endpoints (min/max map exactly).
3. Constant block (zero scale → exact).
4. Random invariance (quantize→dequantize preserves statistics).

**Usage:**
```bash
llama-server -m model.gguf -ctk kivi2 -ctv kivi2 ...
# or, with per-layer split:
llama-server -m model.gguf -ctk f16 -ctv f16 -ctk-cpu kivi2 -ctv-cpu kivi2 ...
```

See [`changes/02-kivi2-quant-type.md`](../changes/02-kivi2-quant-type.md) and [`davetha/llama.cpp-mi210`](https://github.com/davetha/llama.cpp-mi210).

## RotateKV — EVALUATED (not implemented)

**Algorithm:** Fast Walsh–Hadamard Transform (FWHT) + channel reordering to push outliers into a few channels, then 2-bit quant on the rotated space, with **attention sink protection** (the first few tokens' KV kept at full precision — they're referenced by every subsequent query).

**Status:** evaluated but **not implemented**. The FWHT rotation is mathematically identical to TurboQuant's WHT step, and the quantization is simpler. Given that:
1. TurboQuant's WHT (CPU path) is already proven correct,
2. KIVI2 gives a clean 5.3× compression with no rotation needed,

RotateKV would only be worth implementing if attention-sink protection proves necessary for quality at 2-bit — which KIVI's 128-token residual already handles.

## Comparison table

| Method | Bits/value | Compression vs fp16 | gfx90a GPU? | gfx90a CPU? | Status |
|--------|-----------:|--------------------:|:-----------:|:-----------:|--------|
| **fp16** (baseline) | 16.0 | 1× | ✅ | ✅ | default |
| **q8_0** | 8.5 | 1.9× | ✅ | ✅ | mainline |
| **q4_1** (V cache sweet spot) | 5.0 | 3.2× | ✅ | ✅ | mainline |
| **q4_0** | 4.5 | 3.6× | ✅ | ✅ | mainline |
| **KIVI2** | 3.0 | 5.3× | ✅ (scalar) | ✅ | ✅ **implemented** |
| **TurboQuant turbo3** (3-bit) | ~3.0 | ~5.3× | ❌ (wave64) | ✅ (proven) | ⚠️ partial |
| **TurboQuant turbo4** (4-bit) | ~4.0 | ~4.0× | ❌ (wave64) | ✅ (proven) | ⚠️ partial |
| **RotateKV** | ~4.0 | ~4.0× | — | — | evaluated only |

## What we actually run

For the CPU-pinned layers (the DDR4-bandwidth bottleneck):
```
-ctk f16 -ctv f16 -ctk-cpu turbo3 -ctv-cpu turbo3
```
Turbo3 on CPU gives 5.3× compression (proven correct). GPU layers stay at fp16 for maximum quality. This is enabled by the **per-layer KV types** feature ([`changes/01`](../changes/01-per-layer-kv-types.md)).

KIVI2 is available as an alternative (`-ctk-cpu kivi2`) — same 3.0 bpw, hardware-agnostic, and useful if you want to compress GPU layers too (TurboQuant's GPU path is broken on gfx90a; KIVI2's pure-scalar path works everywhere).
