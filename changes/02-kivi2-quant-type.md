# Change 02 — KIVI 2-Bit KV Cache Quantization (`GGML_TYPE_KIVI2`)

A new, hardware-agnostic 2-bit KV-cache quantization type based on the
[KIVI paper (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750).

## What

`GGML_TYPE_KIVI2` — 2-bit asymmetric KV cache quantization:

- **Algorithm:** per-group (group_size = 32) min-max asymmetric quantization.
- **Levels:** 4 `{0,1,2,3}`, `scale = (max-min)/3`, reconstruct as `idx*scale + min`.
- **Block layout:** `block_kivi2` = 2-byte fp16 scale `d` + 2-byte fp16 min `m` + 8 bytes packed 2-bit indices (4 per byte) = **12 bytes / 32 values = 3.0 bits/value** → 5.3× compression vs fp16.
- **Pure scalar C** — no wave-level intrinsics → correct on **every** architecture including gfx90a wave64.

```c
#define QK_KIVI2 32
typedef struct {
    ggml_half d;                 //  2 bytes: scale (delta) = (max-min)/3
    ggml_half m;                 //  2 bytes: min
    uint8_t   qs[QK_KIVI2 / 4];  //  8 bytes: 2-bit indices (4 per byte)
} block_kivi2;                   // 12 bytes total
```

## Files modified (9)

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | `GGML_TYPE_KIVI2 = 47` enum entry. |
| `ggml/src/ggml-common.h` | `block_kivi2` struct + `QK_KIVI2` define + static_assert(12 bytes). |
| `ggml/src/ggml-quants.h` | Declarations: `quantize_row_kivi2_ref`, `dequantize_row_kivi2`, `quantize_kivi2`. |
| `ggml/src/ggml-quants.c` | Implementations: quantize (min-max + 4-level), dequantize, `quantize_kivi2` row dispatcher, `ggml_validate_row_data` case. |
| `ggml/src/ggml.c` | Type traits: `type_name="kivi2"`, `blck_size=32`, `type_size=12`, `is_quantized=true`, to_float/from_float_ref. `ggml_quantize_chunk` case. |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU type traits: `from_float`, `vec_dot` (`ggml_vec_dot_kivi2_f32`), `vec_dot_type=F32`. |
| `ggml/src/ggml-cpu/quants.c` | `quantize_row_kivi2` dispatch → `_ref`. |
| `ggml/src/ggml-cpu/quants.h` | `quantize_row_kivi2` declaration. |
| `common/arg.cpp` | Register `GGML_TYPE_KIVI2` in the `kv_cache_types` list. |

## Algorithm detail

### Quantize (`quantize_row_kivi2_ref`)

```c
for each group of 32 values:
    min = min(values);  max = max(values);
    d = (max - min) / 3;   // 4 levels: 0,1,2,3
    for each value:
        idx = round((value - min) / d);   // clamp to [0,3]
        pack idx as 2 bits (4 per byte)
    store d (fp16), min (fp16), packed qs (8 bytes)
```

### Dequantize (`dequantize_row_kivi2`)

```c
for each block:
    d = fp16_to_fp32(block.d);
    m = fp16_to_fp32(block.m);
    for each of 32 values:
        idx = (block.qs[j/4] >> ((j%4)*2)) & 0x3;
        out[j] = idx * d + m;
```

### Vec dot (`ggml_vec_dot_kivi2_f32`)

Dequantizes the kivi2 block to f32, then dots with the f32 operand. Simple and correct (no fused kernel needed).

## Correctness — all 4 tests PASS

| Test | What it checks | Result |
|------|----------------|--------|
| Exact 4-level | Known values → exact round-trip at each level | ✅ PASS |
| Endpoints | min/max map to level 0/3 exactly | ✅ PASS |
| Constant block | All-same value (zero scale) → exact | ✅ PASS |
| Random invariance | Quantize→dequantize preserves statistics | ✅ PASS |

## Usage

```bash
# Uniform (all layers):
llama-server -m model.gguf -ctk kivi2 -ctv kivi2 ...

# Per-layer (CPU layers kivi2, GPU layers fp16):
llama-server -m model.gguf -ctk f16 -ctv f16 -ctk-cpu kivi2 -ctv-cpu kivi2 ...
```

KIVI2 is the hardware-agnostic alternative to TurboQuant — same 3.0 bpw, and
correct on every CPU architecture, where TurboQuant's wave64 path is broken.

**KIVI2 is CPU-only.** An earlier version of this line claimed it "works on GPU
too". It does not. "Pure scalar C" means architecture-agnostic, not
GPU-capable — every one of the nine files in the table above is host-side
(`ggml-quants.c`, `ggml-cpu/*`), and no CUDA or HIP backend defines a KIVI2
dequant kernel, so a `GGML_TYPE_KIVI2` tensor has no reader on the device. The
per-layer example directly above is the tell: it pairs `-ctk-cpu kivi2` with
`-ctk f16` precisely because the GPU layers must stay fp16.

This is the right tool for CPU-pinned layers re-reading KV from DDR4. It does
nothing for a `-ngl 999` deployment. See [`docs/54`](../docs/54-kv-compression-on-vllm-and-llamacpp.md).

## Patch

→ [`patches/02-kivi2-quant-type.patch`](https://github.com/davetha/llama.cpp-mi210/blob/main/patches/02-kivi2-quant-type.patch)
