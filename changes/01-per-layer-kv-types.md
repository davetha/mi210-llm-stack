# Change 01 — Per-Layer KV Cache Types (`-ctk-cpu` / `-ctv-cpu`)

The headline feature for CPU-hybrid MoE: independent KV cache quantization for
CPU-pinned layers vs GPU layers.

## What

Two new CLI flags:

```
-ctk-cpu, --cache-type-k-cpu TYPE    KV cache data type for K on CPU-pinned layers (inherits -ctk if unset)
-ctv-cpu, --cache-type-v-cpu TYPE    KV cache data type for V on CPU-pinned layers (inherits -ctv if unset)
```

## Why

mimo is a 230B MoE with 25 of its 48 expert layers pinned to CPU via `-ot`. On
those CPU layers, attention re-reads the growing KV from DDR4 every token —
the prefill bottleneck. Compressing **only the CPU layers** with TurboQuant
(turbo3, 5.3× compression, **proven correct on CPU**) cuts DDR4 traffic 5× on
exactly the layers that are bandwidth-bound, while keeping GPU layers at full
fp16 quality. The GPU TurboQuant bug (see [change 03](03-turboquant-wave64-fixes.md))
becomes irrelevant because GPU layers never touch turbo.

## Files modified (12)

| File | Change |
|------|--------|
| `common/common.h` | Added `cache_type_k_cpu` / `cache_type_v_cpu` fields (default `GGML_TYPE_COUNT` = inherit). |
| `common/arg.cpp` | Added `-ctk-cpu` / `-ctv-cpu` CLI options + KIVI2 enum registration. |
| `common/common.cpp` | Pass `cache_type_{k,v}_cpu` into `llama_context_params`. |
| `include/llama.h` | Added `type_k_cpu` / `type_v_cpu` to `llama_context_params`. |
| `src/llama-memory.h` | Added `type_k_cpu` / `type_v_cpu` to `llama_memory_params`. |
| `src/llama-context.cpp` | Pass CPU types to memory params + default params (`GGML_TYPE_COUNT`). |
| `src/llama-model.cpp` | `create_memory()` forwards CPU types to kv-cache constructor. |
| `src/llama-kv-cache.h` | Constructor signature: +2 params (`type_k_cpu`, `type_v_cpu`). |
| `src/llama-kv-cache.cpp` | Sentinel resolution + **per-layer type selection** + **rotation matrix fix**. |
| `src/llama-kv-cache-dsa.cpp` | Pass `GGML_TYPE_COUNT, GGML_TYPE_COUNT` to sub-caches (no per-layer split). |
| `src/llama-kv-cache-iswa.cpp` | Same — pass sentinel to base/SWA sub-caches. |
| `src/llama-memory-hybrid.cpp` | Same — hybrid inherits attn type. |

## How it works

### Sentinel resolution (`llama-kv-cache.cpp`)

```cpp
// GGML_TYPE_COUNT means "inherit the GPU type":
if (type_k_cpu == GGML_TYPE_COUNT) type_k_cpu = type_k;
if (type_v_cpu == GGML_TYPE_COUNT) type_v_cpu = type_v;
```

So unset = backward compatible (all layers use `-ctk`/`-ctv`).

### Per-layer selection (`llama-kv-cache.cpp`)

```cpp
const bool is_cpu_layer = (strcmp(dev_name, "CPU") == 0);
ggml_type layer_type_k = is_cpu_layer ? type_k_cpu : type_k;
ggml_type layer_type_v = is_cpu_layer ? type_v_cpu : type_v;
```

Each layer's KV tensors are created with the device-appropriate type.

### Rotation matrix fix (`llama-kv-cache.cpp`)

TurboQuant needs WHT rotation tensors. Previously they were only created when
`type_k` (the GPU type) was turbo. Now the check includes `type_k_cpu` too:

```cpp
if (turbo_rotation == nullptr &&
    (type_k == GGML_TYPE_TURBO3_0 || ... ||
     type_k_cpu == GGML_TYPE_TURBO3_0 || ...)) {
    turbo_rotation = ggml_new_tensor_2d(...);
```

So `-ctk f16 -ctk-cpu turbo3` correctly sets up the rotation matrices for the
CPU layers even though the GPU layers are plain fp16.

## Build

```
EXIT=0  for both llama-cli and llama-server
```

Built with the standard gfx90a flags (see [`davetha/llama.cpp-mi210/BUILD.md`](https://github.com/davetha/llama.cpp-mi210/blob/main/BUILD.md)).

## Usage

```bash
llama-server -m mimo.gguf -ngl 999 \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1" \
  -ctk f16 -ctv f16 -ctk-cpu turbo3 -ctv-cpu turbo3 -fa on
```

- GPU layers (25–48): fp16 KV — maximum quality.
- CPU layers (0–24): turbo3 KV — 5.3× compressed → 5× less DDR4 traffic.

## Patch

→ [`patches/01-per-layer-kv-types.patch`](https://github.com/davetha/llama.cpp-mi210/blob/main/patches/01-per-layer-kv-types.patch)
