# KV-Cache Quantization × FlashAttention Compatibility Matrix

**Question:** Can you quantize only the K cache (keeping V at f16) and run with
FlashAttention **off**? And *why* does llama.cpp force FlashAttention on whenever
the V cache is quantized?

This is a focused 9-configuration sweep of `-ctk` (K type) × `-ctv` (V type) ×
`-fa` (on/off) to answer both, with a source-level root-cause investigation of
the V-requires-FA constraint.

## TL;DR

1. **Yes — K-only quantization works with FA off.** `-ctk q4_0 -ctv f16 -fa off`
   loads, runs, and produces correct output. This is a valid, useful
   configuration (faster decode than any FA-on config, with VRAM savings).
2. **Yes — `-ctk q8_0 -ctv f16 -fa off` also works.**
3. **Quantized V unconditionally requires FA.** `-ctv q4_0 -fa off` is rejected
   at context creation with `V cache quantization requires flash_attn`. There is
   **no** path to quantize V with FA off.
4. **Root cause (source-verified):** the non-FA attention path computes
   `output = Vᵀ @ softmax(QKᵀ)` via `ggml_mul_mat(V, kq)`, which requires V to be
   **physically transposed** in memory. Block-quantized tensors cannot be
   transposed (transposition interleaves block boundaries that the quantized
   copy/GEMM kernels cannot re-pack). FA avoids this with a **fused tiled kernel
   that dequantizes V in-place per tile** — no transpose needed. K is unaffected
   because it is consumed *directly* by `mul_mat` without transposition.
5. **`turbo3` is not a valid `-ctk`/`-ctv` type** on the stock ROCm 7.1.4 build
   (only the TurboQuant-patched build exposes it).

## Setup

- **Model:** DeepSeek-V2-Lite (16B MoE), `dsv2lite-q8_0.gguf` (~15.9 GB, Q8_0),
  27 blocks, Multi-head Latent Attention (`kv_lora_rank=512`,
  `key_length=192`, `value_length=128`) → small KV cache.
- **Hardware:** AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e). 2× MI210 in
  the host; tests ran single-GPU via `HIP_VISIBLE_DEVICES=1` inside a
  `llama-rocm714:latest` container (`kvtest`).
- **Binary:** `llama-server` at `/src/build/bin/llama-server` (stock ROCm 7.1.4
  build, GNU toolchain, `-DGPU_TARGETS=gfx90a -DGGML_HIP=ON`). This is **not**
  the TurboQuant build — `turbo2/3/4` cache types are unavailable here.
- **Server flags (all configs):** `-ngl 99 -c 4096 -np 1 --host 0.0.0.0 --port
  8099 --no-webui`, plus the per-config `-ctk/-ctv/-fa`.
- **Prompts:**
  - **Correctness** — `"What is 2+2? Answer with just the number."` (14 tokens),
    `n_predict=8`, `temperature=0`.
  - **Throughput** — a ~407-token passage, `n_predict=32`, after one warmup
    request so HIP kernels are JIT'd. (Short-prompt prefill is latency-bound and
    not meaningful; the warm 407-token run saturates the GPU.)
- **Correctness check:** all passing configs emitted `2+2=4` / `4`.

> Note on prefill numbers: see `comprehensive-kv-fa-matrix.md` for ~2000-token
> prompt figures on the same hardware. The relative ordering (FA-off prefill >
> FA-on prefill on gfx90a) is consistent across prompt lengths.

## Results — All 9 Configurations

| # | `-ctk` | `-ctv` | `-fa` | Loads? | Prefill (tok/s) | Decode (tok/s) | Correct? | Total GPU VRAM |
|---|--------|--------|-------|:------:|----------------:|---------------:|:--------:|---------------:|
| A | f16    | f16    | off   | ✅     | **2016.7**      | **156.6**      | ✅ `4`   | 17 516 MB       |
| B | **q4_0** | **f16**  | **off** | ✅ | **1575.4**      | **155.0**      | ✅ `4`   | **17 053 MB**   |
| C | **q8_0** | **f16**  | **off** | ✅ | **1596.6**      | **161.3**      | ✅ `4`   | **17 215 MB**   |
| D | f16    | q4_0   | off   | ❌     | —               | —              | —        | —               |
| E | f16    | q4_0   | on    | ✅     | 243.0†          | 98.2†          | ✅ `4`   | 17 102 MB       |
| F | q4_0   | q4_0   | on    | ✅     | 981.6           | 86.8           | ✅ `4`   | **16 636 MB**   |
| G | q4_0   | f16    | on    | ✅     | 251.3†          | 104.0†         | ✅ `4`   | 16 952 MB       |
| H | q8_0   | q4_1   | on    | ✅     | 913.7           | 79.2           | ✅ `4`   | (≈ F)           |
| I | turbo3 | f16    | off   | ❌     | —               | —              | —        | —               |

† Pre/decode from the short 14-token correctness prompt (FA-on configs E/G were
not re-run at 407 tokens; F and H were). FA-on decode is consistently ~80–110
tok/s on this hardware — well below the FA-off ~155–161 tok/s.

**Config D failure — exact error:**
```
E llama_init_from_model: V cache quantization requires flash_attn
E common_fit_params: encountered an error while trying to fit params to free device memory: failed to create llama_context from model
```
**Config I failure — exact error:**
```
error while handling argument "-ctk": Unsupported cache type: turbo3
allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
```

### What the matrix says

- **The K/V quantization constraint is asymmetric.** Quantizing **K** is always
  optional and independent of FA. Quantizing **V** forces FA on. Config B/C prove
  K-only quant works without FA; config D proves V-only quant cannot bypass FA.
- **On gfx90a, FA off wins on both axes** for single-stream decode/chat: faster
  prefill (2017 vs ~900–980 tok/s) *and* faster decode (156 vs ~80–87 tok/s).
  FA on only helps multi-batch / long-context prefill where its tiling pays off;
  for a single decode stream the ROCm FA kernel adds overhead per step.
- **Best decode-heavy config:** **B (`-ctk q4_0 -ctv f16 -fa off`)** — 155 tok/s
  decode, correct, and 463 MB less VRAM than the f16/f16 baseline. This is the
  sweet spot if you want KV savings without the FA decode tax.
- **Lowest VRAM config that loads:** **F (`-ctk q4_0 -ctv q4_0 -fa on`)** —
  16 636 MB (880 MB under baseline), at the cost of ~2× slower decode.

## Why Quantized V Requires FlashAttention — Root Cause

### The guard

The hard rejection lives in `llama-context.cpp`, in `llama_init_from_model()`
(`src/src/llama-context.cpp:3562`):

```cpp
if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
    LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
    return nullptr;
}
```

There is **no equivalent guard for K** — only a block-size divisibility check
that is itself skipped when FA is disabled. K quantization is unrestricted.

### Why the guard exists: the V-transpose problem

The non-FA and FA attention paths consume V *differently*. In
`llama-graph.cpp` (`src/src/llama-graph.cpp`, `build_attn()`):

**Non-FA path** (lines 2488–2494) — explicit QK→softmax→AV decomposition:
```cpp
if (!v_trans) {
    // note: the V cache is transposed when not using flash attention
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));   // (1) physical transpose
    cb(v, "v_cont", il);
}
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);       // (2) kqv = Vᵀ @ kq
```
K, by contrast, is used **directly** with no transpose (line 2450):
```cpp
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);         // kq = Kᵀ @ Q  (K read as-is)
```

**FA path** (lines 2411–2424) — single fused op:
```cpp
if (v_trans) {
    v = ggml_transpose(ctx0, v);                      // metadata-only; undone if needed
}
cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, ...);
```

The asymmetry is the whole story:

| | K | V (non-FA) | V (FA) |
|---|---|---|---|
| Needs physical transpose? | **No** — fed straight into `mul_mat` | **Yes** — `ggml_cont(ggml_transpose(v))` | **No** — read in native cache layout |
| Works block-quantized? | **Yes** — quantized `mul_mat` reads blocks along the contiguous dim | **No** — transpose interleaves block boundaries | **Yes** — fused kernel dequantizes per tile |

### Why transposing a block-quantized tensor is broken

Block-quantized types (q4_0, q8_0, k-quants, …) pack `blck_size` elements
(32 for q4_0/q8_0, 256 for k-quants) into a single self-contained block:
`[scale | quantized deltas]`. Blocks are laid out contiguously along dimension 0
(`ne[0]`); the per-element stride `nb[0]` equals the block granularity, not 1
element.

`ggml_transpose` is metadata-only (it swaps `nb[1]`/`nb[2]`). The subsequent
`ggml_cont` must then physically re-pack data into the new contiguous order —
which means gathering elements *across* block boundaries and re-blocking them
along what used to be the sequence dimension. ggml's quantized copy/dup kernels
only know how to copy **whole intact blocks row-by-row**; they have no
dequantize→shuffle→requantize path. So `ggml_cont(ggml_transpose(v_quant))`
would either assert, copy garbage, or misalign blocks. Rather than risk silent
numerical corruption in attention (which produces plausible-looking but wrong
tokens), llama.cpp rejects the combination up front.

Two reinforcing ggml facts:

- `ggml_mul_mat` itself asserts `!ggml_is_transposed(a)` for its first operand
  (`ggml.c:3283`). The non-FA code satisfies this by making V contiguous first,
  but that's exactly the step that can't be done for quantized V.
- The block-size divisibility check for K/V quantization
  (`llama-context.cpp:3546–3559`) only runs when FA is **not** disabled —
  another tell that the non-FA path simply isn't wired for quantized caches.

### Is the guard removable?

**No, not as a one-line patch.** Removing the `llama-context.cpp:3562` check
would let context creation succeed, but the first attention op would then hit
either a `ggml_cont`/copy failure on the quantized transposed V or — worse —
silently corrupt output. Making V-quant-without-FA work would require
implementing a transpose-capable dequantize-requantize copy kernel (or a
non-FA attention variant that reads V in native layout), which is real GPU
kernel work, not a config flip. FA is the cheap, correct solution: its fused
kernel already does inline per-tile dequantization, so quantized V "just works."

## KV-Cache VRAM Breakdown (4096 ctx)

Total GPU VRAM = model weights (≈15.9 GB, constant) + KV cache + compute buffer
+ overhead. Deltas are vs config A (f16/f16, FA off):

| Config | K | V | FA | Total VRAM | Δ vs A | What changed |
|--------|------|------|-----|-----------:|-------:|--------------|
| A | f16  | f16  | off | 17 516 MB | —      | baseline |
| C | q8_0 | f16  | off | 17 215 MB | −301 MB | K 8-bit (½ size) |
| B | q4_0 | f16  | off | 17 053 MB | −463 MB | K 4-bit (¼ size) |
| G | q4_0 | f16  | on  | 16 952 MB | −564 MB | K 4-bit + FA compute buffer shrink |
| E | f16  | q4_0 | on  | 17 102 MB | −414 MB | V 4-bit (FA on, required) |
| F | q4_0 | q4_0 | on  | 16 636 MB | −880 MB | both 4-bit + FA |

> FA-on configs also change the compute buffer size vs FA-off, so the deltas
> conflate KV-cache savings with compute-buffer differences. The clean K-only
> comparison is **A→B→C** (all FA off): K quantization saves 300–460 MB at 4096
> context, scaling linearly with context length.

For DeepSeek-V2-Lite's MLA the absolute KV cache is small (the latent is
compressed to `kv_lora_rank=512`), so even the full f16 cache is only ~1–1.2 GB
at 4096 ctx — the 16 GB model dominates. KV quantization matters much more for
standard-attention models and/or long contexts (32k–128k), where the cache, not
the weights, is the memory bottleneck.

## Conclusions

1. **K-only KV quantization + FA off is a first-class working configuration.**
   `-ctk q4_0 -ctv f16 -fa off` (config B) is the recommended decode-optimized
   setup on gfx90a: full correctness, 155 tok/s decode, 463 MB saved.
2. **V quantization is gated on FA by a hard, correct guard** at
   `llama-context.cpp:3562`, not a missing feature flag. The technical reason is
   the non-FA attention's mandatory V transpose, which is incompatible with
   block-quantized memory layout. FA's fused tiled kernel sidesteps the transpose
   entirely via inline dequantization.
3. **On MI210, prefer FA off** for single-stream serving — it is faster on both
   prefill and decode in every like-for-like comparison here. Reserve FA on for
   the one thing it unlocks: quantized V (config F) when VRAM is the binding
   constraint.
4. **`turbo3` requires the TurboQuant build** (`tqbuild`); the stock ROCm 7.1.4
   binary only accepts `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`.

## Reproduction

Container + runner scripts are one-shot; the essential loop per config:
```bash
docker exec -d kvtest bash -c "/src/build/bin/llama-server \
  -m /models/dsv2lite-q8_0.gguf -ngl 99 -c 4096 -np 1 \
  -ctk <TYPE_K> -ctv <TYPE_V> -fa <on|off> \
  --host 0.0.0.0 --port 8099 --no-webui > /tmp/test.log 2>&1"
# wait for 'model loaded' or the 'V cache quantization requires flash_attn' error,
# then POST /completion and read timings.prompt_per_second / .predicted_per_second.
```
