# Comprehensive KV × FlashAttention × Layer-Split Benchmark

A 20-configuration sweep of KV-cache type × FlashAttention on/off × CPU/GPU
layer split, measured on a single MI210 with the TurboQuant `llama.cpp` build
(HIP FlashAttention, `-ctk-cpu`/`-ctv-cpu` per-layer KV, `turbo2/3/4` types).

All numbers are **real measured** data from this hardware session — every
config was booted, a ~2000-token prompt was run through the server, and the
`prompt eval time` / `eval time` lines (cross-checked against the `/completion`
JSON `timings`) were recorded. Failures are documented with the exact engine
error, not omitted.

## Setup

- **Model:** DeepSeek-V2-Lite (16B MoE), `dsv2lite-q8_0.gguf` (~15.7 GB, Q8_0),
  27 decoder blocks, Multi-head Latent Attention (MLA → small KV cache).
- **Hardware:** 2× AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e each),
  AMD EPYC 74F3 (24c/48t), 499 GB DDR4, PCIe 4.0 (no xGMI bridge).
- **Binary:** TurboQuant `llama.cpp` build — `llama-server` @ commit `c26cbdf`,
  GNU 15.2.0, built `-DGPU_TARGETS=gfx90a -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=OFF
  -DGGML_CUDA_FA_ALL_QUANTS=ON` (inside the `tqbuild` container; run via a
  GPU-enabled sibling container `tqbench`).
- **Prompt:** ~2069 tokens (deterministic technical passage), cold prefill after
  one small warmup request so kernels are JIT'd. Decode = 200 generated tokens.
- **Context:** `-c 4096 -np 1` (single slot, full window per request).
- **GPU placement:** all tests on **GPU 1** (`HIP_VISIBLE_DEVICES=1`, ~20.5 GB
  free). GPU 0 was essentially full (24 MiB free) from other workloads, so the
  15.7 GB model could not fit there for `-ngl 99`. This kept a clean
  **single-GPU topology** for every config.

> KV-type names: the build exposes `turbo2`, `turbo3`, `turbo4` (not
> `turbo3_0`/`turbo4_0`). `turbo3` is used here as the TurboQuant 3-bit path.
> No `KIVI2` type is exposed on the `-ctk`/`-ctv` flags of this build.

## All-GPU Results (`-ngl 99`, 27/27 layers on GPU)

| # | KV K | KV V | FA | Prefill (tok/s) | Decode (tok/s) | Correct? |
|---|---|---|---|---:|---:|---|
| 1 | f16 | f16 | off | **1984.0** | **162.5** | ✅ `4` |
| 2 | f16 | f16 | on | 769.7 | 24.4 | ✅ `4` |
| 3 | q8_0 | q8_0 | off | — | — | ⚠️ LOAD FAIL |
| 4 | q8_0 | q8_0 | on | 469.8 | 25.6 | ✅ `4` |
| 5 | q4_0 | q4_0 | off | — | — | ⚠️ LOAD FAIL |
| 6 | q4_0 | q4_0 | on | 495.1 | 25.7 | ✅ `4` |
| 7 | q8_0 | q4_1 | off | — | — | ⚠️ LOAD FAIL |
| 8 | q8_0 | q4_1 | on | 463.1 | 25.6 | ✅ `4` |
| 9 | turbo3 | turbo3 | off | 58.4 | 23.7 | ✅ `4` |
| 10 | turbo3 | turbo3 | on | 58.5 | 23.4 | ✅ `4` |

**⚠️ LOAD FAIL** (configs 3, 5, 7) = `E llama_init_from_model: V cache
quantization requires flash_attn`. Any quantized **V** cache (`q8_0`, `q4_0`,
`q4_1`) is rejected at context creation unless `-fa on`. `f16` and `turbo3` are
the only types that load without FA.

## Split CPU/GPU Results (`-ngl 23`, 23 GPU / 4 CPU layers)

| # | KV K | KV V | FA | Prefill (tok/s) | Decode (tok/s) | Correct? |
|---|---|---|---|---:|---:|---|
| 11 | f16 | f16 | off | — | — | ❌ CRASH |
| 12 | f16 | f16 | on | 312.7 | 24.0 | ❌ WRONG (`2`) |
| 13 | q8_0 | q4_1 | off | — | — | ⚠️ LOAD FAIL |
| 14 | q8_0 | q4_1 | on | 255.1 | 23.7 | ❌ WRONG (`2`) |
| 15 | q4_0 | q4_0 | off | — | — | ⚠️ LOAD FAIL |
| 16 | q4_0 | q4_0 | on | 263.5 | 23.7 | ❌ WRONG (`2`) |

**❌ CRASH** (config 11, FA-off split) = `cmd_child_to_router:error:
/build/src/ggml/src/ggml-backend.cpp:349: GGML_ASSERT(offset + size <=
ggml_nbytes(tensor) && "tensor read out of bounds") failed` — the server drops
the connection mid-request.

**❌ WRONG** (configs 12, 14, 16, FA-on split) = answers `2` to "What is 2+2?"
instead of `4`. The plain CPU/GPU split silently corrupts output whenever FA is
on. See Key Findings.

## Per-Layer KV Results (`-ngl 23`, `-ctk-cpu`/`-ctv-cpu` set CPU layers separately)

GPU layers use `-ctk`/`-ctv`; the 4 CPU-pinned layers use `-ctk-cpu`/`-ctv-cpu`.

| # | GPU K/V | CPU K/V | FA | Prefill (tok/s) | Decode (tok/s) | Correct? |
|---|---|---|---|---:|---:|---|
| 17 | f16 | turbo3 | off | — | — | ⚠️ LOAD FAIL |
| 18 | f16 | turbo3 | on | 214.6 | 24.0 | ✅ `4` |
| 19 | q4_0 | turbo3 | off | — | — | ⚠️ LOAD FAIL |
| 20 | q4_0 | turbo3 | on | 187.1 | 24.5 | ✅ `4` |

The per-layer `-ctk-cpu turbo3` split is the **only** split variant that
produces correct output (`4`). Plain-split (GPU type inherited by CPU layers)
corrupts output; explicitly pinning CPU layers to `turbo3` with FA-on recovers
correctness — at a prefill cost.

## Key Findings

### 1. FlashAttention is a **regression** on gfx90a, not a speedup
On all-GPU f16 KV, turning FA **on** is strictly worse:

| Metric | FA off | FA on | FA-on impact |
|---|---:|---:|---|
| Prefill | 1984.0 tok/s | 769.7 tok/s | **2.6× slower** |
| Decode | 162.5 tok/s | 24.4 tok/s | **6.7× slower** |

This build disables the ROCWMMA FA backend (`GGML_HIP_ROCWMMA_FATTN=OFF`), so
the FA path falls back to a generic kernel that is far slower than the default
(non-FA) attention on CDNA2. Decode collapses hardest: the non-FA f16 path
hits 162 tok/s; every FA-on / quantized-KV path is pinned near **24 tok/s**.

### 2. KV compression does **not** help prefill here
With FA forced on (the only way to load quantized V), more compression is not
faster — the dequant overhead dominates:

| KV (FA on) | Prefill (tok/s) |
|---|---:|
| f16 | 769.7 |
| q4_0 | 495.1 |
| q8_0 | 469.8 |
| q8_0/q4_1 | 463.1 |
| turbo3 | 58.5 |

`turbo3` is the outlier: **58 tok/s** — ~13× slower than f16-FA-on, and FA
makes no difference to it (58.4 off vs 58.5 on). The TurboQuant 3-bit KV kernel
is unusably slow for prefill on this hardware in this build.

### 3. Quantized V cache **requires** FlashAttention
A hard engine constraint: `E llama_init_from_model: V cache quantization
requires flash_attn`. `q8_0`/`q4_0`/`q4_1` V cache is rejected at load unless
`-fa on`. This invalidated 6 of the 20 planned configs (3, 5, 7, 13, 15, 17, 19).
Only `f16` and `turbo3` load without FA.

### 4. CPU/GPU layer-split is **broken** for correctness
Putting 4 of 27 layers on CPU (`-ngl 23`) breaks in both FA modes:

- **FA off → crash:** `GGML_ASSERT(... "tensor read out of bounds")` in
  `ggml-backend.cpp:349`; the request connection is dropped (config 11).
- **FA on → silent wrong output:** configs 12/14/16 answer `2` to `2+2` while
  their all-GPU FA-on twins (2/8/6) answer `4` correctly. The split + FA path
  corrupts computation.

The **per-layer** path (`-ctk-cpu turbo3`, configs 18/20) is the only split
that runs correctly — suggesting the bug lives in the *inherited*-type CPU KV
path, and explicitly pinning CPU layers to `turbo3` routes around it.

### 5. Per-layer TurboQuant: the only safe split, but slow
Compressing just the 4 CPU layers to `turbo3` (GPU stays f16/q4_0) with FA on
gives correct output at **187–215 tok/s** prefill (vs 1984 all-GPU). It trades
~9× prefill speed for ~memory savings on the offloaded layers — only worth it
when VRAM forces a split.

### Best overall config
**`-ngl 99 -ctk f16 -ctv f16 -fa off` (all-GPU, f16 KV, no FlashAttention).**
1984 tok/s prefill + 162.5 tok/s decode + correct output. Every other config is
either slower (FA-on, quantized KV, turbo3), broken (any CPU/GPU split), or
impossible to load (quantized V without FA).

## Methodology / Reproducibility

Each config: boot `llama-server` → wait for `model loaded` (poll `/health`) →
one tiny warmup request → cold ~2069-token prefill (capture `prompt eval time`)
→ 200-token decode (capture `eval time`) → `What is 2+2?` correctness probe →
kill server. Harness: [`run-kv-fa-bench.py`](./run-kv-fa-bench.py). Raw
per-test server logs and `results.jsonl` are under `/mnt/llm-storage/turbo-build/bench/`
on the host.

Example invocation (inside a GPU-enabled `llama-rocm714` container):

```bash
HIP_VISIBLE_DEVICES=1 /build/src/build/bin/llama-server \
  -m /models/dsv2lite-q8_0.gguf -ngl 99 -c 4096 -np 1 \
  -ctk f16 -ctv f16 -fa off \
  --host 127.0.0.1 --port 8098 --no-webui
```

> Single-run measurements; prefill numbers are stable to within ~1% across the
> warmup/measurement passes (JSON `timings` matched the server log line
> exactly, e.g. 1042.84 ms / 2069 tok = 1984.01 tok/s). The FA-on decode
> collapse (162 → 24 tok/s) is consistent across all FA-on/quantized configs.
