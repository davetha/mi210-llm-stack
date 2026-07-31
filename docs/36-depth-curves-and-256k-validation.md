# Depth curves to 150k, and the 256k patch validated in production

**Date**: 2026-07-31 · **Hardware**: 2× MI210 (gfx90a) · **Tool**:
[`eugr/llama-benchy`](https://github.com/eugr/llama-benchy) 0.4.0,
`--latency-mode generation --concurrency 1`

Every other measurement in this repo reports throughput at one or two fixed
context lengths. This is the first sweep, and the first end-to-end confirmation
that `configs/extend_rocm_pa_256k_gfx9.py` works at production depth rather than
in a unit test.

**Both arms ran on both GPUs**, verified by per-card VRAM rather than assumed —
see "How the GPU count was established" below.

---

## The 256k patch is doing its job

```
ROCM_AITER_FA
npar_loops': 10
```

`npar_loops = ceil(ceil(160000/256)/64) = 10`, and **stock vLLM's reduction
switch only handles up to 8**. Cases 9–16 exist solely because of the patch, so
this is direct evidence the custom kernel was used — not an inference from
throughput.

The curve agrees. Decode goes **42.15 → 39.43 t/s** from 120k to 150k, straight
through the stock 131072 ceiling: a 6% decline, not the ~10× Triton-fallback
cliff `docs/28` documents for unpatched builds.

---

## vLLM — Qwen3-30B-A3B **W8A8**, TP=2, `--max-model-len 160000`

| depth | pp2048 t/s | tg128 t/s | TTFT (ms) |
|---:|---:|---:|---:|
| 4,096 | 10,803.53 | 57.80 | 508.6 |
| 8,132 | 9,323.73 | 56.57 | 940.5 |
| 16,000 | 8,397.73 | 55.50 | 1,828 |
| 30,000 | 7,016.41 | 51.48 | 3,847 |
| 60,000 | 5,123.08 | 48.25 | 10,164 |
| 90,000 | 4,039.83 | 44.88 | 19,013 |
| 120,000 | 3,297.37 | 42.15 | 30,914 |
| **150,000** | **2,806.91** | **39.43** | **45,218** |

## llama.cpp — Qwen3-30B-A3B **Q4_K_M**, `-ngl 999`, `-c 160000`

| depth | pp2048 t/s | tg128 t/s | TTFT (ms) |
|---:|---:|---:|---:|
| 8,132 | 3,717.13 | 102.71 | 2,348 |
| 16,000 | 3,467.32 | 94.97 | 4,376 |
| 30,000 | 2,890.28 | 86.75 | 9,252 |
| 60,000 | 2,056.75 | 71.59 | 25,231 |
| 90,000 | 1,593.03 | 61.18 | 48,219 |
| 120,000 | 1,315.32 | 52.86 | 77,554 |
| **150,000** | **1,106.81** | **46.38** | **114,743** |

### The engines split in opposite directions

**vLLM wins prefill and TTFT by a consistent ~2.5×.** At 150k that is 45 s to
first token against 115 s — the dominant term for any long-prompt workload.

**llama.cpp wins decode at every depth**, but the margin collapses: 1.8× at 8k,
**1.18× at 150k**. llama.cpp degrades faster (102.7 → 46.4, a factor of 2.2)
than vLLM (56.6 → 39.4, a factor of 1.4), so the curves converge and would
plausibly cross beyond 150k.

> **Caveat on reading this as an engine comparison.** vLLM ran W8A8 and llama.cpp
> ran Q4_K_M — different quantizations, so part of the decode gap is format
> rather than engine. The *shapes* of the curves are the more reliable signal
> than the absolute difference.

### Decode degrades sub-linearly; prefill does not

From `llama-bench` on a single card, same model, shallow end:

| depth | pp512 t/s | tg128 t/s |
|---:|---:|---:|
| 0 | 2,438.50 | 116.16 |
| 4,096 | 2,062.84 | 108.56 |
| 8,192 | 1,782.49 | 103.15 |
| 16,384 | 1,389.77 | 94.53 |
| 32,768 | 964.21 | 81.76 |

**8× more context costs only 30% of decode throughput.** That matters for
`docs/25` item 1c, whose resized bound assumes KV traffic scales linearly with
context. Out to at least 32k it does not, so the KV term in that calculation is
an over-estimate at depth and the ~3.1× residual is, if anything, understated
there.

TTFT tells the opposite story: 60k → 150k is 2.5× the depth but **4.5× the
TTFT**, which is attention going quadratic exactly where prefill throughput
falls off.

---

## `llama-bench` has a multi-GPU depth bug; `llama-server` does not

Round 28 ran `llama-bench -d 0,4096,8192,16384,32768` on two cards. The `d=0`
rows completed, then it died:

```
[mmhub0] no-retry page fault (src_id:0 ring:144 vmid:3 pasid:35032)
  Faulty UTCL2 client ID: SDMA1 (0x101)
  MAPPING_ERROR:     0x1      <- the page is NOT mapped
  PERMISSION_FAULTS: 0x2
  RW:                0x0      <- a read
```

Round 29 isolated it by changing one variable at a time:

| arm | GPUs | depth | result |
|---|---:|---:|---|
| A | 1 | 4,096 | **survived** |
| B | 1 | full sweep | **survived** |
| C | **2** | 4,096 | **FAULTED** (exit 139) |

Single card is clean at every depth; two cards fault at the *smallest* non-zero
depth. But `llama-server` ran the full sweep to **150,000 on the same two
cards** without faulting.

Same model, same image, same GPUs, overlapping depths — so this is **not**
llama.cpp's multi-GPU KV handling. It is specific to `llama-bench`'s internal
depth-priming path; `llama-server` populates KV through ordinary request
processing and is unaffected. Reproduction:

```bash
llama-bench -m Qwen3-30B-A3B-Q4_K_M.gguf -d 4096 -ngl 999   # 2 GPUs -> fault
HIP_VISIBLE_DEVICES=0 llama-bench ... -d 4096 -ngl 999      # 1 GPU  -> fine
```

**Not the same fault as `docs/29`'s XNACK failure**, despite hitting the same
userspace assertion:

| | XNACK (round 15) | `llama-bench` (round 28) |
|---|---|---|
| hub | `gfxhub0` | **`mmhub0`** |
| client | `TCP` (vector L1, compute) | **`SDMA1`** (DMA engine) |
| `MAPPING_ERROR` | `0x0` — page present | **`0x1` — page absent** |
| `RW` | `0x1` write | **`0x0` read** |
| retry fault first | yes | **no** |
| aftermath | wedged SVM workers, load 70 | died cleanly |

The address, `0x738bfcc00000`, falls in the gap between the two VRAM blocks in
llama-bench's own dump — consistent with an address valid on the *other* device
rather than an out-of-bounds read, which is why the mapping is simply absent.

---

## How the GPU count was established

Worth recording because an earlier version of this writeup asserted it from the
launch flags. `llama-server` logs no device enumeration at all — no
`ggml_cuda_init`, no `MI210`, no `ROCm1` — so the configuration alone did not
prove the model was split. And it would have fit on one card: 17.28 GiB of
weights plus ~15.4 GB of KV at 160k is ~33 GB against a 64 GB card.

Measured instead, per card, before and during:

| | GPU[0] | GPU[1] |
|---|---:|---:|
| idle | 11 MB | 11 MB |
| serving @ 160k ctx | **20.1 GB** | **21.1 GB** |

Near-even split across both. vLLM was never in doubt — `world_size=2`,
14.66 GiB weights and 41.49 GiB KV *per rank*.
