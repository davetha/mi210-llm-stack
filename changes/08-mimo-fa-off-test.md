# Change 08 — FlashAttention audit on production MiMo 230B: FA-on confirmed, FA-off not testable

An audit of the live production **MiMo 230B** (Xiaomi MiMo-V2.5-abliterated, Q4_K,
21-shard, ~174 GB) instance was triggered by the
[KV×FA benchmark matrix](../benchmarks/comprehensive-kv-fa-matrix.md), which
showed FlashAttention is a **2.6× prefill / 6.7× decode regression** on gfx90a for
DeepSeek-V2-Lite. Goal: check production's `-fa` flag, test `-fa off` on MiMo, and
benchmark before/after.

**Outcome:** Production runs `-fa on`. A live FA-off test was **not possible**
this session — VRAM and the production-no-restart constraint block it, and the
production KV types cannot even load under FA-off on this build. The FA-on
baseline for MiMo was measured and a maintenance-window test plan + script are
provided. **No production change was made.**

## 1. What production runs today

From `/mnt/llm-storage/launch-mimo.sh` (identical to
[`configs/launch-mimo.sh`](../configs/launch-mimo.sh)):

```text
-m  /models/mimo-v25/Q4_K/Huihui-MiMo-V2.5-abliterated-Q4_K-00001-of-00021.gguf
-ngl 999
-ot blk.([0-9]|1[0-9]|2[0-4]).ffn.*exps=CPU,          # blks  0-24  -> CPU
    blk.(2[5-9]|3[0-6]).ffn.*exps=ROCm0,              # blks 25-36  -> GPU0
    blk.(3[7-9]|4[0-8]).ffn.*exps=ROCm1               # blks 37-48  -> GPU1
-c 65536 -b 2048 -ub 2048 -np 1
-fa on                                                # <<< FlashAttention ENABLED
-ctk q8_0 -ctv q4_1                                   # <<< quantized V cache
--jinja -a mimo --no-warmup
```

This is a **3-way CPU/GPU split** (FFN experts sharded across CPU + 2× MI210,
attention/embeddings on GPU) with **quantized KV cache** (`q8_0` K, `q4_1` V).

Confirmed live: container `llama-main`, llama-swap model `mimo` = `loaded`,
backend on dynamic port `5803`, `/props` reports `model_ftype: Q4_K - Medium`.

## 2. Architecture: MiMo ≠ DeepSeek-V2-Lite (MLA)

The benchmark's FA-off crash/wrong-output results were specific to
**Multi-head Latent Attention (MLA)**. MiMo uses a **different** attention scheme.
From the GGUF tensor names in the load log:

```text
blk.N.attn_sinks.weight   # sliding-window attention with attention sinks
blk.N.nextn.eh_proj.weight
blk.N.nextn.enorm.weight
blk.N.nextn.hnorm.weight
blk.N.layer_output_norm.weight
```

`attn_sinks` + `nextn` blocks = a **sliding-window hybrid** (Gemma2/Mistral-style
local + global layers with attention sinks), **not MLA**. The chat template also
self-identifies: *"You are MiMo, a helpful AI assistant engineered by Xiaomi."*

**Implication:** the MLA-specific `GGML_ASSERT(... "tensor read out of bounds")`
FA-off crash is not expected to reproduce on MiMo's attention path. But it is not
guaranteed — the split + FA interaction is the unresolved variable (see §5).

## 3. FA-on baseline measured on production (non-disruptive)

Measured live against the production backend (port `5803`, slot idle) via the raw
`/completion` endpoint so timings are clean (no chat-template / `<think>` tokens).
Correctness probe: `"Question: What is 2+2?\nAnswer:"` → **`4`** ✅.

| Test | Prompt tok | Prefill (tok/s) | Decode (tok/s) | Correct? |
|---|---:|---:|---:|---|
| Correctness (n_predict=6) | 11 | — (tiny) | — | ✅ `4` |
| Short prefill (n_predict=8) | 757 | **177.8** | 21.3 | ✅ |
| **~2K prefill (n_predict=16)** | **2737** | **146.0** | 16.4¹ | ✅ |
| Decode (n_predict=128) | 9 | 33.8 | **21.9** | ✅ |

¹ Decode over only 16 tokens is noisy; the 128-token run (21.9 tok/s) is the
reliable decode number.

**FA-on baseline for production MiMo:** **prefill ~146–178 tok/s, decode ~22 tok/s.**

### Context vs. the DSV2-Lite numbers

These are **not** comparable 1:1 to the DSV2-Lite all-GPU figures (770/24 FA-on):
- MiMo is **~15× larger** (230B vs 16B) and runs on a **3-way CPU/GPU split**
  (CPU executes FFN experts for blks 0-24), so prefill is CPU-PCIe-bound, not
  GPU-attention-bound.
- MiMo decode (~22 tok/s) is in fact **comparable** to DSV2-Lite FA-on decode
  (24 tok/s), confirming FA-on decode is attention-kernel-bound at ~22–24 tok/s
  across architectures on this gfx90a build.
- So FA-on's decode penalty (~6.7× vs FA-off on all-GPU DSV2-Lite) is the cost
  MiMo is paying too — *if* FA-off were even loadable here (it isn't, see §4).

## 4. Why FA-off cannot be tested on the current config

Two hard blockers, both independent of the VRAM/restart constraints:

### Blocker A — quantized V cache requires `-fa on` (engine constraint)

Production uses `-ctv q4_1`. The KV×FA matrix proved a **universal llama.cpp
constraint** on this build:

> `E llama_init_from_model: V cache quantization requires flash_attn`

Any quantized **V** cache (`q8_0`, `q4_0`, `q4_1`) is rejected at context creation
unless `-fa on`. This is model-independent — it applies to MiMo too. So a naive
`-fa off` flip on the production launch script **would fail to load**.

The only KV types that load without FA are **`f16`** and `turbo3`. To run FA-off
you must switch to `-ctk f16 -ctv f16` (or `turbo3`). At `n_ctx 65536` over ~49
layers, f16 KV is substantially larger than the current `q8_0/q4_1` and may not
fit alongside the model on these GPUs.

### Blocker B — no spare VRAM for a parallel test instance

| GPU | Total | Used | Free |
|---|---:|---:|---:|
| GPU0 (ROCm0) | 64.0 GB | 43.7 GB | **20.3 GB** |
| GPU1 (ROCm1) | 64.0 GB | 47.5 GB | **16.5 GB** |

Production MiMo already consumes ~91 GB of the combined 128 GB VRAM. The model is
~174 GB; a second instance needs the same ~91 GB VRAM. **No parallel FA-off test
instance can fit.** (System RAM is ample — 481 GB available — but irrelevant; the
GPU-side weights + f16 KV can't spill there on CDNA2, see
[change 07](./07-vllm-cpu-offload-analysis.md).)

### Blocker C — production must not be restarted

The task forbids disrupting the live `mimo` on port 8090 (`ttl: 86400`, the only
loaded model). FA-off requires a full model reload → cannot be done live.

## 5. The open question that only a live test can answer

If Blockers A+B+C were cleared (maintenance window + f16 KV that fits), the one
thing nobody can predict without running it is:

> **Does MiMo's sliding-window split + FA-off stay correct, or does it hit a
> split/FA bug like DSV2-Lite did?**

The MLA crash was attention-specific. MiMo's `attn_sinks` sliding-window path is a
different code path, so it *probably* won't crash — but "probably" is not "verified."
The only certainty from the DSV2-Lite work is that the **plain CPU/GPU split was
broken in *both* FA modes for MLA** (crash off, silent-wrong on). For MiMo the
FA-on split is empirically **correct** (returns `4`), so the FA-on split path is
fine for this architecture. Whether FA-off is also fine is untested.

## 6. Recommendation

1. **Do not change production now.** FA-off is not a safe flag flip on this config
   (quantized-V load failure), and there is no verified upside for MiMo's
   CPU-bound split: the 2.6× prefill win measured on DSV2-Lite was on an
   **all-GPU** topology where attention dominated. MiMo prefill is CPU/PCIe-bound,
   so removing the FA attention penalty may yield far less than 2.6× — possibly
   little — because the bottleneck is elsewhere.
2. **If a prefill win is still wanted**, run the maintenance-window test in §7
   during a scheduled outage. Verify (a) it loads, (b) `2+2 = 4` correctness, (c)
   real prefill/decode numbers. Roll back on any wrong output or crash.
3. **The decode case is the stronger motivation** if FA-off works: DSV2-Lite FA-on
   decode was pinned at ~24 tok/s by the FA kernel; MiMo FA-on decode is the same
   ~22 tok/s. If MiMo FA-off decode behaves like DSV2-Lite FA-off (162 tok/s),
   that is a real win for interactive latency — but only a live test confirms it,
   and only after switching KV to f16/turbo3.

## 7. Maintenance-window test plan (no production restart)

Script: [`configs/launch-mimo-fa-off-test.sh`](../configs/launch-mimo-fa-off-test.sh)
— a standalone launcher for a **separate** container (`llama-mimo-faoff`) on port
**`8099`**, run only after the production `mimo` slot is drained/stopped by an
operator during a maintenance window.

Key differences from production:
- `-fa off` (the variable under test)
- `-ctk f16 -ctv f16` (only types that load without FA)
- `-c 16384` (reduced from 65536 so f16 KV is more likely to fit)
- port `8099`, container `llama-mimo-faoff`, `-a mimo-faoff`

```bash
# 1. Operator drains + stops production mimo first (maintenance window):
#      docker stop llama-main
# 2. Run the FA-off test instance on 8099:
sh /mnt/llm-storage/launch-mimo-fa-off-test.sh
# 3. Correctness + benchmark against http://127.0.0.1:8099
# 4. Stop test, restart production: sh /mnt/llm-storage/launch-mimo.sh <port>
```

Decision rule for the operator:
- Loads + `2+2=4` + prefill > ~200 tok/s **or** decode > ~40 tok/s → FA-off is a
  real win; update `launch-mimo.sh` to f16-KV + `-fa off` (revisit `-c` for VRAM).
- Wrong output or crash → keep `-fa on`; document the architecture-specific bug.
- Loads but no meaningful speedup (CPU/PCIe-bound) → keep current config; the FA
  penalty isn't the bottleneck on the split topology.

## Methodology / Reproducibility

- FA-on baseline: live probes to production backend `127.0.0.1:5803` (llama-swap
  assigned port at session time) via `/completion` (raw, `stream:false`,
  `samplers:["temperature"]`, `temperature:0`). Timings read from the JSON
  `timings` object (`prompt_n/prompt_ms`, `predicted_n/predicted_ms`).
- VRAM: `rocm-smi --showmeminfo vram` inside `llama-main`.
- Architecture: tensor-name inspection from `docker logs llama-main` load log +
  `/props` (`model_ftype`, chat template Xiaomi self-id).
- No production container was restarted, stopped, or reconfigured. Probes used the
  idle slot; the warm-session restore path was left untouched.
