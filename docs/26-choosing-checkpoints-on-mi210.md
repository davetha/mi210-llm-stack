# Choosing a checkpoint on an MI210

**Date**: 2026-07-28 · **Hardware**: 2× MI210 (gfx90a / CDNA2), 64 GB HBM2e each, 512 GB system RAM
**Measurements**: `benchmarks/matrix/results/*.json` · **Mechanism**: `docs/24`

`docs/24` answers "which quantization is fastest" (INT8 W8A8). This document
answers the question that actually comes up when browsing Hugging Face: **given a
model I want to run, which of its published checkpoints should I download, and
how do I start it?**

The short version: **two fields in `config.json` predict almost everything, and
neither of them is the repo name.**

---

## The two fields

```jsonc
"quantization_config": {
  "quant_method": "...",          // <- decides which LOADER runs   (load time)
  "config_groups": {
    "group_0": {
      "weights":            { "num_bits": 8, ... },
      "input_activations":  null   // <- decides which GEMM runs    (throughput)
    }
  }
}
```

They are independent. A checkpoint can be fast to load and slow to run, or the
reverse.

### Why you cannot trust the repo name

Both of these are in this benchmark matrix. Neither name describes the file:

| Repo | Name says | `quant_method` | `input_activations` | Actually is |
|---|---|---|---|---|
| `cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-8bit` | AWQ, 8-bit | `compressed-tensors` | **`null`** | compressed-tensors **W8A16** |
| `RedHatAI/Qwen3-30B-A3B-Thinking-2507-quantized.w8a8` | w8a8 | `compressed-tensors` | 8-bit, `dynamic`, per-token | genuine **W8A8** |

The first one **is not AWQ at all** — it is `format: "pack-quantized"`
compressed-tensors with unquantized activations. It happens to be the fastest
single arm in the whole matrix, and it got there by loading fast and running on
two cards, not by being 8-bit in the way that matters.

Opening `config.json` before downloading costs ten seconds and is the single
highest-value habit in this document.

---

## Axis 1 — `quant_method` decides the loader

vLLM picks a weight loader from the packaging. On a **MoE**, that choice is worth
minutes versus hours, because the slow loader runs per expert per weight
(128 experts × 48 layers, single-threaded Python).

| `quant_method` | Loader | Measured on a MoE |
|---|---|---|
| `compressed-tensors` (`int-quantized`) | fast path | **122.8 s** — 30B W8A8, TP=1 |
| `compressed-tensors` (`pack-quantized`) | fast path | **183.7 s** — 80B W8A16, TP=2 |
| `awq` | AWQ loader | served fine — 30B AWQ-Int4 |
| `gptq` / `gptqmodel` | `moe_wna16_weight_loader` | **>46 min, never reached serving** |
| _(none — bf16)_ | `_load_w13` | **697 s per shard** at TP=2 |

The GPTQ figure is not a timeout artifact. `py-spy` puts every sample in

```
moe_wna16_weight_loader (quantization/moe_wna16.py:445)
```

with the weights already resident in VRAM (32.15 GB). The remaining time is pure
CPU: a `loaded_weight.to(device)` plus a format conversion per expert per weight.
Files that read at 3.0 GB/s under `dd` take 810 s per shard through this path.

**Rule: on a MoE, avoid `gptq` packaging.** It is fine on a dense model, where
there are no per-expert tensors to iterate. This is a *loader* property, not a
statement about GPTQ's accuracy.

**Rule: bf16 is impractical for MoE on this box** for the same reason —
`_load_w13` at 697 s/shard makes a TP=2 load a multi-hour affair.

### The size that follows from this

A 235B at TP=2 through the GPTQ loader costs **~7 hours to load**. That is not a
benchmark artifact, it is the deployment reality, and it is why tier 3 has no
throughput row.

---

## Axis 2 — `input_activations` decides the GEMM

gfx90a's INT8 peak is **181 TOPS, exactly equal to its 181 TFLOP/s bf16 peak**.
So INT8 halves weight-memory traffic at *no arithmetic cost* — but only if the
GEMM is actually int8 × int8. That requires quantized activations.

| `input_activations` | Scheme | GEMM | Reaches `v_mfma_i32_16x16x16i8` |
|---|---|---|---|
| populated (`num_bits: 8, dynamic: true`) | W8A8 | int8 × int8 | **yes** |
| `null` | W8A16 / weight-only | dequant → bf16 | no |

Weight-only 8-bit buys bandwidth and nothing else. That is still worth something
— it is not a *bad* choice — but it does not reach the hardware that makes INT8
the right answer on CDNA2.

Measured at tier 1, with AITER held constant on both sides:

| Checkpoint | TTFT @15k | Prefill |
|---|---:|---:|
| **W8A8** (`input_activations` populated) | **3.20 s** | **4,739 t/s** |
| AWQ-Int4 (weight-only) | 5.07 s | 3,002 t/s |
| FP8 | 14.51 s | 1,047 t/s |

**FP8 is never the answer on this card.** CDNA2 has no FP8 ALU; every weight
upcasts to bf16 before the GEMM. The output is correct, it is just 2.9× slower
than Int4.

---

## W8A8 is far more available than it looks

The reasonable objection to `docs/24` is "W8A8 isn't a popular format, so this
recommendation is useless." That is true of the *GGUF-first hobbyist quantizer*
ecosystem and false of the vLLM serving ecosystem, where W8A8 is close to the
default.

`RedHatAI` alone publishes **~90** `quantized.w8a8` checkpoints, all
`compressed-tensors` `int-quantized` with dynamic per-token activations — i.e.
both axes correct. The ones relevant to this box:

| Repo | Size | Fits 2× 64 GB? |
|---|---:|---|
| `RedHatAI/Qwen3-30B-A3B-Thinking-2507-quantized.w8a8` | 35.2 GB | **yes**, TP=1 with 27 GB KV |
| `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8` | ~35 GB | yes, TP=1 |
| `RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8` | **85.7 GB** | **yes, TP=2** — see below |
| `RedHatAI/GLM-4.6-quantized.w8a8` | ~357 GB | no — CPU offload |
| `RedHatAI/Qwen3-235B-A22B-Instruct-2507-quantized.w8a8` | 239.5 GB | no — CPU offload |
| `RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8` | ~70 GB | yes, TP=2 (dense) |
| `RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8` | ~24 GB | yes, TP=1 (dense) |
| `RedHatAI/QwQ-32B-quantized.w8a8` | ~32 GB | yes, TP=1 (dense) |
| `RedHatAI/gemma-3-27b-it-quantized.w8a8` | ~27 GB | yes, TP=1 (dense) |
| `RedHatAI/MiniMax-M2.5-quantized.w8a8` | large | CPU offload |

### Verified non-Qwen picks

The matrix benchmarked Qwen almost exclusively, which is a sampling artifact of
how the tiers were chosen, not a recommendation. Every row below was checked by
fetching `config.json` — `quant_method`, `input_activations` and `head_dim` are
what the file says, not what the name says:

| Repo | Family | Shape | `head_dim` | W8A8? | Size | Placement |
|---|---|---|---:|---|---:|---|
| `RedHatAI/gemma-3-27b-it-quantized.w8a8` | Gemma 3 | dense, 62L, SWA 1024/6 | **128** | **yes** | ~27 GB | TP=1 |
| `RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w8a8` | Mistral | dense, 40L | **128** | **yes** | ~24 GB | TP=1 |
| `RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8` | Llama 3.3 | dense, 80L | **128** | **yes** | ~70 GB | TP=2 |
| `RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8` | Qwen3-Next | MoE, GDN 3:1 hybrid | 256 | **yes** | 85.7 GB | TP=2 |
| `RedHatAI/MiniMax-M2.5-quantized.w8a8` | MiniMax | MoE, 256e/8, **full-attn** | **128** | **yes** | large | offload |
| `RedHatAI/GLM-4.6-quantized.w8a8` | GLM | MoE, 160e/8, GQA, 92L | **128** | **yes** | ~357 GB | offload |

All six are `compressed-tensors` / `int-quantized` with `input_activations` at
8-bit dynamic per-token. All are ≥131k native context, so none needs the
`--max-model-len` gate raised.

**The dense `head_dim = 128` rows are the sweet spot for this box**, and it is
worth saying plainly because the benchmarked models are not: Gemma-3-27B,
Mistral-Small-24B and Llama-3.3-70B hit **both** fast paths at once — the INT8
MFMA GEMM *and* AITER's `hd128` ASM attention. Qwen3-Next, the model that
produced the headline numbers, gets only the first, because `head_dim = 256`
has no ASM kernel anywhere in AITER. Nothing in the matrix measures a
dense hd128 W8A8 model; on mechanism it should be the strongest configuration
available here, and that is a prediction, not a result.

Community publishers also produce W8A8 — `Avesed/Qwen3.6-35B-A3B-INT8-W8A8`,
`nameistoken/Qwen3.6-35B-A3B-Quark-W8A8-INT8`,
`ramblingpolymath/Qwen3-Coder-30B-A3B-Instruct-W8A8`,
`ArliAI/Mistral-Medium-3.5-128B-INT8-W8A8-Dynamic`. Check both fields on these;
the naming discipline is looser and `Quark`-produced files use a different
packaging.

### Correction to `docs/24`

`docs/24` states that tier 2 used a different format set "because no INT8 W8A8
checkpoint exists for Qwen3-Next-80B." **That is wrong.**
`RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8` exists — 85.73 GB,
`compressed-tensors` / `int-quantized`, `input_activations` 8-bit dynamic
per-token. It is the *Instruct* variant; the *Thinking* variant that the matrix
benchmarked has no W8A8, which is where the claim came from and where it should
have stopped.

**This is the single most promising untested configuration for this box**, and it
is untested only because the GPU host was not reachable when this was written:

- Both axes correct → the fast loader *and* the INT8 MFMA path.
- 85.7 GB at TP=2 ≈ 43 GB/card, leaving ~21 GB/card for KV.
- Qwen3-Next is a 3:1 Gated DeltaNet hybrid, so only 12 of 48 layers hold KV —
  the measured W8A16 build got **1.28 M KV tokens in 14.9 GiB**.
- The measured W8A16 build of the same architecture already produced the best
  numbers in the matrix (2.27 s TTFT, 51.3 t/s decode at 101k) *without* the
  INT8 GEMM path.

Expected but **not measured**: W8A8 should improve prefill over W8A16 on the same
model, by the same mechanism that gave tier 1 its +58%. Do not quote a number for
this until it is run — `benchmarks/matrix/run_arm.sh` with the W8A8 repo is the
whole experiment.

---

## Recommended checkpoint by model class

| Model class | First choice | Fallback | Avoid |
|---|---|---|---|
| **MoE, fits in VRAM** | `compressed-tensors` W8A8 | `compressed-tensors` W8A16, then AWQ-Int4 | **`gptq` packaging**, bf16, FP8 |
| **Dense, fits in VRAM** | `compressed-tensors` W8A8 | AWQ-Int4 or GPTQ-Int4 (both fine — no expert loop) | FP8 |
| **Anything needing >128k context** | GGUF Q8_0 / Q4_K_M on llama.cpp | — | vLLM (see `docs/23`) |
| **Too big for 128 GB VRAM** | **smallest** GGUF you can tolerate (Q4_K_M / IQ3_XS) + llama.cpp `--n-cpu-moe` | — | W8A8 (yes, really — see the inversion below), vLLM CPU offload |

Two notes on the fallbacks:

- **AWQ-Int4 is the reliable universal answer.** It is published for nearly
  everything, it loads at a sane speed, and at 5.07 s TTFT it is 58% behind W8A8
  but 2.9× ahead of FP8. If you are downloading one file and want it to just
  work, this is it.
- **Q8_0 costs almost nothing over Q4_K_M** on llama.cpp — 4.57 s vs 4.43 s,
  despite 31 GB against 18 GB. On this card lighter quantization buys bandwidth,
  not arithmetic, so take the accuracy.

---

## How to load them

### vLLM — MoE, W8A8 or W8A16, ≤128k context

Requires the patches in `configs/` (see `benchmarks/matrix/REPRODUCE.md`);
`enable_int8_moe_rocm.py` is mandatory for W8A8 or vLLM refuses to start with
`NotImplementedError: No Int8 MoE backend supports the deployment configuration`.

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 991 \
  --ipc host --shm-size 32g \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e GPU_MAX_HW_QUEUES=4 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_MHA=1 \
  -e VLLM_PREFER_AITER_FA=1 \
  -e VLLM_TUNED_CONFIG_FOLDER=/tuned \
  vllm-aiter-gfx90a:latest \
  vllm serve <model> \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.92
```

Every one of those knobs earns its place:

| Knob | Why |
|---|---|
| `HSA_NO_SCRATCH_RECLAIM=1` | required on MI200; without it scratch reclaim stalls |
| `NCCL_P2P_DISABLE=1` | these cards are PCIe-only, no XGMI |
| `GPU_MAX_HW_QUEUES=4` | avoids queue oversubscription at TP=2 |
| `VLLM_ROCM_USE_AITER=1` + `_MHA=1` | admits AITER at all |
| `VLLM_PREFER_AITER_FA=1` | **actually selects it** — without this `ROCM_ATTN` is appended first and wins unconditionally |
| `--max-model-len 131072` | **never higher.** Above 128k the graph capture bakes the Triton fallback into *every* request — 10× decode cost. `docs/23` |
| `--group-add 44 --group-add 991` | numeric; `render`/`video` names do not resolve inside the container |

Drop `--tensor-parallel-size 2` for anything under ~45 GB. TP=1 is faster per
card when the model fits; TP=2 is for capacity, and for KV headroom at long
context.

### Verify you got the fast paths — do not assume

Selection is not proof of execution. Grep the server log:

```
Using TRITON Int8 MoE backend            <- W8A8 GEMM path reached
Overriding with ROCM_AITER_FA            <- AITER selected
fwd_hd128_bf16_causal_rtna_group.co      <- ASM kernels actually LOADED
```

The third line is the one that matters and the one that is usually missing. The
80B arms show `Overriding with ROCM_AITER_FA` and then load **only `torch.co`** —
no ASM at all, because Qwen3-Next has `head_dim = 256` and AITER ships fmha ASM
only for `hd128` and `hd192`, upstream, for every architecture. There is nothing
to translate. Backend selected, fast path not taken, +0%.

So: **AITER ASM is worth +12.8% on `head_dim = 128` models and exactly nothing on
`head_dim = 256` models.** Check `head_dim` in `config.json` too while you have
it open.

### llama.cpp — long context, or too big for VRAM

```bash
llama-server -m <model>.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 262144 -ub 2048 \
  --flash-attn on
```

- `-ub 2048` is the measured optimum. 4096 → 4.84 s, 8192 → 6.03 s against
  2048's 4.43 s. Bigger is worse.
- **Do not set `-ngl`, `--n-cpu-moe`, or any tensor override unless you are
  overriding placement entirely.** `common_fit_params()` is all-or-nothing: it
  logs `n_gpu_layers already set by user to 999, abort` and disables auto-fit for
  the whole model, at which point you own every placement decision manually.

llama.cpp is not marginally better at long context, it is the only option:
**230k tokens at 30.4 t/s decode**, against vLLM's 0.7 t/s at 241k. 43×.

---

## MoE that does not fit: experts in system RAM

The box has 128 GB VRAM and **499 GB DDR4**, so the interesting question for
anything above ~120 GB is not "does it fit" but "which parts go where, and what
does that cost." This is well-trodden ground here — `mimo` (230B MoE) has been
running in production with 25 of its 48 expert layers pinned to CPU.

### Use llama.cpp. The alternatives are closed or unusable.

| Engine | Selective expert offload | Status on this box |
|---|---|---|
| **llama.cpp** | `-ot` regex / `--n-cpu-moe` | **production** |
| vLLM `--cpu-offload-gb` | UVA paging, not selective | **measured unusable** — see below |
| vLLM `--moe-expert-cache-size` | LRU expert cache | needs **vLLM 0.25+**; this stack is 0.23.1 |
| KTransformers | yes, its whole purpose | **hard-blocked** — needs AVX-512/AMX, EPYC 74F3 is Zen3/AVX2 |
| SGLang | none | OOMs; no GGUF |
| DeepSpeed-MII | expert-parallel only | no CPU offload at all |

**vLLM's CPU offload is the wrong mechanism, not merely slower.** GLM-4.6
AWQ-Int4 at TP=2 with `--cpu-offload-gb 70` *loads and answers correctly* —
70.51 GB offloaded, 419 s load, 232,368 KV tokens, correctness probe PASS — and
then a single 28k-token request ran past **35 minutes at a steady 505% container
CPU**, computing the whole time, projecting to ~2 hours. It is UVA-based:
parameters live in host memory and are faulted across PCIe on demand, page by
page, with no awareness of which experts a token actually routes to.
llama.cpp instead keeps attention resident and **computes** the CPU-resident
expert layers on the CPU, which is the right structure for a sparse MoE.

**KTransformers is permanently out**, and it is worth being explicit because it
is the obvious thing to reach for. Three independent blockers (`docs/07`), any
one of which is fatal: `sgl-kernel` is CUDA-only; the CPU expert kernels require
AVX-512 or AMX and the EPYC 74F3 is Zen3 with AVX2 only; and the GPU dispatch
kernel requires hidden-dim % 256 == 0. Upgrading the CPU is the only route, and
it is not a software fix.

### Do not split experts across the two cards

Expert parallelism sounds natural for MoE and is wrong here: the MI210s are
**PCIe-linked with no XGMI**, so every routed expert activation crosses the bus.
`NCCL_P2P_DISABLE=1` is in the recommended flags for the same reason. Use TP for
capacity, keep experts CPU-side when they do not fit.

### How much can go to RAM — the planning model

Decode with CPU-resident experts is bounded by **DDR4 bandwidth**, not PCIe and
not the GPU. Per generated token the CPU must read every *active* expert weight
in the layers you pinned:

```
bytes/token  ≈  active_params × bytes_per_param × (CPU layers / total layers)
decode ceiling ≈ effective_DDR4_bandwidth / bytes_per_token
```

Peak DDR4 on this host is **~204 GB/s** (8-channel DDR4-3200, NPS1, so no
cross-NUMA penalty and no NUMA-pinning win either). Peak is not what you get.
Calibrating against the one configuration actually measured here:

| | `mimo` 230B, measured |
|---|---|
| active params | 10B |
| quant | Q4 → 0.5 B/param |
| CPU layers | 25 of 48 (52%) |
| ⇒ bytes/token from DDR4 | ~2.6 GB |
| **measured decode** | **~22 t/s** |
| ⇒ **effective bandwidth** | **~57 GB/s (≈28% of peak)** |

Use **~57 GB/s**, not 204, for planning. Expert reads are scattered and
routing-dependent; they do not stream.

Worked example — GLM-4.6 (357B total, ~32B active) at Q4, all experts CPU-side:
32 × 0.5 = 16 GB/token ⇒ **~3.6 t/s**. Pin only a third of the layers and it is
~11 t/s. That is the shape of the tradeoff: **`--n-cpu-moe` is close to linear in
how many layers you pin**, so pin the fewest that make it fit.

**Prefill is the real problem, not decode.** On a full-attention MoE, prefill
pushes every prompt token through all layers including the CPU-resident ones, and
attention is O(n²) — `mimo` costs **~7.4 s per chunk** re-reading the growing KV
cache from DDR4, while decode stays comfortable at 22 t/s. So:

- **Prefer a hybrid/linear-attention MoE** for CPU offload. Qwen3-Next's 3:1
  Gated DeltaNet means only 12 of 48 layers hold KV at all — the measured 80B
  build fit **1.28 M KV tokens in 14.9 GiB**. A full-attention model of the same
  size is far worse under offload.
- **Compress KV on the CPU layers only.** This repo's `-ctk-cpu` / `-ctv-cpu`
  patch exists precisely for this: 5× less DDR4 traffic on the bandwidth-bound
  layers while GPU layers stay at full precision (`changes/01`).

### The quant calculus inverts once you cross into RAM

This is the part that catches people, and it is the direct consequence of the
bandwidth model above.

**In VRAM you are arithmetic-bound**, and CDNA2's INT8 peak equals its bf16 peak,
so 8-bit weights cost nothing to compute and W8A8 wins. **In system RAM you are
DDR4-bandwidth-bound**, and every extra bit is bytes you must re-read per token.
The same 8 bits that were free on the GPU are now the whole cost.

| GLM-4.6 (357B total, ~32B active), all experts CPU-side | bytes/token | ceiling @ 57 GB/s |
|---|---:|---:|
| W8A8 (1.0 B/param) | 32 GB | **~1.8 t/s** |
| Q4_K_M (~0.5 B/param) | 16 GB | **~3.6 t/s** |
| IQ3_XS (~0.4 B/param) | 13 GB | **~4.5 t/s** |

So `RedHatAI/GLM-4.6-quantized.w8a8` is a **correct** checkpoint by both axes and
still the **wrong** choice at this size — it would be roughly half the speed of
the Q4 GGUF. The crossover is exactly the VRAM boundary:

> **Fits in 128 GB → take the largest quant that fits, W8A8 first.
> Does not fit → take the smallest quant you can tolerate, GGUF Q4/IQ3.**

Two corollaries:

- **Full-attention MoE is a poor offload target regardless of quant.**
  `RedHatAI/MiniMax-M2.5-quantized.w8a8` has both axes right and 256 experts, but
  `attn_type_list` is all-ones — full attention on all 62 layers. That is the
  `mimo` shape, where prefill costs ~7.4 s per chunk re-reading KV from DDR4.
  Prefer a hybrid (GDN, Mamba) or heavily-GQA model when experts go to RAM.
- **Prefer a model that fits over a bigger one that does not.** At tier 2 the
  80B at TP=2 decoded **51.3 t/s** at 101k and prefilled at **4,668 t/s**. The
  357B at IQ3_XS — which very nearly *does* fit — manages 8.5 t/s and **181
  t/s prefill**. Six times the decode gap and **25× on prefill**. Not close.

### Invocation

```bash
llama-server -m <model>.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 -ub 2048 --flash-attn on \
  -ngl 999 --n-cpu-moe 20
```

**`-ngl` and `--n-cpu-moe` are a package deal.** llama.cpp's `common_fit_params()`
auto-fit is all-or-nothing: setting *either* one disables it for the whole model
(`n_gpu_layers already set by user to 999, abort`, then
`tensor_buft_overrides already set by user, abort`), and from that point you own
every placement decision. So either

- **omit both** and let auto-fit place everything — simple, but it will not know
  to keep attention on GPU and experts on CPU; or
- **set both**, as above, and tune `--n-cpu-moe` down until it stops fitting.

There is no middle setting. Attempting one produced two failed GLM-4.6 runs in
this matrix before the mechanism was understood.

For finer control, `-ot` takes a regex over tensor names and overrides placement
per tensor — `-ot "blk\.(2[0-9]|[3-9][0-9])\.ffn.*exps=CPU"` pins experts from
layer 20 up, leaving everything below and all attention on GPU.

### The 357B tier — measured, and it is not what the model predicted

The GGUF arm completed. **GLM-4.6 IQ3_XS on llama.cpp, correctness PASS:**

| | |
|---|---:|
| decode @ short context | **12.83 t/s** |
| decode @ 25,792 tokens | **8.51 t/s** |
| prefill @ 15.2k | 196 t/s (77.5 s TTFT) |
| prefill @ 25.8k | 181 t/s (142.6 s TTFT) |

**Read the placement before reading the number.** llama.cpp's auto-fit put
**135.57 GB of a 139 GB model onto the two cards** — roughly 63 GiB per card.
Only ~3 GiB was ever CPU-resident.

So this is a **capability** result, and a better one than expected: *a 357B model
runs almost entirely inside 128 GB of VRAM at IQ3_XS, on two cards, correctly.*
It is **not** a measurement of CPU offload, and it does not validate the
bandwidth model above — the configuration the model describes (all experts in
RAM) was never run.

Two things it does show:

- **Prefill is the tier-4 problem, exactly as predicted.** 181–196 tok/s against
  the 80B's 4,668. That is 25× slower on a model 4.5× larger, and it is the
  `mimo` signature: a small CPU-resident fraction serializes the whole forward
  pass. **Decode is fine; prefill is what makes this tier painful.**
- **Decode falls 33% from short context to 25.8k** (12.83 → 8.51), steeper than
  the 80B's curve, consistent with 92 layers of GQA KV.

The `~3.6 t/s` projection earlier in this document remains **unvalidated** — it
neither matched nor was contradicted, because it describes a different placement.
`benchmarks/matrix/round2.sh` E7 forces `--n-cpu-moe` to 30/45/60 with this
auto-fit run as the N≈0 anchor, which is the actual test of whether decode
degrades linearly in pinned-layer count.

---

## If your model has no W8A8

Ranked by expected value.

1. **Check whether it does under another name.** Search
   `hf.co/models?search=<model> w8a8` and `<model> INT8`, and open
   `config.json` on anything 8-bit regardless of what the repo is called — the
   `cyankiwi` case above proves the name is unreliable in *both* directions.
2. **Take AWQ-Int4.** 58% behind W8A8, universally available, loads fine.
3. **Quantize it yourself with `llm-compressor`.** This is the tool that produced
   the RedHatAI checkpoints; a `W8A8` recipe with dynamic per-token activations
   is a one-off offline pass and needs no ROCm support, since it runs wherever
   you have the bf16 weights. **Untested on this stack** — it is the obvious
   route and it is written down here so it is not forgotten, not because it has
   been verified end to end.
4. **Do not** try to reach the INT8 path by forcing runtime activation
   quantization on a weight-only checkpoint. The scales in a W8A16 file were
   fitted for a bf16 GEMM; supplying activation scales at runtime changes the
   numerics and needs an accuracy check, not just a throughput measurement. See
   `docs/25` item 6.

---

## Quick reference

Before downloading anything, read three fields from `config.json`:

| Field | Want | Because |
|---|---|---|
| `quantization_config.quant_method` | `compressed-tensors` | fast loader — minutes, not hours |
| `...config_groups.group_0.input_activations` | **not null** | reaches `v_mfma_i32_16x16x16i8` |
| `head_dim` | `128` or `192` | AITER ASM attention exists (+12.8%) |

Two of three is a good checkpoint. The first one alone is the difference between
serving in two minutes and not serving at all.

And one decision before all three — **does it fit in 128 GB?**

| | Bound by | Take |
|---|---|---|
| **Fits in VRAM** | arithmetic (INT8 peak = bf16 peak) | the **largest** quant that fits — W8A8 |
| **Does not fit** | DDR4 bandwidth (~57 GB/s effective) | the **smallest** quant you can tolerate — GGUF Q4/IQ3 |

The same 8 bits that are free on the GPU are the entire cost in RAM.
