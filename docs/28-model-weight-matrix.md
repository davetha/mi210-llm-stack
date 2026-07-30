# Which model weights to run on 2× MI210 — the matrix

**Date**: 2026-07-29 · **Hardware**: 2× MI210 (gfx90a/CDNA2), 64 GB HBM2e each, 499 GB DDR4
**Every number below is measured on this box.** Raw data: `benchmarks/matrix/results/*.json`
**Mechanisms**: `docs/24` (quantization), `docs/26` (checkpoint selection), `docs/27` (matrix cores)

Prefill is tokens/s at a cold 15k prompt. Decode is tokens/s at ~101k context
unless noted. Load is wall-clock to first served request.

---

## The short answer

| Want | Run |
|---|---|
| **Best all-round** | `RedHatAI/*-quantized.w8a8` on vLLM at **TP=2** |
| **Biggest model worth running** | **Qwen3-235B-A22B UD-Q3_K_XL** — 104 GB fits, 698 t/s |
| **Smallest that still flies** | AWQ-Int4 on vLLM, TP=1 |
| **Long context (>128k)** | GGUF **Q8_0 / Q4_K_M** on llama.cpp |
| **Any expert layers on CPU** | a **K-quant** — I-quants lose 36% decode there |
| **Never** | **FP8**, `gptq`-packaged MoE, `compressed-tensors` 4-bit with `type:"int"`, **`HSA_XNACK` unified memory** |

---

## Tier 1 — ~30B MoE (Qwen3-30B-A3B-Thinking-2507)

| Weights | Engine | TP | GiB/rank | Load | Prefill | Decode | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **W8A8** `RedHatAI/...quantized.w8a8` | vLLM | **2** | **14.7** | **60 s** | **7,278** | **43.4** | ★ **best** |
| bf16 `Qwen/Qwen3-30B-A3B-Thinking-2507` | vLLM | 2 | 28.5 | **12,366 s** | 7,578 | **62.6** | faster, unusable load |
| W8A8 | vLLM | 1 | 29.2 | 123 s | 4,739 | 33.8 | ★ **best single-card** |
| AWQ-Int4 `QuantTrio/...AWQ` | vLLM | 1 | **15.7** | 115 s | 3,002 | 17.2 | ★ secondary — smallest |
| GGUF Q8_0 | llama.cpp | — | ~31 GB | — | 3,326 | 29.1 @230k | ★ long context |
| GGUF Q4_K_M | llama.cpp | — | ~18 GB | — | 3,416 | — | secondary long ctx |
| FP8 `Qwen/...FP8` | vLLM | 1 | 29.2 | 119 s | **1,047** | 9.3 | ✗ **avoid** |

**bf16 is the trap here.** It wins throughput — +4% prefill, +44% decode — and
costs **206× the load time**: 3.4 hours against 60 seconds, same model, same
cards. For anything that restarts more than about twice a month that is not a
close call. `docs/25` traces the cause to `hsakmt_ioctl` in the bf16 loader path.

**AWQ-Int4 is the memory pick**: 15.7 GiB, comfortably one card, and still 3×
faster than FP8.

---

## Tier 2 — ~80B MoE (Qwen3-Next-80B-A3B)

| Weights | Engine | TP | GiB/rank | Load | Prefill | Decode | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **W8A8** `RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8` | vLLM | 2 | 38.3 | **143 s** | **7,253** | 45.2 | ★ **best prefill** |
| W8A16 `cyankiwi/...AWQ-8bit` | vLLM | 2 | 41.0 | 184 s | 6,679 | **51.3** | ★ **best decode** |
| AWQ-Int4 `cyankiwi/...AWQ-4bit` | vLLM | 1 | 45.4 | 160 s | 4,487 | 45.9 | ★ single-card |
| GGUF Q4_K_M | llama.cpp | — | ~46 GB | — | 2,264 | **53.8 @230k** | ★ long context |

**This tier has a genuine split.** W8A8 takes prefill by 8.5%; W8A16 takes
decode by 14%. Pick by workload — long prompts and short answers favour W8A8,
long generations favour W8A16.

Two non-obvious facts about these checkpoints, neither visible in the repo name:

- **The W8A8 build has its MTP head stripped** (0 MTP tensors vs 4,625 in the
  cyankiwi builds). Irrelevant today — speculation loses on this hardware
  (`docs/25`) — but it is a real capability difference.
- **`head_dim = 256`**, so this model reaches *no* ROCm attention fast path: no
  AITER ASM, no custom paged attention, no benefit from the 256k patch. It is
  still the fastest thing in the matrix, entirely on architecture (`docs/27`).

---

## Dense models — much slower per token, plan accordingly

| Weights | TP | GiB/rank | Load | Prefill | Decode |
|---|---:|---:|---:|---:|---:|
| `RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8` | 2 | 33.9 | 421 s | **843** | 6.8 @64k |

**843 t/s against the 80B MoE's 7,253** — 8.6× slower on a *smaller* model.
Dense 70B activates all 70B parameters per token; Qwen3-Next activates 3B. Both
checkpoints are correct W8A8 at `head_dim = 128`, and this one loads the AITER
ASM kernels the 80B cannot.

> **Active parameters dominate everything else.** Fast paths are a multiplier on
> top; they are not a substitute. If you are choosing between a dense model and
> a sparse MoE of similar size on this box, the MoE wins by roughly the ratio of
> active parameters.

Use dense models when you need their quality or ecosystem, not for throughput.

---

## Tier 4 — ~357B MoE (GLM-4.6)

| Weights | Engine | On GPU | Prefill | Decode |
|---|---|---:|---:|---:|
| GGUF **UD-IQ2_M** `unsloth/GLM-4.6-GGUF` | llama.cpp | 135.02 GB | **217.8** | **10.85** @25.8k |
| GGUF IQ3_XS `bartowski/...` | llama.cpp | 135.57 GB | 195.9 | 8.51 @25.8k |
| AWQ-Int4 `bullpoint/GLM-4.6-AWQ` | vLLM + `--cpu-offload-gb 70` (**UVA**) | — | — | ✗ **unusable** |
| AWQ-Int4, same weights | vLLM + `--offload-backend prefetch --enforce-eager` | 131.16 | **695.9** | see below |
| AWQ-Int4, same weights | vLLM + `--offload-backend prefetch`, graphs on | — | ✗ **NCCL crash** | — |

**vLLM's *UVA* offload is not viable at this tier** — one 28k request ran past
35 minutes at steady 505% CPU. llama.cpp does the same work at 8.5 t/s.

That result was previously written up here as "vLLM's CPU offload is not
viable", which claims more than was measured. vLLM 0.23 has **two** offload
backends and only one was tested:

| Flag | Backend | Mechanism |
|---|---|---|
| `--cpu-offload-gb` | `UVAOffloader` | Weight **stays** in pinned host memory; the GEMM reads it across PCIe as it runs. No staging, no overlap. |
| `--offload-group-size` | `PrefetchOffloader` | Weight is **copied** to a static GPU buffer pool on a separate stream, double-buffered, layer N+1 landing while layer N computes. |

The one that was measured is the one that cannot overlap. `--offload-params
experts` further restricts either backend to the expert stacks, which on GLM-4.6
is ~168 GB of the 176 GB.

**The prefetch backend works, but only with `--enforce-eager`.** At 75% offload
it first crashed during load with `NCCL error: unhandled cuda error` in
`ncclAllReduce` (worker TP1), then hung with the container still alive. A control
run of `t35-w8a8` at TP=2 loaded normally on the same GPUs minutes later — 130 s,
41.88 GiB KV — proving it was the offloader, not the hardware.

`--enforce-eager` fixes it, which also identifies the cause. `PrefetchOffloader`
splices a private `torch.cuda.Stream` into CUDA-graph captures *by design* — its
docstring says events "allow the copy stream to join CUDA graph captures" — and
it patches module forwards to insert `wait_prefetch`/`start_prefetch` ops into
the captured graph. **RCCL collectives inside that capture are what break.**
Removing graph capture removes the collision. So the feature is not broken on
ROCm; it is incompatible with graph capture there, which is a more useful fact.

```bash
--offload-backend prefetch --offload-params experts \
--offload-group-size 4 --offload-num-in-group 3 \
--enforce-eager                     # REQUIRED — without it, NCCL dies at load
```

**What is and is not accelerated in that result.** Verified from the `.co` load
lines, not inferred:

| AITER component | State |
|---|---|
| ASM flash attention (prefill) | **active** — `fwd_hd128_bf16_rtna_group.co` loaded, matching GLM-4.6's `head_dim=128` |
| Custom paged attention (decode) | active — `pa_v1` JIT at `head_size=128, gqa_ratio=12` |
| AITER linear / GEMM | **off** — `VLLM_ROCM_USE_AITER_LINEAR=0` |
| AITER MoE | **off** — `VLLM_ROCM_USE_AITER_MOE=0` |

So 695.9 t/s is with ASM attention but **stock GEMM and stock MoE**, leaving
headroom. Note `fmoe` — the highest MACs/instruction kernel on this box
(`docs/33`) — is doubly unavailable to this arm: disabled by env, *and* all 8
gfx90a objects are `noquant{Fp16,Bf16}`, so they cannot consume AWQ-Int4 weights.

**`VLLM_PREFER_AITER_FA` is dead on vLLM 0.23.** The startup log says
`Unknown vLLM environment variable detected: VLLM_PREFER_AITER_FA`. AITER FA is
still selected, but on backend priority order rather than by the flag:

```
Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
Overriding with ROCM_AITER_FA out of potential backends:
  ['ROCM_AITER_FA', 'ROCM_ATTN', 'TRITON_ATTN']
```

The +8% @15k / +33% @101k this doc previously credited to the flag was a real
measurement, but the flag no longer causes it. Harmless to keep; do not rely on
it, and do not assume its absence means AITER FA is off — check the `.co` line.

**vLLM has no managed-memory path at all** — zero references to
`hipMallocManaged`/`cudaMallocManaged` in the package. So the llama.cpp
unified-memory arms (round 15) and this are *not* the same mechanism: that one
is OS demand paging, page-granular and unscheduled; this one is explicit,
tensor-granular and prefetched.

### Do NOT use `HSA_XNACK=1` unified memory on this host

Tried, and it does not merely lose — it damages the host. `HSA_XNACK=1` plus
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` with `-ngl 999` on GLM-4.6 IQ3_XS never
finished loading:

```
llama-server: hsa-runtime/core/runtime/runtime.cpp:2026:
  Runtime::VMFaultHandler: Assertion `false && "GPU memory access fault."'
```

The kernel log shows this is **not** a missing-page fault: `MAPPING_ERROR: 0x0`
and `WALKER_ERROR: 0x0` (the page is mapped), with `PERMISSION_FAULTS: 0x5` and
`RW: 0x1` — a device **write** denied on a resident managed page, which the
driver then failed to promote, degrading a recoverable `retry page fault` into a
fatal `no-retry` one. Retry faults are enabled, so `amdgpu.noretry` is not the
problem. Full analysis in `docs/29`.

The abort left an rwsem with no live owner, and amdgpu's SVM eviction workers
stacked up behind it:

```
INFO: task kworker/44:10 blocked for more than 122 seconds.
Workqueue: events svm_range_evict_svm_bo_worker [amdgpu]
... blocked on an rw-semaphore, but the owner is not found.
```

Load average hit **70 and was still climbing**. Killing the arm drained it
(70 → 34 → 20) and both GPUs recovered without a reset — `rocminfo` enumerated
normally at 42–47 °C idle — but nothing stopped on its own: the harness
recorded the failure and advanced to the next `-ub` value, re-triggering it.

The prefill-amortisation argument for RAM-resident weights with GPU compute is
untouched by this — **the fault happens at load, before any of it is
exercised**. The idea is not refuted; this route to it is. `--offload-backend
prefetch` on vLLM pursues the same goal with explicit staged copies and no
XNACK. Kernel 7.0.0-28-generic, ROCm 7.14.

`benchmarks/matrix/round15_unified_mem.sh` now refuses to run without
`ALLOW_UVM_HANG=1`.

**UD-IQ2_M wins both axes** — +11.2% prefill and **+27.5% decode** on a matched
15k/25.8k pair. An earlier partial run put the prefill gain at +6.1% (208 t/s);
the completed arm measures 217.8, and the decode gap is the larger effect.

### Do not hand-place experts — `--n-cpu-moe` loses to auto-fit at every value

Matched arms, same model (IQ3_XS), same prompts, only the placement changing:

| Placement | Prefill @15k | Prefill @25.8k | Decode @25.8k |
|---|---:|---:|---:|
| **auto-fit** (135.57 GB on GPU) | **195.9** | **180.9** | **8.51** |
| `--n-cpu-moe 60` | 159.7 | 151.9 | 7.47 |
| `--n-cpu-moe 70` | 153.7 | 145.8 | 7.06 |
| `--n-cpu-moe 80` | 140.9 | 140.4 | 6.77 |
| `--n-cpu-moe 92` (all experts) | 141.0 | 134.1 | 6.36 |

Monotone: every expert layer moved to the CPU costs throughput, and there is no
knee to tune toward. Auto-fit beats the best manual value by **23% prefill** and
**14% decode**.

An earlier note here claimed the curve went "flat past 60". It does not — that
came from comparing an auto-fit *cold16k* number against `--n-cpu-moe`
*longctx* numbers. Matched workloads show a steady decline on both.

**Reach for `--n-cpu-moe` only when the model does not otherwise fit.** It is a
capacity mechanism, not a performance one.

### The CPU expert path is issue-bound, not DDR4-traffic-bound

Worth stating because the traffic model predicts the opposite and is wrong.

The argument for raising `-ub` on the CPU path: with `E` experts and `k` active
per token, distinct experts touched per chunk is `E·(1−(1−k/E)^c)`, which
saturates at `E` once `c ≳ E/k`. GLM-4.6 routes 8 of 160, so saturation is ~20
tokens and `-ub 2048` is 100× past it — every chunk drags the whole CPU-resident
expert set across DDR4. Total traffic is then `(n/c)·E·bytes`, **monotonically
decreasing in `c`**, predicting ~2× from 2048 → 4096.

Measured, at `--n-cpu-moe 60`:

| `-ub` | Prefill @15k |
|---:|---:|
| 1024 | 119.6 |
| **2048** | **164.6** |
| 4096 | 158.1 |

**4096 is 4% worse, not 2× better.** The traffic model is refuted — the expert
weights are not what the chunk size is trading against.

Two independent lines say the binding constraint is dequantization work, not
bytes moved:

1. **K-quants beat I-quants by 36% decode on this path** at essentially equal
   byte counts (~135 GB either way, above). Traffic cannot explain a 36% gap
   between two files of the same size; per-weight instruction cost can —
   K-quants are scaled integer blocks AVX2 handles directly, I-quants need a
   codebook lookup with no vector equivalent on Zen3.
2. `/proc/cpuinfo` has no `avx_vnni`, no `avx512_vnni`, no `amx_int8`, and
   `libggml-cpu.so` disassembles to **927 `vpmaddubsw` / 722 `vpmaddwd` / zero
   `vpdpbusd`** — the classic three-instruction AVX2 int8 sequence, which is
   correct here rather than a missed optimisation, and which fixes the per-weight
   cost floor.

So the lever on the CPU path is **thread placement and quant family**, not chunk
size. `-t 24` (physical cores) is worth 18% decode over 48; `-tb` barely matters;
`-ub 2048` was already right.

---

## Tier 3 — ~235B MoE (Qwen3-235B-A22B-Instruct-2507) — **the sweet spot**

| Weights | Engine | On GPU | Prefill @15k | Prefill @25.8k | Decode |
|---|---|---:|---:|---:|---:|
| GGUF **UD-Q3_K_XL** `unsloth/...GGUF` | llama.cpp | 104.2 GB | **698.2** | 628.6 | **23.09** |
| GPTQ-Int4 `...t235-gptq4` | vLLM | — | — | — | ✗ never loads |

**3.4× the prefill and 2.7× the decode of GLM-4.6 (357B), on a model only 34%
smaller.** TTFT at 15k is 21.8 s against GLM's 77.5 s.

The reason is that **it fits**. 104.2 GB sits inside 128 GB of VRAM with room
for KV; GLM-4.6's 139 GB does not, and everything after that is paging.

### Active-parameter scaling — apply it **within one engine**

An earlier version of this section compared llama.cpp measurements against a law
derived from a *vLLM* baseline and concluded GLM-4.6 ran "3.5× below prediction",
calling the gap the cost of not fitting. That mixed engines. Corrected, using each
engine's own 3B-active baseline:

**llama.cpp** (baseline: Qwen3-30B Q4_K_M, 3,415.7 t/s @ 3B active)

| Model | Active | Predicted | Measured | |
|---|---:|---:|---:|---|
| **Qwen3-235B-A22B** UD-Q3_K_XL | 22B | ~466 | **698** | **1.5× above** |
| GLM-4.6 IQ3_XS | 32B | ~320 | 196–218 | ~1.6× below |

**vLLM** (baseline: Qwen3-Next-80B W8A8, 7,253 t/s @ 3B active)

| Model | Active | Predicted | Measured | |
|---|---:|---:|---:|---|
| GLM-4.6 AWQ-Int4 + prefetch offload | 32B | ~680 | **695.9** | on the line |

Within llama.cpp the fits/doesn't-fit spread is **~2.2×, not 3.5×** — and the
235B *beats* its own engine's law, which the cross-engine version hid completely.

**The law is too crude to carry more than a direction.** It ignores quantization
format, attention geometry, and batch configuration, and the vLLM row uses a
different quant (AWQ-Int4) from its own baseline (W8A8) — where tier 1 measures
AWQ-Int4 *slower* than W8A8 at equal TP, so that row should have over-predicted
and did not. Treat these as order-of-magnitude sanity checks.

> **Choose the largest model that still fits VRAM, not the largest you can
> page** — on llama.cpp with auto-fit placement. The 235B is the sweet spot there
> and it is measured, not inferred.

**Deliberate paging is a different regime.** GLM-4.6 AWQ-Int4 under vLLM's
*prefetch* offloader — 75% of expert layers, ~126 GB in host RAM — measures
**695.9 t/s prefill** (TTFT 21.92 s median, three reps at 15k, VRAM 131.16 GB),
against 69.5–77.5 s TTFT for the same model kept on the cards as GGUF.

**How much of that is the offloading is not yet established.** The arms differ in
engine, tensor-parallel layout, batch size *and* placement simultaneously, and
llama.cpp is not generally behind vLLM — at tier 1, GGUF Q4_K_M (3,416) beats
vLLM AWQ-Int4 TP=1 (3,002). The isolating experiment is the 67% and 50% offload
arms of round 18, which vary **only** the offload fraction: if prefill barely
moves, the win is engine and configuration rather than the offload mechanism.
Decode is the opposite story either way — see the offload section.

---

### I-quants vs K-quants at tier 4 — K-quants own decode, and own CPU offload

Same model (GLM-4.6), same placement, ~135 GB on GPU in every case:

| Weights | Family | Prefill @15k | Decode @25.8k |
|---|---|---:|---:|
| UD-IQ2_M | I-quant | **217.8** | 10.85 |
| **UD-Q2_K_XL** | K-quant | 208.5 | **11.52** |
| IQ3_XS | I-quant | 195.9 | 8.51 |

No family sweeps both: UD-IQ2_M takes prefill by 4.5%, UD-Q2_K_XL takes decode
by 6.2%, and IQ3_XS loses to both despite being the largest.

**With experts on the CPU the K-quant advantage widens sharply.** At
`--n-cpu-moe 60`:

| Weights | Prefill | Decode |
|---|---:|---:|
| UD-Q2_K_XL | **166.2** | **10.13** |
| IQ3_XS | 159.7 | 7.47 |

**+36% decode.** K-quants are scaled integer blocks that AVX2 handles directly;
I-quants need a codebook lookup per weight, which has no vector equivalent on
Zen3. On the GPU both reach MFMA — `ggml_cuda_should_use_mmq` admits IQ2_XXS
through IQ4_XS, and every MoE here clears its `n_experts > 64` short-circuit
anyway — so the gap only opens once the CPU is doing the arithmetic.

> **If any expert layer will live on the CPU, pick a K-quant.**

---

## Do NOT use — with the reason

| Weights | Why | Evidence |
|---|---|---|
| **FP8, any model** | CDNA2 has **no FP8 ALU**. `v_mfma_f32_*_fp8_fp8` is rejected by the assembler. Every weight upcasts to bf16, so you pay dequantization and get nothing. | **1,047 t/s vs AWQ-Int4's 3,002 — 2.9× slower** |
| **`gptq`-packaged MoE** | `moe_wna16_weight_loader` runs per expert per weight, single-threaded Python. 128 experts × 48 layers. | **>46 min, never reached serving.** 235B projects to ~7 h |
| **`compressed-tensors` 4-bit with `type:"int"`** | The one ROCm-capable mixed-precision kernel accepts `uint4b8`/`uint4`, **not signed `int4`**. AWQ and GPTQ emit the unsigned form and work. | Fails at load: `Quant type int4 not supported` |
| **W4A8 (any)** | Two independent blockers: no ROCm kernel takes int8 activations, *and* none accepts signed int4 weights. | `docs/27` — kernel-by-kernel refusal |
| **bf16 for a MoE you restart** | 12,366 s load against 60 s for the same model as W8A8. | 206× |
| **`--max-model-len` > 131072 on stock vLLM** | Graph capture bakes the Triton fallback into *every* request. | **10× decode.** Fixed by `configs/extend_rocm_pa_256k_gfx9.py` — 0.75 → 16.0 t/s |
| **Speculative decoding** | Decode here sits **6.4× off its bandwidth bound**, so verifying N+1 tokens costs ~N+1×, not ~1×. Loses even at 100% acceptance. | 6.85 → 6.01 t/s with every token accepted |

---

## Reading a checkpoint before you download it

Four fields, in `config.json` and the tensor index. None is in the repo name.

| Check | Want | Costs you if wrong |
|---|---|---|
| `quantization_config.quant_method` | `compressed-tensors` | hours of load time |
| `...group_0.input_activations` | populated, **`type: "int"`** | `"float"` is FP8 — dead on CDNA2 |
| `head_dim` | 128 or 192 | no AITER ASM prefill (+8–33%) — **bf16 objects only**, no fp16 |
| `num_attention_heads / num_key_value_heads` | **exactly 8 or 16** | no ASM paged attention — that is the entire `pa` coverage (`docs/35`) |
| `kv_lora_rank` | check **before** the head counts | if present the model is MLA and `num_key_value_heads` is vestigial — misreading this produced a 44× KV error (`docs/35`) |
| tensor index for `mtp`/`nextn` | present, if you want MTP | quantizers strip it silently |

Worked example of why the name lies: `cyankiwi/...AWQ-8bit` is **not AWQ** — it
is `compressed-tensors` `pack-quantized` with `input_activations: null`, i.e.
weight-only W8A16. It is a perfectly good checkpoint; it is just not what it
says.

---

## Flags that are worth real throughput

```bash
-e HSA_NO_SCRATCH_RECLAIM=1        # required on MI200
-e NCCL_P2P_DISABLE=1              # PCIe-only, no XGMI
-e VLLM_ROCM_USE_AITER=1
-e VLLM_ROCM_USE_AITER_MHA=1
-e VLLM_PREFER_AITER_FA=1          # DEAD on vLLM 0.23 — see below. Harmless to leave.
  --max-model-len 131072           # never higher on stock vLLM
  --max-num-batched-tokens 8192    # +11% prefill; default is 2048, plateaus at 8192
  --no-enable-prefix-caching       # benchmarking only; leave ON in production
```

llama.cpp: `-ub 2048` (4096 and 8192 are both measurably worse), and if you set
`-ngl` you must also set `--n-cpu-moe` — `common_fit_params()` is all-or-nothing.

### `GGML_CUDA_FORCE_MMQ` — worth it only when VRAM is the constraint

`ggml_cuda_should_use_mmq()` returns false past `ne11 > 256` for Q4_K/Q5_K, so at
`-ub 2048` the attention projections and dense layers dequantize to fp16 and call
hipBLAS instead of using the int8 MFMA kernels. Forcing MMQ on (a compile-time
`#ifdef`, so it needs a separate build — `configs/Dockerfile.llama-forcemmq`)
gives opposite answers on the two ends of the matrix:

| Model | Prefill @15k | Prefill @25.8k | Decode |
|---|---:|---:|---:|
| Qwen3-30B Q4_K_M, baseline | **3,415.7** | **2,891.3** | 84.87 |
| Qwen3-30B Q4_K_M, forced | 3,349.8 | 2,866.4 | 85.04 |
| | **−1.9%** | **−0.9%** | +0.2% |
| GLM-4.6 IQ3_XS, baseline | 204.5 | 180.6 | 8.62 |
| GLM-4.6 IQ3_XS, **forced** | **214.3** | **189.5** | 8.64 |
| | **+4.8%** | **+4.9%** | +0.2% |

**Decode is unmoved in both cases** (+0.2%, noise), which is what the heuristic
predicts — it keys on `ne11`, and decode never leaves the small-batch branch
where MMQ already wins.

The prefill split is the interesting part, and it is probably not about the
arithmetic. The 30B is fully resident, so hipBLAS's fp16 GEMM is simply fast and
MMQ's int8 path is marginally behind it — upstream's cutoff is correct there.
GLM-4.6 saturates VRAM, and the dequantize path has to *materialise an fp16 copy
of the weights* to hand to hipBLAS. MMQ consumes the quantized weights directly.
When bandwidth and VRAM are the binding constraint, skipping that temporary is
worth ~5%; when they are not, it costs 1–2%.

> **Rebuild with `-DGGML_CUDA_FORCE_MMQ=ON` only for models that saturate VRAM.**
> For anything that fits comfortably, upstream's default is the better choice.

**Verify AITER actually ran**, don't assume: the `.co` load line is the proof,
not the backend-selection line.

```
Overriding with ROCM_AITER_FA            <- selected
fwd_hd128_bf16_causal_rtna_group.co      <- ACTUALLY LOADED
```
