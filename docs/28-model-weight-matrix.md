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
| AWQ-Int4, same weights | vLLM + `--offload-backend prefetch` | — | ✗ **NCCL crash** | — |

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

**The prefetch backend does not currently work here either.** At 75% offload it
crashed during load with `NCCL error: unhandled cuda error` in `ncclAllReduce`
(worker TP1), then hung with the container still alive. A control run of
`t35-w8a8` at TP=2 loaded normally on the same GPUs minutes later — 130 s,
41.88 GiB KV cache — so this is the offloader, not the hardware. Untested
mitigations: `--enforce-eager` (it joins a private copy stream into CUDA-graph
captures, the most likely thing to break RCCL) and
`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`. See `docs/29`.

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

That shows up cleanly against the active-parameter law from the dense section:

| Model | Active params | Fits? | Prefill | Predicted from 80B ×(3/active) |
|---|---:|---|---:|---:|
| Qwen3-Next-80B-A3B (W8A8) | 3B | yes | 7,253 | — |
| **Qwen3-235B-A22B** | 22B | **yes** | **698** | ~989 |
| GLM-4.6 (357B) | 32B | **no** | 196 | ~680 |

The 235B lands within ~30% of what active-parameter scaling predicts. GLM-4.6
comes in **3.5× below** its prediction — that gap is the cost of not fitting,
not the cost of being bigger.

> **Choose the largest model that still fits VRAM, not the largest you can
> page.** Crossing the VRAM line costs far more than the parameters buy.

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
| `head_dim` | 128 or 192 | no AITER ASM (+8–33% prefill), no custom paged attention |
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
-e VLLM_PREFER_AITER_FA=1          # +8% @15k, +33% @101k — and it is NOT selected without this
  --max-model-len 131072           # never higher on stock vLLM
  --max-num-batched-tokens 8192    # +11% prefill; default is 2048, plateaus at 8192
  --no-enable-prefix-caching       # benchmarking only; leave ON in production
```

llama.cpp: `-ub 2048` (4096 and 8192 are both measurably worse), and if you set
`-ngl` you must also set `--n-cpu-moe` — `common_fit_params()` is all-or-nothing.

**Verify AITER actually ran**, don't assume: the `.co` load line is the proof,
not the backend-selection line.

```
Overriding with ROCM_AITER_FA            <- selected
fwd_hd128_bf16_causal_rtna_group.co      <- ACTUALLY LOADED
```
