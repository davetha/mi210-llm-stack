# Which quantization wins on an MI210

**Date**: 2026-07-28 · **Hardware**: 2× AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each, PCIe-only
**Software**: ROCm 7.14, vLLM 0.23.1, llama.cpp, amd-aiter 0.1.19, PyTorch 2.11
**Reproduce**: `benchmarks/matrix/REPRODUCE.md` · **Raw data**: `benchmarks/matrix/results/*.json`

The short answer: **INT8 W8A8**, and it was the one format vLLM refused to run.

The longer answer is that on this card three of the biggest performance factors
are not quantization choices at all. They are code paths that upstream disables
on gfx90a — two by architecture checks that do not describe the hardware, and
one by a graph-capture detail that silently bakes a slow kernel into every
request.

---

## Tier 1 — Qwen3-30B-A3B-Thinking-2507 (30B total / 3.3B active, Apache-2.0)

Cold 16k prompt, single MI210, TP=1, prefix caching off, correctness probe
passing on every row.

| Quant | Engine | Weights | TTFT | Prefill | Decode @ ~100k |
|---|---|---:|---:|---:|---:|
| bf16 **(TP=2 — see below)** | vLLM + AITER | 28.5 GiB/rank | **2.03 s** | **7,578 t/s** | **62.6 t/s** |
| **INT8 W8A8** | vLLM + AITER | **29.2 GiB** | **3.20 s** | **4,739 t/s** | **33.8 t/s** |
| Q4_K_M | llama.cpp | ~18 GB | 4.43 s | 3,416 t/s | — |
| Q8_0 | llama.cpp | ~31 GB | 4.57 s | 3,326 t/s | 30.4 @ **230k** |
| AWQ-Int4 | vLLM + AITER | 15.7 GiB | 5.07 s | 3,002 t/s | 17.2 t/s |
| AWQ-Int4 | vLLM stock | 15.7 GiB | 5.72 s | 2,661 t/s | — |
| FP8 | vLLM + AITER | 29.2 GiB | 14.51 s | 1,047 t/s | 9.3 t/s |

### The bf16 row is two cards, and it is not comparable to the rest

**Every other row in that table is TP=1. bf16 is TP=2**, because 57 GiB of
weights will not share a 64 GB card with a KV cache. So its 7,578 t/s — the
highest number anywhere in this matrix, above even the 80B's 6,679 — is two
cards against one, and quoting it beside the W8A8 row compares hardware, not
quantization. This is the same error `docs/26` corrects for tier 2.

What makes it interesting is the **per-card footprint**:

| | bf16 TP=2 | W8A8 TP=1 |
|---|---:|---:|
| weights **per card** | 28.51 GiB | 29.17 GiB |
| KV cache | 28.12 GiB | 27.23 GiB |
| **KV tokens** | **614,304** | 297,472 |
| **model load** | **12,366 s (3.4 h)** | **123 s** |

**bf16 at TP=2 occupies almost exactly the same per-card memory as W8A8 at
TP=1** — 28.5 against 29.2 GiB. Halving the bytes and doubling the cards lands
in the same place. The bf16 arm then gets 2.07× the KV tokens because it has two
cards' worth of KV, and roughly 1.6× prefill / 1.85× decode, which is about what
doubling the hardware should buy.

### E0 ran, and it changes the headline

The TP=2 W8A8 control is in. At **equal hardware**:

| 30B, TP=2 | TTFT | Prefill | Decode @101k | Load |
|---|---:|---:|---:|---:|
| **bf16** | **2.03 s** | **7,578 t/s** | **62.6 t/s** | 12,366 s |
| W8A8 | 2.08 s | 7,278 t/s | 43.4 t/s | **59.7 s** |

**bf16 wins on throughput at equal TP** — 4% on prefill and **44% on decode**.
So "INT8 W8A8 is the fastest format" was an artifact of bf16 being unable to fit
on one card, not a property of the arithmetic.

> **UPDATE — the decode gap was diagnosed, and only half-fixed.** The
> speculation immediately below (poorly-tuned Triton INT8 kernel at batch-1
> decode) was **correct**, and [`docs/43`](43-ck-int8-gemm-gfx90a.md) replaced
> that kernel for **1.480× decode on this 30B**. But it does **not** generalize:
> [`docs/55`](55-vllm-vs-llamacpp-decode-on-the-80b.md) measures the same flag
> on the 80B at **0.982×** — a slight regression — with the selected kernel
> asserted from the server log.
>
> The 30B decode rows below have **not** been re-measured with the CK GEMM
> enabled, so the "bf16 wins decode by 44%" conclusion in this section is
> **unverified as of `docs/43`** and should not be quoted without that caveat.

The decode gap is the surprising half and is worth flagging as unexplained:
INT8 halves weight-memory traffic, so it should *win* a bandwidth-bound decode.
Losing by 44% points at the Triton INT8 MoE kernel being poorly tuned at
batch-1 decode shapes rather than at anything architectural — consistent with
`docs/25` item 4, where vLLM ships tuned `fused_moe` configs for MI300X/MI325X
and none for MI210. Not yet measured.

**But the load times invert the recommendation completely:**

> At TP=2, bf16 buys **+4% prefill and +44% decode** for **207× the load time**
> — 3.4 hours against 60 seconds, same model, same cards, same day.

For a server that restarts more than about twice a month, that is not a close
call. And W8A8 remains the only one of the two that runs at all on a single
card. Load-path analysis in `docs/25` item 1.

The 3.4-hour load is its own finding, and it is not I/O: see `docs/25` item 1,
where `py-spy --native` puts the loader in `hsakmt_ioctl`.

### Attribution, stated precisely

The 3.20 s result has **both** patches active — verified from the server log,
not assumed:

```
Overriding with ROCM_AITER_FA            <- ASM attention selected
fwd_hd128_bf16_causal_rtna_group.co      <- ASM kernels actually loaded
Using TRITON Int8 MoE backend            <- INT8 MoE path reached
```

So the honest decomposition is:

| Step | TTFT | What varied |
|---|---:|---|
| AWQ-Int4, stock vLLM | 5.72 s | — |
| AWQ-Int4 + AITER ASM attention | 5.07 s | attention only, **+12.8%** |
| INT8 W8A8 + AITER ASM attention | 3.20 s | quantization only, **+58%** |

The +58% is INT8 versus AWQ **with AITER held constant on both sides**, which
is the comparison that matters. What is *not* measured is INT8 without AITER,
so nothing here supports a claim about INT8's standalone contribution.

### CORRECTION: on a MoE, "W8A8" is W8A8 dense + **W8A16 experts**

The claim below — that W8A8 reaches `v_mfma_i32_16x16x16i8` — is **true for the
dense linear layers and false for the MoE experts**, which is where nearly all
of a sparse MoE's compute and weight traffic actually live.

Caught by a deliberate check that the tuned MoE config had loaded. It had not,
and the filename vLLM asked for gave the game away:

```
WARNING [fused_moe.py:1071] Using default MoE config...
Config file not found at .../E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
```

`dtype=int8_w8a16` — for a W8A8 checkpoint. In
`fused_moe/config.py::get_config_dtype_str`:

```python
if use_fp8_w8a8:      return "fp8_w8a8"
elif use_fp8_w8a16:   return "fp8_w8a16"
elif use_int8_w8a16:  return "int8_w8a16"
elif use_int4_w4a16:  return "int4_w4a16"
```

**There is no `use_int8_w8a8` branch.** The string is emitted only when
`use_int8_w8a16` is set, so the expert GEMMs are running int8 weights against
**16-bit activations** — dequantize to bf16, exactly like AWQ or GPTQ. The same
server log confirms the dense path is genuinely W8A8:

```
Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8   <- dense: true W8A8
Using TRITON Int8 MoE backend                                          <- experts: int8_w8a16
```

**This resolves the decode anomaly that appeared twice.** W8A8 loses decode to
bf16 (43.43 vs 62.59) and to W8A16 at tier 2 (45.19 vs 51.34) because its
experts never do INT8 arithmetic — they do bf16 arithmetic *plus* a
dequantization step, which is strictly worse than plain bf16 when decode is
compute-bound.

It also explains why prefill still wins: prefill is bandwidth-bound on weights,
where int8 halves traffic regardless of what the GEMM does with it.

So the honest statement for a **sparse MoE** is:

> W8A8 buys **weight bandwidth** (prefill) and not **arithmetic** (decode),
> because the expert GEMMs run w8a16 no matter what the checkpoint says.

The `+58%` prefill figure below stands. The implied claim that it comes from
reaching INT8 matrix units does not — for the experts it comes from halved
memory traffic. On a **dense** model the original claim holds unmodified.

### It is 8-bit *activations* that win, not 8-bit weights

W8A8 is not a widely published format, so the obvious question is whether the
popular 8-bit checkpoints — GPTQ-Int8, AWQ-8bit — get the same result. They do
not, and the reason is structural rather than a tuning gap.

| Format | Weights | Activations | Reaches `v_mfma_i32_16x16x16i8`? |
|---|---|---|---|
| **INT8 W8A8** | int8 | **int8** | **yes** |
| GPTQ-Int8 | int8 | bf16 | no |
| AWQ-8bit | int8 | bf16 | no |

Confirmed by profile, not inferred from timings. `py-spy` on a GPTQ-Int8 arm
puts every sample in:

```
moe_wna16_weight_loader (quantization/moe_wna16.py:445)
```

`moe_wna16` is the **weight-only** path: it dequantizes to bf16 and runs a bf16
GEMM. The 8 bits buy memory traffic and nothing else. Only W8A8 quantizes
activations as well, which is what lets the GEMM be int8 × int8 and reach the
INT8 matrix units where CDNA2 is fastest.

**So the rule on this hardware is "8-bit activations", not "8-bit".** When
choosing a checkpoint, `w8a8` / `compressed-tensors` is the thing to look for;
`w8a16` — however it is labelled — will not reach the hardware that makes INT8
worth choosing here.

### The popular formats also load far more slowly

| Format | Load time (TP=1, same model) |
|---|---:|
| INT8 W8A8 (`compressed-tensors`) | **123 s** |
| GPTQ-Int8 (`auto_gptq` → WNA16) | **>46 min, never reached serving** |

The WNA16 loader does a `loaded_weight.to(device)` plus a format conversion
**per expert per weight** — 128 experts × 48 layers, single-threaded. Weights
reach VRAM early (32.15 GB resident) and the remaining time is pure CPU
conversion.

The GPTQ arm was stopped at 46 minutes, still inside
`moe_wna16_weight_loader`, never having served a request. **That absence is the
result, not a gap in the data**: on this stack a GPTQ-Int8 MoE of this size does
not reach serving in a practical time. No throughput row is reported for it
because none was ever produced, and inventing one from a partial load would be
worse than the empty cell.

AWQ-8bit was dropped for the same reason rather than measured: it uses the same
`moe_wna16` loader and the same w8a16 scheme, so it would have cost another
45+ minutes to confirm a mechanism already established by profile.

> **Later finding — the inference in that paragraph does not generalise.** The
> w8a16 half is right; the loader half is not. Tier 2's
> `cyankiwi/Qwen3-Next-80B-A3B-Thinking-AWQ-8bit` is *named* AWQ but is packaged
> `compressed-tensors` / `pack-quantized`, so it takes the fast loader and was
> resident in **183.7 s**. **Load time follows `quant_method`, not the repo
> name and not the bit width** — only `gptq` packaging hits the pathological
> per-expert loop. Full two-axis breakdown in `docs/26`.

### INT8 wins because CDNA2's INT8 peak equals its bf16 peak

gfx90a provides `v_mfma_i32_16x16x16i8` and peaks at **181 TOPS INT8 — the same
as its 181 TFLOP/s bf16 peak.** So INT8 halves weight-memory traffic against
bf16 at *no arithmetic cost*. That is a different economics from CDNA3, where
FP8 is 2× bf16 and INT8 is not the obvious choice.

**FP8 is a trap here, decisively: 2.9× slower than AWQ-Int4.** CDNA2 has no FP8
ALU, so every weight is upcast to bf16 before the GEMM. The bandwidth saving
does not come close to covering it. The output is correct — it is simply slow.

Note also that **Q8_0 costs almost nothing over Q4_K_M** (4.57 s vs 4.43 s)
despite being ~31 GB against ~18 GB. Consistent with the same ceiling: on this
card lighter quantization buys bandwidth, not arithmetic.

---

## Tier 2 — the 80B, and a comparison that has to be read carefully

| Model | Quant | TP | Prefill @15k | Decode @~101k | KV |
|---|---|---|---:|---:|---:|
| 30B | w8a8 | 1 | **4,739 t/s** (3.20 s) | 33.8 t/s | 27.2 GiB |
| 80B | awq-int4 | 1 | 4,487 t/s (3.39 s) | **45.9 t/s** | 10.4 GiB |
| 80B | awq-int8 | **2** | **6,679 t/s** (2.27 s) | **51.3 t/s** | 14.9 GiB |

**At equal hardware the 30B is slightly faster on prefill.** 4,739 against 4,487
tok/s. The 80B's headline 6,679 tok/s needed a second card, so quoting it beside
a TP=1 row compares two cards against one, not two models.

### Tier 2 completed: W8A8 wins prefill and loses decode

Both 80B checkpoints, TP=2, same flags — the decode number took three attempts
(see `docs/25`) and now exists:

| 80B, TP=2 | prefill @15k | prefill @101k | **decode @101k** | weights | load |
|---|---:|---:|---:|---:|---:|
| **W8A8** (`RedHatAI`) | **7,249** | **4,943** | 45.19 t/s | **38.31 GiB** | **142.8 s** |
| W8A16 (`cyankiwi`) | 6,679 | 4,668 | **51.34 t/s** | 40.95 GiB | 183.7 s |

W8A8 takes prefill by **+8.5%**, uses less memory and loads faster — but **loses
decode by 12%**.

**That is the second independent occurrence of the same pattern.** At tier 1,
W8A8 at TP=2 decoded 43.4 t/s against bf16's 62.6 on identical hardware. Twice
now the more heavily quantized format has lost a bandwidth-bound decode it
should win, since INT8 halves weight traffic.

Two confirmations make the untuned-kernel explanation much more likely than
coincidence: vLLM ships tuned `fused_moe` configs for MI300X/MI325X and **none
for MI210**, so the Triton INT8 MoE path runs on generic heuristics at batch-1
decode shapes. A tuned config for this card now exists
(`E=128,N=768,device_name=AMD_Instinct_MI210.json`) and re-measuring against it
is the open item.

**The decode win is real and architectural.** At the same TP=1, the 80B decodes
**45.9 tok/s against the 30B's 33.8** — 36% faster on a model nearly three times
larger, with a KV cache of 10.4 GiB against 27.2. Qwen3-Next is a 3:1 Gated
DeltaNet hybrid, so only 12 of 48 layers store KV. At long context the KV cache
sets decode throughput, and this is the one place where picking a bigger model
makes serving *cheaper*.

So the defensible tier-2 statements are narrower than "the 80B is faster":
prefill is roughly a wash at equal hardware and favours the 30B slightly;
long-context decode favours the 80B substantially; and the fastest absolute
numbers in this matrix come from the 80B at TP=2.

## The three things that mattered more than the format

### 1. INT8 MoE was disabled on every AMD GPU by a CUDA check

Serving an INT8 W8A8 MoE checkpoint failed outright:

```
NotImplementedError: No Int8 MoE backend supports the deployment configuration.
```

vLLM has exactly one Int8 MoE backend, and in
`fused_moe/experts/triton_moe.py`:

```python
# INT8 requires at least 7.5 (Turing).
device_supports_int8 = (
    current_platform.is_cuda()                        # False on ROCm
    and current_platform.has_device_capability((7, 5))
)
```

`_supports_current_device()` in the **same class** accepts ROCm via
`is_cuda_alike()`. So the kernel is declared runnable on AMD, then has its INT8
scheme refused by a Turing compute-capability test — deciding whether AMD
hardware may use its own INT8 matrix units.

Patch: `configs/enable_int8_moe_rocm.py`, scoped to gfx9. Effect: this arm went
from *not starting at all* to **the fastest result in the matrix**.

### 2. AITER's ASM attention was unreachable, then reachable but never chosen

Stock vLLM cannot use AITER on an MI210 and says nothing about it. Four layers
had to be fixed, and each failed independently:

| Layer | Problem |
|---|---|
| AITER version | image ships 0.1.13; patches target 0.1.17+ → installed **v0.1.19** |
| triton | installing AITER downgrades it to 3.7.0, which **segfaults** on `import aiter` |
| Kernels | AITER ships **no gfx90a code objects** → 242 of 1,425 translated |
| AITER dispatch | refuses gfx90a in ~16 places |
| vLLM gate | `on_mi3xx()` = gfx942\|gfx950 → widened to `on_gfx9()` for attention only |
| vLLM **priority** | admitted but never selected — `ROCM_ATTN` appended first unconditionally |

That last one is the subtle one. After the gate patch the log read:

```
Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'ROCM_AITER_FA', ...]
```

AITER present, admitted, **unused**. Worth **+8% prefill at 15k rising to +33%
at 101k** once actually selected, proven by the code objects loading rather than
inferred:

> **The original "+12.8%" was one point at 15k on AWQ-Int4.** A later A/B on a
> W8A8 model (`round2.sh` E3, backend and `.co` counts both verified) measured
> **+7.7% at 15k and +33.0% at 101k**. The win scales with context because
> attention is O(n²) and takes a growing share of prefill. Two things follow:
> the ASM benefit is **quantization-independent**, and a single short-prompt
> number materially understates it. `docs/27`.

```
fwd_hd128_bf16_causal_rtna_group.co
fwd_hd128_bf16_rtna_group.co
```

### 3. `--max-model-len` above 128k costs 10× decode on *every* request

Not a bound — a kernel selection. The gfx9 gate is evaluated at CUDA-graph
capture against the *configured* max, so a server set to 256k bakes the Triton
fallback into the graph and replays it for 16k requests too. Full derivation in
`docs/23-vllm-gfx90a-cudagraph-decode-cliff.md`.

---

## Long context: llama.cpp, not vLLM

| Engine | Quant | Context | Decode |
|---|---|---:|---:|
| **llama.cpp** | Q8_0 | **230,407** | **30.4 t/s** |
| vLLM | AWQ-Int4 | 241,583 | 0.7 t/s |

**43×.** llama.cpp at 230k decodes nearly as fast as vLLM W8A8 at 101k (30.4 vs
33.8) — 2.3× the context for 10% less throughput. llama.cpp has no equivalent
of the graph-capture gate, so for long-context serving on this hardware it is
not marginally better, it is the only viable option.

---

## Recommendations

| Workload | Use |
|---|---|
| Coding assistant, ≤128k context | **INT8 W8A8 on vLLM** with the AITER + INT8 patches |
| Long context (128k–256k) | **GGUF Q8_0 or Q4_K_M on llama.cpp** |
| Anything | **Not FP8.** 2.9× slower, no ALU on CDNA2 |
| vLLM config | **Never set `--max-model-len` above 131072** |
| llama.cpp config | `-ub 2048` (4096 and 8192 are both measurably worse) |

---

## What is not settled

### Tier 4 — vLLM's CPU offload is not competitive at 357B

GLM-4.6 AWQ-Int4 (357B) **loads and serves correctly** on 2× MI210 with vLLM:

| | |
|---|---|
| CPU-offloaded parameters | **70.51 GB** |
| Model load | 419 s |
| GPU KV cache | 232,368 tokens (7.09× concurrency at 32k) |
| Correctness probe | **PASS** |

**No throughput number is reported, deliberately.** A single 28k-token warmup
request ran past 35 minutes at a steady 505% container CPU — computing, not
stalled — which projects to roughly two hours for one arm. That was stopped
rather than allowed to block the rest of the queue.

The number that would have come out is not the interesting one anyway. vLLM's
CPU offload is UVA-based: parameters live in host memory and are faulted across
PCIe on demand. llama.cpp's `--n-cpu-moe` instead keeps attention resident and
pages only expert weights, which is the right structure for a sparse MoE. The
GGUF arm on llama.cpp is where this tier gets a usable answer.

So the honest tier-4 finding so far is a **capability** result, not a throughput
one: a 357B model does run on two MI210s with 70 GB in system RAM and produces
correct output. Whether it runs *usefully* depends on the offload mechanism, and
vLLM's is the wrong one here.

**The llama.cpp arm has since completed and settles it.** GLM-4.6 **IQ3_XS**,
correctness PASS:

| | |
|---|---:|
| decode @ short ctx | 12.83 t/s |
| decode @ 25,792 tok | **8.51 t/s** |
| prefill @ 25.8k | **181 t/s** |
| placement | **135.57 GB of 139 GB on GPU** (auto-fit) |

Two conclusions, and the second is the one that matters. First, llama.cpp is
usable at this tier where vLLM was not — 8.5 t/s against a projected two hours
per request. Second, **this is barely an offload result at all**: auto-fit put
all but ~3 GiB on the cards, so a 357B model very nearly fits in 128 GB of VRAM
at IQ3_XS. What remains slow is **prefill, at 181 t/s** — 25× behind the 80B.

**I first attributed that prefill to the ~3 GiB spill, and that was wrong.**
A UD-IQ2_M build 16 GB smaller, fitting comfortably, prefills at **208 t/s
against IQ3_XS's 196** — +6.1%, not the order of magnitude the serialization
story predicted. Both saturate VRAM at ~135 GB because auto-fit spends anything
freed on KV.

The cause is the model: **~32B active parameters over 92 GQA layers**, against
Qwen3-Next's 3B active with 12 of 48 layers holding KV. Roughly 10.7× the active
parameters plus far more attention brackets the gap without invoking offload.
So at tier 4, **quantizing harder to clear the VRAM line does not buy prefill** —
pick the quant for accuracy and KV headroom. Selection guidance in `docs/26`.

- **Tiers 2–4 are still running.** 80B (Qwen3-Next, hybrid GDN), 235B
  (GPTQ-Int4 at reduced context), GLM-4.6 (IQ3_XS with CPU offload).
- **The MoE expert GEMMs are still not tuned.** All 48 AWQ layers fall back
  from Marlin to Moe WNA16 — correctly, since Marlin is CUDA PTX and genuinely
  cannot run on AMD. But vLLM ships tuned `fused_moe` configs for MI300X /
  MI308X / MI325X / MI350X / MI355X / R9700 / A100 and **none for MI210**, so
  the kernel runs on generic heuristics. Tuning is queued.
- **The tier-2 format set differs from tier 1** because no INT8 W8A8 checkpoint
  exists for Qwen3-Next-80B-**Thinking**, not because arms were skipped.
  ~~no INT8 W8A8 checkpoint exists for Qwen3-Next-80B~~ — **corrected**:
  `RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8` does exist (85.73 GB,
  `compressed-tensors` / `int-quantized`, activations 8-bit dynamic per-token).
  It is the Instruct variant, so it was not a drop-in for the Thinking arm, but
  the general claim was too broad. It is now the highest-value untested config
  on this box — see `docs/26`.
- **bf16 baseline is deferred to last.** It needs TP=2 (61 GB will not share a
  64 GB card with a KV cache), and TP=2 costs ~3 hours of *pure CPU* in vLLM's
  per-expert MoE loader — 697 s per shard while the same file reads at 3.0 GB/s.
