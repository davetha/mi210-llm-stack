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
| **INT8 W8A8** | vLLM + AITER | **29.2 GiB** | **3.20 s** | **4,739 t/s** | **33.8 t/s** |
| Q4_K_M | llama.cpp | ~18 GB | 4.43 s | 3,416 t/s | — |
| Q8_0 | llama.cpp | ~31 GB | 4.57 s | 3,326 t/s | 30.4 @ **230k** |
| AWQ-Int4 | vLLM + AITER | 15.7 GiB | 5.07 s | 3,002 t/s | 17.2 t/s |
| AWQ-Int4 | vLLM stock | 15.7 GiB | 5.72 s | 2,661 t/s | — |
| FP8 | vLLM + AITER | 29.2 GiB | 14.51 s | 1,047 t/s | 9.3 t/s |

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

AITER present, admitted, **unused**. Worth **+12.8% prefill** once actually
selected, proven by the code objects loading rather than inferred:

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

- **Tiers 2–4 are still running.** 80B (Qwen3-Next, hybrid GDN), 235B
  (GPTQ-Int4 at reduced context), GLM-4.6 (IQ3_XS with CPU offload).
- **The MoE expert GEMMs are still not tuned.** All 48 AWQ layers fall back
  from Marlin to Moe WNA16 — correctly, since Marlin is CUDA PTX and genuinely
  cannot run on AMD. But vLLM ships tuned `fused_moe` configs for MI300X /
  MI308X / MI325X / MI350X / MI355X / R9700 / A100 and **none for MI210**, so
  the kernel runs on generic heuristics. Tuning is queued.
- **The tier-2 format set differs from tier 1** because no INT8 W8A8 checkpoint
  exists for Qwen3-Next-80B, not because arms were skipped.
- **bf16 baseline is deferred to last.** It needs TP=2 (61 GB will not share a
  64 GB card with a KV cache), and TP=2 costs ~3 hours of *pure CPU* in vLLM's
  per-expert MoE loader — 697 s per shard while the same file reads at 3.0 GB/s.
