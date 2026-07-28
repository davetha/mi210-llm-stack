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
