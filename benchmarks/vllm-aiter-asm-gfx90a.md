# AITER ASM attention under vLLM on gfx90a — measured

**Date**: 2026-07-27 · **Hardware**: AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e, 104 CU)
**Software**: ROCm 7.14.0, PyTorch 2.11, amd-aiter 0.1.17, vLLM 0.23.1.dev1+g9ddef7117
**Model**: Qwen/Qwen3-14B, bf16, 40 layers, 40 Q heads / 8 KV heads, head_dim 128
**Reproduce**: `configs/enable_vllm_aiter_gfx90a.py` then `benchmarks/bench_vllm_serving.py`

`asm-attention-gfx90a.md` measured AITER's hand-written ASM attention kernels in
isolation and found them 1.86x faster than PyTorch SDPA at prefill and 1.72x
faster than the HIP kernel at decode. It closed by naming the obvious open
question: *whether vLLM actually reaches these kernels is a separate question.*

This is the answer to that question. Two parts, and the first matters more than
the second.

---

**The short answer: yes, but only on long prompts under concurrency — 1.23x
output throughput at 4096-token prompts, and nothing at all on short ones.**
And the credit does not go where the microbenchmarks suggested: the ASM *decode*
kernel — the one measured at 1.72x in isolation — is worth about 1% here, which
is inside this harness's run-to-run noise. Essentially all of the gain comes
from the prefill side.

| | 128-token prompts | 4096-token prompts |
|---|---|---|
| concurrency 1 | 1.02x | 1.01x |
| concurrency 8 | 1.00x | **1.23x** |
| concurrency 32 | 1.02x | **1.23x** |

---

## Part 1 — vLLM could not reach AITER on gfx90a at all

Before any measurement was possible, a gate had to be found and removed. It is
worth describing precisely, because it fails silently and every log line it
produces looks healthy.

`vllm/_aiter_ops.py` documents its own dispatch rule in prose:

> All check functions (`is_*_enabled`) are decorated with `@if_aiter_supported`,
> which verifies: (1) platform is ROCm, (2) device arch is gfx9, and (3) aiter
> library is installed.

gfx90a **is** gfx9, so by that description an MI210 qualifies. The code
implementing the check does something else:

```python
def is_aiter_found_and_supported() -> bool:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx
        return on_mi3xx()          # _ON_MI3XX = gfx942 or gfx950
    return False
```

`on_mi3xx()` is MI300-series only. On an MI210 it is False, so
`if_aiter_supported` short-circuits — and because it returns `None` rather than
`False`, every `is_*_enabled()` query answers falsy. vLLM then drops
`ROCM_AITER_FA` and `ROCM_AITER_UNIFIED_ATTN` from the backend candidate list
and logs, entirely calmly:

```
Found incompatible backend(s) [TURBOQUANT] with AttentionType.DECODER.
Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
```

Nothing in that line says AITER was considered and rejected. Setting
`VLLM_ROCM_USE_AITER=1` changes it not at all. A second, independent
`on_mi3xx()` in `AiterFlashAttentionBackend.supports_compute_capability()`
rejects the backend again even if the first gate is widened.

`configs/enable_vllm_aiter_gfx90a.py` changes both to `on_gfx9()` — the
condition the surrounding documentation already claims. It deliberately leaves
the other two `on_mi3xx()` sites alone, since those guard FP8 scaled-GEMM paths
that CDNA2 genuinely cannot execute.

**This corrects a premise this work started from.** The expectation was that
vLLM's AITER gates were already `is_on_gfx9()` and no vLLM patching would be
needed. That string does appear at `_aiter_ops.py:1315` — but inside the
docstring quoted above, not in executable code.

### Selecting the backend is a second, separate step

Widening the gate makes `ROCM_AITER_FA` *selectable*, not *selected*. vLLM's
ROCm priority list puts `ROCM_ATTN` first, so the default still avoids AITER:

```
Overriding with ROCM_ATTN out of potential backends:
    ['ROCM_ATTN', 'ROCM_AITER_FA', 'ROCM_AITER_UNIFIED_ATTN', 'TRITON_ATTN']
```

`ROCM_ATTN` reaches no AITER code at all — its prefill is a Triton kernel and
its decode is vLLM's own paged attention. Two further points cost real time
here and are worth recording:

- **`VLLM_ATTENTION_BACKEND` is not a recognised variable in vLLM 0.23.x.**
  Setting it produces `WARNING: Unknown vLLM environment variable detected` and
  is otherwise ignored. Backend choice goes through
  `--attention-config '{"backend":"ROCM_AITER_FA"}'`.
- **The ATOM plugin registers a vLLM *platform* plugin**, which replaces
  `RocmPlatform` wholesale and therefore replaces backend selection. Every run
  here sets `VLLM_PLUGINS=` so that dispatch is vLLM's own.

### Which kernel each path actually reaches

Only `ROCM_AITER_FA` touches the ASM kernels, and only its prefill does so
unconditionally:

| path | prefill | decode |
|---|---|---|
| `ROCM_ATTN` (vLLM default) | Triton | vLLM paged attention |
| `ROCM_AITER_FA` | **`fmha_v3_fwd` ASM** | `torch.ops.aiter.paged_attention_v1` (HIP) |
| `ROCM_AITER_FA` + `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` | **`fmha_v3_fwd` ASM** | **`pa_fwd_asm` ASM** |

The ASM decode kernel needs the shuffled KV layout, which is a separate env var
from any `VLLM_ROCM_USE_AITER_*` flag. `VLLM_ROCM_USE_AITER_PAGED_ATTN` does
not select it.

### Proof the ASM kernels load

With `AITER_LOG_LEVEL=info`, from the running server's log (the EngineCore
subprocess inherits the server's stdout, so a plain redirect captures it):

```
[aiter] LoadKernel: _ZN5aiter37fmha_fwd_hd128_bf16_causal_rtna_groupE
    hsaco: .../aiter_meta/hsa//gfx90a/fmha_v3_fwd/MI300/fwd_hd128_bf16_causal_rtna_group.co
[aiter] LoadKernel: _ZN5aiter27pa_bf16_noquant_gqa8_1tg_4wE
    hsaco: .../aiter_meta/hsa//gfx90a/pa/pa_bf16_noquant_gqa8_1tg_4w.co
```

Both paths are `gfx90a/`, i.e. the ported code objects from
`repatch_gfx942_to_gfx90a.py`, not a gfx942 binary loaded by accident.

Note what this does and does not prove. It proves the kernels load and are
reached. It does **not** prove every decode step uses `pa_fwd_asm` — see the
batch-size heuristic below, which is the single most important fact for
interpreting the numbers.

### The ASM decode kernel is gated on batch size, inside AITER

`aiter/ops/attention.py:_should_use_asm_kernel` refuses the ASM path unless

```python
total_heads = num_seqs * num_heads > 2 * cu_num
```

On a 104-CU MI210 serving Qwen3-14B (40 Q heads), that is
`num_seqs > 5.2`, so **the ASM decode kernel is not used below 6 concurrent
requests** no matter how it is configured. This is not a gfx90a quirk; it is
AITER's own heuristic, and it agrees with the microbenchmarks, which found the
HIP and ASM decode kernels equal until the kernel becomes bandwidth-bound.

Any concurrency-1 decode number below is therefore a measurement of the HIP
kernel in both configurations, and should show no difference by construction.

---

## Part 2 — what the kernels are worth end to end

Qwen3-14B bf16 on one MI210, 128 output tokens per request, `ignore_eos`, no
prefix caching. Every request in every cell below succeeded; there are no
dropped requests hiding in these averages.

| prompt | conc | stock ttft | aiter-fa ttft | asm ttft | stock tpot | aiter-fa tpot | asm tpot | stock tok/s | aiter-fa tok/s | **asm tok/s** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 66.3 | 66.1 | 65.9 | 25.29 | 25.01 | 24.89 | 39.1 | 39.5 | **39.7** |
| 128 | 8 | 299.5 | 333.9 | 323.0 | 38.40 | 38.22 | 38.14 | 197.6 | 197.2 | **198.1** |
| 128 | 32 | 959.7 | 1001.5 | 1039.1 | 45.36 | 44.80 | 43.80 | 608.1 | 610.8 | **619.3** |
| 4096 | 1 | 1407.3 | 1362.1 | 1374.8 | 26.23 | 26.02 | 25.95 | 27.0 | 27.4 | **27.4** |
| 4096 | 8 | 5109.4 | 4392.2 | 4469.0 | 117.97 | 95.89 | 93.31 | 50.7 | 61.5 | **62.4** |
| 4096 | 32 | 12501.2 | 10556.3 | 10716.2 | 394.16 | 321.50 | 315.27 | 64.9 | 79.0 | **79.9** |

TTFT and TPOT in ms, mean. Throughput is output tokens/s across the whole case.

### Long prompts under load: 1.23x

The only cells that move are 4096-token prompts at concurrency 8 and 32, and
they move together: **1.23x** output throughput, **1.25x** better TPOT
(394 → 315 ms at concurrency 32), and **14%** lower TTFT (12.5 → 10.7 s).

### Short prompts: nothing

At 128-token prompts the three configurations are within 2% of each other at
every concurrency — which is within run-to-run noise for this harness. A
128-token prefill is too small to occupy an MI210, so a faster attention kernel
has nothing to speed up, and decode at these batch sizes is bound by streaming
28 GB of weights rather than by the attention kernel.

### Concurrency 1: nothing, at either prompt length

Single-stream serving sees no benefit at all (1.01–1.02x). This is expected and
was predictable from the gating described above:

- Decode cannot use `pa_fwd_asm` at all below 6 concurrent requests, because
  `num_seqs * 40 heads` fails AITER's `> 2 * 104 CU` test.
- Prefill *does* use the ASM kernel, but at concurrency 1 it buys only 2.3%
  (TTFT 1407 → 1375 ms at 4096 tokens). The ASM prefill kernel's advantage
  comes from packing several variable-length sequences into one launch; with
  one sequence there is little to pack.

**The consistent story across all six cells is that these kernels pay off only
when the GPU has enough parallel work to expose their advantage.** That is the
same conclusion the microbenchmarks reached — they found the decode kernels
identical below ~700 GB/s and 1.7x apart once bandwidth-bound — now confirmed
at the serving layer.

### Where the gain actually comes from — not the decode kernel

`aiter-fa` and `aiter-fa-asm` differ in exactly one kernel, so the difference
between those two columns is the ASM decode kernel's entire end-to-end
contribution:

| prompt | conc | ASM decode kernel worth |
|---|---:|---:|
| 128 | 32 | +1.4% |
| 4096 | 8 | +1.5% |
| 4096 | 32 | +1.1% |

**About 1% — and a repeat of the same configuration varies by up to 1.0%
(see below), so this is at the edge of measurement noise and cannot be
distinguished from zero.**

The 1.72x decode speedup measured in isolation almost entirely disappears into
everything else a decode step does — the QKV and MLP GEMMs, RMSNorm, sampling,
and the scheduler — none of which the kernel touches. This is the clearest
result here, and it is a caution about reading microbenchmarks as forecasts: a
1.7x kernel was worth ~1% of the workload, if anything.

The direction is at least consistent — `aiter-fa-asm` is ahead in all three
cells where the ASM decode kernel can engage — but proving a 1% effect would
need repeated trials this run does not have.

The remaining ~22% comes from the switch to `ROCM_AITER_FA` itself, which
changes prefill from Triton to the ASM `fmha_v3_fwd` **and** changes decode
from vLLM's paged attention to AITER's HIP `paged_attention_v1`. This
experiment does not separate those two, because isolating the ASM prefill would
mean reverting the aiter ASM enablement and re-running — see the limitations
below.

### Correctness

Every configuration was checked before its throughput was quoted. All three
answer a short factual prompt identically, and all three recover a fact planted
at the start of a ~2,600-token prompt (`RECALL_OK: True`), which a broken
long-context attention kernel would lose.

Outputs are **not** bit-identical across backends — different attention kernels
give slightly different bf16 rounding, and under greedy decoding that
eventually diverges the continuation. On the long probe all three state the
planted code correctly and then continue differently. That is expected, not a
defect; what would be a defect is a wrong or degenerate answer, and there is
none.

### llama.cpp, as a loose reference only

`llama-coder-80b` on the second MI210 — **a different model, so this is context,
not a comparison**: Qwen3-Coder-Next-80B MoE at Q4_K_M with q8_0 KV cache and 8
parallel slots, versus a 14B dense model at bf16 with bf16 KV.

| prompt | conc | llama.cpp 80B-Q4 tok/s | vLLM 14B-bf16 (asm) tok/s |
|---|---:|---:|---:|
| 128 | 1 | 68.3 | 39.7 |
| 128 | 32 | 150.6 | 619.3 |
| 4096 | 1 | 31.4 | 27.4 |
| 4096 | 32 | 44.9 | 79.9 |

The single-stream advantage is llama.cpp's 4-bit weights (an 80B MoE with ~3B
active parameters at Q4 moves far less memory per token than 28 GB of bf16).
The throughput advantage under load is vLLM's continuous batching against
llama.cpp's 8 fixed slots — at concurrency 32 llama.cpp queues, and its TTFT
degrades to 17.9 s (128-token prompts) and 61.9 s (4096-token), against 1.0 s
and 10.7 s for vLLM. None of this says anything about the ASM kernels.

---

## Method

Three configurations, differing only in attention:

| label | flags |
|---|---|
| `stock` | `VLLM_ROCM_USE_AITER=0` → `ROCM_ATTN` |
| `aiter-fa` | AITER on, `ROCM_AITER_FA`, shuffle layout off → ASM prefill, HIP decode |
| `aiter-fa-asm` | AITER on, `ROCM_AITER_FA`, shuffle layout on → ASM prefill, ASM decode |

The AITER master switch also enables AITER's linear, MoE, RMSNorm and FP8-BMM
ops. All of those are pinned **off** in both `aiter-*` configs, so the only
difference between `stock` and `aiter-fa-asm` is the attention implementation.
Left on, the comparison would have measured "AITER vs not" instead of "ASM
attention vs not" — and AITER's RMSNorm in particular is unvalidated on gfx90a.

`aiter-fa` vs `aiter-fa-asm` is the cleaner of the two contrasts: those two
differ in exactly one kernel, the decode one.

Every server ran with `--no-enable-prefix-caching`, `--max-model-len 8192`,
`--gpu-memory-utilization 0.85`, `--max-num-seqs 64`, on one MI210 with the
second card left running its usual workload. Prompts are built to an exact
token count and are unique per request, so prefill is never a cache hit. Output
length is pinned with `ignore_eos`, so decode work is identical across
configurations. TTFT is the first streamed token; TPOT is the mean inter-token
gap after it.

Exact invocation:

```bash
# ASM prefill + ASM decode
VLLM_PLUGINS= HIP_VISIBLE_DEVICES=0 \
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 \
VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0 \
VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0 \
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 AITER_LOG_LEVEL=info \
vllm serve Qwen/Qwen3-14B --port 8000 --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --max-num-seqs 64 \
    --attention-config '{"backend":"ROCM_AITER_FA"}'

python benchmarks/bench_vllm_serving.py --label aiter-fa-asm \
    --out results-aiter-fa-asm.json --model Qwen/Qwen3-14B
```

### Two operational notes that cost time

**`vllm serve` leaves an orphan.** Killing the parent leaves a `VLLM::EngineCore`
process whose cmdline does not match `vllm serve`; it holds ~58 GB of VRAM
indefinitely. The next server then dies with `Free memory on device cuda:0
(8.19/63.98 GiB) ... less than desired GPU memory utilization`, which reads like
a configuration problem and is not. Tear down by matching `VLLM::` as well, and
wait on `rocm-smi` actually reporting the memory back rather than on a fixed
sleep.

**HIP device order is the reverse of `rocm-smi`'s.** `HIP_VISIBLE_DEVICES=0` is
the card `rocm-smi` lists as `GPU[1]`. Picking the wrong one lands on the busy
card and fails with the same misleading free-memory error.

---

## What this does not measure

- **ASM prefill in isolation.** The 22-point remainder is the `ROCM_AITER_FA`
  backend as a whole: ASM prefill *and* AITER's HIP paged decode, versus
  Triton prefill and vLLM's paged decode. Separating them would need a run with
  the aiter ASM enablement reverted so `fmha_v3_fwd` falls back to CK, which
  was out of scope here.
- **Prompts beyond 4096 tokens.** `--max-model-len` was 8192. The trend from
  128 to 4096 is strongly increasing, so the 1.23x is likely a floor rather
  than a ceiling for long-context serving.
- **Anything but bf16 dense.** FP8, INT8 and MoE paths are untouched; CDNA2 has
  no FP8 ALU and the GEMM tuning configs have no gfx90a rows (see
  `asm-attention-gfx90a.md`).
- **Multi-GPU.** Single MI210. The pair has no xGMI bridge.
- **Variance.** Each cell is one run of 8–96 requests. `aiter-fa-asm` was
  re-run end to end as a reproducibility check and agreed to within **1.0%**
  at the noisiest cell:

  | prompt | conc | first run | repeat |
  |---|---:|---:|---:|
  | 128 | 1 | 39.7 | 39.7 |
  | 128 | 8 | 198.1 | 199.5 |
  | 128 | 32 | 619.3 | 625.4 |
  | 4096 | 1 | 27.4 | 27.3 |
  | 4096 | 8 | 62.4 | 62.5 |
  | 4096 | 32 | 79.9 | 80.1 |

  So the 1.23x at long context is far outside noise and is real; every
  sub-2% difference in this document is not, and should be read as "no
  measurable difference".
