# The gfx90a decode cliff: `--max-model-len` bakes a slow kernel into the CUDA graph

**Date**: 2026-07-28 · **Hardware**: AMD Instinct MI210 (gfx90a / CDNA2)
**Software**: ROCm 7.14, vLLM 0.23.1.dev1+g9ddef7117, amd-aiter 0.1.19
**Model**: Qwen3-30B-A3B-Thinking-2507, AWQ-Int4, TP=1

Setting `--max-model-len` above 128k on gfx90a costs **10x decode throughput
for every request**, including short ones that never approach the limit. The
mechanism is not the one it looks like, and the symptom appears with no warning
at request time.

## The measurement

Same model, same card, same harness, prefix caching disabled. The **only**
deliberate difference is `--max-model-len`:

| `--max-model-len` | actual prompt | decode |
|---|---:|---:|
| 131072 | 101,346 tok | **17.2 tok/s** |
| 262144 | 110,000 tok | **1.7 tok/s** |

Context lengths differ by 9%. Decode differs by 10x.

Prefill is **unaffected** — 5.72 s vs 5.67 s TTFT at 15k on the same pair of
configurations. Whatever this is, it is specific to decode.

## What it is not

The obvious suspect is `use_rocm_custom_paged_attention()` in
`vllm/platforms/rocm.py:346-382`, whose gfx9 branch requires

```python
and max_seq_len <= 128 * 1024
```

and which logs, on failure:

```
Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.
```

That looked conclusive and it was **not**: the warning appeared in *neither*
run. A first pass at this stopped there and concluded the ceiling was not
involved.

## What it actually is

The gate is real. It is evaluated at **CUDA graph capture time**, against a
different value than at request time.

`vllm/v1/worker/gpu/model_states/default.py:152-156`:

```python
if for_capture:
    max_seq_len = self.max_model_len
else:
    max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
```

During capture, `max_seq_len` is the **configured** `max_model_len`, not any
real sequence length. So with `--max-model-len 262144`:

1. Capture evaluates the gate with `max_seq_len = 262144`.
2. `262144 > 128*1024`, so the fast custom paged-attention kernel is rejected.
3. The Triton fallback is **baked into the captured graph**.
4. Every subsequent decode replays that graph — with the slow kernel — no
   matter how short the request is.

`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:231-237` has the same pattern
for hybrid models.

This also explains the missing warning. It is emitted once, during capture,
before the server reports ready — not on the request path where it was being
looked for.

The Triton fallback is especially bad here because **aiter ships no gfx90a
Triton configs at all** — only `gfx942-*.json` and `gfx950-*.json` — so it runs
on generic defaults. See `docs/19-aiter-operator-port-matrix.md`.

## Consequences

- **Do not set `--max-model-len` above 131072 on gfx90a** unless you need it.
  It is not a harmless upper bound; it is a decode-path kernel selection.
- A vLLM server configured for 256k serves *16k* requests at 1/10th decode
  speed. Nothing at request time indicates this.
- The 256k workload cannot be served fast by vLLM on this hardware at all.
  llama.cpp has no equivalent gate and carries that case.

Every vLLM arm in `benchmarks/matrix/` is therefore pinned to
`--max-model-len 131072` with long-context prompts at 110k, so the matrix
measures quantization rather than measuring this one fallback repeatedly.

## Confidence, and what is still unverified

| Claim | Confidence | Basis |
|---|---|---|
| `max_seq_len = max_model_len` during capture | **High** | Direct source read, `default.py:152-156` |
| gfx9 gate is `max_seq_len <= 128*1024` | **High** | Direct source read, `rocm.py:366` |
| `cudagraph_capture_sizes` does not depend on `max_model_len` | **High** | `config/vllm.py:1719-1730` derives from `max_num_seqs` |
| `max_num_batched_tokens` does not inflate here | **High** | `arg_utils.py:2647-2654` only fires with chunked prefill off |
| The capture-time warning was actually emitted | **UNVERIFIED** | Implied by the source; not observed |

The last row is the honest gap. The causal chain is read from source, not
demonstrated end to end. The decisive experiment is to force the capture-time
gate to use a runtime-representative length and check whether the cliff
disappears:

```python
# vllm/v1/worker/gpu/model_states/default.py:152-156
max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
# if for_capture:
#     max_seq_len = self.max_model_len
```

Cheaper checks first: capture the server log from the very first line and grep
for the fallback warning before "startup complete"; and bisect `max-model-len`
across 131072 to confirm the cliff is a step at exactly that boundary rather
than a gradient.

## RESOLVED — the ceiling was a switch statement

**Update 2026-07-28.** The "why 128k" question this document leaves open has a
concrete answer, and it is smaller than expected. From `csrc/rocm/attention.cu`:

```c
const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
// reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
// (partition size) = 128K context length
switch (npar_loops) {
  case 1: LAUNCH_CUSTOM_REDUCTION(1); break;
  ...
  case 8: LAUNCH_CUSTOM_REDUCTION(8); break;
  default: TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
}
```

8 × 64 × 256 = 131,072 exactly. **Missing template instantiations, not a
correctness or memory limit.** `configs/extend_rocm_pa_256k_gfx9.py` adds cases
9–16 to the gfx9 launcher and raises that branch's gate to 262,144.

**Numerically verified** — `tests/test_rocm_pa_256k.py`, custom vs Triton, with
an explicit assertion that the gate selected the custom path:

| seq_len | `npar_loops` | stock | patched |
|---:|---:|---|---|
| 131,072 | 8 | PASS `6.62e-03` | PASS `6.62e-03` |
| 139,264 | 9 | gate declined | **PASS `4.15e-03`** |
| 262,144 | 16 | gate declined | **PASS `5.29e-03`** |
| 266,240 | 17 | declined | **still declined** |

Identical control on both builds; exactly the patched lengths fail on stock; the
ceiling still holds at 17. Details in `docs/25` item 2.

**This does not close the row above.** The unverified claim is that the
*capture-time warning was actually emitted* — that the causal chain runs end to
end rather than merely being readable in the source. Raising the gate is
consistent with the diagnosis but does not demonstrate it: if the cliff were
caused by something else that also keys off `max_model_len`, this patch could
still remove it. `benchmarks/matrix/round2.sh` E9 is the measurement, against
the concrete baseline of **0.7485 t/s at 241k** (two reps, 0.7468 / 0.7502).

### Who this patch does and does not help

The same gate filters on head size:

```python
and (head_size == 64 or head_size == 128)
```

So **`head_dim = 256` models never reach the custom kernel at any context
length** — Qwen3-Next-80B decodes on Triton always, patched or not. This is the
second independent place that `head_dim = 256` costs this architecture a fast
path; the first is AITER's ASM attention, which ships only `hd128`/`hd192`
(`docs/25` item 7). Beneficiaries are the `head_dim` 64/128 models: the 30B
Qwen3, Llama-3.3, Gemma-3, Mistral-Small, GLM-4.6.

**And that raises a caveat on the causal story worth stating plainly.** The 80B
decodes at **51.3 t/s at 101k on the Triton path**, so Triton decode is not
inherently catastrophic. The 30B's 0.7485 t/s at 241k differs in two ways at
once — 2.4× the context, and full attention on all 48 layers against the 80B's
12-of-48 GDN hybrid. So "the cliff is the Triton fallback" is the documented
mechanism, not a demonstrated sufficient cause; part of the gap could be context
scaling that would exist regardless of kernel. E9 separates them by changing
only the image.

## Upstream

Worth reporting, and the report is now concrete: extend the gfx9 reduction
dispatch past 8 cases. That is strictly simpler than the three options below,
which were written before the cause was known. Closest existing issue is
[#44014](https://github.com/vllm-project/vllm/issues/44014) — a 10x decode
cliff from cudagraph mode on a hybrid Mamba+MoE model, same symptom and
magnitude, different model class.

Full research trail with citations:
`benchmarks/matrix/research/raw-decode-cliff-opencode.md`.
