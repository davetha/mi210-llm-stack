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

## Upstream

Worth reporting. Either raise the gfx9 threshold, or evaluate the gate during
capture against a representative runtime length, or make the captured graph's
attention dispatch metadata-driven rather than baked. Closest existing issue is
[#44014](https://github.com/vllm-project/vllm/issues/44014) — a 10x decode
cliff from cudagraph mode on a hybrid Mamba+MoE model, same symptom and
magnitude, different model class.

Full research trail with citations:
`benchmarks/matrix/research/raw-decode-cliff-opencode.md`.
