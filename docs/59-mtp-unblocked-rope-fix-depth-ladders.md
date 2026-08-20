# MTP unblocked: the rope fix and the depth ladders

2026-08-18 → 08-20. The MTP × quantised-model cells on vLLM hard-failed
under torch.compile with a dynamo fake-tensor error. Two days of
instrumentation images later, the root cause was **one line**: the MTP
drafter passes M-RoPE positions as `(3, N)` — 3 spatial rope dims, N tokens —
and `rotary_embedding/base.py:forward_static` did `positions.flatten()`,
making `num_tokens = 3N` and the query view invalid. On bf16 the trace
resolved by luck; on int8/fp8 legs it crashed; under compile the same bug
masked itself as a fake-tensor symbolic-view error.

**The fix** (upstream as vLLM PR #52973):

```python
if positions.dim() == 2:
    positions = positions[-1]   # text tokens: 3 mrope rows are identical
```

plus a buffer-growth guard in `llm_base_proposer.py` (the dispatcher pads
token counts to capture buckets — 2048→6144 — and the hidden-states slice
silently clamps). Image `local/vllm-mi210:mi210.5-aiter-mtpfix-min` carries
both; **it is the only image that serves every quant × spec combination on
this box**, including the re-derived legs that plain mi210.5 compile-fails.

## The MoE depth ladder (Qwen3.6-35B-A3B, w8a8+AITER)

| n | decode median | band | acceptance |
|---|---:|---|---:|
| 0 | 59.2 | tight | — |
| 3 | **139.8** | [99.3, 143.8] | high |
| 5 | 133.6 | [94.0, 167.5] | — |
| 7 | 131.6 | [89.3, **181.7**] | — |

**n=3 is the median-quality pick**; deeper lowers the median but raises the
ceiling — n7's best run (181.7) is the fastest decode ever measured on this
box. Deep-k only pays on well-drafting (structured/repetitive) traffic.

## The dense depth ladder (Qwen3.8-27B-ablit, w8a8+AITER)

The ladder the dynamo bug ate; filled 2026-08-20, r9 everywhere:

| n | decode median | band | acceptance |
|---|---:|---|---:|
| 1 |  60.9 | [55.6, 64.2] | 0.87 |
| **2** |  **80.0** | [69.6, 84.5] | 0.83 |
| 3 |  75.1 | [61.3, 103.7] | 0.63 |
| **5** | **108.5** | [59.8, 119.6] | 0.59 |
| 7 |  62.6 | [55.4, **129.9**] | 0.35 |

- **n2 is the robust pick**: median +31% over n1 and its *floor* (69.6) sits
  above the n1 *ceiling* (64.2) — every n2 run beat every n1 run.
- **n5 is the upside pick**: highest dense median ever measured here, but
  bimodal (4 of 9 runs in the 60s).
- **n7 is over-drafted**: acceptance collapses to 0.35; median falls back to
  n1 territory despite a record 129.9 single run.
- Mechanism note: the depth optimum sits **deeper on w8a8+AITER than on
  bf16** — a cheaper verification loop makes deep drafts pay more per
  accepted token. Depth preference is a property of target-loop cost, not
  just the drafter.

## The production ladder for qwen38 (dense)

```
18.3  today (w8a8, TP2, AITER off)
  → 42.3  + VLLM_ROCM_USE_AITER=1
  → 80.0  + MTP num_speculative_tokens=2   (robust)
  → 108.5 + n=5                            (bimodal; quote with band)
```

A 4.4–5.9× decode upgrade behind flags on the existing w8a8 checkpoint —
same artifact, same VRAM, prefill unchanged (~3000 tok/s throughout).

## A lineage warning for future quant legs

The fp8-dynamic and W4A16 dense legs were pruned from storage mid-campaign
and re-derived from the published W8A8 recipe
(`sweep/tools/quantize_leg.py`; llmcompressor preset names are
`FP8_DYNAMIC` / `W4A16` — underscore, no dash). The re-derived legs serve
(only on the mtpfix image) but run ~49% below the lost originals — the
recipe keeps GDN/norm/embed in bf16 and the originals' recipes died with
them. Two lessons: **push every derived artifact to the HF account, not just
W8A8/W8A16**; and GPTQ writes `actorder=static` by default where the
known-good W8A8 carries `null` — set it off to match the serving lineage.
