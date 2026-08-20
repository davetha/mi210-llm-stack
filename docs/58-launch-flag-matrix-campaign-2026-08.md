# The launch-flag matrix campaign (2026-08): both models, both stacks

The campaign of 2026-08-16 → 08-20. Question: for the two production models —
**Qwen3.8-27B-ablit (dense)** and **Qwen3.6-35B-A3B (MoE)** — what does each
stack (vLLM 0.27.2rc0-mi210.5 lineage, llama.cpp fork `a496e3358`) actually
deliver per launch flag, and which knobs matter?

Raw records: `big:/home/dave/sweep/results/{vllm,llama}.jsonl` (227 records,
gate-v2 correctness on every cell). Full analysis: `big:/home/dave/sweep/
LAUNCH-FLAG-MATRIX-{vllm,llama}.md`. This page is the summary that survives.

## Headline numbers (2× MI210, single-stream decode / prefill tok/s)

| model × stack | decode | prefill 512/2048/8192 |
|---|---:|---|
| dense 27B w8a8, vLLM TP2 (production today) | 18.3 | 2375 / 2831 / 2687 |
| dense 27B w8a8 + **AITER** (int8 GEMM) | 42.3 | 2833 / 3237 / 3031 |
| dense 27B w8a8 + AITER + **MTP n2** | **80.0** | ~2700 / 3100 / 2950 |
| dense 27B w8a8 + AITER + MTP n5 | 108.5 ⚠ bimodal | same |
| MoE 35B w8a8 + **MTP n3** + AITER | **139.8** | 6835 / 10745 / 9016 |
| MoE 35B, llama.cpp IQ4_NL (non-spec) | 108 | 3192 / 4573 / 6129 |
| dense 27B GGUF Q4_K_M, llama.cpp | 37.9 | ~650 pp512 |

⚠ = scatter-flagged; quote with band ([59](59-mtp-unblocked-rope-fix-depth-ladders.md) has the ladders).

## The four findings that change how older results read

1. **The DIVERGE base rate is zero.** base-bf16 ran three times with an
   identical cell_id — all three pairwise MATCH. vLLM on this box is
   launch-deterministic for a fixed config, so any cross-config divergence is
   signal, not noise.
2. **Tensor-parallelism is itself a numerics parameter.** TP1 vs TP2 at
   identical `max_num_seqs` diverges on all six probes. The earlier "explained
   by batch width" closure was wrong.
3. **The compile/JIT cache policy is a numerics parameter.** Identical spec,
   shared vs per-cell cache dirs → outputs differ on all six probes. Stale
   AITER-cache poisoning is possible, not hypothetical (n=1 pair, disclosed).
   Per-cell isolation (the harness default) is load-bearing.
4. **MTP changes output at every depth including n=1.** Offline re-keying
   (`sweep/tools/rekey_gate.py`) shows the fixed ladder diverges from
   speculation-off on the *same six probes* at n1/n3/n5/n7; adaptive adds a
   seventh. Not a depth artifact — speculation itself moves text.

## Quant picks (unchanged by the campaign, now measured)

- **MoE**: IQ4_NL (best decode) or Q4_K_M; Q6_K_P dominates nothing.
- **Dense**: Q4_K_M for speed, Q5_K_M the quality step. **Q6_K's
  condemnation was a probe defect** (a trailing space in the raw-completion
  prompt flipped 2 of 3 quants); re-adjudicated post-fix: gate MATCH.
  One space in a probe prompt can condemn a quant — always suspect the probe
  first.

## llama.cpp specifics that transfer

- **MoE prefill is structurally ~2× behind vLLM** — threads flat, `-ub` mixed,
  fa/GDN-chunk/MMQ already optimal: no host-side lever exists. vLLM is the
  prefill stack for that model.
- **Split-mode census on ROCm**: layer works; **row is impossible on any ROCm
  build** (`llama-model.cpp:991` — the HIP backend has never had split
  buffers; issue #25594 is the CUDA-side cousin, not this); **tensor
  hard-faults** with the same Tensile signature as the §8 offload family.
- **Adaptive MTP (PR 27210) as shipped loses at short generations**; with
  `--spec-draft-n-max 6` it reaches fixed-n7 parity (66.1 median, acceptance
  0.57→0.73).

## Suffix decoding: a reproduced correctness blocker

vLLM 0.27.2rc0-mi210.5 + arctic-inference 0.2.0, `--speculative-config
{"method":"suffix"}`: **2/2 runs serve a premature stop on a deterministic
count at temperature 0** (`1 2 3 … 10 1 2`, 25/48 tokens, identical truncation
both runs). Lossless speculation must reproduce greedy exactly — this alters
the served distribution. Do not serve. Drafting itself works (63% acceptance);
the failure is in the accept/stop path. Clean repro signature; candidate
upstream issue.

## Upstream filings from this campaign

vLLM: #52973 (M-RoPE 2-D positions fix — the MTP unblock), #52974 (NVFP4
W4A16 GEMM), #52975 (W4A16 tile tuning), #52983 (magic-bias dequant),
#52984 (narrow-tile rung M≤16), #52985 (fused W8A16 fp8 GEMM).
llama.cpp: #27405 (pin host state buffer), #27406/#27410/#27411 (IQ4_NL /
IQ4_XS / MXFP4 MMQ clauses), #27423 (adaptive ubatch, gated on pipeline
parallel).
