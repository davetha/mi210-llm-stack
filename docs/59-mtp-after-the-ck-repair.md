# Native MTP loses on the 27B W8A8 at 88.6% acceptance — verification is free, drafting is not

`docs/39` left MTP and n-gram as **re-runnable once the decode gap closes**, on
the reasoning that speculation lost here because verification could not be cheap
while target decode sat ~3× off its bandwidth bound. `docs/57` closed that gap on
one arm: the repaired CK W8A8 path moved Qwen3.6-27B decode by 2.9–3.5×.

So the re-run was run. Measured 2026-08-09 on one MI210, vLLM
`0.26.1rc0+mi210.1`, image `local/vllm-mi210:dsa7-aiterint8`, checkpoint
`Avesed/Qwen3.6-27B-INT8-W8A8` — the same server, flags, mounts and device
config as `docs/57`, with **only `--speculative-config` differing between arms**.

**The precondition was met and the conclusion still inverted.** Verification
turned out to be as cheap as predicted. Drafting turned out to cost more than a
full forward pass of the model it is drafting for, which makes the line
unwinnable at *any* acceptance rate rather than merely unprofitable.

## The result

`--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":N}'`.
3 reps, salted prefixes, `build/bench_llamabench_style.py`.

| tg256 | baseline | N=1 | N=2 | N=3 |
|---|---:|---:|---:|---:|
| d0 | **36.13** ± 0.00 | 30.27 ± 0.12 | 27.04 ± 0.25 | 24.53 ± 0.19 |
| d8192 | **34.42** ± 0.27 | 28.37 ± 0.00 | 24.72 ± 0.02 | 20.27 ± 2.41 |
| d32768 | **29.62** ± 0.16 | 24.09 ± 0.01 | 19.66 ± 0.01 | 16.52 ± 0.01 |
| ratio @ d0 | 1.000 | **0.838** | **0.748** | **0.679** |
| ratio @ d32768 | 1.000 | **0.813** | **0.664** | **0.558** |

The baseline reproduces `docs/57` (34.41 ± 1.07 there, 36.13 here; pp2048 @
d32768 1058.87 vs 1055.51), and every arm logged
`Selected AiterInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`, so no arm
silently fell back to Triton.

**The drafting was excellent.** This is not an acceptance failure:

| N | drafts | accepted | mean accepted | per-position acceptance |
|---:|---:|---:|---:|---|
| 1 | 1,224 | 1,084 | 0.886 | 88.6% |
| 2 | 893 | 1,412 | 1.581 | 86.4% / 71.7% |
| 3 | 749 | 1,571 | 2.098 | 87.0% / 68.2% / 54.5% |

88.6% at position 0 sits exactly in the 84.8–89.1% band `docs/25` measured for
native MTP on the 80B. The head is good. The head is not the problem.

## Where it goes

Convert throughput into cost per engine step. With mean accepted length `L`
tokens per step and observed ratio `r`, the step costs `L / r` baseline steps.
Fitting `cost(N) = a + bN` at d0 over N = 1, 2, 3:

```
step cost:  N=1 2.251   N=2 3.449   N=3 4.562
fit:        a = 1.11    b = 1.16
```

- **`a = 1.11` — verification is free, as predicted.** Running the target at
  M = 2…4 instead of M = 1 costs 11% more. That is exactly what `docs/57`'s
  isolated GEMM shape said it would: M=1 30.60 µs vs M=8 33.64 µs, 8× the
  arithmetic for 10% more time. The bandwidth-roofline argument for speculation
  was correct.
- **`b = 1.16` — each drafted token costs 1.16 full target decode steps.** One
  MTP layer plus the shared `lm_head` costs *more than all 64 layers of the
  model it drafts for*.

That second number closes the line by itself. Break-even needs
`1 + N > a + bN`, i.e. `b < 1 − (a−1)/N`. At d0 that is `b < 0.89`, and even the
degenerate ceiling of 100% acceptance cannot help while `b > 1`: **every extra
drafted token costs more than the token it might save.** No acceptance rate
reachable by any draft head fixes a `b` above 1.

## Why one layer costs more than sixty-four

Bytes moved per step, from the checkpoint's safetensors header:

| | bytes/token | time | achieved |
|---|---:|---:|---|
| target forward (64 layers, W8A8) | 26.93 GB | 27.68 ms | **973 GB/s — 83% of 1170** |
| MTP draft forward (1 layer + shared bf16 `lm_head`) | 3.29 GB | 32.11 ms | **102 GB/s — 9% of 1170** |

The draft reads **12.2%** of the target's bytes and takes **16% longer**. It runs
at **one tenth** of the target's achieved bandwidth.

102 GB/s is not a new number for this repo. Round 62 in
[`docs/52`](52-data-parallel-plus-asm-attention.md) measured decode at
**113 GB/s of 1170 (10%)** and diagnosed it as latency/occupancy-bound — not
enough concurrent work in flight. The MTP head is that diagnosis in its purest
form: a one-layer forward is too little work to fill a 104-CU card, so it pays
fixed per-forward cost (launch, sampling, hidden-state plumbing, its own graph)
against almost no arithmetic. The 64-layer target reaches 83% of the bound for
the opposite reason — it is deep enough to amortize the same fixed cost sixty-four
times over.

So speculation on this box asks you to pay a **latency-bound** forward in order
to save **bandwidth-bound** work. That trade is inverted here, and it is inverted
*more* the better the target gets: `docs/57` improving the target by 2.9–3.5×
made the draft head's fixed cost a *larger* fraction of the step, not a smaller
one. Closing the decode gap did not enable speculation. It made speculation
worse.

`lm_head` is worth calling out separately: it is in the checkpoint's quantization
`ignore` list, so it stays **bf16 at 2.543 GB** (vocab 248,320 × 5,120), and vLLM
logs `Detected MTP model. Sharing target model lm_head weights with the draft
model.` That single tensor is 77% of the draft's byte traffic. A quantized or
truncated-vocabulary draft head is the one lever that would move `b` materially.

## A second, separate defect: prefill regresses and does not scale with N

| pp2048 | baseline | N=1 | N=2 | N=3 |
|---|---:|---:|---:|---:|
| d8192 | **1467.09** | 1134.14 | 1124.78 | 1118.24 |
| d32768 | **1058.87** | 797.89 | 798.15 | 781.90 |

Prefill loses **23–25%**, and the loss is **flat across N**. Speculation does no
drafting during prefill, so this is not draft cost — it is a fixed structural
consequence of turning the feature on. The likely mechanism, unverified: this
model is **48 of 64 layers `linear_attention`** (GatedDeltaNet), and
`vllm/v1/attention/backends/gdn_attn.py:112` calls
`_init_reorder_batch_threshold(1, self.use_spec_decode)` — enabling speculation
changes the batch reordering threshold for every GDN layer, whether or not a
draft is in flight.

That is worth chasing on its own account, independent of speculation: a 23%
prefill regression from a decode-side feature flag is a defect, not a tradeoff.
It is **not** chased here because the `b > 1` result already closes the
speculation line and the prefill path does not depend on it.

## Verdict

**Closed. Do not re-run native MTP or n-gram on this hardware without first
changing the draft head's cost.** The `docs/39` re-run condition — "after the
decode gap closes" — has now been tested and was the wrong condition. The right
condition is:

> a draft forward must cost less than `1 − (a−1)/N` target forwards.
> Measured: 1.16. Needed: < 0.89 at N=1. That is a **1.3× reduction to break
> even** and ~1.5× to be interesting, and it must come from the draft's fixed
> cost, not its bytes.

Things that would legitimately reopen it, in order of plausibility:

1. **A cheaper draft head.** Quantize the shared `lm_head` (77% of draft bytes)
   or draft over a truncated vocabulary. This attacks bytes, and bytes are only
   12% of the problem — so it is necessary but probably not sufficient.
2. **Amortizing the draft's fixed cost.** Anything that makes a one-layer
   forward stop being latency-bound: fusing the draft into the target's graph,
   or drafting for many sequences at once so the head runs at useful batch. Note
   the connection to [`docs/58`](58-cu-masking-on-gfx90a.md) — the draft head is
   exactly a workload that cannot fill this GPU.
3. **A model whose MTP head is a larger fraction of the model.** `b` is a ratio;
   a shallower target makes it worse, a deeper one better.

Not reopened by: a better acceptance rate (already 88.6%), a bigger `N` (worse
at every step), or further target-side optimization (which makes `b` worse).

## Reproduce

```bash
./mtp_serve.sh base          # docs/57 baseline, unchanged
./mtp_serve.sh mtp 1         # + {"method":"qwen3_5_mtp","num_speculative_tokens":1}
./mtp_ab.sh                  # all four arms, 3 depths, 3 reps, with the gates
```

[`benchmarks/matrix/mtp_serve.sh`](../benchmarks/matrix/mtp_serve.sh) and
[`benchmarks/matrix/mtp_ab.sh`](../benchmarks/matrix/mtp_ab.sh). Both assert the
CK kernel-selection line and print the resolved speculative method before
benching; `docs/55` records four separate incidents of measuring the wrong arm,
and this harness refuses to be the fifth.

Two notes for anyone repeating this:

- vLLM warns `method 'qwen3_5_mtp' is deprecated and replaced with mtp`, then
  resolves `SpeculativeConfig(method='mtp', num_spec_tokens=N)` and
  `Resolved architecture: Qwen3_5MTP`. That is the correct path, not a fallback.
- `mtp_num_hidden_layers = 1`, so `n_predict = 1` and every `N > 1` re-runs the
  *same* MTP layer. vLLM says so: *"Enabling num_speculative_tokens > 1 will run
  multiple times of forward on same MTP layer, which may result in lower
  acceptance rate."* The per-position acceptance above (87.0 / 68.2 / 54.5) is
  that warning being correct.
