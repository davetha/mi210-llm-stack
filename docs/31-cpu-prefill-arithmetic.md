# CPU-side prefill: the bottleneck is not what the README says, and chunk size runs the wrong way

**Date**: 2026-07-30 · **Hardware**: EPYC 74F3 (Zen3, 24c/48t, AVX2 — no AVX-512,
no VNNI, no bf16), 499 GB DDR4 8-channel · 2× MI210 (gfx90a)
**Config under discussion**: `configs/launch-mimo.sh`

## 1. The stated bottleneck does not match the config

README, on `mimo`:

> prefill is **CPU-dominated** (~7.4 s per chunk reading the growing KV cache back
> from DDR4), even though decode is GPU-bound at ~22 t/s

But `configs/launch-mimo.sh:18` splits by **tensor type**, not by layer:

```
-ngl 999 \
-ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1"
```

Only `ffn.*exps` tensors go to CPU. With `-ngl 999`, **all 48 attention blocks are
on GPU, and the entire KV cache is in HBM** (`-ctk q8_0 -ctv q4_1`).

**So there is no DDR4 KV re-read, and the O(n²) attention runs on the 1.6 TB/s
side.** The placement is right — this is the correct arrangement and it was
already correct a week ago. What is wrong is the *diagnosis* written around it.

This matters because the stated diagnosis points at sparse-attention work, which
by the arithmetic in §4 is aimed at the smaller term at these context lengths.

## 2. What the 7.4 s/chunk actually is

25 layers of Q4_K expert-FFN GEMM over DDR4. With ~104 GB of CPU-resident expert
weights:

**104 GB / 7.4 s ≈ 14 GB/s effective**, against ~150 GB/s achievable on
8-channel DDR4.

**~10× off its bandwidth bound** — structurally the same finding as `docs/25`
item 1c on the GPU side, and worth treating as the same class of problem: a fixed
per-unit-of-work cost that is neither arithmetic nor bandwidth.

## 3. Chunk size runs the opposite way on the CPU path

With `E` experts, `k` active per token and chunk size `c`, the number of distinct
experts touched per chunk is `E·(1 − (1 − k/E)^c)`, which **saturates at `E`** once
`c ≳ E/k`. For MiMo-class routing (~10 of 512), saturation is at **~51 tokens**.

At `-ub 2048` the chunk is 40× past saturation, so **every chunk drags essentially
the entire CPU-resident expert set across DDR4**. Total CPU-side expert traffic is
therefore `(n/c) · E · bytes_per_expert` — **monotonically decreasing in `c`.**

At `-c 65536` with ~104 GB CPU-resident:

| `-ub` | chunks | distinct experts/chunk | total DDR4 expert traffic |
|---:|---:|---:|---:|
| **2048** (current) | 32 | ~512 (saturated) | **~3,330 GB** |
| 4096 | 16 | ~512 | ~1,660 GB |
| **8192** | 8 | ~512 | **~832 GB** |
| 16384 | 4 | ~512 | ~416 GB |
| 32768 | 2 | ~512 | ~208 GB |

**~~`-ub 2048 → 8192` cuts the dominant prefill cost ~4×.~~ REFUTED BY MEASUREMENT —
see the correction below.**

The existing sweep says the opposite — `t35-q4km` prefill went 3,416 → 3,131 →
2,498 tok/s at `-ub` 2048 → 4096 → 8192 — but that arm is **entirely on GPU**,
where the cost structure is inverted (larger micro-batches blow the L2/LDS tiling
and buy nothing, because no weights are being re-read from host).

`round13_cpu_prefill.sh` already says this in its own header:

> `-ub` IS RE-SWEPT ON PURPOSE. The existing `-ub 2048` optimum was measured with
> everything resident on GPU. With 60 expert layers on CPU the bottleneck moves,
> and the best micro-batch for a GPU pipeline is not obviously the best for a CPU
> one — larger batches amortise CPU thread dispatch but blow the L3 tiles.

That is the right instinct with the wrong mechanism attached: the dominant term is
not thread dispatch or L3 tiling, it is **how many times the expert set crosses
DDR4**, and that is `n/c`.

### Thread count, from the literature

CPU-MoE decode is bandwidth-bound on non-AMX x86 — MoE-Lightning
(**2411.11217**, ASPLOS'25), HybriMoE (**2504.05897**), CoX-MoE
(**2605.17889**, DAC'26) all agree. Once bandwidth-saturated, SMT siblings add
contention rather than throughput, so **`-t 24` / `-tb 24` (physical cores) should
beat 48.** That is arm A of round 13 and the reasoning in its header is correct.

### Status: this experiment has never produced a number

There are no `glm-cpu-*` results in `benchmarks/matrix/results/`. Rounds 11, 13,
14 and 17 all died on harness defects — the recent commit trail is
*"Stop waiting on regexes: four deadlocks, one root cause"*, *"A port race was
silently killing arms"*, *"Move the 104 GB fetch outside the bench lock"*,
*"I built a deadlock into the anti-deadlock helper"*, *"The `--n-cpu-moe` decode
sweep never ran; it 400'd on every rep"*.

**The experiment aimed at the machine's actual bottleneck is written and has never
run.** Fixing the runner is currently worth more than any new optimization idea.

## 3b. CORRECTION: the sweep ran, and the traffic model is wrong

Round 13 completed after this was drafted. Results, GLM-4.6 `iq3_xs`,
`--n-cpu-moe 60`:

| arm | prefill tok/s | TTFT (cold16k) | decode (longctx) |
|---|---:|---:|---:|
| `glm-cpu-ub1024` | 120 | 126.76 | 7.25 |
| `glm-cpu-t24tb24` (ub2048) | **165** | **92.77** | 7.13 |
| `glm-cpu-ub4096` | 158 | 96.07 | 7.43 |
| `glm-cpu-t24tb48` (ub2048) | 161 | 94.20 | **7.40** |
| `glm-cpu-t48tb48` (ub2048) | 158 | 95.46 | 5.86 |

**Larger micro-batch does not help.** ub2048 (165) beats ub4096 (158) and
crushes ub1024 (120). The predicted ~4× from `-ub 8192` is **refuted**; the
optimum is at or near the existing 2048.

**Why the model was wrong — and the answer was already in this document's own
"how this could be wrong" section:** the CPU expert path is **compute-bound, not
traffic-bound.** Q4_K dequant on AVX2 with no VNNI and no bf16 dot product costs
enough per byte that the number of passes over the expert set is not the
governing term. `(n/c)·E·bytes` correctly describes the *traffic* and the traffic
is not what is being paid for. The 14 GB/s effective figure should be read as
"the CPU cannot dequant-and-multiply faster than this," not "DDR4 is being
re-read inefficiently."

**Thread count went the predicted way, but marginally:** `t24tb24` (physical
cores) 165 vs `t48tb48` 158 — ~4%, not the meaningful win the
bandwidth-saturation argument implies. Consistent with compute-bound: SMT hurts a
little, but the cores are busy, not stalled. Note decode disagrees with prefill —
`t24tb48` posts the best decode (7.40) and `t48tb48` the worst (5.86), so the
split flags `-t`/`-tb` were worth having separate.

**Scope limit, stated because it is the one thing that could rescue the
prediction:** this ran on GLM-4.6 **iq3_xs** with `--n-cpu-moe 60`, not on mimo
**Q4_K** with the `-ot` regex. I-quants are markedly more expensive to
dequantize on CPU than K-quants, which biases this arm *toward* compute-bound. A
K-quant arm could in principle land differently. But this is the closest
available test and it goes against the prediction, so **treat the chunk-size
inversion as refuted unless a Q4_K arm says otherwise.**

Consequence for §5: since the CPU path is compute-bound, the crossover arithmetic
there (which assumed a DDR4-bandwidth-limited CPU cost of ~2.3 ms/expert) is
**optimistic for the CPU side.** Real CPU cost per expert is higher, so the
crossover sits at *fewer* than 27 tokens/expert/layer and the case for shipping
weights to the GPU is correspondingly stronger than stated.

## 4. Why sparse attention is the wrong target at these context lengths

Attention FLOPs overtake FFN/expert FLOPs only when `4·n·d_model > 2·active_params`.
For ~10B active and `d_model ≈ 6144`:

```
n > 10e9 / (2 · 6144) ≈ 814,000 tokens
```

**Below ~800k, attention is the smaller term** and sparsifying it attacks the
wrong cost. MInference (**2407.02490**, 10× at 1M), FlexPrefill
(**2502.20766**), XAttention (**2503.16428**, 13.5×) and MoA (**2406.14909**) are
real results, but at 64k–256k they are mis-aimed here — and their kernels need
FA2 tiling / `cp.async` / warp-32 reductions anyway. If they ever become relevant,
adopt the *pattern-selection logic* (which runs in torch) and drive a block mask
over the existing CK or ASM bf16 attention; never port their Triton.

## 5. Placement should be an optimisation problem, not a fit order

The 25/48 split was chosen by what fits. The literature has the principled
version, and — unusually — has been evaluated on nearly this hardware.

- **Fiddler** (**2402.07033**, ICLR'25, `efeslab/fiddler`) Algorithm 1 decides per
  expert per step: `if cpu_lat(s) > gpu_lat(s) + trans_lat() then run on GPU`.
- **DALI** (**2602.03495**, *"A Workload-Aware Offloading Framework for Efficient
  MoE Inference on Local PCs"*) solves it as a 0-1 integer program minimising
  `max(T_cpu, T_gpu)`. Its eval box is **EPYC 7532 (Zen2, AVX2-only), PCIe4** —
  the closest published analogue to this host. Reports **7.62× prefill / 3.97×
  decode vs llama.cpp**, and 2.00×/1.32× vs HybriMoE.

**Fiddler's stated justification is dead here, and that is worth knowing before
reading it.** Its CPU kernel is explicitly `AVX512_BF16`; its eval CPUs are
Skylake-SP and Sapphire Rapids (AMX). CoX-MoE puts it bluntly: *"the limited
per-core matrix-multiplication throughput of AVX is insufficient to enable
meaningful compute offloading"* — AVX-512 ≈ 18 TFLOPS bf16/socket against AMX
≈ 144. **Zen3/Milan has no AVX-512, no VNNI, no bf16 dot product.**

**But the inversion survives on bandwidth rather than FLOPs.** One Mixtral-size
expert ≈ 352 MB:

| path | cost |
|---|---:|
| ship weights to GPU over PCIe4 @ ~24 GB/s | **14.6 ms** |
| read in place from DDR4 @ ~150 GB/s | **2.3 ms** |

CPU-execute wins ~6× at batch 1 — because DDR4 beats PCIe, not because AVX2 beats
a GPU. Crossover where shipping becomes worth it:
`14.6 ms ≈ 0.53 ms · N` → **N ≈ 27 tokens per expert per layer.**

**So the split is load-dependent, and the two regimes sit on opposite sides of
it:** likely **under-offloading to CPU at batch-1 decode** and **over-offloading
at `-ub 2048` prefill**. Note this interacts with §3 — raising `-ub` pushes prefill
further past the crossover, which argues for *fewer* CPU-resident experts at large
`-ub`. The two knobs should be swept together, not independently.

**Prediction is better than the older literature suggests**, if prefetch ever gets
built: ProMoE (**2410.22134**) reports **84.7%** expert-prediction accuracy,
degrading only to ~79.7% striding several layers ahead, with adjacent-layer
hidden-state cosine similarity of **90–91.7%**. Fiddler App.C finds expert
popularity flat but **stable across input domains** (mean 0.71, std 0.08), so
offline popularity profiling is a cheap, legitimate input. The older
Mixtral-offloading figure (**2312.17238**, 0.6–0.8 recall) is the low-water mark,
not the state of the art.

**One counterintuitive datum.** MoE-Lightning (**2411.11217**) chooses the
*opposite* arrangement — experts on GPU, **attention on CPU** — because CPU MoE
FFN is memory-bound while its CPU attention kernel beats KV-cache transfer.
Almost certainly wrong for a latency target, but it means "attention on GPU" is a
choice to defend rather than an axiom.

## How this could be wrong

- Larger `-ub` grows the compute buffer and may not fit alongside a ~104 GB model
  at `-c 65536`. The sweep should walk up until allocation fails, not jump.
- If the CPU path is **compute**-bound rather than traffic-bound — plausible, since
  Q4_K dequant on AVX2 has no VNNI to lean on — chunk size will not help and
  thread placement is the lever instead. Both arms already exist in round 13; the
  14 GB/s vs 150 GB/s figure argues for traffic-bound, but it is one data point
  derived from a README sentence rather than a direct measurement.
- DALI is built on KTransformers, which `docs/07` proves cannot serve on this box.
  Port the **placement algorithm** into the `-ot` regex generator; do not attempt
  the engine.
- Prefetch as a transfer-hider is weak here: ~80% recall against two cards
  contending for host PCIe will not reliably hide 14.6 ms.

## Confidence

| Claim | Confidence | Basis |
|---|---|---|
| Attention + KV are on GPU, not DDR4 | **High** | `launch-mimo.sh:18` + `-ngl 999` |
| Expert traffic scales as `(n/c)·E·bytes` | **High** | saturation arithmetic, `c ≫ E/k` |
| `-ub 8192` cuts CPU prefill traffic ~4× | **Medium-High** | arithmetic; unmeasured on this box |
| 14 GB/s effective on the CPU path | **Medium** | derived from the README's 7.4 s figure |
| CPU-execute beats PCIe-ship at batch 1 | **Medium-High** | bandwidth arithmetic, both rates measured elsewhere |
| Crossover ≈ 27 tokens/expert/layer | **Medium** | Fiddler's model, re-derived for AVX2 |
| Sparse attention mis-aimed below ~800k | **High** | FLOP crossover arithmetic |
