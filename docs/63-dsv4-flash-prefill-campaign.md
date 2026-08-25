# DeepSeek-V4-Flash prefill on 2× MI210: +49% cumulative, and where the ceiling actually is

**Date:** 2026-08-25 · **Model:** DeepSeek-V4-Flash-0731-Abliterated (DS4-Quality128 GGUF, 284B total / 13B active MoE, 43 layers, head_dim 512, MLA/MQA) · **Stack:** llama.cpp fork [`davetha/llama.cpp-mi210`](https://github.com/davetha/llama.cpp-mi210), branch [`mi210-dsv4-flash-prefill`](https://github.com/davetha/llama.cpp-mi210/tree/mi210-dsv4-flash-prefill) · **Hardware:** 2× MI210 (gfx90a), ROCm 7.14

Prefill/TTFT campaign on the production DSV4-Flash server. Three patches shipped; **eleven hypotheses killed with measurements**. The negative results and the utilization data are the more useful half of this document — they define where the ceiling is and why.

---

## Results

Server-side `prompt_per_second` (never client wall-clock), measured in co-tenant-idle windows, 2 reps each.

| context | campaign start | after sparse attn (prior work) | **final (this campaign)** | cumulative |
|---|---|---|---|---|
| 16K | — | 568 | **589** | — |
| 60K | 362 (TTFT 166 s) | 501 (TTFT 120 s) | **538.5** (TTFT 111.4 s) | **+48.8%** |
| 100K | — | 451 | **~491** | — |

Attribution, honestly split: the 362 → 501 step is the **prior** gather-sparse-attention work (see the fork branch history). **This campaign is the 501 → 538.5 step at 60K (+7.5%) and 451 → 491 at 100K (+8.9%).** The headline +49% is cumulative across both.

Decode is unchanged throughout (~23 t/s @60K); this was a prefill/TTFT campaign.

### The three shipped patches

| # | patch | numerics | 60K | 100K |
|---|---|---|---|---|
| 1 | **Lightning-indexer MFMA `K_VECS_PER_BLOCK` 32 → 64** | bit-identical | 501 → 528 (+5.3%) | 451 → 487 (+8.1%) |
| 2 | **fattn-mma runtime-gated `V_is_K_view` at DKQ=512** | bit-identical | 528 → 532 (+0.8%) | 487 → 491 |
| 3 | **`GGML_CUDA_FORCE_MMQ=ON`** | int8 activations (not bit-identical) | 532 → 538.5 (+1.1%) | — |

Patches 1 and 2 are **bit-identical** — verified against llama.cpp's own reference implementations via `test-backend-ops` (`LIGHTNING_INDEXER` 144/144, `FLASH_ATTN_EXT` 2920/2920). Patch 3 quantizes activations to int8; needle-in-haystack passes 5/5 at all depths, but it is a (small) quality trade and was deployed only after an explicit decision to accept it.

#### 1. Indexer `K_VECS_PER_BLOCK` 32 → 64 — the biggest win

The MFMA lightning-indexer kernel stages the whole Q tile (64 heads × 128 = **16 KB per block**) into LDS, then processes `K_VECS_PER_BLOCK` keys. With 32 keys/block, that 16 KB Q load is repeated `n_kv/32` times per query — and **Q, not K, is the dominant per-block traffic**. Doubling the kv tile halves the block count in the kv dimension, halving redundant Q reloads. LDS goes to ~50 KB (fits the 64 KB budget); occupancy is unchanged.

The gain **grows with context** (+5.3% at 60K → +8.1% at 100K) because the indexer is quadratic — exactly the regime that hurts TTFT most. `K_VECS=128` is not possible: it exceeds the LDS budget.

#### 2. `V_is_K_view` at DKQ=512 — see the upstream section

#### 3. `FORCE_MMQ`

Routes the ~14% of prefill spent in rocBLAS projection GEMMs (`Cijk_*`) through quantized MMQ, which also eliminated half the `convert_unary` dispatches (38k → 16k). +1.1% prefill, +0.8% decode.

---

## What the GPU is actually doing (hardware counters)

`rocprofv3 --pmc OccupancyPercent MemUnitBusy MemUnitStalled VALUBusy`, production build, prefill.

| kernel | % time | Occ% | VALU | MemBusy | VGPR | LDS | reading |
|---|---|---|---|---|---|---|---|
| `mul_mat_q` (MoE experts) | ~36% | 23 | 46 | 36 | 84 | 0 | half-empty tiles — MoE sparsity, structural |
| `flash_attn_sparse_kernel` (ours) | 20 | **68** | **78** | **4** | 80 | 16.9 KB | compute/latency-bound, well-optimized |
| `flash_attn_ext_f16` (fattn-mma) | 9.4 | 12 | 17 | 12 | **128** | 0 | register-capped ceiling |
| `lightning_indexer_kernel_mfma` | 6.1 | — | — | — | — | — | was 12.2% before patch 1 |
| `quantize_mmq_q8_1` | 2.4 | 71 | 20 | **84** | 16 | 0 | memory-bound, inherent |

Two things worth extracting:

**The sparse attention kernel is now compute-bound.** It reads 78% VALU-busy at only 4% memory-busy. Before the int4-vectorized LDS staging (prior work) it was ~73% *memory*-stalled on the scattered gather. The counters confirm that optimization moved it clean off the memory wall.

**The low-utilization kernels are structurally limited, not badly written.** `mul_mat_q` underutilizes because MoE routing gives each expert only a handful of tokens per ubatch, so the MFMA tiles run half-empty — inherent to sparse MoE. `flash_attn_ext_f16` is at VGPR=128, which structurally caps occupancy near 50% (and is precisely why forcing occupancy=2 spilled and lost 1.2%).

---

## ⚠️ Correction: GPU overlap is *not* zero

An earlier note in this project's records claimed the two GPUs did not overlap (`0.00s`) and that "the last 2×" was sitting in that idle time. **That is wrong for the current stack.**

Measured from the kernel-trace DB — 370k dispatches, per-agent busy intervals merged, warmup excluded:

```
steady-state wall: 62409 ms
agent 1 (GPU0) busy: 60044 ms = 96.2%
agent 2 (GPU1) busy: 58657 ms = 94.0%
both-idle:            363 ms =  0.6%
```

llama.cpp's pipeline parallelism (`n_copies=4`, enabled automatically with `-sm layer` + `-ngl 99` + `offload_kqv`) is **already working near-perfectly**. The imagined 2× lever does not exist; the stack had already banked it. Raising `GGML_SCHED_MAX_COPIES` 4 → 8 confirmed this — a wash (538.7 → 538.5). GPU1's residual ~6% is intra-ubatch dependency plus ~2% layer imbalance (43 layers cannot be split evenly at one-layer granularity), not queue depth.

The methodological lesson repeats one this project has learned before: **a stale measurement is worse than no measurement**, because it directs effort at a lever that isn't there. Re-measure before optimizing against an old number.

---

## Killed hypotheses

Every one of these was built, gated for correctness, and A/B'd against production in a co-tenant-idle window. All are reverted.

| # | hypothesis | result |
|---|---|---|
| 1 | MoE MMQ `IQ2_XXS` occupancy 1 → 2 | **wash** (16K 567.5 vs 567.7; 60K server-side 562.9 vs 563.6) |
| 2 | MoE MMQ `IQ2_XXS` tile `I` 64 → 32 | **breaks correctness** — needle emits garbage (`<<<<<`). `I=64` is a *requirement* for this quant's load/store path, despite the config contract claiming tile dims are results-neutral |
| 3 | fattn-mma DKQ=512 occupancy 1 → 2 | **−1.2%** (register pressure at head_dim 512) |
| 4 | sparse kernel `TK` 16 → 32 | **−10.7%** (32 KB LDS halves resident blocks) |
| 5 | sparse kernel `TK` 16 → 8 | **−0.5%** (doubled barriers beat the occupancy gain) |
| 6 | sparse kernel: cache K value across QK-dot and PV | **wash** — compiler was already doing it; +16 VGPR cancelled the saving |
| 7 | sparse kernel: two-phase split (precompute scores for ILP, then sequential softmax) | **−7.6%** — forced register spilling and defeated the compiler's own scheduling |
| 8 | `-ub` re-sweep after FORCE_MMQ | **1024 still optimal** (2048 −1.6%, 1536 wash) — FORCE_MMQ did not shift the optimum |
| 9 | `GGML_SCHED_MAX_COPIES` 4 → 8 | **wash** (see the overlap section) |
| 10 | QK-MMA rewrite of the sparse kernel | killed *before* implementation by an env-gated diagnostic: skip-reduce and skip-PV showed compute was only ~27% of the kernel, ceiling ~+3.5% |
| 11 | gather-sparse attention for **decode** | dense decode kernel wins at every NWAVES (23.8 vs 18.1 t/s) — decode is MoE-weight-bandwidth-bound, not attention-bound |

Experiments 4–7 are worth reading together: four independent attempts to restructure the sparse kernel, all wash-or-worse. Experiment 7 is the most informative — handing the compiler *explicitly independent* reductions to pipeline made it **slower**, which is the signature of a kernel already at its instruction-scheduling optimum.

Experiment 2 is the safety lesson: a config knob documented as "should not affect results, only speed" **did** affect results. The needle gate caught it; a perf-only A/B would have shipped corrupt output.

---

## The ceiling

Measured from every direction, all agreeing:

| dimension | state | evidence |
|---|---|---|
| kernel code | at compute/latency optima | 4 sparse-kernel restructures: wash or worse |
| kernel configs | upstream-optimal or correctness-constrained | occupancy / tile-dim / TK / nbatch all swept |
| multi-GPU scheduling | 95%+ busy, 0.6% dual-idle | dispatch-trace interval analysis |
| launch flags | tuned, re-verified after every shipped patch | `-ub` swept twice |

**Whole-model 16K = 601 · 60K = 545 · 100K ≈ 491.** (An earlier revision of this document called 589/538.5 the ceiling — see the correction below; it was not.)

### Held: `top_k` 384 (a quality trade, not deployed)

`GGML_DSV4_INDEXER_TOPK=384` (runtime knob, off by default) caps the indexer's key selection at 384 instead of the model's designed 512:

| | 16K | 60K | needle |
|---|---|---|---|
| top-512 (whole model, shipped) | 589 | 538.5 | 5/5 |
| top-384 | **611** | **560** | 5/5 |

It is the only measured way to cross **600 tok/s at 16K** on this hardware. It is **not deployed**, because it changes what the model computes: the model attends to 384 of its 512 selected compressed keys, so outputs change. Needle passes at all depths, but needle only tests retrieval — it is a weak proxy for generation and multi-hop reasoning quality. **A perplexity A/B is the honest gate before anyone deploys this**, and it has not been run.

### Not attempted: MoE grouped-GEMM

The one genuinely unbuilt lever — pack multiple experts' tokens into full MFMA tiles instead of per-expert half-empty ones, addressing the `mul_mat_q` 23% occupancy directly. Estimated **1–2 weeks** (quant-format-correct implementation against a path that experiment 2 proved fragile, plus correctness debugging), high risk, single-digit-% expected payoff. The `-ub` sweep (experiment 8) is weak evidence *against* its premise: packing more tokens per expert was available for free and did not help.

---

## Reproducing

```bash
# Build (fork branch mi210-dsv4-flash-prefill)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx90a -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_CUDA_FORCE_MMQ=ON -DLLAMA_CURL=OFF
cmake --build build -j 24 --target llama-server

# Serve (sparse attention on; -ub 1024 is the measured optimum, do not raise it)
GGML_DSV4_SPARSE_ATTN=1 GGML_CUDA_REGISTER_HOST=1 \
  ./build/bin/llama-server -m DeepSeek-V4-Flash-0731-Abliterated-DS4-Quality128.gguf \
  -ngl 99 -sm layer -np 1 -c 131072 -fa on -t 24 -b 8192 -ub 1024
```

Measurement discipline that made these numbers trustworthy:

- **Server-side timings only.** `timings.prompt_per_second` from the completion response — never client wall-clock, which co-tenant load makes meaningless.
- **Co-tenant-idle windows.** A second model shares these GPUs; benchmarking against it both slows *and* contaminates results. Every A/B verified the co-tenant was quiet, and re-measured the baseline in the same window rather than trusting a stored number.
- **Correctness gate on every change**, including "performance-only" ones — `test-backend-ops` for op-level identity, needle-in-haystack at multiple depths for end-to-end coherence. Experiment 2 is why.
- **ROCm VRAM-release lag** — `docker rm -f` returns before VRAM frees; every relaunch waits for VRAM to actually drop before starting, or the next server boots into a starved GPU and reports garbage.


---

## ⚠️ Correction: 589/538.5 was not the ceiling — a fork-local patch was costing 2%

After declaring the campaign closed, an isolated benchmark of the fork's own inherited
patches found that **`MI210_MOE_J` — the MoE J-tile-selection heuristic — is a net loss.**

It sizes `ncols_sel` against the expected per-expert load in `mul_mat_q_switch_J`, on the
reasoning that dead tiles are expensive because *"CDNA takes the stream-k branch, so those
dead tiles still occupy the persistent-block decomposition."* But the CDNA MMQ config in
this tree runs **`stream_k = false`**, where out-of-range tiles early-return cheaply. The
premise does not hold on this hardware, so shrinking J only adds tiles.

Measured twice, independently:

| | pp8192 / 16K | pp16384 / 60K |
|---|---|---|
| pristine `upstream/master`, with the heuristic | 380.59 ± 0.27 | 329.08 ± 0.10 |
| pristine `upstream/master`, without | **386.09 ± 0.50** | **332.18 ± 0.43** |
| production config, with (was shipping) | 589.3 | 538.6 |
| production config, without | **600.9** | **545.8** |

**+1.9% at 16K, +1.4% at 60K**, TTFT@60K 111.4 → 109.9 s, needle 3/3, decode unchanged.
Numerically neutral: J is a tile-*selection* parameter; accumulation order along K is
unchanged. Deployed and verified live at **16K = 601.1, 60K = 545.4**.

This crosses **600 tok/s at 16K on the whole model**, which the campaign had concluded was
only reachable via the quality-traded `top_k=384` setting. It was not.

The lesson generalises past this patch: **inherited fork patches deserve the same isolated
A/B as new ones.** This one had been carried for months on a rationale that silently stopped
matching the config it ran on. The campaign spent a dozen experiments looking for new wins
while a regression sat in the build.

### Tier-2 upstream candidates, both killed by measurement

- **GPU multi-pass top-k** — wash (386.04 vs 386.09; 332.33 vs 332.18). An earlier revision
  of this document claimed it "removes a CPU fallback above ncols 1024". **That was wrong**:
  upstream's non-CUB path already uses a GPU bitonic argsort. Top-k is ~1.3% of kernel time,
  so any gain is structurally below noise.
- **MMQ J-selection** — regression, as above.

The upstream list is therefore just the **MFMA head-size cap lift** (+15.9%/+16.1% on
pristine upstream, ready) and the **state-restore pin** (crash workaround for the open
issue #20176).
