# Backlog addendum: two items resized, one inverted, two new

**Date**: 2026-07-30 · Companion to `docs/25-optimization-backlog.md`.
Literature claims carry arXiv IDs; every number here is either from this repo or
shown with its arithmetic.

---

## 1c — RESIZED. The decode gap is ~3×, not 6.4×

Item 1c compares observed decode against **peak** HBM (1.6 TB/s) and omits KV
traffic. Redone against the achieved bandwidths measured elsewhere in this repo —
73% of peak for decode-shape GEMM (`docs/20`), 593–611 GB/s for HIP paged
attention (`aiter-cdna2` README) — on the same arm (Llama-3.3-70B W8A8, TP=2,
60k ctx):

| term | bytes/rank | at achieved BW | ms |
|---|---:|---|---:|
| weights | 36.4 GB | 1.17 TB/s | 31.1 |
| KV — 80 layers × 8 KV heads ÷ 2 ranks, 320 KiB/tok total | 9.8 GB | 0.60 TB/s | 16.4 |
| **predicted** | 46.2 GB | | **47.5 → 21 tok/s** |
| observed | | | **146 → 6.85 tok/s** |

**~3.1× unexplained, not 6.4×.** Roughly 40% of what the item calls unexplained is
peak-vs-achieved plus the omitted KV term.

This matters for prioritisation, not just bookkeeping: 3× is reachable by
instruction-count work (see `docs/30`), whereas 6.4× invited hunting for a bigger
structural cause than may exist.

### The literature anchor 1c was missing

Hazy Research's megakernel work (`github.com/HazyResearch/Megakernels`,
2025-05-27) measured **stock vLLM and SGLang at ~50% of H100 bandwidth at batch
1**, rising to **78%** when the entire forward pass is fused into one persistent
kernel — **2.5× on well-supported CUDA hardware**. Mirage MPK closed most of a
remaining 1.45× on A100.

So batch-1 decode sitting well off its bandwidth bound is the **normal** failure
mode, not an AMD or CDNA2 artifact. This box is a worse instance of a known 2×.

**Consequence for the speculation conclusion.** Item 1c's *"speculation loses on
every architecture"* is too strong. On optimized CUDA decode (~1.45× off the
bound) the premise still roughly holds. It loses **here because of the 3×** — so
the MTP and n-gram arms should be re-run *after* the gap closes rather than being
treated as permanently closed.

**What does not port:** every megakernel implementation is warp-specialized around
**TMA + `cp.async` + `wgmma` + wave32**. gfx90a has none of those and is wave64;
the on-GPU interpreter's synchronisation layer breaks. The idea (fewer, larger,
persistent kernels) is portable; the code is not.

**UPDATE 2026-08-01**: run in round 38e (`docs/42`). Result: **99.9% kernel
coverage** during graph-mode decode — launch/CPU share is 0.1%, the gap is
in-kernel, and the eager-mode control measured 82% gap fraction (3.5× slower
wall), so graph capture is what stands between this box and that fate.

**The decomposition 1c says has never been done, concretely:**
`rocprofv3 --kernel-trace --hip-trace` on decode steady-state (skip prefill and
the first token). Sum of kernel durations vs wall clock gives the launch-bound
fraction; the dispatch→kernel-start gap gives per-launch cost. That separates all
three unprofiled suspects in one run. For roofline,
`rocprof-compute profile --roof-only` — MI210 resolves to the **`MI200`** target
directory, metrics under **`gfx90a`**; L2-Fabric is block **17** (`17.1.2` hit%,
`17.1.3` read BW), speed-of-light is block **1**. Measure steady-state only, and
remember wave64 changes the occupancy arithmetic against any CUDA intuition.

---

## 1b — INVERTED. W4A16-fp16 is the better target than W4A8-int8

Item 1b calls a Triton W4A8-int kernel the highest-value open lead, following
QServe and QQQ. **Their premise does not hold on CDNA2.**

- QServe (**2405.04532**, 2024-05-07) and QQQ (**2406.09904**, 2024-06-14) exist
  because int8 tensor cores are **2× fp16** on Hopper/Ada. Their entire design
  goal is feeding int8 MMA without stalling on dequant.
- On gfx90a, int8, f16 and bf16 MFMA **all cap at 181** — `docs/20` measured int8
  GEMM as *a wash* at prefill shapes (102 TOP/s vs bf16's 95 TFLOP/s), winning
  only at decode shapes and only on bytes.
- So W4A8's motivation evaporates. **W4A16-fp16 gets the same MFMA rate, a better
  mantissa (10 explicit bits vs 7), and skips per-token activation quantization
  entirely** — one fewer pass over activations and one fewer reduction per token,
  on a machine that `docs/30` shows is issue-bound.
- **And the path already exists.** `docs/27` found W4A8 blocked by two things, one
  of which was that no ROCm kernel accepts *signed* `int4` —
  `TritonW4A16LinearKernel` wants `uint4b8`/`uint4`, which is exactly what **AWQ
  and GPTQ already emit**. W4A16 sidesteps both blockers: no new checkpoint
  format, no registry work, no new kernel registration.

**Portability note worth keeping.** QServe explicitly **avoids `ldmatrix`** — the
int4-storage/int8-compute type mismatch defeats it — and uses offline weight
reordering instead, with dequant described as "three logical operations per four
weights." That makes the *algorithm* unusually portable. The NVIDIA couplings are
`lop3.b32` (→ 2 gfx9 ops, or 1 `v_perm_b32`) and the warp-32 `mma` fragment
layout (→ re-derive for K=16 wave64). QQQ's per-channel **shift-by-4** trick is
cheaper still — 1 `v_lshlrev_b32` per pair, fold the ÷16 into the epilogue — but
its per-group path depends on `prmt` and should not be ported.

**First experiment:** serve an existing AWQ-Int4 checkpoint, dump the Triton
W4A16 kernel, count ins/MFMA per `docs/30`. If it is ~100:1, the `docs/24`
result (AWQ-Int4 5.07 s vs W8A8 3.20 s despite half the bytes) is fully explained
and the fix is the unpack, not the format.

**How this could be wrong:** if the W4A16 kernel is already tight, int4's loss is
elsewhere — group-wise scale loads or tiling — and this line is mis-aimed.

---

## 2 — The cliff does not explain the gap, but the gradient question is still open

Worth stating because it is the tempting explanation for 1c and it is wrong.

The gfx9 gate is `max_seq_len <= 128 * 1024`. `round6_spec_dense.sh` runs the 70B
arm at `--max-model-len 131072`, which **equals** `128*1024`, so the gate
**passes** and the fast custom kernel is captured.
`npar_loops = ceil(ceil(131072/256)/64) = 8` — exactly the last valid case in the
switch. **No fallback, no cliff.** The arm sits one token below its own cliff.

But `docs/23` lists as **UNVERIFIED** whether the boundary is a step or *a
gradient below it*, and proposes bisecting `--max-model-len` to find out. Capture
bakes geometry derived from 131072 (512 partitions) while requests are 60k (~235
partitions). If the reduction does not early-exit per sequence, part of the 3× is
self-inflicted config.

**A 20-minute test that is already scoped and has never run.** Serve the 70B arm
at `--max-model-len 65536` and re-measure decode at 60k.

**UPDATE 2026-08-01**: run in round 38 on the 30B (`docs/42`) — 131k vs 32k
capture serving identical 27,852-token requests came out 0.986×, a null. No
self-inflicted geometry tax; the gradient question is settled for this regime.

---

## NEW — 9. DP=2 instead of TP=2 for anything that fits one card

TP=2 returns **1.28×**, not 2×, from doubling both bandwidth and compute:

| arm | decode @101k |
|---|---:|
| `t35-w8a8` TP=1 (`results/t35-w8a8-longctx.json`) | **33.80 tok/s** |
| `t35-w8a8` TP=2 (`docs/25` 1c) | **43.40 tok/s** |

With `NCCL_P2P_DISABLE=1` — correctly set, there is no xGMI — a decode step is
~2 host-staged allreduces per layer. Sub-linear scaling of that shape is per-layer
collective latency, not bandwidth.

`docs/04:63-66` measured tensor-parallel (2342 t/s) against expert-parallel
(811 t/s, 2.9× worse) and concluded correctly for this fabric. But **"data
parallel" / DP=2 appears nowhere in this repo.** Two independent single-card
engines behind the existing llama-swap/litellm router pay **zero** inter-GPU
traffic. For any model that fits one 64 GB card — Qwen3-30B-A3B W8A8 and the
smaller arms — DP=2 should dominate TP=2 on aggregate throughput.

This is a config change, not code. It is the cheapest untested item in the
backlog.

**How this could be wrong:** DP=2 doubles throughput but does nothing for
single-request latency, and halves the KV cache available per replica. It is a
throughput answer, not a TTFT answer. TP stays forced for anything exceeding one
card.

---

## NEW — 10. Low-bit allreduce is the one unclaimed lever on this fabric

Given the 1.28× above, compressing the collective is the direct attack:

- "Towards Low-bit Communication for Tensor Parallel LLM Inference"
  (**2411.07942**, Apple/CMU) — 16 → **4.2 bits** average, 98–99.5% of original
  quality.
- Flash Communication (**2412.04964**); Communication Compression
  (**2411.09510**) — 3.5–4.5× comm reduction, up to 2× TTFT.

**None are implemented for ROCm.** It is ISA-independent — quantize the reduced
tensor, no MFMA shape or wave-width dependency — which makes it a rare
build-it-yourself item that carries no gfx90a risk.

**Already dead, do not retry:** Flux (**2406.06858**) and PyTorch Async-TP both
require **P2P / CUDA-IPC**. Flux's own paper notes it could not run on a tested
RTX 4090 server for exactly this reason. Two independent blockers on this box.

---

## Additions to "Closed — do not retry"

| Lead | Verdict |
|---|---|
| **"int8 is free on CDNA2, so spend arithmetic to buy back bits"** | **Closed.** 181 = 181 is a *peak* fact; decode runs at ~21% of peak (M=16 int8 measured 38.8 TOP/s) and is issue-bound. Adding passes enlarges the term that causes the gap. |
| Ozaki / ozIMMU int8 emulation of higher precision (**2306.11975**) | **Closed for inference.** Needs `s(s+1)/2` int8 GEMMs — **3× passes** for ~int16-effective precision at s=2. Backwards at decode. Follow-ons (**2504.08009**, **2508.00441**) all move *away* from int8 toward fp8. |
| Atom (**2310.19102**) | W4**A4** — int4 activations. No int4 MFMA on gfx90a; assembler rejects every `i4` form. |
| T-MAC (**2407.00088**), LUT-GEMM (**2206.09557**) | Replace multiply with table lookup on a chip that *has* 181 TOPS of matrix units. LUT-GEMM's own paper reports it underperforming dequant-based CUTLASS. Raises issue pressure — the exact wrong direction given `docs/30`. T-MAC is CPU-only anyway. |
| QuIP# (**2402.04396**), AQLM (**2401.06118**), QTIP (**2406.11235**), GPTVQ | Codebook **gather** then dequant-to-float. Optimises FLOPs (irrelevant here) and pessimises issue rate plus adds random access (both costly here). |
| QQQ per-group path | Depends on `prmt`. Per-channel shift-by-4 only. |
| Flux / PyTorch Async-TP (**2406.06858**) | Require P2P / CUDA-IPC. |
| Expert parallelism | Already closed by `docs/04`: 811 vs 2342 t/s. Correct answer for this fabric. |
| `HIP_FORCE_DEV_KERNARG=1` | Already correctly marked NO-EFFECT in `env/gfx90a-common.env` (gated on `isGfx94x`). Noted only so it is not re-litigated. |
