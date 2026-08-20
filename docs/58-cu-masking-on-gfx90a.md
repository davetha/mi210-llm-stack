# CU masking works on gfx90a, graph replay and all — but it buys isolation, not concurrency

Phase 0 for a prefill/decode co-scheduling experiment, measured 2026-08-09 on
one MI210 (gfx90a), ROCm 7.14, image `local/vllm-mi210:dsa7`. Probe source:
[`benchmarks/cu_mask_probe.hip`](../benchmarks/cu_mask_probe.hip).

Nothing here is a serving result. This document exists because the co-scheduling
idea rests on five hardware assumptions that this repo had never tested, three of
which are undocumented for CDNA2, and it is cheaper to falsify them with a
200-line HIP program than to discover them inside a vLLM patch.

Four of the five hold. The fifth does not, and it is the one that reframes the
whole line.

## Why look at this at all

`docs/42` closed the launch-overhead explanation for graph-mode batch-1 decode:
**99.9% kernel coverage**, 0.1% gap fraction. There is no meaningful wall time
*between* kernels. But round 62 in [`docs/52`](52-data-parallel-plus-asm-attention.md)
measured what is happening *inside* them — **113 GB/s of 1170 (10%)** achieved
HBM bandwidth and **VALU 15.5 inst/cycle of ~104 (15%)** — and diagnosed decode
as latency/occupancy-bound: not enough concurrent work in flight.

`docs/43` gives the occupancy shape concretely. On a **104-CU** card, at M=1:

| kernel | qkv_proj grid | o_proj grid |
|---|---:|---:|
| Triton | 40 workgroups | 16 |
| CK | 80 | 32 |

So a batch-1 decode step never fills the machine. The question this raises is not
"how do we make one request use 104 CUs" — round 62 says the work is not there to
spread — but "can something else use the CUs decode is not touching, without
disturbing decode."

HIP exposes `hipExtStreamCreateWithCUMask()`. AMD Research's RAPID-Serve
(**2601.11822**, Masood/Gaur/Jayasena, 2026-01-16) reports up to 4.1× unconstrained
and 32× SLO-constrained throughput from exactly this mechanism on "AMD Instinct
GPUs" — the abstract does not name a part number, and CDNA3 is not CDNA2, so
that paper is motivation, not evidence for this box.

## What was measured

The probe creates masked streams, reads the mask back, and launches a kernel in
which each workgroup reads `HW_ID` (GFX9: `CU_ID [11:8]`, `SH_ID [12]`,
`SE_ID [15:13]`) and marks its physical slot. `hipSuccess` is not evidence;
the recorded CU IDs are.

### 1. The mask is accepted and read back intact

| mask | bits set | `hipExtStreamGetCUMask` | physical CUs that ran |
|---|---:|---|---:|
| bits 0–15 | 16 | IDENTICAL | **16** |
| bits 16–31 | 16 | IDENTICAL | **16** |
| bits 0–51 | 52 | IDENTICAL | **52** |
| bits 52–103 | 52 | IDENTICAL | **52** |
| all | 104 | IDENTICAL | **104** |

Placement is exact, not approximate. Bits 0–51 and 52–103 shared **zero** CUs.

### 2. The flat mask is de-interleaved across shader engines

This is the part that will silently produce a wrong experiment if assumed.

```
bit  0 -> se0/sh0/cu0      bit  8 -> se0/sh0/cu1
bit  1 -> se1/sh0/cu0      bit  9 -> se1/sh0/cu1
bit  2 -> se2/sh0/cu0      bit 10 -> se2/sh0/cu1
...
bit  7 -> se7/sh0/cu0      bit 11 -> se3/sh0/cu1
```

MI210 is **8 shader engines × 13 CUs**, all in SH0, and the driver maps
`SE = bit % 8`, `CU_within_SE = bit / 8`. A *contiguous* run of 13 bits therefore
spans **all eight SEs**, one or two CUs each — not one SE. Any sweep that assumes
"bits 0..N-1 gives me N adjacent CUs" is measuring a different partition than it
thinks it is, and any attempt to isolate a workload to one SE's L1/LDS needs
stride-8 bits, not a contiguous run.

### 3. Disjoint masks do execute concurrently

Two 512-workgroup spin kernels, ~36 ms each:

| arrangement | wall | vs serial |
|---|---:|---:|
| single kernel, masked to 52 CUs | 35.82 ms | — |
| serial: A then B | 71.61 ms | 1.00× |
| concurrent, **disjoint** masks (0–51 / 52–103) | 35.97 ms | **1.99×** |
| concurrent, **two ordinary unmasked streams** | 35.97 ms | **1.99×** |

### 4. HIP graph capture and replay preserve the mask

This was the open question — it is undocumented, and the plausible failure mode
(the mask binds to a dedicated HSA queue at stream creation, while
`hipGraphLaunch` may route nodes through runtime-managed queues) would have
killed the line outright, because vLLM decode is graph-replayed.

It does not fail:

| launch | CUs used |
|---|---:|
| direct launch on a 16-CU masked stream | 16 |
| **same work captured to a graph, replayed on that masked stream** | **16** |
| same graph replayed on an unmasked stream | 104 |

The mask follows the stream the graph is *launched* on, which is the useful
semantics: one captured decode graph can be replayed wide or narrow.

### 5. …and that is exactly why the premise is weaker than it looks

Row 4 of the concurrency table is the finding. **Two unmasked streams reached the
same 1.99× as two disjoint-masked streams.** HIP already overlaps independent
streams on this hardware. CU masking did not enable the overlap and did not
improve it.

So masking does not buy *concurrency*. What it can buy is *isolation* — a bound
on how much a co-resident workload can perturb a latency-sensitive one. That is a
real and different property, and it is worth strictly less than the framing the
literature invites.

**Read this result narrowly.** The probe kernel is a pure-ALU spin loop. It
touches no HBM, so it says nothing about the contention that actually matters
here: round 62 already measured decode at 10% of achievable bandwidth, which
means a co-scheduled prefill is competing for the *other* 90% of a shared
resource that no CU mask partitions. Two disjoint CU sets still share L2, the
HBM controllers, and the board power budget.

## What must be measured before any vLLM patch

The interference matrix, with memory-bound kernels rather than a spin loop:
CK W8A8 at M=1, fused MoE at decode M, and paged attention at depth on the
decode side; large-M GEMM and chunk-sized attention on the prefill side. Sweep
the decode allocation over `{16, 24, 32, 40, 48, 64, 80, 104}` CUs — **as
stride-8 bit patterns, per finding 2** — and compare four arrangements: serial,
two unmasked streams, disjoint masks, deliberately overlapping masks. The second
of those is the real control, and finding 5 says it will be hard to beat.

Two constraints to design around:

- A masked stream is granted a **dedicated HSA queue**, against a hardware budget
  of 32 compute queues (8 per ACE). A policy that creates masked streams per
  request will exhaust them.
- The mask can only be set at **stream creation**. There is no setter. An
  adaptive policy has to pre-create its partitions and route work between them,
  not re-mask a live stream.

## How this line dies

- **Against DP=2, not against nothing.** `docs/52` already measured DP=2 plus ASM
  paged attention at **1.118× aggregate, 0.888× TPOT** and adopted it — two
  independent single-card engines, zero inter-GPU traffic, no kernel work. That
  is the incumbent for "use the box better," and it is already deployed. A CU
  masking result that does not beat it is not a result.
- **Power, which no CU mask partitions.** `docs/53` found both cards pinned at
  199–200 W against a 300 W cap, now deployed at 250 W, and prefill gaining
  +8.4% at 300 W while decode is flat. Prefill on this box is *already*
  power-limited. Co-scheduling prefill next to decode contends for the one
  resource `docs/53` identified as binding, and the CU mask does nothing about
  it. This is the most likely way the aggregate win fails to appear.
- If HBM contention dominates, disjoint CUs will give <10% aggregate gain.
- If ordinary continuous batching reproduces the gain — the null that finding 5
  already suggests at the kernel level.

## Reproduce

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -e HIP_VISIBLE_DEVICES=0 -v "$PWD:/w" -w /w --entrypoint bash \
  local/vllm-mi210:dsa7 -c \
  'hipcc -O2 --offload-arch=gfx90a cu_mask_probe.hip -o cu_mask_probe && ./cu_mask_probe'
```

Runs in under a minute and needs no model.
