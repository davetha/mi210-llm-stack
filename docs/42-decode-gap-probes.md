# Rounds 38–39: three nulls, a same-day retraction, and the decode gap is in-kernel

**Date**: 2026-08-01 · Model: Qwen3-30B-A3B W8A8 · Images: `vllm-mi210:v0.26.1rc0`
(rounds 38/38b/38e), `vllm-mi210:latest` = 0.23.1 (round 39) · TP=2, P2P on,
AITER FA on, tuned MoE config off · Scripts:
`benchmarks/matrix/round38_decode_gap.sh`, `round38b_noasync.sh`,
`round38e_profile.sh`, `round39_async_023.sh` · Results:
`benchmarks/matrix/results/rd38-*.json`, `rd39-*.json`

`docs/39` re-derived batch-1 decode as ~3.1× off its bandwidth bound and scoped
a profiling decomposition that had never been run. Before running it, three
config-only suspects had zero measurements anywhere in this repo. This round
measured all four things. Headline: **the three cheap suspects are nulls, the
decomposition finally ran, and it puts the gap in-kernel — 99.9% kernel
coverage during decode under graph capture.** Along the way it measured async
scheduling — default-on in both shipped versions — at **1.064×** (0.23.1) to
**1.110×** (0.26.1rc0) on decode. An earlier revision of this document
concluded from that number that 0.26's synchronous path had regressed and the
default masked it; §2 retracts that claim with the measurement that refutes
it. This document is rounds 38 and 39 together, kept as one record including
the wrong turn.

## 1. The four arms

One variable per arm, same image, same 27,852-token longctx requests —
`LONGCTX_TOKENS` was pinned round-wide precisely so `run_arm.sh`'s per-arm
clamp could not hand different arms different workloads.

| arm | variable | longctx decode | vs base |
|---|---|---:|---:|
| `rd38-base` | — (131k max-len, auto clocks) | 56.58 t/s | — |
| `rd38-clkhigh` | perf level pinned high (1700 MHz) | 54.79 | 0.968× |
| `rd38-async` | `--async-scheduling` | 54.93 | 0.971× |
| `rd38-len32k` | 32k max-len, same requests | 55.76 | 0.986× |

Prefill (cold16k 8398–8415 t/s) and TTFT (1.80–1.81 s / 3.74–3.75 s) were flat
across all four arms. The spread of the three no-op arms, ±2–3%, is the noise
floor for single-variable decode comparisons at this scale.

### Clock pinning: null, with a lesson about the sampler

Every arm ran a 1 Hz sysfs sampler (`pp_dpm_sclk` starred level + hwmon power,
both cards). The pooled distributions look alarming — base spent 77% of
samples at 800 MHz, median power 42 W — but that pool is dominated by the idle
stretches of an arm (load, teardown, between reps), so it cannot resolve what
clocks did *during decode*. The evidence that DPM was not throttling decode is
the A/B itself: pinning 1700 MHz moved decode 0.968×, i.e. nothing. The
`clkhigh` sampler line (94% at 1700, median 61 W) proves the pin took; the
number proves it didn't matter. docs/30's issue-bound conclusion does not have
a hidden clock component.

### Capture geometry: null — docs/39 item 2 is answered

Arm A captured CUDA graphs for `--max-model-len 131072` (npar_loops geometry
for 512 partitions), arm D for 32768 (128 partitions). Both served identical
27,852-token requests: 56.58 vs 55.76 t/s, 0.986×, inside the noise floor.
Serving with 4× oversized captured geometry costs nothing at these request
sizes — the paged-attention reduction is not paying for unused partitions, and
there is no self-inflicted config loss in the production `--max-model-len
131072` setting. (`docs/23`'s gradient question below the gate is settled for
this regime.)

## 2. The vacuous arm, and the A/B that replaced it

`rd38-async` measured 0.971× — which would have been recorded as "async
scheduling: null" except for one check: **the base arm's serverlog also says
"Asynchronous scheduling is enabled."** Async scheduling is ON BY DEFAULT in
0.26.1rc0 (`SchedulerConfig.async_scheduling: bool | None = None`, resolved to
enabled). The two arms were identical configurations. That is the fourth
sighting of this project's most persistent defect class — two arms silently
identical, producing a plausible number — after rounds 31, 32, and 37, and the
first one caught before it reached a doc.

Round 38b ran the informative direction, `--no-async-scheduling`, with the
check the vacuous arm lacked (the arm refuses to report if its serverlog still
shows async enabled):

| workload | metric | async ON (default) | async OFF | ON/OFF |
|---|---|---:|---:|---:|
| cold16k | prefill | 8414.8 | 8386.7 | 1.003× |
| longctx | prefill | 6879.3 | 6883.9 | 0.999× |
| longctx | **decode** | **56.58** | **50.96** | **1.110×** |
| longctx | ttft | 3.74 | 3.75 | 0.997× |

**Async scheduling is worth 1.110× on batch-1 decode on this box**, and the
shipped images already have it. Nothing to change in production; the number
matters for attribution.

### The regression it masks — RETRACTED, same day (round 39)

The original text of this section claimed 0.26's synchronous decode path was
~0.92–0.94× of 0.23.1's, a 6–8% regression masked by async scheduling, on the
premise that "0.23.1 *had no async scheduling*." **The premise is false.**
`async_scheduling: bool | None = None` exists in v0.23.1rc0 with the same
resolution to enabled, and round 36's serverlogs — checked only after the
claim shipped — say "Asynchronous scheduling is enabled" in **all three**
version arms. The arithmetic compared 0.23.1 *with* async against 0.26
*without* it and called the difference a version regression.

Round 39 measured the cell that comparison was missing (0.23.1, async off;
`round39_async_023.sh`, pins identical to rounds 38/38b, async state asserted
from the serverlog per arm):

| version | async ON | async OFF | ON/OFF |
|---|---:|---:|---:|
| 0.23.1 | 54.64 | 51.36 | 1.064× |
| 0.26.1rc0 | 56.58 | 50.96 | 1.110× |

Sync path across versions: **0.992×** — dead center of the ±2–3% noise floor.
Async-on across versions: 1.035×. **There is no regression, masked or
otherwise.** The synchronous path is flat from 0.23.1 to 0.26.1rc0; the async
*implementation* got somewhat more effective (1.064× → 1.110×, plausibly the
#28250 scheduling-config rework); flat sync plus better async is exactly the
+2.7% round 36 reported end-to-end. What survives of this section: async
scheduling is worth 6–11% on this box, both versions ship it enabled, and a
version sweep's end-to-end number can still hide compensating pairs — this
one just didn't contain one.

The method failure is worth naming precisely: the original claim rested on a
*remembered* property of 0.23.1 ("had no async") that one grep of an existing
serverlog would have falsified. Verified: the claim survived less than two
hours.

## 3. The decomposition: 99.9% kernel coverage — the gap is in-kernel

The docs/39 profiling decomposition finally ran: rocprofv3 kernel-trace on
TP=1 offline decode (8k-token prompt, 1024 generated tokens, in-process
engine), union-of-intervals kernel coverage over the trace's final 8 s of pure
decode. Two attempts, both modes:

| mode | generate() wall | decode-tail coverage | gap fraction | launches/s |
|---|---:|---:|---:|---:|
| graphs (production) | 21.1 s | **99.9%** | 0.1% | 46,659 |
| enforce_eager | 74.9 s | 18.2% | **81.8%** | 13,827 |

Two conclusions, one per row:

1. **In production graph mode, decode is wall-to-wall kernels.** The
   launch/CPU-bound share of the ~3× gap to the bandwidth bound is 0.1%.
   docs/30's instruction-count argument is now corroborated by direct
   measurement: the residual is *inside* the kernels — how many instructions
   each one issues per useful byte — not between them. Every lead that attacks
   launch overhead at batch 1 (fewer kernels for scheduling's sake, launch
   batching) is dead on this box. Persistent-megakernel ideas survive only via
   their locality/fusion argument, not their launch argument.
2. **Eager mode is a catastrophe and CUDA graphs are doing enormous work**:
   3.5× slower wall, 82% of decode time spent in gaps between kernels. Any
   configuration that silently drops to eager (a capture failure, an
   `enforce_eager` debug flag left on) costs more than every optimization in
   this repo combined. Worth knowing exactly, not approximately.

Where the time goes (whole-trace totals, graph run — prefill included, so
ratios not exact for decode, but the ordering held in the eager run too):
`scaled_mm_kernel` 10.9 s, `fused_moe_kernel` 4.7 s, `add_rmsnorm_quant` 2.0 s,
`paged_attention_ll4mi` 1.9 s, `topkGating` 0.7 s, `dynamic_scaled_int8_quant`
0.7 s, `wvSplitK` 0.6 s. The int8 GEMM and the MoE kernel are the targets;
everything else is a rounding error. That points the next kernel work at the
`docs/39` 1b W4A16 audit and at `scaled_mm`/`wvSplitK` decode shapes.

### What it cost to get the trace: the profiler breaks the thing it profiles

Three failed attempts, each one layer deeper, all one root cause: **rocprofv3's
preloaded tool prints "Streaming Performance Monitor (SPM) is not supported on
gfx90a devices" onto the STDOUT of every subprocess the profiled process
spawns.** Anything that parses a child's stdout is poisoned:

- Round 38's phase 2: died; the error was discarded by a `tail -8` (a
  self-inflicted silent truncation — the round's own summary then "analyzed"
  the corpse trace at face value and printed a 0.2% coverage line that meant
  nothing).
- 38c (full logs): AITER's `cpp_extension.py:258` does
  `int(HIP_VERSION.split(".")[0])` on raw `hipconfig --version` stdout —
  `int()` ate the SPM sentence. AITER imports lazily on the first attention
  call, which is why init, weight load, and a 24 s torch compile all succeeded
  first.
- 38d (parse patched): AITER then JIT-built `module_aiter_core` and
  `module_rmsnorm_quant` — first use in any fresh container; serving
  containers build them cleanly because no profiler is attached — and the
  build died on the literal `(` of "(SPM)" reaching a shell command line.
- 38e (worked): stop patching consumers, drain the producer. Run the identical
  workload twice in the same container — once un-profiled so every JIT build
  and compile lands in container-local caches with clean stdout, then under
  rocprofv3 where everything cache-hits. Success requires the completion
  marker from BOTH runs, so a dead profiled run cannot pass on the warmup's.
  Profiling the warm system is also the methodologically correct measurement.

Upstream candidates, neither filed: rocprofiler printing warnings to a child's
*stdout* (stderr would be harmless), and AITER's unguarded `int()` on
subprocess output. The ephemeral in-container patch is generated by
`round38e_profile.sh`; the shipped images are untouched.

## 4. Scoreboard after this round

| lead | status |
|---|---|
| DPM/clock throttling during decode | **Closed** — 0.968×, pin verified by sampler |
| Capture-geometry decode tax (docs/39 §2) | **Closed** — 0.986×, same-workload A/B |
| `--async-scheduling` as a new win | **Closed** — already default-on; worth 1.110× and already banked |
| Launch overhead at batch-1 decode | **Closed** — 0.1% gap fraction under graphs |
| 0.26 synchronous-path decode regression | **Refuted** — round 39: sync path 0.992× across versions; see the §2 retraction |
| In-kernel efficiency of `scaled_mm` / `fused_moe` | **Open, and now the only game in town** for the ~3× — via docs/39 1b (W4A16 audit) and docs/30 instruction work |
| Spec-decode re-run condition (docs/39 1c) | Unchanged — the 3× persists; it is in-kernel, so the "re-run after the gap closes" trigger has not fired |
