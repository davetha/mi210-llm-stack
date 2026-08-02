# The five open leads, measured

`docs/49` closed the `fmoe` ASM investigation and left five leads standing. All
five were run. **Four are nulls or corrections, one is a confirmed diagnosis
that redirects the next piece of work.** The two most-recommended leads — the
ones ranked first and second — both failed, and the reasons are worth more than
the ranking was.

## Summary

| # | lead | result |
|---|---|---|
| 1a | tune CK a8w8 int8 GEMM for gfx90a | **null** — 12/12 shapes within ±1.1% |
| 1b | tune CK 2-stage MoE for gfx90a | **impossible** — zero valid kernel pairs exist |
| 2 | DP=2 instead of TP=2 | **1.068×** throughput, median TTFT 50% worse |
| 3 | AWQ-Int4 instructions-per-MFMA | **confirmed** — 209:1, unpack-bound |
| 4 | wire a `pa_fwd_asm` call site | **no code needed** — one env var; 1.033× tput, 0.957× TPOT |
| 5 | low-bit allreduce | **closed by arithmetic** — 0.102% prize |

The only actionable outcomes are **item 4** (an environment variable that
engages a kernel this card has never used, worth a few percent under
concurrency) and **item 3** (a confirmed diagnosis pointing at the int4 unpack).
Everything else is closed.

## 1a — tuning the int8 GEMM buys nothing (round 52)

`a8w8_tuned_gemm.csv` genuinely has zero gfx90a rows (26 gfx942, 553 gfx950),
so the 1.48× win from `docs/43` has always run on a default heuristic config.
That was true. It was also harmless:

```
Total shapes: 12 | Updated: 0 (improved: 0, new: 0) | Skipped: 12
Threshold: >= 1.0% improvement to update

(1,    2048, 2048)     9.60 ->   9.69 us   -0.95%
(1,    2560, 2048)    10.89 ->  10.98 us   -0.84%
(2,    2560, 2048)    10.81 ->  10.78 us   +0.34%
(128,  2560, 2048)    53.33 ->  53.31 us   +0.04%
(2048, 2560, 2048)   234.38 -> 233.86 us   +0.22%
```

All twelve shapes land between **−1.13% and +0.34%**, symmetric about zero —
the signature of measurement noise, not of a better kernel that narrowly missed
the bar. Across 83 int8 candidate instances the tuner found nothing that beats
the default. The missing table rows were real; they were not costing anything.

### The FP8 tax, which is a separate and fixable finding

`csrc/ck_gemm_a8w8/gen_instances.py` emits **both** `abI8` and `abF8` instances
for every kernel — its own comment says so — giving 83 of each. gfx90a has no
FP8 MFMA (`docs/49` proves this at the assembler), so half of every tuning
build is unusable. And the cost is not merely 2×: one instance,

```
a8w8_rowwise_256x256x256x128_..._intrawave_v3_abF8_dF32_eB16.cpp
```

consumed **2073 s of CPU on a single translation unit** while the other 165 had
already finished under `ninja -j 38`. The whole build sat on one straggler
compiling a kernel the hardware rejects.

`configs/skip_fp8_tune_instances_gfx90a.py` removes them. Verified: instance mix
drops to 83 `abI8`, 0 `abF8`.

**Two traps worth recording.** First, that straggler makes a healthy build look
dead — load average falls from 38 to 1.78 with both GPUs at 0%, which is
indistinguishable from a hang unless you check for compiler processes. A run was
killed on that misreading before the pattern was understood. The discriminator:

```
docker exec <c> ps -eo pid,stat,wchan:16,comm    # ninja/hipcc => building
find <jit build dir> -name '*.o' | wc -l         # should be climbing
```

Second, there are **four copies** of `gen_instances.py` in the image and the
live one is `site-packages/aiter_meta/csrc/...`, not `/src/aiter/csrc/...`.
Patching only `/src` reports "patched" and changes nothing. The config script
patches both roots and hard-fails if neither takes.

## 1b — the CK 2-stage MoE cannot be tuned, because it does not exist here

`TUNE_ONLY=cktile` correctly selected `gen_2stages_task` and skipped the ASM
generators, so the gfx90a `get_1stage_file_info()` crash never fired — that part
worked exactly as designed, without a patch. Arch detection was also correct:
every row reports `gfx90a, 104`.

Every shape then failed identically:

```
Error: please check errRatio, stage1 and stage2 should be valid together!
0  gfx90a  104     1  2048  384  ...  -1  -1  -1
0  gfx90a  104  2048  2048  768  ...  -1  -1  -1
[20 rows x 28 columns]     tuner rc=1     tuned rows produced: 0
```

`us1 = us2 = us = -1` on all 20 shapes: **no valid CK 2-stage kernel pair exists
for int8 MoE on gfx90a.** Not "untuned" — absent.

**This corrects `docs/49`.** That document concluded, from
`aiter/fused_moe.py` calling `ck_moe_stage1_fwd`/`ck_moe_stage2_fwd`, that
"vLLM's AITER MoE path goes through Composable Kernel". The call exists; the
kernels do not. Round 55 independently shows Triton `fused_moe_kernel_gptq_awq`
doing the actual work on the AWQ arm. The honest statement is that the MoE path
on this card is **Triton**, and both the ASM tree (`docs/49`) and the CK 2-stage
path (here) are unavailable to it.

## 2 — DP=2 wins on throughput, loses on TTFT (round 54)

| metric | TP=2 | DP=2 (sum) | DP/TP |
|---|---:|---:|---:|
| output tok/s | 210.96 | 225.22 | **1.068×** |
| request/s | 0.82 | 0.88 | 1.073× |
| median TTFT | 996 ms | ~1494 ms | **1.50× worse** |
| median TPOT | 48.41 ms | ~42.9 ms | better |

Real but modest, and it comes with a 50% median-TTFT regression. `docs/39`
item 9 predicted DP "should dominate TP=2 on aggregate throughput". It wins, but
by 6.8%, not the near-2× the collective-overhead arithmetic suggested.

**Why the prediction overshot, stated because the error is instructive.** The
8.25 ms/token of collective overhead (item 5 below) is **per decode step**.
Under concurrency 16, one step emits ~16 tokens, so that cost amortises to
~0.5 ms/token and DP's structural advantage shrinks with it. Batch-1 arithmetic
does not transfer to a batched throughput regime. Any future reasoning from
per-step costs must divide by batch size before predicting a serving result.

DP remains the right answer for pure aggregate throughput on models that fit one
card, and the wrong answer if TTFT matters.

## 3 — AWQ-Int4 is unpack-bound. CONFIRMED (round 55)

Thresholds were fixed **before** the measurement: ~100:1 confirms the unpack
dominates; ~10:1 means the kernel is tight and `docs/39` item 1b is mis-aimed.

```
kernel                        instrs   mfma   ins/mfma
fused_moe_kernel_gptq_awq        836      4      209.0
fused_moe_kernel_gptq_awq        775      4      193.8
fused_moe_kernel_gptq_awq        975      8      121.9
fused_moe_kernel_gptq_awq       1082     16       67.6
fused_moe_kernel_gptq_awq        897     16       56.1
triton_w4a16_gemm_kernel        2050     48       42.7
```

The worst AWQ MoE kernel issues **209 instructions per MFMA — 99.5% of issue
slots are not matrix math.** The unpack histogram names the cost directly:

```
v_bfe_u32          49-65      v_lshlrev_b32   30-34
v_lshrrev_b32      34-54      v_cvt_f32_ubyte0   32
v_and_b32          33-43      v_perm_b32       8-16
```

On a machine `docs/30` shows is issue-bound rather than FLOP-bound, this fully
explains `docs/24`'s anomaly — AWQ-Int4 at **5.07 s vs W8A8's 3.20 s despite
moving half the bytes**. Halving weight traffic cannot pay for 200 extra
instructions per matrix op.

Note the MoE kernels (56–209) are far worse than the dense GEMM (42.7), and this
is a MoE model, so the penalty concentrates exactly where the model spends its
time.

`docs/39` item 1b is **confirmed**: the fix is the unpack, not the format.
W4A16-fp16 keeps the same MFMA rate, gets a better mantissa than W4A8-int8, and
skips per-token activation quantisation. QServe's "three logical operations per
four weights" and QQQ's per-channel shift-by-4 are the specific targets.

## 4 — `pa_fwd_asm` needs no code. `docs/37` §5 is wrong

That document records that `pa_fwd_asm` "has no call site in vLLM" and that
wiring one is real work. Both halves are false in this build:

```
vllm/_aiter_ops.py:2912   def pa_fwd_asm(...)
vllm/_aiter_ops.py:2947   def paged_attention_common(...)
rocm_aiter_fa.py:1318     rocm_aiter_ops.paged_attention_common(...)   LIVE
```

`pa_fwd_asm` carries an explicit note that it is *deliberately* not decorated
with `@if_aiter_supported`, so there is **no architecture gate**. And unlike the
`fmoe` orphans of `docs/49`, `hsa/gfx90a/pa/pa_asm.csv` is populated — 8 rows
including `pa_bf16_noquant_gqa8_1tg_4w.co`, matching `t35-w8a8` exactly
(head_dim 128, GQA 32/4 = 8, bf16).

Two **non-architectural** gates explain the silence:

1. `rocm_aiter_fa.py:1283` reaches the call only under
   `envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`, a plain bool defaulting to False.
2. `aiter/ops/attention.py:_should_use_asm_kernel()` requires
   `num_seqs * num_heads > 2 * cu_num`. On a 104-CU MI210 that is **208**. At
   TP=2 this model has 16 heads per rank, so ASM engages only at
   **num_seqs ≥ 14**.

Gate 2 is why every prior round missed it: **every decode measurement in this
repo is batch-1 single-stream**, giving `1 × 16 = 16` against a threshold of 208.
`paged_attention_ll4mi` — the 13.1% of decode in `docs/45` — is a heuristic
fallback, not a missing kernel.

### Round 56: it engages, and it is slightly faster

Both arms are the same image and model at concurrency 32; only
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` moves.

```
                        HIP        ASM    ASM/HIP
output tok/s         302.26     312.10     1.033x
median TTFT ms      1027.66    1002.40     0.975x
median TPOT ms        76.57      73.24     0.957x

HIP arm:  pa .co objects loaded: <NONE>                      LoadKernel 4
ASM arm:  pa .co objects loaded: pa_bf16_noquant_gqa8_1tg_4w.co   LoadKernel 6
```

**Engagement is proven, not assumed.** The ASM arm loaded exactly the object
predicted for this model and two extra kernels; the control loaded none. Without
that line the throughput delta would be unattributable, since either gate could
have silently declined.

**Read the size honestly.** +3.3% throughput and −4.3% TPOT both point the same
way, but `docs/46` puts this rig's decode bar at **1.036× for a single pair of
arms**, and 1.033× sits just under it. That bar was measured on batch-1
single-stream arms rather than a concurrency-32 throughput run, so it is
indicative rather than binding here — which cuts both ways and is exactly why
this needs a repeat before it is called a win. The TPOT delta (0.957×) is the
larger of the two and the more directly attributable to a decode-path kernel.

**What is not in doubt:** this is the first configuration in this project to
reach AITER's ASM paged attention on gfx90a, it cost one environment variable,
and it did not regress anything. It applies only under concurrency — at
`num_seqs < 14` the heuristic declines and the flag does nothing.

## 5 — low-bit allreduce is dead. Closed by arithmetic

```
bytes per allreduce         4 KiB     (1 token x 2048 hidden x bf16)
allreduces per decode step  96        (48 layers x 2)
bytes per decode step       384 KiB

data movement @25 GB/s      31.5 us = 0.137% of the 23.04 ms decode step

ideal TP=2 (perfect halve)  14.79 ms
actual TP=2                 23.04 ms
shortfall                    8.25 ms/token = 85.9 us x 96 collectives
  of which DATA time is      31.5 us  =  0.38%
```

The collective cost on this fabric is **99.6% fixed latency, 0.4% bandwidth**.
Compressing bf16 → 4 bits saves ~23.6 µs/token = **0.102% of decode**, roughly
**35× below the 3.6% noise floor** (`docs/46`) — unmeasurable even if
implemented perfectly.

`docs/39` item 10 called this "the one unclaimed lever on this fabric". It is
unclaimed because it cannot pay, and the item was written without doing this
multiplication. The correct reading of the same numbers is that the lever is
**eliminating** collectives, not shrinking them — which is item 2.

Caveat: "ideal TP=2 = perfect halving" is generous, so 8.25 ms is an upper bound
on collective overhead. The conclusion survives by three orders of magnitude.

## What this changes

**Closed, do not retry:** low-bit allreduce; tuning the a8w8 GEMM for gfx90a;
tuning the CK 2-stage MoE (no kernels exist to tune).

**Corrected:** `docs/37` §5 (`pa_fwd_asm` has a live call site and no arch gate);
`docs/49` (the MoE path here is Triton, not CK); `docs/39` item 9 (DP wins by
6.8%, not ~2×) and item 10 (dead).

**Deployable today:** `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` on concurrent
workloads. It engages ASM paged attention (proven by object load), measured
1.033× throughput and 0.957× TPOT, cost nothing, and regressed nothing. Repeat
it before trusting the magnitude. It is inert at `num_seqs < 14`, so it does
nothing for single-stream serving.

**The one live target:** the int4 unpack. 209 instructions per MFMA on the
kernel family a MoE model leans on hardest is the largest single inefficiency
this project has measured, and unlike the ASM tree it is ordinary portable code.

## A note on how four of these were found

Three of the five leads were ranked by reading configuration tables — missing
`gfx90a` rows, `cu_num=80` where 104 was needed — and inferring that the absence
was costing performance. In two cases (1a, 1b) the absence was real and cost
nothing: the default config was already optimal, or no kernel existed to
configure. In a third (`docs/49`'s CK claim) a call site was mistaken for an
executing backend.

The two leads that produced something actionable were found by measuring what
actually runs: counting instructions in emitted AMDGCN (item 3) and checking
which `.co` objects a server loads (item 4). **A gap in a config table is a
hypothesis, not a finding.**
