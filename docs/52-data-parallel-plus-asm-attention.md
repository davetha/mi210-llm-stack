# DP=2 + ASM paged attention: 1.118×, and why it beat its parts

The best throughput result this project has measured, and the reason it works is
not the reason either half was expected to work.

## The result

`round61_dp2_plus_asm_pa.sh`, `t35-w8a8`, 4096-token prompts, aggregate
throughput at matched total offered load:

| total conc | TP=2 | DP=2 | DP+ASM | DP/TP | DP+ASM/TP | **ASM gain** |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 224.68 | 225.59 | 240.58 | 1.004× | 1.071× | **1.066×** |
| 32 | 403.11 | 425.56 | 450.76 | 1.056× | 1.118× | **1.059×** |

Latency, measured here rather than inherited from `docs/50`:

| | TTFT | TPOT |
|---|---:|---:|
| TP=2 | 1003.63 ms | 73.48 ms |
| DP=2 | 1534.18 ms | 68.54 ms |
| **DP+ASM** | **1496.53 ms** | **65.23 ms** |

**Engagement proven, not assumed.** `dp2` replicas loaded no `pa` objects;
`dp2asm` replicas loaded `pa_bf16_noquant_gqa8_1tg_4w.co`. Replica balance is
within 0.6% on all four pairs (112.86/112.73, 212.66/212.90, 120.33/120.25,
226.09/224.67), so summing the two replicas is legitimate rather than hiding a
lopsided split.

## The three-arm design, and why two arms would have been useless

`docs/50` round 54 measured DP=2 at 1.068× over TP=2 — but it never set
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`, and `rocm_aiter_fa.py:1283` gates the whole
`paged_attention_common` call on it. So its DP replicas could not reach the ASM
path at all. Its number is DP-only with ASM switched off.

Running `dp2asm` against `tp2` alone would therefore have confounded two
changes. The middle arm exists so the **ASM gain** column can be computed
against a DP baseline measured in the same session on the same hardware.

## Why it beat its parts

`docs/51` round 57 measured ASM paged attention at TP=2 across six concurrency
points and found it worth **1.008–1.013×** — inside the noise bar, and the round
closed with "AMD's `2 * cu_num` constant is correct, do not change it."

Here the same kernel is worth **~6%**. Five times more.

**Head count does not explain it.** The gate is
`num_seqs * num_heads > 2 * cu_num` = 208. At TP=2, conc 32 gives
32 × 16 = 512 heads. At TP=1, conc 16-per-replica gives 16 × 32 = 512 heads.
Identical head counts, gains differing by 5×.

The explanation that fits is **Amdahl**. At TP=2 a decode step carries ~96
host-staged collectives at ~86 µs each (`docs/51` §5), which is 8.25 ms/token of
fixed cost that no attention kernel can touch. Any improvement to attention is
diluted by that denominator. At TP=1 the collectives are gone entirely, so
attention becomes a much larger share of what remains and the same kernel
speedup shows up proportionally bigger.

**So the two changes compound rather than add**, and that is only visible
because they were measured together. `docs/51`'s conclusion needs qualifying:
AMD's threshold constant is correct *for the TP=2 shape it was measured in*, not
universally — the ASM kernel's value depends on what else is competing for the
decode step.

## The tradeoff, corrected

`docs/50` characterised DP as "a throughput answer only — it cannot help
single-request latency." The TTFT half holds: 1.49× worse. But **TPOT improves
11%** (73.48 → 65.23 ms), and inter-token latency is what a streaming client
actually experiences after the first token.

So the honest characterisation is:

- throughput **1.118× better**
- per-token latency **0.888× (better)**
- time-to-first-token **1.49× worse**
- KV cache per replica **halved** — each card holds the full weights

Better on both axes that matter for batch and offline work. For interactive
serving it is a genuine trade: 1.0 s → 1.5 s to first token, against 11% faster
streaming thereafter. Only applies to models that fit one card; TP stays forced
above that.

## Deployment

Two changes, no code:

1. Two single-card engines at `TP=1`, one per GPU via `HIP_VISIBLE_DEVICES`,
   behind the existing llama-swap/litellm router.
2. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` on both.

Gate 2 of `_should_use_asm_kernel` still applies per replica: at TP=1 each owns
all 32 heads, so ASM engages from `num_seqs >= 7`. Below that the replica falls
back to the HIP kernel and this reduces to plain DP.

## Round 60: the env knobs, for completeness

One real result of four (`round60_env_knob_sweep.sh`):

| arm | c8 | vs base | c32 | vs base |
|---|---:|---:|---:|---:|
| `GPU_MAX_HW_QUEUES=4` (stock) | 132.96 | 1.000× | 382.61 | 1.000× |
| `queues=1` | 132.95 | 1.000× | 382.09 | 0.999× |
| `queues=8` | 132.40 | 0.996× | 382.32 | 0.999× |
| **`HSA_ENABLE_SDMA=0`** | **117.43** | **0.883×** | **343.57** | **0.898×** |
| `--block-size 32` | 132.49 | 0.996× | 384.75 | 1.006× |

**`HSA_ENABLE_SDMA=0` costs 10–12%**, consistent across both points and far
outside the 1.036× bar. Not a win — SDMA is on by default — but it is the first
direct evidence that the host-staged collective path genuinely rides the DMA
engines. With no P2P on this box, all ~96 collectives per decode step go through
SDMA, and disabling it makes you pay.

`GPU_MAX_HW_QUEUES` is **inert** across 1/4/8.

**`--block-size 32` did not answer its question.** The intent was to check
whether block size 32 falls off the `pa` ASM path, since `pa_asm.csv` carries
block-size 16. Round 60 never set `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT`, so the
ASM paged-attention path was unreachable in every arm — the `.co` lines show
only `fmha` prefill objects, which KV block size does not affect. Still open.

## Method notes

**Concurrency 1 was deliberately excluded from round 60.** `docs/51` measured a
~7.6% second-arm ordering bias at that point in this harness (two functionally
identical arms differing by 0.924× at conc 1 while agreeing to 0.2% at conc 8
and 32, reproduced across two independent rounds). Including it would have
handed every arm after the baseline a fake penalty.

**Round 61's summary table appeared empty in an early inspection** — the data
rows begin with digits and matched none of the patterns in the `grep` used to
read the log. The script was correct; the reading of it was not. Worth
recording because the same mistake nearly produced a "the analysis block is
broken" entry here.
