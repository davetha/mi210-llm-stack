# DFlash2 depth × context × text profile, after the attention fix

2026-08-22.

Question: now that attention is partitioned, does DFlash2 still earn its keep at
long context? Five arms (n = 0/1/2/4/8), three context lengths, two text
profiles, **3 replicates per cell**, all on identical v3 kernels so drafter depth
is the only variable. [`spec_sweep.py`](spec_sweep.py) and [`code_profile.py`](code_profile.py).

Decode tok/s, median of 3, TTFT excluded (timed from first content token;
token count from `usage.completion_tokens`, never from SSE chunks):

| n | 2K list | 2K prose | 41K list | 41K prose | 101K list | 101K prose |
|--:|--------:|---------:|---------:|----------:|----------:|-----------:|
| 0 | 37.1 | 37.1 | 36.0 | 36.0 | 32.4 | 32.4 |
| 1 | 49.6 | 43.2 | 24.1 | 20.6 | 13.3 | 11.3 |
| 2 | 62.1 | 44.5 | 31.9 | 21.5 | 17.1 | 12.1 |
| 4 | 91.1 | 47.6 | 54.8 | 27.6 | 33.5 | 17.3 |
| **8** | **186.5** | **56.2** | **98.9** | 27.9 | **56.5** | 18.2 |

Third profile, real code generation, n=8 only (see "why no n=0 arm" below):

| context | n=8 | vs n=0 | acceptance | tok/draft |
|--------:|----:|-------:|-----------:|----------:|
| 2K | 138.6 | 3.73x | 0.703 | 5.62 |
| 41K | 76.9 | **2.14x** | 0.735 | 5.88 |
| 101K | 43.3 | **1.33x** | 0.735 | 5.88 |

### Findings

**1. Without a drafter, decode is text-independent to the decimal.** n=0 gives
37.1/37.1, 36.0/36.0, 32.4/32.4 for list/prose. Every bit of text-dependence in
this system is draft acceptance, nothing else. That is also why the code profile
needs no n=0 arm — the existing n=0 row is the reference for any profile.

**2. n=8 is optimal among drafters at every single cell.** No intermediate depth
wins anywhere. This is the opposite of the old MTP ladder (where n=2 was the
robust pick) and it follows from the drafter architecture: DFlash2 is
block-diffusion, so one pass proposes the whole block and the drafter's cost is
roughly independent of n. Shallow n pays a full long-context drafter pass to buy
one or two tokens, which is why **n=1 at 101K is 0.41x of no-spec despite 1.000
acceptance**. Acceptance was perfect and it still lost — the cost is the pass,
not the misses.

**3. Tokens accepted per draft SATURATES on prose.** 0.74 / 0.85 / 1.59 / 1.66
for n=1/2/4/8. On list it scales cleanly: 1.00 / 1.59 / 3.98 / 7.91. Drafting
deeper on unpredictable text does not yield more usable tokens, it only widens
the verify. That is the whole mechanism.

**4. On genuinely unpredictable prose at long context, spec decode is a net
loss**: 27.9 vs 36.0 at 41K (0.78x) and 18.2 vs 32.4 at 101K (0.56x). No depth
fixes it — n=0 beats all four drafter arms at both lengths.

**5. For the traffic this box actually serves, DFlash2 still wins**, because code
sits near the list regime (acceptance 0.70-0.74, ~5.9 tokens/draft), not the
prose one: **2.14x at 41K and 1.33x at 101K**. The honest caveat is that 1.33x
is a much thinner margin than the ~5x DFlash2 was worth before the attention
fix, and it keeps shrinking with context.

### Answer to "does DFlash2 help at large context?"

Yes for code, and the margin narrows sharply with length (3.73x -> 2.14x ->
1.33x). No for prose, where it is actively harmful beyond ~40K. Before the
attention work it was a flat ~5x regardless of context or profile; partitioning
attention removed most of what it was compensating for.

**Recommendation: keep n=8.** The only configuration that would beat it is
spec-decode-off for a long-context prose workload, which is not what this box
does.

### Method notes

- The driver's own `config check` prints "none" for every arm: it greps
  `"num_speculative_tokens"` against `docker inspect .Args`, where quotes are
  JSON-escaped as `\"`, so it can never match. It was verified as broken, not
  trusted. `arm_watch.sh` recorded the true depth and mount count per container
  start as an independent audit trail; all five arms confirmed correct.
- `pkill -f arm_watch.sh` self-matches its own ssh command line and kills the
  connection before the shell can report. Match on `arm[_]watch.sh` instead.
- Do not splice numbers across probes. An earlier prose comparison mixed
  `probe_prose.py` with `ctx_probe.py` and produced 0.60x where the
  internally-consistent figure is 0.78x. Different prose prompts have different
  acceptance; only compare within one probe.
