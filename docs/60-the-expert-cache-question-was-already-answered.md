# The MoE expert-cache question was already answered — and the model it was asked about is gone

A recurring proposal — *"trace the router, simulate an LRU/LFU expert cache,
then implement one"* — arrived again on 2026-08-09. It should not be run. The
study exists, it was run on this hardware three days earlier, and it came back
negative. Separately, the production deployment the proposal describes no longer
exists on this box.

This document imports that result into the hub so the next person greps
`ledger.jsonl` and stops, and corrects two errors in
[`guides/moe-expert-cache-vllm.md`](../guides/moe-expert-cache-vllm.md) that
would otherwise be used as experimental ground truth.

## 1. The routing study was already run

`mi210-vllm/docs/ROUTING.md`, measured **2026-08-06** on 2× MI210 against
`glm52-int4int8`. It is the whole of "phase 1: collect routing traces" and most
of "phase 2: replay an offline cache simulator":

- `build/route_probe_sitecustomize.py` instruments the MoE router and records
  per-layer expert-id selections, bind-mounted into both workers via
  `PYTHONPATH`.
- `build/drive_routing.py` drives a mixed workload — English, code, math,
  Chinese, Spanish, German.
- `build/analyze_routing.py` replays it: per-layer Gini, normalized entropy,
  static top-K coverage, and an **LRU replay over the decode stream** with the
  prefill and decode streams separated and K swept.

The result:

| metric | value | meaning |
|---|---|---|
| per-layer normalized entropy | ~0.91 | near-uniform over 256 experts |
| per-layer Gini | ~0.51 | moderate skew |
| top-10% of experts per layer | ~42% of accesses | a hot set exists, does not dominate |
| top-20% | ~60% | diminishing |
| LRU vs static top-K | tracks it closely | temporal locality is real, and small |

GLM-5.2 activates **8 of 256 experts per token**, so a perfect cache of the
hottest ~26 experts per layer still misses most accesses — and after the offload
window there is only **~2.6 GiB of VRAM free per card**, which is a handful of
experts, not a cache. Verdict, quoted: *"caching will not help. The cost is
reading 8 experts × ~76 layers per token over PCIe, regardless of which experts
they are."*

The same document also closes n-gram speculation on that arm at **0.92×
overall** — worst, at 0.85×, on exactly the repetitive prompts where it should
win — for the same reason: nothing is resident, so every accepted draft still
streams its experts over the bus.

## 2. The MiMo deployment the proposal targets is gone

The proposal is framed on *"25 of 48 expert layers execute on the CPU."* That
placement is real and is still described in this repo's README, but as of
2026-08-09 it is not what the box runs:

- `/models/mimo-v25/Q4_K/…` does not exist. No MiMo weights remain on
  `/mnt/llm-storage` (largest occupants are now `glm52-gguf-q4km` 434 GB,
  `glm52-int4int8` 378 GB).
- llama-swap is not running. `CURRENT-SERVING.md` and the live litellm config
  show the served set as **coder / deephat / glm-5.2 / laguna**, routed
  `open-webui -> litellm -> llama-server`, with no MiMo entry.

So the *specific* claim the cache proposal optimizes — a static 25/48 CPU split
that "is what fits" — describes a configuration that cannot currently be
measured, while the model that replaced it has already been traced and returned
near-uniform routing.

## 3. Two errors corrected in the guide

`guides/moe-expert-cache-vllm.md` was written against vLLM 0.25 and has since
gone wrong in two ways that matter, because it reads as ground truth:

- It documents `--moe-expert-cache-size N` as a **shipped "vLLM 0.25+"
  feature**. It is not. Current vLLM `main` exposes only UVA and a group/layer
  `PrefetchOffloader` in `vllm/config/offload.py`; the expert cache is still
  **open PR [vllm-project/vllm#37190](https://github.com/vllm-project/vllm/pull/37190)**
  (opened 2026-03-16, last touched 2026-08-05, unmerged). That PR is also
  BF16/limited-FP8, `--enforce-eager`, no EP>1, and synchronous per-miss H2D —
  unusable as-is for a Q4_K GGUF path and hostile to graph-mode decode.
- It repeats `VLLM_USE_AITER=0` with the rationale *"AITER rejects gfx90a — only
  targets gfx942/gfx950."* This repo has spent twenty documents disproving that:
  `docs/19`, `docs/18`, `docs/35`, `docs/43`, `docs/57`. Following the guide
  today turns off the CK int8 GEMM that `docs/57` measures at **2.9–3.5×
  decode**.

Both are fixed in this change, with the guide re-headed as historical.

## 4. What the proposal got backwards

Two claims were offered as measured successes that this repo records as
failures. Correcting them here because they will be cited again:

- **"vLLM `PrefetchOffloader` already demonstrated staged H2D expert movement
  and strong prefill."** It did not run. `docs/29 §2`: it crashed in
  `ncclAllReduce` and hung the harness for an hour.
  `benchmarks/matrix/round18_prefetch_eager.sh` is the `--enforce-eager` retry
  of that crash; its header opens "WHAT FAILED". The 695.9 t/s figure in those
  headers is a target string, not a result.
- **"Selective UVA already measured direct GPU access to pinned host weights and
  showed the opposite prefill/decode tradeoff."** No such measurement exists.
  `round20_uva_retest.sh` states predictions, not results. The only UVA-family
  result is `docs/29 §1`: XNACK unified memory VM-faulted at weight load and
  wedged the kernel workers at load average 70. `round15_unified_mem.sh` refuses
  to run without `ALLOW_UVM_HANG=1`. `docs/29` says the amortization argument is
  "untested, not refuted" — which is still true, and is not the same as
  supporting a cache.

## 5. What is actually still open

Narrow, and worth stating so the line closes cleanly rather than being
re-proposed a third time:

- **Does a Q4_K llama.cpp MoE route more skewed than GLM-5.2 did?** GLM-5.2's
  near-uniform 8-of-256 routing is a property of that model, not a law. A model
  with fewer experts and higher top-k could concentrate. This is a *port of
  existing tooling* — `route_probe_sitecustomize.py` +
  `analyze_routing.py` already do the collection and the LRU/top-K replay — not
  a new simulator project, and it needs a resident MoE model to point at.
- `docs/31 §5` already frames placement as an optimization problem and names
  Fiddler (expert popularity) and ProMoE (prediction) as the principled
  versions. That framing stands; what it lacked was a measurement, and the
  measurement now exists and is negative for the cache.

**Do not build a cache before a trace on the actual served model shows a hot set
that GLM-5.2 did not have.** Entropy 0.91 is the number to beat.
