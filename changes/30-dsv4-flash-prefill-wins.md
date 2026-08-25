# 30 — DeepSeek-V4-Flash prefill: indexer Q-reload fix, MLA V-aliasing, FORCE_MMQ

**Date**: 2026-08-25
**Status**: done, validated, deployed
**Branch**: [`mi210-dsv4-flash-prefill`](https://github.com/davetha/llama.cpp-mi210/tree/mi210-dsv4-flash-prefill) · full write-up in [`docs/63`](../docs/63-dsv4-flash-prefill-campaign.md)

## What changed

Three patches to the DSV4-Flash prefill path on 2× MI210 — plus a fourth found afterwards
(see the follow-up at the end). **60K prefill 501 → 545.4 tok/s, 16K 568 → 601.1, 100K
451 → 491**, TTFT@60K 120 → 110.0 s. Cumulative with the prior gather-sparse-attention
work: **362 → 545.4 at 60K (+50.7%)**. Decode unchanged.

Eleven further hypotheses were built and killed with measurements — those, and the
GPU-utilization data that closes out the campaign, are in `docs/63`.

### 1. Lightning-indexer MFMA `K_VECS_PER_BLOCK` 32 → 64 (bit-identical)

`ggml/src/ggml-cuda/lightning-indexer.cu`, CDNA `n_embd==128 && n_head==64` MFMA branch.

The kernel stages the entire Q tile (64 heads × 128 halves = **16 KB per block**) into LDS
and then processes `K_VECS_PER_BLOCK` keys against it. Q — not K — is the dominant
per-block global traffic, and with a 32-key tile that 16 KB load is repeated `n_kv/32`
times for every query. Doubling the kv tile halves the block count in the kv dimension and
therefore halves redundant Q reloads. LDS rises to ~50 KB (within the 64 KB budget);
occupancy is unchanged. `K_VECS=128` does not fit.

**60K 501 → 528 (+5.3%), 100K 451 → 487 (+8.1%).** The gain grows with context because the
indexer is quadratic — the regime that dominates TTFT. Took the indexer from 12.2% to 6.1%
of prefill kernel time.

Correctness: `test-backend-ops -o LIGHTNING_INDEXER` **144/144 pass** (GPU vs CPU reference,
every kv size and K type) — bit-identical, not merely "needle passes".

### 2. Runtime-gated `V_is_K_view` for DKQ=512 (bit-identical) — **upstream candidate**

`ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `ggml_cuda_flash_attn_ext_mma_f16_case`.

Upstream hardcodes `constexpr bool V_is_K_view = DKQ == 576;` — the MLA K-tile-reuse
optimization (read V straight from the already-staged K tile, skipping V's separate load
and LDS staging) is enabled *only* at head-dim 576. But DSV4-Flash's HCA/dense attention
runs at **DKQ=512 and passes the identical tensor as K and V** (`build_attn_mha(q, k_all,
k_all, …)` — two call sites in upstream's own `src/models/deepseek4.cpp`). So V==K, yet
the optimization was off.

Fix: detect aliasing at runtime (identity *or* view-of-K) and select the `true`
instantiation for DKQ=512 only when V actually aliases K. DKQ=576 stays unconditionally
true; every other shape stays false and is byte-identical to before.

Safety, checked explicitly: the multi-stage loading path carries
`static_assert(!V_is_K_view)`, so enabling it where `nstages > 1` would be a **compile
error**. All 20 DKQ=512 configs across every architecture (ampere / turing / volta /
blackwell / cdna / rdna) have `nstages_target = 1`, and `nstages` can only be ≤ that — so
the guarded branch is never instantiated. Safe on NVIDIA as well as CDNA.

**60K 528 → 532, 100K 487 → 491 (+0.8%).** Modest here because fattn-mma is 9.4% of DSV4
prefill; the win scales with how much of a model's attention runs through this path.

Correctness: `test-backend-ops -o FLASH_ATTN_EXT` **2920/2920 pass** (includes all the
non-aliased shapes that must keep taking the `false` path), plus needle 3/3.

### 3. `GGML_CUDA_FORCE_MMQ=ON` (int8 activations — a quality trade)

Build flag, not a code change. Routes the ~14% of prefill spent in rocBLAS projection
GEMMs (`Cijk_*`) through quantized MMQ, which also halves `convert_unary` dispatches
(38k → 16k).

**16K 582 → 589 (+1.1%), 60K 532 → 538.5 (+1.05%)**, decode +0.8%.

This one is **not** bit-identical — it quantizes activations to int8. Needle passes 5/5 at
all depths, and it is the standard llama.cpp quantized path, but it is a real (small)
quality trade and was deployed only after an explicit decision to accept it.

## Also added

- `GGML_DSV4_INDEXER_TOPK` — runtime override for the indexer's top-k, **off by default**.
  Built as a scout for indexer selection-sensitivity; `=384` measures **611 tok/s @16K**
  (the only way found to cross 600 on this hardware) and 560 @60K with needle 5/5, but it
  changes what the model computes and is **not deployed**. A perplexity A/B is the honest
  gate before anyone turns it on.
- Prefill-representative `LIGHTNING_INDEXER` perf cases in `tests/test-backend-ops.cpp`
  (kv 8192/16384 × 1024 queries) — the existing cases top out at kv=256, far too small to
  show Q-reload behaviour.

## Deployed configuration

```
GGML_DSV4_SPARSE_ATTN=1  GGML_CUDA_REGISTER_HOST=1
-ngl 99 -sm layer -np 1 -c 131072 -fa on -t 24 -b 8192 -ub 1024
build: -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DGGML_CUDA_FORCE_MMQ=ON
```

`-ub 1024` was re-swept after FORCE_MMQ and remains the optimum (2048 is −1.6%). Do not
raise it.


## Follow-up (2026-08-25): reverting `MI210_MOE_J`

A fourth change, found after this set was written: the fork's MoE J-tile-selection
heuristic was a **net loss** and has been reverted. Its premise (that CDNA takes the
stream-k branch) does not match the CDNA MMQ config, which runs `stream_k = false`.

**16K 589.3 -> 600.9, 60K 538.6 -> 545.8** (+1.9% / +1.4%), needle 3/3, numerically
neutral. Independently confirmed on pristine upstream, where the heuristic costs 1.4%.
Deployed; live numbers **16K 601.1, 60K 545.4**. See `docs/63` for the full account.
