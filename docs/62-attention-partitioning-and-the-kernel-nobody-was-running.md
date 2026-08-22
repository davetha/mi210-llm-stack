# Attention partitioning on gfx90a — and the kernel nobody was running

2026-08-22. Three results. The middle one is a lesson about measurement, not
about kernels, and it is the reason the other two nearly went to waste.

Continues from [docs/61](61-dflash2-and-the-silent-cudagraph-downgrade.md),
which left long-context decode as the open ceiling: 168 tok/s at 2K but 30.6 at
41K and 13.6 at 101K, with base (no-spec) decode collapsing from 33.9 to 6.2 over
the same span. That page blamed the hybrid GDN+attention KV layout for forcing
the Triton decode path. That attribution was **half right, and the wrong half
was load-bearing** — see §1.

Production now runs **178.8 / 92.0 / 49.0 tok/s at 2K / 41K / 101K**: 3.0× at
41K and 3.6× at 101K, same checkpoint, same VRAM envelope, output verified
against the previous config.

## 1. The layout gate was never the blocker

docs/61 identified `has_native_kv_cache_layout` in
`chunked_prefill_paged_decode.py` as what forced hybrid models onto Triton:
stride-padded KV blocks fail the packed-blocks test, `use_custom` goes false.

Two things about that turned out to be wrong.

**The "stride padding" is not padding.** It is an exact 2× K/V interleave.
`GPUModelRunner._update_hybrid_attention_mamba_layout` re-strides the cache to
`(hidden, 2*hidden, ...)` because ROCM_ATTN declares its shape as
`(2, num_blocks, …)`, so `get_kv_cache_block_dim` returns 1. The server prints
the numbers itself once you make it: `key_cache.stride(0)=851968 vs packed
425984`.

**The HIP kernel is not stride-blind.** `csrc/rocm/attention.cu` takes
`kv_block_stride` and `kv_head_stride` as kernel parameters and the host fills
them straight from the tensor:

```cpp
int kv_block_stride = key_cache.stride(0);
int kv_head_stride  = key_cache.stride(1);
```

The earlier claim that it "takes no stride arguments" came from reading the
Python wrapper in `_custom_ops.py`, which simply does not expose them.

> **What is upstream and what is not** (checked 2026-08-22 against
> `vllm-project/vllm@main`, added after this page first went up because the
> original wording invited the wrong assumption):
> `has_native_kv_cache_layout` **is** upstream, and is called from both
> `chunked_prefill_paged_decode.py` and `rocm_attn.py`. But
> `_update_hybrid_attention_mamba_layout` — the function that actually produces
> the 2× interleave — is **fork-local**; upstream contains no `as_strided_` on a
> KV cache anywhere. So upstream carries the *gate* while nothing upstream
> creates the layout it guards against, and the interleave described here is a
> property of this fork, not of vLLM generally.
>
> Likewise the free kernel of §2 does not exist upstream at all, and upstream's
> `use_rocm_custom_paged_attention` admits only `block_size == 16 or 32`, so
> there is no upstream instance of that bug to fix. §2 is a fork-only repair.
>
> None of this touches §3–§4: the partitioning patches do not depend on the KV
> layout, and the collapsing grids they fix are verbatim upstream — with
> `prefix_prefill.py` byte-identical between this fork and upstream/main.

So the layout gate was relaxed behind an env flag and the result measured. **The
patch was inert** — all eight gate probes returned outputs byte-identical to
baseline, because a different check fires first:

```python
# vllm/platforms/rocm.py, in use_rocm_custom_paged_attention()
and (block_size <= 64 or not _ON_GFX90A)
```

with a comment recording a prior measurement: the free paged-attention kernel
*returns incorrect results* on gfx90a for `block_size > 64`. Qwen3.8-27B runs at
block 784. The custom kernel is refused on **numerical-correctness** grounds, and
no amount of layout reasoning changes that.

## 2. Why the free kernel was wrong above block_size 64

Two independent bugs in `paged_attention_ll4mi_QKV_mfma16_free_kernel`, both
taking the slot within a physical KV block from the **partition-local** token
index rather than the global one:

| | expression | correct when |
|---|---|---|
| K | `kphysical_block_offset = klocal_token_idx % block_size` | `block_size` divides `T_PAR_SIZE` (256) |
| V | `v_ptr` folds in `(rowid * VTOKENS_PER_LANE) % block_size`, hoisted out of the `vtoken_depth` loop | `block_size` divides **both** 64 and 256 |

Geometry: `NTHR=256`, `WARP_SIZE=64` → `NWARPS=4`, `TOKENS_PER_WARP=64`,
`VTOKENS_PER_LANE=16`, `VTLOOP=4`. The V base drops both the partition base and
the `vtoken_depth * 64` term.

That predicts the failure set exactly, and **256 is the case that proves the
diagnosis**: it divides 256 but not 64, so it should fail on V alone while 64
passes. It does.

| block_size | before | after |
|---:|---:|---:|
| 16 / 32 / 64 | ok | ok |
| 128 | 1.13 | 3.7e-3 |
| 256 | 1.56 | 4.2e-3 |
| 512 | 2.82 | 4.0e-3 |
| 784 | 2.92 | 4.3e-3 |
| 1024 | 3.21 | 3.8e-3 |

(max relative error vs a float32 reference; clean at head_size 128 and 256.)

Fixed on fork branch `pa-free-blocksize-fix`. `VLLM_ROCM_FREE_PA_LARGE_BLOCK=1`
opts in to `block_size > 64` on gfx90a; the gate stays conservative by default,
because a correct kernel is necessary but not sufficient to widen it.

**A second copy of the K bug lives in the GFX12 (RDNA4) free kernel**, found by
running the patch registry's `obsolete_when` predicate against the patched tree
and noticing it still matched. It is fixed on the same branch but **not tested** —
no RDNA4 card here. It is applied anyway because it is a provable no-op wherever
the current code is already correct: when `block_size` divides 256 the global and
partition-local indices are congruent, so only the already-broken cases change.
It matters more there than on GFX9 — the `_ON_GFX12X` gate branch carries no
`block_size` restriction at all, so large block sizes are admitted rather than
routed to Triton.

## 3. The measurement lesson: production was never running that kernel

With the custom path closed, the remaining move was to fix the Triton fallback.
Its decode grid is `(num_seqs, num_kv_heads)` — single digits of programs on a
104-CU die, one program walking the whole sequence. Adding flash-decoding
partitioning (a kernel accumulating unnormalised softmax numerators plus running
max/denominator, and a reducer) gave, at the kernel level, 16× at 16K rising to
60× at 200K.

Deployed to production: **167.5 / 31.1 / 12.4 tok/s** against a 168 / 30.6 / 13.6
baseline. Nothing.

The same patch, spec decode **off**:

| context | shipped | partitioned | speedup |
|--------:|--------:|------------:|--------:|
| 2K | 33.7 | 26.5 | 0.79× |
| 41K | 5.5 | **28.4** | **5.2×** |
| 100K | 2.6 | **30.1** | **11.6×** |

A 5–12× win that measures as exactly zero in production is not a subtle effect
being swamped. It means the code never ran.

**With speculative decoding on, every decode step has `query_len = num_spec + 1`.**
DFlash2 at n=8 makes that 9. So `max_query_len > 1`, and
`chunked_prefill_paged_decode` routes to `context_attention_fwd` — the *prefill*
kernel — before reaching the decode kernel, which additionally early-returns via
`filter_by_query_len`. The ROCm HIP free kernel does the same thing: it returns
immediately when `query_start_loc[i+1] - query_start_loc[i] != 1`.

So **both** of the preceding sections optimise a path a spec-decode deployment
never touches. The generalisable form: *if a change measures as a no-op, check
whether the configuration reaches that code at all before concluding the change
was worthless.* The no-spec arm is what turned "this patch does nothing" into
"this patch does a great deal, somewhere production does not go."

## 4. The kernel production actually runs

`_fwd_kernel`'s grid is `(batch, head, cdiv(max_input_len, BLOCK_M))`. For a
9-token verify with `BLOCK_M = 32` (784 is not a power of two) the third
dimension is **1** — a handful of programs, each scanning the entire cached
context serially. Exactly the starvation the decode path had, in the kernel that
was actually hot.

Partitioning the cached-context scan there, as a specialised route gated on
`query_len <= 32` so ordinary prefill still goes through `_fwd_kernel` untouched:

| context | baseline | patched | speedup |
|--------:|---------:|--------:|--------:|
| 2K | 168 | 178.8 | 1.06× |
| 41K | 30.6 | **92.0** | **3.0×** |
| 101K | 13.6 | **49.0** | **3.6×** |

Both text profiles improve (list 30.6 → 73.4, prose 6.9 → 17.5 at 41K). Quote the
range, not a point: a separate run at the same length gave 92.0, so 73–92 is the
honest band for 41K list traffic.

With spec decode off, the decode-path patch flattens the curve almost entirely —
26.5 / 28.4 / 30.1 across 2K / 41K / 100K — and the list-vs-prose gap at 41K
closes to 28.3 / 29.1. That gap, which [docs/61](61-dflash2-and-the-silent-cudagraph-downgrade.md)
reported as text-dependence, is substantially attention serialisation.

## 5. Two sizing traps, both silent

**The cudagraph bound.** Decode runs inside full cudagraphs, and
`RocmAttentionMetadataBuilder.build_for_cudagraph_capture()` does
`seq_lens.fill_(1)`. A partition grid sized from the runtime `max_seq_len` is
therefore captured at ~1 partition and **silently truncates the tail of every
long sequence at replay** — wrong answers, not a slowdown. The bound must come
from something fixed for the life of a captured graph; here, the block table's
width times the physical block size, which covers `max_model_len`. Oversized
grids are safe (surplus partitions return without storing, and the reducer
recomputes the partition count from the true per-sequence length). Undersized
ones are not.

**Do not then size for occupancy at that bound.** The bound is 262,640 while
real requests are far shorter, so "enough partitions to fill the device at 256K"
yields ~33 partitions of 8192 — of which **5** are non-empty at 41K. Taking the
finest partition the scratch budget affords gives 40 instead. Surplus partitions
cost one early-exiting program each; empty ones cost the whole win.

## 6. Correctness

Every kernel change was gated against the path it replaces, not against a
description of what it should do.

- Free-kernel fix: custom vs Triton through the real dispatch entry point,
  `block_size` 16…1024 × head_size {128, 256} × seq_len {1024, 5000}. **20 of 32
  cases fail on the unfixed image and all pass on the fixed one** — a test that
  passed everywhere would have proved nothing.
- Both partitioning patches: partitioned vs unpartitioned on identical tensors,
  over ragged batches, zero context, both causal modes, query lengths 1/5/9/17,
  and wide-block-table shapes that emulate a captured graph. Each suite asserts
  the partitioner is *engaged* — a declined route sends both arms down the same
  code and passes regardless.
- Production gate: 8/8, with all four needle-at-depth probes (2K/20K/60K/100K)
  byte-identical to the previous config.

One probe DIFFed: an open-ended prose answer came back "recorded to durable
storage" where the baseline said "recorded in a durable log". Swapping attention
kernels changes reduction order, which can flip a token on open-ended text. That
is why the gate separates semantic checks from digest comparison and treats the
digest as a signal to inspect rather than a verdict.

## What this does not claim

- Production runs these as **two Python files bind-mounted** over the image's
  modules, not as a rebuilt image. That is deliberate — it needs no rebuild — and
  load-bearing: dropping a mount silently costs ~3× long-context decode and
  errors nothing. `launch-qwen38.sh` carries the warning.
- The image built for the free-kernel fix passed tier 0 only (`--no-gpu`). The
  kernel is verified by the test above, not by the repo's hardware tiers, and it
  is **not** what production runs.
- The GFX12 fix is derivation-only. No RDNA4 card was involved.
- TTFT is untouched. 101K still costs ~93 s of prefill, and at agent-sized
  contexts that remains what the wait actually feels like.
