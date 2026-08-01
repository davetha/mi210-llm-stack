# Rounds 40–41: the CK int8 GEMM is worth 1.48× decode, and it was three gates away

**Date**: 2026-08-01 · Model: Qwen3-30B-A3B W8A8 · Image:
`vllm-mi210:v0.26.1rc0-ckgemm-warm` · Patch:
`configs/enable_aiter_ck_gemm_gfx90a.py` · Scripts:
`benchmarks/matrix/probe_a8w8_ck_gfx90a.py`, `round40_ck_gemm_ab.sh`,
`round41_ck_profile.sh` · Results: `benchmarks/matrix/results/rd40-*.json`

`docs/42` put the ~3× decode gap **in-kernel** and named the target:
`scaled_mm_kernel` at 44% of decode kernel time, which is vLLM's *generic
Triton fallback*. This is the follow-through. vLLM already ships a better int8
GEMM for ROCm — `AiterInt8ScaledMMLinearKernel` → `aiter.gemm_a8w8_CK` — and it
was unreachable on gfx90a for three independent reasons, none of them a
hardware limit.

**Result: decode 55.71 → 82.48 tok/s, 1.480×.** The largest single win measured
in this project — larger than AITER flash attention (1.19–1.33× prefill),
larger than PCIe P2P, an order of magnitude larger than three vLLM versions.

## 1. The measurement

Round 40, one image, one variable (`VLLM_ROCM_USE_AITER_LINEAR`), TP=2, 27,852-token
requests:

| workload | metric | Triton | AITER CK | factor |
|---|---|---:|---:|---:|
| longctx | **decode** | 55.71 | **82.48** | **1.480×** |
| longctx | prefill | 6904.30 | 6980.34 | 1.011× |
| longctx | ttft | 3.74 | 3.69 | 0.987× |
| cold16k | prefill | 8411.77 | 8555.25 | 1.017× |
| cold16k | ttft | 1.80 | 1.77 | 0.982× |

Decode is steady to ±0.1 across three reps (55.7/55.7/55.7 vs 82.4/82.5/82.5)
at matched prompt sizes, and **both arms pass the correctness probe** with
coherent output — this is not a fast-wrong kernel. Prefill barely moves, which
is the expected shape: prefill runs at large M where both kernels saturate the
card; decode runs at M=1 where they do not.

At TP=1 (round 41, offline, 1024 tokens) the same flag is worth **1.708×**
(20.12 s → 11.78 s). The win is *larger* without tensor parallelism, because
TP=2's per-layer collectives are a fixed cost the GEMM speedup cannot touch.

### Proving the arm ran what it claims

vLLM does **not** log which int8 kernel it selected —
`choose_scaled_mm_linear_kernel()` returns the first supported entry silently
(`kernels/linear/__init__.py:566-577`). So "the flag was set" is not evidence.
Both rounds assert on AITER's module-import line instead:
`module_gemm_a8w8` must be **present in the CK arm and absent in the control**.
That assertion earned its place immediately — it caught round 40's first
attempt, below.

## 2. Why it was unreachable: three gates, none of them hardware

**Gate 1 — the build.** `aiter.gemm_a8w8_CK` fails to JIT-build with
`fatal error: 'gemm_a8w8_manifest.h' file not found`, which is a missing
*code-generation* step, not a compile error. Instance codegen resolves the
current arch's CU count through `GFX_CU_NUM_MAP` in
`aiter/jit/utils/build_targets.py`, that table holds only gfx942 / gfx950 /
gfx1250, and the lookup raises `RuntimeError: Unknown gfx 'gfx90a'` — so
nothing is generated and the compile dies later on the absent header. The
module already knows the architecture (`build_targets.py:11` has
`1: "gfx90a"`); only the CU table omits it. MI210 reports **104 CUs**.

The sharp edge, and why this is worth reporting upstream: that lookup runs
**even when the tuned CSV has no rows for your architecture**
(`a8w8_tuned_gemm.csv` is 553 gfx950 rows + 26 gfx942 + zero gfx90a), so an
arch that should fall back to default kernels gets a hard error *instead of the
fallback*. Bypassing the CSV proves the templates are fine on CDNA2:
generation then emits 9 instances, a manifest and a lookup header, with this
project's `AITER_A8W8_NO_FP8` guard applied. Related-but-different reports
exist for the same class of bug ([ROCm/aiter#1415], [#1552] — arch missing from
a mapping dict) and for gfx90a builds generally ([#179], fp8 kernels); the
CU-map path appears unreported.

**Gate 2 — selection.** `AiterInt8ScaledMMLinearKernel.is_supported()` gates on
`rocm_aiter_ops.is_linear_enabled()`, decorated `@if_aiter_supported` →
`on_mi3xx()` = gfx942|gfx950. `enable_vllm_aiter_gfx90a.py` deliberately
narrowed its carve-out to **attention only**, so linear was blocked by our own
patch as well as upstream's.

**Gate 3 — registration, found the hard way.** With gates 1 and 2 open, serving
still died at the first `qkv_proj`:

```
AttributeError: '_OpNamespace' 'vllm' object has no attribute 'rocm_aiter_w8a8_gemm'
```

`register_ops_once()` carries `@if_aiter_supported` too, so on CDNA2 **not one
AITER custom op is ever registered**. The attention carve-out never tripped
over this because the ASM flash-attention backend calls into aiter directly
rather than through `torch.ops.vllm`; the linear path is the first thing here
to need a registered op. Registration only *declares* ops — every impl imports
its aiter symbol lazily inside the call — so declaring MI300-only ops on gfx90a
costs nothing and cannot run them; their own `is_*_enabled()` gates stay
`on_mi3xx()`.

Ordering matters: this patch **must** run after `enable_vllm_aiter_gfx90a.py`,
whose `is_aiter_attention_supported` predicate it reuses, and it asserts that
rather than producing a subtly broken file.

## 3. The mechanism: Triton leaves 60–85% of the card idle at M=1

Round 41 profiled both configurations and diffed the decode-window kernel
tables. Because the window is a fixed 8 s and CK fits more tokens into it,
every *unchanged* kernel scales by the token ratio 1.708× — `fused_moe`
1158.6→1947 (predicted 1979), `paged_attention` 639.8→1045 (predicted 1093),
`add_rmsnorm_quant` 186.3→327 (predicted 318). That cross-check is what makes
the GEMM row readable:

| kernel | Triton | CK |
|---|---:|---:|
| `scaled_mm_kernel` | 3977.1 ms | **gone** |
| `ck::kernel_gemm_xdl_cshuffle_v3_multi_d` | — | **1218.6 ms** |

Same GEMM calls per token in both (102 and 107 = 48 layers × qkv+o_proj), so
nothing structural changed. Per call: **Triton 95.7 µs, CK 16.3 µs — 5.87×.**

The grid sizes say why:

| | qkv_proj (N=5120) | o_proj (N=2048) |
|---|---:|---:|
| Triton workgroups | **40** | **16** |
| CK workgroups | **80** | **32** |

On a **104-CU** card at M=1, Triton's tile heuristic emits 40 and 16
workgroups — most of the GPU is idle on every decode step. CK's default
instance uses narrower N-tiles and emits twice as many. This is an occupancy
failure at batch 1, not an arithmetic one, which is also why prefill (large M,
both kernels saturating) shows nothing.

### The part that does NOT reconcile — stated, not smoothed over

Isolated microbenchmarks of the two kernels do **not** reproduce the in-model
ratio. At the served shapes they measure Triton 42–44 µs and CK 32–36 µs, i.e.
**1.17–1.35×**, against 5.87× in the model. Two hypotheses were tested and
both failed:

- *TP=2's narrower per-rank shapes favour CK more* — **false**, CK's edge is
  *smaller* at TP=2 (o_proj 1.516× → 1.162×).
- *the served weight layout (a transposed view, since the Cutlass base stores
  `weight.t()`; `is_weak_contiguous` accepts transposed) penalises Triton* —
  **false**, the as-served layout is if anything *faster* for Triton
  (o_proj 57.9 → 42.2 µs).

What remains is partly explicable — the isolated CK number includes ~17 µs of
Python/dispatch overhead per call that CUDA-graph replay hides in the model
(33 µs isolated vs 16.3 µs in-graph is consistent), and `docs/42`'s clock
sampling shows decode does not hold peak SCLK the way a tight GEMM loop does —
but that does not add up to 5.87× and **is not claimed to**. The end-to-end
number is the measurement; the microbenchmark is the thing that fails to
predict it. Anyone extending this should trust the served A/B over the isolated
kernel timing.

## 4. Round 40's first attempt, and the fourth sighting

The first round-40 run **failed**, correctly: the CK arm crashed during load on
gate 3 above, and the `module_gemm_a8w8` assertion refused to report its
numbers. Recorded because the alternative — a crashed arm silently reported as
a null — is this project's most repeated failure.

Also fixed in passing: `VLLM_ROCM_USE_AITER_LINEAR=0` was **hardcoded** in
`serve_vllm_aiter.sh`, which would have run both arms on Triton and reported
the null as a result. That is the **fourth** instance of a hardcoded value in an
arm-launch path (after `VLLM_IMAGE` in round 31, `NCCL_P2P_DISABLE` in round
32, `VLLM_PREFER_AITER_FA` in round 37). Every one of them fails by making two
arms silently identical, which is why each produced a plausible number rather
than a crash.

## 5. What this changes

- **Serving**: apply `configs/enable_aiter_ck_gemm_gfx90a.py` and set
  `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_LINEAR=1`. Both are required;
  the patch alone is inert.
- **`docs/42`'s scoreboard**: the in-kernel lead is no longer just a direction.
  44% of decode kernel time was Triton leaving the card idle, and reclaiming it
  is worth 1.48× at TP=2 / 1.71× at TP=1.
- **Still open, and now more attractive**: `a8w8_tuned_gemm.csv` has no gfx90a
  rows, so all of the above is CK's *default* instance selection. AITER ships
  `gemm_a8w8_tune.py`. Round 34's lesson (a partial tuned config was 0.79×)
  says tuning must cover the served shapes or be left alone — but a full tune
  is the obvious next lever.
- **`fused_moe_kernel`** is now the largest decode kernel by a wide margin. It
  is the next target, via the same question this round asked: is vLLM running a
  generic fallback where an AITER kernel exists?

[ROCm/aiter#1415]: https://github.com/ROCm/aiter/issues/1415
[#1552]: https://github.com/ROCm/aiter/issues/1552
[#179]: https://github.com/ROCm/aiter/issues/179
