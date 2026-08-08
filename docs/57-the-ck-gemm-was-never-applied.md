# The CK int8 GEMM was never applied — and tuning it is a null result

Two findings from a session on Qwen3.6-27B W8A8, measured 2026-08-07 on one
MI210 (gfx90a), vLLM `0.26.1rc0+mi210.1`, image `local/vllm-mi210:dsa7`.

1. `configs/enable_aiter_ck_gemm_gfx90a.py` had **bit-rotted** against this vLLM
   and was aborting before it wrote anything. The CK GEMM this project measured
   at 1.662× decode in [`docs/43`](43-ck-int8-gemm-gfx90a.md) was **not running**
   in the deployed image. Applying it end-to-end is worth **2.9–3.5× decode**.
2. Tuning `a8w8_tuned_gemm.csv` for gfx90a — the gap
   [`docs/56`](56-the-tuning-gap-on-gfx90a.md) documents as 0/579 rows — is a
   **null result**. 12 shapes, 1,032 candidates, every one within −1.57%…+0.07%.
   The default heuristic is already right for this card at decode shapes.

## The patch was aborting silently-ish

`enable_aiter_ck_gemm_gfx90a.py` patches three sites. On this image it printed:

```
  patched .../aiter/jit/utils/build_targets.py: gfx90a -> 104 CUs
  patched .../vllm/_aiter_ops.py: is_linear_enabled carve-out
FATAL: register_ops_once anchor matched 0 times in .../vllm/_aiter_ops.py,
       expected exactly 1. Upstream changed the method; re-derive this patch
       rather than forcing it.
```

The second line is a lie of timing, not of fact: the write is atomic, so the
`is_linear_enabled` carve-out that *did* match was **never written to disk**.
`--check` afterwards correctly reported `is_linear_enabled carve-out: not
patched`, which is the only reason this was caught at all.

Cause: vLLM ≥ 0.26.1 decorates the method itself —

```python
    @if_aiter_attention_supported          # upstream, current
    def register_ops_once() -> None:
```

— where the patch expected the older `@if_aiter_supported` + hand-rolled body
guard. Upstream had independently arrived at the same carve-out this script
performs, so the site needs no patch. The anchor matched 0 times and the
assert-on-match-count discipline (correctly) refused to force it.

The fix adds an `_REGISTER_EQUIV` case: when upstream's decorator is present,
treat the site as satisfied and continue rather than abort. `check()` gained the
same case — it was returning exit 1 on a correctly-patched image, which would
have made `verify_gfx90a_image.sh` report a working image as broken.

Verified on a clean `dsa7`: applies, opens both gates, idempotent on re-run,
and `--check` still exits 1 on a genuinely unpatched image.

```
is_linear_enabled(): True
AiterInt8.is_supported(90): (True, None)
```

Server log line that proves it took (this is the one to grep for):

```
INFO [__init__.py:670] Selected AiterInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
```

Without the patch the same line reads `TritonInt8ScaledMMLinearKernel`, and
nothing else in the log differs. That is the whole failure mode: a silent
fallback to the generic Triton kernel that `docs/43` measured at 44% of decode
kernel time.

## What it is worth end-to-end

`Avesed/Qwen3.6-27B-INT8-W8A8` (29.10 GiB, int8 channel weights + int8 per-token
activations), 1× MI210, bf16, `--attention-backend ROCM_AITER_FA`,
`--max-num-batched-tokens 16384`, prefix caching on. Same server, same flags,
only the kernel differs:

| depth | tg256 Triton | tg256 AITER | speedup |
|---|---:|---:|---:|
| 0 | 10.12 | 35.62 | 3.5× |
| 8192 | 10.04 | 33.42 | 3.3× |
| 16384 | 9.92 | 31.35 | 3.2× |
| 32768 | 9.59 | 28.34 | 3.0× |

Prefill gains too — 907.7 → 1,055.5 t/s at d32768 (+17%) — since the same linear
layers carry prefill.

`docs/43` measured this kernel in isolation at **1.662× median decode** on
Qwen3-30B-A3B shapes. End-to-end on a dense 27B it is **2.9–3.5×**. The
difference is that the isolated probe compared kernel-to-kernel, while
end-to-end the Triton fallback was also costing scheduler and fusion overhead.

Full sweep, `build/bench_llamabench_style.py`, 3 reps, mean ± stddev:

| test | t/s |
|---|---:|
| pp2048 | 1814.00 ± 6.65 |
| tg256 | 34.41 ± 1.07 |
| pp2048 @ d8192 | 1462.92 ± 2.32 |
| tg256 @ d8192 | 32.48 ± 0.06 |
| pp2048 @ d16384 | 1195.07 ± 0.20 |
| tg256 @ d16384 | 31.18 ± 0.02 |
| pp2048 @ d32768 | 1055.51 ± 1.62 |
| tg256 @ d32768 | 28.11 ± 0.05 |
| pp2048 @ d65536 | 867.02 ± 1.11 |
| tg256 @ d65536 | 24.40 ± 0.01 |
| pp2048 @ d98304 | 839.04 ± 1.66 |
| tg256 @ d98304 | 20.68 ± 0.02 |
| pp2048 @ d131072 | 726.75 ± 4.80 |
| tg256 @ d131072 | 17.94 ± 0.01 |

Against llama.cpp Q8_0 on the same card and model (`llama-bench`, 1× MI210):
vLLM wins prefill 1.6–2.9× at every depth; decode is 34.41 vs 30.52 at d0 but
crosses over by d16384 and llama.cpp is 31% ahead at d131072. The crossover is
attention-side, not GEMM-side — the GDN layers run Triton/FLA in both engines
and vLLM's per-token overhead grows faster with context here.

> **Measurement hazard, cost one bad run.** With `--enable-prefix-caching`, a pp
> measurement that reuses the same 2048-token delta replays it from cache. A
> first pass reported **19,048 t/s @ d8192** and **7,396 t/s @ d0**. The harness
> now salts the delta seed per process run. Any pp number above ~2,000 t/s on
> this card is cache contamination, not a result.

## Tuning the CK GEMM: nothing there

[`docs/56`](56-the-tuning-gap-on-gfx90a.md) records `a8w8_tuned_gemm.csv` as
0 gfx90a rows / 579 total. Round 52's script tunes the 30B's shapes; this run
tuned Qwen3.6-27B's, at hidden 5120, decode-only M ∈ {1, 2, 8}:

| layer | N | K |
|---|---:|---:|
| qkv_proj | 7168 | 5120 |
| o_proj | 5120 | 4096 |
| gate_up | 34816 | 5120 |
| down_proj | 5120 | 17408 |

`gemm_a8w8_tune.py --compare --update_improved`, FP8 instances skipped via
`configs/skip_fp8_tune_instances_gfx90a.py`, 1,032 candidates across 2 GPUs.
It finished in ~25 minutes, not the 1.5–2 h the round-52 notes imply — the
ccache was warm.

```
Total shapes: 12 | Updated: 0 (improved: 0, new: 0) | Skipped: 12
Threshold: >= 3.0% improvement to update
```

Every shape between **−1.57% and +0.07%**. Representative rows:

| shape (M, N, K) | pre (µs) | post (µs) | Δ |
|---|---:|---:|---:|
| 1, 7168, 5120 | 30.60 | 30.58 | +0.07% |
| 1, 5120, 4096 | 19.42 | 19.69 | −1.42% |
| 1, 34816, 5120 | 146.09 | 146.12 | −0.01% |
| 1, 5120, 17408 | 76.67 | 76.86 | −0.24% |
| 8, 34816, 5120 | 150.90 | 150.79 | +0.07% |

**Why nothing moved.** Compare M=1 and M=8 for one shape: 30.60 vs 33.64 µs.
Eight times the arithmetic for 10% more time. These GEMMs are at the bandwidth
roofline at decode shapes, and tile-config tuning schedules *compute*. An
isolated probe showed the same thing beforehand — `aiter.gemm_a8w8` at
M=1 (0.054 ms) and M=16 (0.060 ms) on 8192×8192 — which is worth taking as the
cheap pre-check before committing a tuning round.

`--update_improved` wrote nothing, so the deployed config is untouched. This
closes the `a8w8_tuned_gemm.csv` question for decode: the gap in that table is
real and does not matter. It says nothing about prefill-shaped M (128, 2048),
which was not tuned here.

## Order of operations

`enable_aiter_ck_gemm_gfx90a.py` still requires `enable_vllm_aiter_gfx90a.py`
first — it reuses the `is_aiter_attention_supported` predicate that script
inserts, and asserts on its absence. And both patches remain necessary but not
sufficient: serving needs `VLLM_ROCM_USE_AITER=1` **and**
`VLLM_ROCM_USE_AITER_LINEAR=1`. A run with the patches and without the flags is
measuring Triton, which is exactly how this went unnoticed.

Grep the server log for `Selected .*ScaledMMLinearKernel` before trusting any
W8A8 number from this box.
