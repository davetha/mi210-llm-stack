# The low-level knobs, measured

Eleven leads, run after `docs/50` closed the previous five. **Seven closed, one
is a pattern-coverage gap rather than a performance result, and the largest
finding is not software at all.** Two harness defects are recorded here as
findings in their own right, because each would have produced a confident and
wrong entry in this repo.

## Summary

| # | lead | result |
|---|---|---|
| 1 | `_should_use_asm_kernel` threshold | **AMD's constant is correct** — do not change |
| 2 | B-preshuffle W8A8 GEMM | **dead** — FP8-only path |
| 3 | int8 rmsnorm+quant fusion | **pass matches 0 patterns** — coverage gap |
| 4 | `fmha_v3` config selection | **nothing to tune** — complete cross-product |
| 5 | AITER Triton op swaps | **mostly dead** — MLA-only or mxfp4/fp8-only |
| 6 | fp8 KV cache | **dead** — `Unsupported dtype: torch.float8_e4m3fn` |
| 9 | clocks under load | **power-limited at 200 W of a 300 W maximum** |
| 11 | AITER codegen flags | no env hook; needs a source patch |
| 7, 8, 10 | DP+shuffle, roofline, env knobs | not yet run |

## 9 — the largest lever is the power cap, and it is not software

Everything else here fights for 1–3%. This is 50% of a power budget.

```
card2  power1_cap = 200W    power1_cap_max = 300W
card3  power1_cap = 200W    power1_cap_max = 300W
```

Sampled every 20 s through round 57's two-hour sweep:

```
09:05:11   junction 67/68C   sclk 1375/1350MHz   power 199/200W
09:06:11   junction 69/71C   sclk 1255/1250MHz   power 199/200W
09:08:52   junction 69/71C   sclk 1360/1355MHz   power 199/200W
```

Under sustained load the cards sit at **1235–1375 MHz against a 1700 MHz boost**,
at 67–71 °C, with power pinned at **exactly 199–200 W in every sample**. Flat
power with headroom on temperature is the signature of a **power cap**, not a
thermal limit.

**Not changed, deliberately.** These are passive MI210s whose cooling depends
entirely on chassis fans driven by `gpu-fan-control.service`. Raising the cap by
50% raises the thermal load by roughly the same, and whether that fan curve has
the headroom is not determinable from `rocm-smi`. It is also a failure mode that
degrades hardware slowly rather than erroring loudly. Recommended sequence:
verify the fan curve above 71 °C, raise 200 → 250 W, run a sustained *prefill*
load, and watch whether clocks actually rise. Prefill should benefit more than
decode — prefill is compute-bound, while decode is memory-bound and `mclk` is
already at its 1600 MHz maximum.

**A correction:** an earlier note in this session reported "full 1700 MHz boost,
no throttling". That came from a brief sample taken at low load between
requests. It was wrong about the sustained case.

## 1 — the ASM paged-attention threshold is correctly calibrated

`aiter/ops/attention.py:_should_use_asm_kernel` ends with
`num_seqs * num_heads > 2 * cu_num`, i.e. 208 on a 104-CU MI210. `docs/50` left
open whether that factor of 2 — calibrated by AMD on CDNA3 — transfers to CDNA2.

`configs/asm_pa_threshold_gfx90a.py` adds `AITER_PA_ASM_FORCE=1|0` after the
`head_size != 128` capability check, making the heuristic overridable. Two
servers, six concurrency points each:

| conc | heads | HIP tok/s | ASM tok/s | ASM/HIP | TPOT ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 7.60 | 7.54 | 0.992× | 1.011× |
| 2 | 32 | 16.97 | 16.81 | 0.991× | 1.010× |
| 4 | 64 | 31.33 | 31.27 | 0.998× | 1.005× |
| 8 | 128 | 58.96 | 59.03 | 1.001× | 1.002× |
| 16 | 256 | 105.26 | 106.58 | **1.013×** | 0.997× |
| 32 | 512 | 177.27 | 178.64 | **1.008×** | 0.992× |

**Monotonic crossing between 64 and 128 heads.** ASM is genuinely slower below
it and faster above. Individual points sit inside the 1.036× noise bar, but the
monotonic trend through six points is the signal, and it says the heuristic
declines ASM exactly where ASM loses. AMD's constant is slightly conservative
and substantively right.

Validity proven both directions: `rd57-asm` loaded
`pa_bf16_noquant_gqa8_1tg_4w.co`, `rd57-hip` loaded none.

**Item 1 closes. Do not change the constant.** The carve-out is kept as an
instrument, not a setting.

> **QUALIFIED 2026-08-02 — see `docs/52`.** This was measured entirely at TP=2,
> and the conclusion holds only for that shape. Round 61 ran the same kernel
> under TP=1 (each replica owning all 32 heads instead of 16) and measured it
> worth **~6%**, five times the 1.008–1.013× seen here. Head count does not
> explain the difference — conc 32 at TP=2 and conc 16-per-replica at TP=1 both
> give 512 heads. The likely cause is Amdahl: at TP=2 the ~96 host-staged
> collectives per decode step contribute 8.25 ms/token of fixed cost that
> dilutes any attention-kernel gain, and removing them makes attention a much
> larger share of what remains. So the constant is correctly calibrated *for the
> TP=2 configuration it was tested in*; the ASM kernel's value depends on what
> else competes for the decode step.

## 3 — the int8 fusion pass matches nothing

`docs/46` established that vLLM's `rms_quant_fusion.py` registry is FP8/FP4 only
(zero int8 keys), that AITER has the kernel (`_aiter_ops.py:723` asserts
`quant_dtype in [torch.int8, FP8_DTYPE]` and calls
`aiter.rmsnorm2d_fwd_with_dynamicquant`), and that the pass is gated behind
`rocm_aiter_ops.is_enabled()`. It carved that gate out, **validated it safe**,
then measured `fuse_allreduce_rms` — which closed on shape grounds — and never
set `fuse_norm_quant`.

Setting it, with both arms on the same image so the flag is the only variable:

```
RocmAiterRMSNormQuantFusionPass  Replaced 0 patterns     <- BOTH arms
RocmAiterRMSNormQuantFusionPass  completed in 0.1 ms
```

**The pass runs and matches nothing.** This is a *pattern-coverage* finding, not
a performance one: the ROCm fusion pass does not recognise a W8A8 Qwen3-MoE
graph. The likely reason is that int8 activation quantization happens inside the
compressed-tensors linear layer's `apply_weights`, not as a graph node adjacent
to the rmsnorm, so there is no `rmsnorm → int8_quant` pair to rewrite.

So the gap `docs/46` identified is real and remains open, but closing it needs a
**new pattern**, not a flag. Nothing in the current stack can fuse it.

### The second gap, unmeasured

`compilation/passes/fusion/act_quant_fusion.py` has the same shape:

```python
FUSED_OPS = { kFp8StaticTensorSym, kNvfp4Dynamic, kFp8Dynamic128Sym, kFp8Dynamic64Sym }
```

Zero int8. AITER's `silu_and_mul_quant` is dtype-generic, but vLLM only wraps
`act_mul_and_**fp8**_group_quant`. For a MoE model this may matter more than the
norm gap — silu+mul fires on every expert activation (~384/token at 8 experts ×
48 layers) versus ~96 for the norms. There is no ROCm act+quant fusion pass
analogous to `RocmAiterRMSNormQuantFusionPass`, so this would need code written,
not enabled.

## 6 — fp8 KV cache is unavailable

```
AssertionError: Unsupported dtype: torch.float8_e4m3fn
  Worker_TP0/TP1, ~2.5 minutes into engine init
```

Note the flavour: **`e4m3fn`** (CUDA/OCP), not **`e4m3fnuz`** (AMD CDNA, per
`aiter/utility/dtypes.py`). vLLM accepts `--kv-cache-dtype fp8` at config time
and logs *"Using fp8 data type to store kv cache"*, then fails in worker init.

The reason is a **dtype-flavour rejection, not "no FP8 MFMA"** — KV cache is
storage, not compute, so the absence of fp8 matrix instructions would not by
itself have blocked it. That distinction only became available by capturing the
actual error.

Two related claims made earlier in this session were **wrong** and are withdrawn:

- *"Quantized KV short-circuits the ASM batch heuristic, giving ASM at batch 1."*
  It does not. `int8` is not in vLLM's `CacheDType` at all, and `high_precision`
  is never passed by vLLM (it stays at AITER's default of 1), so **both** early
  returns in `_should_use_asm_kernel` are unreachable from vLLM.
- The bandwidth half of the lead dies with the dtype rejection.

`turboquant_k8v4` (FP8 keys + 4-bit values, 2.6× reduction, +1.17% PPL) remains
the only route to a smaller KV footprint, but it uses its own backend and would
cost the AITER FA prefill win.

bf16 baseline, for reference: 902,176 KV tokens; 50.4 / 150.8 / 233.9 tok/s at
concurrency 1 / 8 / 32.

## 2, 4, 5, 11 — closed

**2. B-preshuffle GEMM is FP8-only.** Three independent signals: the vLLM class
is `AiterPreshuffledPerToken**Fp8**ScaledMMLinearKernel(FP8ScaledMMLinearKernel)`
(`scaled_mm/aiter.py:128`); its `can_implement` gates on
`current_platform.fp8_dtype()`; and the CK instances are templated on
`DeviceGemmHelper**F8**Flatmm`. Our int8 path uses
`AiterInt8ScaledMMLinearKernel` at `:31` and never touches it. The zero gfx90a
rows in `a8w8_bpreshuffle_tuned_gemm.csv` are correct and harmless.

*This was called wrong twice before being settled* — first guessed FP8, then
"corrected" to int8 on the strength of the function name
`preshuffled_per_token_w8a8_gemm`. `w8a8` there is generic 8-bit-weight/
8-bit-activation; the concrete dtype is fp8. The same
"config-table gap is a hypothesis, not a finding" trap `docs/50` named.

**4. `fmha_v3` has nothing to tune.** The 24 gfx90a rows are a complete
cross-product: 2 head dims (128, 192×128) × 2 masks (causal, non-causal) ×
2 modes (batch, group) × 3 bf16 rounding modes (rtz, rtna, rtne) = 24. Those
suffixes are *rounding modes*, not tile configs, so the row is determined by
problem shape plus a numerics choice — there is no performance heuristic that
could mis-select, and no tuner exists because there is nothing to tune. Observed
in use: `fwd_hd128_bf16_rtna_group.co` and its causal sibling.

**5. AITER's Triton op surface is mostly unusable here.** `kv_cache` is
MLA-only (`cat_and_cache_mla`), `gather_kv_b_proj` likewise, and `activation`'s
two quant fusions are mxfp4/fp8-only. Only `fused_silu_mul`, `softmax`, `topk`
and `gmm` remain as candidates. vLLM already wires AITER Triton from *other*
subpackages (`attention.mha_v3`, `unified_attention`, `mha`, `triton_rope_and_cache`).

**11. The codegen flags have no env hook.** They are hardcoded at
`aiter/jit/core.py:862-876` (`--amdgpu-kernarg-preload-count=32`,
`--lsr-drop-solution=1`, `-enable-post-misched=0`,
`-amdgpu-early-inline-all=true`). Changing them requires patching source and
rebuilding, which puts this below its already-low confidence.

## Two harness defects, recorded as findings

Both would have written a confident and wrong claim into this repo.

**Writing the conclusion into the instrument.** Round 58 printed *"fp8 KV is
unavailable on gfx90a"* from a string hardcoded in the script, on a startup
failure whose root cause it had not captured (it tailed 25 lines, caught only
the API-server wrapper, and its cleanup trap deleted the container). Round 59
counted *its own explanatory prose* as fusion evidence, because it grepped the
round log — which contains that prose — instead of the container log. The fp8
guess happened to be right; the fusion one was not checkable at all. **A grep
that can match your own narration is not a measurement.**

**A ~7.6% concurrency-1 ordering bias.** Round 59b's two arms were functionally
identical (the pass replaced 0 patterns in both), yet:

```
conc 1    OFF 33.26    ON 30.72    0.924x
conc 8   OFF 255.02   ON 254.69    0.999x
conc 32  OFF 418.91   ON 418.01    0.998x
```

Round 59 produced **the same 0.924×** at concurrency 1 (30.61/33.12) with a
different image pair. Two independent rounds, identical artifact, second arm
always slower, and only at concurrency 1 — where `num_prompts = 1 × 8 = 8` is
too few to amortise warmup.

**Any single-pair concurrency-1 comparison in this harness carries a ~7.6%
ordering bias.** It does not affect round 57 (monotonic across six points,
driven by conc ≥ 8), but it is exactly the defect that manufactures a
"regression" or a "win" out of arm ordering, and low-concurrency arms need more
prompts or interleaved ordering before they can be trusted.

## What is left

- **The power cap** — 200 W of 300 W, pending a thermal-headroom check.
- **The act+quant fusion gap** — needs a pattern written, not a flag.
- Items 7 (DP + shuffle_kv combined), 8 (rocprof-compute roofline), 10 (env knob
  sweep), not yet run.
