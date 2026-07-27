# Block-scaled FP8 GEMM on gfx90a: it needs a decoder, not a tuning config

**Date**: 2026-07-27
**Hardware**: 2x AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, PyTorch 2.11.0, amd-aiter 0.1.17, vLLM 0.23.0, Triton 3.x, Python 3.14

This answers two questions left open by
[`benchmarks/vllm-aiter-asm-gfx90a.md`](../benchmarks/vllm-aiter-asm-gfx90a.md)
part 3, which found that a block-quantized FP8 checkpoint runs correctly on an
MI210, keeps its 1.75x weight-memory saving, preserves the ASM attention
kernels — and serves 10-15x slower, because the only reachable FP8 GEMM peaks
at 8.3 TFLOP/s against bf16's ~96.

1. Is AITER's `AiterFp8BlockScaledMMKernel` dequant-based, and therefore
   something gfx90a could run, or does it need FP8 arithmetic?
2. That doc concluded the blocker was a missing tuning config, since vLLM names
   the exact missing file. Is that right?

**The answers are "dequant-based, but it is a Triton kernel here, so enabling it
gains nothing structural" and "no".** The gap is not tiling. gfx90a lacks an FP8
*decode* instruction, so Triton emulates `e4m3 -> fp16` in software at ~29 VALU
ops per value, and that emulation — not the matmul — is the kernel.

Replacing the emulation with a 3-op bit reinterpretation, which is **exact** for
every byte a block-quantized checkpoint can contain, makes the same kernel
5.5-6x faster with no tuning at all.

---

## Headline results

1. **AITER's block-scaled FP8 GEMM does not need FP8 hardware.** It loads FP8,
   converts in registers, and does the MFMA in fp16 — `v_mfma_f32_16x16x16f16`,
   with **zero** FP8 or BF8 opcodes anywhere in the generated code. It runs on
   gfx90a and is numerically exact.

2. **But on gfx90a it is not the CK or ASM kernel.** `gemm_a8w8_blockscale`
   checks `_hip_blockscale_supported()`, whose arch list is
   `{gfx940, gfx941, gfx942, gfx950}`, and falls back to *aiter's own Triton
   kernel*. Enabling `AiterFp8BlockScaledMMKernel` on gfx90a therefore swaps one
   Triton kernel for another. It is not a route to hand-written ASM.

3. **The real bottleneck is FP8 decoding, not tiling.** vLLM's kernel is 11,997
   instructions, of which **64 are MFMA** and **7,106 are `v_cmp_ne_u16` /
   `v_cndmask_b32`** — the software emulation of an IEEE-correct `e4m3 -> fp16`
   conversion. gfx942 does this with `v_cvt_pk_f32_fp8`; gfx90a has no such
   instruction.

4. **The emulation cost is specific to e4m3.** Decoding 4 values costs 117 VALU
   ops for e4m3 (both OCP `fn` and AMD `fnuz`), against **11** for e5m2 and 11
   for int8 — both of which gfx90a decodes with a shift. e4m3's 4-bit exponent
   has to be re-biased and its denormals handled; e5m2's exponent field is
   already fp16-shaped.

5. **A bit-trick decode is exact and removes the storm.** 11,997 instructions
   drop to 3,954 and the cmp/cndmask count drops from 7,106 to 194, with the
   same 64 MFMA doing the same arithmetic. Bit-exact on all 254 non-NaN e4m3
   byte patterns.

6. **Tuning helps too, but far less than the decode fix**, and the two compose.
   Tuning alone takes the stock kernel from 12.6x slower than bf16 to 4.1x; the
   decode fix *untuned* already beats the fully tuned stock kernel; together
   they reach 0.59-1.78x of bf16.

7. **In serving this is worth 8.6-10.7x.** `Qwen/Qwen3-14B-FP8` on one MI210
   goes from 2.7 to **28.9 tok/s** at 128 tokens / concurrency 1, and TPOT from
   373 ms to 34 ms. FP8 now serves at ~0.73x of bf16 throughput while using
   1.75x less weight memory — where it was 0.07x before.

8. **At decode, FP8 is faster than bf16**, which the shared-181-TFLOP/s argument
   does not predict: decode is bound by streaming weights, not by arithmetic,
   and FP8 weights are half the bytes. At M=1 `gate_up_proj` the fast kernel
   runs at **0.59x of bf16's time**.

---

## 1. What does AITER's FP8 block GEMM actually dispatch to?

vLLM's `AiterFp8BlockScaledMMKernel.apply_block_scaled_mm`
(`vllm/model_executor/kernels/linear/scaled_mm/aiter.py:401`) picks between two
aiter entry points:

```python
if self.use_triton:
    gemm_a8w8_blockscale_op = rocm_aiter_ops.triton_gemm_a8w8_blockscale
else:
    gemm_a8w8_blockscale_op = rocm_aiter_ops.gemm_a8w8_blockscale
```

`use_triton` requires `is_triton_gemm_w8a8_tuned(n, k)`, a hardcoded list of 11
`(N, K)` pairs — none of which are Qwen3-14B's — so a Qwen3-14B FP8 checkpoint
takes the second branch.

That branch is not the CK kernel either. In
`aiter/ops/gemm_op_a8w8.py:775`:

```python
if not _hip_blockscale_supported():
    # No CK code object for this arch -> triton
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale as _gemm_a8w8_blockscale_triton,
    )
    ...
    return _gemm_a8w8_blockscale_triton(xq, wq, x_scale, w_scale, dtype=dtype)
```

with

```python
_BLOCKSCALE_HIP_PREBUILT_ARCHES = frozenset({"gfx940", "gfx941", "gfx942", "gfx950"})
```

Measured on the card:

```
aiter get_gfx()             : gfx90a
prebuilt arches             : ['gfx940', 'gfx941', 'gfx942', 'gfx950']
_hip_blockscale_supported() : False
```

`module_gemm_a8w8_blockscale` is also absent from aiter's prebuilt `.so` set —
only the non-blockscale `module_gemm_a8w8.so` ships — so there is no gfx90a code
object to load, exactly as the fallback comment says.

**So both branches of `AiterFp8BlockScaledMMKernel` land in Triton on gfx90a.**
Opening the gate PR #12 closed would not expose a hand-written FP8 kernel,
because none exists for this arch. That is the direct answer to the question
that motivated the gate.

### The CK kernel behind that gate really does need FP8 hardware

The arch list is not conservatism. CK picks its matrix instruction by operand
type, and for `f8_t` the selection has no CDNA2 case
(`composable_kernel/include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp:1576`):

```cpp
template <>
constexpr auto GetMfma<f8_t, 16, 16, f8_t, true, false>()
{
#if defined(__gfx12__)
    return MfmaInstr::wmma_f32_16x16x16_f8f8_gfx12;
#elif defined(__gfx11__)
    return MfmaInstr::wmma_unsupport_16x16_gfx11;
#else
    return MfmaInstr::mfma_f32_16x16x32f8f8;
#endif
}
```

Everything that is not RDNA3/RDNA4 falls through to `mfma_f32_16x16x32f8f8`.
Asking the assembler directly:

| instruction | gfx90a | gfx942 |
|---|---|---|
| `v_mfma_f32_16x16x32_fp8_fp8` (what CK selects for `f8_t`) | **REJECTED** | OK |
| `v_mfma_f32_32x32x16_fp8_fp8` | **REJECTED** | OK |
| `v_cvt_pk_fp8_f32` (the FP8 decoder) | **REJECTED** | OK |
| `v_mfma_f32_16x16x16f16` (what Triton actually emits) | **OK** | OK |

So the honest answer to "is AITER's FP8 block GEMM dequant-based or
FP8-hardware-dependent" is **both, depending on which implementation**:

* the **CK and ASM** implementations are genuinely FP8-hardware-dependent and
  cannot be built for CDNA2 at all;
* the **Triton** implementation — the only one gfx90a can reach — is
  dequant-based and runs fine.

Which is why enabling the AITER kernel on gfx90a is not worth a gate change: it
would route to Triton, the same class of kernel vLLM already uses.

### It does not need FP8 hardware

Called with an explicit config (the arch's config file is missing — see below),
aiter's Triton blockscale kernel runs and is exact:

```
M=64    N=5120   K=5120    RAN  mean_rel_err=0.00000  CORRECT
M=256   N=7168   K=5120    RAN  mean_rel_err=0.00000  CORRECT
M=4096  N=5120   K=5120    RAN  mean_rel_err=0.00000  CORRECT
```

Disassembling the code it generated (`llvm-objdump -d --mcpu=gfx90a` on the
`.hsaco` Triton emitted, the same technique used to build
[`19-aiter-operator-port-matrix.md`](19-aiter-operator-port-matrix.md)):

```
_gemm_a8w8_blockscale_kernel ... GRID_MN_2560 ...
    MFMA instructions : {'v_mfma_f32_16x16x16f16': 128}
    FP8/BF8 opcodes   : NONE
```

No `v_cvt_pk_fp8_f32`, no `v_mfma_*_fp8_*`. The kernel loads FP8, decodes to
fp16 in registers, and multiplies with the fp16 matrix cores gfx90a has had
since CDNA1. **The design is dequant-based, exactly as hypothesised — a
weight-only FP8 GEMM never needed FP8 arithmetic.**

### Why it still fails out of the box

```
AssertionError: Required config file doesn't exist:
  .../aiter/ops/triton/configs/gemm/gfx90a-GEMM-A8W8_BLOCKSCALE.json
```

Only `gfx1201` ships one. This is a hard assert, not a warning — which is why
the kernel is unreachable rather than merely slow.

---

## 2. The bottleneck is the FP8 decoder, not the tiling

The 128 MFMA above sit inside **11,604 instructions**. vLLM's own kernel is the
same shape:

| kernel | total | MFMA | `v_cmp_ne_u16` + `v_cndmask_b32` |
|---|---:|---:|---:|
| `_w8a8_triton_block_scaled_mm` (vLLM) | 11,997 | 64 | 7,106 |
| `_gemm_a8w8_blockscale_kernel` (aiter) | 11,604 | 128 | 6,984 |

A `v_cmp`/`v_cndmask` pair is a branchless conditional select. Thousands of them
in a GEMM inner loop is not addressing arithmetic; it is a decision tree. It is
the IEEE-correct `e4m3 -> fp16` conversion, open-coded: one branch per
infinity, NaN and denormal case, per value.

Isolating it in a kernel that does nothing but convert confirms the cost and
pins it to the format:

| source dtype | VALU ops to decode 4 values | decoded with |
|---|---:|---|
| `e4m3fn` (OCP) | 117 | software emulation |
| `e4m3fnuz` (AMD) | 118 | software emulation |
| `e5m2` (OCP) | **11** | `v_lshl_or_b32` — a shift |
| `int8` | **11** | `v_cvt_f16_i16_sdwa` — hardware |

Two things follow.

**AMD's `fnuz` format does not help.** It drops inf and negative zero, which is
where one might expect the saving to come from, but Triton's lowering is
essentially identical — 118 ops against 117. Requantizing a checkpoint to fnuz
would buy nothing.

**The problem is e4m3's exponent, not FP8 as such.** e5m2 has 5 exponent bits
and bias 15 — byte-identical in layout to fp16's exponent field — so decoding it
is a left shift. e4m3 has 4 exponent bits and bias 7, so every value needs
re-biasing and the denormal boundary moves.

This corrects the conclusion in
[`benchmarks/vllm-aiter-asm-gfx90a.md`](../benchmarks/vllm-aiter-asm-gfx90a.md),
which read the missing-config warning and inferred that tuning was the blocker.
The warning is real and tuning does help, but an 11.6x gap was never going to be
a tiling problem: the kernel was spending ~99% of its instruction budget
decoding its inputs.

---

## 3. The fix: decode e4m3 with three instructions

The expensive conversion is expensive because it is exact for inputs that
block-quantized weights cannot contain. Reinterpreting the bits is nearly free:

```
e4m3fn:  s eeee mmm          (exponent bias 7)
fp16:    s eeeee mmmmmmmmmm  (exponent bias 15)

h = ((u & 0x80) << 8) | ((u & 0x7f) << 7)
```

This puts the 4 exponent bits at 13:10 and the 3 mantissa bits at 9:7, producing
a valid fp16 that carries bias 15 where the value assumed bias 7. The result is
therefore exactly `2^-8` times the true value — and exactly so for denormals
too, since an e4m3 denormal (exponent field 0) maps to an fp16 denormal and the
same factor applies. Folding `2^8` per operand back in — one multiply by 65536
on the fp32 accumulator, per output element, outside the K loop — restores the
true product.

Verified over the entire input domain, not sampled:

```
exact on all 254 non-NaN bytes: max_abs_diff=0.0
NaN bytes [127, 255] decode to [480.0, -480.0] (expected, unused)
```

The two NaN patterns decode to +-480.0 rather than NaN. Block-quantized weights
contain no NaN, and the dispatch predicate refuses anything that is not a plain
e4m3 `[128,128]` block-scaled GEMM with fp32 scales.

### What it costs and what it buys

| kernel | instructions | MFMA | cmp/cndmask |
|---|---:|---:|---:|
| `_w8a8_triton_block_scaled_mm` | 11,997 | 64 | 7,106 |
| bit-trick decode | **3,954** | 64 | **194** |

Same 64 MFMA instructions, same arithmetic, same operand layout, same launch
config — 3x fewer instructions and 36x less conditional-select traffic.

Timed on an MI210 (free card; the second MI210 hosts an unrelated model and was
verified idle), Qwen3-14B projection shapes, output checked against a
dequantized reference on every call:

| M | N | K | bf16 us | stock Triton us | bit-trick us | speedup vs stock | vs bf16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7168 | 5120 | 88.1 | 1134.0 | **204.3** | 5.6x | 2.3x slower |
| 32 | 5120 | 5120 | 71.6 | 1142.6 | **183.7** | 6.2x | 2.6x slower |
| 256 | 5120 | 5120 | 145.4 | 2250.0 | **322.5** | 7.0x | 2.2x slower |
| 4096 | 5120 | 5120 | 2570.8 | 27975.7 | **5005.1** | 5.6x | 1.9x slower |

Mean relative error against the dequantized reference is `1e-6` at every shape —
**identical to the stock kernel's own error**, which is the point: this is not a
precision-for-speed trade.

Re-checked on real `Qwen/Qwen3-14B-FP8` tensors rather than synthetic Gaussians,
since the only data-dependent risk is that operands 2^-8 smaller reach fp16's
denormal range:

| projection | N | K | fast vs fp32 ref | stock vs fp32 ref | fast vs stock |
|---|---:|---:|---:|---:|---:|
| `layers.0.self_attn.q_proj` | 5120 | 5120 | 0.001408 | 0.001408 | 0.000001 |
| `layers.20.mlp.gate_proj` | 17408 | 5120 | 0.001410 | 0.001410 | 0.000002 |
| `layers.39.mlp.down_proj` | 5120 | 17408 | 0.001404 | 0.001404 | 0.000002 |
| `layers.39.self_attn.o_proj` | 5120 | 5120 | 0.001407 | 0.001407 | 0.000001 |
| `layers.5.mlp.down_proj` | 5120 | 17408 | 0.001408 | 0.001408 | 0.000004 |

The 0.0014 is bf16 output rounding and is the same for both kernels; the two
disagree with each other by at most `4e-6`. The denormal concern is real but
immaterial: values that underflow are below 2^-15 of the block maximum, and a
block is scaled so its maximum is 448.

The remaining ~2x against bf16 is the honest ceiling area. On CDNA2 FP8 and bf16
share one 181 TFLOP/s peak, so FP8 can never *beat* bf16 here; the goal is to
match it while holding 1.75x less weight memory.

---

## 4. Does tuning close the gap?

Partly, and it composes with the decode fix rather than substituting for it.

Full config sweep (480 configurations: `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`,
`num_warps`, `num_stages`), every configuration checked against a dequantized
reference before being timed, winner re-checked before its number is quoted.
All times in microseconds on one MI210, Qwen3-14B projection shapes.

| M | layer | bf16 | vLLM stock | vLLM tuned | aiter stock | aiter tuned | fast stock | **fast tuned** | fast tuned / bf16 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | qkv_proj | 90.7 | 1140.4 | 372.6 | 880.3 | 204.5 | 207.2 | **112.5** | 1.24x |
| 1 | o_proj | 71.0 | 1136.1 | 211.6 | 865.6 | 165.1 | 204.6 | **78.5** | 1.11x |
| 1 | gate_up_proj | 451.7 | 3258.6 | 876.7 | 2393.5 | 779.6 | 510.6 | **265.5** | **0.59x** |
| 1 | down_proj | 228.3 | 3675.7 | 685.9 | 2675.7 | 487.7 | 670.2 | **242.0** | 1.06x |
| 32 | qkv_proj | 97.3 | 1149.9 | 506.0 | 1040.7 | 295.3 | 191.8 | **151.7** | 1.56x |
| 32 | o_proj | 77.5 | 1143.4 | 379.0 | 1032.5 | 225.5 | 186.8 | **117.8** | 1.52x |
| 32 | gate_up_proj | 475.3 | 3284.2 | 1194.6 | 3485.0 | 1107.1 | 506.0 | **363.1** | **0.76x** |
| 32 | down_proj | 250.4 | 3704.3 | 1116.4 | 3297.1 | 674.5 | 603.9 | **373.0** | 1.49x |
| 4096 | qkv_proj | 3254.0 | 39078.5 | 14932.9 | 31301.3 | 25977.0 | 6317.4 | **5766.9** | 1.77x |
| 4096 | o_proj | 2343.6 | 27950.6 | 10830.1 | 22301.7 | 18727.6 | 4553.2 | **4169.1** | 1.78x |

Reading it:

* **Tuning alone recovers 2.6-5.4x** and leaves FP8 at 3.0-4.6x slower than
  bf16. Real, and worth having, but not the answer.
* **The decode fix alone, untuned, beats the exhaustively tuned stock kernel**
  at every shape — 207.2 vs 372.6 at M=1 `qkv_proj`. That is the cleanest
  statement of which problem was the big one.
* **Together they reach 0.59-1.78x of bf16**, from 11.4-16.1x.
* **aiter's kernel is consistently faster than vLLM's** once tuned (204.5 vs
  372.6 at M=1 `qkv_proj`), because its config schema exposes `NUM_KSPLIT` and
  its winner splits K 8 ways. It is still 1.8x behind the decode fix, and it is
  a Triton kernel either way.

### FP8 beats bf16 at decode, which the arithmetic argument does not predict

"FP8 cannot beat bf16 on CDNA2 because they share one 181 TFLOP/s peak" is
correct about arithmetic and wrong about decode. At M=1 the GEMM is not
arithmetic-bound at all, it is bound by streaming the weights — and FP8 weights
are half the bytes. Hence 0.59x at `gate_up_proj`, the largest weight matrix
in the model.

Adding split-K to the fast kernel (partial sums into an fp32 buffer via global
atomics) pushes this further at decode, most at the layers with the longest
reduction:

| layer | M | bf16 | fast tuned | fast + split-K | best NUM_KSPLIT | weight-stream bound |
|---|---:|---:|---:|---:|---:|---:|
| qkv_proj | 1 | 91.8 | 112.5 | **95.1** | 8 | 22.9 |
| o_proj | 1 | 72.2 | **78.5** | 83.8 | 16 | 16.4 |
| gate_up_proj | 1 | 451.2 | **265.5** | 283.2 | 4 | 111.4 |
| down_proj | 1 | 229.7 | 242.0 | **131.9** | 16 | 55.7 |

`down_proj` has K=17408, the longest K in the model, and gains 1.8x. The
"weight-stream bound" column is the time to read the FP8 weights once at
1.6 TB/s — still 2-4x below what any of these kernels achieve, so decode
headroom remains.

Split-K is **not** in the shipped patch. It needs an fp32 scratch buffer, global
atomics (so summation order stops being deterministic) and a `NUM_KSPLIT` key
that vLLM's stock config schema does not carry. The simple kernel already
delivers the bulk of the win; split-K is the obvious next step, with the numbers
above to justify it.

### The activation quantizer is not a second bottleneck

`per_token_group_quant_fp8` encodes bf16 -> e4m3, the direction that also lacks
hardware. It is a flat ~17.5 us at decode sizes (launch-bound, not
conversion-bound) — 2.8-15% of the GEMM and 0.2-1.0% at prefill. vLLM uses a
custom HIP op there, not a Triton FP8 cast, so it never hit the emulation path.

---

## 5. End-to-end: does it show up in serving?

Yes — about 9-11x. Same harness, model and server flags as
[`benchmarks/vllm-aiter-asm-gfx90a.md`](../benchmarks/vllm-aiter-asm-gfx90a.md)
part 3, so the rows are directly comparable. `Qwen/Qwen3-14B-FP8`, one MI210,
`stock` attention config in both cases, so the GEMM is the only difference.

| prompt | conc | FP8 before | **FP8 after** | gain | bf16 (ASM attn) | FP8 after / bf16 |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 2.7 tok/s | **28.9** | **10.7x** | 39.7 | 0.73x |
| 128 | 8 | 19.6 tok/s | **168.8** | **8.6x** | 198.1 | 0.85x |
| 4096 | 1 | 2.1 tok/s | **18.7** | **8.9x** | 27.4 | 0.68x |

TPOT was pinned at 373-383 ms in every FP8 cell before — the signature of a
kernel doing no useful work. It is now **34.0-36.0 ms**, against bf16's 25-26 ms,
and it moves with load again.

The bf16 column had ASM attention enabled and FP8 here did not, so 0.68-0.85x
understates the remaining gap slightly in FP8's favour on attention and
overstates it on nothing. The honest summary is that FP8 now serves at roughly
three-quarters of bf16 throughput while using **1.75x less weight memory
(15.71 vs 27.52 GiB) and holding 1.40x more KV cache (240,992 vs 172,000
tokens)** — which is a trade worth making, where 0.07x was not.

---

## 6. Verdict

**The FP8 question on CDNA2 is not closed by hardware.** The earlier conclusion —
that FP8 works, keeps its memory saving, but is unusable for serving — was
correct as a measurement and wrong about the cause. It was not the missing
tuning config, though that config was genuinely missing and worth ~3x. It was
that gfx90a has no FP8 *decoder* instruction, so Triton emulated `e4m3 -> fp16`
at ~29 VALU ops per value and the GEMM spent ~99% of its instruction budget
unpacking its inputs.

Three instructions of bit manipulation, exact over every byte a block-quantized
checkpoint can contain, remove that. With the tuned configs it takes FP8 serving
from 0.07x of bf16 to 0.73x, and takes the decode GEMM itself to **below** bf16
on the largest projections.

On the AITER question specifically: **leave the gate closed.** Not because the
kernel needs FP8 hardware — the one gfx90a can reach does not — but because on
this arch it resolves to a Triton kernel that the patched vLLM path already
beats by 1.8x. `AiterFp8BlockScaledMMKernel` would be worth reaching on gfx942,
where it is CK or ASM. Here it is neither.

### What shipped

* [`configs/enable_fast_fp8_dequant_gfx90a.py`](../configs/enable_fast_fp8_dequant_gfx90a.py)
  — the kernel and its dispatch predicate, gated to ROCm + `on_gfx9() and not
  on_mi3xx()` + e4m3fn operands + fp32 scales + `[128,128]` blocks. Anything
  else falls through to the stock kernel untouched. `--check` and `--revert`
  round-trip to a byte-identical file.
* [`configs/mi210-fp8-block-configs/`](../configs/mi210-fp8-block-configs/) —
  tuned configs in vLLM's on-disk format, for the four Qwen3-14B shapes. Copy
  into `vllm/model_executor/layers/quantization/utils/configs/`. These are tuned
  for the fast kernel; without the patch they are still valid, just not optimal.
* [`benchmarks/bench_fp8_fast_dequant_gfx90a.py`](../benchmarks/bench_fp8_fast_dequant_gfx90a.py)
  — the four-way comparison and config sweep that produced the tables above.
* [`benchmarks/probe_fp8_convert_cost_gfx90a.py`](../benchmarks/probe_fp8_convert_cost_gfx90a.py)
  — the convert-only measurement that pins the cost to e4m3.

### Not done

* Split-K, measured and worth 1.8x more on `down_proj` at decode (above).
* Shapes beyond Qwen3-14B have no tuned configs; the sweep script generates them.
* The `M=4096` row covers `qkv_proj` and `o_proj` only — the sweep was stopped
  before the two largest prefill shapes, which are the least interesting case
  (prefill is compute-bound, where FP8 provably cannot beat bf16).
