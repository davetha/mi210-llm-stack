# Which quantization formats can reach gfx90a's matrix cores

**Date**: 2026-07-28 · **Method**: assembler acceptance + vLLM kernel-registry gates
**Related**: `docs/24` (measurements), `docs/26` (what to download), `docs/19` (ASM port matrix)

The question behind this document: **W8A8 wins, but is it the only format that
wins, or just the only one tested?** Specifically — is there another quantization
scheme that reaches `v_mfma_i32_16x16x16i8`, or that benefits from the translated
AITER ASM kernels?

Two answers, and they are different in kind:

- **The AITER ASM kernels are not coupled to W8A8 at all.** They are bf16
  *attention*. They already help every format.
- **W8A8 is the only format that reaches quantized arithmetic on this chip** —
  not by a tuning gap, but because int8 is the only sub-16-bit matrix dtype
  gfx90a has, and because no ROCm-reachable vLLM kernel feeds int8 activations to
  a mixed-precision GEMM.

---

## 1. The ASM kernels are quantization-independent

This is worth stating first because it is the more useful of the two answers and
it is easy to get backwards.

The 242 kernels that survived translation break down like this (`docs/19`):

| Family | Portable | What it does |
|---|---:|---|
| `fmha_v3_bwd` | 138 | attention backward (training) |
| `fmha_v3_fwd` | **48** | **attention forward — this is the one that matters** |
| `topksoftmax` | 22 | MoE routing |
| `mla` | 11 | MLA attention (no MLA model in the matrix) |
| `pa` | 8 | paged attention |
| `fmoe` | 8 | MoE |
| (top-level) | 7 | misc |
| `i8gemm` | **0 of 9** | **INT8 GEMM — none translated** |
| `bf16gemm` | 0 of 22 | bf16 GEMM |
| `fmoe_2stages` | 0 of 186 | MoE |

The kernels that actually load at runtime are

```
fwd_hd128_bf16_causal_rtna_group.co
fwd_hd128_bf16_rtna_group.co
```

— `bf16`, forward attention. **Attention consumes activations and the KV cache,
not quantized weights.** Nothing in that path knows or cares how the FFN weights
were packed.

The measurement already proves it, and it is easy to misread: the **+12.8%** from
enabling AITER was measured on **AWQ-Int4**, not on W8A8.

| Arm | TTFT | |
|---|---:|---|
| AWQ-Int4, stock vLLM | 5.72 s | |
| AWQ-Int4 + AITER ASM | **5.07 s** | **+12.8%** |

So the ASM work is a free win on **every** vLLM arm whose model has
`head_dim ∈ {128, 192}`. The one thing that gates it is head dimension, not
format — and Qwen3-Next's `head_dim = 256` gets nothing, because AITER ships no
`hd256` fmha ASM for any architecture (`docs/25` item 7).

**Note also that `i8gemm` is 0 of 9 portable.** W8A8's win did *not* come from
AITER ASM GEMM — the server log says `Using TRITON Int8 MoE backend`. The two
findings are independent and were nearly conflated.

---

## 2. int8 is the only sub-16-bit matrix dtype on gfx90a

Settled by assembling each candidate instruction and asking whether the target
accepts it — the same method `docs/19` used for the port matrix, and the same one
that caught the bad `D3E1 → D3CD` patch.

```bash
llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding
```

| Instruction | gfx90a | gfx942 |
|---|---|---|
| `v_mfma_i32_16x16x16i8` | **ok** `0xd5d3` | rejected |
| `v_mfma_i32_32x32x8i8` | **ok** `0xd4d3` | rejected |
| `v_mfma_i32_16x16x32_i8` | rejected | ok |
| `v_mfma_f32_16x16x16f16` | **ok** | ok |
| `v_mfma_f32_16x16x16bf16_1k` | **ok** `0xe7d3` | ok `0xe1d3` |
| `v_mfma_f64_16x16x4f64` | **ok** | ok |
| `v_mfma_i32_16x16x32i4` | **rejected** | rejected |
| `v_mfma_i32_32x32x16i4` | **rejected** | rejected |
| `v_mfma_i32_4x4x4i4` | **rejected** | rejected |
| `v_mfma_f32_16x16x32_fp8_fp8` | **rejected** | ok |
| `v_mfma_f32_16x16x32_bf8_bf8` | **rejected** | ok |
| `v_mfma_f32_16x16x8_xf32` | rejected | ok |
| `v_smfmac_i32_16x16x32_i8` (sparse) | **rejected** | ok |
| `v_dot4c_i32_i8` | ok | ok |

Three conclusions, each of which closes a plausible idea:

- **There is no INT4 matrix path.** Not on gfx90a, and not on gfx942 either — the
  `i4` MFMA forms were a CDNA1 feature and are gone. So a W4A4 scheme buys
  nothing over unpacking to int8 or bf16. **Closed.**
- **There is no FP8 or sparse path.** Already known for FP8; `smfmac` confirms
  2:4 structured sparsity is CDNA3-only too. **Closed.**
- **The int8 shapes are disjoint between generations.** gfx90a has K=16
  (`16x16x16i8`), gfx942 has K=32 (`16x16x32_i8`), and each assembler rejects the
  other's. This is why 482 of the 1,180 blocked ASM kernels are blocked on
  `int8` — not because CDNA2 lacks INT8, but because it lacks *gfx942's* INT8.

So the complete matrix-core dtype set on this chip is **{fp64, fp32, bf16, fp16,
int8}**. Below 16 bits there is exactly one option, and it is int8.

### Which is why the format landscape collapses to two buckets

| Scheme | Activations | Path on gfx90a |
|---|---|---|
| **W8A8 int8** | int8 | **`v_mfma_i32_16x16x16i8`** |
| AWQ-Int4, GPTQ, W8A16, Q4_K_M, Q8_0 | bf16/fp16 | dequantize → 16-bit MFMA |
| FP8 (any) | fp8 | upcast → 16-bit MFMA, plus decode cost |
| W4A4 / NVFP4 / MXFP4 | fp4/int4 | no kernel, no instruction |

And because **CDNA2's INT8 peak equals its bf16 peak (181 TOPS = 181 TFLOP/s)**,
bucket 1 wins purely on halved memory traffic, never on arithmetic. That single
fact explains the whole measured matrix: why Q8_0 costs almost nothing over
Q4_K_M (4.57 s vs 4.43 s — both dequantize to the same 16-bit path), and why FP8
is catastrophic rather than merely disappointing (dequantizes *and* has no
decoder).

---

## 3. W4A8-INT8: the one format that should work, and does not

This is the idea worth chasing, because on paper it dominates W8A8 on exactly the
axis that matters here.

**The premise is sound.** Unpack int4 weights to int8 in-register (cheap VALU),
then run `v_mfma_i32_16x16x16i8`. You get AWQ's memory footprint *and* W8A8's
arithmetic. On a chip where INT8 is bandwidth-limited rather than
arithmetic-limited, halving the weight bytes again is the right direction.

**Both halves of the software exist:**

- vLLM ships `compressed_tensors_w4a8_int.py`.
- llm-compressor produces the checkpoints, e.g.
  `alishafique/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a8int8-llmcompressor` —
  verified `int-quantized`, weights int4 group-128, `input_activations`
  `num_bits: 8, type: "int", dynamic: true, strategy: "token"`.

**But no kernel on this hardware accepts int8 activations.** The scheme itself is
platform-agnostic and delegates to `choose_mp_linear_kernel()`. Auditing every
kernel in that registry for `act_type == torch.int8`:

| Kernel | Accepts int8 activations | Gate |
|---|---|---|
| `marlin.py` | **yes** (`marlin_act_int8_process_scales`) | `if not current_platform.is_cuda()` — **CUDA PTX** |
| `cutlass.py` | no (fp8 only) | CUDA + capability 90 |
| `dynamic_4bit.py` | **yes** — literally "unpack int4 → int8, int8 × int8 → int32" | `if not current_platform.is_cpu()` + **ARM/KleidiAI** |
| `machete.py` | no (fp16/bf16) | CUDA + capability 90 |
| `humming.py` | — | CUDA |
| `exllama.py` | no (fp16 only) | `is_cuda_alike` |
| `triton_w4a16.py` | no — `act_type not in (float16, bfloat16)` → reject | **ROCm ok**, but w4a16 |
| `rdna_hybrid_w4a16.py` | no (fp16/bf16) | ROCm, but `_on_gfx1x` (**RDNA only**) |
| `conch.py`, `allspark.py` | no | — |

**Every ROCm-reachable mixed-precision kernel requires fp16 or bf16
activations.** The two kernels that do int8 activations are Marlin (CUDA PTX) and
KleidiAI (ARM CPU). On gfx90a, a W4A8-int8 checkpoint therefore either finds no
kernel or falls back to a w4a16 path that dequantizes — losing precisely the
property that made it interesting.

**Critically, this is not the Int8-MoE bug again.** That one was a real defect: a
working kernel sitting behind a `current_platform.is_cuda()` test that did not
describe the hardware, and widening the gate turned the slowest arm into the
fastest. Here there is **no kernel being wrongly excluded** — Marlin genuinely
cannot run on AMD, and KleidiAI is an ARM CPU library. Widening a gate would
crash, not accelerate.

The distinction is the one `docs/25` closes with: *does the kernel actually exist
for this target?* For Int8 MoE it did. For W4A8 it does not.

### CONFIRMED by running it, and the blocker is worse than predicted

`benchmarks/matrix/round2.sh` E4 served
`alishafique/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a8int8-llmcompressor` on
gfx90a. It refuses at load, and the error enumerates every candidate:

```
ValueError: Failed to find a kernel that can implement the WNA16 linear layer. Reasons:
 RDNA3W4A16LinearKernel cannot implement due to: RDNA3 W4A16 kernel requires gfx1100
 TritonW4A16LinearKernel cannot implement due to: Quant type int4 not supported;
                                                  supported: [ScalarType.uint4b8, ScalarType.uint4]
 ConchLinearKernel  cannot implement due to: Weight type (int4) not supported by
                                             ConchLinearKernel, supported types are:
                                             [uint4, uint8, uint4b8, uint8b128]
```

The static audit was right that no kernel takes int8 activations. **It missed a
second, independent blocker: the int4 *weight packing* is also unsupported.**

`TritonW4A16LinearKernel` — the one ROCm-capable candidate — accepts
`uint4b8` and `uint4` but **not `int4`**. compressed-tensors with
`type: "int", symmetric: true` emits signed `int4`; GPTQ and AWQ emit the
unsigned-with-bias `uint4b8` form. So this checkpoint would fail on gfx90a
**even as plain W4A16**, before activations enter the picture.

Two further details worth recording:

- **vLLM routed it to `compressed_tensors_w4a8_fp8.py`**, not `w4a8_int` — the
  traceback names that file. Whatever the config says, the scheme selection
  landed on the FP8 variant, which is dead on CDNA2 regardless.
- **It degraded to "the WNA16 linear layer"** — i.e. even the attempted path was
  weight-only. The int8-activation path was never in play.

**Verdict: closed, with two blockers rather than one.** A Triton w4a8-int kernel
for gfx90a would need to handle signed `int4` weights *and* int8 activations,
neither of which any ROCm-reachable kernel does today.

**What it would take:** a Triton w4a8-int kernel — unpack int4 → int8, dynamic
per-token activation quantization, `tl.dot` with int8 operands accumulating to
int32. Triton on ROCm emits `v_mfma_i32_16x16x16i8` for int8 `tl.dot`, so the
hardware path is reachable from Triton; what is missing is the kernel and its
registration in the mixed-precision registry. That is real work — a new kernel,
not a gate patch — and it is the highest-value remaining item in `docs/25`.

**Expected payoff, stated as a bound rather than a promise.** W8A8 at tier 1 is
29.17 GiB of weights against AWQ-Int4's 15.7 GiB. W4A8 would land near the
latter with the former's arithmetic. Since the measured W8A8 → AWQ-Int4 gap is
58% of prefill and is *bandwidth*, halving weight bytes again should help — but
activation-quantization overhead and int4 unpacking both cost something, and
neither is measured. Do not quote a number until one exists.

---

## 4. `head_dim = 256` reaches none of the fast paths — and still wins

Worth isolating, because it is the most counter-intuitive result in the matrix.

Qwen3-Next-80B has `head_dim = 256`. Enumerating what that costs it on ROCm:

| Fast path | Available at `head_dim 256`? | Why |
|---|---|---|
| AITER ASM attention | **no** | AITER ships fmha ASM for `hd128`/`hd192` only — upstream, every architecture |
| ROCm custom paged attention | **no** | `CALL_CUSTOM_LAUNCHER` instantiates `head_size` **64 and 128** only; `default: TORCH_CHECK(false, "Unsupported head size")` |
| The 256k reduction patch | **no** | same gate — it never reaches the custom kernel to begin with |
| INT8 MFMA GEMM | yes | GEMM does not depend on head dim |

Both attention gaps are **upstream instantiation gaps, not misapplied
platform checks.** This is the distinction that matters when triaging: the Int8
MoE bug was a working kernel behind a bad `is_cuda()` test, and widening it
turned the slowest arm into the fastest. Here the Python gate
(`head_size == 64 or head_size == 128`) describes the C++ dispatch exactly.
Widening it produces `TORCH_CHECK(false, "Unsupported head size: 256")`, not
speed. Instantiating `HEAD_SIZE=256` is not a drop-in either — the QKV kernel's
LDS tiling (`shared_logits[NWARPS][4][16][4]`) and register budget are written
around the supported head dims.

**So the 80B decodes on Triton, always, with every ROCm attention fast path
unavailable to it — and it is still the fastest arm in this matrix**: 2.27 s
TTFT at 15k, 51.3 t/s decode at 101k.

That is worth sitting with. It means the tier-2 result is **entirely
architectural** — 3B active parameters and a 3:1 Gated DeltaNet hybrid that
keeps only 12 of 48 layers holding KV (10.4 GiB at TP=1, 1.28 M tokens in
14.9 GiB at TP=2). None of the kernel work in this repo touched it.

Two conclusions follow, pulling in opposite directions, and both are real:

- **Kernel optimization has a ceiling that model choice does not.** The AITER
  ASM work is worth +12.8%; picking a hybrid-attention model is worth more than
  that at long context, and it composes with everything else.
- **The 80B has the most headroom left of anything measured**, precisely
  because none of the fast paths apply. An `hd256` fmha kernel or an
  `HEAD_SIZE=256` paged-attention instantiation would benefit the one model that
  currently gets nothing — which is a better argument for that work than the
  +12.8% ceiling suggests on its own.

## Summary

| Idea | Verdict |
|---|---|
| Other formats reach the INT8 MFMA | **No.** W8A8 is the only one, today |
| INT4 matrix path | **Closed** — no `i4` MFMA on gfx90a or gfx942 |
| FP8 / sparse (`smfmac`) matrix path | **Closed** — CDNA3 only |
| AITER ASM helps only W8A8 | **Wrong** — it is bf16 *attention*, helps every format at `head_dim` 128/192; the +12.8% was measured on AWQ-Int4 |
| AITER ASM INT8 GEMM | **Does not exist on gfx90a** — `i8gemm` is 0 of 9 portable |
| W4A8-int8 | **Closed by measurement.** Two blockers: no ROCm kernel takes int8 activations, *and* none accepts signed `int4` weights (they want `uint4b8`/`uint4`) |
| Custom paged attention at `head_dim 256` | **No kernel** — `head_size` 64/128 only. The gate is accurate, not conservative |
| The 80B's speed came from kernels | **No** — it reaches *none* of the ROCm attention fast paths and still wins. Purely architectural |
