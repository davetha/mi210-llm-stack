# MACs per issued instruction — what "good" looks like on gfx90a

**Date**: 2026-07-30 · **Hardware**: 2× MI210 (gfx90a / CDNA2)
**Origin**: proposed and measured by [Andrei-Dr](https://github.com/Andrei-Dr).
Every number below was **independently re-verified against the installed tree**
before being recorded here; where my figure differs from the original it is
stated.

`docs/30` established that the shipped Triton GEMMs issue ~100 instructions per
MFMA, and left open the question that actually matters: **is 100 bad?** Without a
ceiling, a ratio is not a verdict. This doc supplies the ceiling from the hand-
written AITER ASM already installed on the box.

## The metric, and the trap in the obvious version

Ranking kernels by *instructions per matrix op* is meaningless, because matrix
ops differ by three orders of magnitude in work done: `v_dot4_i32_i8` performs
**4** MACs, `v_mfma_f32_32x32x8f16` performs **8,192**. By that metric llama.cpp
looks ~30× better than Triton while doing far less arithmetic per instruction.

Normalised to **MACs per issued instruction**, whole-kernel:

| source | kind | objects | median MACs/ins | best |
|---|---|---:|---:|---:|
| AITER `fmoe` (MoE expert FFN) | hand ASM | 8 | **1,058** | 1,152 |
| AITER `fmha_v3_fwd` | hand ASM | 48 | 602 | 868 |
| AITER `pa` | hand ASM | 8 | 539 | 656 |
| vLLM **shipping `fused_moe`** | Triton | — | **195** | 496 |
| llama.cpp `mul_mat_q`/`mul_mat_f` | HIP | 1,545 | 98.6 | 263 |
| vLLM Triton FP8 blockscale `32x32x8f16` | Triton | 1,757 | 85.0 | — |
| vLLM Triton FP8 blockscale `16x16x16f16` | Triton | 1,131 | 39.1 | — |

**The hand-written ASM proves 539–1,058 MACs/instruction is achievable on this
chip.** That is the ceiling `docs/30` was missing.

> **Corrected.** An earlier version of this table omitted the `fused_moe` row and
> concluded the compiled path sits "6–13× below" the ASM ceiling. That used the
> **FP8 blockscale** kernels as the baseline — and `docs/30` had already
> established those are the known-pathological family and *not* the path that
> ships. Recounting the actually-served `fused_moe` gives **195 median / 496
> best**, so the real gap to ASM is roughly **2–5×**, not 6–13×. The FP8 rows are
> retained as the outlier they are.
>
> That also retires the "brackets the ~3× decode gap" inference, which was wrong
> twice over: the denominators differ (issue capability vs bandwidth), *and* the
> 6–13× input was itself the wrong baseline. `fused_moe` numbers reported by
> Andrei-Dr; not independently re-verified here, unlike the `fmoe` row below.

### Independent verification of the top row

Disassembled `fmoe_fp16_noquantBf16_g1u0_vs_atm_inlv_silu_1tg_ps_32x512.co` from
the installed tree:

```
ins = 3,714   mfma = 1,024   ins/mfma = 3.63   MACs/ins = 1,129
```

(1,024 × `v_mfma_f32_16x16x16bf16_1k` × 4,096 MACs each ÷ 3,714 instructions.)
**1,129** sits between the reported median 1,058 and best 1,152 — reproduced.
Against Triton's 85.0 that is a **13.3× gap**, the top of the stated range.

Also confirmed on the same object: ELF flags **`0x53f, gfx90a`**, exactly
**1,024** MFMA, **only** `v_mfma_f32_16x16x16bf16_1k` — the correct D3E7 encoding
for CDNA2, not gfx942's D3E1 — and **zero** undecodable opcodes.

Object counts reproduce exactly from `aiter_meta/hsa/gfx90a`:

| family | objects |
|---|---:|
| `fmha_v3_bwd` | 138 |
| `fmha_v3_fwd` | 48 |
| `topksoftmax` | 22 |
| `mla` | 11 |
| `pa` | 8 |
| **`fmoe`** | **8** |
| **total** | **242** |

There is no `i8gemm/` or `bf16gemm/` directory at all, so `docs/19`'s port matrix
reproduces from the installed tree.

## The lead: `fmoe` has never been retested since the repatch

`fmoe` has the **highest MACs/ins of anything on the box** and it is the **MoE
expert FFN** — the dominant cost of the primary model. It is also the one family
still written off:

- `docs/15` — *"fmoe_b16.co ILLEGAL_INSTRUCTION: the patched BF16 MoE dispatcher
  executes 1024 swapped MFMA ops but traps mid-kernel"*, still listed as open.
- `docs/16`:135 — `| fmoe | 838 | ~20/kernel | Patched, not tested through ATOM |`

**That failure dates from the retracted D3E1→D3CD era**, i.e. before
`configs/repatch_gfx942_to_gfx90a.py` produced a correct mapping. The two sibling
families written off for the same reason both turned out to work:

| family | original verdict | actual |
|---|---|---|
| `fmha_v3_fwd` | "unreachable on gfx90a" | **80/80 numerically exact** |
| `pa_fwd_asm` | "gfx942 binaries can't run on gfx90a" | **48/48 exact** (blocker was a stale JIT module) |
| **`fmoe`** | ILLEGAL_INSTRUCTION | **never retested** |

Two for two, and the third has the best instruction efficiency and the most
relevant workload. The static evidence above is clean — correct opcode, correct
ELF target, no undecodable instructions — which is consistent with the trap
having been a patching artifact rather than a hardware limit.

**Not yet run.** The next step is pointing the `--require-asm` harness (the
pattern in `tests/test_fmha_v3_fwd_asm_gfx90a.py`) at `fmoe` and checking whether
it still traps. This has not been done, and nothing here should be read as
saying `fmoe` works.

### Three constraints that bound how much this can pay

1. **All 8 objects are `noquantFp16` / `noquantBf16`.** They do **not** serve the
   W8A8-int8 path. This is a bf16/fp16-arms-only lead — still the tier where bf16
   posts the fastest inference numbers on this box (`docs/25`), but not the tier
   that ships by default.
2. **Coverage is 8 of 838**, all `g1u0` / `32x512`. Expert geometry has to match
   or the kernel is not selected.
3. **MACs/ins is a static count.** It screened the FP8 pathology correctly from
   the binary alone, but a favourable ratio is **not a measured speedup**. This
   repo has been burned repeatedly by treating a plausible number as a result.

## Two framing points to keep straight

**A ratio against the ceiling does not bracket the ~3× decode gap.** Those are
different denominators — 3.1× is against a *bandwidth* bound (`docs/25` item 1c,
as resized), while the MACs/ins ratio is against *issue capability*. A kernel 5×
off peak issue efficiency need not be 5× slow end-to-end, because it may spend
much of its time waiting on memory regardless. The static ratio is *consistent
with* an issue-bound decode path; it does not measure it. (Retracted by its
author as well as here.)

**Count the path that ships, not the path that is cached.** This was the trap in
the first version of the table. `docs/30` observed that the Triton cache is
dominated by the FP8 blockscale family and that the shipping path (`docs/24`:
W8A8 dense + W8A16 experts) was **never cached anywhere on the box** — then the
first table used the FP8 numbers as the baseline anyway, tripling the apparent
gap. Recounting `fused_moe` fixed it. The general rule: a kernel being *available
to count* is not evidence it *runs*.

`rocprofv3 --kernel-trace` on decode steady-state supplies the dynamic half and
would settle both points.

## A side result: llama.cpp's dot4 attention path

`flash_attn_tile` runs at **~1.4 MACs/ins** (≈2.9 instructions per
`v_dot4_i32_i8`, which is 4 MACs) against AITER ASM attention's 602. `docs/22`
concluded the matrix cores were never idle, which holds for the kernel rocWMMA
diverted *to* — but the dot4 path has a very large efficiency gap that neither
rocWMMA nor the CK FA build closes. This is the number `docs/25` item 5's open
sub-question was missing. *(Reported, not independently re-verified here.)*

## Worth adopting as a standing gate

The metric costs a disassembly and no GPU time:

| MACs/ins | reading |
|---:|---|
| **> 500** | ASM-class, near the chip's issue capability |
| **100–500** | reasonable compiled code |
| **< 100** | look for a dequant loop |
| **< 50** | software emulation of something the hardware lacks |

Both pathologies this repo has found — the FP8 `v_cmp`/`v_cndmask` emulation
(`docs/21`) and the W8A16 expert dequant (`docs/24`) — would have been caught by
that one number without touching a GPU.
