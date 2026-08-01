#!/usr/bin/env python3
"""Does AITER's CK int8 GEMM beat vLLM's Triton scaled_mm on gfx90a?

Round 38e's profile put `scaled_mm_kernel` at 10.9 s of trace kernel time --
the single largest consumer, ~52% -- and that kernel is vLLM's GENERIC TRITON
FALLBACK (compressed_tensors/triton_scaled_mm.py). vLLM 0.26 already wires a
CK path (`_aiter_ops.py:541` -> `aiter.gemm_a8w8_CK`) behind
VLLM_ROCM_USE_AITER_LINEAR, which this project pins to 0 because
enable_vllm_aiter_gfx90a.py deliberately narrowed the AITER gate to attention.

This probe answers, WITHOUT touching any serve path, whether widening that
gate is worth it:

  1. Does gemm_a8w8_CK build CK instances for gfx90a at all, or does it
     decline the device / emit garbage?
  2. Is it numerically correct against an fp32 reference?
  3. Is it FASTER than the Triton kernel at the shapes that actually matter --
     decode (M=1..16) far more than prefill?

A "yes" to all three makes the gate-widening patch the highest-value item on
the board. A numeric failure or a loss at M=1 kills it in 20 minutes instead
of an evening, which is the point of running this before writing the patch.

Shapes are the real ones for Qwen3-30B-A3B at TP=1: qkv_proj (K=2048,
N=5120) and o_proj (K=4096, N=2048). The MoE experts do NOT come through
here -- they are fused_moe's problem, a separate lead.

Per-tensor scales, matching how compressed-tensors W8A8 stores this
checkpoint (docs/24). CK is asked for bf16 out, which is what the model runs.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
# Decode shapes first: M=1 is the batch-1 case that the whole decode gap is
# about. Prefill shapes are included so a win/loss there is visible too, but
# they are not what this probe is for.
M_DECODE = [1, 2, 4, 8, 16]
M_PREFILL = [512, 2048, 8192]
SHAPES = [("qkv_proj", 2048, 5120), ("o_proj", 4096, 2048)]

REPS = 50
WARMUP = 10


def make_inputs(m, k, n):
    """Same numbers, each kernel in ITS OWN natural layout.

    These two kernels do not agree on layout, and forcing one to eat the
    other's would measure a transpose rather than a GEMM:
      CK      wants WQ [N, K] and w_scale [1, N];
      Triton  asserts weight [K, N] AND scale_b [N, 1] (triton_scaled_mm.py
              :157,165) and requires a weakly-contiguous weight, so a bare
              .t() view is both wrong-shaped for the scale and a stride trap.
    Both weights are materialised contiguous from identical values.
    """
    torch.manual_seed(0)
    # int8 in [-127, 127]; realistic magnitudes matter for accumulation error.
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=DEV)
    b_nk = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=DEV)
    b_kn = b_nk.t().contiguous()
    as_ = torch.full((m, 1), 0.01, dtype=torch.float32, device=DEV)
    bs_1n = torch.full((1, n), 0.02, dtype=torch.float32, device=DEV)
    bs_n1 = torch.full((n, 1), 0.02, dtype=torch.float32, device=DEV)
    return a, b_nk, b_kn, as_, bs_1n, bs_n1


def reference(a, b_nk, as_, bs_1n):
    """fp32 reference: dequantize, matmul, in float."""
    return (a.float() @ b_nk.float().t()) * as_ * bs_1n


def bench(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6  # microseconds


def main():
    from aiter import gemm_a8w8_CK
    from aiter.jit.utils.chip_info import get_gfx

    gfx = get_gfx()
    print(f"arch: {gfx}")
    if gfx != "gfx90a":
        print(f"ABORT: expected gfx90a, got {gfx} -- this probe is about CDNA2")
        return 1

    # vLLM's Triton path -- the kernel the profile caught. Import the same
    # entry point the quantization layer uses, so a win here is a win there.
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    print(f"\n{'shape':<10} {'M':>6} {'CK us':>10} {'Triton us':>10} "
          f"{'speedup':>9} {'max_err':>10}  verdict")
    print("-" * 68)

    ck_failed = 0
    rows = []
    for name, k, n in SHAPES:
        for m in M_DECODE + M_PREFILL:
            a, b_nk, b_kn, as_, bs_1n, bs_n1 = make_inputs(m, k, n)
            ref = reference(a, b_nk, as_, bs_1n)
            scale = max(ref.abs().max().item(), 1e-9)

            # --- CK. A crash here IS the answer; do not let it kill the run.
            try:
                out_ck = gemm_a8w8_CK(a, b_nk, as_, bs_1n, None, torch.bfloat16)
                rel = (out_ck.float() - ref).abs().max().item() / scale
                t_ck = bench(gemm_a8w8_CK, a, b_nk, as_, bs_1n, None,
                             torch.bfloat16)
                ck_ok = rel < 1e-2
            except Exception as exc:  # noqa: BLE001 -- reporting is the job
                print(f"{name:<10} {m:>6}   CK FAILED: {type(exc).__name__}: "
                      f"{str(exc)[:60]}")
                ck_failed += 1
                continue

            # --- Triton, same values, its own layout.
            try:
                out_tr = triton_scaled_mm(a, b_kn, as_, bs_n1, torch.bfloat16)
                t_tr = bench(triton_scaled_mm, a, b_kn, as_, bs_n1,
                             torch.bfloat16)
                rel_tr = (out_tr.float() - ref).abs().max().item() / scale
                if rel_tr >= 1e-2:
                    # The baseline being wrong invalidates the comparison just
                    # as surely as the candidate being wrong.
                    print(f"{name:<10} {m:>6}   TRITON NUMERICALLY WRONG "
                          f"(rel {rel_tr:.3g}) -- comparison void")
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"{name:<10} {m:>6}   Triton FAILED: "
                      f"{type(exc).__name__}: {str(exc)[:50]}")
                continue

            speedup = t_tr / t_ck
            verdict = "CK wins" if speedup > 1.05 else (
                "Triton wins" if speedup < 0.95 else "tie")
            if not ck_ok:
                verdict = f"CK WRONG (rel {rel:.3g})"
            tag = "decode" if m in M_DECODE else "prefill"
            print(f"{name:<10} {m:>6} {t_ck:10.1f} {t_tr:10.1f} "
                  f"{speedup:8.3f}x {rel:10.2e}  {verdict} [{tag}]")
            rows.append((tag, speedup, ck_ok, rel_tr))

    print()
    if ck_failed:
        print(f"{ck_failed} CK invocations failed outright.")
    if not rows:
        print("NO COMPARABLE ROWS -- CK does not run here. Gate-widening is dead.")
        return 1

    bad = [r for r in rows if not r[2]]
    if bad:
        print(f"NUMERIC FAILURES: {len(bad)}/{len(rows)} shapes. A fast wrong "
              "kernel is worse than the slow right one -- do NOT wire this in.")
        return 1

    dec = [r[1] for r in rows if r[0] == "decode"]
    pre = [r[1] for r in rows if r[0] == "prefill"]
    if dec:
        print(f"decode  shapes: median speedup {sorted(dec)[len(dec)//2]:.3f}x "
              f"(min {min(dec):.3f}x, max {max(dec):.3f}x)")
    if pre:
        print(f"prefill shapes: median speedup {sorted(pre)[len(pre)//2]:.3f}x "
              f"(min {min(pre):.3f}x, max {max(pre):.3f}x)")
    print()
    print("READING THIS. Decode is what matters -- round 38e put the ~3x gap "
          "in-kernel at batch 1.\nA decode median above ~1.15x justifies "
          "widening the AITER gate to linear ops and\nre-running the serve "
          "A/B; near 1.0x means the Triton kernel is already competitive\nand "
          "the 52%% of decode time it owns has to be attacked another way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
