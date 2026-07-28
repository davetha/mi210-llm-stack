#!/usr/bin/env python3
"""Numeric acceptance test for the 256k paged-attention patch on gfx9.

`configs/extend_rocm_pa_256k_gfx9.py` extends the ROCm custom paged-attention
reduction switch from `npar_loops` 8 to 16, lifting its context ceiling from
131,072 to 262,144. That patch is cheap to apply and easy to believe, which is
exactly why it needs this test.

**A throughput win is not an acceptance signal.** `docs/14` records what that
mistake costs: a binary patch swapped bf16 MFMA for f16 MFMA, the kernel ran at
full speed, and it computed the wrong thing for weeks before anyone checked.

What this checks
----------------
The custom kernel is compared against **vLLM's Triton paged-attention**, both
driven through the real `chunked_prefill_paged_decode` entry point -- once
normally, once with `use_rocm_custom_paged_attention` forced to return False.
Triton is the right reference because it is exactly what the 128k gate falls
back to today, so "the patch is sound" means precisely "the custom kernel agrees
with the fallback it replaces".

An earlier version compared against a hand-written fp32 dense reference, and
that failed at **every** length including `npar_loops = 1`, which upstream has
always supported. The lesson is worth keeping: a reference that reimplements the
paged KV layout is a second thing that can be wrong, and when it is, it fails
identically at all lengths and tells you nothing about the patch. Two kernels
consuming the *same* tensors cannot disagree for layout reasons.

The control cases carry the weight. Agreement at `npar_loops <= 8` is
upstream-verified behaviour, so if those pass the harness is sound, and any
failure at 9..16 is the patch:

    131,072  npar_loops =  8  -- last length upstream supports (CONTROL)
    139,264  npar_loops =  9  -- FIRST length that needs the patch
    196,608  npar_loops = 12
    262,144  npar_loops = 16  -- new ceiling
    266,240  npar_loops = 17  -- must still be REFUSED, loudly

That last case matters as much as the rest: the ceiling must still exist. A
patch that removed the guard entirely would pass every other case here and be
more dangerous than the bug it fixed.

Usage
-----
    python3 tests/test_rocm_pa_256k.py
    python3 tests/test_rocm_pa_256k.py --dtype float16 --head-size 64

Against a STOCK build the 9+ cases should report REFUSED -- that is the expected
stock behaviour and confirms the test exercises the patched path at all.
"""

from __future__ import annotations

import argparse
import sys

import torch

PARTITION_SIZE = 256
WARP_SIZE = 64

CASES = [
    (131_072, 8, "CONTROL: last length upstream supports"),
    (139_264, 9, "FIRST length requiring the patch"),
    (196_608, 12, "mid-range"),
    (262_144, 16, "new ceiling"),
]
OVER_CEILING = (266_240, 17, "must still be refused")


def npar_loops(seq_len: int) -> int:
    parts = (seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    return (parts + WARP_SIZE - 1) // WARP_SIZE


def build_inputs(seq_len, num_heads, num_kv_heads, head_size, block_size, dtype, device):
    """One decode token against `seq_len` of context.

    The layout here is dictated by the call site, not chosen. Two traps:

    * **K and V have different ranks.** The Triton decode kernel is passed
      `x=key_cache.shape[4]` and `stride_k_cache_0..4`, but only
      `stride_v_cache_0..3` -- so K is 5-D
      `[blocks, kv_heads, head_size/x, block_size, x]` and V is 4-D
      `[blocks, kv_heads, head_size, block_size]`. `real_block_size` is read as
      `value_cache.shape[3]`.
    * **This is NOT what `RocmAttentionBackend.get_kv_cache_shape` returns.**
      That declares `(2, num_blocks, block_size, num_kv_heads, head_size)`,
      which is the allocation shape; it is re-viewed before reaching this op.
      Trusting the declared shape here produces order-1 errors at every length,
      which is exactly how the first version of this test failed.

    `x` is the 16-byte vectorisation width: 8 for 2-byte dtypes.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks + 16
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    assert head_size % x == 0, f"head_size {head_size} not divisible by x={x}"

    torch.manual_seed(0xA1DE)
    query = torch.randn(1, num_heads, head_size, dtype=dtype, device=device)
    key_cache = torch.randn(
        total_blocks, num_kv_heads, head_size // x, block_size, x,
        dtype=dtype, device=device,
    )
    value_cache = torch.randn(
        total_blocks, num_kv_heads, head_size, block_size,
        dtype=dtype, device=device,
    )
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": torch.arange(num_blocks, dtype=torch.int32,
                                    device=device).unsqueeze(0),
        "seq_lens": torch.tensor([seq_len], dtype=torch.int32, device=device),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device=device),
    }


def run_decode(inp, seq_len, num_kv_heads, head_size, block_size, scale,
               dtype, device, force_triton: bool):
    """Drive the production decode entry point, optionally forcing Triton.

    The call site does `from vllm.platforms.rocm import
    use_rocm_custom_paged_attention` *inside* the function body, so rebinding the
    module attribute takes effect at call time. That is what lets both arms run
    the identical surrounding code.
    """
    from vllm.platforms import rocm as rocm_platform  # noqa: PLC0415
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: PLC0415
        chunked_prefill_paged_decode,
    )

    num_heads = inp["query"].shape[1]
    output = torch.empty(1, num_heads, head_size, dtype=dtype, device=device)
    one = torch.tensor(1.0, device=device)

    original = rocm_platform.use_rocm_custom_paged_attention
    if force_triton:
        rocm_platform.use_rocm_custom_paged_attention = lambda *a, **k: False
    try:
        chunked_prefill_paged_decode(
            query=inp["query"], key=None, value=None, output=output,
            kv_cache_dtype="auto",
            key_cache=inp["key_cache"], value_cache=inp["value_cache"],
            block_table=inp["block_table"],
            query_start_loc=inp["query_start_loc"],
            seq_lens=inp["seq_lens"],
            max_seq_len=seq_len, max_query_len=1,
            k_scale=one, v_scale=one, sm_scale=scale,
        )
    finally:
        rocm_platform.use_rocm_custom_paged_attention = original
    return output[0].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--head-size", type=int, default=128, choices=[64, 128])
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=16, choices=[16, 32])
    # Two independent bf16 kernels over a 262k-term softmax will not agree to
    # more than a few parts in a hundred. This bound catches a dropped partition
    # group, which is an order-1 error, not a rounding difference.
    ap.add_argument("--rtol", type=float, default=2e-2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no GPU visible", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    print(f"device: {arch}  dtype: {args.dtype}  head_size: {args.head_size}")
    if "gfx9" not in arch:
        print(f"WARNING: this patch targets gfx9; {arch} is not covered", file=sys.stderr)

    scale = args.head_size ** -0.5
    failures = []

    for seq_len, expect_npar, note in CASES:
        got_npar = npar_loops(seq_len)
        assert got_npar == expect_npar, f"test bug: {seq_len} -> {got_npar}"
        label = f"seq={seq_len:>7,}  npar={got_npar:>2}  {note}"

        # Assert the custom path is actually SELECTED before comparing anything.
        # Without this the test has a silent-pass mode: if the gate declined,
        # both arms would run Triton and agree perfectly, and a broken patch
        # would look flawless. This is the same failure this repo hit with
        # ROCM_AITER_FA -- backend admitted, never chosen, credited anyway.
        from vllm.platforms.rocm import use_rocm_custom_paged_attention  # noqa: PLC0415
        if not use_rocm_custom_paged_attention(
            dtype, args.head_size, args.block_size,
            args.num_heads // args.num_kv_heads, seq_len, 0, "auto", None, None,
        ):
            print(f"FAIL     {label}  gate DECLINED -- custom path never ran")
            failures.append((label, "gate declined; comparison would be Triton vs Triton"))
            continue

        inp = build_inputs(seq_len, args.num_heads, args.num_kv_heads,
                           args.head_size, args.block_size, dtype, device)
        common = (seq_len, args.num_kv_heads, args.head_size, args.block_size,
                  scale, dtype, device)
        try:
            custom = run_decode(inp, *common, force_triton=False)
        except RuntimeError as exc:
            kind = ("kernel refused; patch not applied?"
                    if "Unsupported npar_loops" in str(exc) else str(exc))
            print(f"{'REFUSED' if 'npar_loops' in str(exc) else 'ERROR':8} {label}")
            print(f"         -> {str(exc).splitlines()[0]}")
            failures.append((label, kind))
            continue

        triton = run_decode(inp, *common, force_triton=True)
        denom = triton.abs().max().clamp_min(1e-6)
        err = ((custom - triton).abs().max() / denom).item()
        ok = err <= args.rtol
        print(f"{'PASS' if ok else 'FAIL':8} {label}  rel_err={err:.2e}")
        if not ok:
            failures.append((label, f"rel_err {err:.2e} > {args.rtol}"))

    # The ceiling must STILL be enforced. Silently truncating here would be a
    # worse bug than the one this patch fixes, and it would look like a pass.
    seq_len, _, note = OVER_CEILING
    label = f"seq={seq_len:>7,}  npar={npar_loops(seq_len):>2}  {note}"
    inp = build_inputs(seq_len, args.num_heads, args.num_kv_heads,
                       args.head_size, args.block_size, dtype, device)
    try:
        run_decode(inp, seq_len, args.num_kv_heads, args.head_size,
                   args.block_size, scale, dtype, device, force_triton=False)
    except RuntimeError as exc:
        if "Unsupported npar_loops" in str(exc):
            print(f"PASS     {label}  (refused as required)")
        else:
            print(f"FAIL     {label}  wrong error: {str(exc).splitlines()[0]}")
            failures.append((label, "wrong error"))
    else:
        # NOTE: reaching here is not automatically a bug -- if the Python gate
        # refuses first, the call quietly runs Triton and returns. Distinguish
        # the two, because "gate held" and "guard removed" look identical from
        # the return value alone.
        from vllm.platforms.rocm import use_rocm_custom_paged_attention  # noqa: PLC0415
        would_use_custom = use_rocm_custom_paged_attention(
            dtype, args.head_size, args.block_size,
            args.num_heads // args.num_kv_heads, seq_len, 0, "auto", None, None,
        )
        if would_use_custom:
            print(f"FAIL     {label}  gate ACCEPTED an over-ceiling length")
            failures.append((label, "gate lets npar_loops=17 through to the kernel"))
        else:
            print(f"PASS     {label}  (gate declined; Triton served it)")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for label, why in failures:
            print(f"  {label}\n    {why}")
        return 1
    print("all cases passed: custom paged attention agrees with Triton to 256k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
