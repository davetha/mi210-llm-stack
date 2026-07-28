#!/usr/bin/env python3
"""Numeric acceptance test for the 256k paged-attention patch on gfx9.

`configs/extend_rocm_pa_256k_gfx9.py` extends the ROCm custom paged-attention
reduction switch from `npar_loops` 8 to 16, lifting its context ceiling from
131,072 to 262,144. That patch is cheap to apply and easy to believe, which is
exactly why it needs this test.

**A throughput win is not an acceptance signal.** `docs/14` in this repo records
what that mistake costs: a binary patch swapped bf16 MFMA for f16 MFMA, the
kernel ran at full speed, and it computed the wrong thing for weeks before
anyone checked. The kernel here is *correct by construction* only if the
reduction genuinely handles more than 8 partition groups, and the way to know
that is to compare its output against a reference at lengths that require 9+.

What this checks
----------------
For sequence lengths straddling the old ceiling, the custom kernel's output is
compared against vLLM's Triton paged-attention on identical inputs:

    131,072  npar_loops =  8  -- last length upstream supports (control)
    139,264  npar_loops =  9  -- FIRST length that needs the patch
    196,608  npar_loops = 12
    262,144  npar_loops = 16  -- new ceiling
    266,240  npar_loops = 17  -- must still be REFUSED, loudly

The 9-partition case is the real test. If the patch were merely accepted by the
compiler without the reduction actually iterating the extra groups, this is
where the mismatch appears -- the tail of the sequence would be dropped and the
softmax denominator would be wrong, which shows up as a large relative error
rather than a subtle one.

The 17 case matters too: the ceiling must still exist and must still fail with
`TORCH_CHECK`, not silently truncate. A patch that removes the guard entirely
would pass every other case here and be far more dangerous than the bug it fixed.

Usage
-----
    python3 tests/test_rocm_pa_256k.py
    python3 tests/test_rocm_pa_256k.py --dtype float16 --head-size 64

Exit code is 0 only if every comparison passes AND the over-ceiling case is
refused. Requires a build with the patch applied; run against a stock build to
see the 9+ cases fail with "Unsupported npar_loops", which is the expected
stock behaviour and confirms the test is actually exercising the path.
"""

from __future__ import annotations

import argparse
import sys

import torch

# 8 * 64 * 256 upstream; 16 * 64 * 256 patched.
PARTITION_SIZE = 256
WARP_SIZE = 64

CASES = [
    (131_072, 8, "control: last length upstream supports"),
    (139_264, 9, "FIRST length requiring the patch"),
    (196_608, 12, "mid-range"),
    (262_144, 16, "new ceiling"),
]
OVER_CEILING = (266_240, 17, "must still be refused")


def npar_loops(seq_len: int) -> int:
    parts = (seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    return (parts + WARP_SIZE - 1) // WARP_SIZE


def build_inputs(seq_len, num_heads, num_kv_heads, head_size, block_size, dtype, device):
    """One sequence at `seq_len`, laid out as vLLM's decode path expects."""
    num_blocks = (seq_len + block_size - 1) // block_size
    # Allocate a little slack so the block table never indexes the final block
    # partially -- that is a separate edge case and not what is under test.
    total_blocks = num_blocks + 16

    torch.manual_seed(0xA1DE)
    query = torch.randn(1, num_heads, head_size, dtype=dtype, device=device)
    key_cache = torch.randn(
        total_blocks, num_kv_heads, head_size, block_size, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        total_blocks, num_kv_heads, head_size, block_size, dtype=dtype, device=device
    )
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    return query, key_cache, value_cache, block_table, seq_lens


def reference_attention(query, key_cache, value_cache, block_table, seq_len,
                        num_kv_heads, head_size, block_size, scale):
    """Dense reference in fp32. Slow and obviously correct -- that is the point.

    Reconstructs K/V from the paged layout, then does a plain softmax(QK^T)V so
    the comparison does not depend on any of the machinery under test.
    """
    num_heads = query.shape[1]
    gqa = num_heads // num_kv_heads
    blocks = block_table[0, : (seq_len + block_size - 1) // block_size]

    # [num_blocks, kv_heads, head_size, block_size] -> [kv_heads, seq, head_size]
    k = key_cache[blocks].permute(1, 0, 3, 2).reshape(num_kv_heads, -1, head_size)
    v = value_cache[blocks].permute(1, 0, 3, 2).reshape(num_kv_heads, -1, head_size)
    k = k[:, :seq_len, :].float()
    v = v[:, :seq_len, :].float()

    q = query[0].float()                                   # [heads, head_size]
    out = torch.empty(num_heads, head_size, dtype=torch.float32, device=q.device)
    for h in range(num_heads):
        kv = h // gqa
        logits = (q[h] @ k[kv].T) * scale                  # [seq]
        probs = torch.softmax(logits, dim=-1)
        out[h] = probs @ v[kv]
    return out


def run_custom(query, key_cache, value_cache, block_table, seq_lens, max_seq_len,
               num_kv_heads, head_size, block_size, scale, dtype, device):
    num_heads = query.shape[1]
    max_num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    output = torch.empty(1, num_heads, head_size, dtype=dtype, device=device)
    exp_sums = torch.empty(1, num_heads, max_num_partitions,
                           dtype=torch.float32, device=device)
    max_logits = torch.empty_like(exp_sums)
    tmp_output = torch.empty(1, num_heads, max_num_partitions, head_size,
                             dtype=dtype, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    torch.ops._rocm_C.paged_attention(
        output, exp_sums, max_logits, tmp_output, query,
        key_cache, value_cache, num_kv_heads,
        scale, block_table, seq_lens, query_start_loc,
        block_size, max_seq_len, None, "auto",
        torch.tensor(1.0, device=device), torch.tensor(1.0, device=device), None,
    )
    return output[0].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--head-size", type=int, default=128, choices=[64, 128])
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=16, choices=[16, 32])
    # bf16 through a 262k-term softmax accumulates real error; this bound is set
    # to catch a dropped partition group (which is order-1 wrong), not to certify
    # bit-exactness against fp32.
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
        assert got_npar == expect_npar, f"test bug: {seq_len} -> {got_npar} != {expect_npar}"
        label = f"seq={seq_len:>7,}  npar={got_npar:>2}  {note}"

        q, kc, vc, bt, sl = build_inputs(
            seq_len, args.num_heads, args.num_kv_heads, args.head_size,
            args.block_size, dtype, device)
        try:
            got = run_custom(q, kc, vc, bt, sl, seq_len, args.num_kv_heads,
                             args.head_size, args.block_size, scale, dtype, device)
        except RuntimeError as exc:
            if "Unsupported npar_loops" in str(exc):
                print(f"REFUSED  {label}\n         -> {exc}".rstrip())
                failures.append((label, "kernel refused; patch not applied?"))
            else:
                print(f"ERROR    {label}\n         -> {exc}".rstrip())
                failures.append((label, str(exc)))
            continue

        want = reference_attention(q, kc, vc, bt, seq_len, args.num_kv_heads,
                                   args.head_size, args.block_size, scale)
        err = ((got - want).abs().max() / want.abs().max().clamp_min(1e-6)).item()
        ok = err <= args.rtol
        print(f"{'PASS' if ok else 'FAIL':8} {label}  rel_err={err:.2e}")
        if not ok:
            failures.append((label, f"rel_err {err:.2e} > {args.rtol}"))

    # The ceiling must still be enforced. Silently truncating here would be a
    # worse bug than the one this patch fixes, and it would look like a pass.
    seq_len, expect_npar, note = OVER_CEILING
    label = f"seq={seq_len:>7,}  npar={npar_loops(seq_len):>2}  {note}"
    q, kc, vc, bt, sl = build_inputs(
        seq_len, args.num_heads, args.num_kv_heads, args.head_size,
        args.block_size, dtype, device)
    try:
        run_custom(q, kc, vc, bt, sl, seq_len, args.num_kv_heads,
                   args.head_size, args.block_size, scale, dtype, device)
    except RuntimeError as exc:
        if "Unsupported npar_loops" in str(exc):
            print(f"PASS     {label}  (refused as required)")
        else:
            print(f"FAIL     {label}  wrong error: {exc}")
            failures.append((label, f"wrong error: {exc}"))
    else:
        print(f"FAIL     {label}  kernel ACCEPTED an over-ceiling length")
        failures.append((label, "guard removed -- kernel accepted npar_loops=17"))

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for label, why in failures:
            print(f"  {label}\n    {why}")
        return 1
    print("all cases passed: custom paged attention is numerically sound to 256k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
