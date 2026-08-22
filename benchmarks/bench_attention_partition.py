#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for the two attention partitioning patches on gfx90a.

Measures the kernels in isolation, with the partitioner toggled by environment
variable, so the result is not confounded by the rest of the model. Both paths
are exercised because a spec-decode deployment only ever reaches the second:

  decode  -- kernel_paged_attention_2d, query_len == 1
             (VLLM_TRITON_PA_SEQ_PARTITION)
  verify  -- context_attention_fwd's cached-context scan, query_len == n + 1
             (VLLM_TRITON_VERIFY_CTX_PARTITION)

Shapes default to production's: block_size 784 (attention aligned to the mamba
page size of a hybrid GDN model), head_size 256, hybrid interleaved KV.

The block table is sized from --ctx-bound rather than from the sequence being
measured. That matters: the partition count is derived from the block table's
width, because a captured cudagraph must not have its grid sized from a runtime
sequence length. Benchmarking with a table sized to the actual sequence would
report a partition count no real deployment ever gets.

Usage:
    python3 build/bench_attention_partition.py
    python3 build/bench_attention_partition.py --path verify --batch 1 4
"""

import argparse
import math
import os
import time

import torch

DTYPE = torch.bfloat16


def _make_cache(num_blocks, block_size, num_kv_heads, head_size, device):
    """Hybrid interleaved KV, as _update_hybrid_attention_mamba_layout builds."""
    from vllm.v1.attention.ops.paged_attn import PagedAttention

    kv = torch.randn(2, num_blocks, block_size, num_kv_heads, head_size,
                     dtype=DTYPE, device=device) * 0.5
    hidden = kv.shape[2:].numel()
    kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
    return PagedAttention.split_kv_cache(kv, num_kv_heads, head_size)


def _blocks_for(ctx_bound, seq_len, block_size):
    return max((seq_len + block_size - 1) // block_size,
               (ctx_bound + block_size - 1) // block_size)


def _time(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1e3


def bench_decode(args, device):
    from vllm.v1.attention.ops import chunked_prefill_paged_decode as mod

    nqh = args.kv_heads * args.queries_per_kv
    print("%9s %6s | %11s %11s %8s %s"
          % ("seq_len", "batch", "unpart", "partitioned", "speedup", "part"))
    for seq_len in args.seq_lens:
        for batch in args.batch:
            torch.manual_seed(0)
            bps = _blocks_for(args.ctx_bound, seq_len, args.block_size)
            num_blocks = batch * bps + 4
            k_cache, v_cache = _make_cache(num_blocks, args.block_size,
                                           args.kv_heads, args.head_size, device)
            bt = torch.randperm(num_blocks, device=device)[: batch * bps]\
                .reshape(batch, bps).to(torch.int32)
            q = torch.randn(batch, nqh, args.head_size, dtype=DTYPE,
                            device=device) * 0.5
            kw = dict(
                query=q, key=None, value=None, output=torch.zeros_like(q),
                kv_cache_dtype="auto", key_cache=k_cache, value_cache=v_cache,
                block_table=bt,
                query_start_loc=torch.arange(batch + 1, dtype=torch.int32,
                                             device=device),
                seq_lens=torch.full((batch,), seq_len, dtype=torch.int32,
                                    device=device),
                max_seq_len=seq_len, max_query_len=1,
                k_scale=torch.tensor(1.0, device=device),
                v_scale=torch.tensor(1.0, device=device),
                sm_scale=1.0 / math.sqrt(args.head_size),
            )

            def run():
                mod.chunked_prefill_paged_decode(**kw)

            os.environ["VLLM_TRITON_PA_SEQ_PARTITION"] = "0"
            mod._seq_partition_enabled.cache_clear()
            base = _time(run, args.iters)
            os.environ["VLLM_TRITON_PA_SEQ_PARTITION"] = "1"
            mod._seq_partition_enabled.cache_clear()
            part_ms = _time(run, args.iters)
            psize = mod._choose_partition_size(
                batch, args.kv_heads, bps * args.block_size, 32,
                num_query_heads=nqh, head_size_padded=args.head_size)
            print("%9d %6d | %10.3fms %10.3fms %7.2fx  %d"
                  % (seq_len, batch, base, part_ms,
                     base / part_ms if part_ms else 0, psize))
    os.environ.pop("VLLM_TRITON_PA_SEQ_PARTITION", None)
    mod._seq_partition_enabled.cache_clear()


def bench_verify(args, device):
    from vllm.v1.attention.ops import prefix_prefill as mod

    nqh = args.kv_heads * args.queries_per_kv
    print("%9s %5s %6s | %11s %11s %8s %s"
          % ("ctx", "qlen", "batch", "unpart", "partitioned", "speedup", "part"))
    for ctx in args.seq_lens:
        for batch in args.batch:
            torch.manual_seed(0)
            total = ctx + args.query_len
            bps = _blocks_for(args.ctx_bound, total, args.block_size)
            num_blocks = batch * bps + 4
            k_cache, v_cache = _make_cache(num_blocks, args.block_size,
                                           args.kv_heads, args.head_size, device)
            b_loc = torch.randperm(num_blocks, device=device)[: batch * bps]\
                .reshape(batch, bps).to(torch.int32)
            tot_q = batch * args.query_len
            q = torch.randn(tot_q, nqh, args.head_size, dtype=DTYPE,
                            device=device) * 0.5
            k = torch.randn(tot_q, args.kv_heads, args.head_size, dtype=DTYPE,
                            device=device) * 0.5
            kw = dict(
                q=q, k=k, v=torch.randn_like(k), o=torch.zeros_like(q),
                kv_cache_dtype="auto", k_cache=k_cache, v_cache=v_cache,
                b_loc=b_loc,
                b_start_loc=torch.arange(0, tot_q + 1, args.query_len,
                                         dtype=torch.int32, device=device),
                b_seq_len=torch.full((batch,), total, dtype=torch.int32,
                                     device=device),
                max_seq_len=total, max_input_len=args.query_len,
                k_scale=torch.tensor(1.0, device=device),
                v_scale=torch.tensor(1.0, device=device),
                sm_scale=1.0 / math.sqrt(args.head_size),
                skip_decode=True, causal=False,
            )

            def run():
                mod.context_attention_fwd(**kw)

            os.environ["VLLM_TRITON_VERIFY_CTX_PARTITION"] = "0"
            mod._verify_partition_enabled.cache_clear()
            base = _time(run, args.iters)
            os.environ["VLLM_TRITON_VERIFY_CTX_PARTITION"] = "1"
            mod._verify_partition_enabled.cache_clear()
            part_ms = _time(run, args.iters)
            psize = mod._choose_verify_partition(
                batch, nqh, bps * args.block_size,
                block_m=max(16, 1 << (args.query_len - 1).bit_length()),
                head_dim=args.head_size)
            print("%9d %5d %6d | %10.3fms %10.3fms %7.2fx  %d"
                  % (ctx, args.query_len, batch, base, part_ms,
                     base / part_ms if part_ms else 0, psize))
    os.environ.pop("VLLM_TRITON_VERIFY_CTX_PARTITION", None)
    mod._verify_partition_enabled.cache_clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=["decode", "verify", "both"],
                   default="both")
    p.add_argument("--seq-lens", type=int, nargs="+", dest="seq_lens",
                   default=[2048, 40960, 102400, 200000])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4])
    p.add_argument("--block-size", type=int, default=784)
    p.add_argument("--head-size", type=int, default=256)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--queries-per-kv", type=int, default=8)
    p.add_argument("--query-len", type=int, default=9,
                   help="verify path only: num_speculative_tokens + 1")
    p.add_argument("--ctx-bound", type=int, default=262_144,
                   help="max_model_len; the block table is sized from this, "
                        "because that is what a captured cudagraph gets")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible")
    device = torch.device("cuda")

    print("block_size=%d head_size=%d kv_heads=%d q_per_kv=%d ctx_bound=%d"
          % (args.block_size, args.head_size, args.kv_heads,
             args.queries_per_kv, args.ctx_bound))
    if args.path in ("decode", "both"):
        print("\n== paged decode (query_len 1) ==")
        bench_decode(args, device)
    if args.path in ("verify", "both"):
        print("\n== spec-decode verify (query_len %d) ==" % args.query_len)
        bench_verify(args, device)


if __name__ == "__main__":
    main()
