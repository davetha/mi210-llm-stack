"""Monkey-patch pa_fwd_asm to log exact arguments before calling the real kernel.

This will tell us EXACTLY what ATOM passes to the kernel, so we can reproduce
the fault in a standalone test and identify the root cause.
"""
import os
os.environ.setdefault("HSA_COREDUMP_PATTERN", "/tmp/coredump_%p.log")

import triton
import torch
import aiter
import functools

# Save original
_orig_pa_fwd_asm = aiter.pa_fwd_asm
_call_count = [0]

def logged_pa_fwd_asm(Q, K, V, block_tables, context_lens, block_tables_stride0,
                       max_qlen=1, K_QScale=None, V_QScale=None, out_=None,
                       qo_indptr=None, high_precision=1, kernelName=None):
    _call_count[0] += 1
    n = _call_count[0]

    print(f"\n{'='*60}", flush=True)
    print(f"pa_fwd_asm call #{n}:", flush=True)
    print(f"  Q: shape={Q.shape}, dtype={Q.dtype}, strides={Q.stride()}", flush=True)
    print(f"  K: shape={K.shape}, dtype={K.dtype}, strides={K.stride()}", flush=True)
    print(f"  V: shape={V.shape}, dtype={V.dtype}, strides={V.stride()}", flush=True)
    print(f"  block_tables: shape={block_tables.shape}, dtype={block_tables.dtype}", flush=True)
    print(f"  block_tables values: {block_tables.flatten()[:20].tolist()}", flush=True)
    print(f"  context_lens: {context_lens.tolist()}", flush=True)
    print(f"  block_tables_stride0: {block_tables_stride0}", flush=True)
    print(f"  max_qlen: {max_qlen}", flush=True)
    print(f"  high_precision: {high_precision}", flush=True)
    if K_QScale is not None:
        print(f"  K_QScale: shape={K_QScale.shape}", flush=True)
    if V_QScale is not None:
        print(f"  V_QScale: shape={V_QScale.shape}", flush=True)
    if qo_indptr is not None:
        print(f"  qo_indptr: {qo_indptr.tolist()}", flush=True)

    # K/V memory stats
    k_elements = K.numel()
    v_elements = V.numel()
    print(f"  K total elements: {k_elements} ({k_elements*2/1e6:.1f}MB)", flush=True)
    print(f"  V total elements: {v_elements} ({v_elements*2/1e6:.1f}MB)", flush=True)

    # Check block_tables validity
    max_block_id = block_tables.max().item()
    num_blocks_k = K.shape[0]
    print(f"  Max block_tables entry: {max_block_id}, K blocks: {num_blocks_k}", flush=True)
    if max_block_id >= num_blocks_k:
        print(f"  *** WARNING: block_tables references block {max_block_id} but K only has {num_blocks_k} blocks!", flush=True)

    # Check context_lens vs available blocks
    for i, cl in enumerate(context_lens.tolist()):
        blocks_needed = (cl + 15) // 16  # ceiling division by block_size
        bt_entries = block_tables[i].tolist()
        valid_bt = [x for x in bt_entries if x >= 0 and x < num_blocks_k]
        if blocks_needed > len(valid_bt):
            print(f"  *** WARNING: seq {i} context_len={cl} needs {blocks_needed} blocks but only {len(valid_bt)} valid entries", flush=True)

    print(f"  Calling kernel...", flush=True)

    # Call original
    result = _orig_pa_fwd_asm(Q, K, V, block_tables, context_lens, block_tables_stride0,
                              max_qlen, K_QScale, V_QScale, out_, qo_indptr,
                              high_precision, kernelName)
    print(f"  Result: shape={result.shape if hasattr(result, 'shape') else type(result)}", flush=True)
    print(f"{'='*60}", flush=True)
    return result

# Install patch
aiter.pa_fwd_asm = logged_pa_fwd_asm

# Also patch in the module where ATOM imports it
import atom.model_ops.base_attention as ba
if hasattr(ba, 'aiter'):
    ba.aiter.pa_fwd_asm = logged_pa_fwd_asm
import atom.model_ops.attention_mha as am
if hasattr(am, 'aiter'):
    am.aiter.pa_fwd_asm = logged_pa_fwd_asm

print("pa_fwd_asm monkey-patch installed — will log all calls", flush=True)
