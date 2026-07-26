"""Definitive test: pa_fwd_asm through ATOM's FULL pipeline.

Strategy: Monkey-patch reshape_and_cache to write data in the CORRECT
physical layout (matching what pa_fwd_asm reads), then see if the kernel works.

If this works: the issue is reshape_and_cache writing in wrong layout.
If this faults: the issue is something else entirely.
"""
import os
os.environ.setdefault("HSA_COREDUMP_PATTERN", "/tmp/coredump_%p.log")

import triton
import torch
import aiter

# Monkey-patch reshape_and_cache to log and verify writes
_orig_rac = aiter.reshape_and_cache
_rac_calls = [0]

def traced_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                               kv_cache_dtype="auto", k_scale=None, v_scale=None,
                               asm_layout=False):
    _rac_calls[0] += 1
    n = _rac_calls[0]
    print(f"\n[reshape_and_cache #{n}]", flush=True)
    print(f"  key: {key.shape}, dtype={key.dtype}", flush=True)
    print(f"  value: {value.shape}, dtype={value.dtype}", flush=True)
    print(f"  key_cache: {key_cache.shape}, strides={key_cache.stride()}", flush=True)
    print(f"  value_cache: {value_cache.shape}, strides={value_cache.stride()}", flush=True)
    print(f"  slot_mapping: {slot_mapping.shape}, sample={slot_mapping[:10].tolist()}", flush=True)
    print(f"  asm_layout: {asm_layout}", flush=True)
    print(f"  kv_cache_dtype: {kv_cache_dtype}", flush=True)

    # Call original
    result = _orig_rac(key, value, key_cache, value_cache, slot_mapping,
                       kv_cache_dtype, k_scale, v_scale, asm_layout)
    print(f"  write done", flush=True)
    return result

aiter.reshape_and_cache = traced_reshape_and_cache

# Also trace pa_fwd_asm
_orig_pa = aiter.pa_fwd_asm
_pa_calls = [0]

def traced_pa_fwd_asm(Q, K, V, block_tables, context_lens, block_tables_stride0,
                       max_qlen=1, K_QScale=None, V_QScale=None, out_=None,
                       qo_indptr=None, high_precision=1, kernelName=None):
    _pa_calls[0] += 1
    n = _pa_calls[0]
    print(f"\n[pa_fwd_asm #{n}]", flush=True)
    print(f"  Q: {Q.shape}, strides={Q.stride()}", flush=True)
    print(f"  K: {K.shape}, strides={K.stride()}", flush=True)
    print(f"  V: {V.shape}, strides={V.stride()}", flush=True)
    print(f"  block_tables: {block_tables.shape}, max={block_tables.max().item()}", flush=True)
    print(f"  context_lens: {context_lens.tolist()}", flush=True)
    print(f"  stride0={block_tables_stride0}, max_qlen={max_qlen}, hp={high_precision}", flush=True)

    # Check: is K contiguous?
    print(f"  K is_contiguous: {K.is_contiguous()}", flush=True)
    print(f"  V is_contiguous: {V.is_contiguous()}", flush=True)

    # Check block validity
    k_blocks = K.shape[0]
    bt_max = block_tables.max().item()
    print(f"  K blocks={k_blocks}, max_bt={bt_max}", flush=True)
    if bt_max >= k_blocks:
        print(f"  *** BLOCK TABLE OVERFLOW: bt_max={bt_max} >= k_blocks={k_blocks}", flush=True)

    print(f"  Calling kernel...", flush=True)
    result = _orig_pa(Q, K, V, block_tables, context_lens, block_tables_stride0,
                      max_qlen, K_QScale, V_QScale, out_, qo_indptr,
                      high_precision, kernelName)
    print(f"  kernel returned OK", flush=True)
    return result

aiter.pa_fwd_asm = traced_pa_fwd_asm

# Patch in ATOM modules too
import atom.model_ops.base_attention as ba
ba.aiter.reshape_and_cache = traced_reshape_and_cache
ba.aiter.pa_fwd_asm = traced_pa_fwd_asm
import atom.model_ops.attention_mha as am
am.aiter.reshape_and_cache = traced_reshape_and_cache
am.aiter.pa_fwd_asm = traced_pa_fwd_asm

print("=== Trace patches installed ===", flush=True)
print("Will log all reshape_and_cache and pa_fwd_asm calls", flush=True)
