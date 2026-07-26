#!/usr/bin/env python3
"""Test MLA decode - fixed KV tensor shape (4D)"""
import torch, traceback

print(f"GPU: {torch.cuda.get_device_name(0)}")
import aiter

SEQ_LEN = 128
NUM_HEADS_Q = 128
HEAD_DIM = 128
NUM_KV_HEADS = 1
PAGE_SIZE = 1
NUM_PAGES = SEQ_LEN  # page_size=1, so each page is 1 token

q = torch.randn(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
# KV must be 4D: [num_pages, page_size, num_kv_heads, head_dim]
kv = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
kv_indptr = torch.tensor([0, NUM_PAGES], dtype=torch.int32, device="cuda")
kv_page_indices = torch.arange(NUM_PAGES, dtype=torch.int32, device="cuda")
kv_last_page_lens = torch.tensor([PAGE_SIZE], dtype=torch.int32, device="cuda")

MAX_WORKS = 256
work_metadata_ptrs = torch.empty(MAX_WORKS, dtype=torch.uint64, device="cuda")
work_info_set = torch.empty(MAX_WORKS, dtype=torch.int32, device="cuda")
work_indptr = torch.empty(2, dtype=torch.int32, device="cuda")
reduce_indptr = torch.empty(2, dtype=torch.int32, device="cuda")
reduce_final_map = torch.empty(MAX_WORKS * 64, dtype=torch.int32, device="cuda")
reduce_partial_map = torch.empty(MAX_WORKS * 64, dtype=torch.int32, device="cuda")

print(f"Config: seq={SEQ_LEN}, heads={NUM_HEADS_Q}, pages={NUM_PAGES}, page_size={PAGE_SIZE}")
print(f"Q shape: {q.shape}, KV shape: {kv.shape}")

try:
    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        HEAD_DIM, NUM_KV_HEADS, False,
        work_metadata_ptrs, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        PAGE_SIZE, 1,
    )
    num_works = work_indptr[-1].item()
    print(f"Metadata: {num_works} works")
except Exception as e:
    print(f"Metadata error: {e}")
    traceback.print_exc()
    exit(1)

output = torch.zeros(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
NUM_SPLITS = max(1, num_works)
split_data = torch.empty(NUM_SPLITS, 1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
split_lse = torch.empty(NUM_SPLITS, 1, NUM_HEADS_Q, dtype=torch.float32, device="cuda")
num_kv_splits_indptr = work_indptr.clone()

print(f"Running mla_decode_stage1_asm_fwd...")
try:
    aiter.mla_decode_stage1_asm_fwd(
        q, kv,
        qo_indptr, kv_indptr,
        kv_page_indices, kv_last_page_lens,
        num_kv_splits_indptr,
        work_info_set[:max(1,num_works)],
        work_indptr, work_info_set,
        1, PAGE_SIZE, NUM_KV_HEADS,
        1.0 / (HEAD_DIM ** 0.5),
        split_data, split_lse,
        output,
    )
    print("*** DECODE SUCCESS! MLA ASM DECODE RAN ON GFX90A! ***")
    print(f"Output: min={output.min():.4f} max={output.max():.4f} mean={output.mean():.4f}")
except Exception as e:
    err = str(e)[:200]
    if "Memory" in err or "fault" in err.lower():
        print(f"Memory Fault: {err[:120]}")
    elif "Illegal" in err:
        print(f"ILLEGAL INSTRUCTION: {err[:120]}")
    elif "heuristic" in err.lower():
        print(f"Heuristic: {err[:120]}")
    else:
        print(f"Error: {err[:120]}")
    traceback.print_exc()
