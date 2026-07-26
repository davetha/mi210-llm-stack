#!/usr/bin/env python3
"""
Fix MLA metadata: use larger page_size and proper causal=False for decode.
Then benchmark the decode kernel on gfx90a.
"""
import torch, time, traceback

print(f"GPU: {torch.cuda.get_device_name(0)}")
import aiter

SEQ_LEN = 1024
NUM_HEADS_Q = 128
HEAD_DIM = 128
NUM_KV_HEADS = 1
PAGE_SIZE = 128
NUM_PAGES = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE

q = torch.randn(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
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

# Compute metadata with causal=False for decode
try:
    aiter.get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        HEAD_DIM, NUM_KV_HEADS, False,
        work_metadata_ptrs, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        PAGE_SIZE, 1,
    )
    num_works = work_indptr[-1].item()
    print(f"Metadata: {num_works} works, work_indptr={work_indptr.tolist()}")
    if num_works == 0:
        print("Still 0 works - trying with larger kv_granularity")
        aiter.get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_lens,
            HEAD_DIM, NUM_KV_HEADS, False,
            work_metadata_ptrs, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            PAGE_SIZE, 4,
        )
        num_works = work_indptr[-1].item()
        print(f"Metadata (kv_gran=4): {num_works} works")
except Exception as e:
    print(f"Metadata error: {e}")
    traceback.print_exc()
    exit(1)

output = torch.zeros(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
NUM_SPLITS = max(1, num_works)
split_data = torch.empty(NUM_SPLITS, 1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
split_lse = torch.empty(NUM_SPLITS, 1, NUM_HEADS_Q, dtype=torch.float32, device="cuda")
num_kv_splits_indptr = work_indptr.clone()

print(f"\nRunning decode (splits={NUM_SPLITS})...")
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
    print("*** DECODE COMPLETED ***")
    print(f"Output: min={output.min():.6f} max={output.max():.6f} mean={output.mean():.6f}")
    print(f"Split data: min={split_data.min():.6f} max={split_data.max():.6f}")
    print(f"LSE: min={split_lse.min():.6f} max={split_lse.max():.6f}")

    # Benchmark
    for _ in range(10):
        aiter.mla_decode_stage1_asm_fwd(
            q, kv, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
            num_kv_splits_indptr, work_info_set[:max(1,num_works)],
            work_indptr, work_info_set, 1, PAGE_SIZE, NUM_KV_HEADS,
            1.0/(HEAD_DIM**0.5), split_data, split_lse, output,
        )
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        aiter.mla_decode_stage1_asm_fwd(
            q, kv, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
            num_kv_splits_indptr, work_info_set[:max(1,num_works)],
            work_indptr, work_info_set, 1, PAGE_SIZE, NUM_KV_HEADS,
            1.0/(HEAD_DIM**0.5), split_data, split_lse, output,
        )
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 100 * 1000
    print(f"\nBenchmark: {ms:.3f}ms per decode step")
except Exception as e:
    err = str(e)[:200]
    if "Memory" in err:
        print(f"Memory Fault: {err[:100]}")
    else:
        print(f"Error: {err[:100]}")
    traceback.print_exc()
