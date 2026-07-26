#!/usr/bin/env python3
"""Full MLA decode pipeline: metadata + decode + reduce = actual output"""
import torch, time, traceback

print(f"GPU: {torch.cuda.get_device_name(0)}")
import aiter

SEQ_LEN = 1024
NUM_HEADS_Q = 128
HEAD_DIM = 128
NUM_KV_HEADS = 1
PAGE_SIZE = 128
NUM_PAGES = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
NUM_KV_SPLITS = 8

q = torch.randn(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")

qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
kv_indptr = torch.tensor([0, NUM_PAGES], dtype=torch.int32, device="cuda")
kv_page_indices = torch.arange(NUM_PAGES, dtype=torch.int32, device="cuda")
kv_last_page_lens = torch.tensor([PAGE_SIZE], dtype=torch.int32, device="cuda")

# Use decode_update_mla_metadata_v1 for decode mode
MAX_WORKS = 256
work_metadata_ptrs = torch.empty(MAX_WORKS, dtype=torch.uint64, device="cuda")
work_info_set = torch.empty(MAX_WORKS, dtype=torch.int32, device="cuda")
work_indptr = torch.empty(2, dtype=torch.int32, device="cuda")
reduce_indptr = torch.empty(2, dtype=torch.int32, device="cuda")
reduce_final_map = torch.empty(MAX_WORKS * 64, dtype=torch.int32, device="cuda")
reduce_partial_map = torch.empty(MAX_WORKS * 64, dtype=torch.int32, device="cuda")
num_kv_splits_indptr = torch.empty(2, dtype=torch.int32, device="cuda")

print(f"Config: seq={SEQ_LEN}, heads={NUM_HEADS_Q}, splits={NUM_KV_SPLITS}")

try:
    aiter.decode_update_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens,
        NUM_KV_SPLITS,
        work_metadata_ptrs, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        num_kv_splits_indptr,
        PAGE_SIZE, 1,
    )
    num_works = work_indptr[-1].item()
    print(f"Decode metadata: {num_works} works")
    print(f"  work_indptr: {work_indptr.tolist()}")
    print(f"  num_kv_splits_indptr: {num_kv_splits_indptr.tolist()}")
except Exception as e:
    print(f"Trying get_mla_metadata_v1 with is_causal=True...")
    try:
        aiter.get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_lens,
            HEAD_DIM, NUM_KV_HEADS, True,
            work_metadata_ptrs, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            PAGE_SIZE, 1,
        )
        num_works = work_indptr[-1].item()
        print(f"Metadata (causal=True): {num_works} works")
        num_kv_splits_indptr = work_indptr.clone()
    except Exception as e2:
        print(f"Metadata error: {e2}")
        # Just use manual split setup
        num_works = NUM_KV_SPLITS
        work_indptr = torch.tensor([0, NUM_KV_SPLITS], dtype=torch.int32, device="cuda")
        work_info_set = torch.arange(NUM_KV_SPLITS, dtype=torch.int32, device="cuda")
        num_kv_splits_indptr = torch.tensor([0, NUM_KV_SPLITS], dtype=torch.int32, device="cuda")
        print(f"Using manual metadata: {num_works} works")

output = torch.zeros(1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
split_data = torch.empty(NUM_KV_SPLITS, 1, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
split_lse = torch.empty(NUM_KV_SPLITS, 1, NUM_HEADS_Q, dtype=torch.float32, device="cuda")

print(f"\nRunning decode stage 1...")
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
    print(f"Stage 1 done! split_data: min={split_data.min():.4f} max={split_data.max():.4f}")
    print(f"           split_lse: min={split_lse.min():.4f} max={split_lse.max():.4f}")
    print(f"           output: min={output.min():.6f} max={output.max():.6f}")

    if output.abs().max() > 0:
        print("\n*** FULL PIPELINE WORKING! Output has non-zero values! ***")
    else:
        print("\nOutput still zeros - trying mla_reduce_v1...")
        try:
            aiter.mla_reduce_v1(
                split_data, split_lse,
                reduce_indptr, reduce_final_map, reduce_partial_map,
                output,
            )
            print(f"After reduce: min={output.min():.6f} max={output.max():.6f}")
        except Exception as e:
            print(f"Reduce error: {str(e)[:100]}")

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
    print(f"\nBenchmark: {ms:.3f}ms per decode step ({1/ms*1000:.0f} steps/sec)")

except Exception as e:
    err = str(e)[:200]
    print(f"Error: {err[:100]}")
    traceback.print_exc()
