#!/usr/bin/env python3
"""
FIXED: Correct MLA shapes from AITER Python wrapper analysis
- num_kv_splits = 1 (hardcoded in wrapper)
- Q: [S, 128, 576] (head_size = kv_lora_rank + qk_rope_head_dim)
- KV: [pages, page_size, 1, 576]
- splitData: [S, 1, 128, 512] (output reshaped: bs, kv_splits, heads, v_head_dim)
- splitLse: [S, 1, 128, 1]
- sm_scale = 1/sqrt(576)
"""
import torch, aiter, time

print(f"GPU: {torch.cuda.get_device_name(0)}")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
NUM_HEADS = 128
NUM_KV_HEADS = 1
PAGE_SIZE = 128

for S in [1, 16, 64, 128, 256, 512]:
    NUM_PAGES = max(1, (S + PAGE_SIZE - 1) // PAGE_SIZE)

    q = torch.randn(S, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")

    qo = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    ki = torch.tensor([0, NUM_PAGES], dtype=torch.int32, device="cuda")
    kpi = torch.arange(NUM_PAGES, dtype=torch.int32, device="cuda")
    klpl = torch.tensor([min(S, PAGE_SIZE)], dtype=torch.int32, device="cuda")

    # splitData = output reshaped: [bs=S, kv_splits=1, heads=128, v_head_dim=512]
    sd = torch.empty(S, 1, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    sl = torch.empty(S, 1, NUM_HEADS, 1, dtype=torch.float32, device="cuda")

    sm_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    try:
        aiter.mla_prefill_asm_fwd(q, kv, qo, ki, kpi, klpl,
                                  S, sm_scale, sd, sl)
        print(f"S={S:4d}: SUCCESS! data=[{sd.min():.2f},{sd.max():.2f}] lse=[{sl.min():.2f},{sl.max():.2f}]")

        if S >= 64:
            for _ in range(5):
                aiter.mla_prefill_asm_fwd(q, kv, qo, ki, kpi, klpl, S, sm_scale, sd, sl)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                aiter.mla_prefill_asm_fwd(q, kv, qo, ki, kpi, klpl, S, sm_scale, sd, sl)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 20 * 1000
            print(f"        {ms:.2f}ms = {S/(ms/1000):.0f} tok/s")
    except Exception as e:
        print(f"S={S:4d}: FAULT: {str(e)[:80]}")
        break
