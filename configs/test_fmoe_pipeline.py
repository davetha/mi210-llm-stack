#!/usr/bin/env python3
"""Test fmoe pipeline: topk_softmax_asm -> moe_sorting_fwd -> fmoe_g1u1

Required for any MoE model (DeepSeek, MiMo, Qwen3-MoE, etc.)
"""
import sys, os, time, json, traceback
import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")


def safe_call(fn):
    t0 = time.time()
    try:
        result = fn()
        torch.cuda.synchronize()
        return {"status": "PASS", "ms": round((time.time() - t0) * 1000, 2), "result": result}
    except RuntimeError as e:
        msg = str(e)
        l = msg.lower()
        cls = ("ILLEGAL_INSTR" if "illegal" in l else
               "MEM_FAULT" if any(x in l for x in ["memory fault", "faulting"]) else
               "CRASH" if any(x in l for x in ["abort", "segfault"]) else
               "RUNTIME_ERROR")
        return {"status": cls, "error": msg[:500]}
    except Exception as e:
        return {"status": type(e).__name__, "error": str(e)[:500]}


def test_fmoe_pipeline():
    """Full MoE pipeline test."""
    print("\n=== FMOE PIPELINE TEST ===")
    import aiter

    # MiMo-shape: 256 experts, top-8 (matches topksoftmax_4x256x8.co)
    # Use 128 experts to also validate that variant
    M = 64         # tokens
    E = 256        # experts (MiMo shape - matches topksoftmax_4x256x8)
    K = 8          # top-k (MiMo)
    H = 7168       # hidden (MiMo actual)
    I = 2048       # intermediate (MiMo actual per-expert)

    print(f"\nConfig: M={M}, E={E}, K={K}, H={H}, I={I}")

    # Step 0: Create inputs
    gating = torch.randn(M, E, dtype=torch.float32, device="cuda")
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") * 0.1

    # Weight matrices - check layout expectations
    # AITER expects gate/down in specific layout. Try [E, I, H] for gate (mm: H->I)
    # and [E, H, I] for down (mm: I->H)
    gate_w = torch.randn(E, I, H, dtype=torch.bfloat16, device="cuda") * 0.01
    down_w = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * 0.01

    out = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")

    # Step 1: topk_softmax_asm
    print("\n[1/3] topk_softmax_asm")
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(M * K, dtype=torch.int32, device="cuda")
    r = safe_call(lambda: aiter.topk_softmax_asm(topk_w, topk_i, token_expert_indices, gating, True))
    print(f"  topk: {r['status']}")
    if r["status"] != "PASS":
        print(f"  err: {r.get('error', '')[:200]}")
        return r
    print(f"  wsum[0]={topk_w[0].sum().item():.4f}, idx[0]={topk_i[0].tolist()}")

    # Step 2: moe_sorting_fwd
    # Signature: (topk_ids, topk_weights, sorted_token_ids, sorted_weights,
    #             sorted_expert_ids, num_valid_ids, moe_buf, num_experts, unit_size)
    print("\n[2/3] moe_sorting_fwd")
    # Allocate sorted buffers
    # sorted_token_ids: [align(M*K, 128)] padded to 128-element boundary
    align_m_k = ((M * K + 127) // 128) * 128
    sorted_token_ids = torch.empty(align_m_k, dtype=torch.int32, device="cuda")
    sorted_weights = torch.empty(align_m_k, dtype=torch.float32, device="cuda")
    sorted_expert_ids = torch.empty(align_m_k, dtype=torch.int32, device="cuda")
    num_valid_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    # moe_buf - size depends on expert count; allocate large
    moe_buf_size = E * align_m_k * 2  # rough estimate
    moe_buf = torch.zeros(moe_buf_size, dtype=torch.int32, device="cuda")

    # unit_size - check what this means (probably element size in bytes for bf16 = 2)
    r = safe_call(lambda: aiter.moe_sorting_fwd(
        topk_i, topk_w, sorted_token_ids, sorted_weights, sorted_expert_ids,
        num_valid_ids, moe_buf, E, 2  # unit_size=2 for bf16
    ))
    print(f"  sorting: {r['status']}")
    if r["status"] != "PASS":
        print(f"  err: {r.get('error', '')[:300]}")
        # Try alternative unit_size
        for us in [1, 4, 8, 16]:
            r2 = safe_call(lambda: aiter.moe_sorting_fwd(
                topk_i, topk_w, sorted_token_ids, sorted_weights, sorted_expert_ids,
                num_valid_ids, moe_buf, E, us
            ))
            print(f"  retry unit_size={us}: {r2['status']}")
            if r2["status"] == "PASS":
                r = r2
                break
        if r["status"] != "PASS":
            return r

    # Step 3: fmoe_g1u1
    # Signature: (out, input, gate, down, sorted_token_ids, sorted_weights,
    #             sorted_expert_ids, num_valid_ids, topk, input_scale, fc1_scale, fc2_scale)
    print("\n[3/3] fmoe_g1u1")
    input_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    fc1_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    fc2_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

    r = safe_call(lambda: aiter.fmoe_g1u1(
        out, hidden, gate_w, down_w,
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        K, input_scale, fc1_scale, fc2_scale
    ))
    print(f"  fmoe: {r['status']}")
    if r["status"] == "PASS":
        print(f"  out shape: {out.shape}")
        print(f"  out[0, :8]: {out[0, :8].tolist()}")
        print(f"  out mean: {out.mean().item():.4f}, std: {out.std().item():.4f}")
        # Check for NaN/Inf
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        print(f"  has_nan: {has_nan}, has_inf: {has_inf}")
        r["has_nan"] = has_nan
        r["has_inf"] = has_inf
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        r = test_fmoe_pipeline()
        print("\n" + "=" * 60)
        print(f"FINAL: {r.get('status', '?')}")
        print("=" * 60)
        with open("/tmp/fmoe_pipeline_result.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    except Exception as e:
        traceback.print_exc()
