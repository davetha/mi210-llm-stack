#!/usr/bin/env python3
"""Test fmoe using AITER's own helper functions (correct buffer sizing).

Uses aiter.fused_moe_bf16_asm.moe_sorting_ck + aiter.fmoe for the full pipeline.
This is the SAME path that ATOM uses internally.
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
        return {"status": cls, "error": msg[:600]}
    except Exception as e:
        return {"status": type(e).__name__, "error": str(e)[:600]}


def test_fmoe_via_helper():
    """Use moe_sorting_ck helper (same path ATOM uses)."""
    print("\n=== FMOE VIA AITER HELPER (moe_sorting_ck + aiter.fmoe) ===")
    import aiter
    from aiter.fused_moe_bf16_asm import moe_sorting_ck, asm_moe, BLOCK_SIZE_M
    from aiter import ActivationType

    # MiMo shape: 256 experts, top-8, hidden=7168, inter=2048
    # But inter_dim*2 for gate+up concat in w1
    M = 64
    E = 256
    K = 8
    H = 7168  # model_dim
    I = 2048  # inter_dim (per-expert)

    print(f"\nConfig: M={M}, E={E}, K={K}, H={H}, I={I}")

    # Create inputs
    hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") * 0.1
    gating = torch.randn(M, E, dtype=torch.float32, device="cuda")

    # Weights - AITER expects:
    # w1: [E, inter_dim*2, dim]  (gate+up concatenated, N,K layout)
    # w2: [E, dim, inter_dim]
    w1 = torch.randn(E, I * 2, H, dtype=torch.bfloat16, device="cuda") * 0.01
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * 0.01
    print(f"  w1: {w1.shape}, w2: {w2.shape}")

    # Step 1: topk_softmax_asm
    print("\n[1/3] topk_softmax_asm")
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    tei = torch.empty(M * K, dtype=torch.int32, device="cuda")
    r = safe_call(lambda: aiter.topk_softmax_asm(topk_w, topk_i, tei, gating, True))
    print(f"  {r['status']}")
    if r["status"] != "PASS":
        return r

    # Step 2: moe_sorting_ck (the helper that allocates proper buffers)
    print("\n[2/3] moe_sorting_ck (helper allocates buffers)")
    r = safe_call(lambda: moe_sorting_ck(topk_i, topk_w, E, H, torch.bfloat16, BLOCK_SIZE_M))
    print(f"  {r['status']}")
    if r["status"] != "PASS":
        print(f"  err: {r.get('error', '')[:400]}")
        return r
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = r["result"]
    print(f"  sorted_ids: {sorted_ids.shape}")
    print(f"  sorted_weights: {sorted_weights.shape}")
    print(f"  sorted_expert_ids: {sorted_expert_ids.shape}")
    print(f"  num_valid_ids: {num_valid_ids.tolist()}")
    print(f"  moe_buf: {moe_buf.shape}")

    # Step 3: aiter.fmoe (the unified API)
    print("\n[3/3] aiter.fmoe")
    r = safe_call(lambda: aiter.fmoe(
        moe_buf, hidden, w1, w2,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, K
    ))
    print(f"  {r['status']}")
    if r["status"] == "PASS":
        print(f"  moe_buf after: shape={moe_buf.shape}, mean={moe_buf.mean().item():.4f}")
        has_nan = torch.isnan(moe_buf).any().item()
        has_inf = torch.isinf(moe_buf).any().item()
        print(f"  has_nan: {has_nan}, has_inf: {has_inf}")
        r["has_nan"] = has_nan
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


def test_asm_moe_highlevel():
    """Use the high-level asm_moe function which does everything."""
    print("\n=== ASM_MOE HIGH-LEVEL (aiter.fused_moe_bf16_asm.asm_moe) ===")
    from aiter.fused_moe_bf16_asm import asm_moe
    from aiter import ActivationType

    M, E, K = 64, 256, 8
    H, I = 7168, 2048

    hidden = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") * 0.1
    gating = torch.randn(M, E, dtype=torch.float32, device="cuda")
    w1 = torch.randn(E, I * 2, H, dtype=torch.bfloat16, device="cuda") * 0.01
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * 0.01

    # Get routing
    import aiter
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    tei = torch.empty(M * K, dtype=torch.int32, device="cuda")
    aiter.topk_softmax_asm(topk_w, topk_i, tei, gating, True)

    print(f"  Calling asm_moe(hidden, w1, w2, topk_w, topk_i)")
    r = safe_call(lambda: asm_moe(
        hidden, w1, w2, topk_w, topk_i,
        a16=False,
        activation=ActivationType.Silu
    ))
    print(f"  result: {r['status']}")
    if r["status"] == "PASS":
        out = r.get("result")
        if out is not None:
            print(f"  out shape: {out.shape if hasattr(out, 'shape') else type(out)}")
            if hasattr(out, "shape"):
                print(f"  out mean: {out.mean().item():.4f}")
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = {}
    for name, fn in [
        ("fmoe_helper", test_fmoe_via_helper),
        ("asm_moe_highlevel", test_asm_moe_highlevel),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\nTOP EXC: {e}")
            traceback.print_exc()
            results[name] = {"status": "EXC", "error": str(e)[:300]}

    print("\n" + "=" * 60)
    print("SUMMARY")
    for n, r in results.items():
        print(f"  {n}: {r.get('status', '?')}")

    with open("/tmp/fmoe_helper_result.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
