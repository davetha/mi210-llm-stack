#!/usr/bin/env python3
"""
Test patched AITER ASM kernels on gfx90a. Iterative - run, see what fails, fix.

Focus on the high-level ASM dispatch APIs that internally select .co files:
- aiter.topk_softmax_asm
- aiter.fmoe_g1u1 / fmoe_g1u1_a16
- aiter.moe_stage1_g1u1
- aiter.pa_fwd_asm / pa_ps_fwd_asm
- aiter.gemm_a16w16_asm
- aiter.fmha_v3_fwd_asm (if exists)
"""
import sys, os, time, traceback, json
import torch

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("HSA_COREDUMP_PATTERN", "/tmp/coredump_aiter_%p.log")


def banner(s):
    line = "=" * 60
    print(f"\n{line}\n{s}\n{line}")


def safe_call(fn, name=""):
    t0 = time.time()
    try:
        result = fn()
        torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000
        return {"status": "PASS", "ms": round(ms, 2), "result": result}
    except RuntimeError as e:
        ms = (time.time() - t0) * 1000
        msg = str(e)
        l = msg.lower()
        if "illegal_instruction" in l or "illegal shader" in l:
            cls = "ILLEGAL_INSTR"
        elif "memory_fault" in l or "memory access" in l or "faulting address" in l:
            cls = "MEM_FAULT"
        elif "invalid_resource_handle" in l:
            cls = "BAD_HANDLE"
        elif "abort" in l or "segfault" in l or "core dumped" in l:
            cls = "CRASH"
        else:
            cls = "RUNTIME_ERROR"
        return {"status": cls, "ms": round(ms, 2), "error": msg[:800]}
    except Exception as e:
        ms = (time.time() - t0) * 1000
        return {"status": type(e).__name__, "ms": round(ms, 2), "error": str(e)[:800]}


def inspect_sig(fn_name, fn):
    """Try to get the function signature."""
    import inspect
    try:
        sig = inspect.signature(fn)
        return str(sig)
    except (ValueError, TypeError):
        # C extension - try docstring
        doc = (fn.__doc__ or "").strip().split("\n")[0] if fn.__doc__ else "no doc"
        return doc


# =============================================================
# 1. TOPK SOFTMAX ASM
# =============================================================
def test_topk_softmax_asm():
    banner("TEST 1: aiter.topk_softmax_asm (22 .co files patched)")
    import aiter

    # Correct signature from error message:
    # topk_softmax_asm(topk_weights, topk_indices, token_expert_indices, gating_output, need_renorm)
    # token_expert_indices is a pre-allocated output for the flattened token-expert pairs

    configs = [
        ("smoke_4x128x8", 4, 128, 8),     # matches .co name topksoftmax_4x128x8
        ("mimo_4096x256x8", 4096, 256, 8),# MiMo shape
    ]

    results = []
    for name, M, N, K in configs:
        print(f"\n  [{name}] M={M}, N={N}, K={K}")
        gating = torch.randn(M, N, dtype=torch.float32, device="cuda")
        topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
        topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
        # token_expert_indices: flattened [M*K] of (token, expert) pairs encoded as token*E + expert
        # Or just pre-allocated buffer of size M*K
        token_expert_indices = torch.empty(M * K, dtype=torch.int32, device="cuda")

        r = safe_call(lambda: aiter.topk_softmax_asm(
            topk_w, topk_i, token_expert_indices, gating, False
        ), name=f"topk_softmax_asm({M}x{N},k={K})")
        print(f"    -> {r['status']} ({r.get('ms', 0)}ms)")
        if r["status"] == "PASS":
            print(f"    weights[0]: {topk_w[0][:K].tolist()}")
            print(f"    indices[0]: {topk_i[0][:K].tolist()}")
            wsum = topk_w[0].sum().item()
            valid_idx = (topk_i >= 0).all().item() and (topk_i < N).all().item()
            print(f"    weight sum: {wsum:.4f}, idx valid: {valid_idx}")
            r["weight_sum"] = wsum
            r["idx_valid"] = valid_idx
        else:
            print(f"    err: {r.get('error', '')[:300]}")
        results.append((name, r))
    return results


# =============================================================
# 2. TOPK PER ROW (decode + prefill)
# =============================================================
def test_topk_per_row():
    banner("TEST 2: aiter.top_k_per_row_decode_fast / prefill_fast")
    import aiter

    # Find the signature
    for variant in ["decode_fast", "prefill_fast", "decode", "prefill"]:
        api = f"top_k_per_row_{variant}"
        if hasattr(aiter, api):
            fn = getattr(aiter, api)
            # Get the actual signature by calling with wrong args
            try:
                fn()
            except Exception as e:
                print(f"  aiter.{api}: {str(e)[:200]}")

    # These probably need (output_vals, output_idx, input, K)
    M, N, K = 64, 256, 8
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.empty(M, K, dtype=torch.float32, device="cuda")
    idx = torch.empty(M, K, dtype=torch.int32, device="cuda")

    results = []
    for api in ["top_k_per_row_decode_fast", "top_k_per_row_prefill_fast"]:
        if not hasattr(aiter, api):
            continue
        print(f"\n  [{api}]")
        fn = getattr(aiter, api)
        # Try (out, idx, x, K)
        r = safe_call(lambda: fn(out, idx, x, K))
        print(f"    (out, idx, x, K): {r['status']}")
        if r["status"] == "PASS":
            print(f"    out[0]: {out[0].tolist()}")
            print(f"    idx[0]: {idx[0].tolist()}")
            # verify descending order
            sorted_check = all(out[0, i] >= out[0, i+1] for i in range(K-1))
            print(f"    sorted desc: {sorted_check}")
        else:
            print(f"    err: {r.get('error', '')[:300]}")
        results.append((api, r))
    return results


# =============================================================
# 3. BF16 GEMM ASM
# =============================================================
def test_gemm_a16w16_asm():
    banner("TEST 3: aiter.gemm_a16w16_asm (22 bf16gemm .co files patched)")
    import aiter

    # Signature: gemm_a16w16_asm(A, B, out, bias=None, splitK=None, kernelName=None, bpreshuffle=False)
    M, N, K = 2048, 2048, 2048
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = A @ B
    print(f"  ref: mean={C_ref.mean().item():.4f}, std={C_ref.std().item():.4f}")

    r = safe_call(lambda: aiter.gemm_a16w16_asm(A, B, out))
    print(f"  (A, B, out): {r['status']}")
    if r["status"] == "PASS":
        diff = (out - C_ref).abs().max().item()
        rel = diff / C_ref.abs().max().item()
        print(f"    out mean: {out.mean().item():.4f}, diff: {diff:.4f}, rel: {rel:.4f}")
        # perf
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            aiter.gemm_a16w16_asm(A, B, out)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            flops = 2 * M * N * K
            print(f"    perf: {ms:.2f}ms = {flops/ms/1e9:.1f} TFLOPS")
        r["max_diff"] = diff
        r["rel_diff"] = rel
    else:
        print(f"    err: {r.get('error', '')[:300]}")
    return r


# =============================================================
# 4. FMOE G1U1 (the BIG category - 838 .co files)
# =============================================================
def test_fmoe_g1u1():
    banner("TEST 4: aiter.fmoe_g1u1 (838 fmoe .co files patched)")
    import aiter

    # Signature from error:
    # fmoe_g1u1(out, input, gate, down, sorted_token_ids, sorted_weights, ...)
    # Needs pre-sorted routing. Use moe_sorting_fwd first.

    M, E, K = 64, 32, 8
    H, I = 2048, 1024

    hs = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(E, I, H, dtype=torch.bfloat16, device="cuda") * 0.01
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * 0.01
    gate = torch.randn(M, E, dtype=torch.float32, device="cuda")

    # Step 1: topk_softmax_asm
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty(M * K, dtype=torch.int32, device="cuda")
    print(f"  Step 1: topk_softmax_asm")
    r_topk = safe_call(lambda: aiter.topk_softmax_asm(topk_w, topk_i, token_expert_indices, gate, False))
    print(f"    -> {r_topk['status']}")
    if r_topk["status"] != "PASS":
        return r_topk

    # Step 2: moe_sorting_fwd - need to discover its signature
    print(f"  Step 2: discover moe_sorting_fwd signature")
    try:
        aiter.moe_sorting_fwd()
    except Exception as e:
        print(f"    sig: {str(e)[:300]}")

    return {"status": "STAGED", "step1_topk": r_topk, "note": "Need moe_sorting_fwd setup"}


# =============================================================
# 5. PA (paged attention) ASM
# =============================================================
def test_pa_fwd_asm():
    banner("TEST 5: aiter.pa_fwd_asm / pa_ps_fwd_asm (56 pa .co files)")
    import aiter

    for api in ["pa_fwd_asm", "pa_ps_fwd_asm"]:
        if hasattr(aiter, api):
            fn = getattr(aiter, api)
            print(f"  aiter.{api} sig: {inspect_sig(api, fn)}")

    # Paged attention forward (prefill) needs:
    # - Q, K, V tensors
    # - KV cache layout
    # - block table
    # This is complex to set up correctly without knowing exact layout.
    # Defer to ATOM integration test.

    print("  NOTE: PA ASM needs specific paged KV cache layout - will be tested via ATOM integration")
    return {"status": "DEFER", "note": "Test via ATOM paged attention path"}


# =============================================================
# 6. FUSED MOE STAGE1 (alternative entry point)
# =============================================================
def test_moe_stage1():
    banner("TEST 6: aiter.moe_stage1_g1u1 / moe_sorting_fwd")
    import aiter

    # moe_sorting_fwd prepares sorted_token_ids from gating
    # moe_stage1_g1u1 takes sorted inputs

    M, E, K = 64, 32, 8
    H, I = 2048, 1024
    hs = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn(M, E, dtype=torch.float32, device="cuda")

    # First need topk + sorting
    print("  Step 1: topk_softmax_asm to get routing")
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    r_topk = safe_call(lambda: aiter.topk_softmax_asm(topk_w, topk_i, gate))
    print(f"    topk: {r_topk['status']}")
    if r_topk["status"] != "PASS":
        return r_topk

    # moe_sorting_fwd - dispatch
    print(f"  Step 2: moe_sorting_fwd")
    print(f"    aiter.moe_sorting_fwd sig: {inspect_sig('moe_sorting_fwd', aiter.moe_sorting_fwd)}")
    r_sort = safe_call(lambda: aiter.moe_sorting_fwd(topk_w, topk_i, M, E, K),
                        name="moe_sorting_fwd")
    print(f"    sort: {r_sort['status']}")
    if r_sort["status"] != "PASS":
        print(f"    err: {r_sort.get('error', '')[:300]}")
    return r_sort


# =============================================================
# 7. Flash attention ASM (if exists)
# =============================================================
def test_fmha_asm():
    banner("TEST 7: fmha_v3_fwd ASM APIs (56 .co files)")
    import aiter

    fmha_apis = [a for a in dir(aiter) if "fmha" in a.lower() and "asm" in a.lower()]
    print(f"  fmha ASM APIs: {fmha_apis}")
    if not fmha_apis:
        # try non-asm
        fmha_apis = [a for a in dir(aiter) if "fmha" in a.lower()]
        print(f"  all fmha APIs: {fmha_apis[:20]}")

    # We have CK flash attention working, ASM may not be exposed directly
    return {"status": "DEFER", "note": "Use CK flash_attn_func which works at 2M tok/s"}


# =============================================================
# Main
# =============================================================
if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}
    tests = [
        ("topk_softmax_asm", test_topk_softmax_asm),
        ("topk_per_row", test_topk_per_row),
        ("gemm_a16w16_asm", test_gemm_a16w16_asm),
        ("fmoe_g1u1", test_fmoe_g1u1),
        ("moe_stage1", test_moe_stage1),
        ("pa_fwd_asm", test_pa_fwd_asm),
        ("fmha_asm", test_fmha_asm),
    ]

    for name, fn in tests:
        try:
            r = fn()
            results[name] = r
        except Exception as e:
            print(f"\n  TOP-LEVEL EXCEPTION in {name}: {e}")
            traceback.print_exc()
            results[name] = {"status": "TOP_EXC", "error": str(e)[:400]}

    print("\n" + "#" * 60)
    print("# FINAL SUMMARY")
    print("#" * 60)
    for name, r in results.items():
        if isinstance(r, list):
            print(f"  {name}:")
            for item in r:
                if len(item) == 3:
                    api, label, rr = item
                    print(f"    {api}{label}: {rr.get('status', '?')}")
                else:
                    print(f"    {item}")
        else:
            print(f"  {name}: {r.get('status', '?') if isinstance(r, dict) else r}")

    with open("/tmp/aiter_asm_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed: /tmp/aiter_asm_test_results.json")
