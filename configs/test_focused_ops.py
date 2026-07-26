#!/usr/bin/env python3
"""Focused tests for individual AITER ASM kernels with correct signatures."""
import sys, os, time, traceback, json
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
               "MEM_FAULT" if any(x in l for x in ["memory fault", "memory access", "faulting"]) else
               "BAD_HANDLE" if "resource_handle" in l else
               "CRASH" if any(x in l for x in ["abort", "segfault", "core dumped"]) else
               "RUNTIME_ERROR")
        return {"status": cls, "ms": round((time.time() - t0) * 1000, 2), "error": msg[:600]}
    except Exception as e:
        return {"status": type(e).__name__, "ms": round((time.time() - t0) * 1000, 2), "error": str(e)[:600]}


def test_topk_softmax_asm():
    """Quick re-validation."""
    print("\n[Test] topk_softmax_asm")
    import aiter
    M, N, K = 4096, 256, 8
    gating = torch.randn(M, N, dtype=torch.float32, device="cuda")
    topk_w = torch.empty(M, K, dtype=torch.float32, device="cuda")
    topk_i = torch.empty(M, K, dtype=torch.int32, device="cuda")
    tei = torch.empty(M * K, dtype=torch.int32, device="cuda")
    r = safe_call(lambda: aiter.topk_softmax_asm(topk_w, topk_i, tei, gating, True))
    print(f"  need_renorm=True: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        print(f"  wsum[0]={topk_w[0].sum().item():.4f} (expect ~1.0 with renorm)")
    return r


def test_topk_per_row_decode():
    """top_k_per_row_decode_fast signature: (logits, next_n, seqLens, indices, numRows, values)"""
    print("\n[Test] top_k_per_row_decode_fast")
    import aiter

    # From signature: this is a "speculative decoding" top-k - logits is [batch, vocab_size]
    # next_n is how many next tokens to consider
    # seqLens is per-row sequence length
    # Returns top-k values and indices per row
    batch = 4
    vocab = 128256  # Llama 3 vocab size
    next_n = 8

    logits = torch.randn(batch, vocab, dtype=torch.float32, device="cuda")
    seqLens = torch.tensor([vocab] * batch, dtype=torch.int32, device="cuda")
    indices = torch.empty(batch, next_n, dtype=torch.int32, device="cuda")
    values = torch.empty(batch, next_n, dtype=torch.float32, device="cuda")

    r = safe_call(lambda: aiter.top_k_per_row_decode_fast(
        logits, next_n, seqLens, indices, batch, values
    ))
    print(f"  result: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        print(f"  top val[0]: {values[0, 0].item():.4f}, idx: {indices[0, 0].item()}")
        sorted_ok = all(values[0, i] >= values[0, i+1] for i in range(next_n - 1))
        print(f"  sorted desc: {sorted_ok}")
    else:
        print(f"  err: {r.get('error', '')[:300]}")
    return r


def test_topk_per_row_prefill():
    """top_k_per_row_prefill_fast signature: (logits, rowStarts, rowEnds, indices, values, ...)"""
    print("\n[Test] top_k_per_row_prefill_fast")
    import aiter

    # Prefill variant takes ragged batches
    M = 4096
    vocab = 128256
    next_n = 8

    logits = torch.randn(M, vocab, dtype=torch.float32, device="cuda")
    rowStarts = torch.tensor([0, 512, 1024, 2048, 4096], dtype=torch.int32, device="cuda")
    rowEnds = torch.tensor([512, 1024, 2048, 4096, 4096], dtype=torch.int32, device="cuda")
    indices = torch.empty(M, next_n, dtype=torch.int32, device="cuda")
    values = torch.empty(M, next_n, dtype=torch.float32, device="cuda")

    r = safe_call(lambda: aiter.top_k_per_row_prefill_fast(
        logits, rowStarts, rowEnds, indices, values
    ))
    print(f"  result: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        print(f"  top val[0]: {values[0, 0].item():.4f}")
    else:
        print(f"  err: {r.get('error', '')[:300]}")
    return r


def test_gemm_a16w16_asm():
    """Signature: gemm_a16w16_asm(A, B, out, bias=None, splitK=None)"""
    print("\n[Test] gemm_a16w16_asm (bf16gemm)")
    import aiter

    M, N, K = 2048, 2048, 2048
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    C_ref = A @ B

    r = safe_call(lambda: aiter.gemm_a16w16_asm(A, B, out))
    print(f"  result: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        diff = (out - C_ref).abs().max().item()
        rel = diff / C_ref.abs().max().item()
        print(f"  diff: {diff:.4f}, rel: {rel:.4f}")
        # perf
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.time()
            aiter.gemm_a16w16_asm(A, B, out)
            torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000
            print(f"    perf: {ms:.2f}ms = {2*M*N*K/ms/1e9:.1f} TFLOPS")
        r["diff"] = diff
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


def test_pa_fwd_asm():
    """Signature: pa_fwd_asm(Q, K, V, block_tables, context_lens, block_tables_stride0, ...)
    CSV only has Gqa=8 (Llama3-70B: 64Q/8KV) and Gqa=16. No MHA entry."""
    print("\n[Test] pa_fwd_asm (paged attention, GQA=8 Llama3-70B shape)")
    import aiter

    # Llama 3 70B shape: 64 Q heads, 8 KV heads, head_dim=128, GQA ratio=8
    batch = 32
    num_q_heads = 64
    num_kv_heads = 8
    head_dim = 128
    max_ctx = 4096
    page_size = 128  # KV cache page size
    pages_per_seq = max_ctx // page_size  # 32

    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    total_pages = batch * pages_per_seq
    K_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    V_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    block_tables = torch.arange(batch * pages_per_seq, dtype=torch.int32, device="cuda").reshape(batch, pages_per_seq)
    context_lens = torch.full((batch,), max_ctx, dtype=torch.int32, device="cuda")

    print(f"  Q={Q.shape}, K_cache={K_cache.shape}, gqa={num_q_heads//num_kv_heads}")
    # hp=None means use whatever default; try hp=0 first (matches CSV)
    r = safe_call(lambda: aiter.pa_fwd_asm(Q, K_cache, V_cache, block_tables, context_lens,
                                            block_tables.stride(0), high_precision=0))
    print(f"  hp=0: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        out = r.get("result")
        if hasattr(out, "shape"):
            print(f"  out shape: {out.shape}")
            print(f"  out[0,0,:4]: {out[0, 0, :4].tolist()}")
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


def test_pa_ps_fwd_asm():
    """pa_ps_fwd_asm uses CSR-style page indices. Same GQA constraints."""
    print("\n[Test] pa_ps_fwd_asm (persistent split, GQA=8)")
    import aiter

    batch = 32
    num_q_heads = 64
    num_kv_heads = 8
    head_dim = 128
    max_ctx = 4096
    page_size = 128
    pages_per_seq = max_ctx // page_size

    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    total_pages = batch * pages_per_seq
    K_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    V_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1

    kv_indptr = torch.tensor([i * pages_per_seq for i in range(batch + 1)], dtype=torch.int32, device="cuda")
    kv_page_indices = torch.arange(batch * pages_per_seq, dtype=torch.int32, device="cuda")
    context_lens = torch.full((batch,), max_ctx, dtype=torch.int32, device="cuda")
    softmax_scale = 1.0 / (head_dim ** 0.5)

    print(f"  Q={Q.shape}, K_cache={K_cache.shape}")
    r = safe_call(lambda: aiter.pa_ps_fwd_asm(
        Q, K_cache, V_cache, kv_indptr, kv_page_indices, context_lens, softmax_scale,
        high_precision=0
    ))
    print(f"  hp=0: {r['status']} ({r.get('ms')}ms)")
    if r["status"] == "PASS":
        out = r.get("result")
        if hasattr(out, "shape"):
            print(f"  out shape: {out.shape}")
            print(f"  out[0,0,:4]: {out[0, 0, :4].tolist()}")
    else:
        print(f"  err: {r.get('error', '')[:400]}")
    return r


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = {}
    for name, fn in [
        ("topk_softmax_asm", test_topk_softmax_asm),
        ("topk_per_row_decode", test_topk_per_row_decode),
        ("topk_per_row_prefill", test_topk_per_row_prefill),
        ("gemm_a16w16_asm", test_gemm_a16w16_asm),
        ("pa_fwd_asm", test_pa_fwd_asm),
        ("pa_ps_fwd_asm", test_pa_ps_fwd_asm),
    ]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  TOP-LEVEL EXC: {e}")
            traceback.print_exc()
            results[name] = {"status": "EXC", "error": str(e)[:300]}

    print("\n" + "#" * 60)
    print("SUMMARY")
    print("#" * 60)
    for name, r in results.items():
        s = r.get("status", "?") if isinstance(r, dict) else str(r)
        ms = r.get("ms", "") if isinstance(r, dict) else ""
        print(f"  {name}: {s} ({ms}ms)")

    with open("/tmp/aiter_focused_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
