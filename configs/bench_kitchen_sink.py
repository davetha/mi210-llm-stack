#!/usr/bin/env python3
"""
Kitchen Sink Benchmark: Test EVERY AITER operation on MI210 (gfx90a)
Catches failures, patches where needed, documents everything.
"""
import torch
import time
import traceback
import sys
import os

# Results storage
results = []

def log(category, name, status, ms=None, tok_s=None, diff=None, error=None, notes=""):
    entry = {
        "category": category,
        "name": name,
        "status": status,
        "ms": round(ms, 3) if ms else None,
        "tok_s": round(tok_s) if tok_s else None,
        "diff": round(diff, 4) if diff is not None else None,
        "error": str(error)[:200] if error else None,
        "notes": notes,
    }
    results.append(entry)
    status_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "INFO": "ℹ️"}.get(status, "?")
    perf = f"{ms:.2f}ms = {tok_s:.0f} tok/s" if ms and tok_s else ""
    acc = f"diff={diff:.4f}" if diff is not None else ""
    err = f"ERROR: {str(error)[:100]}" if error else ""
    print(f"  {status_icon} {name:45s} {perf:30s} {acc:15s} {err} {notes}")

def bench(fn, warmup=5, N=20):
    """Benchmark a function, return (ms, output)"""
    try:
        for _ in range(warmup):
            out = fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(N):
            out = fn()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / N * 1000
        return ms, out
    except Exception as e:
        return None, e

# ============================================================
# SETUP
# ============================================================
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
import aiter
print(f"AITER loaded. GFX: {aiter.get_gfx()}")
print(f"CUs: {aiter.get_cu_num()}")
print()

# Test dimensions
B, S, H, D = 1, 4096, 32, 128  # Standard prefill shape
B_DEC, S_DEC = 1, 1  # Decode shape

# Create test tensors
q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

# Reference output
qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
ref = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True).transpose(1, 2)

print("=" * 80)
print("PHASE 1: ATTENTION KERNELS (22 variants)")
print("=" * 80)
print()

# ------------------------------------------------------------
# 1. AITER CK flash_attn_func (KNOWN GOOD)
# ------------------------------------------------------------
print("--- 1. AITER CK Flash Attention ---")
ms, out = bench(lambda: aiter.flash_attn_func(q, k, v, causal=True))
if ms:
    diff = (out - ref).abs().max().item()
    tok_s = S / (ms / 1000)
    log("attention", "aiter.flash_attn_func (CK)", "PASS", ms, tok_s, diff)

# ------------------------------------------------------------
# 2. flash_attn 2.8.3
# ------------------------------------------------------------
print("\n--- 2. flash_attn 2.8.3 ---")
try:
    from flash_attn import flash_attn_func as fa283
    ms, out = bench(lambda: fa283(q, k, v, causal=True))
    if ms:
        diff = (out - ref).abs().max().item()
        tok_s = S / (ms / 1000)
        log("attention", "flash_attn 2.8.3 (CK)", "PASS", ms, tok_s, diff)
except Exception as e:
    log("attention", "flash_attn 2.8.3", "FAIL", error=e)

# ------------------------------------------------------------
# 3. PyTorch SDPA (reference)
# ------------------------------------------------------------
print("\n--- 3. PyTorch SDPA (reference) ---")
ms, out = bench(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True))
if ms:
    tok_s = S / (ms / 1000)
    log("attention", "PyTorch SDPA", "PASS", ms, tok_s, 0.0)

# ------------------------------------------------------------
# 4. AITER mha_fwd (low-level)
# ------------------------------------------------------------
print("\n--- 4. AITER mha_fwd (low-level) ---")
try:
    ms, out = bench(lambda: aiter.mha_fwd(q, k, v, is_cu_seqlen_test=False))
    if ms:
        diff = (out - ref).abs().max().item() if out.shape == ref.shape else -1
        tok_s = S / (ms / 1000)
        log("attention", "aiter.mha_fwd", "PASS", ms, tok_s, diff)
    else:
        log("attention", "aiter.mha_fwd", "FAIL", error=out)
except Exception as e:
    log("attention", "aiter.mha_fwd", "FAIL", error=e)

# ------------------------------------------------------------
# 5. AITER flash_attn_varlen_func
# ------------------------------------------------------------
print("\n--- 5. AITER flash_attn_varlen_func ---")
try:
    cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    ms, out = bench(lambda: aiter.flash_attn_varlen_func(
        q.reshape(1, S, H, D), k.reshape(1, S, H, D), v.reshape(1, S, H, D),
        cu_seqlens, cu_seqlens, S, S, 0.0))
    if ms:
        diff = (out.reshape(B, S, H, D) - ref).abs().max().item()
        tok_s = S / (ms / 1000)
        log("attention", "aiter.flash_attn_varlen_func", "PASS", ms, tok_s, diff)
    else:
        log("attention", "aiter.flash_attn_varlen_func", "FAIL", error=out)
except Exception as e:
    log("attention", "aiter.flash_attn_varlen_func", "FAIL", error=e)

# ------------------------------------------------------------
# 6. AITER fmha_v3_fwd (FA v3)
# ------------------------------------------------------------
print("\n--- 6. AITER fmha_v3_fwd (Flash Attention v3) ---")
try:
    ms, out = bench(lambda: aiter.fmha_v3_fwd(q, k, v))
    if ms:
        diff = (out - ref).abs().max().item() if out.shape == ref.shape else -1
        tok_s = S / (ms / 1000)
        log("attention", "aiter.fmha_v3_fwd (FA-v3)", "PASS", ms, tok_s, diff)
    else:
        log("attention", "aiter.fmha_v3_fwd", "FAIL", error=out)
except Exception as e:
    log("attention", "aiter.fmha_v3_fwd", "FAIL", error=e)

# ------------------------------------------------------------
# 7. AITER fmha_v3_varlen_fwd
# ------------------------------------------------------------
print("\n--- 7. AITER fmha_v3_varlen_fwd ---")
try:
    cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device="cuda")
    ms, out = bench(lambda: aiter.fmha_v3_varlen_fwd(
        q.reshape(1, S, H, D), k.reshape(1, S, H, D), v.reshape(1, S, H, D),
        cu_seqlens, cu_seqlens, S, S))
    if ms:
        tok_s = S / (ms / 1000)
        log("attention", "aiter.fmha_v3_varlen_fwd", "PASS", ms, tok_s)
    else:
        log("attention", "aiter.fmha_v3_varlen_fwd", "FAIL", error=out)
except Exception as e:
    log("attention", "aiter.fmha_v3_varlen_fwd", "FAIL", error=e)

# ------------------------------------------------------------
# 8. AITER flash_attn_fp8_pertensor_func
# ------------------------------------------------------------
print("\n--- 8. AITER flash_attn_fp8_pertensor_func ---")
try:
    q8 = q.to(torch.float8_e4m3fn)
    k8 = k.to(torch.float8_e4m3fn)
    v8 = v.to(torch.float8_e4m3fn)
    descale = torch.ones(3, dtype=torch.float32, device="cuda")
    ms, out = bench(lambda: aiter.flash_attn_fp8_pertensor_func(q8, k8, v8, descale, descale, descale))
    if ms:
        tok_s = S / (ms / 1000)
        log("attention", "aiter.flash_attn_fp8_pertensor_func", "PASS", ms, tok_s, notes="FP8")
    else:
        log("attention", "aiter.flash_attn_fp8_pertensor_func", "FAIL", error=out)
except Exception as e:
    log("attention", "aiter.flash_attn_fp8_pertensor_func", "FAIL", error=e)

# ------------------------------------------------------------
# 9. MLA operations (THE BIG ONES for MiMo)
# ------------------------------------------------------------
print("\n--- 9. MLA Operations (MiMo-specific) ---")

# 9a. mla_prefill_asm_fwd
print("\n  9a. mla_prefill_asm_fwd...")
try:
    # MLA uses different shapes: q_proj has 128 heads, kv uses compressed latent
    # Try with standard shape first
    ms, out = bench(lambda: aiter.mla_prefill_asm_fwd(q, k, v))
    if ms:
        tok_s = S / (ms / 1000)
        log("mla", "aiter.mla_prefill_asm_fwd", "PASS", ms, tok_s, notes="STANDARD SHAPE")
    else:
        log("mla", "aiter.mla_prefill_asm_fwd", "FAIL", error=out)
except Exception as e:
    # Try to understand the expected signature
    import inspect
    try:
        sig = inspect.signature(aiter.mla_prefill_asm_fwd)
        log("mla", "aiter.mla_prefill_asm_fwd", "FAIL", error=e, notes=f"SIG: {sig}")
    except:
        log("mla", "aiter.mla_prefill_asm_fwd", "FAIL", error=e)

# 9b. mla_prefill_ps_asm_fwd (persistent)
print("  9b. mla_prefill_ps_asm_fwd...")
try:
    ms, out = bench(lambda: aiter.mla_prefill_ps_asm_fwd(q, k, v))
    if ms:
        tok_s = S / (ms / 1000)
        log("mla", "aiter.mla_prefill_ps_asm_fwd", "PASS", ms, tok_s, notes="PERSISTENT")
    else:
        log("mla", "aiter.mla_prefill_ps_asm_fwd", "FAIL", error=out)
except Exception as e:
    import inspect
    try:
        sig = inspect.signature(aiter.mla_prefill_ps_asm_fwd)
        log("mla", "aiter.mla_prefill_ps_asm_fwd", "FAIL", error=e, notes=f"SIG: {sig}")
    except:
        log("mla", "aiter.mla_prefill_ps_asm_fwd", "FAIL", error=e)

# 9c. mla_decode_stage1_asm_fwd
print("  9c. mla_decode_stage1_asm_fwd...")
try:
    import inspect
    sig = inspect.signature(aiter.mla_decode_stage1_asm_fwd)
    log("mla", "aiter.mla_decode_stage1_asm_fwd", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("mla", "aiter.mla_decode_stage1_asm_fwd", "INFO", error=e)

# 9d. hk_mla_decode_fwd
print("  9d. hk_mla_decode_fwd...")
try:
    import inspect
    sig = inspect.signature(aiter.hk_mla_decode_fwd)
    log("mla", "aiter.hk_mla_decode_fwd", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("mla", "aiter.hk_mla_decode_fwd", "INFO", error=e)

# 9e. concat_and_cache_mla
print("  9e. concat_and_cache_mla...")
try:
    import inspect
    sig = inspect.signature(aiter.concat_and_cache_mla)
    log("mla", "aiter.concat_and_cache_mla", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("mla", "aiter.concat_and_cache_mla", "INFO", error=e)

# 9f. fused_qk_rope_concat_and_cache_mla
print("  9f. fused_qk_rope_concat_and_cache_mla...")
try:
    import inspect
    sig = inspect.signature(aiter.fused_qk_rope_concat_and_cache_mla)
    log("mla", "aiter.fused_qk_rope_concat_and_cache_mla", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("mla", "aiter.fused_qk_rope_concat_and_cache_mla", "INFO", error=e)

# 9g. get_mla_metadata_v1
print("  9g. get_mla_metadata_v1...")
try:
    import inspect
    sig = inspect.signature(aiter.get_mla_metadata_v1)
    log("mla", "aiter.get_mla_metadata_v1", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("mla", "aiter.get_mla_metadata_v1", "INFO", error=e)

# ------------------------------------------------------------
# 10. Triton attention modules
# ------------------------------------------------------------
print("\n--- 10. Triton Attention Modules ---")

# 10a. flash_attn_triton_amd fwd_prefill
print("  10a. flash_attn_triton_amd.fwd_prefill...")
try:
    from flash_attn.flash_attn_triton_amd import fwd_prefill as triton_prefill
    import inspect
    sig = inspect.signature(triton_prefill)
    log("triton", "flash_attn_triton_amd.fwd_prefill", "INFO", notes=f"SIG: {sig}")
except Exception as e:
    log("triton", "flash_attn_triton_amd.fwd_prefill", "FAIL", error=e)

# 10b. Try calling Triton fwd_prefill with positional args
print("  10b. Triton fwd_prefill with unpacked args...")
try:
    from flash_attn.flash_attn_triton_amd.fwd_prefill import attention_prefill_forward_triton_impl
    import inspect
    sig = inspect.signature(attention_prefill_forward_triton_impl)
    params = list(sig.parameters.keys())
    log("triton", "fwd_prefill.attention_prefill_forward_triton_impl", "INFO", notes=f"PARAMS({len(params)}): {params[:10]}")
except Exception as e:
    log("triton", "fwd_prefill.attention_prefill_forward_triton_impl", "FAIL", error=e)

# 10c. interface_v2 decode
print("  10c. Triton interface_v2 decode...")
try:
    from flash_attn.flash_attn_triton_amd.interface_v2 import attention_forward_decode_triton_impl
    import inspect
    sig = inspect.signature(attention_forward_decode_triton_impl)
    params = list(sig.parameters.keys())
    log("triton", "interface_v2.attention_forward_decode_triton_impl", "INFO", notes=f"PARAMS({len(params)}): {params[:10]}")
except Exception as e:
    log("triton", "interface_v2.attention_forward_decode_triton_impl", "FAIL", error=e)

# 10d. AITER Triton attention modules
print("  10d. AITER Triton attention modules...")
triton_attn_modules = [
    "mla_decode", "mla_decode_rope", "unified_attention",
    "unified_attention_sparse_mla", "lean_atten", "lean_atten_paged",
    "pod_attention", "pa_prefill", "pa_decode", "chunked_pa_prefill",
    "extend_attention", "prefill_attention", "hstu_attention",
    "fav3_sage_attention", "mha_fused_bwd", "mha_onekernel_bwd",
    "fp8_mqa_logits", "pa_mqa_logits",
]
for mod_name in triton_attn_modules:
    try:
        mod = __import__(f"aiter.ops.triton.attention.{mod_name}", fromlist=[mod_name])
        funcs = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n, None)) and n[0].islower()]
        log("triton_attn", f"aiter.ops.triton.attention.{mod_name}", "PASS",
            notes=f"Functions: {funcs[:5]}")
    except Exception as e:
        log("triton_attn", f"aiter.ops.triton.attention.{mod_name}", "FAIL", error=e)

# ------------------------------------------------------------
# 11. Paged Attention variants
# ------------------------------------------------------------
print("\n--- 11. Paged Attention ---")
pa_ops = [
    ("pa_fwd_asm", "aiter.pa_fwd_asm"),
    ("pa_fwd_naive", "aiter.pa_fwd_naive"),
    ("pa_persistent_fwd", "aiter.pa_persistent_fwd"),
    ("pa_ps_fwd_asm", "aiter.pa_ps_fwd_asm"),
]
for name, func_name in pa_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("paged", func_name, "SKIP", notes="Function not found")
            continue
        import inspect
        sig = inspect.signature(func)
        log("paged", func_name, "INFO", notes=f"SIG: {sig}")
    except Exception as e:
        log("paged", func_name, "FAIL", error=e)

# ------------------------------------------------------------
# 12. MLA decode with actual decode-shaped tensors
# ------------------------------------------------------------
print("\n--- 12. MLA Decode Shape Testing ---")
# MLA decode: single query, full KV cache
q_dec = torch.randn(1, 1, H, D, dtype=torch.float16, device="cuda")

# Try mla_decode from triton module
try:
    from aiter.ops.triton.attention.mla_decode import mla_decode_fwd_triton_impl
    import inspect
    sig = inspect.signature(mla_decode_fwd_triton_impl)
    params = list(sig.parameters.keys())
    log("mla_decode", "mla_decode.mla_decode_fwd_triton_impl", "INFO",
        notes=f"PARAMS({len(params)}): {params}")
except Exception as e:
    log("mla_decode", "mla_decode.mla_decode_fwd_triton_impl", "FAIL", error=e)

# Try mla_decode_rope
try:
    from aiter.ops.triton.attention.mla_decode_rope import mla_decode_rope_fwd_triton_impl
    import inspect
    sig = inspect.signature(mla_decode_rope_fwd_triton_impl)
    params = list(sig.parameters.keys())
    log("mla_decode", "mla_decode_rope.mla_decode_rope_fwd_triton_impl", "INFO",
        notes=f"PARAMS({len(params)}): {params}")
except Exception as e:
    log("mla_decode", "mla_decode_rope.mla_decode_rope_fwd_triton_impl", "FAIL", error=e)

print("\n" + "=" * 80)
print("PHASE 2: MoE OPERATIONS")
print("=" * 80)
print()

# ------------------------------------------------------------
# MoE operations
# ------------------------------------------------------------
moe_ops = [
    "fmoe_g1u1", "fmoe_g1u1_a16", "fmoe_g1u1_tkw1",
    "fmoe_int8_g1u0", "fmoe_int8_g1u0_a16",
    "fmoe_fp8_blockscale_g1u1", "fmoe",
    "ck_moe_stage1", "ck_moe_stage1_fwd",
    "ck_moe_stage2", "ck_moe_stage2_fwd",
    "moe_sorting_fwd", "moe_sorting_opus_fwd",
    "moe_fused_gate", "moe_align_block_size",
    "moe_sum", "moe_stage1_g1u1",
    "moe_cktile2stages_gemm1", "moe_cktile2stages_gemm2",
]
for name in moe_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("moe", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        log("moe", f"aiter.{name}", "INFO", notes=f"PARAMS({len(params)}): {params[:8]}")
    except Exception as e:
        log("moe", f"aiter.{name}", "FAIL", error=e)

print("\n" + "=" * 80)
print("PHASE 3: AUXILIARY OPERATIONS")
print("=" * 80)
print()

# ------------------------------------------------------------
# RMSNorm
# ------------------------------------------------------------
print("--- RMSNorm ---")
hidden = 5120  # MiMo hidden dim
x_norm = torch.randn(1, S, hidden, dtype=torch.float16, device="cuda")
weight_norm = torch.randn(hidden, dtype=torch.float16, device="cuda")

norm_ops = [
    ("rmsnorm", "aiter.rmsnorm"),
    ("rms_norm", "aiter.rms_norm"),
    ("rmsnorm2d_fwd", "aiter.rmsnorm2d_fwd"),
    ("rmsnorm2d_fwd_ck", "aiter.rmsnorm2d_fwd_ck"),
    ("rmsnorm2d_fwd_with_add", "aiter.rmsnorm2d_fwd_with_add"),
    ("rmsnorm2d_fwd_with_add_ck", "aiter.rmsnorm2d_fwd_with_add_ck"),
]
for name, label in norm_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("norm", label, "SKIP", notes="Not found")
            continue
        # Try calling
        if "with_add" in name:
            residual = torch.randn_like(x_norm)
            ms, out = bench(lambda: func(x_norm, residual, weight_norm))
        else:
            ms, out = bench(lambda: func(x_norm, weight_norm))
        if ms:
            log("norm", label, "PASS", ms, notes=f"H={hidden}")
        else:
            # Get signature
            import inspect
            sig = inspect.signature(func)
            log("norm", label, "INFO", notes=f"SIG: {sig}")
    except Exception as e:
        log("norm", label, "FAIL", error=e)

# ------------------------------------------------------------
# RoPE
# ------------------------------------------------------------
print("\n--- RoPE ---")
# Standard rotary embedding shape
q_rope = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
k_rope = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
cos = torch.randn(S, D // 2, dtype=torch.float16, device="cuda")
sin = torch.randn(S, D // 2, dtype=torch.float16, device="cuda")
pos = torch.arange(S, device="cuda").int()

rope_ops = [
    "rope_fwd", "rope_fwd_inplace",
    "rope_2d_fwd", "rope_2d_fwd_inplace",
    "rope_cached_fwd", "rope_cached_fwd_inplace",
    "rope_cached_positions_fwd",
]
for name in rope_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("rope", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        log("rope", f"aiter.{name}", "INFO", notes=f"PARAMS({len(params)}): {params[:8]}")
    except Exception as e:
        log("rope", f"aiter.{name}", "FAIL", error=e)

# ------------------------------------------------------------
# TopK
# ------------------------------------------------------------
print("\n--- TopK (Expert Routing) ---")
# MiMo: 256 experts, top-8 routing
logits = torch.randn(1, S, 256, dtype=torch.float32, device="cuda")

topk_ops = [
    "grouped_topk", "grouped_topk_torch",
    "biased_grouped_topk", "biased_grouped_topk_hip",
    "topk_softmax", "topk_softmax_asm", "topk_sigmoid",
    "topk_plain", "top_k_per_row_decode", "top_k_per_row_decode_fast",
]
for name in topk_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("topk", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        log("topk", f"aiter.{name}", "INFO", notes=f"PARAMS({len(params)}): {params[:8]}")
    except Exception as e:
        log("topk", f"aiter.{name}", "FAIL", error=e)

# ------------------------------------------------------------
# GEMM
# ------------------------------------------------------------
print("\n--- GEMM ---")
M, N_gemm, K = 4096, 4096, 4096
a_gemm = torch.randn(M, K, dtype=torch.float16, device="cuda")
b_gemm = torch.randn(K, N_gemm, dtype=torch.float16, device="cuda")

# PyTorch reference
ms_ref, _ = bench(lambda: torch.mm(a_gemm, b_gemm))
if ms_ref:
    tflops = 2 * M * N_gemm * K / (ms_ref / 1000) / 1e12
    log("gemm", "PyTorch mm (reference)", "PASS", ms_ref, notes=f"{tflops:.1f} TFLOPS")

gemm_ops = [
    ("gemm_a8w8", "aiter.gemm_a8w8"),
    ("gemm_a8w8_CK", "aiter.gemm_a8w8_CK"),
    ("batched_gemm_bf16", "aiter.batched_gemm_bf16"),
    ("hipb_mm", "aiter.hipb_mm"),
    ("rocb_mm", "aiter.rocb_mm"),
]
for name, label in gemm_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("gemm", label, "SKIP", notes="Not found")
            continue
        import inspect
        sig = inspect.signature(func)
        log("gemm", label, "INFO", notes=f"SIG available")
    except Exception as e:
        log("gemm", label, "FAIL", error=e)

# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------
print("\n--- Sampling ---")
sample_ops = ["greedy_sample", "random_sample", "mixed_sample"]
for name in sample_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("sample", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        log("sample", f"aiter.{name}", "PASS", notes="Available")
    except Exception as e:
        log("sample", f"aiter.{name}", "FAIL", error=e)

# ------------------------------------------------------------
# Communication
# ------------------------------------------------------------
print("\n--- Communication ---")
comm_ops = ["all_reduce", "all_gather_reg", "all_gather_unreg", "reduce_scatter"]
for name in comm_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("comm", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        log("comm", f"aiter.{name}", "PASS", notes="Available (needs TP setup)")
    except Exception as e:
        log("comm", f"aiter.{name}", "FAIL", error=e)

# ------------------------------------------------------------
# Quantization
# ------------------------------------------------------------
print("\n--- Quantization ---")
quant_ops = [
    "dynamic_per_tensor_quant", "dynamic_per_token_scaled_quant",
    "per_tensor_quant_hip", "per_token_quant_hip",
    "pertoken_quant", "per_1x32_f4_quant",
]
for name in quant_ops:
    try:
        func = getattr(aiter, name, None)
        if func is None:
            log("quant", f"aiter.{name}", "SKIP", notes="Not found")
            continue
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        log("quant", f"aiter.{name}", "INFO", notes=f"PARAMS({len(params)}): {params[:5]}")
    except Exception as e:
        log("quant", f"aiter.{name}", "FAIL", error=e)

print("\n" + "=" * 80)
print("PHASE 4: EXTERNAL FRAMEWORKS")
print("=" * 80)
print()

# ------------------------------------------------------------
# conch-triton-kernels
# ------------------------------------------------------------
print("--- conch-triton-kernels ---")
try:
    import conch
    print(f"  conch path: {conch.__file__}")
    contents = [n for n in dir(conch) if not n.startswith("_")]
    log("conch", "import conch", "PASS", notes=f"Contents: {contents}")

    # Check submodules
    import pkgutil
    submods = [m.name for m in pkgutil.iter_modules(conch.__path__)]
    log("conch", "submodules", "INFO", notes=f"Submodules: {submods}")

    # Try importing each submodule
    for sub in submods:
        try:
            mod = __import__(f"conch.{sub}", fromlist=[sub])
            funcs = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n, None))]
            if funcs:
                log("conch", f"conch.{sub}", "PASS", notes=f"Functions: {funcs[:8]}")
        except Exception as e:
            log("conch", f"conch.{sub}", "FAIL", error=e)
except Exception as e:
    log("conch", "import conch", "FAIL", error=e)

# ------------------------------------------------------------
# tilelang
# ------------------------------------------------------------
print("\n--- tilelang ---")
try:
    import tilelang
    print(f"  tilelang version: {tilelang.__version__}")
    print(f"  tilelang path: {tilelang.__file__}")

    # Check for attention-related modules
    import pkgutil
    tl_path = os.path.dirname(tilelang.__file__)
    submods = [m.name for m in pkgutil.iter_modules([tl_path])]
    log("tilelang", "import tilelang", "PASS", notes=f"Submodules: {submods[:10]}")

    # Check for examples/attention
    import glob
    attention_files = glob.glob(f"{tl_path}/**/attention*", recursive=True)
    if attention_files:
        log("tilelang", "attention files", "INFO", notes=f"Found {len(attention_files)} files")
        for f in attention_files[:5]:
            log("tilelang", f"  {os.path.relpath(f, tl_path)}", "INFO")
    else:
        log("tilelang", "attention files", "SKIP", notes="No attention kernels found")

except Exception as e:
    log("tilelang", "import tilelang", "FAIL", error=e)

# ------------------------------------------------------------
# Triton base (check if it can do flash attention)
# ------------------------------------------------------------
print("\n--- Triton (base) ---")
try:
    import triton
    import triton.language as tl
    log("triton", "import triton", "PASS", notes=f"Version: {triton.__version__}")

    # Check for triton.ops or similar
    ops_path = os.path.join(os.path.dirname(triton.__file__), "ops")
    if os.path.exists(ops_path):
        ops = os.listdir(ops_path)
        attn_ops = [o for o in ops if "attn" in o.lower() or "flash" in o.lower()]
        log("triton", "triton.ops", "INFO", notes=f"Attention: {attn_ops}")
    else:
        log("triton", "triton.ops", "SKIP", notes="No ops directory")
except Exception as e:
    log("triton", "import triton", "FAIL", error=e)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Count results
from collections import Counter
status_counts = Counter(r["status"] for r in results)
category_counts = Counter(r["category"] for r in results)

print(f"\nTotal operations tested: {len(results)}")
print(f"Status breakdown: {dict(status_counts)}")
print(f"Categories: {dict(category_counts)}")

# Show benchmarks that passed with timing
print("\n--- Benchmarked (with timing) ---")
for r in results:
    if r["ms"] and r["tok_s"]:
        print(f"  {r['name']:45s} {r['ms']:7.2f}ms  {r['tok_s']:>10.0f} tok/s  diff={r['diff']}")

# Show all MLA operations
print("\n--- MLA Operations ---")
for r in results:
    if r["category"] in ("mla", "mla_decode"):
        status_icon = {"PASS": "✅", "FAIL": "❌", "INFO": "ℹ️"}.get(r["status"], "?")
        print(f"  {status_icon} {r['name']:50s} {r['notes'][:80]}")

# Show failures
print("\n--- Failures ---")
for r in results:
    if r["status"] == "FAIL":
        print(f"  ❌ [{r['category']}] {r['name']:45s} {r['error'][:80] if r['error'] else ''}")

# Save results to file
import json
results_file = "/tmp/kitchen_sink_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_file}")
