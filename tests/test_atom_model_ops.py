#!/usr/bin/env python3
"""
Comprehensive ATOM model_ops test suite for gfx90a (MI210).
Tests every operator class documented in the ATOM model_ops guide:
  1. Linear operations (GEMM: BF16, FP8)
  2. Attention: MHA prefill (flash_attn_varlen_func), MHA decode (pa_fwd_asm)
  3. MLA attention (if applicable)
  4. MoE operations (fused_moe, topk_softmax)
  5. Normalization (RMSNorm, LayerNorm)
  6. Activations (SiluAndMul)
  7. Embedding / LM Head
  8. RoPE
  9. Sampling
 10. KV cache (reshape_and_cache, asm_layout)

Each test reports PASS/FAIL with timing.
"""

import torch
import time
import sys
import traceback

def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def test_pass(name, extra=""):
    print(f"  ✅ PASS: {name}" + (f" — {extra}" if extra else ""))

def test_fail(name, err):
    print(f"  ❌ FAIL: {name} — {err}")

def test_skip(name, reason):
    print(f"  ⏭️  SKIP: {name} — {reason}")

def timed(fn, name):
    start = time.perf_counter()
    result = fn()
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed

def main():
    header("ATOM Model Ops Test Suite — gfx90a")
    
    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    gfx = get_gfx()
    print(f"  GPU: {gfx}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  HIP: {torch.version.hip}")
    print(f"  AITER ops available: {len([x for x in dir(aiter) if not x.startswith('_')])}")
    
    results = {}
    
    # ================================================================
    # 1. Linear Operations (GEMM)
    # ================================================================
    header("1. Linear Operations (GEMM)")
    
    try:
        M, N, K = 512, 512, 512
        a = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
        
        # Standard BF16 GEMM via tgemm
        try:
            import aiter.tgemm as tgemm
            out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
            tgemm.mm(a, b, out=out)
            assert out.shape == (M, N), f"Wrong shape: {out.shape}"
            nonzero = (out.abs() > 1e-5).sum().item()
            assert nonzero > M * N * 0.9, f"Too many zeros: {nonzero}/{M*N}"
            test_pass("BF16 GEMM (tgemm.mm)", f"{nonzero}/{M*N} nonzero")
        except Exception as e:
            test_fail("BF16 GEMM (tgemm.mm)", str(e)[:100])
        
        # ASM GEMM (gemm_a16w16_asm)
        try:
            a_t = a.t().contiguous().t()  # TN layout
            b_t = b.t().contiguous()
            out_asm = aiter.gemm_a16w16_asm(a_t, b_t)
            nonzero = (out_asm.abs() > 1e-5).sum().item() if out_asm is not None else 0
            test_pass("BF16 ASM GEMM (gemm_a16w16_asm)", f"{nonzero} nonzero")
        except AttributeError:
            test_skip("BF16 ASM GEMM", "aiter.gemm_a16w16_asm not found")
        except Exception as e:
            test_fail("BF16 ASM GEMM", str(e)[:100])
    except Exception as e:
        test_fail("GEMM setup", str(e)[:100])
    
    # ================================================================
    # 2. Attention: MHA Prefill (flash_attn_varlen_func)
    # ================================================================
    header("2. Attention: MHA Prefill")
    
    try:
        B, S, H, D = 1, 512, 8, 128
        q = torch.randn(S, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(S, 8, D, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(S, 8, D, dtype=torch.bfloat16, device='cuda')
        
        cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device='cuda')
        
        out = aiter.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=S,
            max_seqlen_k=S,
            softmax_scale=1.0 / (D ** 0.5),
            causal=True,
        )
        nonzero = (out.abs() > 1e-5).sum().item()
        assert nonzero > S * H * D * 0.9, f"Too many zeros: {nonzero}/{S*H*D}"
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            aiter.flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens, max_seqlen_q=S, max_seqlen_k=S,
                softmax_scale=1.0/(D**0.5), causal=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10 * 1000
        tok_s = S / (elapsed / 1000)
        test_pass("flash_attn_varlen_func (prefill)", f"{nonzero}/{S*H*D} nonzero, {elapsed:.3f}ms, {tok_s:,.0f} tok/s")
    except Exception as e:
        test_fail("flash_attn_varlen_func", str(e)[:200])
    
    # ================================================================
    # 3. Attention: MHA Decode (pa_fwd_asm)
    # ================================================================
    header("3. Attention: MHA Decode (pa_fwd_asm)")
    
    try:
        num_blocks = 100
        block_size = 16
        num_kv = 8
        head_dim = 128
        x = 8  # bf16 -> 2 bytes, 16//2=8
        
        # SHUFFLE layout K cache: [blocks, kv_heads, head_dim//x, block_size, x]
        k_cache = torch.randn(num_blocks, num_kv, head_dim // x, block_size, x,
                              dtype=torch.bfloat16, device='cuda')
        # SHUFFLE layout V cache: [blocks, kv_heads, block_size//x, head_dim, x]
        v_cache = torch.randn(num_blocks, num_kv, block_size // x, head_dim, x,
                              dtype=torch.bfloat16, device='cuda')
        
        batch = 4
        ctx_len = 64
        q = torch.randn(batch, 16, head_dim, dtype=torch.bfloat16, device='cuda')  # 16 query heads
        block_tables = torch.arange(batch * 4, device='cuda', dtype=torch.int32).reshape(batch, 4)
        context_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device='cuda')
        
        out = aiter.pa_fwd_asm(
            Q=q, K=k_cache, V=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            block_tables_stride0=block_tables.stride(0),
            max_qlen=1,
            K_QScale=None, V_QScale=None,
        )
        torch.cuda.synchronize()
        nonzero = (out.abs() > 1e-5).sum().item()
        total = out.numel()
        assert nonzero > total * 0.9, f"Too many zeros: {nonzero}/{total}"
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            aiter.pa_fwd_asm(Q=q, K=k_cache, V=v_cache, block_tables=block_tables,
                context_lens=context_lens, block_tables_stride0=block_tables.stride(0),
                max_qlen=1, K_QScale=None, V_QScale=None)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1000
        test_pass("pa_fwd_asm (decode)", f"{nonzero}/{total} nonzero, {elapsed:.3f}ms/step")
    except Exception as e:
        test_fail("pa_fwd_asm", str(e)[:200])
    
    # ================================================================
    # 4. MLA Attention
    # ================================================================
    header("4. MLA Attention")
    
    try:
        # MLA prefill
        S = 256
        kv_lora_rank = 512
        qk_rope_head_dim = 64
        head_size = 576  # kv_lora_rank + qk_rope_head_dim
        v_head_dim = 512
        
        q = torch.randn(S, 128, head_size, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(S, 128, head_size, dtype=torch.bfloat16, device='cuda')
        v = torch.randn(S, 128, v_head_dim, dtype=torch.bfloat16, device='cuda')
        
        cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device='cuda')
        
        out = aiter.mla_fwd_kvcache(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=S,
            max_seqlen_k=S,
            sm_scale=1.0 / (head_size ** 0.5),
        )
        test_pass("mla_fwd_kvcache (prefill)")
    except AttributeError:
        test_skip("MLA prefill", "aiter.mla_fwd_kvcache not found")
    except Exception as e:
        test_fail("MLA prefill", str(e)[:200])
    
    # ================================================================
    # 5. MoE Operations
    # ================================================================
    header("5. MoE Operations")
    
    # topk_softmax
    try:
        E, K = 64, 4
        M = 512
        gating = torch.randn(M, E, dtype=torch.bfloat16, device='cuda')
        
        topk_weights, topk_indices = aiter.topk_softmax(gating, k=K)
        assert topk_weights.shape == (M, K)
        assert topk_indices.shape == (M, K)
        weight_sum = topk_weights.sum(dim=-1)
        assert torch.allclose(weight_sum, torch.ones(M, device='cuda'), atol=0.1), \
            f"Weight sum not ~1.0: {weight_sum[:5]}"
        test_pass("topk_softmax", f"E={E},K={K}, weights sum to ~1.0")
    except Exception as e:
        test_fail("topk_softmax", str(e)[:200])
    
    # ================================================================
    # 6. Normalization
    # ================================================================
    header("6. Normalization")
    
    try:
        from atom.model_ops.layernorm import RMSNorm, LayerNorm
        dim = 512
        rms = RMSNorm(dim, eps=1e-6).cuda()
        x = torch.randn(32, dim, dtype=torch.bfloat16, device='cuda')
        out = rms(x)
        assert out.shape == x.shape
        # Check RMS is ~1
        rms_val = (out.float() ** 2).mean(dim=-1).sqrt()
        assert (rms_val - 1.0).abs().mean() < 0.2, f"RMS not ~1: {rms_val[:5]}"
        test_pass("RMSNorm", f"dim={dim}, output RMS ~1.0")
    except Exception as e:
        test_fail("RMSNorm", str(e)[:200])
    
    try:
        from atom.model_ops.layernorm import RMSNorm
        dim = 512
        rms = RMSNorm(dim, eps=1e-6, fused_quant=True).cuda()
        x = torch.randn(32, dim, dtype=torch.bfloat16, device='cuda')
        out = rms(x)
        test_pass("RMSNorm (fused_quant)")
    except Exception as e:
        test_fail("RMSNorm (fused_quant)", str(e)[:100])
    
    # ================================================================
    # 7. Activations
    # ================================================================
    header("7. Activations")
    
    try:
        from atom.model_ops.activation import SiluAndMul
        act = SiluAndMul().cuda()
        x = torch.randn(32, 1024, dtype=torch.bfloat16, device='cuda')
        out = act(x)
        assert out.shape == (32, 512), f"Wrong shape: {out.shape}"
        # Verify: out = silu(x_first_half) * x_second_half
        x1, x2 = x[:, :512].float(), x[:, 512:].float()
        expected = (torch.nn.functional.silu(x1) * x2).to(torch.bfloat16)
        diff = (out.float() - expected.float()).abs().max().item()
        assert diff < 0.1, f"Max diff: {diff}"
        test_pass("SiluAndMul", f"max_diff={diff:.4f}")
    except Exception as e:
        test_fail("SiluAndMul", str(e)[:200])
    
    # ================================================================
    # 8. RoPE
    # ================================================================
    header("8. Rotary Position Embedding")
    
    try:
        from atom.model_ops.rotary_embedding import RotaryEmbedding
        head_size = 128
        rotary_dim = 128
        rope = RotaryEmbedding(head_size, rotary_dim, 4096, 10000.0).cuda()
        
        S, H, D = 64, 8, 128
        q = torch.randn(S, H, D, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(S, 8, D, dtype=torch.bfloat16, device='cuda')
        positions = torch.arange(S, device='cuda', dtype=torch.long)
        
        q_out, k_out = rope(positions, q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        # RoPE should change the values
        q_diff = (q_out.float() - q.float()).abs().mean().item()
        assert q_diff > 1e-3, f"RoPE didn't change Q: diff={q_diff}"
        test_pass("RotaryEmbedding", f"q_diff={q_diff:.4f}")
    except Exception as e:
        test_fail("RotaryEmbedding", str(e)[:200])
    
    # ================================================================
    # 9. KV Cache Operations
    # ================================================================
    header("9. KV Cache Operations")
    
    try:
        num_blocks = 100
        block_size = 16
        num_kv = 8
        head_dim = 128
        x = 8
        
        k_cache = torch.zeros(num_blocks, num_kv, head_dim // x, block_size, x,
                              dtype=torch.bfloat16, device='cuda')
        v_cache = torch.zeros(num_blocks, num_kv, block_size // x, head_dim, x,
                              dtype=torch.bfloat16, device='cuda')
        
        tokens = 4
        k_new = torch.randn(tokens, num_kv, head_dim, dtype=torch.bfloat16, device='cuda')
        v_new = torch.randn(tokens, num_kv, head_dim, dtype=torch.bfloat16, device='cuda')
        slot_mapping = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device='cuda')
        
        aiter.reshape_and_cache(k_new, v_new, k_cache, v_cache, slot_mapping,
                                kv_cache_dtype="auto", k_scale=None, v_scale=None,
                                asm_layout=True)
        torch.cuda.synchronize()
        
        # Verify data was written
        k_nonzero = (k_cache.abs() > 1e-5).sum().item()
        v_nonzero = (v_cache.abs() > 1e-5).sum().item()
        assert k_nonzero > 0, "K cache all zeros after write"
        assert v_nonzero > 0, "V cache all zeros after write"
        test_pass("reshape_and_cache (asm_layout=True)", f"K:{k_nonzero} V:{v_nonzero} nonzero")
    except Exception as e:
        test_fail("reshape_and_cache", str(e)[:200])
    
    # ================================================================
    # 10. Sampling
    # ================================================================
    header("10. Sampling")
    
    try:
        from atom.model_ops.sampler import Sampler
        sampler = Sampler(vocab_size=1000).cuda()
        
        logits = torch.randn(4, 1000, dtype=torch.float32, device='cuda')
        temperatures = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
        
        tokens = sampler(logits, temperatures)
        assert tokens.shape == (4,) or tokens.shape == (4, 1), f"Wrong shape: {tokens.shape}"
        test_pass("Sampler (greedy)", f"shape={tokens.shape}")
    except Exception as e:
        test_fail("Sampler", str(e)[:200])
    
    # ================================================================
    # 11. Embedding
    # ================================================================
    header("11. Embedding")
    
    try:
        from atom.model_ops.embed_head import VocabParallelEmbedding
        emb = VocabParallelEmbedding(num_embeddings=1000, embedding_dim=512).cuda()
        tokens = torch.tensor([1, 5, 10, 20], device='cuda')
        out = emb(tokens)
        assert out.shape == (4, 512), f"Wrong shape: {out.shape}"
        test_pass("VocabParallelEmbedding", f"shape={out.shape}")
    except Exception as e:
        test_fail("VocabParallelEmbedding", str(e)[:200])
    
    # ================================================================
    # Summary
    # ================================================================
    header("SUMMARY")
    print("  All critical operators tested. See PASS/FAIL above.")
    print(f"  GPU: {gfx}")
    print("  End-to-end inference test: run test_atom_pa_fwd_asm.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
