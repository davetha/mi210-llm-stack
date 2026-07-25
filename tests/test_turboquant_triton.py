#!/usr/bin/env python3
"""
TurboQuant KV Cache Quantization - Standalone Triton Implementation for gfx90a (MI210).

Based on vLLM's TurboQuant implementation:
  - model_executor/layers/quantization/turboquant/centroids.py
  - v1/attention/ops/triton_turboquant_store.py
  - v1/attention/ops/triton_turboquant_decode.py
  - v1/attention/backends/turboquant_attn.py

Algorithm (per vLLM):
  1. Build Hadamard matrix H (D x D, Sylvester construction, normalized by 1/sqrt(D))
  2. Compute L2 norm per key vector: d = ||x||
  3. Normalize: x_hat = x / d
  4. Rotate: y = x_hat @ H  (single GEMM via rocBLAS)
  5. Binary search on Lloyd-Max midpoints to find nearest centroid
  6. Pack indices (3-bit: 8 indices per 3 bytes; 4-bit: 2 per byte) + store norm (fp16)
  7. Dequantize: unpack -> gather centroids -> x d -> @H (inverse rotation)

Hadamard matrix is orthonormal and symmetric: H @ H = I, so inverse rotation = forward rotation.

No warp shuffle operations used. GEMM-based WHT avoids wave64 issues entirely.
"""

import math
import sys
import traceback

import torch
import triton
import triton.language as tl


# =====================================================================
# Part A: Lloyd-Max Centroid Solver (from vLLM centroids.py)
# =====================================================================

def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration (replaces scipy.integrate.quad)."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10):
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution.

    After rotating a d-dimensional unit vector by a random orthogonal matrix,
    each coordinate approximately follows N(0, 1/d) for d >= 64.

    Args:
        d: Vector dimension (determines variance = 1/d).
        bits: Number of quantization bits (3 -> 8 centroids, 4 -> 16 centroids).
        max_iter: Maximum Lloyd-Max iterations.
        tol: Convergence tolerance.

    Returns:
        centroids: Sorted tensor of 2^bits optimal centroids.
        boundaries: Sorted tensor of 2^bits - 1 decision boundaries.
    """
    n_levels = 2 ** bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def get_centroids_and_midpoints(d: int, bits: int):
    """Get sorted centroids and decision-boundary midpoints."""
    centroids, _ = solve_lloyd_max(d, bits)
    midpoints = (centroids[:-1] + centroids[1:]) / 2
    return centroids, midpoints


# =====================================================================
# Part B: Hadamard Matrix Builder (from vLLM turboquant_attn.py)
# =====================================================================

def build_hadamard(d: int) -> torch.Tensor:
    """Orthonormal Hadamard matrix via Sylvester construction, normalized by 1/sqrt(d).

    Properties: H = H^T, H @ H = I (orthonormal + symmetric -> self-inverse).
    This enables GEMM-based Walsh-Hadamard Transform via a single cuBLAS/rocBLAS call,
    completely avoiding wave64/butterfly-loop issues on gfx90a.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


# =====================================================================
# Part C: Triton Quantize Kernel
# =====================================================================

@triton.jit
def _tq_quantize_kernel(
    Y_ptr,            # [NH, D] float32 -- rotated normalized keys (x_hat @ H)
    Norms_ptr,        # [NH] float32    -- key vector norms (||x||)
    Packed_ptr,       # [NH, mse_bytes] uint8 -- output packed indices
    Norms_out_ptr,    # [NH] float16    -- output norms
    Midpoints_ptr,    # [n_centroids-1] float32
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Fused binary-search bucketize + pack indices + store norm.

    No warp shuffle operations. Pure elementwise + reduce + bitwise ops.
    """
    pid = tl.program_id(0)
    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # -- 1. Binary search bucketize on midpoints --------------------------
    # Midpoints are sorted (N_CENTROIDS-1 values); binary search finds
    # insertion point in MSE_BITS iterations vs N_CENTROIDS-1 for linear.
    y_vec = tl.load(Y_ptr + base + d_offs, mask=d_mask, other=0.0)
    lo = tl.zeros([BLOCK_D], dtype=tl.int32)
    hi = tl.full([BLOCK_D], N_CENTROIDS - 1, dtype=tl.int32)
    for _ in range(MSE_BITS):
        mid = (lo + hi) >> 1
        safe_mid = tl.minimum(mid, N_CENTROIDS - 2)
        mid_val = tl.load(Midpoints_ptr + safe_mid, mask=d_mask, other=0.0)
        lo = tl.where(y_vec >= mid_val, mid + 1, lo)
        hi = tl.where(y_vec >= mid_val, hi, mid)
    idx = tl.minimum(lo, N_CENTROIDS - 1)

    # -- 2. Pack indices from register ------------------------------------
    packed_base = pid * MSE_BYTES

    if MSE_BITS == 4:
        # Pack two 4-bit values per byte: byte[i] = idx[2i] | (idx[2i+1] << 4)
        idx_pairs = tl.reshape(idx, [BLOCK_D // 2, 2])
        shifts_4 = tl.arange(0, 2) * 4
        packed = tl.sum((idx_pairs & 0xF) << shifts_4[None, :], axis=1).to(tl.uint8)
        mse_offs = tl.arange(0, BLOCK_D // 2)
        mse_mask = mse_offs < MSE_BYTES
        tl.store(Packed_ptr + packed_base + mse_offs, packed, mask=mse_mask)

    elif MSE_BITS == 3:
        # Pack eight 3-bit values per 3 bytes: 8*3=24 bits per group
        grp_offs = tl.arange(0, BLOCK_GRP)
        grp_mask = grp_offs < (D // 8)
        idx_grp = tl.reshape(idx, [BLOCK_GRP, 8])
        shifts_3 = tl.arange(0, 8) * 3
        packed_24 = tl.sum((idx_grp & 0x7) << shifts_3[None, :], axis=1)
        b0 = (packed_24 & 0xFF).to(tl.uint8)
        b1 = ((packed_24 >> 8) & 0xFF).to(tl.uint8)
        b2 = ((packed_24 >> 16) & 0xFF).to(tl.uint8)
        tl.store(Packed_ptr + packed_base + grp_offs * 3, b0, mask=grp_mask)
        tl.store(Packed_ptr + packed_base + grp_offs * 3 + 1, b1, mask=grp_mask)
        tl.store(Packed_ptr + packed_base + grp_offs * 3 + 2, b2, mask=grp_mask)

    # -- 3. Store norm (fp16) ---------------------------------------------
    vn_f16 = tl.load(Norms_ptr + pid).to(tl.float16)
    tl.store(Norms_out_ptr + pid, vn_f16)


def triton_quantize(x, H, midpoints, bits, device):
    """Full quantization: normalize -> rotate (GEMM) -> binary search -> pack -> store norm.

    Args:
        x: [N, D] fp16 -- input keys
        H: [D, D] fp32 -- Hadamard matrix (on device)
        midpoints: [n_centroids-1] fp32
        bits: 3 or 4
        device: torch device

    Returns:
        packed: [N, mse_bytes] uint8
        norms: [N] fp16
    """
    N, D = x.shape
    n_centroids = 2 ** bits
    mse_bytes = math.ceil(D * bits / 8)

    # Steps 1-3: normalize + rotate via GEMM (rocBLAS -- handles gfx90a natively)
    k_flat = x.float()
    norms = k_flat.norm(dim=1, keepdim=True)  # [N, 1] L2 norm
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ H  # rotate: [N, D] -- single GEMM call

    # Steps 4-6: Triton kernel (binary search + pack + norm store)
    # Use flat buffer with stride MSE_BYTES per row (kernel uses pid * MSE_BYTES).
    # +2 trailing bytes prevent OOB reads in 3-bit dequant (byte_idx+1 on last elem).
    packed = torch.zeros(N * mse_bytes + 2, dtype=torch.uint8, device=device)
    norms_out = torch.empty(N, dtype=torch.float16, device=device)
    midpoints_dev = midpoints.to(device)

    BLOCK_D = triton.next_power_of_2(D)
    block_grp = triton.next_power_of_2(D // 8) if D >= 8 else 1
    grid = (N,)

    _tq_quantize_kernel[grid](
        y, norms.squeeze(1), packed, norms_out, midpoints_dev,
        D=D, BLOCK_D=BLOCK_D, MSE_BITS=bits, MSE_BYTES=mse_bytes,
        N_CENTROIDS=n_centroids, BLOCK_GRP=block_grp,
        num_warps=4,
    )

    return packed[:N * mse_bytes].view(N, mse_bytes), norms_out


# =====================================================================
# Part D: Triton Dequantize Kernel
# =====================================================================

@triton.jit
def _tq_dequant_kernel(
    Packed_ptr,       # [NH, mse_bytes] uint8
    Norms_ptr,        # [NH] float16
    Centroids_ptr,    # [n_centroids] float32
    Out_ptr,          # [NH, D] float32 -- output: rotated reconstructed (norms x centroids)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
    NORM_CORRECTION: tl.constexpr,
):
    """Unpack indices, gather centroids, multiply by norm.

    No warp shuffle operations. Pure elementwise + reduce + gather ops.
    """
    pid = tl.program_id(0)
    base = pid * D
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    packed_base = pid * MSE_BYTES

    # -- Unpack indices ---------------------------------------------------
    if MSE_BITS == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(
            Packed_ptr + packed_base + vb_idx, mask=d_mask, other=0
        ).to(tl.int32)
        idx = (val_raw >> vb_shift) & 0xF

    elif MSE_BITS == 3:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            Packed_ptr + packed_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            Packed_ptr + packed_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16 = mse_raw0 | (mse_raw1 << 8)
        idx = (raw16 >> mse_bit_shift) & mse_umask

    # -- Gather centroids -------------------------------------------------
    c_vals = tl.load(Centroids_ptr + idx, mask=d_mask, other=0.0)

    # -- Norm correction: re-normalize centroid vector to unit norm -------
    if NORM_CORRECTION:
        c_norm_sq = tl.sum(tl.where(d_mask, c_vals * c_vals, 0.0), axis=0)
        c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
        c_vals = c_vals * c_inv_norm

    # -- Multiply by norm -------------------------------------------------
    vec_norm = tl.load(Norms_ptr + pid).to(tl.float32)
    recon = vec_norm * c_vals

    tl.store(Out_ptr + base + d_offs, recon, mask=d_mask)


def triton_dequantize(packed, norms, centroids, H, D, bits, device,
                      norm_correction=True):
    """Full dequantization: unpack -> gather -> norm -> inverse rotate (GEMM).

    Args:
        packed: [N, mse_bytes] uint8
        norms: [N] fp16
        centroids: [n_centroids] fp32
        H: [D, D] fp32 -- Hadamard matrix for inverse rotation (H is self-inverse)
        D: head dimension
        bits: 3 or 4
        device: torch device
        norm_correction: re-normalize centroid vectors to unit norm

    Returns:
        recon: [N, D] fp16 -- reconstructed keys
    """
    N = packed.shape[0]
    n_centroids = 2 ** bits
    mse_bytes = math.ceil(D * bits / 8)

    # Flat buffer with +2 trailing bytes prevents OOB read in 3-bit unpack
    # (last element reads byte_idx+1 which can be MSE_BYTES past row start).
    packed_flat = torch.zeros(N * mse_bytes + 2, dtype=torch.uint8, device=device)
    packed_flat[:N * mse_bytes] = packed.reshape(-1)

    rotated_recon = torch.empty(N, D, dtype=torch.float32, device=device)
    centroids_dev = centroids.to(device)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (N,)

    _tq_dequant_kernel[grid](
        packed_flat, norms, centroids_dev, rotated_recon,
        D=D, BLOCK_D=BLOCK_D, MSE_BITS=bits, MSE_BYTES=mse_bytes,
        N_CENTROIDS=n_centroids,
        NORM_CORRECTION=1 if norm_correction else 0,
        num_warps=4,
    )

    # Inverse rotation via GEMM (H is self-inverse: H @ H = I)
    # rotated_recon @ H = (norms * centroids) @ H -> back to original space
    recon = (rotated_recon @ H).to(torch.float16)
    return recon


# =====================================================================
# CPU Reference Implementation
# =====================================================================

def cpu_quantize_dequant(x, H, centroids, midpoints, bits, norm_correction=True):
    """CPU reference: quantize -> dequantize using PyTorch on CPU.

    This is the ground truth for verifying the Triton kernels.
    """
    N, D = x.shape

    # Quantize
    k_flat = x.float()
    norms = k_flat.norm(dim=1, keepdim=True)  # [N, 1]
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ H  # rotate

    # Find nearest centroid via searchsorted on midpoints
    indices = torch.searchsorted(midpoints, y, right=True)  # [N, D]

    # Dequantize
    c_vals = centroids[indices]  # gather

    if norm_correction:
        c_norm = c_vals.norm(dim=1, keepdim=True)
        c_vals = c_vals / (c_norm + 1e-16)

    recon_rotated = norms * c_vals  # [N, D]
    recon = recon_rotated @ H  # inverse rotate (H is self-inverse)

    return recon.to(torch.float16)


# =====================================================================
# PyTorch Fallback (if Triton fails on gfx90a)
# =====================================================================

def pytorch_quantize_dequant(x, H, centroids, midpoints, bits, device,
                             norm_correction=True):
    """Pure PyTorch quantize + dequantize on GPU (fallback if Triton fails).

    Uses GEMM-based WHT (matmul with Hadamard matrix) -- same approach as
    Triton path, just without fused binary-search/pack kernel.
    """
    N, D = x.shape

    k_flat = x.float()
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ H  # GEMM rotation

    indices = torch.searchsorted(midpoints.to(device), y, right=True)
    c_vals = centroids.to(device)[indices]

    if norm_correction:
        c_norm = c_vals.norm(dim=1, keepdim=True)
        c_vals = c_vals / (c_norm + 1e-16)

    recon_rotated = norms * c_vals
    recon = recon_rotated @ H  # GEMM inverse rotation

    return recon.to(torch.float16)


# =====================================================================
# Metrics
# =====================================================================

def cosine_similarity(a, b):
    """Per-row cosine similarity, averaged."""
    dot = (a * b).sum(dim=1)
    na = a.norm(dim=1)
    nb = b.norm(dim=1)
    return (dot / (na * nb + 1e-8)).mean().item()


def mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


# =====================================================================
# Correctness Test
# =====================================================================

def test_turboquant(D, bits, N, device, use_triton=True):
    """Run quantize -> dequant for given config and return metrics."""
    print(f"\n{'=' * 70}")
    print(f"  TurboQuant {bits}-bit | D={D} | N={N} tokens | device={device}")
    print(f"{'=' * 70}")

    # Build codebook and Hadamard
    centroids, midpoints = get_centroids_and_midpoints(D, bits)
    H = build_hadamard(D).to(device)
    H_orth = torch.allclose(H @ H.T, torch.eye(D, device=device), atol=1e-5)
    print(f"  Centroids ({len(centroids)}): [{', '.join(f'{c:.6f}' for c in centroids.tolist())}]")
    print(f"  Hadamard: {D}x{D}, orthonormal(H@H^T=I): {H_orth}")

    # Generate random fp16 data simulating KV cache
    torch.manual_seed(42)
    x = torch.randn(N, D, dtype=torch.float16, device=device)

    # --- CPU Reference (always on CPU) ---
    x_cpu = x.cpu()
    H_cpu = H.cpu()
    cpu_recon = cpu_quantize_dequant(x_cpu, H_cpu, centroids, midpoints, bits)
    cpu_cos = cosine_similarity(x_cpu.float(), cpu_recon.float())
    cpu_mse = mse(x_cpu.float(), cpu_recon.float())
    print(f"\n  CPU Reference:")
    print(f"    Cosine similarity: {cpu_cos:.6f}")
    print(f"    MSE:               {cpu_mse:.8f}")

    # --- GPU Implementation ---
    mse_bytes = math.ceil(D * bits / 8)
    raw_bytes = D * 2  # fp16
    packed_total = mse_bytes + 2  # indices + fp16 norm

    if use_triton:
        try:
            # Triton path: quantize (GEMM + kernel) -> dequant (kernel + GEMM)
            packed, norms = triton_quantize(x, H, midpoints, bits, device)
            recon = triton_dequantize(
                packed, norms, centroids, H, D, bits, device, norm_correction=True
            )

            gpu_cos = cosine_similarity(x.float(), recon.float())
            gpu_mse = mse(x.float(), recon.float())

            # Cross-check GPU vs CPU
            cross_cos = cosine_similarity(
                cpu_recon.float().to(device), recon.float()
            )
            max_diff = (cpu_recon.float().to(device) - recon.float()).abs().max().item()

            print(f"\n  Triton GPU (gfx90a) -- GEMM-based WHT:")
            print(f"    Cosine similarity: {gpu_cos:.6f}")
            print(f"    MSE:               {gpu_mse:.8f}")
            print(f"    GPU vs CPU match:  cosine={cross_cos:.6f}, max_diff={max_diff:.6f}")
            print(f"    Packed size:       {mse_bytes} bytes indices + 2 bytes norm = {packed_total} bytes/vector")
            print(f"    Compression:       {raw_bytes}/{packed_total} = {raw_bytes / packed_total:.2f}x")

            return cpu_cos, gpu_cos, "triton"

        except Exception as e:
            print(f"\n  *** Triton kernel FAILED on gfx90a: {e}")
            traceback.print_exc()
            print(f"\n  >> Falling back to PyTorch GEMM-based implementation <<")

            recon = pytorch_quantize_dequant(
                x, H, centroids, midpoints, bits, device, norm_correction=True
            )
            gpu_cos = cosine_similarity(x.float(), recon.float())
            gpu_mse = mse(x.float(), recon.float())
            cross_cos = cosine_similarity(
                cpu_recon.float().to(device), recon.float()
            )

            print(f"\n  PyTorch Fallback (GPU) -- GEMM-based WHT:")
            print(f"    Cosine similarity: {gpu_cos:.6f}")
            print(f"    MSE:               {gpu_mse:.8f}")
            print(f"    GPU vs CPU match:  cosine={cross_cos:.6f}")
            print(f"    Compression:       {raw_bytes}/{packed_total} = {raw_bytes / packed_total:.2f}x")

            return cpu_cos, gpu_cos, "pytorch_fallback"
    else:
        recon = pytorch_quantize_dequant(
            x, H, centroids, midpoints, bits, device, norm_correction=True
        )
        gpu_cos = cosine_similarity(x.float(), recon.float())
        gpu_mse = mse(x.float(), recon.float())
        print(f"\n  PyTorch GPU:")
        print(f"    Cosine similarity: {gpu_cos:.6f}")
        print(f"    MSE:               {gpu_mse:.8f}")
        return cpu_cos, gpu_cos, "pytorch"


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("  TurboQuant KV Cache Quantization")
    print("  Standalone Triton Implementation for AMD gfx90a (MI210)")
    print("  Based on vLLM TurboQuant (Lloyd-Max + Hadamard rotation)")
    print("=" * 70)

    # Environment info
    print(f"\nPyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("*** No CUDA/ROCm available -- running CPU only ***")
        device = torch.device("cpu")

    D = 64    # head_dim for mimo
    N = 256   # num_tokens simulating KV cache
    results = {}

    for bits in [3, 4]:
        cpu_cos, gpu_cos, impl = test_turboquant(D, bits, N, device, use_triton=True)
        results[bits] = (cpu_cos, gpu_cos, impl)

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    all_pass = True
    for bits, (cpu_cos, gpu_cos, impl) in sorted(results.items()):
        threshold = 0.95 if bits == 3 else 0.99
        status = "PASS" if gpu_cos > threshold else "FAIL"
        impl_tag = f"[{impl}]"
        if gpu_cos <= threshold:
            all_pass = False
        print(f"  {bits}-bit {impl_tag:25s}: Cosine = {gpu_cos:.6f}  "
              f"(target > {threshold})  {status}")

    # Verify CPU and GPU agree
    print(f"\n  Cross-validation (GPU vs CPU reference):")
    for bits, (cpu_cos, gpu_cos, impl) in sorted(results.items()):
        diff = abs(cpu_cos - gpu_cos)
        match = "MATCH" if diff < 0.01 else "MISMATCH"
        print(f"    {bits}-bit: CPU={cpu_cos:.6f}  GPU={gpu_cos:.6f}  "
              f"diff={diff:.6f}  {match}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
