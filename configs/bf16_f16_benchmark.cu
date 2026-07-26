#include <hip/hip_runtime.h>
#include <cstdio>

// ============================================================
// BF16 → F16 Conversion Benchmark for gfx90a
// Tests multiple approaches to find the fastest
// ============================================================

// Types
typedef unsigned short u16;
typedef unsigned int u32;

// BF16 bit layout: [S(1)][E(8)][M(7)] = 16 bits
// F16 bit layout:  [S(1)][E(5)][M(10)] = 16 bits
// BF16 exponent bias: 127 (same as FP32)
// F16 exponent bias: 15

// ============================================================
// Method 1: Float intermediate (bf16 → fp32 → fp16)
// Most accurate, but potentially slow (2 conversions)
// ============================================================
__device__ u16 bf16_to_f16_via_float(u16 bf16_bits) {
    // Pad BF16 to FP32 by appending 16 zero bits to the mantissa
    u32 fp32_bits = ((u32)bf16_bits) << 16;
    
    // Convert FP32 → FP16 using the F32→F16 conversion instruction
    // __float2half uses v_cvt_f16_f32 hardware instruction
    float fp32 = __uint_as_float(fp32_bits);
    _Float16 fp16 = (_Float16)fp32;
    return *(u16*)&fp16;
}

// ============================================================
// Method 2: Bit manipulation (direct exponent remap)
// BF16: S(1) E(8) M(7) → bias 127
// F16:  S(1) E(5) M(10) → bias 15
// 
// For normal-range values (BF16 exp 102..127 → F16 exp 0..15):
// - Check if value is within F16 range
// - Remap exponent: new_exp = old_exp - 127 + 15 = old_exp - 112
// - Append zeros to mantissa: M(7) → M(10) by shifting
// ============================================================
__device__ u16 bf16_to_f16_bitwise(u16 bf16_bits) {
    u32 sign = (bf16_bits >> 15) & 1;
    u32 exp = (bf16_bits >> 7) & 0xFF;
    u32 mant = bf16_bits & 0x7F;
    
    // F16 exponent = BF16 exponent - 112 (bias difference)
    // F16 bias = 15, BF16 bias = 127, diff = 112
    if (exp == 0) {
        // Zero or denormal → just return signed zero
        return sign << 15;
    }
    if (exp == 0xFF) {
        // Inf or NaN
        return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
    }
    
    int32_t f16_exp = (int32_t)exp - 112;
    if (f16_exp <= 0) {
        // Underflow to zero (too small for F16)
        return sign << 15;
    }
    if (f16_exp >= 0x1F) {
        // Overflow to infinity (too large for F16)
        return (sign << 15) | 0x7C00;
    }
    
    // Pack: S(1) | E(5) | M(10)
    // Mantissa: BF16 has 7 bits → shift left by 3 to get 10 bits
    u32 f16_mant = mant << 3;
    return (sign << 15) | (f16_exp << 10) | f16_mant;
}

// ============================================================
// Method 3: Vectorized conversion (2 values at once)
// Uses v_perm_b32 to shuffle two bf16 into two f16
// ============================================================
__device__ u32 bf16x2_to_f16x2_via_perm(u32 bf16x2) {
    // bf16x2 contains two BF16 values packed in 32 bits
    // Convert each via float intermediate, packed
    u16 lo = bf16x2 & 0xFFFF;
    u16 hi = bf16x2 >> 16;
    
    u16 f16_lo = bf16_to_f16_via_float(lo);
    u16 f16_hi = bf16_to_f16_via_float(hi);
    return ((u32)f16_hi << 16) | f16_lo;
}

// ============================================================
// Method 4: Bulk conversion kernel via float
// ============================================================
__global__ void convert_bulk_via_float(
    const u16* __restrict__ bf16_in,
    u16* __restrict__ f16_out,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        f16_out[idx] = bf16_to_f16_via_float(bf16_in[idx]);
    }
}

// ============================================================
// Method 5: Bulk conversion kernel via bit manipulation
// ============================================================
__global__ void convert_bulk_bitwise(
    const u16* __restrict__ bf16_in,
    u16* __restrict__ f16_out,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        f16_out[idx] = bf16_to_f16_bitwise(bf16_in[idx]);
    }
}

// ============================================================
// Method 6: MFMA with inline conversion (no separate pass)
// Loads BF16, converts to F16, feeds directly to MFMA
// ============================================================
typedef _Float16 __attribute__((vector_size(8))) f16x4_t;
typedef float __attribute__((vector_size(16))) f32x4_t;

__global__ void test_mfma_with_conversion(
    const u16* __restrict__ bf16_q,
    const u16* __restrict__ bf16_k,
    f32x4_t* __restrict__ output
) {
    // Load 4 BF16 values
    u16 b0 = bf16_q[threadIdx.x * 4 + 0];
    u16 b1 = bf16_q[threadIdx.x * 4 + 1];
    u16 b2 = bf16_q[threadIdx.x * 4 + 2];
    u16 b3 = bf16_q[threadIdx.x * 4 + 3];
    
    // Convert to F16
    f16x4_t a;
    ((u16*)&a)[0] = bf16_to_f16_via_float(b0);
    ((u16*)&a)[1] = bf16_to_f16_via_float(b1);
    ((u16*)&a)[2] = bf16_to_f16_via_float(b2);
    ((u16*)&a)[3] = bf16_to_f16_via_float(b3);
    
    // Load K values similarly
    u16 k0 = bf16_k[threadIdx.x * 4 + 0];
    u16 k1 = bf16_k[threadIdx.x * 4 + 1];
    u16 k2 = bf16_k[threadIdx.x * 4 + 2];
    u16 k3 = bf16_k[threadIdx.x * 4 + 3];
    
    f16x4_t b;
    ((u16*)&b)[0] = bf16_to_f16_via_float(k0);
    ((u16*)&b)[1] = bf16_to_f16_via_float(k1);
    ((u16*)&b)[2] = bf16_to_f16_via_float(k2);
    ((u16*)&b)[3] = bf16_to_f16_via_float(k3);
    
    // MFMA with F16 inputs
    f32x4_t c = {0.0f, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    output[threadIdx.x] = c;
}

// ============================================================
// Host-side test and benchmark
// ============================================================
#include <cstdlib>
#include <cmath>

// BF16 packing helper (FP32 → BF16)
u16 float_to_bf16_bits(float f) {
    u32 bits = *(u32*)&f;
    return (u16)(bits >> 16);
}

int main() {
    // Get device
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Wavefront: %d\n", prop.warpSize);
    
    // Test data
    const int N = 1 << 20;  // 1M elements
    u16* h_bf16 = (u16*)malloc(N * sizeof(u16));
    u16* h_f16_float = (u16*)malloc(N * sizeof(u16));
    u16* h_f16_bitwise = (u16*)malloc(N * sizeof(u16));
    
    // Generate test values (attention score range: -10 to +10)
    for (int i = 0; i < N; i++) {
        float val = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        h_bf16[i] = float_to_bf16_bits(val);
    }
    
    // Device buffers
    u16 *d_bf16, *d_f16;
    hipMalloc(&d_bf16, N * sizeof(u16));
    hipMalloc(&d_f16, N * sizeof(u16));
    hipMemcpy(d_bf16, h_bf16, N * sizeof(u16), hipMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Warmup
    hipLaunchKernelGGL(convert_bulk_via_float, blocks, threads, 0, 0, d_bf16, d_f16, N);
    hipDeviceSynchronize();
    
    // Benchmark Method 1: Float intermediate
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    float ms;
    hipEventRecord(start);
    for (int iter = 0; iter < 100; iter++) {
        hipLaunchKernelGGL(convert_bulk_via_float, blocks, threads, 0, 0, d_bf16, d_f16, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    float gb_s_float = (float)(N * sizeof(u16) * 2) / (ms / 100) / 1e6; // read+write
    printf("\nMethod 1 (float intermediate): %.2f ms/100iter = %.1f GB/s\n", ms, gb_s_float);
    
    hipMemcpy(h_f16_float, d_f16, N * sizeof(u16), hipMemcpyDeviceToHost);
    
    // Benchmark Method 2: Bit manipulation
    hipEventRecord(start);
    for (int iter = 0; iter < 100; iter++) {
        hipLaunchKernelGGL(convert_bulk_bitwise, blocks, threads, 0, 0, d_bf16, d_f16, N);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    float gb_s_bitwise = (float)(N * sizeof(u16) * 2) / (ms / 100) / 1e6;
    printf("Method 2 (bitwise):            %.2f ms/100iter = %.1f GB/s\n", ms, gb_s_bitwise);
    
    hipMemcpy(h_f16_bitwise, d_f16, N * sizeof(u16), hipMemcpyDeviceToHost);
    
    // Correctness check: compare methods
    int mismatches = 0;
    float max_diff = 0;
    for (int i = 0; i < N; i++) {
        if (h_f16_float[i] != h_f16_bitwise[i]) {
            mismatches++;
            // Calculate the actual FP value difference
            _Float16 vf = *(_Float16*)&h_f16_float[i];
            _Float16 vb = *(_Float16*)&h_f16_bitwise[i];
            float diff = fabsf((float)vf - (float)vb);
            if (diff > max_diff) max_diff = diff;
        }
    }
    printf("\nCorrectness: %d/%d mismatches between methods, max_diff=%f\n", mismatches, N, max_diff);
    
    // Also verify against reference: convert a few values and check
    printf("\nSample conversions (BF16 → F16):\n");
    for (int i = 0; i < 10; i++) {
        float orig = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
        u16 bf16 = float_to_bf16_bits(orig);
        
        // Via float
        u32 fp32_bits = ((u32)bf16) << 16;
        float as_float = *(float*)&fp32_bits;  // BF16 value as FP32
        _Float16 f16_val = (_Float16)as_float;
        u16 f16_bits = *(u16*)&f16_val;
        
        printf("  %.6f (bf16=0x%04x) → f16=0x%04x (%.6f) diff=%.2e\n",
               as_float, bf16, f16_bits, (float)f16_val,
               fabsf(as_float - (float)f16_val));
    }
    
    // Test MFMA with conversion
    printf("\n=== MFMA with inline conversion test ===\n");
    const int MFMA_N = 256;  // 4 wavefronts × 64 threads
    u16* d_mfma_q;
    u16* d_mfma_k;
    f32x4_t* d_mfma_out;
    hipMalloc(&d_mfma_q, MFMA_N * 4 * sizeof(u16));
    hipMalloc(&d_mfma_k, MFMA_N * 4 * sizeof(u16));
    hipMalloc(&d_mfma_out, MFMA_N * sizeof(f32x4_t));
    
    // Fill with test data
    u16* h_q = (u16*)malloc(MFMA_N * 4 * sizeof(u16));
    for (int i = 0; i < MFMA_N * 4; i++) {
        h_q[i] = float_to_bf16_bits(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
    hipMemcpy(d_mfma_q, h_q, MFMA_N * 4 * sizeof(u16), hipMemcpyHostToDevice);
    hipMemcpy(d_mfma_k, h_q, MFMA_N * 4 * sizeof(u16), hipMemcpyHostToDevice);
    
    hipLaunchKernelGGL(test_mfma_with_conversion, MFMA_N/64, 64, 0, 0, d_mfma_q, d_mfma_k, d_mfma_out);
    hipDeviceSynchronize();
    
    f32x4_t* h_out = (f32x4_t*)malloc(sizeof(f32x4_t));
    hipMemcpy(h_out, d_mfma_out, sizeof(f32x4_t), hipMemcpyDeviceToHost);
    printf("MFMA output[0]: [%.4f, %.4f, %.4f, %.4f]\n",
           (*h_out)[0], (*h_out)[1], (*h_out)[2], (*h_out)[3]);
    
    // Benchmark MFMA with conversion
    hipEventRecord(start);
    for (int iter = 0; iter < 1000; iter++) {
        hipLaunchKernelGGL(test_mfma_with_conversion, MFMA_N/64, 64, 0, 0, d_mfma_q, d_mfma_k, d_mfma_out);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&ms, start, stop);
    printf("MFMA+convert: %.2f ms/1000iter = %.3f us/iter\n", ms, ms / 1000.0);
    
    // Cleanup
    hipFree(d_bf16); hipFree(d_f16);
    hipFree(d_mfma_q); hipFree(d_mfma_k); hipFree(d_mfma_out);
    free(h_bf16); free(h_f16_float); free(h_f16_bitwise); free(h_q); free(h_out);
    hipEventDestroy(start); hipEventDestroy(stop);
    
    printf("\nDone!\n");
    return 0;
}
