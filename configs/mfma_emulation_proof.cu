#include <hip/hip_runtime.h>

// Use GCC vector_size syntax which Clang also supports
// The builtins expect specific vector types

// For __builtin_amdgcn_mfma_f32_16x16x16f16:
//   Input: vector of 4 _Float16 (8 bytes)
//   Acc: vector of 4 float (16 bytes)
typedef _Float16 __attribute__((vector_size(8)))  f16x4_t;
typedef float    __attribute__((vector_size(16))) f32x4_t;
typedef float    __attribute__((vector_size(64))) f32x16_t;

// For __builtin_amdgcn_mfma_f32_32x32x4bf16:
//   Input: vector of 2 short (4 bytes) - packed bf16
typedef short __attribute__((vector_size(4)))  bf16x2_packed_t;

// For __builtin_amdgcn_mfma_f32_16x16x32_bf16:
//   Input: vector of 8 __bf16 (16 bytes)
typedef __bf16 __attribute__((vector_size(16))) bf16x8_t;

// ============================================================
// Test 1: v_mfma_f32_16x16x16f16 — F16 MFMA (same tile as MLA needs)
// ============================================================
__global__ void test_f16_mfma(f32x4_t* out) {
    f16x4_t a = {1.0f16, 2.0f16, 3.0f16, 4.0f16};
    f16x4_t b = {5.0f16, 6.0f16, 7.0f16, 8.0f16};
    f32x4_t c = {0.0f, 0.0f, 0.0f, 0.0f};
    c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
    *out = c;
}

// ============================================================
// Test 2: v_mfma_f32_32x32x4bf16 — BF16 larger tile (padding approach)
// ============================================================
__global__ void test_bf16_32x32x4(f32x4_t* out) {
    bf16x2_packed_t a = {0x3F80, 0x4000};  // 1.0, 2.0 in bf16 as raw shorts
    bf16x2_packed_t b = {0x4200, 0x4400};  // 3.0, 4.0 in bf16
    f32x16_t c = {};
    c = __builtin_amdgcn_mfma_f32_32x32x4bf16(a, b, c, 0, 0, 0);
    out[0] = (f32x4_t){c[0], c[1], c[2], c[3]};
}

// (16x16x32_bf16 removed — requires gfx950)
