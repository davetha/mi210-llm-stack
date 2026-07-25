# Why V Cache Quantization Requires FlashAttention

## The Technical Reason (Source-Verified)

### The Constraint
**File**: `llama-context.cpp:3562-3565`
```cpp
if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
    LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
    return nullptr;
}
```

### Why It Exists — The V Transpose Problem

The non-FA attention path computes `output = V^T × softmax(QK^T)` using `ggml_mul_mat()`. For this GEMM, V must be in a specific layout.

**File**: `llama-graph.cpp:2488-2494`
```cpp
// Non-FA path:
if (!v_trans) {
    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));  // V must be transposed!
}
ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);     // Then GEMM
```

**Block-quantized tensors CANNOT be transposed** — transposing scrambles the block structure:
- Q4_0 block: [scale(2 bytes) + 16 × 4-bit values] = 18 bytes per 32 elements
- Blocks are contiguous along ONE dimension
- Transposing moves elements across block boundaries
- The GEMM kernel can't read the scrambled blocks

### Why K Works Without FA

K is used DIFFERENTLY:
```cpp
ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);  // K used DIRECTLY, no transpose
```

K is already in the correct layout for `mul_mat(k, q)`. Block-quantized K works because the GEMM kernel reads K's blocks along the contiguous dimension — exactly what block-quantized GEMM supports.

### Why FA Works With Quantized V

FA (`ggml_flash_attn_ext`) is a **fused tiled kernel**:
```
For each tile:
    Load K block → dequantize inline → compute QK^T
    Load V block → dequantize inline → multiply by attention weights
```

FA reads V in its **native cache layout** (no transpose needed). The inline dequantization happens per-tile within the fused kernel. This is why FA supports any quantized V type.

### Summary

| Path | K Quantized | V Quantized | Why |
|---|---|---|---|
| Non-FA (standard attention) | ✅ Works | ❌ Blocked | K used directly; V needs transpose which breaks blocks |
| FA (flash attention) | ✅ Works | ✅ Works | Fused kernel reads both K and V in native layout with inline dequant |

### The Fix Path
To enable quantized V without FA, you need either:
1. A transpose-capable dequant-requant kernel (dequant V → f16 → transpose → GEMM)
2. A native-layout non-FA attention variant (read V in cache layout without transpose)
3. Option 1 adds memory traffic (dequant temporary); Option 2 requires kernel rewrite

### The K-Only Sweet Spot
Since K works without FA but V doesn't, the optimal no-FA config is:
```
-ctk q4_0 -ctv f16 -fa off
```
This gives compressed K cache + fast FA-off attention. Verified working on gfx90a.
