# gfx90a Architecture Constraints

Everything that does **not** work on the AMD MI210 (gfx90a / CDNA2) and why — learned the hard way on 2× MI210, PCIe-linked, ROCm 7.14.

## CDNA2 vs CDNA3: Vector Core vs Matrix Core

| Feature | CDNA2 (gfx90a / MI210) | CDNA3 (gfx942 / MI300X) |
|---------|------------------------|-------------------------|
| Compute units | 64 CU | 304 CU |
| Matrix engine | **MFMA (Matrix Core, fp16/bf16)** | **MFMA + WMA (Wave Matrix Accumulator)** |
| fp8 / bf8 | ❌ no | ✅ yes |
| Wavefront | **64 lanes (wave64)** | 64 lanes |
| xGMI / P2P | ❌ not on MI210 (PCIe only) | ✅ |

### Why rocWMMA FlashAttention loses

llama.cpp's `rocWMMA`-based FlashAttention (`GGML_HIP_ROCWMMA_FATTN`) is a
**performance** loss on gfx90a, not a correctness or capability one. Measured on
an MI210 with rocWMMA 2.2.1, it runs correctly and is 18-26% slower at prefill
(1,761 -> 1,440 tok/s at 16k; 1,713 -> 1,273 tok/s at 24k), degrading further
with context length.

The reason is that ggml already reaches the matrix cores without it. With the
flag OFF, gfx90a falls through to the modern `fattn-mma-f16` kernel, whose
gfx90a code object is full of `v_mfma_f32_16x16x16f16`. Turning rocWMMA ON only
*diverts* attention to the older `fattn-wmma-f16` kernel, which emits the same
MFMA instruction with worse blocking.

**This is not a tuning problem — do not retry it.** The flag does not turn the
matrix cores on; it chooses between two kernels that both use them, and picks
the worse one. Confirmed by three independent benchmark runs and by
disassembling the shipped library.

**Must build with:** `-DGGML_HIP_ROCWMMA_FATTN=OFF` — but note that setting it
`ON` against the rocWMMA shipped with ROCm 7.1 (version 2.0.0) does nothing at
all, because llama.cpp blacklists that exact version on CDNA. See
[`22-rocwmma-flash-attention-gfx90a.md`](22-rocwmma-flash-attention-gfx90a.md).

**What works instead:** standalone [FlashAttention 2.8.3 with the CK (Composable Kernel) backend](../guides/build-flashattention-gfx90a.md) — CK has a gfx90a codepath that rocWMMA lacks.

## Wave64 vs Wave32 — the silent killer

This is the single most important constraint. It broke TurboQuant and it will break any CUDA kernel naively ported to HIP without a wave64 audit.

- **NVIDIA GPUs** and **AMD RDNA** consumer cards run **32-lane warps**.
- **AMD CDNA** (including gfx90a/MI210) runs **64-lane wavefronts**.

The compiler defines `__gfx90a__` → maps to `CDNA2`, but it does **NOT** define `__GFX9__`. As a result:

```cpp
// ggml-cuda's vendor shim:
int ggml_cuda_get_physical_warp_size() {
#ifdef __GFX9__
    return 64;   // <-- NOT taken on gfx90a!
#else
    return 32;   // <-- this path runs
#endif
}
```

So the C++ code *thinks* warps are 32 lanes, but the **HIP intrinsics** (`__shfl`, `__ballot`, `__shfl_sync`) operate on the hardware `warpSize = 64`. When a 128-thread block launches (= 2 wavefronts on gfx90a), two logical 32-thread "warps" share each 64-lane wavefront. Width-less `__shfl_sync(mask, var, src)` defaults to the full 64-lane wavefront → **cross-block data contamination**.

### Concrete TurboQuant example

```c
// set-rows.cu — the corruption was here:
uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k);
//                                                              ^^^^^^
// no explicit width → defaults to 64-lane wavefront on gfx90a
// threads 32..63 gather from the wrong half-wave

const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
//              ^^^^^^^^                                       ^^^^
// __ballot returns 64-bit on wave64; the shim macro truncated to uint32_t
// → lanes 32-63 silently dropped
```

See [`changes/03-turboquant-wave64-fixes.md`](../changes/03-turboquant-wave64-fixes.md) and [`davetha/llama.cpp-mi210`](https://github.com/davetha/llama.cpp-mi210) for the patches.

## fp8 is dead on gfx90a

- `torch._scaled_mm` is **gated to MI300+** (gfx942). On gfx90a it raises "not supported on this GPU".
- vLLM's fp8 Marlin kernels are CUDA-only.
- Consequence: the only fp8 "abliterated" model checkpoints (e.g. Qwen3-Coder-Next fp8) **cannot run** on the MI210. Use llama.cpp Q4 instead — it's *faster anyway* on these memory-bandwidth-bound cards.

## P2P / peer-DMA impossible on CDNA2 (PCIe)

The two MI210s are **PCIe-linked with no xGMI bridge**. Native peer-to-peer DMA:

```
hipDeviceEnablePeerAccess(...)  // crashes / returns error on CDNA2
```

This means:
- **No direct GPU↔GPU tensor copy** over the fabric. Cross-card traffic must bounce through host RAM.
- Tensor-parallel all-reduce pays PCIe latency on every layer.
- **Expert parallelism on PCIe is NOT recommended** — each expert activation would cross the PCIe bus; measured **worse** than tensor-parallel (see [`docs/04`](04-moe-engine-survey.md)).

## NUMA: NPS1 (single node)

The EPYC 74F3 is configured NPS1 — one NUMA node spanning all 24 cores and all 499 GB. So there's no cross-NUMA penalty for CPU-offloaded experts, but also no NUMA-pinning win to be had. DDR4 bandwidth (~204 GB/s peak) is the CPU-side ceiling.

## What IS supported on gfx90a today

| Capability | Status | Notes |
|------------|--------|-------|
| HIP compute (kernels) | ✅ | Standard HIP path works. |
| rocBLAS GEMM | ✅ | Full speed; the backbone of all matmuls. |
| Triton 3.7.1 | ✅ | JIT-compiles wave64-native kernels automatically. No wave-level bugs. |
| FlashAttention 2.8.3 (CK backend) | ✅ | Must build from `origin/main`, not a pinned old tag (see [guide](../guides/build-flashattention-gfx90a.md)). |
| TurboQuant (HIP kernels) | ❌ | wave64 corruption — use the [Triton GEMM version](https://github.com/davetha/turboquant-triton-amd) instead. |
| fp8 / `_scaled_mm` | ❌ | gated to gfx942+. |
| rocWMMA FlashAttention | ❌ | CDNA3+ only. Build with `GGML_HIP_ROCWMMA_FATTN=OFF`. |
| P2P peer-DMA | ❌ | CDNA2 + no xGMI. |

## Independent confirmation

MI50 (gfx906), another older wave64 part, shows the **identical** TurboQuant corruption — [llama.cpp discussion #21526](https://github.com/ggml-org/llama.cpp/discussions/21526). The TurboQuant fork's validated HIP arches are RDNA3 / RDNA4 / CDNA3 / CDNA4 — **CDNA2/gfx90a is not listed**.
