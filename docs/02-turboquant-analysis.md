# TurboQuant Analysis — Wave64 Failure & the Triton GEMM Solution

## What TurboQuant is

[**TurboQuant**](https://arxiv.org/abs/2504.19874) (Google Research / DeepMind, presented at ICLR-2026) is a KV-cache compression pipeline claiming ~5.2× near-lossless compression:

1. **Walsh–Hadamard rotation (WHT)** — rotate each key/value vector so outlier energy spreads evenly across coordinates.
2. **Lloyd-Max scalar quantization** — optimal per-coordinate quantizer for the post-rotation `N(0,1/d)` distribution.
3. **QJL projection** (optional) — further dimensionality reduction.

Forks evaluated:
- [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) — **maintained**, HIP/ROCm backend (the one we built against).
- `AmesianX/TurboQuant` — archived/abandoned.
- `phil2Sat/llamacpp-rocmcpp-turboquant` — only ships a gfx900 (Vega) prebuilt.

## The wave64 problem

TurboQuant **compiles and runs on gfx90a at full GPU speed (~108 t/s) but produces garbage output** (`|- 0 100<!--ast0000ágenes...`).

The smoking gun: the **same binary with `-ctv q8_0` is coherent**, and **`-ngl 0` (pure CPU) turbo3 is coherent**. So the algorithm and CPU implementation are correct — **the HIP GPU kernel is wrong**.

### Root cause

The MI210 runs **64-lane wavefronts**. The turbo kernels assume **32-lane warps** (written/validated for NVIDIA + RDNA + CDNA3/CDNA4). They use width-less `__shfl_sync` / `__ballot_sync` calls:

```c
// width-less shuffle → defaults to 64-lane wavefront on gfx90a
uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k);

// ballot truncated to uint32_t → lanes 32-63 silently dropped
const uint32_t ballot = __ballot_sync(0xffffffff, (idx >> 2) & 1);
```

When a 128-thread block launches (= 2 wavefronts), two logical 32-thread "warps" share each wavefront. The packed `qs`/`signs` bytes come out scrambled → garbage on read.

**Independent confirmation:** MI50/gfx906 (another wave64 part) shows the identical corruption — [llama.cpp discussion #21526](https://github.com/ggml-org/llama.cpp/discussions/21526).

### What is NOT the bug (proven by isolation)

| Component | Status | How proven |
|-----------|--------|------------|
| WHT butterfly (shared-mem) | ✅ correct | Uses shared memory + `__syncthreads()`, not warp shuffles — correct on any hardware. |
| General `__shfl`/`__ballot` in the codebase | ✅ correct | `q8_0` path uses the same shuffles and is coherent. |
| Lloyd-Max centroid math | ✅ correct | CPU round-trip cosine > 0.98. |
| Graph-level Q/V rotation (`turbo-wht.cu`) | ✅ correct | Shared-memory + `syncthreads`. |
| **TBQ-specific quantize path** (`set-rows.cu`) | ❌ **the bug** | The `qs`/`signs` packing uses width-less shuffles + truncated ballot. |

The corruption is in the **TurboQuant-specific quantize/dequantize kernels**, not in the general shuffle infrastructure (which `q8_0` proves works) and not in the WHT math (which is shared-memory and platform-agnostic).

## The wave64 patch attempt

We applied 4 categories of genuine fixes (each changed the corruption pattern, proving they affect the computation):

1. **Ballot macro** (`vendors/hip.h:59`): `uint32_t` → `uint64_t`. Root cause of sign-bit truncation.
2. **`__shfl_sync` width ×5** (`set-rows.cu`): added explicit `WARP_SIZE` parameter to 5 calls.
3. **turbo3 signs ballot** (`set-rows.cu:381`): `uint32_t` → `uint64_t` + physical-lane indexing.
4. **turbo2 signs ballot** (`set-rows.cu:510`): same pattern.

See [`changes/03-turboquant-wave64-fixes.md`](../changes/03-turboquant-wave64-fixes.md).

**Result:** output improved from total garbage → semi-corrupted (real words, wrong meaning). But the corruption is **pervasive** — the same pattern exists in `mmvq-tq.cu`, `convert.cu`, the fattn path. A full multi-kernel wave64 port is needed for GPU-correct TurboQuant. All code-level debugging was exhausted via SSH; the remaining bug requires `rocprof` GPU tracing to identify the unidentified numerical-precision path.

## The Triton solution: GEMM-based WHT

Instead of auditing every wave-level intrinsic, **replace the WHT rotation with a GEMM**. The Hadamard transform is a linear operation: `y = x · H`. So precompute `H` and do a single rocBLAS matrix multiply. A GEMM has no wave-level intrinsics — it's block-level, and Triton compiles it to wave64-native code automatically.

**Implementation:** [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd) — pure Triton 3.7.1, zero wave-level ops, no scipy dependency.

### Quality results

| Bits | Compression vs fp16 | Cosine similarity (GPU vs CPU) | Verdict |
|------|--------------------:|-------------------------------:|---------|
| 3-bit | ~5.3× | **0.9838** | ✅ PASS (threshold 0.95) |
| 4-bit | ~4.0× | **0.9955** | ✅ PASS (threshold 0.99) |

GPU (Triton on gfx90a) and CPU (PyTorch reference) match to `max_diff < 0.01`.

## The per-layer workaround (what actually shipped)

The **CPU** TurboQuant path is proven correct. So instead of fixing the GPU kernels, we added **per-layer KV cache types** (`-ctk-cpu turbo3 -ctv-cpu turbo3`) to compress *only* the 25 CPU-pinned expert layers — exactly the layers that are DDR4-bandwidth-bound — while keeping GPU layers at full fp16. The GPU TurboQuant bug becomes irrelevant.

See [`changes/01-per-layer-kv-types.md`](../changes/01-per-layer-kv-types.md).

→ **Live in:** [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd)
→ **Patches:** [`davetha/llama.cpp-mi210`](https://github.com/davetha/llama.cpp-mi210)
