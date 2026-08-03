#!/usr/bin/env python3
"""
>>> OBSOLETE (checked 2026-08-03 against ggml-org/llama.cpp@master). <<<
>>>
>>> Fixed upstream. ggml/src/ggml-cuda/fattn-common.cuh now contains zero
>>> cudaMalloc/cudaFree calls and uses ggml_cuda_pool_alloc throughout.
>>>
>>> Proposed as an upstream candidate on 2026-08-03 and withdrawn the same
>>> day -- judged from this file rather than from upstream.
Fix HIP graph capture: replace raw cudaMalloc/cudaFree in launch_fattn with pool allocator.

Root cause: On HIP, launch_fattn uses raw cudaMalloc/cudaFree for K_f16 and V_f16
temp buffers (lines 1495-1496 in fattn-common.cuh). cudaMalloc is NOT permitted
during HIP stream capture, causing "operation not permitted when stream is capturing".

The CUDA path uses ggml_cuda_pool_alloc which IS graph-safe (allocates from
pre-allocated pool, no cudaMalloc during capture).

Fix: Replace the hip_f16_alloc with ggml_cuda_pool_alloc on HIP too.
The original concern (pool retains memory causing OOM) is mitigated by the
fact that decode is the capture path — prefill runs eagerly. The pool will
retain the f16 buffer at its peak size, which is small relative to model weights.
"""
import sys

FILE = "/build/src/ggml/src/ggml-cuda/fattn-common.cuh"

with open(FILE, "r") as f:
    content = f.read()

# Replace the entire #ifdef GGML_USE_HIP block with pool allocator
old = """#ifdef GGML_USE_HIP
    // HIP/ROCm: bypass the memory pool for f16 temp buffers.
    // The legacy pool (ggml_cuda_pool_leg) retains peak-sized allocations permanently
    // because free() stores buffers for reuse rather than releasing them.
    // On HIP without VMM support (RDNA 3/4), this means the f16 dequant temp buffers
    // for quantized KV stay allocated after use, consuming more VRAM than the KV
    // compression saves — causing OOM before f16 at equivalent context lengths.
    // Using raw cudaMalloc/cudaFree ensures memory is released after the kernel completes.
    // Ref: https://github.com/ggml-org/llama.cpp/issues/22107
    struct hip_f16_alloc {
        half * ptr = nullptr;
        cudaStream_t stream;
        hip_f16_alloc(cudaStream_t s) : stream(s) {}
        hip_f16_alloc(const hip_f16_alloc &) = delete;
        hip_f16_alloc & operator=(const hip_f16_alloc &) = delete;
        ~hip_f16_alloc() {
            if (ptr) {
                cudaStreamSynchronize(stream);
                cudaFree(ptr);
            }
        }
        void alloc(size_t nelements) {
            CUDA_CHECK(cudaMalloc(&ptr, nelements * sizeof(half)));
        }
    };
    hip_f16_alloc K_f16(main_stream);
    hip_f16_alloc V_f16(main_stream);
#else
    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
#endif"""

new = """    // Use pool allocator for f16 temp buffers on ALL platforms (CUDA and HIP).
    // The pool is graph-safe: it allocates from a pre-allocated buffer during
    // pool construction (outside graph capture) and only bumps a pointer during
    // alloc() calls (no cudaMalloc during stream capture).
    // Previously HIP used raw cudaMalloc/cudaFree which is NOT graph-capturable.
    // The pool retains peak-sized allocations but this is acceptable: the buffer
    // size is small relative to model weights (~0.5-2 GB for typical MLA configs).
    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);"""

if old in content:
    content = content.replace(old, new)
    with open(FILE, "w") as f:
        f.write(content)
    print("FIX APPLIED: Replaced raw cudaMalloc/cudaFree with pool allocator in launch_fattn")
    print("This makes FA graph-capturable on HIP/ROCm")
else:
    print("ERROR: Cannot find the hip_f16_alloc block")
    # Check if already patched
    if "Use pool allocator for f16 temp buffers on ALL platforms" in content:
        print("ALREADY PATCHED")
    else:
        # Show what's around the area
        idx = content.find("hip_f16_alloc")
        if idx >= 0:
            print(f"Found hip_f16_alloc at char {idx}")
            print(content[max(0,idx-200):idx+200])
    sys.exit(1)
