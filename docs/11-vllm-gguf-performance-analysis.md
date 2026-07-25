# Why GGUF Is 8× Slower in vLLM

## The Translation Overhead Problem

GGUF quantization formats (Q4_K_M, Q8_0) were designed for llama.cpp's custom kernels. vLLM uses PyTorch + Triton + custom CUDA kernels. When vLLM loads GGUF, it operates as a translation layer.

### llama.cpp (Native GGUF)
```
Q4_K_M block → [fused dequant+matmul kernel] → result
One kernel launch, peak memory bandwidth, hand-tuned per format
```

### vLLM (GGUF Compatibility Layer)
```
Q4_K_M block → [dequant kernel] → fp16 tensor → [rocBLAS matmul] → result
Two kernel launches, fp16 intermediate, no fusion
```

### The Performance Gap

| Issue | llama.cpp (native) | vLLM (compatibility) |
|---|---|---|
| Kernel fusion | Dequant+matmul in ONE kernel | Separate dequant + separate matmul |
| Memory passes | Read quantized ONCE | Read quantized → write fp16 → read fp16 → matmul |
| AWQ/GPTQ Marlin | N/A | Fused kernel → 741 tok/s |
| GGUF Marlin equiv. | IS the native kernel | Doesn't exist in vLLM |

Benchmark (same model, same GPU):
- GGUF in vLLM: 93 tok/s
- AWQ in vLLM: 741 tok/s (8× faster, uses Marlin)
- GGUF in llama.cpp: ~160 tok/s (native kernels)

### On gfx90a Specifically
The gap may be SMALLER on gfx90a because:
- Marlin is NVIDIA-only (not available on AMD)
- AWQ on AMD uses rocBLAS + separate dequant (similar to GGUF path)
- Expected gap: 2-3× not 8×

### Practical Implication
On gfx90a, llama.cpp is the FASTER choice for GGUF models. Its native kernels beat vLLM's translation layer. vLLM's advantages (PagedAttention, expert offload, prefix caching) must overcome the 2-4× speed penalty.
