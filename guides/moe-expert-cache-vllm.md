# vLLM MoE Expert Cache on gfx90a

How to set up vLLM with `--moe-expert-cache-size` on the MI210 — the best
alternative to llama.cpp for MoE inference when you need high concurrency.

## What the expert cache does

`--moe-expert-cache-size N` (vLLM 0.25+) implements an **LRU expert weight cache**
in VRAM: experts are loaded on demand when first routed to, and evicted
(least-recently-used) when the cache is full. This is vLLM's equivalent of
llama.cpp's `-ot ...=CPU` selective offload, but automatic.

## Environment flags for gfx90a

```bash
export VLLM_USE_AITER=0          # disable AITER (gfx90a not in its target list)
export VLLM_USE_TRITON_FLASH_ATTN=1   # use Triton FA, not the broken rocWMMA path
```

| Flag | Why |
|------|-----|
| `VLLM_USE_AITER=0` | AITER (AMD Instinct Triton Extensions for ROCm) rejects gfx90a — only targets gfx942/gfx950. |
| `VLLM_USE_TRITON_FLASH_ATTN=1` | Forces the Triton attention backend (wave64-safe) instead of the rocWMMA path (CDNA3+ only). |

## Single-GPU testing first

The two MI210s are PCIe-linked (no xGMI). Multi-GPU (TP=2) works but has edge cases (peer-DMA, uneven expert split). **Always test single-GPU first:**

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_USE_AITER=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=1 \
  --entrypoint python3 llama-vllm025:gfx90a \
  -m vllm.entrypoints.openai.api_server \
  --model /models/your-moe-model \
  --dtype half \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

Once single-GPU is confirmed working, add TP=2:

```bash
  -e HIP_VISIBLE_DEVICES=0,1 \
  ... --tensor-parallel-size 2 ...
```

## Model format conversion

vLLM does **not** read GGUF. You need HuggingFace format (safetensors):

```bash
# If you only have GGUF, convert:
# Option A: use the original HF checkpoint + the quantization config
# Option B: convert GGUF → HF via llama.cpp's convert_lora_to_gguf.py / export tools
```

For int4 compressed-tensors models (like Qwen3-Next-80B-A3B), use the original
HF checkpoint directly — vLLM reads compressed-tensors natively.

## MoE config tuning

vLLM warns `Using default MoE config. Performance might be sub-optimal!` because
there's no tuned fused-MoE config for gfx90a. A hand-written config captures
nearly all the gains without the ~17 min/batch autotune cost:

```
BLOCK_SIZE_M: 16/32/64 (by batch bucket)
BLOCK_SIZE_N: 64
BLOCK_SIZE_K: 64   (must divide the K dim)
GROUP_SIZE_M: 1
SPLIT_K: 1          (REQUIRED for int4 — crashes without it)
num_warps: 4        (beats 8 on gfx90a)
num_stages: 2
waves_per_eu: 2
```

> Under TP=2 the MoE intermediate N shards 512→256, so the config filename
> changes from `E=512,N=512,...` to `E=512,N=256,...`.

## When to use vLLM vs llama.cpp

| Need | Use |
|------|-----|
| Many concurrent users / aggregate throughput | **vLLM** (scales to thousands of t/s) |
| Model exceeds VRAM (>128 GB) | **llama.cpp** (`-ot` CPU split — vLLM can't offload) |
| Max single-stream speed | **llama.cpp** (~71 vs ~55 t/s on 80B) |
| GGUF-only model | **llama.cpp** (no conversion) |

See [`docs/04-moe-engine-survey.md`](../docs/04-moe-engine-survey.md) for the full comparison.
