# vLLM MoE Expert Cache on gfx90a

> ⚠️ **Two things in this guide were wrong; corrected 2026-08-09. Read
> [`docs/60`](../docs/60-the-expert-cache-question-was-already-answered.md)
> first.** `--moe-expert-cache-size` is **not a shipped vLLM feature** — it is
> still open PR
> [vllm-project/vllm#37190](https://github.com/vllm-project/vllm/pull/37190).
> And the `VLLM_USE_AITER=0` advice below is actively harmful on this card
> today. The MoE-config section is still good.
>
> Separately, the expert-cache *idea* has been measured and lost:
> `mi210-vllm/docs/ROUTING.md` traced the router on GLM-5.2 and found
> near-uniform routing (per-layer entropy ~0.91, top-10% of experts covering 42%
> of accesses), so no VRAM cache concentrates the working set.

## What the expert cache does — and does not

`--moe-expert-cache-size N` would implement an **LRU expert weight cache** in
VRAM: experts loaded on demand when first routed to, evicted least-recently-used
when full — vLLM's equivalent of llama.cpp's `-ot ...=CPU` selective offload,
but automatic.

**The flag is not in any image you have.** Current vLLM `main` exposes only UVA
and a group/layer `PrefetchOffloader` in `vllm/config/offload.py`; the cache is
PR #37190, opened 2026-03-16 and unmerged as of 2026-08-05. Passing
`--moe-expert-cache-size` to the stack as built is an unrecognised-argument
error, so the recipe below cannot be followed as written.

"Unmerged" is not by itself an objection here — `patches/registry.yaml` in
`mi210-vllm` carries five unmerged vLLM branches on purpose, and `aiter-cdna2`
goes considerably further than cherry-picking a PR. If #37190 were worth having,
the correct move would be to add it to the registry with an `obsolete_when`
predicate, exactly like the others.

It is not worth having, for two independent reasons. **Mechanically**, #37190 is
BF16 with limited FP8, requires `--enforce-eager`, does not support EP>1, and
copies synchronously on every miss — it cannot serve a Q4_K GGUF path at all,
and `--enforce-eager` alone costs the 99.9%-coverage graph-mode decode that
`docs/42` measured (eager was 3.5× slower wall). **Empirically**, the routing
trace says a cache would not pay even if it were free — see `docs/60`.

## Environment flags for gfx90a

```bash
export VLLM_USE_TRITON_FLASH_ATTN=1   # use Triton FA, not the broken rocWMMA path
```

| Flag | Why |
|------|-----|
| `VLLM_USE_TRITON_FLASH_ATTN=1` | Forces the Triton attention backend (wave64-safe) instead of the rocWMMA path (CDNA3+ only). |

> ⚠️ This guide used to say `VLLM_USE_AITER=0`, on the grounds that "AITER
> rejects gfx90a — only targets gfx942/gfx950." **That is true of stock AITER,
> false as a claim about the hardware, and out of date on this stack.**
>
> Stock AITER ships ASM code objects for gfx942/gfx950/gfx1250 only and checks
> the architecture by name in every dispatch path, so on an *unpatched* MI210
> `VLLM_ROCM_USE_AITER=1` has no effect at all and the engine silently serves a
> generic fallback. The old advice was reacting to something real.
>
> What changed is not upstream — it is us.
> [`aiter-cdna2`](https://github.com/davetha/aiter-cdna2) binary-repatches
> **242 of 1,422** gfx942 code objects to gfx90a and opens the gating
> predicates (`on_mi3xx()`, enumerated in `docs/35`). With that applied, ASM
> flash attention is 80/80 configs numerically exact (`docs/19`), ASM
> paged-attention decode 48/48 exact (`docs/18`), and the CK int8 GEMM is worth
> **2.9–3.5× decode** on W8A8 (`docs/57`).
>
> So the advice is conditional, not "always on":
>
> - **Patched image** (`aiter-cdna2` + `enable_vllm_aiter_gfx90a.py` +
>   `enable_aiter_ck_gemm_gfx90a.py` — what `mi210-vllm` builds): serve W8A8
>   with `VLLM_ROCM_USE_AITER=1` **and** `VLLM_ROCM_USE_AITER_LINEAR=1`, then
>   grep the log for `Selected .*ScaledMMLinearKernel` before trusting a number.
> - **Stock ROCm/vLLM image**: the flag genuinely does nothing. Not harmful, not
>   a fix.
> - **⚠️ Never for sliding-window models, on any image.** `ROCM_AITER_FA`
>   **silently corrupts sliding-window attention on gfx90a** — no crash, no
>   assert, no log line, healthy server, normal throughput, garbage output.
>   Verified on `poolside/Laguna-S-2.1-INT4` (36 `sliding_attention` / 12
>   `full_attention`); `TRITON_ATTN` fixes it. See
>   [`aiter-cdna2/docs/aiter-fa-sliding-window-corruption.md`](https://github.com/davetha/aiter-cdna2/blob/main/docs/aiter-fa-sliding-window-corruption.md).

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
