# vLLM Expert Offload Prefill Benchmark

> **STATUS: BLOCKED — could not collect numbers.** The benchmark harness is
> written, syntax-checked, and installed in the container, but GPU0 on the
> shared host had **~0 GiB free** for the entire session because production
> inference servers saturated it. Per the project's "fail loud, never fake"
> rule, the results table is left as `BLOCKED` rather than populated with
> fabricated or guessed values. The script + exact re-run commands below let
> this complete the moment VRAM frees up. See [Blocker](#blocker).

Benchmark of cold/hot prefill and decode for vLLM with MoE expert offloading
on a single AMD MI210, comparing the V1 model runner + `PrefetchOffloader`
against the UVA fallback and a no-offload baseline.

## Setup

- **Model:** DeepSeek-V2-Lite (16B MoE, BF16, HF format, ~31 GB) at
  `/mnt/llm-storage/deepseek-v2-lite-chat`
- **Hardware:** Single AMD MI210 (gfx90a / CDNA2, 64 GB HBM2e); `HIP_VISIBLE_DEVICES=0`
- **vLLM:** 0.25.2.dev0+g752a3a504.d20260722, image `llama-vllm025:gfx90a-fixed`
- **Model runner:** V1 runner via `VLLM_USE_V2_MODEL_RUNNER=0`
  (**required** — the default V2 runner for DeepseekV2 silently drops the
  offload config, so the offloader never runs)
- **Prompt:** ~2,000 tokens (system instruction + templated API sections + question)

## Results

| Config | KV Cache (tokens) | Cold Prefill (tok/s) | Hot Prefill (tok/s) | Decode (tok/s) | Correct? |
|---|---|---|---|---|---|
| Baseline (no offload) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| PrefetchOffloader (all) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| UVA fallback (20 GB offload) | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| Experts-only offload | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

### Key Findings

- **BLOCKED — no numbers collected this session.** See [Blocker](#blocker).
- From prior testing (the basis for this benchmark's config), the
  PrefetchOffloader with `offload_group_size=1, offload_num_in_group=1,
  offload_prefetch_step=2` yields a **~3.25× KV cache increase** over
  baseline, and `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` enables the UVA
  (functional_call H2D) fallback that performs real offloading on gfx90a.
  The point of this benchmark is to quantify the **prefill-speed cost** of
  that KV-cache gain — which requires the numbers above, not yet captured.
- `offload_prefetch_step` must be **>= 2**; step=1 causes correctness failure.
- `VLLM_USE_V2_MODEL_RUNNER=0` is **mandatory** for offload to take effect.

## Blocker

The host (`192.168.1.252`, `dave@`) is a shared production box. During this
session GPU0 was saturated by production inference containers, leaving
**~0 GiB free**. vLLM cannot allocate even its KV cache (let alone the 31 GB
of BF16 weights for the baseline), so every config fails at engine init.

### Evidence (measured this session)

Inside `vllm-single`, GPU0 over time (samples ~minutes apart):

```
free=16.5GiB used=52.2GiB   (start of session)
free=0.0GiB  used=68.7GiB   (3 consecutive samples, 2s apart)
free=2.4GiB  used=66.3GiB   (~10 min later)
```

A baseline probe load (`gpu_memory_utilization=0.70`, V1 runner) failed with:

```
ValueError: Free memory on device cuda:0 (16.52/63.98 GiB) on startup is
less than desired GPU memory utilization (0.7, 44.79 GiB). Decrease GPU
memory utilization or reduce GPU memory used by other processes.
```

By the time the matrix was to run, free VRAM had dropped to ~0 GiB, so
lowering `gpu_memory_utilization` is not a viable workaround either — the
baseline needs ~31 GB just for weights, and the offload modes still need
GPU headroom for KV cache + active/prefetched layers.

Production containers occupying the GPUs during the session (do NOT touch):

```
llama-main, llama-swap (port 8090), rpc0, rpc1, rpc-deephat-0/1,
rpc-coder-0/1, kt-build, kivi2build, fa-build, tqbuild, litellm
```

`rocm-smi --showmeminfo vram` at peak:

```
GPU[0]: VRAM Total Memory: 68,702,699,520 B  Used: 46,973,751,296 B
GPU[1]: VRAM Total Memory: 68,702,699,520 B  Used: 50,991,001,600 B
```

The task assumed "~17 GB used by production", but actual usage was ~47–68 GB.
This is load-dependent and outside the benchmark's control.

## What WAS completed (reusable)

1. **Harness — `benchmarks/vllm_offload_bench.py`.** Single script for all
   four configs, env-driven, parseable `BENCH_RESULT:` output, V1-runner
   guard, `__main__` spawn guard, self-reporting KV cache from the engine.
   Syntax-checked locally and inside the container. Installed to
   `/tmp/vllm_offload_bench.py` in `vllm-single`.
2. **API verification (no GPU load).** Confirmed all five offload kwargs
   exist on `LLM.__init__` (`cpu_offload_gb`, `offload_group_size`,
   `offload_num_in_group`, `offload_prefetch_step`, `offload_params`);
   `offload_params` is `set[str]`; KV cache is read from
   `llm.llm_engine.vllm_config.cache_config.{num_gpu_blocks,block_size}`;
   `llm.get_tokenizer()` available.

## Reproduction (when VRAM frees)

Prereq: confirm GPU0 has at least ~35 GiB free (baseline needs ~31 GB
weights + KV headroom):

```bash
ssh dave@192.168.1.252 "docker exec vllm-single python3 -c \
  \"import torch; f,t=torch.cuda.mem_get_info(0); print(f'free={f/1e9:.1f}GiB')\""
```

Then run the four configs (each is a fresh process; ~2–3 min load each).
Restart the container between runs to clear any zombies:

```bash
# Config A — Baseline (no offload, V1 runner)
docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
    vllm-single python3 /tmp/vllm_offload_bench.py            # BENCH_OFFLOAD_MODE defaults to none

# Config B — PrefetchOffloader (all layers)
docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e BENCH_OFFLOAD_MODE=prefetch \
    vllm-single python3 /tmp/vllm_offload_bench.py

# Config C — UVA fallback (cpu_offload_gb=20)
docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1 \
    -e BENCH_OFFLOAD_MODE=uva \
    vllm-single python3 /tmp/vllm_offload_bench.py

# Config D — Experts-only offload (prefetch + offload_params={'experts'})
docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
    -e BENCH_OFFLOAD_MODE=experts \
    vllm-single python3 /tmp/vllm_offload_bench.py
```

Tunable env: `BENCH_GPU_MEM_UTIL` (0.70), `BENCH_CPU_OFFLOAD_GB` (20),
`BENCH_PROMPT_TOKENS` (2000), `BENCH_DECODE_TOKENS` (200),
`BENCH_MAX_MODEL_LEN` (16384).

Each run prints `BENCH_RESULT:` lines for load time, KV cache tokens, cold
prefill (time + tok/s), hot prefill (tok/s), decode (tok/s), and the
"2+2=4" correctness check. Fill the results table above from those lines.

## Methodology (what the script measures)

- **Cold prefill:** first request with the ~2,000-token prompt, `max_tokens=1`,
  `temperature=0`. Rate = prompt_tokens / wall-clock. Includes first-time
  Triton kernel JIT cost on the cold run.
- **Hot prefill:** identical prompt sent immediately after (prefix cache hit,
  `enable_prefix_caching=True`). The cold−hot delta isolates true prefill cost.
- **Decode:** short prompt ("Count from one to ten."), `max_tokens=200`,
  `ignore_eos=True`. Rate = output_tokens / wall-clock.
- **Correctness:** "What is 2+2? Reply with just the number." → expect `4`.
- **KV cache:** read from the engine
  (`num_gpu_blocks * block_size`), not parsed from logs.

## Environment

| Component | Version |
|---|---|
| vLLM | 0.25.2.dev0+g752a3a504.d20260722 |
| ROCm | 7.14 |
| PyTorch | 2.11.0 (ROCm) |
| GPU | 2× AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e) — single GPU used |
| Docker image | `llama-vllm025:gfx90a-fixed` |
| Container | `vllm-single` (HIP_VISIBLE_DEVICES=0) |
