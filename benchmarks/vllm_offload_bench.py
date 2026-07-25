#!/usr/bin/env python3
"""vLLM expert-offload prefill benchmark (DeepSeek-V2-Lite, gfx90a / MI210).

Measures cold/hot prefill, decode, and correctness for the 16B MoE model under
four offload configurations, all using the **V1 model runner**
(`VLLM_USE_V2_MODEL_RUNNER=0`). The V1 runner is REQUIRED for the
PrefetchOffloader: with the default V2 runner the offload config is silently
dropped, so the offloader never runs and the numbers are meaningless.

Config selected via `BENCH_OFFLOAD_MODE` env var:

  - `none`     baseline, no offload
  - `prefetch` PrefetchOffloader (offload_group_size=1, offload_num_in_group=1,
               offload_prefetch_step=2). offload_prefetch_step MUST be >= 2;
               step=1 causes correctness failure.
  - `uva`      UVA fallback (cpu_offload_gb=20). Run with
               `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` on the docker exec.
  - `experts`  prefetch + offload_params={'experts'} (experts only offloaded)

Other env vars:
  - BENCH_MODEL            model path  (default /mnt/llm-storage/deepseek-v2-lite-chat)
  - BENCH_GPU_MEM_UTIL     gpu_memory_utilization (default 0.70)
  - BENCH_CPU_OFFLOAD_GB   cpu_offload_gb for uva mode (default 20)
  - BENCH_MAX_MODEL_LEN    max_model_len (default 16384)
  - BENCH_PROMPT_TOKENS    cold/hot prompt length (default 2000)
  - BENCH_DECODE_TOKENS    decode gen length (default 200)

Example (Config B, prefetch):
    docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
        vllm-single python3 /tmp/vllm_offload_bench.py

Example (Config C, uva):
    docker exec -e VLLM_USE_V2_MODEL_RUNNER=0 \
                -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1 \
        vllm-single python3 /tmp/vllm_offload_bench.py

Output is parseable: each result line is prefixed with `BENCH_RESULT:`.

NOTE on multiprocessing: vLLM spawns workers. Keep everything inside main()
under the `if __name__ == "__main__":` guard so the spawn path stays clean.
"""

import os
import sys
import time
from typing import Dict, Optional

MODEL_DEFAULT = "/mnt/llm-storage/deepseek-v2-lite-chat"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_prompt(tokenizer, target_tokens: int = 2000) -> str:
    """Build a deterministic ~target_tokens prompt.

    Structure mirrors the real workload: a system instruction followed by
    several templated "API documentation" sections, then a short question.
    Trimmed to ~target_tokens with the model's own tokenizer so the measured
    token count is exact.
    """
    sys_msg = (
        "You are a meticulous technical assistant. Answer the final question "
        "using only the provided API reference. If the reference does not "
        "contain the answer, say you do not know.\n\n"
    )
    section = (
        "## Endpoint: /v1/resources/{id}\n"
        "Method: GET\n"
        "Auth: Bearer token\n"
        "Rate limit: 120 requests per minute per token\n"
        "Query parameters:\n"
        "  - id (string, required): unique resource identifier\n"
        "  - fields (string, optional): comma-separated projection list\n"
        "  - expand (string, optional): related entities to inline\n"
        "  - version (integer, optional): pin a specific revision\n"
        "Response 200:\n"
        "  {\"id\": \"abc\", \"name\": \"sample\", \"created_at\": \"2026-07-01T00:00:00Z\",\n"
        "   \"status\": \"active\", \"tags\": [\"alpha\", \"beta\"], \"owner\": {\"id\": 1}}\n"
        "Errors: 404 not_found, 401 unauthorized, 429 rate_limited, 500 internal\n"
        "Notes: the fields projection does not apply to nested owner objects.\n\n"
    )
    body = (section * 64)  # ~64 sections is well above 2000 tokens
    question = "\nQuestion: what is the rate limit and which HTTP status is returned when it is exceeded?\nAnswer:"

    # Trim the body to ~target_tokens using the tokenizer.
    overhead = len(tokenizer.encode(sys_msg + question, add_special_tokens=False))
    section_ids = tokenizer.encode(section, add_special_tokens=False)
    n_sections = max(1, (target_tokens - overhead) // max(1, len(section_ids)))
    prompt = sys_msg + (section * n_sections) + question

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    # If still over, hard-trim by token ids to land near target_tokens.
    if len(ids) > int(target_tokens * 1.05):
        trimmed = ids[:target_tokens]
        prompt = tokenizer.decode(trimmed) + question
    return prompt


def correctness_check(llm) -> Dict:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=20, seed=0)
    t0 = time.perf_counter()
    outs = llm.generate(["What is 2+2? Reply with just the number."], sp)
    dt = time.perf_counter() - t0
    text = outs[0].outputs[0].text.strip()
    ok = "4" in text and ("14" not in text.split()[0] if text else True)
    # Accept any token that is exactly "4" or starts with "4"
    first_token = text.split()[0] if text else ""
    ok = first_token.strip(".,;:!?") == "4"
    return {"text": text, "correct": ok, "time_s": dt}


def run_prefill(llm, prompt: str):
    """Single prefill request (max_tokens=1). Returns (elapsed, token_count)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp)
    dt = time.perf_counter() - t0
    tok_count = len(outs[0].prompt_token_ids)
    return dt, tok_count


def run_decode(llm, n_tokens: int = 200):
    """Decode benchmark: short prompt, n_tokens generation."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=n_tokens, seed=0,
                        ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(["Count from one to ten."], sp)
    dt = time.perf_counter() - t0
    out_tok = len(outs[0].outputs[0].token_ids)
    return dt, out_tok


def build_kwargs(mode: str, model: str, gpu_mem_util: float,
                 max_model_len: int, cpu_offload_gb: float) -> Dict:
    """Construct LLM() kwargs for the requested offload mode."""
    base = dict(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=True,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        dtype="bfloat16",
    )
    if mode == "none":
        return base
    if mode == "prefetch":
        base.update(offload_group_size=1, offload_num_in_group=1,
                    offload_prefetch_step=2)
        return base
    if mode == "uva":
        base.update(cpu_offload_gb=cpu_offload_gb)
        return base
    if mode == "experts":
        base.update(offload_group_size=1, offload_num_in_group=1,
                    offload_prefetch_step=2, offload_params={"experts"})
        return base
    raise ValueError(f"unknown BENCH_OFFLOAD_MODE={mode!r} "
                     f"(expect none|prefetch|uva|experts)")


def main() -> int:
    mode = _env("BENCH_OFFLOAD_MODE", "none")
    model = _env("BENCH_MODEL", MODEL_DEFAULT)
    gpu_mem_util = float(_env("BENCH_GPU_MEM_UTIL", "0.70"))
    cpu_offload_gb = float(_env("BENCH_CPU_OFFLOAD_GB", "20"))
    max_model_len = int(_env("BENCH_MAX_MODEL_LEN", "16384"))
    prompt_tokens = int(_env("BENCH_PROMPT_TOKENS", "2000"))
    decode_tokens = int(_env("BENCH_DECODE_TOKENS", "200"))

    # Safety check: V1 runner is mandatory for offload to actually take effect.
    v2 = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "")
    if mode != "none" and v2 not in ("0", "false", "False"):
        print(f"BENCH_RESULT: WARN VLLM_USE_V2_MODEL_RUNNER={v2!r}; offload "
              f"config will be silently dropped by the V2 runner. Set "
              f"VLLM_USE_V2_MODEL_RUNNER=0.", file=sys.stderr)

    print(f"BENCH_RESULT: mode={mode} model={model} gpu_mem_util={gpu_mem_util} "
          f"max_model_len={max_model_len} prompt_tokens={prompt_tokens} "
          f"decode_tokens={decode_tokens} cpu_offload_gb={cpu_offload_gb}",
          flush=True)

    from vllm import LLM

    kwargs = build_kwargs(mode, model, gpu_mem_util, max_model_len, cpu_offload_gb)

    t_load0 = time.perf_counter()
    llm = LLM(**kwargs)
    load_time = time.perf_counter() - t_load0

    # KV cache size from the engine's own bookkeeping.
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        kv_tokens = int(cc.num_gpu_blocks) * int(cc.block_size)
    except Exception as e:  # accessor drift across versions
        kv_tokens = -1
        print(f"BENCH_RESULT: WARN could not read cache_config: {e}",
              file=sys.stderr)

    print(f"BENCH_RESULT: load_time_s={load_time:.2f} kv_cache_tokens={kv_tokens}",
          flush=True)

    tokenizer = llm.get_tokenizer()
    prompt = build_prompt(tokenizer, prompt_tokens)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"BENCH_RESULT: prompt_token_count={actual_tokens}", flush=True)

    # Cold prefill (first request, no cache).
    cold_dt, _ = run_prefill(llm, prompt)
    cold_rate = actual_tokens / cold_dt if cold_dt > 0 else float("nan")
    print(f"BENCH_RESULT: cold_prefill_time_s={cold_dt:.4f} "
          f"cold_prefill_toks={actual_tokens} cold_prefill_rate_tps={cold_rate:.1f}",
          flush=True)

    # Hot prefill (identical prompt immediately after -> prefix cache hit).
    hot_dt, _ = run_prefill(llm, prompt)
    hot_rate = actual_tokens / hot_dt if hot_dt > 0 else float("nan")
    print(f"BENCH_RESULT: hot_prefill_time_s={hot_dt:.4f} "
          f"hot_prefill_rate_tps={hot_rate:.1f}", flush=True)

    # Decode benchmark.
    dec_dt, out_tok = run_decode(llm, decode_tokens)
    dec_rate = out_tok / dec_dt if dec_dt > 0 else float("nan")
    print(f"BENCH_RESULT: decode_time_s={dec_dt:.4f} decode_out_tokens={out_tok} "
          f"decode_rate_tps={dec_rate:.1f}", flush=True)

    # Correctness.
    corr = correctness_check(llm)
    print(f"BENCH_RESULT: correct={corr['correct']} answer={corr['text']!r}",
          flush=True)

    print(f"BENCH_RESULT: DONE mode={mode}", flush=True)
    return 0


if __name__ == "__main__":
    # Guard is required: vLLM uses 'spawn' multiprocessing; workers re-import
    # this module and must not re-execute main().
    sys.exit(main())
