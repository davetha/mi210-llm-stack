# Serving qwen38 fast — and what "slow" actually feels like

2026-08-20. After the depth-ladder campaign ([59](59-mtp-unblocked-rope-fix-depth-ladders.md)),
production qwen38 was redeployed with the tuned config and then investigated
for a day over a user-perceived slowdown. This page is the launch reference
and the incident record, so none of it has to be re-derived.

## The canonical launch (verified 63–72 tok/s effective)

```bash
/home/dave/launch-qwen38.sh        # on big; full text below for other boxes
```

- image `local/vllm-mi210:mi210.5-aiter-mtpfix-min` (the only image that
  serves every quant × spec combination)
- model `/models/qwen38-27b-ablit-w8a8`, TP2, `--gpu-memory-utilization 0.72`
  (leaves room for the qwen35 tenant on GPU1)
- `--max-model-len 262144` (KV cache 486K tokens, 1.86× concurrency at full
  length), `--max-num-batched-tokens 8192`
- `VLLM_ROCM_USE_AITER=1` (the int8 GEMM: 18 → 42 tok/s)
- `--speculative-config {"method":"mtp","num_speculative_tokens":2}`
  (the robust depth: 80 t/s median, floor above n1's ceiling)

Net effect vs the pre-2026-08-20 production: **~4× decode** on the same
checkpoint, same VRAM envelope.

## What "slow" actually is on this box (measured, not guessed)

1. **TTFT dominates the feel.** Agent prompts average ~49K tokens (long
   sessions: 110K). Average first-token wait 6.7s, 97% of it prefill. The
   prefix cache (81% hit rate) is already the only reason it is 6.7s and
   not ~25s. There is no flag left that fixes this — it is prefill physics
   at ~3000 tok/s.
2. **Co-running penalty:** while a 110K-KV request decodes, concurrent
   requests see ~6 tok/s. Mitigated (not eliminated) by mnbt 8192.
3. **Restart cost:** every container restart pays ~6 min of AITER JIT
   compilation. `AITER_ROOT_DIR` does **not** redirect the a8w8 build cache
   (container-local path). Fix candidate: bake compiled modules into a
   derived image.

## The measurement trap (cost an evening; do not repeat)

Under speculative decoding, **vLLM coalesces accepted draft tokens into
multi-token SSE chunks (~2.7 tokens/chunk)**. A streaming probe that counts
`data:` chunks reports ~27 t/s when the true rate is ~72. An entire A/B
ladder (mnbt, prefix caching, 32K vs 256K, image swap) was internally
consistent at the wrong number because every arm shared the probe bug.

**Rate claims must come from `usage.completion_tokens / wall time` on a
non-streaming request** (or summed per-chunk token deltas). Same lesson
family as the trailing-space probe: the probe is the suspect until the
exact-count method agrees.

## What was exonerated along the way

Clocks/thermals/power (mclk 1600, unthrottled, 52 °C, 93 W busy), prefix
caching (on/off identical throughput — keep it ON for the 4× TTFT saving),
context length (32K vs 256K identical), image (mi210.5-aiter vs mtpfix-min
identical), the idle qwen35 co-tenant (0% CPU when idle).
