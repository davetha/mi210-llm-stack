#!/usr/bin/env bash
# Re-run round 19's W8A8 arms for the decode numbers they lost.
#
# WHY THEY WERE LOST. round19_aiter_components.sh did not set LONGCTX_TOKENS, and
# run_arm.sh's auto-clamp only recognised --ctx-size / -c, which are the
# llama.cpp spellings. Round 19 is a vLLM round using --max-model-len 131072, so
# the clamp never fired, the longctx workload asked for the default 262144
# tokens, and every rep returned:
#
#   HTTP 400: This model's maximum context length is 131072 tokens.
#             However, you requested 256 output tokens and your prompt contains
#             at least 130817 input tokens...
#
# All four arms, decode lost. That is the identical failure run_arm.sh's clamp
# block was written to prevent -- its own comment records it killing the whole
# --n-cpu-moe decode sweep, "four arms, all four failed, none of them for any
# reason worth knowing." Same bug, different dialect. run_arm.sh now clamps on
# --max-model-len too; this round sets LONGCTX_TOKENS explicitly regardless,
# because the arms must not depend on that fix having landed.
#
# 101000 IS CHOSEN, NOT ARBITRARY. docs/28 records t35-w8a8 TP=2 decode as
# 43.4 t/s "@101k", so measuring at the same context makes this directly
# comparable to the published baseline rather than to nothing.
#
# WHAT THE PREFILL ARMS ALREADY SETTLED. cold16k came through clean and was flat:
#
#   base 8351.5 | linear 8344.3 | moe 8364.8 | all 8337.7   t/s
#
# a 0.32% spread, i.e. noise. So VLLM_ROCM_USE_AITER_LINEAR and _MOE do nothing
# for W8A8 PREFILL. Decode is the remaining question and the more plausible one:
# AITER's MoE path would matter most at small batch, which is exactly where
# prefill throughput cannot see it.
#
# NOTE ON THE RESEARCH NOTE THIS ROUND WAS TESTING. benchmarks/matrix/research/
# marks both flags "NO-EFFECT -- fails gate". The stated reason is wrong -- they
# are not architecture-gated, and setting MOE=0 actively REMOVES the AITER MoE
# backend (docs/28, round 19 header). But on the int8 path the conclusion is so
# far correct, which is worth stating plainly rather than quietly dropping.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/t35-w8a8
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 19b (W8A8 decode, AITER components) ==="

run() {  # run <label>
    local label="$1"
    echo "--- $label  [env: ${ARM_ENV:-none}] ---"
    VLLM_EXTRA_ENV="${ARM_ENV:-}" \
    LONGCTX_TOKENS=101000 ARM_TIMEOUT=7200 READY_TIMEOUT=2400 \
        "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        || echo "!! $label failed (recorded)"
}

ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=0"
run t35w8a8d-base
ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_USE_AITER_MOE=0"
run t35w8a8d-linear
ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=1"
run t35w8a8d-moe
ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_USE_AITER_MOE=1"
run t35w8a8d-all

echo "=== $(date -u +%T) round 19b done ==="
echo "--- W8A8 decode vs AITER components (docs/28 baseline: 43.4 t/s @101k) ---"
for l in t35w8a8d-base t35w8a8d-linear t35w8a8d-moe t35w8a8d-all; do
    f="results/$l-longctx.json"
    if [ -f "$f" ]; then
        python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-18s prompt=%6s prefill=%9.1f decode=%7.2f' % (sys.argv[2],
    d.get('actual_prompt_tokens'), d.get('implied_prefill_tps_median') or 0,
    d.get('decode_tps_median') or 0))" "$f" "$l"
    else
        echo "  $l: no longctx result"
    fi
done
