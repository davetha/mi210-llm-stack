#!/usr/bin/env bash
# Chunk-size sweep for long-context prefill on the patched 256k build.
#
# WHY. The 256k patch took decode at 241k from 0.7485 to 16.0 t/s -- a 21x
# win -- but TTFT is still 338.5 s, i.e. 714 tok/s prefill against the same
# model's 3,002 tok/s at 15k. Most of that 4.2x gap is not a bug:
#
#   attention is O(n^2), so per-token attention cost scales linearly with
#   context. At 16x the tokens (15k -> 241k) per-token attention cost is 16x.
#   If attention is ~20% of prefill at 15k, the predicted slowdown is
#   (0.8 + 16*0.2) / 1.0 = 4.0x. Observed: 4.2x. The gap is essentially all
#   quadratic attention, which no kernel work removes.
#
# What IS tunable is how the prefill is chopped up. vLLM's default
# max_num_batched_tokens is 2048 (DEFAULT_MAX_NUM_BATCHED_TOKENS in
# config/scheduler.py), so a 241,912-token prompt runs as ~119 sequential
# chunks. Each chunk re-launches the full kernel stack and, at 2048 tokens,
# presents a small GEMM to a 30B MoE. Larger chunks mean fewer launches and
# better utilisation, traded against activation memory.
#
# FALSIFIABLE. If prefill is already bound by quadratic attention rather than
# by launch overhead, chunk size will barely move TTFT. That is a useful
# answer too -- it would say the remaining 338 s is physics, and effort should
# go to prefix caching instead.
#
# NOTE the baseline here is cold-cache by construction (--no-enable-prefix-
# caching). In production, prefix caching is the far larger lever for repeated
# long prompts; this sweep is about the cold path only.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

echo "=== $(date -u +%T) waiting for earlier work to finish ==="
while ps -eo cmd | grep -qE "[b]in/round2.sh|[b]in/round2_followup.sh" \
   || docker ps --format '{{.Names}}' | grep -q '^bench-'; do
    sleep 120
done
echo "=== $(date -u +%T) starting prefill sweep ==="

# 2048 is the default and is already measured (338.5 s / 714 t/s), but re-run
# it here so all four points come from the same build and the same day.
for CHUNK in 2048 8192 16384 32768; do
    echo "--- max-num-batched-tokens=$CHUNK  (~$((241912 / CHUNK)) chunks for a 241k prompt) ---"
    VLLM_IMAGE=rocm-vllm-aiter-gfx90a:pa256k \
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=262144 \
        "$BIN/run_arm.sh" "t35-awq-256k-chunk$CHUNK" 35B awq vllm-aiter "$BASE/t35-awq" \
        --max-model-len 262144 --no-enable-prefix-caching \
        --max-num-batched-tokens "$CHUNK" \
        || echo "!! chunk=$CHUNK failed (recorded)"
done

echo "=== $(date -u +%T) prefill sweep complete ==="
echo "baseline to beat: TTFT 338.5 s / 714 tok/s at 241,912 tokens (chunk=2048)"
