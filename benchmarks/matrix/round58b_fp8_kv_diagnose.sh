#!/usr/bin/env bash
# Round 58b: WHY did --kv-cache-dtype fp8 fail to start?
#
# Round 58's fp8 arm exited during startup and the round reported "fp8 KV is
# unavailable on gfx90a". That line was a GUESS baked into the script, not a
# finding: the harness tailed only 25 lines on failure, which captured the API
# server's wrapper traceback --
#     RuntimeError: Engine core initialization failed. See root cause above.
# -- and then the cleanup trap removed the container, taking the actual root
# cause with it. "See root cause above" was never captured.
#
# The failure could be a genuine capability rejection, a KV-cache-size
# recomputation that OOMs, a config validation error, or something unrelated to
# fp8 entirely. Those have completely different consequences and only one of
# them closes the lead.
#
# This round does one thing: start that exact arm and keep ALL of the logs.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 58b: diagnose fp8 KV startup failure ==="

# NO cleanup trap. The container must survive for its logs to be readable.
docker rm -f rd58b-fp8 >/dev/null 2>&1 || true

TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1" \
    "$BIN/serve_vllm_aiter.sh" "$MODEL" rd58b-fp8 8116 \
    --max-model-len 32768 --kv-cache-dtype fp8 >/dev/null

t=0
while [ $t -lt 900 ]; do
    curl -sf http://127.0.0.1:8116/health >/dev/null 2>&1 && { echo "  UNEXPECTED: it started fine after ${t}s"; break; }
    docker ps --format '{{.Names}}' | grep -q '^rd58b-fp8$' || { echo "  exited after ${t}s"; break; }
    sleep 10; t=$((t+10))
done

echo ""
echo "=== FULL CONTAINER LOG -> $LOGS/rd58b-fp8.full ==="
docker logs rd58b-fp8 > "$LOGS/rd58b-fp8.full" 2>&1 || true
echo "  captured $(wc -l < "$LOGS/rd58b-fp8.full") lines"
echo ""
echo "=== the actual root cause (first real exception, not the wrapper) ==="
grep -anE "Error|Exception|assert|not support|unsupported|Invalid|raise " "$LOGS/rd58b-fp8.full" \
  | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib|Engine core initialization failed" \
  | head -20
echo ""
echo "=== fp8 / kv-cache mentions ==="
grep -anEi "fp8|kv.cache.dtype|kv_cache_dtype|e4m3" "$LOGS/rd58b-fp8.full" | head -15
echo ""
echo "=== last 40 lines verbatim ==="
tail -40 "$LOGS/rd58b-fp8.full"

docker rm -f rd58b-fp8 >/dev/null 2>&1 || true
echo ""
echo "=== $(date -u +%T) round 58b done ==="
echo "If the root cause is a capability rejection, fp8 KV closes on gfx90a."
echo "If it is OOM or a config error, the lead is still open and the harness"
echo "needs fixing rather than the conclusion being written."
