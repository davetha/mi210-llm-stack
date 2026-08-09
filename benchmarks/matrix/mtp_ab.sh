#!/usr/bin/env bash
# docs/57 baseline vs native MTP at N=1,2,3 on Qwen3.6-27B-INT8-W8A8, 1x MI210.
# Only --speculative-config differs between arms.
set -uo pipefail

PORT=8033
DEPTHS="${DEPTHS:-0,8192,32768}"
REPS="${REPS:-3}"
OUT=~/cu-mask-probe/results
mkdir -p "$OUT"

wait_ready() {
  local name="$1"
  for i in $(seq 1 240); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -qx "$name" || { echo "DIED"; return 1; }
    sleep 5
  done
  echo "TIMEOUT"; return 1
}

spec_metrics() {
  curl -s "http://localhost:${PORT}/metrics" 2>/dev/null \
    | grep -E "^vllm:spec_decode" | grep -v "^#" || true
}

run_arm() {
  local arm="$1" n="${2:-}"
  local name="mtp-${arm}${n:+-$n}"
  local tag="${arm}${n:+$n}"
  echo "############ ARM $tag ############"
  ~/cu-mask-probe/mtp_serve.sh "$arm" ${n:+$n} >/dev/null 2>&1 &
  sleep 20
  wait_ready "$name" || { docker logs "$name" 2>&1 | tail -30; docker rm -f "$name" >/dev/null 2>&1; return 1; }

  echo "--- kernel gate ---"
  docker logs "$name" 2>&1 | grep -E "Selected .*ScaledMMLinearKernel" | tail -1
  docker logs "$name" 2>&1 | grep -E "SpeculativeConfig\(" -o | tail -1
  docker logs "$name" 2>&1 | grep -oE "num_spec_tokens=[0-9]+" | tail -1

  # A short warm request so the first real measurement is not a cold graph.
  curl -s "http://localhost:${PORT}/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"w8a8","prompt":"hello","max_tokens":8}' >/dev/null 2>&1

  python3 /tmp/bench_llamabench_style.py "http://localhost:${PORT}" w8a8 \
      "$DEPTHS" "$REPS" "q36-27b-w8a8 $tag" 2>&1 | tee "$OUT/bench-$tag.txt"

  echo "--- spec metrics ($tag) ---" | tee -a "$OUT/bench-$tag.txt"
  spec_metrics | tee -a "$OUT/bench-$tag.txt"

  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
}

run_arm base
for n in 1 2 3; do run_arm mtp "$n"; done
echo "ALL DONE"
