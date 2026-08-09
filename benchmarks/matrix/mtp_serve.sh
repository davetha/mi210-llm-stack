#!/usr/bin/env bash
# Serve Qwen3.6-27B-INT8-W8A8 exactly as docs/57 did, optionally with native MTP.
#
#   ./mtp_serve.sh base        -> the docs/57 baseline, unchanged
#   ./mtp_serve.sh mtp 1       -> same + {"method":"qwen3_5_mtp","num_speculative_tokens":1}
#
# Only the speculative config differs between arms. Every other flag, env var,
# mount and device is copied from the exited `vllm-w8a8` container that
# produced the docs/57 numbers.
set -euo pipefail

ARM="${1:-base}"
NSPEC="${2:-1}"
NAME="mtp-${ARM}${2:+-$2}"
PORT=8033
IMAGE=local/vllm-mi210:dsa7-aiterint8

docker rm -f "$NAME" >/dev/null 2>&1 || true

SPEC=()
if [ "$ARM" = "mtp" ]; then
  SPEC=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${NSPEC}}")
fi

docker run -d --name "$NAME" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 991 \
  --ipc=private --shm-size=32g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v /mnt/llm-storage/qwen36-27b-w8a8:/models/w8a8:ro \
  -v /mnt/llm-storage/aiter-cache:/cache/aiter \
  -p ${PORT}:8000 \
  "$IMAGE" \
  vllm serve /models/w8a8 \
    --served-model-name w8a8 \
    --dtype bfloat16 \
    --max-model-len 200000 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 16384 \
    --attention-backend ROCM_AITER_FA \
    --enable-prefix-caching \
    "${SPEC[@]}" \
    --port 8000 --host 0.0.0.0 >/dev/null

echo "started $NAME"

# Wait for health, or die with the tail of the log.
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "READY after ${i}0s"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "CONTAINER DIED"
    docker logs "$NAME" 2>&1 | tail -60
    exit 1
  fi
  sleep 10
done

echo "--- gates ---"
LOG=$(docker logs "$NAME" 2>&1)
echo "$LOG" | grep -E "Selected .*ScaledMMLinearKernel" | tail -2 || echo "!! no kernel-selection line"
echo "$LOG" | grep -iE "speculative|qwen3_5_mtp|num_speculative|draft" | grep -viE "^$" | tail -12 || true
echo "--- weight-load warnings (missing/unexpected) ---"
echo "$LOG" | grep -iE "missing|unexpected|not used|skipping weight" | tail -12 || echo "(none)"
