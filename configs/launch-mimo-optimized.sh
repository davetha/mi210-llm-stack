#!/bin/sh
# launch-mimo-optimized.sh — K-only quantization + FA-off for mimo 230B
#
# Purpose: deploy -ctk q4_0 -ctv f16 -fa off proven optimal in benchmarks.
#
# WHAT CHANGED from launch-mimo.sh:
#   -ctk q8_0 → -ctk q4_0    # K cache 4.63→2.45 GB (47% smaller)
#   -ctv q4_1 → -ctv f16     # V cache f16 required for FA-off (3.2× larger)
#   -fa on   → -fa off       # avoid broken gfx90a FA fallback
#
# VRAM IMPACT at 65536 context (calculated from model architecture):
#   K(q8_0)  4.63 GB → K(q4_0)  2.45 GB  (-2.18 GB)
#   V(q4_1)  1.82 GB → V(f16)   5.81 GB  (+4.00 GB)
#   Total:   6.45 GB →          8.26 GB  (+1.81 GB)
#
# Free VRAM before change: GPU0 ~20 GB, GPU1 ~17 GB
# Expected free after:      GPU0 ~18 GB, GPU1 ~15 GB   ✅ fits
#
# See changes/11-mimo-k-only-quant-deployment.md for full analysis.
#
# Auto-restores a warm session file on startup (if one exists).
# Run warm-mimo-session.sh once to create the warm session file.

PORT="$1"
HOST_SESSION_DIR="/mnt/llm-storage/mimo-sessions"
CONTAINER_SESSION_DIR="/models/mimo-sessions"
SESSION_FILE="${MIMO_SESSION_FILE:-warm-system-prompt.bin}"
mkdir -p "$HOST_SESSION_DIR" 2>/dev/null || true

docker run --rm --name llama-main --network host --init \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g -v /mnt/llm-storage:/models \
  --entrypoint /src/build/bin/llama-server llama-rocm714:latest \
  -m "/models/mimo-v25/Q4_K/Huihui-MiMo-V2.5-abliterated-Q4_K-00001-of-00021.gguf" \
  -ngl 999 \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1" \
  --host 127.0.0.1 --port "$PORT" -c 65536 -b 2048 -ub 2048 -np 1 \
  -fa off \
  -ctk q4_0 -ctv f16 \
  --jinja -a mimo --no-warmup \
  --slot-save-path "$CONTAINER_SESSION_DIR/" &
CONTAINER_PID=$!

# Wait for health, then auto-restore warm session if it exists
for i in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
    if [ -f "${HOST_SESSION_DIR}/${SESSION_FILE}" ]; then
      echo "[launch-mimo] restoring warm session ${SESSION_FILE}..." >&2
      RESTORE=$(curl -s -X POST "http://127.0.0.1:${PORT}/slots/0?action=restore" \
        -H "Content-Type: application/json" \
        -d "{\"filename\":\"${SESSION_FILE}\"}" 2>&1)
      echo "[launch-mimo] restore result: ${RESTORE}" >&2
    else
      echo "[launch-mimo] no warm session file at ${HOST_SESSION_DIR}/${SESSION_FILE}; cold start" >&2
    fi
    break
  fi
  sleep 2
done

wait $CONTAINER_PID
