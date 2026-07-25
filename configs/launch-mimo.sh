#!/bin/sh
# launch-mimo.sh — mimo with persistent KV session support
# Auto-restores a warm session file on startup (if one exists).
# Run warm-mimo-session.sh once to create the warm session file.
PORT="$1"
HOST_SESSION_DIR="/mnt/llm-storage/mimo-sessions"     # host path for file checks
CONTAINER_SESSION_DIR="/models/mimo-sessions"           # container path for --slot-save-path
SESSION_FILE="${MIMO_SESSION_FILE:-warm-system-prompt.bin}"
mkdir -p "$HOST_SESSION_DIR" 2>/dev/null || true

# Start mimo in background so we can restore the session after it is ready
docker run --rm --name llama-main --network host --init \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g -v /mnt/llm-storage:/models \
  --entrypoint /src/build/bin/llama-server llama-rocm714:latest \
  -m "/models/mimo-v25/Q4_K/Huihui-MiMo-V2.5-abliterated-Q4_K-00001-of-00021.gguf" \
  -ngl 999 \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1" \
  --host 127.0.0.1 --port "$PORT" -c 65536 -b 2048 -ub 2048 -np 1 -fa on \
  -ctk q8_0 -ctv q4_1 --jinja -a mimo --no-warmup \
  --slot-save-path "$CONTAINER_SESSION_DIR/" &
CONTAINER_PID=$!

# Wait for health, then auto-restore warm session if it exists (check HOST path)
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
      echo "[launch-mimo] hint: run warm-mimo-session.sh ${PORT} to create one" >&2
    fi
    break
  fi
  sleep 2
done

# Block until container exits (llama-swap requires the wrapper to stay alive)
wait $CONTAINER_PID
