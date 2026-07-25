#!/bin/sh
# warm-mimo-session.sh PORT [SESSION_FILENAME]
# Sends a representative system prompt to mimo, then saves the KV session.
# Run this ONCE after deploying launch-mimo.sh. Future mimo restarts will
# auto-restore this session, making the first request after restart instant
# (only NEW tokens are prefilled, not the cached system prompt).
#
# To re-warm (e.g. after changing your system prompt), just re-run this script.
PORT="${1:-5803}"
SESSION_FILE="${2:-warm-system-prompt.bin}"
SESSION_DIR="/mnt/llm-storage/mimo-sessions"
mkdir -p "$SESSION_DIR"

# Representative system prompt — adjust to match your actual client workload.
# If using opencode with oh-my-openagent (~18K tokens), paste that full system
# prompt here instead. The key is: the saved KV must match what your client
# typically sends as the prefix of every request.
cat > /tmp/mimo-warmup-payload.json <<EOF
{
  "model": "mimo",
  "messages": [
    {"role": "system", "content": "You are Mimo, a helpful, harmless, and honest AI assistant. You provide accurate, concise, and well-reasoned responses. When working with code, you follow best practices and explain your reasoning step by step. You are knowledgeable about software engineering, system administration, mathematics, and general technical topics. If you are unsure, you say so rather than guessing. You format code blocks with proper syntax highlighting and use markdown for structure."}
  ],
  "max_tokens": 1,
  "temperature": 0,
  "stream": false
}
EOF

echo "=== sending warmup prompt to mimo on :${PORT} ==="
START=$(date +%s.%N)
RESULT=$(curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @/tmp/mimo-warmup-payload.json)
END=$(date +%s.%N)
WALL=$(awk "BEGIN{print $END-$START}")
echo "warmup prompt took ${WALL}s"
echo "usage: $(echo "$RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get(\"usage\",{}))" 2>&1)"

echo
echo "=== saving session to ${SESSION_FILE} ==="
curl -s -X POST "http://127.0.0.1:${PORT}/slots/0?action=save" \
  -H "Content-Type: application/json" \
  -d "{\"filename\":\"${SESSION_FILE}\"}" 2>&1

echo
echo "=== verifying session file ==="
ls -lh "${SESSION_DIR}/${SESSION_FILE}" 2>&1
echo
echo "Done. Future mimo starts will auto-restore this session."
echo "The first request after a restart will only need to prefill NEW tokens."
