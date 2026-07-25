#!/bin/sh
# test-iq2xxs-allvram.sh — All-VRAM mimo with IQ2_XXS dynamic expert quant
# THE TEST: Does fitting the entire model in VRAM eliminate the DDR4 bottleneck?
# Expected: 10-16x decode improvement (2.7 -> 24-42 tok/s) per Kimi theoretical analysis
#
# PREREQUISITES:
#   1. IQ2_XXS download complete at /mnt/llm-storage/mimo-v25/UD-IQ2_XXS/
#   2. Production stopped: docker stop llama-swap && docker stop llama-main rpc0 rpc1
#
PORT="${1:-8095}"
MODEL_DIR="/mnt/llm-storage/mimo-v25/UD-IQ2_XXS"
FIRST_SHARD=$(ls "${MODEL_DIR}"/*UD-IQ2_XXS*00001*.gguf 2>/dev/null | head -1)

if [ -z "$FIRST_SHARD" ]; then
    echo "ERROR: No IQ2_XXS shards found in ${MODEL_DIR}"
    echo "Download with: hf download unsloth/MiMo-V2.5-GGUF --include '*UD-IQ2_XXS*' --local-dir ${MODEL_DIR}"
    exit 1
fi

echo "[test-iq2xxs] Using model: ${FIRST_SHARD}"
echo "[test-iq2xxs] All layers on GPU (-ngl 999, NO CPU split)"
echo "[test-iq2xxs] Starting on port ${PORT}..."

# ALL LAYERS ON GPU - no -ot CPU split needed since IQ2_XXS is ~84GB = fits in 128GB VRAM
docker run --rm --name llama-iq2xxs-test --network host --init \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g -v /mnt/llm-storage:/models \
  -e LD_LIBRARY_PATH=/models/turbo-build/src/build/bin \
  -e GGML_CUDA_DISABLE_GRAPHS=1 \
  --entrypoint /models/turbo-build/src/build/bin/llama-server llama-rocm714:latest \
  -m "${FIRST_SHARD}" \
  -ngl 999 \
  --host 127.0.0.1 --port "$PORT" -c 65536 -b 2048 -ub 2048 -np 1 \
  -fa on -ctk q8_0 -ctv f16 \
  --jinja -a mimo-iq2 --no-warmup &

echo "[test-iq2xxs] Waiting for health..."
for i in $(seq 1 300); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
    echo "[test-iq2xxs] READY after ${i}x2s"
    echo ""
    echo "=== CORRECTNESS TEST ==="
    curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"messages":[{"role":"user","content":"What is 2+2? One word."}],"max_tokens":20}' \
      2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message']; print('CONTENT:',c.get('content','')); print('REASONING:',c.get('reasoning_content','')[:80]); u=d['usage']; t=d.get('timings',{}); print(f'TOKS={u[\"prompt_tokens\"]} PREFILL={u[\"prompt_tokens\"]/(t.get(\"prompt_ms\",1)/1000):.0f}tok/s')" 2>&1
    echo ""
    echo "=== PREFILL BENCHMARK (long prompt) ==="
    LONGPROMPT="Explain in detail: ROCm, CDNA2, HBM2e, gfx90a, MFMA, MoE, KV cache, flash attention on AMD MI210. "
    curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"${LONGPROMPT}${LONGPROMPT}${LONGPROMPT}\"}],\"max_tokens\":50}" \
      2>&1 | python3 -c "import sys,json; d=json.load(sys.stdin); u=d['usage']; t=d.get('timings',{}); print(f'PROMPT={u[\"prompt_tokens\"]} toks | PREFILL={u[\"prompt_tokens\"]/(t.get(\"prompt_ms\",1)/1000):.0f} tok/s')" 2>&1
    echo ""
    echo "=== VRAM USAGE ==="
    docker exec llama-iq2xxs-test rocm-smi --showmeminfo vram 2>&1 | grep -E "gpu|memory|VRAM" | head -6
    echo ""
    echo "=== DONE - compare to production 392 tok/s ==="
    exit 0
  fi
  sleep 2
done
echo "[test-iq2xxs] FAILED to become healthy"
exit 1
