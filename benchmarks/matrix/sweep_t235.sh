#!/bin/sh
# Tier 3: Qwen3-235B-A22B GPTQ-Int4 (Apache-2.0, 235B total / 22B active).
#
# ~125 GB of weights against 128 GB of VRAM. That does not leave room for a KV
# cache on top, so this arm is deliberately run at a REDUCED context rather
# than pretending 128k fits: 32768 tokens of KV for a full-attention GQA model
# at 94 layers is already several GB, and the alternative is an OOM that says
# nothing useful.
#
# TP=2 is unavoidable here -- no single 64 GB card holds 125 GB -- which also
# means paying vLLM's per-expert MoE load cost measured in tier 1 (~697 s per
# shard at TP=2, pure CPU in _load_w13, not I/O). Budget hours for the load.
#
# 22B active parameters versus tier 1's 3.3B is the real variable being tested
# at this tier: it is the first arm where compute rather than memory bandwidth
# should dominate prefill.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

while ps -eo cmd | grep -qE "[f]ollowup_chain|[b]f16_last|[s]weep_t80"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

# Refuse to start on an incomplete download rather than half-load a model.
if ! grep -q "OK .*t235-gptq4" $B/dl-big.status 2>/dev/null; then
  echo "### tier-3 SKIPPED: t235-gptq4 download not verified complete"
  exit 0
fi

echo "### tier-3 sweep start $(date -u)"
TP=2 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh t235-gptq4 235B gptq-int4 vllm-aiter $B/t235-gptq4 \
  --max-model-len 32768 --safetensors-load-strategy=prefetch \
  --gpu-memory-utilization 0.95 \
  || echo "!! t235-gptq4 failed (recorded)"
echo "### tier-3 sweep done $(date -u)"
