#!/bin/sh
# Tier 3: Qwen3-235B-A22B GPTQ-Int4, retried with the MoE loader fix.
#
# The first attempt paced at 810 s per shard across 32 shards -- 6h58m for one
# arm, blocking every tier behind it. That is the per-expert loader cost:
# narrow() leaves the mmap'd checkpoint slice non-contiguous, and copy_ then
# degrades into a strided host-to-device gather, once per expert per layer.
# See configs/fast_moe_expert_load.py.
#
# This run doubles as the fix's validation. If shards do not land dramatically
# faster than 810 s, source-side contiguity was not the whole story and the
# destination tensor is the next suspect -- expert_data is itself a narrowed
# device tensor and may be a scatter.
#
# ~125 GB of weights against 128 GB of VRAM leaves no room for a KV cache, so
# context is deliberately reduced rather than pretending 128k fits.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin
while ps -eo cmd | grep -qE "[s]weep_glm|[b]f16_last|[r]erun_t80_cold|[r]etry_t80_awq8|[s]weep_int8_formats"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### tier-3 start (with MoE loader fix) $(date -u)"
TP=2 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh t235-gptq4 235B gptq-int4 vllm-aiter $B/t235-gptq4 \
  --max-model-len 32768 --safetensors-load-strategy=prefetch \
  --gpu-memory-utilization 0.95 \
  || echo "!! t235-gptq4 failed (recorded)"
echo "### tier-3 done $(date -u)"
echo "### LOADER FIX VALIDATION (baseline was 810 s/shard):"
grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds" $B/logs/t235-gptq4.serverlog 2>/dev/null | tail -1
