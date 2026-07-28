#!/bin/sh
# t80 AWQ-8bit retried at TP=2.
#
# At TP=1 it OOMed cleanly: 54 GB of weights against a 64 GB card leaves no
# room for a KV cache, and vLLM reported 63.53 GiB allocated with 214 MiB free.
# That is a genuine capacity limit, not a bug, and it is worth having on record
# as the point where 8-bit stops fitting on a single MI210.
#
# TP=2 halves the per-card weight footprint. It also pays vLLM per-expert MoE
# load cost, but this model is 10 shards rather than 16 and loaded in 164 s at
# TP=1, so the penalty should be tolerable here in a way it was not for bf16.
B=/mnt/llm-storage/bench-matrix
while ps -eo cmd | grep -qE "[s]weep_t80|[s]weep_t235|[s]weep_glm|[b]f16_last|[r]erun_t80_cold|[t]une_moe_targeted"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### t80-awq8 retry at TP=2 $(date -u)"
TP=2 LONGCTX_TOKENS=110000 READY_TIMEOUT=12000 \
  $B/bin/run_arm.sh t80-awq8 80B awq-int8 vllm-aiter $B/t80-awq8 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! t80-awq8 still failing at TP=2 (recorded)"
echo "### t80-awq8 retry done $(date -u)"
