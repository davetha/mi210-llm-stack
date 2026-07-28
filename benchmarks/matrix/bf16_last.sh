#!/bin/sh
# BF16 baseline, deliberately run LAST -- after every other tier.
#
# 61 GB of weights cannot share a 64 GB card with a KV cache, so this arm has
# no choice but TP=2, and TP=2 on this MoE costs roughly three hours of PURE
# CPU in vLLM's per-expert loader: _load_w13 narrows each expert tensor per TP
# rank into a non-contiguous view and then copies per expert, 128 experts x 48
# layers, measured at ~697 s per shard while the same file reads at 3.0 GB/s.
#
# Three hours for one baseline row, ahead of three entirely unmeasured tiers,
# is the wrong trade -- so this waits for all of them. The load time is itself
# the reportable result here; the throughput number is confirmatory.
B=/mnt/llm-storage/bench-matrix
while ps -eo cmd | grep -qE "[f]ollowup_chain|[s]weep_t80|[s]weep_t235|[s]weep_glm"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### bf16 baseline start $(date -u)"
TP=2 LONGCTX_TOKENS=110000 READY_TIMEOUT=20000 \
  $B/bin/run_arm.sh t35-bf16 35B bf16 vllm-aiter $B/t35-bf16 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! bf16 failed (recorded)"
echo "### bf16 done $(date -u)"
