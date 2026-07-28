#!/bin/sh
# bf16 baseline, run absolutely last.
#
# Requires TP=2 (61 GB will not share a 64 GB card with a KV cache), and TP=2
# on this model measured 697 s/shard -- about three hours. The loader patch that
# was supposed to fix that is REFUTED (810.81 vs 810.49 s/shard on the 235B), so
# nothing has changed and the cost stands.
#
# Three hours for a baseline whose shape is already known -- slowest and
# largest -- is worth less than the tier-4 number that is genuinely missing, so
# it waits for GLM.
B=/mnt/llm-storage/bench-matrix
while ps -eo cmd | grep -qE "[/]rerun_glm3.sh|[/]run_queue2.sh"; do sleep 60; done
while docker ps --format '{{.Names}}' | grep -q '^bench-'; do sleep 30; done
echo "### bf16 baseline start $(date -u)"
TP=2 ARM_TIMEOUT=7200 LONGCTX_TOKENS=110000 READY_TIMEOUT=20000 \
  $B/bin/run_arm.sh t35-bf16 35B bf16 vllm-aiter $B/t35-bf16 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! bf16 failed (recorded)"
echo "### bf16 done $(date -u)"
