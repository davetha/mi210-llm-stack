#!/bin/sh
# GLM-4.6 AWQ cold16k, re-run.
#
# The original attempt produced no result: the client blocked 17 minutes reading
# a stream the server had already answered 200 OK, accumulating zero CPU in
# poll_schedule_timeout while both GPUs sat idle. Root cause was urllib's
# timeout being PER READ rather than a request deadline, so the full 3600 s
# budget applied to a single dead read. bench_matrix.py now caps reads at
# max(600, timeout/4) and warmup at 15 minutes.
#
# Without this re-run the 400B tier would have a long-context number and no
# TTFT number, which is the half that the coding-assistant question actually
# cares about.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin
while ps -eo cmd | grep -qE "[/]sweep_glm.sh|[/]rerun_glm_gguf.sh|[/]sweep_t235.sh|[/]bf16_last.sh|[/]rerun_t80_cold.sh|[/]retry_t80_awq8.sh|[/]sweep_int8_formats.sh|[/]tune_moe_targeted.sh"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### GLM-4.6 AWQ cold16k re-run $(date -u)"
TP=2 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh glm46-awq-cold 400B awq-int4 vllm-aiter $B/glm-awq \
  --max-model-len 32768 --safetensors-load-strategy=prefetch \
  --cpu-offload-gb 70 --gpu-memory-utilization 0.95 \
  || echo "!! glm46-awq cold re-run failed (recorded)"
echo "### done $(date -u)"
