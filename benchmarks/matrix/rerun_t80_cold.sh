#!/bin/sh
# Re-run the t80 cold16k arm voided by a false-positive cold-cache assertion.
# The first request paid 124.8 s of Triton JIT compilation against a 3.4 s
# steady state; the harness read that as prefix caching, since "first slow
# because of warmup" and "later fast because of caching" are indistinguishable
# from the outside. bench_matrix.py now issues a discarded warmup request at
# the real prompt shape before timing.
B=/mnt/llm-storage/bench-matrix
while ps -eo cmd | grep -qE "[s]weep_t80|[s]weep_t235|[s]weep_glm|[b]f16_last|[t]une_moe_targeted"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### t80-awq4 cold16k re-run (with warmup) $(date -u)"
TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=9000 \
  $B/bin/run_arm.sh t80-awq4 80B awq-int4 vllm-aiter $B/t80-awq \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! t80-awq4 rerun failed (recorded)"
echo "### rerun done $(date -u)"
