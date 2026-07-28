#!/bin/sh
# Targeted MoE tuning for MI210, run LAST and deliberately narrow.
#
# The full sweep was abandoned after 42 minutes. benchmark_moe.py --tune walks
# a config space that GROWS per batch size -- observed stages of 704, 2,660,
# 4,990 and 7,810 candidates, the last at ~1.07 s/it, i.e. 2h14m for that stage
# alone with more batch sizes still queued. On a serialized GPU that is
# realistically 6-10 hours.
#
# That was the wrong trade against three entirely unmeasured tiers, for a
# config that is shape-specific (E=128, N=768, so tier 1 only) and that targets
# the AWQ path -- already beaten on this hardware by INT8 W8A8, which the
# tuning would not obviously touch.
#
# So: tune the batch sizes the benchmark actually exercises rather than the
# whole grid, and do it after everything else has been measured.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

while ps -eo cmd | grep -qE "[s]weep_t80|[s]weep_t235|[s]weep_glm|[b]f16_last"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

mkdir -p $B/moe-configs
for BS in 1 64; do
  echo "### targeted MoE tuning, batch-size $BS  $(date -u)"
  docker rm -f moe-tune >/dev/null 2>&1
  timeout 5400 docker run --rm --name moe-tune \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
    --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
    -v /mnt/llm-storage:/models -v $BIN:/bin2 \
    -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
    --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
    /bin2/benchmark_moe.py --model /models/bench-matrix/t35-awq \
    --tune --tp-size 1 --dtype auto --seed 1234 --batch-size $BS \
    --save-dir /models/bench-matrix/moe-configs 2>&1 | tail -12
  echo "### batch-size $BS done $(date -u); configs so far:"; ls $B/moe-configs/ 2>/dev/null
done

# Only re-benchmark if a config actually exists. An unchanged number from a
# config that was never produced reads as "tuning did not help", which is a
# different claim entirely.
if ls $B/moe-configs/*.json >/dev/null 2>&1; then
  $B/install_moe_config.sh
  echo "### tuned AWQ re-benchmark $(date -u)"
  TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=6000 \
    VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs \
    $BIN/run_arm.sh t35-awq-moetuned 35B awq-tuned vllm-aiter $B/t35-awq \
    --max-model-len 131072 --safetensors-load-strategy=prefetch \
    || echo "!! tuned arm failed (recorded)"
  echo "### PROOF the config was loaded (absence = tuning bought nothing):"
  grep -c "Using configuration from" $B/logs/t35-awq-moetuned.serverlog 2>/dev/null
else
  echo "### SKIPPED re-benchmark: targeted tuning also produced no config"
fi
echo "### targeted tuning done $(date -u)"
