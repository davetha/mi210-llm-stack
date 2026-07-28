#!/bin/sh
# Serialized follow-up chain. Everything here needs the GPUs, so it runs one at
# a time behind the tier-1 sweep. Kept as ONE script rather than several queued
# jobs precisely so two cannot wake in the same gap and collide -- a second
# model loading against a busy card fails with a free-memory error that reads
# like a config problem.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

wait_gpus() {
  while ps -eo cmd | grep -q "[s]weep_t35_rest"; do sleep 60; done
  while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
}

echo "### chain start $(date -u)"
wait_gpus

# 1. w8a8, retried with the ROCm INT8 MoE patch. Previously died with
#    "No Int8 MoE backend supports the deployment configuration".
echo "### [1/4] w8a8 retry $(date -u)"
TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=6000 \
  $BIN/run_arm.sh t35-w8a8 35B w8a8 vllm-aiter $B/t35-w8a8 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! w8a8 still failing (recorded)"
wait_gpus

# 2. q8_0 long context, re-run with the context size the prompt actually needs.
#    The first attempt asked for 262144 tokens from a server started with
#    --ctx-size 131072 and got HTTP 400 -- a harness misconfiguration, not a
#    model or kernel problem.
echo "### [2/4] q8_0 longctx at true 256k $(date -u)"
TP=1 LONGCTX_TOKENS=250000 READY_TIMEOUT=6000 \
  $BIN/run_arm.sh t35-q8-256k 35B q8_0 llamacpp $B/t35-gguf-q8 \
  --ctx-size 262144 \
  || echo "!! q8 256k failed (recorded)"
wait_gpus

# 3. Tune the Triton fused-MoE kernel for MI210. vLLM ships configs for
#    MI300X/MI308X/MI325X/MI350X/MI355X/R9700/A100 and none for MI210, so the
#    AWQ path runs on heuristic block sizes -- get_moe_wna16_block_config()
#    uses tuned BLOCK_SIZE_N/K only when a config supplies them.
echo "### [3/4] MoE tuning for MI210 $(date -u)"
mkdir -p $B/moe-configs
docker rm -f moe-tune >/dev/null 2>&1
docker run --rm --name moe-tune \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
  -v /mnt/llm-storage:/models -v $BIN:/bin2 \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
  --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
  /bin2/benchmark_moe.py --model /models/bench-matrix/t35-awq \
  --tune --tp-size 1 --dtype auto --seed 1234 \
  --save-dir /models/bench-matrix/moe-configs 2>&1 | tail -30
echo "### tuning output:"; ls -la $B/moe-configs/ 2>/dev/null
wait_gpus

# 4. Re-benchmark AWQ with the tuned config, to measure whether tuning bought
#    anything. If no config was produced, SKIP rather than run -- an unchanged
#    number would otherwise read as "tuning did not help" when in fact nothing
#    was applied.
if ls $B/moe-configs/*.json >/dev/null 2>&1; then
  echo "### [4/4] AWQ re-benchmark with tuned MoE config $(date -u)"
  TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=6000 \
    $BIN/run_arm.sh t35-awq-moetuned 35B awq vllm-aiter $B/t35-awq \
    --max-model-len 131072 --safetensors-load-strategy=prefetch \
    || echo "!! tuned AWQ arm failed (recorded)"
else
  echo "### [4/4] SKIPPED: tuning produced no config JSON, nothing to measure"
fi

echo "### chain done $(date -u)"
