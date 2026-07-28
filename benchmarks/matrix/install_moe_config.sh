#!/bin/sh
# Install the tuned MoE config into the image, then make sure an arm actually
# used it.
#
# vLLM resolves MoE configs from two places (fused_moe.py get_moe_configs):
#   1. $VLLM_TUNED_CONFIG_FOLDER   -- user override, checked first
#   2. <package>/fused_moe/configs -- shipped defaults
#
# Installing into (2) is what makes this work with the ALREADY-RUNNING chain:
# editing a live sh script shifts byte offsets and corrupts execution, and the
# chain's own step 4 passes MOE_CONFIG_DIR, which vLLM does not read at all.
# Baking the file into the image means every subsequent run picks it up with no
# environment variable and no edit.
#
# Without this the tuned arm would have reported "no improvement" from a config
# that was never loaded -- a false negative indistinguishable from a real one.
B=/mnt/llm-storage/bench-matrix
S=/opt/python/lib/python3.14/site-packages
CFG=$S/vllm/model_executor/layers/fused_moe/configs

# Poll tightly: the chain starts its next arm seconds after tuning exits.
while ! ls $B/moe-configs/*.json >/dev/null 2>&1; do
  docker ps --format "{{.Names}}" | grep -q moe-tune || {
    ls $B/moe-configs/*.json >/dev/null 2>&1 || { echo "### tuner exited producing NO config"; exit 1; }
    break
  }
  sleep 5
done

echo "### tuned config(s) produced:"
ls -la $B/moe-configs/

C=moe-cfg-install
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C -v $B/moe-configs:/tuned --entrypoint sleep \
  rocm-vllm-aiter-gfx90a:latest infinity >/dev/null
docker exec $C sh -c "cp -v /tuned/*.json $CFG/ && ls $CFG/ | grep -i MI210"
docker commit $C rocm-vllm-aiter-gfx90a:latest >/dev/null
docker rm -f $C >/dev/null 2>&1
echo "### tuned config baked into rocm-vllm-aiter-gfx90a:latest"

# Verify an arm actually loads it. vLLM logs "Using configuration from <path>
# for MoE layer." on a hit -- that line is the only proof, and its absence
# means the tuning bought nothing regardless of what the timings say.
echo "### will verify via: grep 'Using configuration from' in the next arm's serverlog"
