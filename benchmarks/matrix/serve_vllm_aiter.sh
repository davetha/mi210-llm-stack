#!/usr/bin/env bash
# Serve with the gfx90a AITER ASM stack enabled. Same as serve_vllm.sh but on
# the patched image, with the AITER attention flags on.
set -euo pipefail
MODEL_DIR="${1:?}"; NAME="${2:?}"; PORT="${3:?}"; shift 3
# Overridable so an arm can A/B a patched build against the default without a
# second copy of this script. Hardcoding it meant round2's E9 would have quietly
# benchmarked the stock image while reporting a patched result -- the same shape
# of error as crediting AITER for a run that never loaded its kernels.
IMAGE="${VLLM_IMAGE:-rocm-vllm-aiter-gfx90a:latest}"
H=/mnt/llm-storage
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
  -v "$H":/models -v /var/cache/mi210-ccache:/ccache \
  -p "${PORT}:8000" \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 -e GPU_MAX_HW_QUEUES=4 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 \
  -e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e VLLM_TUNED_CONFIG_FOLDER=${VLLM_TUNED_CONFIG_FOLDER:-} \
  -e AITER_LOG_LEVEL=info -e VLLM_PREFER_AITER_FA=${VLLM_PREFER_AITER_FA:-0} \
  -e VLLM_PLUGINS= \
  -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=100G -e CCACHE_DEPEND=1 \
  -e PYTHONHASHSEED=0 \
  `# Per-arm environment passthrough, unquoted on purpose so several "-e K=V"` \
  `# pairs split into separate docker arguments. Without this, an arm that sets` \
  `# e.g. VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 would run with the flag` \
  `# silently absent and report the result as though it had been applied --` \
  `# the same class of error as VLLM_IMAGE being hardcoded above.` \
  ${VLLM_EXTRA_ENV:-} \
  --entrypoint vllm "$IMAGE" serve "${MODEL_DIR/#$H//models}" \
    --served-model-name bench --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size "${TP:-1}" --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --seed 1234 "$@"
echo "started $NAME on :$PORT"
