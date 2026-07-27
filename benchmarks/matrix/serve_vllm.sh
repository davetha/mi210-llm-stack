#!/usr/bin/env bash
# Launch vLLM on the MI210 pair for one arm of the quantization matrix.
#
#   ./serve_vllm.sh <model-dir> <container-name> <port> [extra vllm args...]
#
# Every knob that could silently invalidate a benchmark is set here rather than
# left to defaults, and the reasons are recorded inline. The two that matter
# most:
#
#   --no-enable-prefix-caching
#       vLLM V1 enables automatic prefix caching by DEFAULT. With it on, the
#       second and later repetitions of a cold-prompt benchmark return a TTFT
#       near zero, because the prefix was already in the block cache. That
#       reads as a spectacular result and is entirely an artifact. The harness
#       also defends against this with UUID-seeded prompts and a --verify-cold
#       assertion, but defence in depth is warranted: this single flag is the
#       difference between a real number and a fictional one.
#
#   NCCL_P2P_DISABLE=1
#       The two MI210s in this box have NO xGMI bridge. Left enabled, RCCL
#       probes for a peer-to-peer path that does not exist and stalls during
#       collective setup for tensor-parallel runs.
#
# Note on device indices: HIP and rocm-smi enumerate these two cards in
# OPPOSITE orders on this host (rocm-smi GPU[0] is HIP_VISIBLE_DEVICES=1). This
# script uses both cards, so it does not matter here -- but it matters a great
# deal for any single-card run, and it is why no HIP_VISIBLE_DEVICES is set
# below by default.
set -euo pipefail

MODEL_DIR="${1:?usage: serve_vllm.sh <model-dir> <name> <port> [extra args]}"
NAME="${2:?missing container name}"
PORT="${3:?missing port}"
shift 3

IMAGE="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"
HOST_MODELS="/mnt/llm-storage"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --device /dev/kfd --device /dev/dri \
  --group-add 44 --group-add 991 \
  `# NUMERIC gids, not names. --group-add resolves names against the` \
  `# CONTAINER's /etc/group, not the host's. This host has video=44 and` \
  `# render=991, but the rocm/vllm image has no "render" entry, so` \
  `# --group-add render fails with "Unable to find group render". Without` \
  `# membership of the group owning /dev/kfd and /dev/dri/renderD*, the` \
  `# container starts but sees no GPUs. Verify with: getent group video render` \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32G \
  -v "$HOST_MODELS":/models \
  -v /var/cache/mi210-ccache:/ccache \
  -p "${PORT}:8000" \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e GPU_MAX_HW_QUEUES=4 \
  -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=100G -e CCACHE_DEPEND=1 \
  -e PYTHONHASHSEED=0 \
  -e VLLM_LOGGING_LEVEL=INFO \
  --entrypoint vllm \
  "$IMAGE" serve "${MODEL_DIR/#$HOST_MODELS//models}" \
    --served-model-name bench \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --seed 1234 \
    "$@"

echo "started $NAME on :$PORT  (model $MODEL_DIR)"
echo "follow with: docker logs -f $NAME"
