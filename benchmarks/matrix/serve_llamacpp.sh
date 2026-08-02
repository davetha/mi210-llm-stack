#!/usr/bin/env bash
# Launch llama-server on the MI210 pair for one GGUF arm of the matrix.
#
#   ./serve_llamacpp.sh <gguf-file-or-dir> <container-name> <port> [extra args...]
#
# Two settings here are results from earlier work in this repo, not guesses:
#
#   -DGGML_HIP_ROCWMMA_FATTN=OFF  (baked into llama-rocm714:latest)
#       Not a workaround for broken rocWMMA. ggml ALREADY reaches the matrix
#       cores on CDNA2 through fattn-mma-f16; enabling rocWMMA diverts
#       attention to the older fattn-wmma-f16 kernel, which emits the same
#       v_mfma_f32_16x16x16f16 with worse blocking and runs 18-26% slower,
#       worsening with context length. See docs/22.
#
#   -ub 2048
#       Physical batch size. Measured 15% faster on 16k prefill than the
#       default (10,524ms -> 9,160ms) on this hardware.
#
# Deliberately NOT set: -ctv q4_1. Quantizing the V cache is catastrophic here
# -- 11.5x slower on prefill in earlier measurement. KV quantization is a
# separate axis from weight quantization and would confound this matrix, so
# both caches stay at their default precision for every arm.
set -euo pipefail

MODEL="${1:?usage: serve_llamacpp.sh <gguf> <name> <port> [extra args]}"
NAME="${2:?missing container name}"
PORT="${3:?missing port}"
shift 3

# LLAMA_IMAGE swaps the engine build without touching anything else, for A/B
# against a differently-compiled llama.cpp. GGML_CUDA_FORCE_MMQ is a #ifdef
# rather than an env var, so the only way to test it is a second image
# (configs/Dockerfile.llama-forcemmq, same pinned commit, one flag changed).
IMAGE="${LLAMA_IMAGE:-llama-rocm714:latest}"
HOST_MODELS="/mnt/llm-storage"

# Accept a directory and find the first shard inside it. GGUF repos often ship
# split files (`-00001-of-00005.gguf`); llama.cpp opens the whole set given the
# FIRST shard, so sort and take the head rather than whatever glob order gives.
if [ -d "$MODEL" ]; then
    FOUND="$(find "$MODEL" -name '*.gguf' | sort | head -1)"
    [ -n "$FOUND" ] || { echo "no .gguf under $MODEL" >&2; exit 2; }
    MODEL="$FOUND"
fi

# 999 keeps every layer on GPU, which is right whenever the model fits.
# NGL=auto drops the flag so llama.cpp does its own fitting.
if [ "${NGL:-999}" = "auto" ]; then
    NGL_FLAG=""
else
    NGL_FLAG="--n-gpu-layers ${NGL:-999}"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

# Per-arm environment passthrough, unquoted on purpose so multiple "-e K=V"
# pairs split into separate docker arguments.
#
# Exists for the unified-memory arms, which need HSA_XNACK=1 and
# GGML_CUDA_ENABLE_UNIFIED_MEMORY=1. Deliberately NOT set globally: xnack+ is a
# different code-object target from xnack-, and the 242 translated AITER ASM
# kernels were built for the default, so turning XNACK on for every container
# could quietly make them unloadable on the vLLM arms.
# shellcheck disable=SC2086
# LLAMA_NO_LOCAL_GPU=1 omits the GPU devices entirely. Required for an --rpc
# arm: production gives its main llama-server NO --device flags, so all compute
# goes through the rpc-servers. Leaving them in makes llama.cpp enumerate the 2
# local ROCm devices AND the 2 RPC backends -- four backends over two physical
# cards, double-allocating them. Measured consequence: prefill 0.658x and then
# a GPU VM fault (rocr VMFaultHandler assertion) partway through long context.
if [ "${LLAMA_NO_LOCAL_GPU:-0}" = "1" ]; then
    DEV_FLAGS=""
else
    DEV_FLAGS="--device /dev/kfd --device /dev/dri"
fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" \
  $DEV_FLAGS \
  --group-add 44 --group-add 991 \
  `# numeric gids: --group-add resolves names in the CONTAINER's /etc/group` \
  ${LLAMA_EXTRA_ENV:-} \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32G \
  -v "$HOST_MODELS":/models \
  -v /var/cache/mi210-ccache:/ccache \
  -p "${PORT}:8000" \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e GPU_MAX_HW_QUEUES=4 \
  "$IMAGE" \
    --model "${MODEL/#$HOST_MODELS//models}" \
    --alias bench \
    --host 0.0.0.0 --port 8000 \
    ${NGL_FLAG} \
    `# NGL=auto omits --n-gpu-layers entirely so llama.cpp fits the model to` \
    `# free VRAM itself. Forcing 999 DISABLES that fitting -- it aborts with` \
    `# "n_gpu_layers already set by user to 999, abort" and then OOMs trying` \
    `# to place every layer. Correct for models that fit; wrong for the ~400B` \
    `# tier, which is chosen precisely because it does NOT fit and must page` \
    `# experts into system RAM.` \
    --ubatch-size 2048 \
    --flash-attn on \
    --parallel 1 \
    --no-mmap \
    --seed 1234 \
    "$@"

echo "started $NAME on :$PORT  (model $MODEL)"
