#!/bin/sh
# GLM-4.6 IQ3_XS, third attempt -- with NO manual placement at all.
#
# Attempt 1: --n-gpu-layers 999 -> "n_gpu_layers already set by user to 999,
#            abort", then OOM trying to place all 148 GB on 128 GB of VRAM.
# Attempt 2: NGL=auto but kept --n-cpu-moe 20 -> "tensor_buft_overrides already
#            set by user, abort", same OOM.
#
# llama.cpp's common_fit_params() is all-or-nothing: ANY manual placement
# override -- layer count or buffer-type override -- disables the fitting pass
# entirely rather than fitting around it. So for a model that genuinely exceeds
# VRAM, the correct configuration is to specify NOTHING and let it choose.
#
# That also means --n-cpu-moe, the flag that would keep attention resident and
# page only experts, cannot be combined with auto-fit. If the automatic split
# turns out to be poor, the alternative is an explicit -ngl AND -n-cpu-moe pair
# chosen by hand, not a mixture.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin
while ps -eo cmd | grep -q "[/]run_queue2.sh"; do sleep 60; done
while docker ps --format '{{.Names}}' | grep -q '^bench-'; do sleep 30; done
echo "### GLM-4.6 IQ3_XS, auto-fit with no overrides $(date -u)"
TP=1 NGL=auto ARM_TIMEOUT=10800 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs \
  --ctx-size 32768 \
  || echo "!! glm46-iq3xs failed a third time (recorded)"
echo "### resulting split ==="
grep -oiE "offloaded [0-9]+/[0-9]+ layers to GPU|CPU buffer size = *[0-9.]+ MiB|ROCm[01] buffer size = *[0-9.]+ MiB" \
  $B/logs/glm46-iq3xs.serverlog 2>/dev/null | head -5
echo "### done $(date -u)"
