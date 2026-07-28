#!/bin/sh
# GLM-4.6 IQ3_XS, re-run with llama.cpp's own memory fitting enabled.
#
# The first attempt OOMed at load:
#   W common_fit_params: failed to fit params to free device memory:
#     n_gpu_layers already set by user to 999, abort
#   E allocating 70673.58 MiB on device 1: cudaMalloc failed: out of memory
#
# serve_llamacpp.sh hardcoded --n-gpu-layers 999, which DISABLES llama.cpp's
# fitting pass -- it aborts rather than choosing a split, then tries to place
# all ~148 GB on 128 GB of VRAM. Right for every model that fits; wrong for the
# one tier picked precisely because it does not.
#
# NGL=auto omits the flag so llama.cpp fits the model itself, and --n-cpu-moe
# 20 keeps attention on GPU while paging experts to the 499 GB of system RAM.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin
while ps -eo cmd | grep -qE "[s]weep_glm.sh|[s]weep_t235|[b]f16_last|[r]erun_t80_cold|[r]etry_t80_awq8|[s]weep_int8_formats|[t]une_moe_targeted"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
echo "### GLM-4.6 IQ3_XS re-run, NGL=auto $(date -u)"
TP=1 NGL=auto LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs \
  --ctx-size 32768 --n-cpu-moe 20 \
  || echo "!! glm46-iq3xs failed again (recorded)"
echo "### how did it split? ==="
grep -oiE "offloaded [0-9]+/[0-9]+ layers to GPU|CPU buffer size = *[0-9.]+ MiB|ROCm[01] buffer size = *[0-9.]+ MiB" \
  $B/logs/glm46-iq3xs.serverlog 2>/dev/null | head -5
echo "### done $(date -u)"
