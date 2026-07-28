#!/bin/sh
# Revised queue. GPTQ-Int8 and AWQ-8bit are dropped, deliberately.
#
# GPTQ-Int8 ran 46 minutes without reaching serving, still inside
# moe_wna16_weight_loader with all 32 GB of weights already resident in VRAM.
# That is not a missing number, it IS the number: the format is impractical to
# load on this stack. And the question those two arms existed to answer --
# whether the popular 8-bit formats match W8A8 -- is already settled by profile
# rather than by timing: both route through moe_wna16, the weight-only path
# that dequantizes to bf16 and never reaches v_mfma_i32_16x16x16i8.
#
# AWQ-8bit uses the same loader and the same w8a16 scheme, so spending another
# 45+ minutes on it would confirm a mechanism already established. The GPU time
# goes to genuine gaps instead.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

arm() {
  label=$1; tier=$2; quant=$3; engine=$4; dir=$5; tp=$6; lc=$7; shift 7
  echo ""; echo "######## $(date -u +%H:%M:%S)  $label (tp=$tp) ########"
  [ -d "$dir" ] || { echo "SKIP $label: $dir missing"; return 0; }
  TP=$tp LONGCTX_TOKENS=$lc ARM_TIMEOUT=${AT:-3600} NGL=${NGL_OVERRIDE:-999} \
    READY_TIMEOUT=${RT:-9000} \
    "$BIN/run_arm.sh" "$label" "$tier" "$quant" "$engine" "$dir" "$@" \
    || echo "!! $label nonzero (recorded, continuing)"
  while docker ps --format '{{.Names}}' | grep -q '^bench-'; do sleep 15; done
}

echo "=== revised queue start $(date -u) ==="

# 1. Tier-4 with the offload mechanism suited to a sparse MoE. This is the
#    genuinely missing number: vLLM's UVA offload was stopped at 35 min of
#    warmup, so tier 4 currently has a capability result and no throughput.
AT=10800 RT=20000 NGL_OVERRIDE=auto arm glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs 1 28000 \
  --ctx-size 32768 --n-cpu-moe 20

# 2. Tier-2 cold16k, voided when JIT compilation was read as prefix caching.
arm t80-awq4 80B awq-int4 vllm-aiter $B/t80-awq 1 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch

# 3. Tier-2 AWQ-8bit, which OOMed at TP=1 with 214 MiB free. compressed-tensors,
#    so it should NOT hit the slow WNA16 loader.
RT=15000 arm t80-awq8 80B awq-int8 vllm-aiter $B/t80-awq8 2 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch

# 4. bf16 baseline last. Needs TP=2 and may still cost hours -- the loader fix
#    was refuted, so nothing has changed that.
RT=20000 arm t35-bf16 35B bf16 vllm-aiter $B/t35-bf16 2 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch

echo ""; echo "=== revised queue complete $(date -u) ==="
