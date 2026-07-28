#!/bin/sh
# ONE serial runner for all remaining arms.
#
# This replaces eight independent scripts that each waited on a list of the
# others. That scheme deadlocked twice: sweep_t235 waited on bf16_last while
# bf16_last waited on sweep_t235, and because every other job waited on one or
# both, nothing could start. The failure is silent -- every script alive, GPUs
# idle, no error -- and looks exactly like "the queue is slow".
#
# A single script cannot deadlock against itself. Order is by value, so if this
# is interrupted the results that change conclusions already exist.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

arm() {
  label=$1; tier=$2; quant=$3; engine=$4; dir=$5; tp=$6; lc=$7; shift 7
  echo ""
  echo "######## $(date -u +%H:%M:%S)  $label  (tp=$tp longctx=$lc) ########"
  [ -d "$dir" ] || { echo "SKIP $label: $dir missing"; return 0; }
  TP=$tp LONGCTX_TOKENS=$lc ARM_TIMEOUT=${AT:-3600} NGL=${NGL_OVERRIDE:-999} \
    READY_TIMEOUT=${RT:-12000} \
    "$BIN/run_arm.sh" "$label" "$tier" "$quant" "$engine" "$dir" "$@" \
    || echo "!! $label returned nonzero (recorded, continuing)"
  while docker ps --format '{{.Names}}' | grep -q '^bench-'; do sleep 15; done
}

echo "=== serial queue start $(date -u) ==="

# 1. Validates configs/fast_moe_expert_load.py against its own 810 s/shard
#    baseline. Highest value: it either confirms a real fix or retires it.
RT=20000 arm t235-gptq4 235B gptq-int4 vllm-aiter $B/t235-gptq4 2 28000 \
  --max-model-len 32768 --safetensors-load-strategy=prefetch --gpu-memory-utilization 0.95

# 2. Do the POPULAR 8-bit formats reach W8A8? Prediction: no, because they are
#    w8a16 -- int8 weights, bf16 activations -- so they dequantize to a bf16
#    GEMM and never touch v_mfma_i32_16x16x16i8.
for d in t35-gptq8:gptq-int8 t35-awq8:awq-int8; do
  arm "${d%%:*}" 35B "${d##*:}" vllm-aiter "$B/${d%%:*}" 1 110000 \
    --max-model-len 131072 --safetensors-load-strategy=prefetch
done

# 3. Tier-4 with the offload mechanism that suits a sparse MoE. vLLM's UVA
#    offload was stopped after a 35-minute warmup at 28k.
AT=10800 RT=20000 NGL_OVERRIDE=auto arm glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs 1 28000 \
  --ctx-size 32768 --n-cpu-moe 20

# 4. Tier-2 gaps: cold16k was voided by the JIT-as-cache false positive, and
#    awq8 OOMed at TP=1 with 214 MiB free.
arm t80-awq4 80B awq-int4 vllm-aiter $B/t80-awq 1 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch
RT=15000 arm t80-awq8 80B awq-int8 vllm-aiter $B/t80-awq8 2 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch

# 5. Bounded MoE tuning, then the bf16 baseline last -- it needs TP=2 and may
#    still cost hours in the per-expert loader if the fix does not apply to it.
echo ""; echo "######## $(date -u +%H:%M:%S)  targeted MoE tuning ########"
sh $B/tune_moe_targeted.sh 2>&1 | tail -20

RT=20000 arm t35-bf16 35B bf16 vllm-aiter $B/t35-bf16 2 110000 \
  --max-model-len 131072 --safetensors-load-strategy=prefetch

echo ""; echo "=== serial queue complete $(date -u) ==="
ls -la $B/results/ | tail -5
