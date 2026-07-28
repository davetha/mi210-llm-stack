#!/bin/sh
# Tier 4: GLM-4.6 (MIT, 357B total / ~32B active).
#
# This is the tier the task frames around RAM offload, and it is the only tier
# where the model genuinely does not fit: IQ3_XS is ~148 GB against 128 GB of
# VRAM, so ~20-30 GB of experts must live in the 499 GB of system RAM.
#
# llama.cpp carries this rather than vLLM for two measured reasons, not
# preference:
#   - vLLM bakes the Triton attention fallback into its CUDA graph whenever
#     max_model_len exceeds 128k on gfx9, costing 10x decode (docs/23). At this
#     model size a reduced context is already forced, but the gate is one more
#     thing to fight.
#   - llama.cpp handles MoE expert offload well on this box: earlier work here
#     sustained ~21 tok/s on 230B-class models with -ncmoe, and it just decoded
#     at 29 tok/s with 230k context resident.
#
# -ncmoe 20 is a STARTING POINT, not a tuned value. If it OOMs, that is a
# reportable limit and the arm records it rather than silently reducing scope.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

while ps -eo cmd | grep -qE "[f]ollowup_chain|[s]weep_t80|[s]weep_t235"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

if ! grep -q "OK .*glm-gguf-iq3xs" $B/dl-big.status 2>/dev/null; then
  echo "### tier-4 GGUF SKIPPED: download not verified complete"
else
  echo "### tier-4 GLM-4.6 IQ3_XS start $(date -u)"
  # Modest context: KV for a 357B full-attention model is expensive, and the
  # point of this arm is whether a 357B runs at all with offload, not its
  # long-context behaviour.
  TP=1 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
    $BIN/run_arm.sh glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs \
    --ctx-size 32768 --n-cpu-moe 20 \
    || echo "!! glm46-iq3xs failed (recorded)"
fi
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

# The AWQ arm on vLLM is the comparison point, and is expected to be the harder
# of the two: 189 GB needs ~70 GB of vLLM CPU offload, and the AWQ MoE path on
# ROCm is the WNA16 fallback rather than Marlin. Run it anyway -- "vLLM cannot
# practically serve this tier on an MI210" is a result worth having measured
# rather than asserted.
if grep -q "OK .*glm-awq" $B/dl-big.status 2>/dev/null; then
  echo "### tier-4 GLM-4.6 AWQ on vLLM start $(date -u)"
  TP=2 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
    $BIN/run_arm.sh glm46-awq 400B awq-int4 vllm-aiter $B/glm-awq \
    --max-model-len 32768 --safetensors-load-strategy=prefetch \
    --cpu-offload-gb 70 --gpu-memory-utilization 0.95 \
    || echo "!! glm46-awq failed (recorded)"
else
  echo "### tier-4 AWQ SKIPPED: download not verified complete"
fi

echo "### tier-4 done $(date -u)"
