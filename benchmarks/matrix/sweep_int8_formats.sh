#!/bin/sh
# Do the POPULAR 8-bit formats reach W8A8 speed on CDNA2?
#
# W8A8 won tier 1 outright (3.20 s TTFT, 4,739 tok/s) but compressed-tensors
# W8A8 is not a widely published format. GPTQ-Int8 and AWQ-8bit are, so if they
# perform the same the practical recommendation changes completely.
#
# HYPOTHESIS: they will NOT match it, and the reason is the activations.
#   W8A8       int8 weights AND int8 activations -> int8 x int8 GEMM, which can
#              use gfx90a v_mfma_i32_16x16x16i8 (181 TOPS, equal to bf16 peak)
#   GPTQ-Int8  int8 weights, bf16 activations (w8a16) -> dequantize to bf16 and
#   AWQ-8bit   run a bf16 GEMM. Memory saving, but no INT8 arithmetic.
#
# If that holds, the rule for this hardware is not "pick 8-bit" but "pick 8-bit
# ACTIVATIONS" -- which is a much more useful thing to tell someone choosing a
# checkpoint, and is falsifiable here.
#
# Both are ~32 GB so they fit one card at TP=1.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

while ps -eo cmd | grep -qE "[s]weep_t235|[s]weep_glm|[b]f16_last|[r]erun_t80_cold|[r]etry_t80_awq8|[t]une_moe_targeted"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

for pair in "t35-gptq8:gptq-int8" "t35-awq8:awq-int8"; do
  d=$(echo $pair | cut -d: -f1); q=$(echo $pair | cut -d: -f2)
  [ -d "$B/$d" ] || { echo "### $d missing, skipping"; continue; }
  echo "### $d ($q) $(date -u)"
  TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=9000 \
    $BIN/run_arm.sh $d 35B $q vllm-aiter $B/$d \
    --max-model-len 131072 --safetensors-load-strategy=prefetch \
    || echo "!! $d failed (recorded)"
  # Which MoE path did it actually take? This is the whole point of the run:
  # 'Int8 MoE backend' means w8a8 and the INT8 matrix path; anything else means
  # it dequantized to bf16 and the 8 bits bought only memory.
  echo "### MoE path for $d:"
  grep -oiE "Using [A-Z0-9]+ Int8 MoE backend|Using [A-Z0-9]+ .*MoE backend|Moe WNA16" \
    $B/logs/$d.serverlog 2>/dev/null | sort -u | head -3
  while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
done
echo "### 8-bit format comparison done $(date -u)"
