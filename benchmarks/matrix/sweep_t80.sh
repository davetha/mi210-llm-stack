#!/bin/sh
# Tier 2: Qwen3-Next-80B-A3B-Thinking (Apache-2.0, 80B total / 3B active).
#
# This is the tier where 256k context is actually interesting rather than
# merely large. The model is a 3:1 hybrid -- three Gated DeltaNet (linear
# attention) layers per full-attention layer -- so only 12 of 48 layers hold a
# KV cache. A full-attention 80B at 256k would not fit; this one should.
#
# w8a8 INT8 is the format that won tier 1 by a wide margin (3.20 s vs 4.43 s
# for the next best), but no INT8 W8A8 checkpoint exists for this model, so the
# closest available is AWQ 8-bit. Worth stating plainly: the tier-2 comparison
# is not the same set of formats as tier 1, because the checkpoints do not
# exist, not because they were skipped.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin

# Wait for every earlier GPU job: the tier-1 chain, then the bf16 baseline.
while ps -eo cmd | grep -qE "[f]ollowup_chain|[b]f16_last"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

echo "### tier-2 sweep start $(date -u)"

# AWQ 4-bit, ~46 GB -> fits one card at TP=1, which also avoids the ~3h
# per-expert TP=2 load cost measured in tier 1.
TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=9000 \
  $BIN/run_arm.sh t80-awq4 80B awq-int4 vllm-aiter $B/t80-awq \
  --max-model-len 131072 --safetensors-load-strategy=prefetch \
  || echo "!! t80-awq4 failed (recorded)"
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

# AWQ 8-bit, ~54 GB. Tight on a 64 GB card once KV and graph capture are
# accounted for; if it OOMs that is a reportable limit, not a bug to hide.
TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=9000 \
  $BIN/run_arm.sh t80-awq8 80B awq-int8 vllm-aiter $B/t80-awq8 \
  --max-model-len 65536 --safetensors-load-strategy=prefetch \
  || echo "!! t80-awq8 failed (recorded)"
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done

# GGUF on llama.cpp, at the real 256k. llama.cpp has no equivalent of vLLM's
# graph-capture attention gate, so this is the arm that can actually answer the
# 256k question on this hardware.
TP=1 LONGCTX_TOKENS=250000 READY_TIMEOUT=9000 \
  $BIN/run_arm.sh t80-q4km 80B q4_k_m llamacpp $B/t80-gguf-q4km \
  --ctx-size 262144 \
  || echo "!! t80-q4km failed (recorded)"

echo "### tier-2 sweep done $(date -u)"
