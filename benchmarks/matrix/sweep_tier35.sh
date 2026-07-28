#!/usr/bin/env bash
# Tier-1 sweep: Qwen3-30B-A3B-Thinking-2507 across every quantization that
# has a real repo, on both engines.
#
#   nohup ./sweep_tier35.sh > /mnt/llm-storage/bench-matrix/sweep-t35.log 2>&1 &
#
# Runs unattended. Each arm serves, probes correctness, benchmarks, tears down.
# An arm that fails is recorded as a FAILED result rather than skipped -- see
# run_arm.sh for why that matters.
#
# Two deliberate choices:
#
# LONGCTX IS RUN AT 110k, NOT 256k, for the vLLM arms.
#   vLLM's use_rocm_custom_paged_attention() requires max_seq_len <= 128*1024
#   on gfx9 (platforms/rocm.py:328). Above it, paged decode falls back to a
#   Triton kernel for which aiter ships no gfx90a configs at all, and decode
#   collapsed to 0.7 tok/s in the first 256k run. Benchmarking every vLLM arm
#   above the ceiling would measure that one fallback repeatedly instead of
#   measuring the quantizations. The 256k question is asked separately, and
#   llama.cpp -- which has no such gate -- carries it.
#
# TP=1 WHEREVER THE WEIGHTS FIT ON ONE CARD.
#   vLLM's per-expert MoE loader narrows each expert tensor per TP rank,
#   producing a non-contiguous view, then copies per expert. On this model
#   (128 experts x 48 layers) TP=2 measured ~697 s per shard. The w8a8 arm is
#   run at BOTH TP=1 and TP=2 purely to settle whether TP is really the cause:
#   same model, same bytes, only the sharding differs. Without that pair the
#   45x gap seen so far is confounded with a 4x difference in model size.
set -uo pipefail

BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
export READY_TIMEOUT=${READY_TIMEOUT:-14400}   # BF16 at TP=2 is genuinely hours

arm() {  # arm <label> <quant> <engine> <dir> <tp> [extra...]
    local label="$1" quant="$2" engine="$3" dir="$4" tp="$5"; shift 5
    # vLLM stays under the 128k paged-decode ceiling; llama.cpp has no such
    # gate, so it gets the full 256k the task actually asks about.
    local lc=110000
    [ "$engine" = "llamacpp" ] && lc=262144
    echo ""
    echo "############ $(date -u +%H:%M:%S)  $label  (tp=$tp longctx=$lc) ############"
    TP="$tp" LONGCTX_TOKENS="$lc" \
        "$BIN/run_arm.sh" "$label" 35B "$quant" "$engine" "$dir" "$@" \
        || echo "!! $label returned nonzero (recorded, continuing)"
}

# --- vLLM arms that fit on one card -------------------------------------
arm t35-w8a8-tp1  w8a8  vllm "$BASE/t35-w8a8"  1 --max-model-len 131072 --safetensors-load-strategy=prefetch
arm t35-fp8-tp1   fp8   vllm "$BASE/t35-fp8"   1 --max-model-len 131072 --safetensors-load-strategy=prefetch

# --- the controlled TP comparison ---------------------------------------
# Same weights as t35-w8a8-tp1, only sharding differs. Load time is the
# measurement of interest here, not throughput.
arm t35-w8a8-tp2  w8a8  vllm "$BASE/t35-w8a8"  2 --max-model-len 131072 --safetensors-load-strategy=prefetch

# --- the BF16 baseline, which has no choice but TP=2 ---------------------
# 61 GB of weights cannot share a 64 GB card with a KV cache. Expected to take
# hours to load; that cost is itself a reportable result.
arm t35-bf16-tp2  bf16  vllm "$BASE/t35-bf16"  2 --max-model-len 131072 --safetensors-load-strategy=prefetch

# --- llama.cpp / GGUF, including the real 256k attempt ------------------
arm t35-q4km      q4_k_m llamacpp "$BASE/t35-gguf-q4km" 1 --ctx-size 131072
arm t35-q8        q8_0   llamacpp "$BASE/t35-gguf-q8"   1 --ctx-size 131072

echo ""
echo "############ sweep complete $(date -u) ############"
ls -la "$BASE/results/"
