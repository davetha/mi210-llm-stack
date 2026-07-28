#!/usr/bin/env bash
# Speculative decoding: the one idea worth taking from the ROCmFPX look.
#
# WHY THIS AND NOT THE FORMATS. ROCmFPX targets RDNA (gfx1151/1100/1030/1200);
# its FP3/4/6/8 formats have no matrix hardware to land on here -- docs/27
# establishes by assembler that gfx90a's matrix cores accept only
# {fp64, fp32, bf16, fp16, int8}. But multi-token prediction is a SCHEDULING
# technique, not a kernel, so it is ISA-independent and aims squarely at this
# box's weakest axis: decode. 16.0 t/s at 241k, 43.4 t/s for W8A8 at 101k.
#
# A CHECKPOINT FINDING WORTH THE WHOLE EXERCISE. Qwen3-Next ships MTP weights,
# and quantizers do not agree about keeping them:
#
#   RedHatAI/Qwen3-Next-80B-...-quantized.w8a8   148,311 tensors,     0 MTP
#   cyankiwi/Qwen3-Next-80B-...-AWQ-8bit         226,472 tensors, 4,625 MTP
#   cyankiwi/Qwen3-Next-80B-...-AWQ-4bit         226,472 tensors, 4,625 MTP
#
# The W8A8 build -- which wins prefill at 7,253 vs 6,679 t/s and loads faster
# -- had its MTP head stripped. Neither repo name says so. If MTP pays off,
# the slower-prefill checkpoint may be the better one overall, which would be
# a genuine reversal.
#
# vLLM wiring verified in this build: config/speculative.py maps
# model_type "qwen3_next" -> "qwen3_next_mtp", and registry.py has
# "Qwen3NextMTP". Our checkpoint declares model_type qwen3_next.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

echo "=== $(date -u +%T) waiting for earlier work ==="
while ps -eo cmd | grep -qE "[b]in/round2.sh|[b]in/round2_followup.sh|[b]in/round3_prefill.sh" \
   || docker ps --format '{{.Names}}' | grep -q '^bench-'; do
    sleep 120
done
echo "=== $(date -u +%T) starting speculative sweep ==="

# ---------------------------------------------------------------------------
# S1  Native MTP on the 80B that still has its MTP head.
#     Baseline: t80-awq8 measured 51.3 t/s decode at 101k, 6,679 t/s prefill.
#     Speculative decoding trades prefill/compute for decode, so watch BOTH --
#     a decode win paid for by a prefill loss may not be worth taking.
# ---------------------------------------------------------------------------
for NSPEC in 1 2; do
    echo "--- S1: 80B AWQ-8bit + MTP, num_speculative_tokens=$NSPEC ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=110000 \
        "$BIN/run_arm.sh" "t80-awq8-mtp$NSPEC" 80B awq-int8-mtp vllm-aiter "$BASE/t80-awq8" \
        --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
        --speculative-config "{\"method\":\"qwen3_next_mtp\",\"model\":\"/models/bench-matrix/t80-awq8\",\"num_speculative_tokens\":$NSPEC}" \
        || echo "!! MTP nspec=$NSPEC failed (recorded)"
done

# ---------------------------------------------------------------------------
# S2  n-gram speculation on the recommended format. No draft model, no extra
#     VRAM, no download -- it speculates from repetition in the prompt itself.
#     Baseline: t35-w8a8-tp2 measured 43.4 t/s decode at 101k.
#
#     Expected to help most where output echoes input (summarisation, code
#     edits, RAG). The benchmark prompt is random filler plus a counting
#     instruction, which is close to a WORST case for n-gram -- so read a null
#     result here as "not helped by this workload", not "does not work".
# ---------------------------------------------------------------------------
echo "--- S2: 30B W8A8 TP=2 + ngram speculation ---"
ARM_TIMEOUT=7200 READY_TIMEOUT=2400 LONGCTX_TOKENS=110000 \
    "$BIN/run_arm.sh" t35-w8a8-tp2-ngram 35B w8a8-ngram vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    || echo "!! ngram failed (recorded)"

echo "=== $(date -u +%T) speculative sweep complete ==="
echo "baselines: t80-awq8 51.3 t/s decode @101k | t35-w8a8-tp2 43.4 t/s @101k"
