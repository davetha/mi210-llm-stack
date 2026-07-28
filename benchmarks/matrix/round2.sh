#!/usr/bin/env bash
# Round 2 of the quantization matrix: the experiments docs/26 and docs/27
# predict but do not measure.
#
# Run on the MI210 host:
#   cd /mnt/llm-storage/bench-matrix && ./bin/round2.sh 2>&1 | tee logs/round2.log
#
# SERIAL BY CONSTRUCTION. Round 1 lost hours to two circular deadlocks built out
# of eight scripts that waited on each other. Every arm here runs in sequence in
# one process; a failing arm records status=FAILED and the next one starts. There
# is nothing to coordinate and nothing that can wait on anything else.
#
# Each experiment names the claim it tests, so a result that contradicts the doc
# is recognisable as such rather than being read as noise.
set -uo pipefail

BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODELS=$BASE/models
mkdir -p "$MODELS" "$BASE/results" "$BASE/logs"

fetch() {  # fetch <dest-dir> <hf-repo>
    local dest="$1" repo="$2"
    [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ] && { echo "have $repo"; return 0; }
    echo "### fetching $repo"
    # --connections 1: the Xet CDN signs per-byte-range URLs, so aria2 --split
    # produces holey files that look complete. Round 1 lost a 45 GB download to
    # this before the cause was found.
    python3 "$BIN/fetch_model.py" --repo "$repo" --dest "$dest" --connections 1 --concurrent 8
}

banner() { echo; echo "###############################################################"; echo "# $*"; echo "###############################################################"; }

# ---------------------------------------------------------------------------
# E1  Does W8A8 beat W8A16 on the SAME 80B architecture?
#
# docs/26 predicts yes. The tier-2 measurement used
# cyankiwi/...-AWQ-8bit, which despite the name is compressed-tensors
# pack-quantized with input_activations=null -- weight-only, so it never reached
# v_mfma_i32_16x16x16i8. This repo is int-quantized with 8-bit dynamic per-token
# activations, and is the same model family at the same TP.
#
# Confounder to respect: the W8A16 arm was Thinking, this is Instruct. Different
# post-training, same architecture. Report it as such.
# ---------------------------------------------------------------------------
banner "E1  80B W8A8 (tests docs/26 prediction: W8A8 > W8A16 on Qwen3-Next)"
fetch "$MODELS/t80-w8a8" RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t80-w8a8 80B w8a8 vllm-aiter "$MODELS/t80-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E2  A model that hits BOTH fast paths at once.
#
# Nothing in round 1 does. The 30B W8A8 got INT8 MFMA + hd128 ASM but is a MoE
# with a small active set; Qwen3-Next got INT8 but head_dim=256 so no ASM at all
# (docs/25 item 7). Llama-3.3-70B W8A8 is dense, head_dim=128, 131k native:
# INT8 GEMM *and* AITER ASM attention, simultaneously, verified from config.json.
#
# Verify from the serverlog, not from the timing:
#   grep -c 'fwd_hd128_bf16.*\.co' logs/t70-w8a8.serverlog   -> must be > 0
# ---------------------------------------------------------------------------
banner "E2  Llama-3.3-70B W8A8, dense hd128 (tests docs/26 'sweet spot')"
fetch "$MODELS/t70-w8a8" RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t70-w8a8 70B w8a8 vllm-aiter "$MODELS/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E3  Is the ASM win additive with W8A8, or does it overlap?
#
# The +12.8% in docs/24 was measured on AWQ-Int4. The claim in docs/27 -- that
# the ASM kernels are bf16 attention and therefore quantization-independent --
# predicts the same delta on a W8A8 model. Same model, same flags, AITER off then
# on. This is the decisive A/B and it has never been run on a W8A8 arm.
# ---------------------------------------------------------------------------
banner "E3  AITER A/B on W8A8 (tests docs/27: ASM is quantization-independent)"
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t70-w8a8-noaiter 70B w8a8 vllm "$MODELS/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E4  W4A8-int8: which kernel does vLLM actually pick on gfx90a?
#
# docs/27 audits the mixed-precision registry statically and concludes every
# ROCm-reachable kernel rejects int8 activations, so this must either fail to
# find a kernel or silently fall back to a w4a16 path. A 1.5B model makes that a
# two-minute question instead of an hour.
#
# The result is the LOG, not the throughput. Read it:
#   grep -iE 'kernel|w4a8|marlin|triton|cannot implement' logs/probe-w4a8.serverlog
# A clean start is NOT confirmation the int8 path ran -- that is the exact
# mistake docs/24 documents for ROCM_AITER_FA. Confirm which kernel was chosen.
# ---------------------------------------------------------------------------
banner "E4  W4A8-int8 kernel probe (tests docs/27: no ROCm kernel takes int8 acts)"
fetch "$MODELS/probe-w4a8" alishafique/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a8int8-llmcompressor
READY_TIMEOUT=600 "$BIN/run_arm.sh" probe-w4a8 1.5B w4a8-int8 vllm-aiter "$MODELS/probe-w4a8" \
    --max-model-len 32768 --no-enable-prefix-caching
echo "--- kernel selection for the W4A8 probe ---"
grep -iE "kernel|w4a8|marlin|triton|machete|cannot implement|not supported" \
    "$BASE/logs/probe-w4a8.serverlog" 2>/dev/null | head -40

# ---------------------------------------------------------------------------
# E5  MoE tuning, with the variable vLLM actually reads.
#
# Round 1 set MOE_CONFIG_DIR, which vLLM ignores; the correct name is
# VLLM_TUNED_CONFIG_FOLDER. That mistake would have produced a confident
# "tuning does not help" from a run where the config was never loaded.
#
# Bounded to batch sizes 1 and 64. The full sweep is unbounded -- it grew
# 704 -> 2,660 -> 4,990 -> 7,810 candidates and eats a day per tier.
# ---------------------------------------------------------------------------
banner "E5  MoE tuning with VLLM_TUNED_CONFIG_FOLDER (round 1 used the wrong var)"
"$BIN/tune_moe_targeted.sh" || echo "E5 tuning did not complete -- continuing"

# ---------------------------------------------------------------------------
# E6  bf16 baseline. Still the only missing cell in tier 1.
#
# TP=2 is mandatory: 61 GB will not share a 64 GB card with a KV cache. Expect
# ~697 s/shard in _load_w13 (docs/25 item 1) -- roughly 3 hours before it serves.
# That is why it is here and not first.
# ---------------------------------------------------------------------------
banner "E6  30B bf16 baseline at TP=2 (slow: ~3h load, run last of the vLLM arms)"
fetch "$MODELS/t35-bf16" Qwen/Qwen3-30B-A3B-Thinking-2507
READY_TIMEOUT=14400 LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t35-bf16 35B bf16 vllm-aiter "$MODELS/t35-bf16" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E7  GLM-4.6 GGUF, with the -ngl / --n-cpu-moe interaction handled correctly.
#
# Two round-1 attempts failed on this and the mechanism is now understood:
# common_fit_params() is all-or-nothing, so setting EITHER -ngl or --n-cpu-moe
# disables auto-fit for the whole model. They must be set together.
#
# The sweep also validates the bandwidth model in docs/26, which predicts decode
# is roughly LINEAR in the number of CPU-pinned layers. Three points is enough to
# see whether that holds. If it does not, the model is wrong and the doc must say
# so -- this is a real test, not a formality.
# ---------------------------------------------------------------------------
banner "E7  GLM-4.6 GGUF + --n-cpu-moe sweep (tests docs/26 bandwidth model)"
fetch "$MODELS/glm46-q4km" bartowski/zai-org_GLM-4.6-GGUF
for N in 30 45 60; do
    echo "--- n-cpu-moe=$N ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm46-q4km-ncmoe$N" 400B q4_k_m llamacpp "$MODELS/glm46-q4km" \
        --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$N"
done

banner "round 2 complete"
python3 "$BIN/summarize_results.py" "$BASE/results"
