#!/usr/bin/env bash
# Speculation on a dense model, with a draft that can actually accept.
#
# WHY ROUND 6's D1 DID NOT ANSWER THIS. D1 used ngram on the dense 70B because
# it needs no draft model. Its acceptance was 0.0% -- mean acceptance length
# 1.00 -- so speculation never fired, and the arm cannot say whether
# speculation HELPS a dense model. That was a design error: the benchmark
# prompt is random filler, so prompt-lookup has nothing to find, and I had
# already predicted exactly that for the 80B ngram arm without carrying the
# prediction over to this one.
#
# WHAT D1 DID PRODUCE, which is worth keeping. Comparing wasted speculation
# across architectures:
#
#   dense 70B    acceptance  0.0%          prefill 544 -> 536   -1.5%
#   sparse 30B   acceptance  7.2-30.1%     decode 43.40 -> 32.76  -24.5%
#
# The MoE accepted MORE and lost FAR more. That is the expert-traffic penalty
# showing up directly: MoE verification reads ~2x the experts whether or not
# the drafted tokens are accepted, while dense verification re-reads the same
# weights it was reading anyway. Consistent with the mechanism in docs/25.
#
# THIS ROUND. Llama-3.2-3B-Instruct-quantized.w8a8 as a draft for
# Llama-3.3-70B-Instruct-quantized.w8a8 -- same tokenizer family, both W8A8,
# 4.4 GB so it costs almost no VRAM. If dense speculation works anywhere on
# this box, it works here.
#
# Baseline: t70-w8a8 prefill 843 @15k / 544 @101k. Decode was never recorded
# for that arm (it predates the prompt fix), so t70-w8a8-ngram's 6.38 t/s at
# 0% acceptance is the closest decode reference -- effectively the unspeculated
# rate plus ~0 accepted tokens.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting ==="

if [ ! -d "$BASE/draft-llama32-3b" ] || [ -z "$(ls -A "$BASE/draft-llama32-3b" 2>/dev/null)" ]; then
    echo "--- fetching draft model (4.4 GB) ---"
    python3 "$BIN/fetch_model.py" RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 \
        "$BASE/draft-llama32-3b" --connections 1 --concurrent 4 \
        || { echo "!! draft fetch failed"; exit 1; }
fi

for NSPEC in 1 3; do
    echo "--- dense 70B + 3B draft, num_speculative_tokens=$NSPEC ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=110000 \
        "$BIN/run_arm.sh" "t70-w8a8-draft$NSPEC" 70B w8a8-draft vllm-aiter "$BASE/t70-w8a8" \
        --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
        --speculative-config "{\"model\":\"/models/bench-matrix/draft-llama32-3b\",\"num_speculative_tokens\":$NSPEC,\"draft_tensor_parallel_size\":1}" \
        || echo "!! nspec=$NSPEC failed (recorded)"
    echo "--- acceptance for nspec=$NSPEC (0% means this arm proves nothing) ---"
    grep -ao "Mean acceptance length: [0-9.]*\|Avg Draft acceptance rate: [0-9.]*%" \
        "$BASE/logs/t70-w8a8-draft$NSPEC.serverlog" 2>/dev/null | tail -2
done

echo "=== $(date -u +%T) round 8 complete ==="
