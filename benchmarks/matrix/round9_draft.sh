#!/usr/bin/env bash
# Draft-model speculation on the dense 70B -- corrected invocation.
#
# Round 8 failed on my own config error, not on anything interesting:
#
#   ValueError: Currently, 'draft_tensor_parallel_size' and
#   'tensor_parallel_size' must be the same. Got 1 and 2.
#
# I set draft_tensor_parallel_size=1 thinking a 3B draft had no need of two
# cards. vLLM requires them equal. Dropping the field lets it default to the
# target's TP.
#
# STILL THE OPEN QUESTION. Sparse-MoE speculation loses because verifying N+1
# tokens touches ~2x the experts (docs/25). Dense verification re-reads the
# same weights it was already reading, so the penalty should not exist. Round 6
# tried to test this with ngram and got 0.0% acceptance -- speculation never
# fired, so it proved nothing. A real draft model is the only way to get
# acceptance up on a random-filler prompt.
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
    echo "--- fetching draft model ---"
    python3 "$BIN/fetch_model.py" RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 \
        "$BASE/draft-llama32-3b" --connections 1 --concurrent 4 \
        || { echo "!! draft fetch failed"; exit 1; }
fi

for NSPEC in 1 3; do
    echo "--- dense 70B + 3B draft, num_speculative_tokens=$NSPEC ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=110000 \
        "$BIN/run_arm.sh" "t70-w8a8-draft$NSPEC" 70B w8a8-draft vllm-aiter "$BASE/t70-w8a8" \
        --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
        --speculative-config "{\"model\":\"/models/bench-matrix/draft-llama32-3b\",\"num_speculative_tokens\":$NSPEC}" \
        || echo "!! nspec=$NSPEC failed (recorded)"
    echo "--- acceptance for nspec=$NSPEC (0% means this arm proves NOTHING) ---"
    grep -ao "Mean acceptance length: [0-9.]*\|Avg Draft acceptance rate: [0-9.]*%" \
        "$BASE/logs/t70-w8a8-draft$NSPEC.serverlog" 2>/dev/null | tail -2 \
        || echo "   no acceptance metrics found"
done
echo "=== $(date -u +%T) round 9 complete ==="
