#!/usr/bin/env bash
# Dense-model speculation, third attempt. Two prior failures, neither about
# speculation:
#
#   round 8: ValueError: 'draft_tensor_parallel_size' and 'tensor_parallel_size'
#            must be the same. Got 1 and 2. (My config error.)
#   round 9: ValueError: To serve at least one request with the model's max seq
#            len (131072), 27.0 GiB KV cache is needed, which is larger than
#            the available KV cache memory.
#
# Round 9's OOM is straightforward: the 70B target is 33.88 GiB per rank, the
# 3B draft adds its own weights and KV, and a 131072-token context needs 27 GiB
# of KV on top. Dropping to 65536 halves the KV requirement, which is the
# cheapest fix that keeps the comparison meaningful -- speculation is a decode
# optimisation, and decode behaviour does not depend on the context ceiling as
# long as both arms use the same one.
#
# SO THE BASELINE MOVES TOO. t70-w8a8's numbers were taken at 131072, so they
# are not comparable to a 65536 run. This script measures its own unspeculated
# baseline first, at the same ceiling, and only then the speculated arms.
#
# WHAT IS BEING TESTED. Sparse-MoE speculation lost because verifying N+1
# tokens touches ~2x the experts (docs/25). Dense verification re-reads the same
# weights it was already reading, so that penalty should not exist. Round 6
# tried this with ngram and got 0.0% acceptance -- it proved nothing. A real
# draft model is the only way to get acceptance up on a random-filler prompt.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CTX=65536
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting (ctx=$CTX for all arms) ==="

if [ ! -d "$BASE/draft-llama32-3b" ] || [ -z "$(ls -A "$BASE/draft-llama32-3b" 2>/dev/null)" ]; then
    echo "--- fetching draft model ---"
    python3 "$BIN/fetch_model.py" RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8 \
        "$BASE/draft-llama32-3b" --connections 1 --concurrent 4 \
        || { echo "!! draft fetch failed"; exit 1; }
fi

echo "--- BASELINE: dense 70B, no speculation, ctx=$CTX ---"
ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=60000 \
    "$BIN/run_arm.sh" t70-w8a8-ctx64 70B w8a8 vllm-aiter "$BASE/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len "$CTX" --no-enable-prefix-caching \
    || echo "!! baseline failed (recorded)"

for NSPEC in 1 3; do
    echo "--- dense 70B + 3B draft, num_speculative_tokens=$NSPEC, ctx=$CTX ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=60000 \
        "$BIN/run_arm.sh" "t70-w8a8-draft$NSPEC" 70B w8a8-draft vllm-aiter "$BASE/t70-w8a8" \
        --tensor-parallel-size 2 --max-model-len "$CTX" --no-enable-prefix-caching \
        --speculative-config "{\"model\":\"/models/bench-matrix/draft-llama32-3b\",\"num_speculative_tokens\":$NSPEC}" \
        || echo "!! nspec=$NSPEC failed (recorded)"
    echo "--- acceptance (0% means this arm proves NOTHING) ---"
    grep -ao "Mean acceptance length: [0-9.]*\|Avg Draft acceptance rate: [0-9.]*%" \
        "$BASE/logs/t70-w8a8-draft$NSPEC.serverlog" 2>/dev/null | tail -2 \
        || echo "   no acceptance metrics found"
done
echo "=== $(date -u +%T) round 10 complete ==="
