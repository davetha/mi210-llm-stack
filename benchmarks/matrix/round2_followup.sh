#!/usr/bin/env bash
# Re-runs the arms that failed in round 2 for fixable reasons, after round 2 ends.
#
# Chains on a process match anchored to the exact argv, not a loose `pgrep -f`.
# The first version of round2_chain.sh waited on `pgrep -f "fetch_round2.sh"`
# and deadlocked for eight hours on a shell whose command line merely CONTAINED
# that string.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

echo "=== $(date -u +%T) waiting for round 2 to finish ==="
while pgrep -f "bin/round2.sh" >/dev/null 2>&1 \
   || docker ps --format '{{.Names}}' | grep -q '^bench-'; do
    sleep 120
done
echo "=== $(date -u +%T) round 2 done ==="

# ---------------------------------------------------------------------------
# E7-redo  --n-cpu-moe sweep, against the IQ3_XS build already on disk.
#
# The original arms pointed at $BASE/glm46-q4km, which does not exist, so all
# three exited instantly on "failed to open GGUF file". Nothing about the
# experiment was wrong -- only the path.
#
# IQ3_XS is also the right choice: the auto-fit run that anchors this sweep at
# N~0 was IQ3_XS (12.83 t/s short, 8.51 t/s at 25.8k, 135.57 GB of 139 GB
# resident on GPU). Sweeping a different quant would confound placement with
# precision, which is the same mistake as the TP=2 rows.
#
# docs/26 predicts decode degrades roughly LINEARLY in pinned-layer count. If
# these three points are not close to linear, that model is wrong and the doc
# must say so.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%T) E7-redo: --n-cpu-moe sweep on IQ3_XS ==="
for N in 30 45 60; do
    echo "--- n-cpu-moe=$N (anchor: auto-fit ~0 CPU layers -> 8.51 t/s @25.8k) ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm46-iq3xs-ncmoe$N" 400B iq3_xs llamacpp "$BASE/glm-gguf-iq3xs" \
        --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$N"
done

# ---------------------------------------------------------------------------
# E1-redo  80B W8A8 decode, which the original arm could not measure.
#
# The prompt asked for exactly one word, so decode rate was accidentally
# dependent on the model being a reasoning model: Qwen3-Next-80B-Thinking
# padded to ~170 tokens and produced a number, while the Instruct build of the
# same architecture answered in 3 and got decode_tps=null. bench_matrix.py now
# asks for a count to 60 when max_tokens is large enough to measure decode, so
# output length is fixed by the instruction rather than by verbosity.
#
# This fills the decode half of the W8A8-vs-W8A16 comparison at tier 2 --
# prefill already showed W8A8 ahead, 7,253 against 6,679 t/s.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%T) E1-redo: 80B W8A8 decode with the fixed prompt ==="
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t80-w8a8-decode 80B w8a8 vllm-aiter "$BASE/t80-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E5-redo  MoE tuning, which never ran.
#
# The original failed with "bin/tune_moe_targeted.sh: No such file or
# directory" -- the script lives at the bench-matrix root and was never copied
# into bin/. Same class of failure as E7's missing model path: the experiments
# were fine, the plumbing was not. Both are now deployed.
#
# This matters more than it did before round 2. The TP=2 control showed W8A8
# decoding at 43.4 t/s against bf16's 62.6 on identical hardware -- backwards,
# since INT8 halves weight traffic and should WIN a bandwidth-bound decode.
# The leading explanation is that the Triton INT8 MoE kernel is untuned at
# batch-1 shapes, and vLLM ships tuned fused_moe configs for MI300X/MI325X and
# none for MI210. This is the experiment that tests it.
#
# Bounded to batch sizes 1 and 64: the full sweep grew 704 -> 2,660 -> 4,990 ->
# 7,810 candidates and would eat a day per tier.
# ---------------------------------------------------------------------------
echo "=== $(date -u +%T) E5-redo: targeted MoE tuning ==="
"$BIN/tune_moe_targeted.sh" || echo "E5-redo did not complete -- continuing"

echo "=== $(date -u +%T) followup complete ==="
python3 "$BIN/summarize_results.py" "$BASE/results" 2>/dev/null || true
