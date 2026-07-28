#!/usr/bin/env bash
# Why speculative decoding LOST, and the two things that could change it.
#
# THE MECHANISM, from measurement rather than guesswork.
#
# Acceptance was excellent: 84-89% per position, mean acceptance length
# 1.85-1.89 of a possible 2. The draft is not the problem. So the loss must be
# in verification cost -- and speculation's core assumption is exactly that
# verifying N+1 tokens costs about what decoding 1 costs, because you are
# memory-bound reading all the weights anyway.
#
# That assumption holds for DENSE models and breaks for SPARSE MoE.
# Qwen3-Next-80B routes 10 of 512 experts per token, and adjacent tokens route
# near-independently, so:
#
#   positions   distinct experts   weight traffic
#       1            10.0             1.00x
#       2            19.8             1.98x
#       3            29.4             2.94x
#
# Predicted net at 2 positions: 1.86 (acceptance) / 1.98 (traffic) = 0.94x.
# Observed: 43.09 / 51.34 = 0.84x. Residual 1.12x is the MTP head's own
# forward pass. The model accounts for the result.
#
# TWO TESTS FOLLOW, and they are falsifiable in opposite directions.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

echo "=== $(date -u +%T) waiting for earlier work ==="
while ps -eo cmd | grep -qE "[b]in/round[0-9]_|[f]etch_model.py" \
   || docker ps --format '{{.Names}}' | grep -q '^bench-'; do
    sleep 120
done
echo "=== $(date -u +%T) starting ==="

# ---------------------------------------------------------------------------
# D1  Speculation on a DENSE model, where the batching-is-free assumption holds.
#
# Llama-3.3-70B W8A8 is dense: verifying 2 tokens reads the same weights ONCE,
# so the 1.98x traffic penalty that sank the MoE does not exist. If the
# sparsity explanation is right, speculation should HELP here.
#
# This is the decisive test. Same engine, same flags, ngram so no draft model
# is needed. Baseline: t70-w8a8 measured 843 t/s prefill @15k and 5.2 t/s
# decode @101k (from the noaiter A/B run).
#
# If D1 also loses, the sparsity story is wrong and the cost is somewhere else
# entirely -- which would be worth knowing before any more speculation work.
# ---------------------------------------------------------------------------
echo "--- D1: ngram speculation on the DENSE 70B ---"
ARM_TIMEOUT=7200 READY_TIMEOUT=3600 LONGCTX_TOKENS=110000 \
    "$BIN/run_arm.sh" t70-w8a8-ngram 70B w8a8-ngram vllm-aiter "$BASE/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    || echo "!! D1 failed (recorded)"

# ---------------------------------------------------------------------------
# D2  Tune fused_moe at the batch sizes speculation actually runs at.
#
# The MI210 config produced so far covers batch size 1 ONLY -- the batch-64
# pass did not land. Speculation runs the expert GEMM at batch 2-3, which is
# therefore entirely untuned, and vLLM ships no MI210 configs at all.
#
# HONEST CEILING: even perfectly tuned, the arithmetic above says the best
# case at 2 positions is ~0.94x, i.e. still a loss on this model. Tuning can
# recover the gap between 0.84 and 0.94; it cannot make sparse-MoE speculation
# positive. The reason to run it is that the SAME shapes are what makes W8A8
# lose decode twice over (docs/24), and that is worth much more than
# speculation is.
# ---------------------------------------------------------------------------
echo "--- D2: tune fused_moe at batch 2, 4, 8 (speculation + small-batch decode) ---"
mkdir -p "$BASE/moe-configs"
for BS in 2 4 8; do
    echo "### tuning batch-size $BS $(date -u +%T)"
    docker rm -f moe-tune >/dev/null 2>&1 || true
    timeout 5400 docker run --rm --name moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-awq \
        --tune --tp-size 1 --dtype auto --seed 1234 --batch-size "$BS" \
        --save-dir /models/bench-matrix/moe-configs 2>&1 | tail -4
done
echo "### tuned batch sizes now present:"
python3 - <<'PY'
import json, glob
for f in glob.glob("/mnt/llm-storage/bench-matrix/moe-configs/*.json"):
    d = json.load(open(f))
    print("  ", f.split("/")[-1], sorted(int(k) for k in d if k.isdigit()))
PY

echo "=== $(date -u +%T) round 6 complete ==="
