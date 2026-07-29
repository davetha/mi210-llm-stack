#!/usr/bin/env bash
# Tune the MoE config the W8A8 arm ACTUALLY asks for.
#
# The previous run tuned the wrong model and produced a file vLLM never looked
# at. Its own "did the config load" check caught it:
#
#   WARNING Using default MoE config. Performance might be sub-optimal!
#   Config file not found at .../E=128,N=384,...,dtype=int8_w8a16.json
#
# We produced   E=128,N=768,device_name=AMD_Instinct_MI210.json
# vLLM wanted   E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
#
# Two errors in one filename. N=768 is the AWQ model's intermediate size --
# tune_moe_targeted.sh hardcodes --model t35-awq -- while the W8A8 model is
# N=384. And the dtype tag was absent because the tuner was not told the
# quantization scheme. So the "tuned" re-benchmark (43.43 -> 44.36 t/s) was
# measuring nothing but noise, and without that grep it would have been written
# down as "tuning does not help".
#
# Batches 16 and 32 also both hit the 90-minute timeout (rc=124) and
# contributed nothing, so three hours bought zero additional batch sizes.
# Capped at 8 here: 1, 2 and 4 completed in 5, 6 and 35 minutes respectively,
# and speculation plus single-stream decode live at 1-4 anyway.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CFG=$BASE/moe-configs-w8a8
BATCHES="${BATCHES:-1 2 4 8}"
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting W8A8 MoE tuning ==="
mkdir -p "$CFG"

for BS in $BATCHES; do
    echo "### batch-size $BS  $(date -u +%T)"
    docker rm -f moe-tune >/dev/null 2>&1 || true
    timeout 5400 docker run --rm --name moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
        --tune --tp-size 1 --dtype auto --seed 1234 --batch-size "$BS" \
        --save-dir /models/bench-matrix/moe-configs-w8a8 2>&1 | tail -3
    echo "### files now: $(ls "$CFG" 2>/dev/null | tr '\n' ' ')"
done

echo "=== $(date -u +%T) re-benchmarking W8A8 TP=2 ==="
echo "    baseline: 43.43 t/s decode @101k, 7,278 t/s prefill @15k"
echo "    bf16 on the same hardware: 62.59 t/s decode"
VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs-w8a8 \
LONGCTX_TOKENS=110000 ARM_TIMEOUT=7200 \
    "$BIN/run_arm.sh" t35-w8a8-tp2-tuned2 35B w8a8-tuned vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    || echo "!! failed (recorded)"

echo "--- DID IT LOAD? a 'Using default MoE config' warning means it did not ---"
grep -aiE "Using default MoE config|Config file not found" \
    "$BASE/logs/t35-w8a8-tp2-tuned2.serverlog" 2>/dev/null | head -2 \
    && echo "  ^^ config STILL not found -- any delta is noise" \
    || echo "  no warning: the tuned config was used"
echo "=== $(date -u +%T) complete ==="
