#!/usr/bin/env bash
# Fourth attempt. This one captures the FULL tuner output, because the reason
# the third failed is still unknown -- I piped it through `tail -3` and the only
# thing that survived was a tqdm shutdown traceback, which is exit noise, not
# the error.
#
# WHAT IS ALREADY RULED OUT. The arguments are right:
#   --dtype int8_w8a16   is in choices=["auto","fp8_w8a8","int8_w8a16","int4_w4a16"]
#   --tp-size 2          matches the serving arm (and is the parser default;
#                        the earlier runs passed 1 explicitly, which is why they
#                        produced N=768 instead of N=384)
# So v3 asked for the right thing and something inside the tuner refused.
#
# HISTORY, so this is not attempted a fifth time blind:
#   v1  tuned t35-awq, --tp-size 1, --dtype auto -> E=128,N=768,...json
#       Overwrote itself per batch; only batch 4 survived. ~3 GPU-hours wasted.
#   v2  tuned t35-w8a8, --tp-size 1, --dtype auto -> same wrong filename.
#       Batches 16 and 32 both hit the 90-min timeout. ~3 more hours wasted.
#   v3  correct args, gated on filename -> produced NOTHING, aborted in 5 min.
#       The gate is the only reason that cost minutes instead of hours.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CFG=$BASE/moe-configs-w8a8
WANT="E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json"
RAW=$BASE/logs/tuner_raw.log
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others

rm -rf "$CFG"; mkdir -p "$CFG"
echo "=== $(date -u +%T) DIAGNOSTIC: batch 1, full output to $RAW ==="
docker rm -f moe-tune >/dev/null 2>&1 || true
timeout 5400 docker run --rm --name moe-tune \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
    --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
    -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
    -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
    --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
    /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
    --tune --tp-size 2 --dtype int8_w8a16 --seed 1234 --batch-size 1 \
    --save-dir /models/bench-matrix/moe-configs-w8a8 > "$RAW" 2>&1
rc=$?
echo "  tuner exit=$rc"
echo "--- last 30 non-progress lines ---"
grep -avE "it/s|^\s*$" "$RAW" | tail -30
echo "--- any exception? ---"
grep -aiE "Traceback|Error|Exception|assert|not supported|KeyError|ValueError" "$RAW" \
    | grep -avE "tqdm|sys.meta_path" | tail -10

GOT=$(ls "$CFG" 2>/dev/null | head -1)
echo "  produced: ${GOT:-<nothing>}"
echo "  wanted  : $WANT"
if [ "$GOT" != "$WANT" ]; then
    echo "!! still wrong -- stopping. Full tuner output is in $RAW"
    exit 1
fi

echo "  MATCH -- tuning remaining batch sizes"
for BS in 2 4 8; do
    echo "### batch-size $BS  $(date -u +%T)"
    docker rm -f moe-tune >/dev/null 2>&1 || true
    timeout 5400 docker run --rm --name moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
        --tune --tp-size 2 --dtype int8_w8a16 --seed 1234 --batch-size "$BS" \
        --save-dir /models/bench-matrix/moe-configs-w8a8 2>&1 | grep -avE "it/s" | tail -3
done

echo "=== $(date -u +%T) re-benchmarking W8A8 TP=2 ==="
echo "    baseline 43.43 t/s decode @101k | bf16 same hardware 62.59 t/s"
VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs-w8a8 \
LONGCTX_TOKENS=110000 ARM_TIMEOUT=7200 \
    "$BIN/run_arm.sh" t35-w8a8-tp2-tuned3 35B w8a8-tuned vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    || echo "!! failed (recorded)"
if grep -aqiE "Using default MoE config|Config file not found" \
     "$BASE/logs/t35-w8a8-tp2-tuned3.serverlog" 2>/dev/null; then
    echo "  config STILL not found -- any delta is noise"
else
    echo "  no warning: the tuned config WAS used, so the delta is real"
fi
echo "=== $(date -u +%T) complete ==="
