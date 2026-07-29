#!/usr/bin/env bash
# Third attempt at tuning the W8A8 MoE config -- this time the filename is
# verified BEFORE spending hours on it.
#
# TWO FAILED RUNS, ~6 GPU-hours, zero usable output. Both produced
#   E=128,N=768,device_name=AMD_Instinct_MI210.json
# while vLLM looks for
#   E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
#
# Both halves of that name were wrong, for separate reasons I only found by
# reading benchmark_moe.py:
#
#   N: the config name uses shard_intermediate_size // 2, which is per-RANK.
#      Qwen3-30B-A3B has moe_intermediate_size 768, so --tp-size 1 yields 768
#      and --tp-size 2 yields 384. The serving arm runs TP=2; the tuner ran
#      TP=1. Pointing --model at the right checkpoint did not fix this because
#      N never depended on the checkpoint.
#
#   dtype: benchmark_moe.py sets
#            use_int8_w8a16 = args.dtype == "int8_w8a16"
#      so the tag only appears if --dtype says so verbatim. I passed
#      --dtype auto, which silently means "no tag".
#
# So: --tp-size 2 --dtype int8_w8a16, and a GATE that runs one fast batch and
# aborts unless the produced filename is exactly what vLLM asked for. Batch 1
# took ~5 minutes in the previous runs, so the gate is cheap; the alternative
# is another six hours of tuning a file nothing reads.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CFG=$BASE/moe-configs-w8a8
WANT="E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json"
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others

rm -rf "$CFG"; mkdir -p "$CFG"

tune_one() {  # tune_one <batch-size>
    docker rm -f moe-tune >/dev/null 2>&1 || true
    timeout 5400 docker run --rm --name moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
        --tune --tp-size 2 --dtype int8_w8a16 --seed 1234 --batch-size "$1" \
        --save-dir /models/bench-matrix/moe-configs-w8a8 2>&1 | tail -3
}

echo "=== $(date -u +%T) GATE: tuning batch 1 to verify the filename ==="
tune_one 1
GOT=$(ls "$CFG" 2>/dev/null | head -1)
echo "  produced: ${GOT:-<nothing>}"
echo "  wanted  : $WANT"
if [ "$GOT" != "$WANT" ]; then
    echo "!! FILENAME MISMATCH -- aborting rather than tuning for hours again."
    echo "!! vLLM will not read '${GOT:-<nothing>}'. Fix the invocation first."
    exit 1
fi
echo "  MATCH -- proceeding with the remaining batch sizes"

for BS in 2 4 8; do
    echo "### batch-size $BS  $(date -u +%T)"
    tune_one "$BS"
done
python3 - <<'PY'
import glob, json
for f in glob.glob("/mnt/llm-storage/bench-matrix/moe-configs-w8a8/*.json"):
    d = json.load(open(f))
    print("  final:", f.split("/")[-1], sorted(int(k) for k in d if k.isdigit()))
PY

echo "=== $(date -u +%T) re-benchmarking W8A8 TP=2 ==="
echo "    baseline 43.43 t/s decode @101k | bf16 on same hardware 62.59 t/s"
VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs-w8a8 \
LONGCTX_TOKENS=110000 ARM_TIMEOUT=7200 \
    "$BIN/run_arm.sh" t35-w8a8-tp2-tuned3 35B w8a8-tuned vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    || echo "!! failed (recorded)"

echo "--- DID IT LOAD? ---"
if grep -aqiE "Using default MoE config|Config file not found" \
     "$BASE/logs/t35-w8a8-tp2-tuned3.serverlog" 2>/dev/null; then
    echo "  config STILL not found -- any delta is noise"
    grep -aoE "Config file not found at [^,]*" "$BASE/logs/t35-w8a8-tp2-tuned3.serverlog" | head -1
else
    echo "  no warning: the tuned config WAS used, so the delta is real"
fi
echo "=== $(date -u +%T) complete ==="
