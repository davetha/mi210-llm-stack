#!/usr/bin/env bash
# Round 65: the llama.cpp KV quant configs -- the PRODUCTION path.
#
# WHY THIS MATTERS MORE THAN ROUND 64. docs/47 established that production runs
# llama.cpp/GGUF, not vLLM. Every vLLM result in docs/50-53 is deployed nowhere.
# This round tests the thing actually serving traffic.
#
# THE OBSERVATION. llama-swap-config.yaml runs coder-next with
#     -ctk q8_0 -ctv q8_0
# while EVERY other model on the box uses
#     -ctk q8_0 -ctv q4_1
# (launch-235b, launch-mimo, launch-minimax, launch-winner-256k). Production is
# on the most conservative KV setting in the fleet, and it is not obvious
# whether that is deliberate or copy-paste.
#
# docs/03 bits-per-value table:
#     q8_0   8.5 bpw   1.9x vs fp16
#     q4_1   5.0 bpw   3.2x        <- the doc calls this "the V cache sweet spot"
#     q4_0   4.5 bpw   3.6x
#     KIVI2  3.0 bpw   5.3x        <- implemented in davetha/llama.cpp-mi210,
#                                     but NOT in any image on this box
#
# ARMS. All the same binary and model; only -ctk/-ctv move.
#     q8q8   current production
#     q8q4   what the rest of the fleet runs
#     q4q4   more aggressive, still mainline
# KIVI2 is deliberately NOT an arm: it is not in llama-rocm714:latest, and
# building it is a separate piece of work rather than a flag flip.
#
# THE HEADLINE IS KV CAPACITY, not throughput. Quantized KV should be roughly
# throughput-neutral and buy context. If throughput MOVES a lot, that is worth
# knowing too -- q4_1 dequant is not free.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=llama-rocm714:latest
MODEL=/mnt/llm-storage/coder-next-q4

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
# llama.cpp -m takes the .gguf FILE, not the directory. The first attempt
# passed the directory and every arm died with "failed to load model" -- which
# the harness then reported as "likely an unsupported cache type", a cause it
# had not checked. Resolve the actual file, and fail here rather than three
# arms later.
if [ -d "$MODEL" ]; then
    MODEL=$(find "$MODEL" -name '*.gguf' 2>/dev/null | head -1)
fi
[ -f "$MODEL" ] || { echo "FATAL: no .gguf found (got '$MODEL')"; exit 1; }
echo "model: $MODEL"

# -fa on IS MANDATORY, not optional. llama.cpp refuses quantized V outright:
#     llama_init_from_model: failed to initialize the context:
#         quantized V cache was requested, but this requires Flash Attention
# The first attempt at this round omitted it and both q4_1 arms died, which
# also meant the round was not testing the production configuration at all --
# llama-swap-config.yaml passes `-fa -ctv q8_0`, and launch-235b.sh /
# launch-mimo.sh pass `-fa -ctv q4_1`. So q4_1 is AVAILABLE to production
# today; running q8_0 is a choice rather than a technical limit.
CTX=${CTX:-32768}
READY_TIMEOUT=${READY_TIMEOUT:-900}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 65: llama.cpp KV quant configs (production path) ==="

cleanup() { docker rm -f rd65-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag ctk ctv
    local tag="$1" ctk="$2" ctv="$3" t=0
    echo ""
    echo "=== $(date -u +%T) arm $tag  -ctk $ctk -ctv $ctv ==="
    docker rm -f rd65-srv >/dev/null 2>&1
    docker run -d --name rd65-srv --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v /mnt/llm-storage:/mnt/llm-storage \
      --entrypoint /src/build/bin/llama-server "$IMG" \
      -m "$MODEL" --host 0.0.0.0 --port 8160 \
      --ctx-size "$CTX" -ngl 999 -fa on -ctk "$ctk" -ctv "$ctv" \
      --no-warmup >/dev/null 2>&1

    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8160/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd65-srv$' || {
            echo "  ARM $tag EXITED. Actual error below -- do NOT assume it is the"
            echo "  cache type; the first run of this round died on a bad model path"
            echo "  while claiming an unsupported cache type."
            docker logs rd65-srv 2>&1 | grep -aiE "error|unsupported|unknown|invalid|failed" | head -6
            return 1; }
        sleep 5; t=$((t+5))
    done
    [ $t -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"

    # THE HEADLINE: how much VRAM the KV cache takes for the same context.
    # Cast wide: the exact wording varies by llama.cpp version, and the first
    # attempt matched nothing at all.
    docker logs rd65-srv 2>&1 | grep -aiE "kv.?cache|KV self|KV buffer|K *\(|V *\(" \
      | grep -aiE "MiB|GiB|size" | head -6 | sed 's/^/    /'

    # Throughput + a correctness probe, since q4 is lossy.
    curl -s http://127.0.0.1:8160/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Write a Python function that returns the first 10 prime numbers. Reply with code only.","n_predict":96,"temperature":0,"seed":1234}' \
      2>/dev/null | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin)
    print("    probe:", d.get("content","")[:160].replace(chr(10)," "))
    tm = d.get("timings", {})
    pps = tm.get("predicted_per_second", 0) or 0
    rps = tm.get("prompt_per_second", 0) or 0
    print("    decode %.2f tok/s, prompt %.2f tok/s" % (pps, rps))
except Exception as e: print("    <probe failed>", e)'
    docker rm -f rd65-srv >/dev/null 2>&1
}

run_arm q8q8 q8_0 q8_0 || echo "(q8q8 failed)"
run_arm q8q4 q8_0 q4_1 || echo "(q8q4 failed)"
run_arm q4q4 q4_0 q4_1 || echo "(q4q4 failed)"

echo ""
echo "=== $(date -u +%T) round 65 done ==="
echo "Compare the KV self size lines: q8_0/q8_0 is what production runs today,"
echo "q8_0/q4_1 is what every other model on this box runs, and q4_0/q4_1 is"
echo "more aggressive but still mainline. Per docs/03 the bits-per-value are"
echo "8.5 / 5.0 / 4.5, so V alone moving q8_0 -> q4_1 should cut the V cache"
echo "by ~41%. Read the probes: q4 is lossy and a capacity win with broken"
echo "output is not a win."
