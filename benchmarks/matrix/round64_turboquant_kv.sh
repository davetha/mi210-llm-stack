#!/usr/bin/env bash
# Round 64: TurboQuant KV cache. A capacity multiplier, not a 3% tweak.
#
# WHY THIS IS DIFFERENT FROM EVERY OTHER LEAD IN docs/50-53. Those fought for
# 1-3% throughput. This changes how much context the box can hold:
#
#   turboquant_k8v4      FP8 keys + 4-bit values      2.6x   +1.17% PPL
#   turboquant_4bit_nc   4-bit MSE keys + 4-bit + NC  3.8x   +2.71% PPL
#   turboquant_k3v4_nc   3-bit keys + 4-bit values    3.5x  +10.63% PPL
#   turboquant_3bit_nc   3-bit keys + 3-bit values    4.9x  +20.59% PPL
#
# The last two cost 10-20% perplexity and are not serious options. This round
# tests the top two.
#
# WHY 4bit_nc IS THE PRIMARY CANDIDATE DESPITE k8v4 HAVING BETTER PPL. k8v4
# stores FP8 keys, and fp8 KV is exactly what died on this card in round 58b:
#     AssertionError: Unsupported dtype: torch.float8_e4m3fn
# (AMD CDNA wants e4m3FNUZ, vLLM selected e4m3FN). 4bit_nc uses 4-bit MSE keys
# instead, sidestepping fp8 entirely -- AND it compresses harder, 3.8x vs 2.6x,
# for +2.71% PPL instead of +1.17%. Both are run so the fp8 question gets an
# answer rather than an assumption; k8v4 failing to start would be a result.
#
# FEASIBILITY, CHECKED BEFORE BUILDING THIS:
#   no platform gate in turboquant_attn.py (grep for is_cuda/is_rocm is empty)
#   the decode path is Triton (triton_turboquant_decode.py), so arch-portable
#   supports_head_size() returns head_size > 0 -- no head_dim constraint
#   the one is_cuda_alike() branch is an fp8-format selector for Ampere/Ada,
#     and ROCm IS in that enum, so it evaluates rather than crashing
#
# THE COST, STATED UP FRONT. TurboQuant is its own attention backend. The AITER
# FA backend -- which delivers the measured 1.19-1.33x prefill win (docs/28) --
# does NOT list turboquant in supported_kv_cache_dtypes, so enabling this
# TRADES that prefill win for KV capacity. This round measures both sides of
# that trade rather than reporting only the good half.
#
# THE HEADLINE NUMBER IS NOT THROUGHPUT. It is "GPU KV cache size (tokens)" at
# startup. Round 58 measured the bf16 baseline at 902,176 tokens; 3.8x would be
# ~3.4M. That is the difference between holding a handful of long conversations
# and holding a lot of them.
#
# CORRECTNESS IS NOT OPTIONAL. This is a lossy numerical change with a published
# PPL cost. A capacity win with incoherent output is not a win. Every arm runs
# the same fixed-seed greedy probe and the outputs are printed for comparison.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-8192}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"8 32"}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 64: TurboQuant KV cache ==="

ARMS="base tq4bit tqk8v4"
cleanup() { for a in $ARMS; do docker rm -f "rd64-$a" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED -- full startup log kept:"
            docker logs "$name" > "$LOGS/$name.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|assert|not support|Unsupported" "$LOGS/$name.startup-fail" \
              | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib" | head -8
            echo "    (full log: $LOGS/$name.startup-fail)"
            return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

probe() {  # port outfile
    curl -s "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"bench","prompt":"List the first 10 prime numbers, then name the capital of Japan in one word.","max_tokens":80,"temperature":0,"seed":1234}' \
      2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["text"].strip())
except Exception as e: print(f"<probe failed: {e}>")' > "$2"
}

run_arm() {  # tag port kvdtype
    local name="rd64-$1" port="$2" kv="$3"
    echo ""
    echo "=== $(date -u +%T) arm $name  kv-cache-dtype=$kv ==="
    local extra=""
    [ "$kv" != "auto" ] && extra="--kv-cache-dtype $kv"
    # shellcheck disable=SC2086
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" \
        --max-model-len 32768 $extra >/dev/null
    wait_ready "$port" "$name" || { echo "  ARM $1 FAILED -- reported as a result"; return 1; }

    # THE HEADLINE: how many tokens fit in KV now.
    docker logs "$name" 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'
    docker logs "$name" 2>&1 | grep -aoiE "# GPU blocks: [0-9]+" | tail -1 | sed 's/^/  /'
    # Which attention backend actually got selected -- turboquant should NOT be
    # running AITER FA, and if it somehow is, the comparison is meaningless.
    docker logs "$name" 2>&1 | grep -aoiE "Using [A-Za-z]+ backend|attention backend[: ]+[A-Za-z_]+" | sort -u | head -3 | sed 's/^/  backend: /'
    probe "$port" "$LOGS/$name.probe"

    for c in $CONCS; do
        echo "--- $(date -u +%T) $name @ conc $c ---"
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url "http://127.0.0.1:$port" \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
            --num-prompts $(( c * 8 )) --max-concurrency "$c" \
            --ignore-eos --seed 1234 2>&1 | tee "$LOGS/$name-c$c.bench" \
          | grep -E "Output token throughput|Median TTFT|Median TPOT"
    done
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm base   8150 auto               || { echo "BASELINE FAILED -- aborting"; exit 1; }
run_arm tq4bit 8151 turboquant_4bit_nc || echo "(4bit_nc arm failed; see above)"
run_arm tqk8v4 8152 turboquant_k8v4    || echo "(k8v4 arm failed -- if this is the fp8 dtype error, that matches round 58b)"

echo ""
echo "=== $(date -u +%T) round 64 done ==="
echo "########## CORRECTNESS PROBES -- read these before any throughput number ##########"
for a in $ARMS; do
    echo "-- rd64-$a --"
    head -c 300 "$LOGS/rd64-$a.probe" 2>/dev/null || echo "  (none)"
    echo
done
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "8 32").split()]
ARMS = [("base", "bf16 (AITER FA)"), ("tq4bit", "turboquant_4bit_nc"),
        ("tqk8v4", "turboquant_k8v4")]
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"
def v(a, c, pat):
    p = os.path.join(L, f"rd64-{a}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
hdr = f"{'arm':<22}"
for c in CONCS: hdr += f"{'c'+str(c)+' tok/s':>13}{'vs base':>9}"
hdr += f"{'c'+str(CONCS[-1])+' TTFT':>12}"
print(hdr); print("-" * len(hdr))
for a, label in ARMS:
    row = f"{label:<22}"
    for c in CONCS:
        x, b = v(a, c, TPUT), v("base", c, TPUT)
        row += f"{x:13.2f}" if x else f"{'-':>13}"
        row += (f"{x/b:8.3f}x" if x and b else f"{'-':>9}")
    t = v(a, CONCS[-1], TTFT)
    row += f"{t:12.2f}" if t else f"{'-':>12}"
    print(row)
print()
print("READ THE KV CACHE SIZE LINES ABOVE FIRST -- they are the actual result.")
print("Baseline was 902,176 tokens in round 58. 4bit_nc claims 3.8x (~3.4M) and")
print("k8v4 claims 2.6x (~2.3M). If the token count did NOT grow, the flag did")
print("not take and every throughput number here is measuring the same thing.")
print()
print("THEN READ THE PROBES. This is a LOSSY change -- +2.71% PPL for 4bit_nc,")
print("+1.17% for k8v4. Outputs will not be token-identical to baseline and")
print("that is expected; what matters is that they are still correct. A")
print("capacity win with degraded output is not a win.")
print()
print("FINALLY the throughput. TurboQuant is its own backend, so these arms are")
print("NOT running AITER FA and give up its 1.19-1.33x prefill (docs/28). A")
print("throughput loss here is the PRICE of the capacity, not a bug -- the")
print("question is whether the trade is worth it for long-context serving.")
PY
