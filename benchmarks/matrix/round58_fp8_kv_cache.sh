#!/usr/bin/env bash
# Round 58: fp8 KV cache. Halves the term that dominates long-context decode.
#
# THE CASE. docs/39 item 1c decomposed a decode step on this box: weights
# 36.4 GB/rank and KV 9.8 GB/rank, i.e. KV is ~21% of the bytes at 60k context
# and grows with sequence length while weights do not. Decode is
# memory-bandwidth bound. Halving KV bytes attacks that directly, and unlike
# every quantization lead in docs/50 it does not touch the weights, so it does
# not interact with the W8A8 GEMM path at all.
#
# A CLAIM I MADE AND HAVE TO WITHDRAW. Earlier notes in this project asserted
# that quantized KV also short-circuits AITER's ASM paged-attention batch
# heuristic, giving ASM at batch 1 for free. It does not, via vLLM:
#
#   aiter/ops/attention.py:_should_use_asm_kernel has two early returns --
#       if high_precision == 2:                  return True   # fp8 kv only
#       if kv_cache_tensor_dtype == torch.int8:  return True
#   but vLLM never passes high_precision (grep is empty; AITER defaults it to
#   1), and `int8` is not a member of vLLM's CacheDType at all -- the list is
#   auto/float16/bfloat16/fp8*/turboquant*. Neither branch is reachable.
#
# So this round tests the BANDWIDTH benefit only. Round 57 tests the batch-1
# ASM question directly with an explicit force hook, which is a cleaner
# instrument than hoping a dtype side-effect reaches it.
#
# WHAT COULD MAKE THIS FAIL, AND WHY THAT IS STILL A RESULT. gfx90a has no FP8
# MFMA (docs/49, proven at the assembler). fp8 KV is storage plus convert, not
# fp8 compute, so it SHOULD be fine -- but "should" is why we measure. If the
# server refuses to start with --kv-cache-dtype fp8, that is a clean answer:
# fp8 KV is unavailable on CDNA2 and the whole line closes.
#
# CORRECTNESS IS NOT OPTIONAL HERE. Unlike every other round in this project,
# this one changes NUMERICS -- fp8 e4m3 has ~2 decimal digits of mantissa. A
# throughput win with degraded output is not a win. Both arms run the same
# fixed-seed greedy prompt and the outputs are compared.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-8192}      # longer than round 57: KV is the point here
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"1 8 32"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 58: fp8 KV cache vs bf16 ==="

cleanup() { docker rm -f rd58-bf16 rd58-fp8 >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED during startup -- last lines:"
            docker logs "$name" 2>&1 | tail -25
            return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

# Greedy, fixed seed, same prompt both arms. temperature=0 is set explicitly
# because vllm bench serve no longer defaults to greedy.
probe_correctness() {  # port outfile
    curl -s "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"bench","prompt":"List the first 12 prime numbers, then explain in one sentence why 1 is not prime.","max_tokens":96,"temperature":0,"seed":1234}' \
      2>/dev/null | python3 -c 'import json,sys
try:
    d=json.load(sys.stdin); print(d["choices"][0]["text"].strip())
except Exception as e:
    print(f"<probe failed: {e}>")' > "$2"
    echo "  correctness probe -> $2"
}

bench_at() {  # port conc outfile
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$1" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts $(( $2 * 8 )) --max-concurrency "$2" \
        --ignore-eos --seed 1234 2>&1 | tee "$3" \
      | grep -E "Output token throughput|Median TPOT|Median TTFT"
}

run_arm() {  # name port kvdtype
    local name="$1" port="$2" kv="$3"
    echo ""
    echo "=== $(date -u +%T) arm $name  --kv-cache-dtype $kv ==="
    local extra=""
    [ "$kv" != "auto" ] && extra="--kv-cache-dtype $kv"
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" \
        --max-model-len 32768 $extra >/dev/null
    if ! wait_ready "$port" "$name"; then
        echo "  ARM $name FAILED TO START."
        # DO NOT WRITE THE CONCLUSION HERE. The first version of this script
        # printed "fp8 KV is unavailable on gfx90a" at this point -- a guess
        # hardcoded before any evidence existed. It happened to be right, but
        # the script had no way to know: wait_ready tails only 25 lines, which
        # catches the API server's wrapper ("Engine core initialization failed.
        # See root cause above.") and not the worker's actual exception, and
        # the cleanup trap then deletes the container. Round 58b re-ran it with
        # no trap and full log capture to get the real cause:
        #     AssertionError: Unsupported dtype: torch.float8_e4m3fn
        # -- a dtype-FLAVOUR rejection (e4m3fn vs AMD's e4m3fnuz), not the
        # "no FP8 MFMA" reason one would assume. Capture, then conclude.
        echo "  Root cause NOT captured by this harness -- run round58b to get it."
        docker logs "$name" > "$LOGS/$name.startup-fail" 2>&1 || true
        echo "  full startup log -> $LOGS/$name.startup-fail"
        return 1
    fi
    # KV cache blocks reported at startup -- direct evidence the footprint shrank
    docker logs "$name" 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens|# GPU blocks: [0-9]+" | tail -2 | sed 's/^/  /'
    probe_correctness "$port" "$LOGS/$name.probe"
    for c in $CONCS; do
        echo "--- $(date -u +%T) $name @ concurrency $c ---"
        bench_at "$port" "$c" "$LOGS/$name-c$c.bench"
    done
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm rd58-bf16 8112 auto || { echo "BASELINE FAILED -- aborting"; exit 1; }
run_arm rd58-fp8  8113 fp8  || echo "(fp8 arm failed; see above -- reported as a result, not an error)"

echo ""
echo "=== $(date -u +%T) round 58 done ==="
echo "--- correctness probes (these must be comparable, not identical) ---"
for a in rd58-bf16 rd58-fp8; do
    echo "== $a =="; head -c 400 "$LOGS/$a.probe" 2>/dev/null || echo "(none)"; echo
done
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "1 8 32").split()]
def val(arm, c, pat):
    p = os.path.join(L, f"{arm}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
print(f"{'conc':>5}{'bf16 tok/s':>12}{'fp8 tok/s':>11}{'fp8/bf16':>10}"
      f"{'bf16 TPOT':>11}{'fp8 TPOT':>10}{'TPOT r':>8}")
print("-" * 67)
any_row = False
for c in CONCS:
    b, f = val("rd58-bf16", c, TPUT), val("rd58-fp8", c, TPUT)
    bt, ft = val("rd58-bf16", c, TPOT), val("rd58-fp8", c, TPOT)
    if b is None and f is None: continue
    any_row = True
    r  = f"{f/b:9.3f}x" if b and f else f"{'-':>10}"
    rt = f"{ft/bt:7.3f}x" if bt and ft else f"{'-':>8}"
    print(f"{c:>5}"
          f"{(f'{b:12.2f}' if b else f'{chr(45):>12}')}"
          f"{(f'{f:11.2f}' if f else f'{chr(45):>11}')}{r}"
          f"{(f'{bt:11.2f}' if bt else f'{chr(45):>11}')}"
          f"{(f'{ft:10.2f}' if ft else f'{chr(45):>10}')}{rt}")
if not any_row:
    print("  no paired results -- check whether the fp8 arm started at all")
print()
print("READING THIS. The prize is KV BANDWIDTH, so the effect should grow with")
print("context and with concurrency (more sequences = more KV resident). A flat")
print("ratio across concurrency suggests the win is not coming from KV at all.")
print("Also check the reported GPU KV cache size above: fp8 should roughly")
print("DOUBLE the cached token capacity. If it did not, the flag did not take")
print("and the throughput numbers are measuring nothing.")
print()
print("AND CHECK THE PROBES. fp8 e4m3 carries ~2 decimal digits of mantissa.")
print("The two outputs will not be token-identical and that is expected; what")
print("matters is that the fp8 answer is still correct and coherent. A")
print("throughput win with degraded output is not a win.")
PY
