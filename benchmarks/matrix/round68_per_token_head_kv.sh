#!/usr/bin/env bash
# Round 68: the OTHER KV quantization in vLLM -- int8/int4 per-token-head.
#
# WHY THIS EXISTS. Rounds 64/66/67 measured turboquant and found it costs ~49%
# throughput for its 3.09x capacity. That number was easy to over-generalize
# into "KV quantization is expensive on this card". Round 65 says otherwise:
# llama.cpp doing the same job on the same box, on the production model, cost
#   q8_0/q8_0 -> q8_0/q4_1   76.81 -> 73.88 tok/s   -3.8%
#   q8_0/q8_0 -> q4_0/q4_1   76.81 -> 72.12 tok/s   -6.1%
# Roughly the same compression for a twentieth of the price. So the ~49% is a
# property of turboquant's implementation, not of quantizing KV.
#
# WHAT MAKES per_token_head A DIFFERENT BET. Listing CacheDType shows sixteen
# options. turboquant_* route to their own attention backend. But
# int4_per_token_head / int8_per_token_head / fp8_per_token_head are handled in
# v1/attention/backends/triton_attn.py -- the generic Triton backend, a
# DIFFERENT kernel with a different author and different performance. Round 64's
# result does not transfer to it, in either direction.
#
# THE EVIDENCE THAT TURBOQUANT'S COST IS NOT ABOUT KV BYTES. Round 64 ran both
# turboquant variants:
#   turboquant_k8v4     2.6x compression   119.45 tok/s @ conc 32
#   turboquant_4bit_nc  3.8x compression   119.16 tok/s @ conc 32
# Different byte counts, identical speed -- 0.2% apart. If the penalty came from
# moving KV, the arm moving fewer bytes would have won. It did not. The cost is
# the backend. That is precisely why swapping to a different backend is worth a
# measurement rather than an assumption.
#
# WHAT WOULD MAKE THIS A WIN. int8_per_token_head is 2x compression -- less than
# turboquant's 3.09x, and not enough to reach a 1M context (bf16 holds 902,160
# tokens, so 2x is ~1.8M and DOES clear 1M, unlike bf16). If it costs a few
# percent like llama.cpp rather than half like turboquant, it is the better
# trade at every context length that fits.
#
# WHICH BACKEND ACTUALLY GETS PICKED IS PART OF THE RESULT. The standing prefill
# win on this box comes from AITER FA (1.19-1.33x, docs/28) and turboquant gives
# it up. per_token_head routes through Triton attention, so it likely gives it
# up too -- this round PRINTS the selected backend rather than assuming, because
# assuming is how docs/48 got its mechanism wrong.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

MAXLEN=${MAXLEN:-131072}
OUT_LEN=${OUT_LEN:-64}
CTXS=${CTXS:-"8192 65536"}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

for c in $CTXS; do
    [ $(( c + OUT_LEN + 256 )) -le "$MAXLEN" ] || {
        echo "FATAL: ctx $c + output + template > max-model-len $MAXLEN"; exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 68: int8/int4 per-token-head KV ==="

ARMS="bf16 i8pth i4pth"
cleanup() { docker rm -f rd68-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

probe() {  # port outfile
    curl -s "http://127.0.0.1:$1/v1/completions" -H 'Content-Type: application/json' \
      -d '{"model":"bench","prompt":"List the first 10 prime numbers, then name the capital of Japan in one word.","max_tokens":80,"temperature":0,"seed":1234}' \
      2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["choices"][0]["text"].strip())
except Exception as e: print(f"<probe failed: {e}>")' > "$2"
}

run_arm() {  # tag kvdtype
    local tag="$1" kv="$2" extra="" t=0
    [ "$kv" != "auto" ] && extra="--kv-cache-dtype $kv"
    echo ""
    echo "=== $(date -u +%T) arm $tag (kv=$kv) ==="
    docker rm -f rd68-srv >/dev/null 2>&1
    # shellcheck disable=SC2086
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" rd68-srv 8182 --max-model-len "$MAXLEN" $extra >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8182/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd68-srv$' || {
            echo "  EXITED -- an arm that will not start IS a result, log kept:"
            docker logs rd68-srv > "$LOGS/rd68-$tag.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|assert|not support|Unsupported" "$LOGS/rd68-$tag.startup-fail" \
              | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib" | head -6
            echo "    (full log: $LOGS/rd68-$tag.startup-fail)"
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"

    # Capacity: the reason to want this at all.
    docker logs rd68-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'
    # Fast path: does AITER FA survive, or did we trade it away like turboquant?
    docker logs rd68-srv 2>&1 | grep -aoiE "Using [A-Za-z_]+ backend|attention backend[: ]+[A-Za-z_]+" \
      | sort -u | head -3 | sed 's/^/  backend: /'
    probe 8182 "$LOGS/rd68-$tag.probe"

    for c in $CTXS; do
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8182 \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$c" --random-output-len "$OUT_LEN" \
            --num-prompts 4 --max-concurrency 1 \
            --ignore-eos --seed 1234 > "$LOGS/rd68-$tag-c$c.bench" 2>&1
        local failed
        failed=$(grep -aoE "Failed requests: +[0-9]+" "$LOGS/rd68-$tag-c$c.bench" | grep -aoE "[0-9]+$" | head -1)
        printf "  ctx=%-7s " "$c"
        if [ -n "${failed:-}" ] && [ "$failed" -gt 0 ]; then
            echo "FAILED ($failed rejected) -- see $LOGS/rd68-$tag-c$c.bench"; continue
        fi
        printf "TTFT %8s ms   TPOT %7s ms\n" \
          "$(grep -aoE "Median TTFT \(ms\): +[0-9.]+" "$LOGS/rd68-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)" \
          "$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$LOGS/rd68-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)"
    done
    docker rm -f rd68-srv >/dev/null 2>&1
}

run_arm bf16  auto                 || { echo "BASELINE FAILED -- aborting"; exit 1; }
run_arm i8pth int8_per_token_head  || echo "(int8_per_token_head arm failed; see above)"
run_arm i4pth int4_per_token_head  || echo "(int4_per_token_head arm failed; see above)"

echo ""
echo "=== $(date -u +%T) round 68 done ==="
echo "########## CORRECTNESS PROBES -- read before any speed number ##########"
for a in $ARMS; do
    echo "-- rd68-$a --"; head -c 260 "$LOGS/rd68-$a.probe" 2>/dev/null || echo "  (none)"; echo
done
CTXS="$CTXS" python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CTXS = [int(c) for c in os.environ["CTXS"].split()]
ARMS = [("bf16", "bf16 (baseline)"), ("i8pth", "int8_per_token_head"),
        ("i4pth", "int4_per_token_head")]
def v(tag, c, pat):
    p = os.path.join(L, f"rd68-{tag}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    x = float(m.group(1))
    return x if x > 0 else None
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"

hdr = f"{'arm':<22}"
for c in CTXS: hdr += f"{'c'+str(c)+' TPOT':>13}{'vs bf16':>10}"
print(hdr); print("-" * len(hdr))
for tag, label in ARMS:
    row = f"{label:<22}"
    for c in CTXS:
        x, b = v(tag, c, TPOT), v("bf16", c, TPOT)
        row += f"{x:13.2f}" if x else f"{chr(45):>13}"
        row += f"{x/b:9.3f}x" if x and b else f"{chr(45):>10}"
    print(row)
print()
print("THE BAR THIS HAS TO CLEAR. turboquant_4bit_nc measured 1.508x TPOT at")
print("8192 and 2.352x at 32768 -- and got WORSE with context, not better.")
print("llama.cpp's comparable q8_0/q4_1 cost only 3.8%. If per_token_head lands")
print("near llama.cpp it is the usable option on vLLM; if it lands near")
print("turboquant, then vLLM has no affordable KV compression on gfx90a and")
print("that is the finding.")
print()
print("ALSO READ THE BACKEND LINES ABOVE. If these arms dropped off AITER FA,")
print("they gave up the 1.19-1.33x prefill win (docs/28) to get their capacity,")
print("and the TPOT column above is only half of what the trade costs.")
PY
