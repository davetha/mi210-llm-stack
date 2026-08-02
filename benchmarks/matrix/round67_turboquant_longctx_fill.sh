#!/usr/bin/env bash
# Round 67: finish the turboquant-vs-context curve that round 66 left with a
# hole in it.
#
# WHAT ROUND 66 ACTUALLY MEASURED, AND WHAT IT GOT WRONG.
#   context   bf16 TPOT   tq4 TPOT   ratio
#      8192      16.90      25.48   1.508x
#     32768      19.05      44.81   2.352x
#    131072          -          -   (all 4 requests failed, BOTH arms)
#
# The 131072 row is a script bug, not a result. The server ran with
# --max-model-len 131072 and the benchmark asked for --random-input-len 131072
# --random-output-len 64, so prompt+generation exceeded the window and every
# request was rejected. The openai-chat backend also wraps the prompt in a chat
# template, costing a few more tokens on top. This round reserves headroom
# instead of spending the entire window on the prompt.
#
# WHY THE MISSING ROW MATTERS MORE THAN THE TWO THAT LANDED. The pre-registered
# question was whether turboquant's cost falls as context grows -- KV bytes
# scale with context while a fixed overhead does not, so a falling ratio would
# mean a crossover exists somewhere. The two points say the ratio RISES: 1.508x
# to 2.352x. Going 8k->32k, bf16 TPOT grows 2.15ms while turboquant grows
# 19.33ms, a ~9x steeper slope. That is the signature of a decode kernel that is
# less efficient PER KV BYTE, not one paying a fixed startup cost. If that slope
# holds, long context is where turboquant is worst, which is exactly backwards
# from the reason to want it.
#
# Two points define a line and cannot distinguish a line from a curve, so this
# round adds 65536 and ~131072 to see whether the slope is linear, flattening,
# or accelerating. The deployment question -- is turboquant usable at the long
# contexts that are its only justification -- is decided at the last point.
#
# WHY BOTH ARMS RERUN 8192. Round 66's numbers came from a different process and
# a different server start. Re-measuring one shared point per arm makes the new
# curve self-consistent and shows whether the rig repeats; a large disagreement
# on 8192 would mean the curve is noise and should be read as such.
#
# THE CAPACITY SIDE IS NOT IN QUESTION. Round 66 confirmed it directly from the
# server logs: 902,160 tokens bf16 vs 2,790,992 turboquant, a 3.09x that matches
# the claimed 3.8x closely enough. Nothing here re-litigates that. This round is
# only about what the capacity costs.
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
# Reserve room for the generation AND the chat template wrapper. Round 66 spent
# the whole window on the prompt and got zero completed requests.
CTXS=${CTXS:-"8192 65536 130000"}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

for c in $CTXS; do
    [ $(( c + OUT_LEN + 256 )) -le "$MAXLEN" ] || {
        echo "FATAL: ctx $c + $OUT_LEN output + 256 template > max-model-len $MAXLEN"
        echo "       this is the exact round 66 bug -- fix CTXS or raise MAXLEN"
        exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 67: turboquant vs context, curve filled ==="
echo "    contexts: $CTXS   output: $OUT_LEN   max-model-len: $MAXLEN"

cleanup() { docker rm -f rd67-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag kvdtype
    local tag="$1" kv="$2" extra="" t=0
    [ "$kv" != "auto" ] && extra="--kv-cache-dtype $kv"
    echo ""
    echo "=== $(date -u +%T) arm $tag (kv=$kv) ==="
    docker rm -f rd67-srv >/dev/null 2>&1
    # shellcheck disable=SC2086
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" rd67-srv 8181 --max-model-len "$MAXLEN" $extra >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8181/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd67-srv$' || {
            echo "  EXITED -- log kept at $LOGS/rd67-$tag.startup-fail"
            docker logs rd67-srv > "$LOGS/rd67-$tag.startup-fail" 2>&1 || true
            tail -12 "$LOGS/rd67-$tag.startup-fail"
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"
    docker logs rd67-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'

    for c in $CTXS; do
        # One request at a time, prompt of c tokens, then OUT_LEN decode steps.
        # Decode re-reads all c tokens of KV every step, so c IS the KV size
        # under test and TPOT is the per-token cost of reading it.
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8181 \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$c" --random-output-len "$OUT_LEN" \
            --num-prompts 4 --max-concurrency 1 \
            --ignore-eos --seed 1234 > "$LOGS/rd67-$tag-c$c.bench" 2>&1

        # Report completed/failed BEFORE any timing, so a zero row is visibly a
        # failed run rather than a fast one. Round 66 printed 0.00 and it took
        # reading the raw log to learn nothing had run at all.
        local failed
        failed=$(grep -aoE "Failed requests: +[0-9]+" "$LOGS/rd67-$tag-c$c.bench" | grep -aoE "[0-9]+$" | head -1)
        printf "  ctx=%-7s " "$c"
        if [ -n "${failed:-}" ] && [ "$failed" -gt 0 ]; then
            echo "FAILED ($failed requests rejected) -- see $LOGS/rd67-$tag-c$c.bench"
            continue
        fi
        printf "TTFT %8s ms   TPOT %7s ms\n" \
          "$(grep -aoE "Median TTFT \(ms\): +[0-9.]+" "$LOGS/rd67-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)" \
          "$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$LOGS/rd67-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)"
    done
    docker rm -f rd67-srv >/dev/null 2>&1
}

run_arm bf16 auto               || { echo "BASELINE FAILED -- aborting"; exit 1; }
run_arm tq4  turboquant_4bit_nc || echo "(turboquant arm failed; see above)"

echo ""
echo "=== $(date -u +%T) round 67 done ==="
CTXS="$CTXS" python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CTXS = [int(c) for c in os.environ["CTXS"].split()]
def v(tag, c, pat):
    p = os.path.join(L, f"rd67-{tag}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    x = float(m.group(1))
    return x if x > 0 else None
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"

print(f"{'context':>9}{'bf16 TPOT':>11}{'tq4 TPOT':>10}{'ratio':>9}"
      f"{'bf16 TTFT':>12}{'tq4 TTFT':>11}")
print("-" * 62)
rows = []
for c in CTXS:
    a, b = v("bf16", c, TPOT), v("tq4", c, TPOT)
    fa, fb = v("bf16", c, TTFT), v("tq4", c, TTFT)
    if a and b: rows.append((c, a, b))
    print(f"{c:>9}{(f'{a:11.2f}' if a else f'{chr(45):>11}')}"
          f"{(f'{b:10.2f}' if b else f'{chr(45):>10}')}"
          f"{(f'{b/a:8.3f}x' if a and b else f'{chr(45):>9}')}"
          f"{(f'{fa:12.1f}' if fa else f'{chr(45):>12}')}"
          f"{(f'{fb:11.1f}' if fb else f'{chr(45):>11}')}")

if len(rows) >= 2:
    print()
    print("SLOPE -- ms of TPOT added per 1000 tokens of context, between points.")
    print("This is the number that decides the question. A kernel paying a FIXED")
    print("overhead has a slope no worse than the baseline; a kernel that is")
    print("inefficient per KV byte has a steeper slope, and gets worse forever.")
    print(f"  {'span':>18}{'bf16':>10}{'tq4':>10}{'tq4/bf16':>11}")
    for (c0, a0, b0), (c1, a1, b1) in zip(rows, rows[1:]):
        dk = (c1 - c0) / 1000.0
        sa, sb = (a1 - a0) / dk, (b1 - b0) / dk
        print(f"  {f'{c0}->{c1}':>18}{sa:10.4f}{sb:10.4f}"
              f"{(f'{sb/sa:10.2f}x' if sa else f'{chr(45):>11}')}")
    print()
    c_last, a_last, b_last = rows[-1]
    print(f"AT THE LONGEST CONTEXT MEASURED ({c_last} tokens): bf16 {a_last:.2f} ms/token,")
    print(f"turboquant {b_last:.2f} ms/token, a {b_last/a_last:.2f}x penalty. Read that as")
    print(f"tokens per second per stream: {1000/a_last:.1f} vs {1000/b_last:.1f}.")

print()
print("HOW TO READ THIS AGAINST THE CAPACITY WIN. Round 66 confirmed turboquant")
print("holds 2,790,992 KV tokens against bf16's 902,160 -- 3.09x, and the only")
print("configuration that fits a 1M context at all. The trade is therefore not")
print("'is turboquant fast' but 'is a slower 1M context worth more than a fast")
print("context that cannot reach 1M'. A ratio that GROWS with context makes that")
print("trade worse the further out you go, which is the opposite of the shape")
print("that would have justified it.")
PY
