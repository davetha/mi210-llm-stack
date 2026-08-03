#!/usr/bin/env bash
# Round 78: does the tuned MoE config help DECODE at all? A cheap gate on an
# expensive question.
#
# WHY THIS ROUND EXISTS INSTEAD OF FINISHING THE SWEEP. Round 77 set out to tune
# M=1..2048. The candidate space doubles per batch size and per-candidate cost
# is roughly flat:
#
#     M=1     304 candidates    14 min   (measured)
#     M=2     608              31 min    (measured)
#     M=4   1,216             ~40 min    (projected)
#     M=16  4,864              ~2.7 h
#     M=32  9,728              ~5.4 h    <- already past the 3 h per-size cap
#     M>=64  19k+              beyond reach
#
# So the sweep as configured would spend ~5 GPU-hours and arrive at a config
# covering M=1..16 -- which is precisely the shape docs/41 measured at 0.786x
# prefill, because vLLM's nearest-M matching has no fallback. Paying hours to
# reach a known-harmful artifact is the wrong trade.
#
# THE CHEAP GATE. Single-stream decode at concurrency 1 uses M=1, and M=1 is
# already tuned. TPOT is decode-only -- it excludes the first token -- so it
# reports the tuned kernel's effect regardless of what prefill does. That makes
# a ~20-minute A/B decide whether the remaining ~8 GPU-hours are worth spending:
#
#     TPOT improves  -> the MoE kernel is a real lever, finish the sweep
#     TPOT flat      -> tuning this kernel does not help decode on gfx90a, and
#                       the whole line closes for the price of one round
#
# PREFILL IS EXPECTED TO REGRESS AND IS NOT THE METRIC. The staged config holds
# only M=1 and M=2, so every larger shape -- including chunked prefill -- matches
# M=2 by nearest-M. TTFT is reported so the damage is visible and quantified,
# but a prefill regression here is the KNOWN consequence of a deliberately
# partial config, not a finding. Do not read it as "tuning hurts".
#
# THE ASSERTION, FROM docs/41. vLLM logs
#
#     Using default MoE config. Performance might be sub-optimal! Config file
#     not found at .../E=512,N=256,...,dtype=int4_w4a16.json
#
# when the tuned config is absent. docs/41 records that without gating on this,
# a round reports "tuning doesn't help" from a config that was never loaded --
# a false negative indistinguishable from a real one. So the stock arm MUST log
# it and the tuned arm MUST NOT, and either mismatch aborts.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
SERVER_IMG=${SERVER_IMG:-vllm-mi210:v0.26.1rc0-ckgemm-warm}
MODEL=$BASE/t80-awq
TUNED_DIR_HOST=$BASE/moe-int4-deploy
TUNED_DIR_CTR=/models/bench-matrix/moe-int4-deploy
CFG="E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json"

docker image inspect "$SERVER_IMG" >/dev/null 2>&1 || { echo "FATAL: missing $SERVER_IMG"; exit 1; }
[ -f "$MODEL/config.json" ]      || { echo "FATAL: no model at $MODEL"; exit 1; }
[ -f "$TUNED_DIR_HOST/$CFG" ]    || { echo "FATAL: no tuned config at $TUNED_DIR_HOST/$CFG"; exit 1; }

MAXLEN=${MAXLEN:-40960}
CTXS=${CTXS:-"8192 32768"}
OUT_LEN=${OUT_LEN:-128}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

echo "=== tuned config under test ==="
python3 -c "
import json,sys
c=json.load(open('$TUNED_DIR_HOST/$CFG'))
print('  sizes:', sorted((int(k) for k in c if k.isdigit())))
"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 78: tuned MoE decode gate (W4A16, TP=2) ==="

cleanup() { docker rm -f rd78-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag  want_default_config(yes|no)  extra_env
    local tag="$1" want_default="$2" extra="${3:-}" t=0
    echo ""
    echo "=== $(date -u +%T) arm $tag (expect 'Using default MoE config' = $want_default) ==="
    docker rm -f rd78-srv >/dev/null 2>&1
    VLLM_IMAGE="$SERVER_IMG" TP=2 \
      VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
      VLLM_TUNED_CONFIG_FOLDER="$extra" \
      "$BIN/serve_vllm_aiter.sh" "$MODEL" rd78-srv 8192 --max-model-len "$MAXLEN" >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8192/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd78-srv$' || {
            echo "  EXITED -- log at $LOGS/rd78-$tag.startup-fail"
            docker logs rd78-srv > "$LOGS/rd78-$tag.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|not support|out of memory" "$LOGS/rd78-$tag.startup-fail" \
              | grep -avE "core_client|launch_core_engines|contextlib|triton_kernels|AVX2" | head -5
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"

    # THE GATE. Without this, a null result is indistinguishable from a config
    # that never loaded (docs/41).
    # Count, do not `grep -q`. Under `set -o pipefail`, grep -q exits as soon as
    # it matches, docker logs takes SIGPIPE, and the PIPELINE reports failure --
    # so `... | grep -q X && flag=yes` never sets the flag even when X is
    # present. The first run of this round aborted the stock arm for exactly
    # that reason while printing the matching line directly underneath, and the
    # tuned arm then "passed" for the same broken reason rather than because the
    # config loaded. grep -c reads to EOF, so no SIGPIPE; `|| true` guards the
    # no-match exit code and `head -1` guards the two-value output that a bare
    # `grep -c || echo 0` produces.
    local n saw_default="no"
    n=$( { docker logs rd78-srv 2>&1 || true; } | grep -aci "Using default MoE config" | head -1)
    [ "${n:-0}" -gt 0 ] && saw_default="yes"
    echo "  'Using default MoE config' present: $saw_default (wanted $want_default)"
    if [ "$saw_default" != "$want_default" ]; then
        echo "  ABORTING ARM: config-load state is not what this arm intends."
        echo "  A number measured here would not mean what it appears to mean."
        docker logs rd78-srv 2>&1 | grep -aiE "Using default MoE config|tuned_config|VLLM_TUNED" | head -3 | sed 's/^/    /'
        docker rm -f rd78-srv >/dev/null 2>&1
        return 1
    fi

    for c in $CTXS; do
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$SERVER_IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8192 --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$c" --random-output-len "$OUT_LEN" \
            --num-prompts 4 --max-concurrency 1 --ignore-eos --seed 1234 \
            > "$LOGS/rd78-$tag-c$c.bench" 2>&1
        local failed
        failed=$(grep -aoE "Failed requests: +[0-9]+" "$LOGS/rd78-$tag-c$c.bench" | grep -aoE "[0-9]+$" | head -1)
        printf "  ctx=%-7s " "$c"
        if [ -n "${failed:-}" ] && [ "$failed" -gt 0 ]; then echo "FAILED ($failed rejected)"; continue; fi
        printf "TPOT %7s ms   TTFT %9s ms\n" \
          "$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$LOGS/rd78-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)" \
          "$(grep -aoE "Median TTFT \(ms\): +[0-9.]+" "$LOGS/rd78-$tag-c$c.bench" | grep -aoE "[0-9.]+$" | head -1)"
    done
    docker rm -f rd78-srv >/dev/null 2>&1
}

# Args are (tag, want_default_config, tuned_config_folder). Getting this order
# wrong sets VLLM_TUNED_CONFIG_FOLDER to "yes" and asserts against an empty
# string -- which the first run of this round did. The assertion caught it, but
# name the arguments here so the next reader does not have to rediscover it.
run_arm stock yes ""                || echo "(stock arm failed/aborted)"
run_arm tuned no  "$TUNED_DIR_CTR"  || echo "(tuned arm failed/aborted)"

echo ""
echo "=== $(date -u +%T) round 78 done ==="
CTXS="$CTXS" python3 - <<'PY'
import os, re
L = "/mnt/llm-storage/bench-matrix/logs"
CTXS = [int(c) for c in os.environ["CTXS"].split()]
def v(tag, c, pat):
    p = os.path.join(L, f"rd78-{tag}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    x = float(m.group(1)); return x if x > 0 else None
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"
dash = "-"

print(f"{'ctx':>8}{'stock TPOT':>13}{'tuned TPOT':>13}{'decode':>10}"
      f"{'stock TTFT':>13}{'tuned TTFT':>13}{'prefill':>10}")
print("-" * 80)
gains = []
for c in CTXS:
    sd, td = v("stock", c, TPOT), v("tuned", c, TPOT)
    sf, tf = v("stock", c, TTFT), v("tuned", c, TTFT)
    if sd and td: gains.append(sd / td)      # >1 means tuned decodes FASTER
    print(f"{c:>8}"
          f"{(f'{sd:13.2f}' if sd else f'{dash:>13}')}"
          f"{(f'{td:13.2f}' if td else f'{dash:>13}')}"
          f"{(f'{sd/td:9.3f}x' if sd and td else f'{dash:>10}')}"
          f"{(f'{sf:13.1f}' if sf else f'{dash:>13}')}"
          f"{(f'{tf:13.1f}' if tf else f'{dash:>13}')}"
          f"{(f'{sf/tf:9.3f}x' if sf and tf else f'{dash:>10}')}")

print()
print("DECODE column >1.0 means the tuned M=1 kernel is FASTER. That is the")
print("gate. PREFILL column is expected to be <1.0 and is NOT the metric --")
print("the staged config holds only M=1 and M=2, so every prefill shape matches")
print("M=2 by nearest-M. That regression is the known cost of a deliberately")
print("partial config (docs/41 measured 0.786x from exactly this), not evidence")
print("about the tuned kernel itself.")
print()
if gains:
    best = max(gains)
    if best >= 1.03:
        print(f"VERDICT: decode improves up to {best:.3f}x. The MoE kernel IS a real")
        print("lever on gfx90a. Finishing the M sweep (~8 GPU-hours) is justified,")
        print("and the full config must cover the prefill range to be deployable.")
    elif best >= 0.99:
        print(f"VERDICT: decode is flat (best {best:.3f}x). Tuning the MoE kernel does")
        print("not help single-stream decode on this card. The remaining ~8 GPU-hours")
        print("of sweep are NOT justified -- this closes for the price of one round.")
    else:
        print(f"VERDICT: decode got WORSE ({best:.3f}x) even at the tuned size. The")
        print("tuner's own choice loses to the heuristic at M=1, which is a result")
        print("about the tuner, not just about this config. Do not continue.")
PY
