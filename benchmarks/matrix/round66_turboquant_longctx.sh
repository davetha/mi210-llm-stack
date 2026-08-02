#!/usr/bin/env bash
# SUPERSEDED BY round67_turboquant_longctx_fill.sh -- READ THIS FIRST.
#
# This round has a defect. It serves with --max-model-len 131072 and then asks
# the benchmark for --random-input-len 131072 --random-output-len 64, so at the
# 131072 point prompt+generation exceeds the window and EVERY request is
# rejected. Both arms printed 0.00 for that row. A zero that means "nothing ran"
# is indistinguishable from a zero that means "immeasurably fast" once it is in
# a table, which is why round 67 prints the failed-request count BEFORE any
# timing and refuses to start if the arithmetic does not clear.
#
# Its 8192 and 32768 rows are valid and reproduced in round 67 to within 1.5%.
# The conclusion drawn here was also wrong in an instructive way: the round was
# built to test whether turboquant's cost FALLS with context, and pre-registered
# only "falls" and "flat" as outcomes. The measured answer was neither -- the
# ratio RISES, 1.475x at 8192 to 5.214x at 130000. See docs/54.

# Round 66: does turboquant's cost invert at long context?
#
# THE HYPOTHESIS, AND WHY ROUND 64 COULD NOT SEE IT. Round 64 measured
# turboquant_4bit_nc at 0.511x throughput -- but with 8192-token inputs, where
# the KV cache is small and compression is pure overhead: dequant work with
# almost no bytes saved, plus the loss of the AITER FA backend.
#
# Decode reads the ENTIRE KV every token. So KV bytes scale with context while
# turboquant's overhead is roughly fixed. At long context the ratio should move:
#
#     short ctx   overhead dominates          -> turboquant loses (measured)
#     long ctx    KV bandwidth dominates       -> 3x fewer bytes may win
#
# This matters because vLLM has NO CPU KV offload -- no -ngl equivalent, and
# --cpu-offload-gb moves WEIGHTS not KV -- so on vLLM the active KV must live in
# VRAM and compression is the only lever for long context.
#
# AND THE CAPACITY WALL IS REAL. Round 64 measured:
#     bf16               902,176 KV tokens   <- SHORT of a single 1M context
#     turboquant_4bit_nc 2,791,056 KV tokens
# At bf16 a 1M context does not fit at all on this model.
#
# DESIGN. Concurrency 1 deliberately: one sequence, so decode re-reads that
# sequence's full KV every token and the context length IS the KV size. Running
# concurrency 8 would conflate per-sequence context with batch size.
# Contexts 8k / 32k / 131k, both arms, same image and model.
#
# NOTE ON THE conc-1 BIAS. docs/51 found a ~7.6% second-arm penalty at conc 1
# with only 8 prompts. Here each point uses a SINGLE long request rather than 8
# short ones, and the per-token decode rate is read from vLLM's own timing, so
# that warmup artefact does not apply the same way -- but treat differences
# under ~8% as directional rather than decisive.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model"; exit 1; }
CTXS=${CTXS:-"8192 32768 131072"}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 66: turboquant vs bf16 across context length ==="

cleanup() { docker rm -f rd66-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag kvdtype
    local tag="$1" kv="$2" extra="" t=0
    [ "$kv" != "auto" ] && extra="--kv-cache-dtype $kv"
    echo ""
    echo "=== $(date -u +%T) arm $tag (kv=$kv) ==="
    docker rm -f rd66-srv >/dev/null 2>&1
    # shellcheck disable=SC2086
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" rd66-srv 8180 --max-model-len 131072 $extra >/dev/null
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8180/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd66-srv$' || { echo "  EXITED"; docker logs rd66-srv 2>&1 | tail -12; return 1; }
        sleep 10; t=$((t+10))
    done
    [ $t -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"
    docker logs rd66-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'

    for c in $CTXS; do
        # One request whose PROMPT is c tokens, then 64 decode steps. Decode
        # re-reads all c tokens of KV per step, so c is the KV size under test.
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8180 \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$c" --random-output-len 64 \
            --num-prompts 4 --max-concurrency 1 \
            --ignore-eos --seed 1234 > "$LOGS/rd66-$tag-c$c.bench" 2>&1
        printf "  ctx=%-7s " "$c"
        grep -aE "Median TPOT|Output token throughput" "$LOGS/rd66-$tag-c$c.bench" \
          | grep -aoE "[0-9.]+" | tr '\n' ' '
        echo
    done
    docker rm -f rd66-srv >/dev/null 2>&1
}

run_arm bf16 auto               || { echo "BASELINE FAILED"; exit 1; }
run_arm tq4  turboquant_4bit_nc || echo "(turboquant arm failed)"

echo ""
echo "=== $(date -u +%T) round 66 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CTXS = [int(c) for c in os.environ.get("CTXS", "8192 32768 131072").split()]
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
def v(tag, c):
    p = os.path.join(L, f"rd66-{tag}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(TPOT, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
print(f"{'context':>9}{'bf16 TPOT ms':>14}{'tq4 TPOT ms':>13}{'tq4/bf16':>10}")
print("-" * 46)
for c in CTXS:
    a, b = v("bf16", c), v("tq4", c)
    r = f"{b/a:9.3f}x" if a and b else f"{'-':>10}"
    print(f"{c:>9}{(f'{a:14.2f}' if a else f'{chr(45):>14}')}"
          f"{(f'{b:13.2f}' if b else f'{chr(45):>13}')}{r}")
print()
print("TPOT is the right metric here -- it is per-token decode latency, which")
print("is what KV bandwidth drives. Below 1.0 means turboquant is FASTER.")
print()
print("THE SHAPE IS THE RESULT, not any single row. If the ratio falls as")
print("context grows, compression is paying for its overhead and the crossover")
print("is where it starts winning. If the ratio is flat near 1.5, the cost is")
print("the lost AITER FA backend rather than KV bytes, and no context length")
print("will rescue it.")
PY
