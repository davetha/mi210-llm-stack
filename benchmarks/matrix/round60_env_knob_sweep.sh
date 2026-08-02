#!/usr/bin/env bash
# Round 60: the runtime env knobs nobody has A/B'd. Low confidence, cheap.
#
# WHY THESE FOUR, AND WHY EXPECTATIONS ARE LOW. Every kernel-level lead in
# docs/50 and docs/51 came back null or closed, and the one large finding was a
# power cap. These are the remaining runtime settings that plausibly touch this
# box's specific weaknesses, run because they are cheap rather than because
# they are promising:
#
#   GPU_MAX_HW_QUEUES   currently 4 (set by serve_vllm.sh). More queues means
#                       more concurrent HSA queues; fewer means less scheduler
#                       overhead. On a 2-GPU TP=2 setup with ~96 collectives
#                       per decode step, queue count interacts with how those
#                       serialise. Swept 1 / 4 / 8.
#   HSA_ENABLE_SDMA     toggles the DMA engines for host<->device copies.
#                       RELEVANT HERE specifically because there is no P2P on
#                       this box, so TP collectives are HOST-STAGED -- docs/51
#                       measured that path at ~86 us per collective, 99.6% of
#                       which is fixed latency. If SDMA vs blit changes that
#                       fixed cost, it shows up 96 times per token.
#   --block-size 32     vLLM's paged-attention KV block size, default 16. The
#                       gfx90a pa_asm.csv carries a block-size column reading
#                       16, so 32 may fall off the ASM path entirely -- which
#                       is itself worth knowing, and is checked via .co load.
#
# ONE VARIABLE PER ARM, all against a common baseline, same image, same load.
#
# CONCURRENCY 8 AND 32 ONLY -- NOT 1. docs/51 established a ~7.6% ordering bias
# at concurrency 1 in this harness: round 59b's two functionally IDENTICAL arms
# differed by 0.924x at conc 1 while agreeing to 0.2% at conc 8 and 32, and
# round 59 reproduced the same 0.924x with a different image pair. The cause is
# num_prompts = conc * 8 = 8 being too few to amortise warmup. Including conc 1
# here would hand every second arm a spurious 7.6% penalty.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"8 32"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 60: runtime env knob sweep ==="

ARMS="base q1 q8 nosdma blk32"
cleanup() { for a in $ARMS; do docker rm -f "rd60-$a" >/dev/null 2>&1 || true; done; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED:"; docker logs "$name" 2>&1 | tail -30; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

run_arm() {  # name port extra_env extra_args
    local name="rd60-$1" port="$2" env="$3" args="${4:-}"
    echo ""
    echo "=== $(date -u +%T) arm $name  env[$env] args[$args] ==="
    # shellcheck disable=SC2086
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="$env" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" \
        --max-model-len 32768 $args >/dev/null
    wait_ready "$port" "$name" || { echo "  ARM FAILED -- reported as a result"; return 1; }

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
          | grep -E "Output token throughput|Median TPOT"
    done
    docker logs "$name" > "$LOGS/$name.container" 2>&1 || true
    local co
    co=$(grep -ohE "pa_[a-z0-9_]*\.co|fwd_hd[0-9x]+_bf16[a-z_]*\.co" "$LOGS/$name.container" | sort -u | tr '\n' ' ')
    echo "  ASM .co: ${co:-<none>}"
    echo "$co" > "$LOGS/$name.cofiles"
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm base   8120 "-e GPU_MAX_HW_QUEUES=4" || { echo "BASELINE FAILED"; exit 1; }
run_arm q1     8121 "-e GPU_MAX_HW_QUEUES=1"
run_arm q8     8122 "-e GPU_MAX_HW_QUEUES=8"
run_arm nosdma 8123 "-e GPU_MAX_HW_QUEUES=4 -e HSA_ENABLE_SDMA=0"
run_arm blk32  8124 "-e GPU_MAX_HW_QUEUES=4" "--block-size 32"

echo ""
echo "=== $(date -u +%T) round 60 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "8 32").split()]
ARMS = [("base", "GPU_MAX_HW_QUEUES=4 (stock)"), ("q1", "queues=1"),
        ("q8", "queues=8"), ("nosdma", "HSA_ENABLE_SDMA=0"),
        ("blk32", "--block-size 32")]
def val(arm, c, pat):
    p = os.path.join(L, f"rd60-{arm}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
hdr = f"{'arm':<28}"
for c in CONCS: hdr += f"{'c'+str(c)+' tok/s':>13}{'vs base':>9}"
print(hdr); print("-" * len(hdr))
for a, label in ARMS:
    row = f"{label:<28}"
    for c in CONCS:
        v, b = val(a, c, TPUT), val("base", c, TPUT)
        row += f"{v:13.2f}" if v else f"{'-':>13}"
        row += (f"{v/b:8.3f}x" if v and b else f"{'-':>9}")
    print(row)
print()
for a, label in ARMS:
    f = os.path.join(L, f"rd60-{a}.cofiles")
    s = open(f).read().strip() if os.path.isfile(f) else ""
    print(f"  {a:<8} ASM .co: {s or '<none>'}")
print()
print("docs/46 puts the decode bar at 1.036x for a single pair of arms, and")
print("these are single pairs. Treat anything under that as noise. The .co")
print("lines matter for blk32 specifically: pa_asm.csv carries block-size 16,")
print("so if --block-size 32 drops the pa objects it has left the ASM path and")
print("its throughput is measuring a different kernel, not a different setting.")
print()
print("Concurrency 1 is deliberately absent -- docs/51 measured a ~7.6%")
print("second-arm penalty there in this harness, which would fake a result for")
print("every arm after the baseline.")
PY
