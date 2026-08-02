#!/usr/bin/env bash
# Round 59b: int8 rmsnorm+quant fusion, measured properly this time.
#
# ROUND 59 WAS INVALID. Three defects, all in the harness rather than the idea:
#
#   1. NO PROOF THE FUSION FIRED. The "fusion-pass mentions: 2" it reported
#      were its own explanatory text grepped back out of the round log --
#      the strings `RocmAiterRMSNormQuantFusion` and `fuse_norm_quant` appear
#      in the guidance block the script prints at the end. Real vLLM
#      pattern-match evidence: zero lines. The check also ran before the
#      workload.
#   2. CONFOUNDED ARMS. Control was vllm-mi210:aiterops, fused was
#      vllm-mi210:mastergate. That compares (carve-out + flag) against
#      (neither), not the flag alone. The mastergate image carves out
#      is_enabled(), which enables FOUR ROCm fusion passes, so any difference
#      was unattributable to fuse_norm_quant specifically.
#   3. MIXED SIGNS across concurrency (0.924x, 1.051x, 1.004x) -- exactly what
#      1 and 2 would produce.
#
# WHAT CHANGES HERE:
#   - BOTH arms run vllm-mi210:mastergate. The only difference is the
#     --compilation-config flag. One variable.
#   - Container logs are captured to their own files AFTER the workload, and
#     the fusion evidence is grepped from THOSE, never from this script's
#     stdout. A grep that can match your own narration is not a measurement.
#   - VLLM_LOGGING_LEVEL=DEBUG on both arms so torch.compile pattern counts
#     are actually emitted; set identically so it cannot skew the comparison.
#
# THE PRIOR, RESTATED. docs/46 established that vLLM's rms_quant_fusion.py has
# ZERO int8 patterns (FUSED_OPS/QUANT_OPS are fp8/fp4 only), that AITER has the
# kernel (_aiter_ops.py:723 asserts quant_dtype in [torch.int8, FP8_DTYPE] and
# calls aiter.rmsnorm2d_fwd_with_dynamicquant), and that the pass is only added
# when rocm_aiter_ops.is_enabled() -- carved out by the mastergate image and
# validated safe by round 46. Everything was prepared; the flag was never set.
#
# SO THE PRIMARY OUTPUT OF THIS ROUND IS NOT THROUGHPUT. It is a yes/no:
# does RocmAiterRMSNormQuantFusionPass match this graph? If no, the lead closes
# as "the ROCm pass does not match a W8A8 Qwen3-MoE graph" and no throughput
# number from it means anything. Only if it fires do the timings matter.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:mastergate

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"1 8 32"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 59b: int8 norm+quant fusion, one variable ==="

cleanup() { docker rm -f rd59b-off rd59b-on >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED:"; docker logs "$name" 2>&1 | tail -40; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

run_arm() {  # name port [extra args...]
    local name="$1" port="$2"; shift 2
    echo ""
    echo "=== $(date -u +%T) arm $name  $* ==="
    VLLM_IMAGE="$IMG" TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="-e VLLM_LOGGING_LEVEL=DEBUG" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" \
        --max-model-len 32768 "$@" >/dev/null
    wait_ready "$port" "$name" || return 1

    for c in $CONCS; do
        echo "--- $(date -u +%T) $name @ concurrency $c ---"
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

    # Container log to its OWN file, AFTER the workload. Round 59 grepped its
    # own narration; this cannot, because this file contains only vLLM output.
    docker logs "$name" > "$LOGS/$name.container" 2>&1 || true
    echo "  container log: $(wc -l < "$LOGS/$name.container") lines"
    local asm ck
    asm=$(grep -c "LoadKernel" "$LOGS/$name.container"); asm=${asm:-0}
    ck=$(grep -c "module_gemm_a8w8" "$LOGS/$name.container"); ck=${ck:-0}
    echo "  ASM objects: $asm   CK GEMM refs: $ck   (both must be nonzero)"
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm rd59b-off 8117 || { echo "OFF arm failed"; exit 1; }
run_arm rd59b-on  8118 --compilation-config '{"pass_config":{"fuse_norm_quant":true}}' \
    || echo "(ON arm failed)"

echo ""
echo "=== $(date -u +%T) round 59b done ==="
echo "########## DID THE PASS FIRE? (from container logs only) ##########"
for a in rd59b-off rd59b-on; do
    f="$LOGS/$a.container"
    [ -f "$f" ] || { echo "  $a: no container log"; continue; }
    echo "-- $a --"
    n=$(grep -ciE "RocmAiterRMSNormQuantFusion|rmsnorm_fused_dynamic_quant" "$f"); n=${n:-0}
    r=$(grep -aoiE "[Rr]eplaced [0-9]+ (patterns|nodes)|fusion.*[0-9]+ (match|replacement)" "$f" | head -4)
    echo "   RocmAiterRMSNormQuantFusion / fused-op refs: $n"
    echo "   replacement lines: ${r:-<none>}"
    grep -aiE "RocmAiterRMSNormQuantFusionPass|fuse_norm_quant" "$f" | head -3 | sed 's/^/     /'
done
echo ""
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
print(f"{'conc':>5}{'fuse OFF':>11}{'fuse ON':>11}{'ON/OFF':>9}"
      f"{'OFF TPOT':>10}{'ON TPOT':>10}{'TPOT r':>8}")
print("-" * 64)
for c in CONCS:
    a, b = val("rd59b-off", c, TPUT), val("rd59b-on", c, TPUT)
    at, bt = val("rd59b-off", c, TPOT), val("rd59b-on", c, TPOT)
    if a is None and b is None: continue
    r  = f"{b/a:8.3f}x" if a and b else f"{'-':>9}"
    rt = f"{bt/at:7.3f}x" if at and bt else f"{'-':>8}"
    print(f"{c:>5}"
          f"{(f'{a:11.2f}' if a else f'{chr(45):>11}')}"
          f"{(f'{b:11.2f}' if b else f'{chr(45):>11}')}{r}"
          f"{(f'{at:10.2f}' if at else f'{chr(45):>10}')}"
          f"{(f'{bt:10.2f}' if bt else f'{chr(45):>10}')}{rt}")
print()
print("BOTH ARMS ARE THE SAME IMAGE. The only difference is the")
print("fuse_norm_quant flag, so any delta is attributable to it -- unlike")
print("round 59, which varied the image as well and was therefore void.")
print()
print("IF THE PASS DID NOT FIRE, ignore this table entirely. The finding is")
print("then 'RocmAiterRMSNormQuantFusionPass does not match a W8A8 Qwen3-MoE")
print("graph', which is about pattern coverage, not about performance.")
PY
