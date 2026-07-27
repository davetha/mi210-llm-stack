#!/usr/bin/env bash
# Run one arm of the matrix end to end: serve, wait, benchmark, tear down.
#
#   ./run_arm.sh <label> <tier> <quant> <engine> <model-dir> [extra serve args]
#
#   ./run_arm.sh t35-bf16 35B bf16 vllm /mnt/llm-storage/bench-matrix/t35-bf16 \
#       --max-model-len 262144
#
# A failed arm is a RESULT, not an error to be swallowed. Several quantization
# formats are expected to be unloadable on gfx90a -- AWQ and GPTQ normally
# dispatch to Marlin kernels, which are CUDA-only, and FP8 has no native compute
# on CDNA2. When an arm fails, this writes a JSON record with status=FAILED and
# the captured server error, then moves on. Those records are part of the
# answer: "which formats win on MI210" is not answerable without also knowing
# which ones will not run at all. What must never happen is a format quietly
# disappearing from the results table with no explanation.
set -uo pipefail

LABEL="${1:?usage: run_arm.sh <label> <tier> <quant> <engine> <model-dir> [args]}"
TIER="${2:?}"; QUANT="${3:?}"; ENGINE="${4:?}"; MODEL_DIR="${5:?}"
shift 5

BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
RESULTS=$BASE/results
LOGS=$BASE/logs
PORT=8100
NAME="bench-$LABEL"
READY_TIMEOUT=${READY_TIMEOUT:-2400}   # 40 min: a 400B model off NVMe is slow

mkdir -p "$RESULTS" "$LOGS"

# rocm-smi reads sysfs and reports PHYSICAL cards, so it is the right tool for
# a footprint measurement regardless of the HIP/rocm-smi index disagreement
# documented in env/gfx90a-common.env -- we want the total across both cards,
# and a sum does not care about ordering.
VRAM_CMD="docker exec $NAME rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Total Used' | grep -oE '[0-9]{6,}'"

fail() {
    local reason="$1"
    echo "ARM FAILED: $LABEL -- $reason"
    docker logs "$NAME" > "$LOGS/$LABEL.serverlog" 2>&1 || true
    python3 - "$RESULTS/$LABEL-FAILED.json" "$LABEL" "$TIER" "$QUANT" "$ENGINE" "$reason" <<'PY'
import json, sys
out, label, tier, quant, engine, reason = sys.argv[1:7]
tail = ""
try:
    with open(out.rsplit("/results/", 1)[0] + "/logs/" + label + ".serverlog") as fh:
        tail = "".join(fh.readlines()[-40:])
except OSError:
    pass
json.dump({"label": label, "tier": tier, "quant": quant, "engine": engine,
           "status": "FAILED", "reason": reason, "server_log_tail": tail},
          open(out, "w"), indent=2)
PY
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    return 1
}

echo "=============================================================="
echo "ARM $LABEL  tier=$TIER quant=$QUANT engine=$ENGINE"
echo "  model: $MODEL_DIR"
echo "=============================================================="

case "$ENGINE" in
    vllm)     "$BIN/serve_vllm.sh"     "$MODEL_DIR" "$NAME" "$PORT" "$@" || { fail "server would not start"; exit 1; } ;;
    llamacpp) "$BIN/serve_llamacpp.sh" "$MODEL_DIR" "$NAME" "$PORT" "$@" || { fail "server would not start"; exit 1; } ;;
    *) echo "unknown engine $ENGINE"; exit 2 ;;
esac

# Poll for readiness, but bail early if the container dies -- otherwise a model
# that OOMs on load burns the full READY_TIMEOUT before reporting.
echo "waiting for $NAME to become ready (timeout ${READY_TIMEOUT}s)..."
elapsed=0
until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! docker ps --filter "name=^${NAME}$" --format '{{.Names}}' | grep -q .; then
        fail "container exited during load"; exit 1
    fi
    sleep 10; elapsed=$((elapsed + 10))
    if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
        fail "not ready after ${READY_TIMEOUT}s"; exit 1
    fi
done
echo "ready after ${elapsed}s"

rc=0
for wl in cold16k longctx; do
    echo "--- workload $wl ---"
    python3 "$BIN/bench_matrix.py" \
        --url "http://127.0.0.1:$PORT" --model bench \
        --label "$LABEL" --workload "$wl" \
        --tier "$TIER" --quant "$QUANT" --engine "$ENGINE" \
        --vram-cmd "$VRAM_CMD" \
        --out "$RESULTS/$LABEL-$wl.json" \
        2>&1 | tee -a "$LOGS/$LABEL-$wl.log"
    # PIPESTATUS, not $?: $? is tee's exit status, which is 0 even when the
    # benchmark failed its correctness check or aborted on a cache hit.
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "workload $wl reported a problem"; rc=1; }
done

docker logs "$NAME" > "$LOGS/$LABEL.serverlog" 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "ARM $LABEL done (rc=$rc)"
exit $rc
