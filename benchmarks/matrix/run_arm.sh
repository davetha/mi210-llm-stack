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
    # longctx defaults to 262144, which on vLLM/gfx9 is ABOVE the
    # use_rocm_custom_paged_attention ceiling (max_seq_len <= 128*1024) and
    # therefore measures the Triton fallback rather than the quantization.
    # LONGCTX_TOKENS lets a sweep pin it just under the ceiling; llama.cpp
    # arms, which have no such gate, can leave it at the full 262144.
    extra=()
    if [ "$wl" = "longctx" ] && [ -n "${LONGCTX_TOKENS:-}" ]; then
        extra=(--prompt-tokens "$LONGCTX_TOKENS")
    fi
    python3 "$BIN/bench_matrix.py" \
        --url "http://127.0.0.1:$PORT" --model bench \
        --label "$LABEL" --workload "$wl" \
        --tier "$TIER" --quant "$QUANT" --engine "$ENGINE" \
        --vram-cmd "$VRAM_CMD" \
        "${extra[@]}" \
        --out "$RESULTS/$LABEL-$wl.json" \
        2>&1 | tee -a "$LOGS/$LABEL-$wl.log"
    # PIPESTATUS, not $?: $? is tee's exit status, which is 0 even when the
    # benchmark failed its correctness check or aborted on a cache hit.
    [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "workload $wl reported a problem"; rc=1; }
done

docker logs "$NAME" > "$LOGS/$LABEL.serverlog" 2>&1 || true

# Harvest the engine's own footprint accounting before tearing the container
# down. Sampling rocm-smi does NOT give model size: vLLM preallocates KV cache
# to --gpu-memory-utilization, so a 16 GB model and a 60 GB model both report
# ~90% of the card. The deliverable needs weights-vs-KV separated, and only the
# engine knows the split.
python3 - "$LOGS/$LABEL.serverlog" "$RESULTS" "$LABEL" <<'PY' || true
import json, os, re, sys, glob
logfile, results, label = sys.argv[1:4]
try:
    text = open(logfile, errors="replace").read()
except OSError:
    sys.exit(0)
pats = {
    "weights_gib":  r"Model loading took ([\d.]+) GiB",
    "load_seconds": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
    "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "graph_capture_gib": r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB",
}
found = {}
for key, pat in pats.items():
    m = re.search(pat, text)
    if m:
        found[key] = float(m.group(1).replace(",", ""))
if not found:
    sys.exit(0)
for path in glob.glob(os.path.join(results, label + "-*.json")):
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        continue
    doc["engine_footprint"] = found
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"  footprint merged into {os.path.basename(path)}: {found}")
PY

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "ARM $LABEL done (rc=$rc)"
exit $rc
