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

# Wait for the previous arm's port to actually be released before binding it.
#
# `docker rm -f` returns before the daemon has torn down the port mapping, so
# back-to-back arms race: the next `docker run` gets
#
#   Bind for 0.0.0.0:8100 failed: port is already allocated
#
# and the arm is recorded as "server would not start" with a ZERO-BYTE server
# log -- indistinguishable at a glance from a model that genuinely cannot load.
# That is exactly how the 80B W8A8 decode arm was lost: it never created a
# container at all, and the failure looked like a model problem.
#
# Also sweep up any leftover container still holding the port, which `rm -f` in
# a previous *interrupted* run would not have cleaned.
for _ in $(seq 1 60); do
    holder=$(docker ps -q --filter "publish=$PORT" 2>/dev/null)
    [ -z "$holder" ] && break
    echo "  port $PORT still held by $holder -- removing and waiting"
    docker rm -f $holder >/dev/null 2>&1 || true
    sleep 2
done
# Even with no container holding it, the daemon can lag on the mapping.
for _ in $(seq 1 30); do
    docker ps -q --filter "publish=$PORT" 2>/dev/null | grep -q . || break
    sleep 2
done

case "$ENGINE" in
    # vllm-aiter is the DEFAULT for vLLM arms now: AITER ASM attention measured
    # 12.8% faster prefill than stock ROCM_ATTN on gfx90a, proven by the .co
    # loads rather than inferred. "vllm" remains available for A/B against it.
    vllm-aiter) VLLM_PREFER_AITER_FA=1 "$BIN/serve_vllm_aiter.sh" "$MODEL_DIR" "$NAME" "$PORT" "$@" || { fail "server would not start"; exit 1; } ;;
    vllm)     "$BIN/serve_vllm.sh"     "$MODEL_DIR" "$NAME" "$PORT" "$@" || { fail "server would not start"; exit 1; } ;;
    llamacpp) "$BIN/serve_llamacpp.sh" "$MODEL_DIR" "$NAME" "$PORT" "$@" || { fail "server would not start"; exit 1; } ;;
    *) echo "unknown engine $ENGINE"; exit 2 ;;
esac

# Poll for readiness, but bail early if the container dies -- otherwise a model
# that OOMs on load burns the full READY_TIMEOUT before reporting.
#
# THE CONTAINER-EXITED CHECK IS NOT ENOUGH. A vLLM worker rank can die while the
# container stays up: the engine keeps running and waits forever on the dead
# rank, printing "shm_broadcast: no available block" once a minute. Neither the
# readiness curl nor the container check fires, so the arm burns the FULL
# READY_TIMEOUT. That is exactly how the prefetch-offload arm cost an hour --
# it had already crashed at 22:47 with
#
#   Worker_TP1: RuntimeError: NCCL error: unhandled cuda error in ncclAllReduce
#
# and four more arms behind it would each have repeated the 90-minute wait, for
# a total of six hours to learn nothing new. See docs/29.
#
# The grace counter matters. A single match is not proof: a healthy start can
# log a caught traceback, so the pattern must still be present 60s later before
# the arm is failed. And the pattern is deliberately narrow -- every healthy
# vLLM start logs "ERROR ... Failed to import Triton kernels", so anything
# matching a bare "ERROR" would fail every arm.
FATAL_RE='NCCL error|RuntimeError:|Assertion .* failed|CUDA error: |HSA_STATUS_ERROR'
echo "waiting for $NAME to become ready (timeout ${READY_TIMEOUT}s)..."
elapsed=0
crash_seen=0
until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! docker ps --filter "name=^${NAME}$" --format '{{.Names}}' | grep -q .; then
        fail "container exited during load"; exit 1
    fi
    if [ $((elapsed % 60)) -eq 0 ] && [ "$elapsed" -gt 0 ]; then
        hit=$(docker logs --tail 500 "$NAME" 2>&1 | grep -oE "$FATAL_RE.*" | tail -1)
        if [ -n "$hit" ]; then
            crash_seen=$((crash_seen + 1))
            echo "  fatal pattern in server log (${crash_seen}/2): $hit"
            if [ "$crash_seen" -ge 2 ]; then
                fail "worker crashed during load: $hit"; exit 1
            fi
        else
            crash_seen=0
        fi
    fi
    sleep 10; elapsed=$((elapsed + 10))
    if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
        fail "not ready after ${READY_TIMEOUT}s"; exit 1
    fi
done
echo "ready after ${elapsed}s"

# Clamp the long-context prompt to what the server was actually configured for.
#
# llama.cpp arms pass --ctx-size N, but the longctx workload defaults to 262144
# tokens. Mismatch produces
#   HTTP 400: request (241846 tokens) exceeds the available context size (32768)
# on every rep, and the arm is recorded as a failure that looks like a model or
# placement problem. That silently killed the whole --n-cpu-moe decode sweep --
# four arms, all four "failed", none of them for any reason worth knowing.
#
# 0.85 leaves room for the generated tokens and the harness's own framing.
if [ -z "${LONGCTX_TOKENS:-}" ]; then
    ctx_arg=""
    prev=""
    for a in "$@"; do
        case "$prev" in --ctx-size|-c) ctx_arg="$a" ;; esac
        prev="$a"
    done
    if [ -n "$ctx_arg" ] && [ "$ctx_arg" -gt 0 ] 2>/dev/null; then
        LONGCTX_TOKENS=$(( ctx_arg * 85 / 100 ))
        export LONGCTX_TOKENS
        echo "  longctx clamped to $LONGCTX_TOKENS (server --ctx-size $ctx_arg)"
    fi
fi

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
    # -u: unbuffered. Piping through tee makes stdout a pipe rather than a
    # tty, so Python block-buffers it and a long arm shows NOTHING until the
    # buffer fills or the process exits -- on a 357B model that is 20+ minutes
    # of apparent silence that reads exactly like a hang.
    # ARM_TIMEOUT raises the per-request budget for tiers where a single
    # prefill legitimately runs for many minutes. bench_matrix caps each socket
    # read at max(600, timeout/4), so the default 3600 gives a 900 s read cap --
    # fine everywhere except the ~400B CPU-offloaded arms, where TTFT at 28k can
    # approach that on its own. Aborting there would record a failure that is
    # really just a slow but working configuration.
    python3 -u "$BIN/bench_matrix.py" \
        --timeout "${ARM_TIMEOUT:-3600}" \
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

# ---------------------------------------------------------------- ASM evidence
# WHICH KERNEL RAN IS A MEASUREMENT, NOT AN INFERENCE. The `LoadKernel:` line
# from AITER's C++ runtime is the only direct proof that an ASM code object
# executed rather than a CK or Triton fallback. docs/14 and docs/16 both
# published gfx90a "ASM" throughput that was actually the fallback because
# nothing checked, and docs/19 had to retract every ASM figure predating the
# port matrix for the same reason. A fast number with no proof of ASM is
# probably the fallback.
#
# So record it per arm, at measurement time, beside the result. Attribution
# cannot be reconstructed later from a container that has been torn down.
#
# Harvested from `docker logs` rather than in-process because the runtime writes
# the line straight to fd 1, which is why contextlib.redirect_stdout cannot see
# it (docs/17). Needs AITER_LOG_LEVEL=info, which serve_vllm_aiter.sh sets. On
# stock-vLLM arms no objects appear and "any_asm": false is the correct answer,
# not a failure.
python3 - "$LOGS/$LABEL.serverlog" "$RESULTS/$LABEL-asm.json" "$LABEL" <<'ASMPY' || true
import collections, json, os, re, sys
logfile, out, label = sys.argv[1:4]
objs = collections.Counter()
readable = True
try:
    with open(logfile, errors="replace") as fh:
        for line in fh:
            if "LoadKernel" not in line:
                continue
            for m in re.findall(r"[\w./+-]+\.co\b", line):
                objs[os.path.basename(m)] += 1
except OSError:
    # A missing serverlog must still leave a verdict on disk. Writing nothing
    # would make "the check never ran" indistinguishable from "no ASM loaded",
    # which is the ambiguity this whole block exists to remove.
    readable = False
# Family = leading token of the object name (fmha, pa, fmoe, mla, ...), which is
# how the port matrix in docs/19 is indexed.
fams = collections.Counter()
for co, n in objs.items():
    fams[co.split("_")[0]] += n
json.dump({"label": label, "serverlog_readable": readable,
           "any_asm": bool(objs), "distinct_objects": len(objs),
           "families": dict(sorted(fams.items())),
           "objects": dict(sorted(objs.items()))},
          open(out, "w"), indent=2)
if objs:
    print("  ASM: %d distinct code objects loaded -- %s"
          % (len(objs), ", ".join("%s x%d" % (k, v) for k, v in sorted(fams.items()))))
elif not readable:
    print("  ASM: serverlog unreadable -- attribution UNKNOWN, not proven absent")
else:
    print("  ASM: none loaded -- this arm's numbers are NOT ASM-attributable")
ASMPY

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
