#!/usr/bin/env bash
# Round 38c: redo of round 38's phase 2, which failed rc=1 with its error
# DISCARDED -- the docker output was piped through `tail -8`, so the only
# evidence left was the trace files. Their autopsy: init-phase kernels only
# (embedding, fills, RNG, 5.7 ms total of scaled_mm), zero attention or MoE
# kernels, generate() never reached, death near a hipModuleLaunchKernel after
# ~38k hipStreamIsCapturing calls. The "0.2% coverage" line round 38 printed
# was an analysis of a corpse, not a measurement.
#
# Two fixes here:
#   1. FULL console capture per attempt (tee, no tail). A failure with a
#      discarded error is the exact silent-truncation this repo bans.
#   2. Two attempts, run UNCONDITIONALLY: graph mode (production reality,
#      and the mode that died last time -- its error is now capturable) and
#      enforce_eager (no graph capture to trip over, and an upper bound on
#      launch overhead: eager pays per-kernel launch cost that graph replay
#      amortises, so eager coverage <= graph coverage).
#
# Success is defined by the PROFILE_MARK line printed AFTER generate()
# returns, not by exit codes alone -- rc=0 with no mark means the trace is
# another corpse and is reported as such.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 38c: rocprofv3 decode decomposition, graph + eager ==="

analyze() {  # $1 = prof dir, $2 = label
    PROF_DIR="$1" PROF_LABEL="$2" python3 - <<'PY'
import csv, glob, os
d, label = os.environ["PROF_DIR"], os.environ["PROF_LABEL"]
traces = sorted(glob.glob(f"{d}/**/*kernel_trace*.csv", recursive=True))
print(f"--- analysis: {label} ---")
if not traces:
    print("  NO TRACE FILES"); raise SystemExit
iv, names = [], {}
for t in traces:
    with open(t) as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames or []
        sc = next((c for c in cols if "Start_Timestamp" in c), None)
        ec = next((c for c in cols if "End_Timestamp" in c), None)
        kc = next((c for c in cols if "Kernel_Name" in c), None)
        if not (sc and ec):
            print(f"  {t}: unrecognised columns"); continue
        for row in rd:
            try:
                s, e = int(row[sc]), int(row[ec])
            except (KeyError, ValueError):
                continue
            iv.append((s, e))
            k = row.get(kc, "?") if kc else "?"
            names[k] = names.get(k, 0) + (e - s)
if not iv:
    print("  no kernel records parsed"); raise SystemExit
iv.sort()
t_end = max(e for _, e in iv)
W = 8_000_000_000
w0 = t_end - W
clipped = [(max(s, w0), e) for s, e in iv if e > w0]
merged, cs, ce = [], None, None
for s, e in sorted(clipped):
    if cs is None: cs, ce = s, e
    elif s <= ce: ce = max(ce, e)
    else: merged.append((cs, ce)); cs, ce = s, e
if cs is not None: merged.append((cs, ce))
busy = sum(e - s for s, e in merged)
print(f"  total kernel records: {len(iv)}")
print(f"  8 s decode tail: kernels {len(clipped)}, coverage {100*busy/W:.1f}%, "
      f"gap fraction {100*(1-busy/W):.1f}%, launches/s {len(clipped)/8.0:,.0f}")
print("  top kernels by total time (whole trace):")
for k, ns in sorted(names.items(), key=lambda kv: -kv[1])[:10]:
    print(f"    {ns/1e6:10.1f} ms  {k[:90]}")
PY
}

run_profile() {  # $1 = label, $2 = eager 0|1
    local label="$1" eager="$2"
    local prof="$LOGS/$label"
    rm -rf "$prof"; mkdir -p "$prof"
    echo ""
    echo "=== $(date -u +%T) attempt: $label (enforce_eager=$eager) ==="
    docker run --rm --name "probe-$label" \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
      --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
      -v /mnt/llm-storage:/models -v "$prof":/prof \
      -e HSA_NO_SCRATCH_RECLAIM=1 -e GPU_MAX_HW_QUEUES=4 \
      -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 \
      -e VLLM_PREFER_AITER_FA=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -e PROF_EAGER="$eager" \
      --entrypoint bash vllm-mi210:v0.26.1rc0 -c '
set -e
SDK=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel
cat > /tmp/decode_trace.py <<'"'"'PY'"'"'
import os
import time
import traceback
from vllm import LLM, SamplingParams

try:
    llm = LLM(model="/models/bench-matrix/t35-w8a8", max_model_len=32768,
              gpu_memory_utilization=0.90, tensor_parallel_size=1,
              enable_prefix_caching=False, seed=1234,
              enforce_eager=os.environ.get("PROF_EAGER") == "1")
    tok = llm.get_tokenizer()
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 1400)[:8000]
    prompt = tok.decode(ids)
    sp = SamplingParams(max_tokens=1024, temperature=0.0, ignore_eos=True)
    t0 = time.time()
    out = llm.generate([prompt], sp)
    t1 = time.time()
    n = len(out[0].outputs[0].token_ids)
    print(f"PROFILE_MARK decode_tokens={n} wall_s={t1-t0:.2f}", flush=True)
except Exception:
    traceback.print_exc()
    raise
PY
"$SDK/bin/rocprofv3" --kernel-trace --hip-trace --output-format csv \
    -d /prof -o "$HOSTNAME" -- python3 /tmp/decode_trace.py
' 2>&1 | tee "$LOGS/$label.console" | grep -E --line-buffered \
        "PROFILE_MARK|Error|error|Traceback|Exception|abort|Capturing|graph" | tail -20
    local rc=${PIPESTATUS[0]}
    if grep -q "PROFILE_MARK" "$LOGS/$label.console"; then
        echo "attempt $label: generate() completed ($(grep -o 'PROFILE_MARK.*' "$LOGS/$label.console"))"
        analyze "$prof" "$label"
    else
        echo "attempt $label FAILED (rc=$rc, no PROFILE_MARK). Full log: logs/$label.console; last 25 lines:"
        tail -25 "$LOGS/$label.console" | sed 's/^/    /'
    fi
}

run_profile rd38c-graph 0
run_profile rd38c-eager 1

echo ""
echo "=== $(date -u +%T) round 38c done ==="
echo "READING THIS. Graph-mode coverage is production reality; eager coverage"
echo "is the launch-cost upper bound (per-kernel launches, no replay"
echo "amortisation). Coverage near 100% in graph mode puts the decode gap"
echo "in-kernel (docs/30); a large gap fraction means launch/CPU overhead"
echo "survives even graph replay. If graph mode failed again, the full error"
echo "is now in logs/rd38c-graph.console -- read it before drawing anything."
