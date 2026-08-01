#!/usr/bin/env bash
# Round 41: WHY is the CK GEMM worth 1.48x end to end when the kernel is only
# worth ~1.10x?
#
# Round 40 measured, one image / one variable / correctness probe passing on
# both arms, decode 55.71 -> 82.48 t/s = 1.480x, dead steady across three
# reps (55.7/55.7/55.7 vs 82.4/82.5/82.5), same prompt sizes, TTFT
# unchanged. Prefill barely moved (1.011-1.017x). So the effect is
# decode-specific and real.
#
# It is also LARGER THAN ITS STATED CAUSE, which is a reason to distrust the
# explanation, not the measurement. Microbenchmarks at the shapes actually
# served (TP=2: qkv K=2048 N=2560, o_proj K=2048 N=2048) put CK at
# 1.14-1.34x over Triton, and scaled_mm was 44% of decode kernel time, so
# Amdahl predicts ~1.10x. The first hypothesis -- that TP=2 shapes favour CK
# more than the TP=1 shapes the earlier probe used -- was tested and is
# FALSE: CK's edge is SMALLER at TP=2 (o_proj 1.516x -> 1.162x).
#
# So something other than the GEMM changed when VLLM_ROCM_USE_AITER_LINEAR
# went to 1. Candidates this round can distinguish:
#   - register_ops_once() now registers EVERY aiter op, not just the GEMM,
#     so a fusion pass or an ir_op_priority entry (KernelConfig lists
#     rms_norm/fused_add_rms_norm as ['aiter','native']) may now resolve to
#     an aiter op that previously fell through to native;
#   - the CK path may fold quant/scale work into its epilogue, deleting a
#     separate kernel (dynamic_scaled_int8_quant was 2.9% of decode);
#   - something about graph capture differs between the two paths.
#
# Method: the round 38e harness, run TWICE on the CK image with
# VLLM_ROCM_USE_AITER_LINEAR 0 then 1, and DIFF the decode-window kernel
# tables. Same warm-then-profile structure (rocprofv3 poisons subprocess
# stdout on gfx90a; see 38e for that whole saga) and the same two-marker
# success rule.
#
# NOTE ON TP. This profiles at TP=1 while round 40 served TP=2, because
# tracing wants one process. That is a real difference and the run reports
# its own TP=1 end-to-end factor from the PROFILE_MARK wall times: if TP=1
# also shows ~1.48x the kernel diff explains round 40 directly; if TP=1
# shows ~1.10x then the extra win is TP=2-specific and the kernel table is
# only part of the answer. Either outcome is informative and neither is
# assumed.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 43: rocprofv3 decode decomposition, graph + eager ==="

# The ephemeral AITER fix, generated host-side and bind-mounted read-only so
# the container script needs no nested quoting (a draft that inlined this
# python inside the docker -c single-quoted string broke the outer quoting on
# its own string literals -- caught in review before it shipped). Applied
# inside each --rm container; the image on disk is never modified.
cat > "$LOGS/rd43-aiter-patch.py" <<'EOF'
import pathlib

p = pathlib.Path(
    "/opt/python/lib/python3.14/site-packages/aiter/jit/utils/cpp_extension.py")
s = p.read_text()
old = '    ROCM_VERSION = tuple(int(v) for v in HIP_VERSION.split(".")[:2])'
new = ('    ROCM_VERSION = tuple(int(v) for v in __import__("re")'
       r'.search(r"(\d+)\.(\d+)", HIP_VERSION).groups())')
n = s.count(old)
assert n == 1, f"AITER version-parse anchor matched {n} times, expected 1"
p.write_text(s.replace(old, new))
print("AITER version-parse patch applied (container-local, ephemeral)")
EOF

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

run_profile() {  # $1 = label, $2 = VLLM_ROCM_USE_AITER_LINEAR 0|1
    local label="$1" eager="$2"
    local prof="$LOGS/$label"
    rm -rf "$prof"; mkdir -p "$prof"
    echo ""
    echo "=== $(date -u +%T) attempt: $label (AITER_MOE=$eager) ==="
    docker run --rm --name "probe-$label" \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
      --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
      -v /mnt/llm-storage:/models -v "$prof":/prof \
      -v "$LOGS/rd43-aiter-patch.py":/tmp/aiter_patch.py:ro \
      -e HSA_NO_SCRATCH_RECLAIM=1 -e GPU_MAX_HW_QUEUES=4 \
      -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 \
      -e VLLM_PREFER_AITER_FA=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      -e PROF_EAGER=0 \
      -e VLLM_ROCM_USE_AITER=1 \
      -e VLLM_ROCM_USE_AITER_LINEAR=1 \
      -e VLLM_ROCM_USE_AITER_MOE="$eager" \
      --entrypoint bash vllm-mi210:aiterops -c '
set -e
SDK=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel
python3 /tmp/aiter_patch.py
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
echo "=== WARMUP RUN (no profiler) -- JIT builds and caches populate cleanly ==="
python3 /tmp/decode_trace.py
echo "=== PROFILED RUN (rocprofv3) -- warm caches, no builds ==="
"$SDK/bin/rocprofv3" --kernel-trace --hip-trace --output-format csv \
    -d /prof -o "$HOSTNAME" -- python3 /tmp/decode_trace.py
' 2>&1 | tee "$LOGS/$label.console" | grep -E --line-buffered \
        "PROFILE_MARK|Error|error|Traceback|Exception|abort|Capturing|graph" | tail -20
    local rc=${PIPESTATUS[0]}
    # Two marks required: warmup's AND the profiled run's. A single mark means
    # the warmup succeeded and the profiled run died -- analyzing that trace
    # would repeat round 38's analysis-of-a-corpse mistake.
    local marks
    marks=$(grep -c "PROFILE_MARK" "$LOGS/$label.console" 2>/dev/null)
    marks=${marks:-0}
    if [ "$marks" -ge 2 ]; then
        echo "attempt $label: warmup + profiled run both completed:"
        grep -o 'PROFILE_MARK.*' "$LOGS/$label.console" | sed 's/^/    /'
        analyze "$prof" "$label"
    else
        echo "attempt $label FAILED (rc=$rc, PROFILE_MARK count=$marks, need 2)."
        echo "Full log: logs/$label.console; last 25 lines:"
        tail -25 "$LOGS/$label.console" | sed 's/^/    /'
    fi
}

run_profile rd43-moeoff 0
run_profile rd43-moeon  1

echo ""
echo "=== $(date -u +%T) round 43 done ==="

# The whole point: what MOVED between the two runs, per kernel.
LOGS="$LOGS" python3 - <<'PYDIFF'
import csv, glob, os
from collections import defaultdict
L = os.environ["LOGS"]


def decode_table(d):
    tr = sorted(glob.glob(f"{d}/**/*kernel_trace*.csv", recursive=True))
    iv = []
    for t in tr:
        with open(t) as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            sc = next((c for c in cols if "Start_Timestamp" in c), None)
            ec = next((c for c in cols if "End_Timestamp" in c), None)
            kc = next((c for c in cols if "Kernel_Name" in c), None)
            if not (sc and ec):
                continue
            for r in rd:
                try:
                    st, en = int(r[sc]), int(r[ec])
                except Exception:
                    continue
                iv.append((st, en, r.get(kc, "?") if kc else "?"))
    if not iv:
        return None, 0.0
    end = max(e for _, e, _ in iv)
    W = 8_000_000_000
    w0 = end - W
    per = defaultdict(int)
    tot = 0
    for st, en, k in iv:
        if en <= w0:
            continue
        dur = en - max(st, w0)
        per[k] += dur
        tot += dur
    return per, tot


a, atot = decode_table(f"{L}/rd43-moeoff")
b, btot = decode_table(f"{L}/rd43-moeon")
if not a or not b:
    print("  missing one or both traces -- cannot diff")
    raise SystemExit


def short(k):
    return k.split("(")[0][:52]


print(f"decode-window busy: MoE-off {atot/1e9:.3f} s, MoE-on {btot/1e9:.3f} s "
      "(8 s window each)")
print()
print(f"{'kernel':<54}{'MoE-off ms':>12}{'MoE-on ms':>11}{'delta ms':>10}")
print("-" * 85)
keys = set(a) | set(b)
for k in sorted(keys, key=lambda k: -abs(a.get(k, 0) - b.get(k, 0)))[:14]:
    av, bv = a.get(k, 0) / 1e6, b.get(k, 0) / 1e6
    mark = "  <-- NEW with MoE" if k not in a else (
        "  <-- GONE with MoE" if k not in b else "")
    print(f"{short(k):<54}{av:11.1f}{bv:10.1f}{bv-av:+10.1f}{mark}")
print()
print(f"total decode-window kernel time: {atot/1e6:.1f} ms -> {btot/1e6:.1f} ms")
PYDIFF

echo ""
echo "TP=1 END-TO-END, from the PROFILE_MARK wall times:"
for lab in rd43-moeoff rd43-moeon; do
    w=$(grep -o "PROFILE_MARK decode_tokens=[0-9]* wall_s=[0-9.]*" \
        "$LOGS/$lab.console" 2>/dev/null | tail -1)
    echo "  $lab: ${w:-<no mark>}"
done
echo ""
echo "READING THIS. If the TP=1 wall times differ by ~1.48x, round 40's win is"
echo "explained by whatever the kernel diff shows. If they differ by only"
echo "~1.10x, the GEMM accounts for TP=1 and the rest of round 40's win is"
echo "TP=2-specific -- look at collectives next, not kernels."
