#!/usr/bin/env bash
# CPU-offload PREFILL tuning for the EPYC 74F3, on a model big enough to need it.
#
# WHY THIS IS UNTESTED GROUND. Every llama.cpp arm so far ran on defaults for
# threading -- serve_llamacpp.sh passes --no-mmap and nothing else. In a
# container seeing 48 logical CPUs, llama.cpp will pick ~48 threads, which for
# memory-bound GEMM on Zen3 is usually WORSE than 24: SMT siblings share the
# L2 and the FP pipes, so the second thread per core mostly adds contention.
#
# THE HARDWARE ARGUES FOR TESTING IT. EPYC 74F3 is the frequency-optimised SKU:
# 24 cores spread over 8 CCDs, each CCD keeping its full 32 MB L3. That is
# 256 MB total and ~10.6 MB per core, which is a lot of room for expert tiles --
# but only if threads are placed so they are not thrashing each other's L3.
# AVX2 only, no AVX-512, so per-core throughput is fixed and placement is the
# lever.
#
# PLACEMENT MATTERS FOR THE TEST ITSELF. Auto-fit puts 135.57 of 139 GB on the
# GPUs, leaving ~3 GiB on CPU -- that barely exercises the CPU path at all. So
# these arms pin --n-cpu-moe 60, which is the smallest value that fits (30 and
# 45 OOM) and gives real CPU-side work. Baseline at that placement: 149 t/s
# prefill, 101.5 s TTFT.
#
# -ub IS RE-SWEPT ON PURPOSE. The existing -ub 2048 optimum was measured with
# everything resident on GPU. With 60 expert layers on CPU the bottleneck moves,
# and the best micro-batch for a GPU pipeline is not obviously the best for a
# CPU one -- larger batches amortise CPU thread dispatch but blow the L3 tiles.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/glm-gguf-iq3xs
NC=60
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting CPU-prefill tuning (n-cpu-moe=$NC) ==="

run() {  # run <label> <extra args...>
    local label="$1"; shift
    echo "--- $label : $* ---"
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "$label" 400B iq3_xs llamacpp "$MODEL" \
        --ctx-size 32768 --flash-attn on -ngl 999 --n-cpu-moe "$NC" "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. thread count -------------------------------------------------------
# -t is generation (decode) threads, -tb is batch threads -- and BATCH is what
# prefill uses, so -tb is the flag that should matter most here. Testing them
# separately rather than together, because the right answer for a latency-bound
# decode step and a throughput-bound prefill step is often not the same number.
run "glm-cpu-t24tb24"  -ub 2048 -t 24 -tb 24     # physical cores only
run "glm-cpu-t48tb48"  -ub 2048 -t 48 -tb 48     # all SMT threads
run "glm-cpu-t24tb48"  -ub 2048 -t 24 -tb 48     # SMT for prefill, cores for decode

# --- B. micro-batch, at the best thread setting found above -----------------
BEST=$(python3 - <<'PY'
import glob, json, os
best, bp = None, -1
for lab in ("glm-cpu-t24tb24", "glm-cpu-t48tb48", "glm-cpu-t24tb48"):
    f = f"/mnt/llm-storage/bench-matrix/results/{lab}-cold16k.json"
    if not os.path.exists(f):
        continue
    p = json.load(open(f)).get("implied_prefill_tps_median") or 0
    if p > bp:
        best, bp = lab, p
print(best or "glm-cpu-t24tb24")
PY
)
case "$BEST" in
    glm-cpu-t24tb24) TF="-t 24 -tb 24" ;;
    glm-cpu-t48tb48) TF="-t 48 -tb 48" ;;
    *)               TF="-t 24 -tb 48" ;;
esac
echo "=== best thread config: $BEST ($TF) -- sweeping -ub against it ==="
run "glm-cpu-ub1024" -ub 1024 $TF
run "glm-cpu-ub4096" -ub 4096 $TF

echo "=== $(date -u +%T) round 13 complete ==="
python3 - <<'PY'
import glob, json, os
print("  CPU-offload prefill (n-cpu-moe=60; baseline 149 t/s, 101.5 s TTFT):")
for f in sorted(glob.glob("/mnt/llm-storage/bench-matrix/results/glm-cpu-*-cold16k.json")):
    d = json.load(open(f))
    print(f"    {os.path.basename(f)[:-13]:22} ttft={d['ttft_s_median']:7.1f}s "
          f"prefill={d['implied_prefill_tps_median']:6.0f}")
PY
