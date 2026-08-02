#!/usr/bin/env bash
# Round 62: is steady-state decode bandwidth-bound or issue-bound? Counters.
#
# THE QUESTION, AND WHY IT DIRECTS EVERYTHING ELSE. docs/39 item 1c put the
# decode gap at ~3x against an achieved-bandwidth model. Round 38e proved the
# gap is IN-KERNEL (99.9% of decode wall-clock is inside kernels; launch/CPU is
# 0.1%). docs/30 argues the machine is ISSUE-bound rather than FLOP-bound. But
# nobody has measured achieved HBM bandwidth during steady-state decode, so
# "issue-bound" is an inference from instruction counts rather than a
# measurement of the alternative.
#
# It matters because it decides what future work is worth doing. If decode is
# running near its bandwidth ceiling, then every kernel-level lead in docs/50
# and docs/51 was doomed regardless of merit and the only remaining lever is
# moving fewer bytes (quantization, KV compression). If it is far from the
# ceiling and issue-limited, then fusion and instruction-count work is the
# right direction -- which is what docs/51's int8-fusion gap and the 209:1
# int4 unpack ratio both point at.
#
# WHY NOT rocprof-compute. docs/39 scoped this as `rocprof-compute --roof-only`
# (MI210 resolves to the MI200 target dir, metrics under gfx90a, L2-Fabric is
# block 17, speed-of-light block 1). That binary is NOT in this image and is
# NOT pip-installable ("No matching distribution found for rocprof-compute").
# Only rocprofv3 is present. The counters needed are available through it, so
# this round computes the two numbers that matter by hand rather than rendering
# a full roofline plot.
#
# TWO HAZARDS, BOTH ALREADY PAID FOR BY ROUND 38e. rocprofv3's preloaded tool
# prints "Streaming Performance Monitor (SPM) is not supported on gfx90a
# devices" onto the STDOUT OF EVERY SUBPROCESS the profiled process spawns:
#   38c died because AITER int()s the output of `hipconfig --version`.
#   38d died because the leaked literal "(" in "(SPM)" broke a JIT build's
#       shell command line.
# The structural fix, reused verbatim here: run the workload TWICE in one
# container -- once with NO profiler so every JIT build and graph capture lands
# in the container-local cache with clean stdout, then again under rocprofv3
# where everything is a cache hit and nothing parses a child's output. That is
# also the methodologically correct thing to profile: a warm system.
# The 38d version-parse patch is kept, because cpp_extension's module-level
# parse still runs at import time during the profiled pass, cache or no cache.
#
# WHAT IS COLLECTED, AND THE ARITHMETIC.
#   TCC_EA_RDREQ / TCC_EA_WRREQ  L2<->HBM requests. On CDNA2 a request is 64 B
#                                (32 B for the _32B variants), so bytes = 64 *
#                                requests -- the FETCH_SIZE/WRITE_SIZE basis.
#                                NOTE: --pmc takes RAW counter names. The
#                                `_sum` forms that `--list-avail` prints are
#                                DERIVED EXPRESSION names; passing those makes
#                                rocprofv3 die on SIGTERM with no CSVs, which
#                                is how the first attempt at this round failed.
#   GRBM_GUI_ACTIVE                      GPU-busy cycles -> seconds via sclk.
#   SQ_WAVES, SQ_INSTS_VALU              issue-side load.
#   SQ_INSTS_VALU_MFMA_MOPS_BF16         actual matrix work.
# Achieved GB/s = bytes / busy_seconds, compared against MI210's 1.6 TB/s peak.
# docs/20 measured 73% of peak as reachable for decode-shape GEMM, so ~1.17
# TB/s is the practical ceiling, not 1.6.
#
# SCLK IS NOT 1700 MHz. docs/51 measured 1235-1375 MHz under sustained load,
# power-capped at 200 W. Use a measured clock, not the boost number, or the
# derived seconds are wrong by ~25%.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 62: decode counters (bandwidth vs issue) ==="

# The 38d patch: AITER parses `hipconfig --version` with a bare int(), which
# rocprofv3's stdout pollution breaks. Made tolerant of leading noise.
cat > "$LOGS/rd62-aiter-patch.py" <<'EOF'
import pathlib
p = pathlib.Path("/opt/python/lib/python3.14/site-packages/aiter/jit/utils/cpp_extension.py")
s = p.read_text()
old = '    ROCM_VERSION = tuple(int(v) for v in HIP_VERSION.split(".")[:2])'
new = ('    ROCM_VERSION = tuple(int(v) for v in __import__("re")'
       r'.search(r"(\d+)\.(\d+)", HIP_VERSION).groups())')
n = s.count(old)
assert n == 1, f"AITER version-parse anchor matched {n} times, expected 1"
p.write_text(s.replace(old, new))
print("AITER version-parse patch applied (container-local, ephemeral)")
EOF

docker rm -f probe-rd62 >/dev/null 2>&1 || true
docker run --rm --name probe-rd62 \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
  -v "$BASE":"$BASE" -v "$LOGS":/logs \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_PREFER_AITER_FA=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  --entrypoint bash "$IMG" -c '
set -o pipefail
python3 /logs/rd62-aiter-patch.py || exit 1
PROF=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/rocprofv3

# Offline decode driver: TP=1 so one process owns one GPU and the counters are
# unambiguous. Steady-state only -- prefill is excluded by measuring a second
# generate() after a warm one.
cat > /tmp/decode.py <<"PY"
import os, time
from vllm import LLM, SamplingParams
llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=1,
          max_model_len=8192, gpu_memory_utilization=0.85,
          enforce_eager=False, seed=1234)
sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
prompt = "def fibonacci(n):" * 64
llm.generate([prompt], sp)                     # warm: prefill + graph capture
t0 = time.time()
out = llm.generate([prompt], sp)               # measured
dt = time.time() - t0
n = len(out[0].outputs[0].token_ids)
print(f"PROFILE_MARK tokens={n} seconds={dt:.3f} tok_s={n/dt:.2f}", flush=True)

# Graceful teardown. Attempt 2 died with rocprofv3 catching signal 15 as the
# vLLM process manager force-killed its children at interpreter exit, so this
# drops the LLM first. That fixed the signal but NOT the missing counters --
# attempt 3 exited cleanly (ENGINE_DOWN printed twice) and still produced zero
# CSVs. The real cause was found by testing --pmc on a trivial in-process
# matmul, which wrote counters fine: vLLM runs its GPU work in a SPAWNED
# EngineCore/Worker subprocess, and rocprofv3 traces the parent, so the kernels
# were never in the profiled process at all. Hence
# VLLM_ENABLE_V1_MULTIPROCESSING=0 on the container, which keeps the engine
# in-process. Keeping the graceful teardown regardless -- it is correct.
# NO APOSTROPHES IN THIS BLOCK: it lives inside a single-quoted bash -c.
del llm
import gc
gc.collect()
time.sleep(5)
print("ENGINE_DOWN", flush=True)
PY

echo "=== pass 1: NO profiler (JIT builds, graph capture land in cache) ==="
MODEL='"$MODEL"' python3 /tmp/decode.py 2>&1 | tail -30

echo "=== pass 2: rocprofv3 --pmc on the warm system ==="
MODEL='"$MODEL"' $PROF \
  --pmc GRBM_GUI_ACTIVE SQ_WAVES SQ_INSTS_VALU TCC_EA_RDREQ TCC_EA_WRREQ \
  -d /logs/rd62-prof -o counters --output-format csv \
  -- python3 /tmp/decode.py 2>&1 | tail -30
' 2>&1 | tee "$LOGS/round62-run.log"

echo ""
echo "=== $(date -u +%T) round 62 done ==="
marks=$(grep -ac "PROFILE_MARK" "$LOGS/round62-run.log" | head -1)
# grep -c EXITS 1 on zero matches. `$(grep -c ... || echo 0)` therefore
# yields the string "0\n0" and every [ -lt ] test below dies with
# "integer expected". docs/50 recorded this exact bug in round 55 and it
# was reintroduced here. Pipe through head -1; never `|| echo`.
echo "PROFILE_MARK lines: $marks  (need 2 -- warmup AND profiled run)"
if [ "${marks:-0}" -lt 2 ]; then
    echo "ONLY THE WARMUP SURVIVED. The profiled pass died -- almost certainly"
    echo "another consumer of a polluted child stdout, as in 38c/38d. The trace"
    echo "is not usable; find the new consumer before trusting any number."
fi
python3 - <<'PY'
import csv, glob, os
files = sorted(glob.glob("/mnt/llm-storage/bench-matrix/logs/rd62-prof/**/*counter*.csv", recursive=True))
print(f"counter files: {len(files)}")
if not files:
    print("  NO COUNTER FILES -- see the note above"); raise SystemExit
tot = {}
for f in files:
    with open(f) as fh:
        for row in csv.DictReader(fh):
            name = row.get("Counter_Name") or row.get("counter_name")
            val = row.get("Counter_Value") or row.get("counter_value")
            if not name or val is None: continue
            try: tot[name] = tot.get(name, 0) + float(val)
            except ValueError: pass
for k in sorted(tot):
    print(f"  {k:<24} {tot[k]:,.0f}")
# Raw counter names, NOT the _sum forms. --pmc collects raw counters, so the
# CSV carries "TCC_EA_RDREQ"; the _sum spellings are derived-expression
# names from --list-avail. Attempt 4 collected counters successfully and
# then printed no derivation because these lookups still said _sum and
# silently returned 0.
rd, wr = tot.get("TCC_EA_RDREQ", 0), tot.get("TCC_EA_WRREQ", 0)
busy = tot.get("GRBM_GUI_ACTIVE", 0)
if rd and busy:
    gb = (rd + wr) * 64 / 1e9
    for mhz in (1235, 1300, 1375):
        sec = busy / (mhz * 1e6)
        print(f"  @ {mhz} MHz: {sec*1000:8.2f} ms busy -> {gb/sec:8.1f} GB/s "
              f"({gb/sec/1170*100:5.1f}% of the 1.17 TB/s practical ceiling)")
    valu, waves = tot.get("SQ_INSTS_VALU", 0), tot.get("SQ_WAVES", 0)
    if valu and busy:
        ipc = valu / busy
        # wave64 on 104 CUs x 4 SIMDs: a wave64 VALU op occupies its 16-wide
        # SIMD for 4 cycles, so sustained issue is ~104 instructions/cycle.
        print(f"  VALU issue     : {ipc:8.2f} inst/cycle of ~104 peak "
              f"({ipc/104*100:5.1f}% of issue capacity)")
        print(f"  waves launched : {waves:,.0f}")
    print()
    print("  READ BOTH NUMBERS TOGETHER -- bandwidth alone cannot tell")
    print("  issue-bound from latency-bound:")
    print("    bandwidth ~90%+                -> bandwidth-bound; only moving")
    print("                                      fewer bytes helps.")
    print("    bandwidth low, issue ~high     -> issue-bound, per docs/30;")
    print("                                      fusion is the right direction.")
    print("    BOTH low                       -> latency/occupancy-bound: not")
    print("                                      enough concurrent work to keep")
    print("                                      either resource busy. The lever")
    print("                                      is more work per step, not")
    print("                                      faster kernels.")
    print()
    print("  CAVEAT: these counters span the WHOLE profiled process -- two")
    print("  prefills and two generates -- not steady-state decode alone. And")
    print("  the profiled pass runs ~5x slower than un-profiled, so rocprofv3")
    print("  distorts what it measures. Treat the order of magnitude as sound")
    print("  and the precise figure as not.")
PY
