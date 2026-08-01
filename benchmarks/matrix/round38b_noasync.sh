#!/usr/bin/env bash
# Round 38b: the async-scheduling A/B that round 38 only appeared to run.
#
# Round 38's rd38-async arm passed --async-scheduling and measured no change
# against rd38-base. Verification of the serverlogs shows why: 0.26.1rc0
# enables async scheduling BY DEFAULT -- rd38-base logs "Asynchronous
# scheduling is enabled" with no flag passed. The two arms were identical
# configurations, and that "null" was a vacuous A/B, not a measured one. Same
# defect class as rounds 31/32/37: two arms silently identical, producing a
# plausible number.
#
# The informative direction is the inverse. This arm serves with
# --no-async-scheduling (argparse.BooleanOptionalAction generates the negative
# form; verified against arg_utils.py:347 in this image) and compares against
# rd38-base, which is the async-ON measurement. base/noasync is then what
# async scheduling is worth on this box at concurrency 1.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work (round 38 ahead in queue) ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 38b: --no-async-scheduling vs rd38-base ==="

# Identical pins to round 38 -- this arm must be comparable to rd38-base.
export VLLM_IMAGE=vllm-mi210:v0.26.1rc0
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export TP=2
export READY_TIMEOUT=900
export LONGCTX_TOKENS=27852

"$BIN/run_arm.sh" rd38-noasync 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
    --max-model-len 131072 --no-async-scheduling 2>&1 | tail -14
rc=${PIPESTATUS[0]}

# The check round 38 lacked: prove the arm actually differs from base. If this
# serverlog still says async is enabled, the flag did not take and the arm is
# NOT a measurement of anything.
if grep -q "Asynchronous scheduling is enabled" "$LOGS/rd38-noasync.serverlog" 2>/dev/null; then
    echo "ARM INVALID: rd38-noasync serverlog says async scheduling is STILL enabled."
    echo "The flag did not take; this arm is identical to base and its numbers"
    echo "must not be recorded as an A/B."
    exit 1
fi
echo "verified: no 'Asynchronous scheduling is enabled' line in serverlog"

n=$(grep -c "LoadKernel" "$LOGS/rd38-noasync.serverlog" 2>/dev/null)
n=${n:-0}
echo "arm rd38-noasync rc=$rc  ASM code objects loaded: $n"

echo ""
echo "=== $(date -u +%T) round 38b done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
def get(a, wl, key):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(key) if os.path.isfile(f) else None
rows = [("cold16k", "implied_prefill_tps_median", "prefill"),
        ("cold16k", "ttft_s_median", "ttft"),
        ("longctx", "implied_prefill_tps_median", "prefill"),
        ("longctx", "decode_tps_median", "decode"),
        ("longctx", "ttft_s_median", "ttft")]
print(f"{'workload':<9} {'metric':<8} {'async-ON(base)':>15} {'async-OFF':>12} {'ON/OFF':>9}")
print("-" * 58)
for wl, key, name in rows:
    on, off = get("rd38-base", wl, key), get("rd38-noasync", wl, key)
    if on is None and off is None:
        continue
    c1 = f"{on:15.2f}" if isinstance(on, (int, float)) else f"{'-':>15}"
    c2 = f"{off:12.2f}" if isinstance(off, (int, float)) else f"{'-':>12}"
    fac = f"{on/off:8.3f}x" if isinstance(on, (int, float)) and isinstance(off, (int, float)) and off else f"{'-':>9}"
    print(f"{wl:<9} {name:<8} {c1} {c2} {fac}")
print()
print("ON/OFF above 1.0 on throughput rows (or below 1.0 on ttft rows) is what")
print("default-on async scheduling is worth at concurrency 1 on this box.")
PY
