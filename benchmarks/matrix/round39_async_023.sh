#!/usr/bin/env bash
# Round 39: the async ON/OFF pair on 0.23.1 -- the measurement that decides
# whether docs/42's "masked regression" claim survives.
#
# docs/42 SS2 claimed 0.26's synchronous decode path regressed ~6-8% vs
# 0.23.1, masked by default-on async scheduling. That arithmetic compared
# 0.23.1 WITH async (round 36's arm -- its serverlog says "Asynchronous
# scheduling is enabled", verified 2026-08-01) against 0.26 WITHOUT async
# (rd38-noasync). Apples to oranges: async_scheduling exists in v0.23.1rc0
# with the same bool|None=None default and the same resolution to enabled.
# Round 36 was async-on across ALL THREE versions, so its +2.7% is the
# honest like-for-like number and no "masking" was demonstrated.
#
# What was never measured is 0.23.1 with async OFF. That number picks
# between two worlds:
#
#   0.23.1 ON/OFF ~= 1.11 (like 0.26's): async is worth the same on both
#     versions, the sync paths moved together, docs/42 SS2 is retracted in
#     full -- there is no regression, masked or otherwise.
#   0.23.1 ON/OFF ~= 1.00: async gained value in 0.26 because the sync path
#     under it got slower; a sync-path regression is real after all, just
#     established by a different comparison than docs/42 used.
#
# Two arms, one image (vllm-mi210:latest = 0.23.1), pins identical to rounds
# 38/38b so the 0.26 pair (56.58 / 50.96) is directly comparable. Each arm
# asserts its async state from the serverlog -- the identical-arms defect
# class does not get a fifth sighting.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 39: async ON/OFF on 0.23.1 ==="

export VLLM_IMAGE=vllm-mi210:latest
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export TP=2
export READY_TIMEOUT=900
export LONGCTX_TOKENS=27852

run_one() {  # label want_async(on|off) extra-args...
    local label="$1" want="$2"; shift 2
    echo ""
    echo "=== $(date -u +%T) arm: $label (async should be $want) ==="
    "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" "$@" 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    local n has
    has=$(grep -c "Asynchronous scheduling is enabled" "$LOGS/$label.serverlog" 2>/dev/null)
    has=${has:-0}
    if { [ "$want" = "on" ] && [ "$has" -eq 0 ]; } || \
       { [ "$want" = "off" ] && [ "$has" -gt 0 ]; }; then
        echo "ARM INVALID: $label expected async $want, serverlog says enabled-count=$has."
        echo "Its numbers are not a measurement of the intended configuration."
        return 1
    fi
    echo "verified: async is $want (serverlog enabled-count=$has)"
    n=$(grep -c "LoadKernel" "$LOGS/$label.serverlog" 2>/dev/null)
    n=${n:-0}
    echo "arm $label rc=$rc  ASM code objects loaded: $n"
}

run_one rd39-023-base    on  --max-model-len 131072
run_one rd39-023-noasync off --max-model-len 131072 --no-async-scheduling

echo ""
echo "=== $(date -u +%T) round 39 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
def get(a, wl, key):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(key) if os.path.isfile(f) else None

pairs = [("0.23.1", "rd39-023-base", "rd39-023-noasync"),
         ("0.26.1rc0", "rd38-base", "rd38-noasync")]
print(f"{'version':<10} {'async ON':>10} {'async OFF':>10} {'ON/OFF':>8}   (longctx decode t/s)")
print("-" * 52)
vals = {}
for name, on_arm, off_arm in pairs:
    on = get(on_arm, "longctx", "decode_tps_median")
    off = get(off_arm, "longctx", "decode_tps_median")
    vals[name] = (on, off)
    fac = f"{on/off:7.3f}x" if on and off else "      -"
    o = f"{on:10.2f}" if on else f"{'-':>10}"
    f_ = f"{off:10.2f}" if off else f"{'-':>10}"
    print(f"{name:<10} {o} {f_} {fac}")

on23, off23 = vals.get("0.23.1", (None, None))
on26, off26 = vals.get("0.26.1rc0", (None, None))
if all([on23, off23, on26, off26]):
    print()
    print(f"sync-path (async OFF) 0.23.1 -> 0.26.1rc0: {off26/off23:.3f}x")
    print(f"async-on             0.23.1 -> 0.26.1rc0: {on26/on23:.3f}x")
    print()
    print("If the two lines above are similar, the versions moved together and")
    print("docs/42 SS2's 'masked regression' is retracted in full. If the sync")
    print("line is well below the async line, a sync-path regression is real.")
PY
