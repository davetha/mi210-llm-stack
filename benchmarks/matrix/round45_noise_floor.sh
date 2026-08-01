#!/usr/bin/env bash
# Round 45: how big is the noise floor, actually?
#
# Every A/B in this repo compares two single arms and calls a small delta a
# result. Round 44 showed that is shakier than assumed: the SAME image with the
# SAME config measured decode at
#
#   82.48 (rd40-ck)   82.92 (rd42-baseline)   85.16 (rd44-nopad)
#
# a 3.2% spread, while prefill and TTFT from those same runs reproduced to
# ~0.1%. So decode is the noisy metric and the noise is the same size as
# several published effects (the 1.035x version climb, the 0.977x AITER MoE,
# most of the "null" arms).
#
# This round runs the identical configuration N times -- separate server
# launches, exactly as a real A/B arm does -- and reports mean, sample stdev,
# min/max and a 95% interval per metric. The output is the number every past
# and future single-arm comparison has to be read against.
#
# WHY BETWEEN-ARM, NOT WITHIN-ARM. bench_matrix.py already does 3 reps inside
# one server and takes the median, so within-server jitter is handled. What is
# NOT handled is variation between server launches -- allocator layout, graph
# capture, expert placement, page mapping. That is the variance an A/B is
# actually exposed to, because each arm is its own launch.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

REPS="${REPS:-5}"
IMAGE=vllm-mi210:aiterops
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: missing $IMAGE"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 45: noise floor, ${REPS} identical arms ==="

# Pinned to the current best-known-good serving configuration, so the interval
# this produces applies directly to the arms it will be used to judge.
export VLLM_IMAGE="$IMAGE"
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=0
export TP=2
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

for i in $(seq 1 "$REPS"); do
    echo ""
    echo "=== $(date -u +%T) identical arm $i/$REPS ==="
    "$BIN/run_arm.sh" "rd45-r$i" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 131072 2>&1 | tail -6
    echo "arm rd45-r$i rc=${PIPESTATUS[0]}"
done

echo ""
echo "=== $(date -u +%T) round 45 done ==="
REPS="$REPS" python3 - <<'PY'
import json, os, statistics as st
R = "/mnt/llm-storage/bench-matrix/results"
reps = int(os.environ["REPS"])

metrics = [("longctx", "decode_tps_median", "decode t/s"),
           ("longctx", "implied_prefill_tps_median", "longctx prefill"),
           ("longctx", "ttft_s_median", "longctx ttft"),
           ("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
           ("cold16k", "ttft_s_median", "cold16k ttft")]

print(f"{'metric':<17}{'n':>3}{'mean':>10}{'stdev':>9}{'cv%':>7}{'min':>10}{'max':>10}{'spread%':>9}")
print("-" * 75)
summary = {}
for wl, key, name in metrics:
    vals = []
    for i in range(1, reps + 1):
        f = os.path.join(R, f"rd45-r{i}-{wl}.json")
        if os.path.isfile(f):
            v = json.load(open(f)).get(key)
            if isinstance(v, (int, float)):
                vals.append(v)
    if len(vals) < 2:
        print(f"{name:<17}{len(vals):>3}   insufficient data")
        continue
    m = st.mean(vals); s = st.stdev(vals)
    cv = 100 * s / m if m else float("nan")
    spread = 100 * (max(vals) - min(vals)) / m if m else float("nan")
    summary[name] = (m, s, cv, spread)
    print(f"{name:<17}{len(vals):>3}{m:10.2f}{s:9.3f}{cv:7.2f}{min(vals):10.2f}{max(vals):10.2f}{spread:9.2f}")

print()
print("95% interval for a SINGLE arm (mean +/- 1.96 sd), and the smallest")
print("A/B ratio that clears it:")
for name, (m, s, cv, spread) in summary.items():
    lo, hi = m - 1.96 * s, m + 1.96 * s
    # Two independent arms: sd of the difference is sqrt(2)*s.
    min_ratio = 1 + 1.96 * (2 ** 0.5) * s / m if m else float("nan")
    print(f"  {name:<17} {lo:8.2f} .. {hi:8.2f}   ratio must exceed {min_ratio:.3f}x")

print()
print("READING THIS. The right-hand column is the bar a single-arm A/B has to")
print("clear to mean anything at 95% confidence, given two independent launches.")
print("Any past result below its metric's bar is indistinguishable from noise")
print("and should be re-stated as inconclusive, not as a small effect.")
PY
