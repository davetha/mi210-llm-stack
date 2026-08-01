#!/usr/bin/env bash
# Does three minor versions of upstream vLLM buy anything on gfx90a?
#
# Everything in docs/28 and docs/37 was measured on 0.23.1. Two source-built
# images now exist and neither has a single benchmark against it. This is the
# largest unmeasured variable in the repo, and the answer is unknown in BOTH
# directions -- newer is not automatically faster, and a regression would be
# just as important to find.
#
#   vllm-mi210:latest       0.23.1 + patches (AMD-supported base)
#   vllm-mi210:v0.26.0      source build, needs our int8 patch
#   vllm-mi210:v0.26.1rc0   source build, upstream int8 fix, 300 commits ahead
#
# Model is the 30B W8A8 -- the checkpoint docs/28 characterises most thoroughly,
# and small enough (30 GB) that three images x two workloads stays inside an
# hour. Nothing here needs an 80B to answer the question.
#
# AITER MUST ACTUALLY RUN, AND THIS ROUND CHECKS IT -----------------------------
#
# Round 32's arms all selected ROCM_ATTN and loaded ZERO gfx90a code objects,
# because the image of the day was missing prefer_aiter_fa_gfx90a.py. Its AITER
# rows measured nothing and had to be reported as inconclusive.
#
# All three images now carry that patch (verify-gfx90a passes on each), so
# VLLM_PREFER_AITER_FA=1 should put ROCM_AITER_FA ahead of ROCM_ATTN and the ASM
# kernels should load. That makes this ALSO the first sweep where the AITER
# carve-out is exercised end to end -- so every arm asserts on LoadKernel rather
# than assuming it. An arm with zero ASM loads is reported as such and its
# numbers are not comparable to one that had them.
#
# ONE VARIABLE. P2P is pinned, the tuned MoE config is off (round 34 measured it
# as a 20% regression), the model and flags are identical. Only the image moves.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 36: vLLM version sweep, patches held constant ==="

IMAGES="latest v0.26.0 v0.26.1rc0"
for t in $IMAGES; do
    docker image inspect "vllm-mi210:$t" >/dev/null 2>&1 \
      || { echo "FATAL: missing image vllm-mi210:$t"; exit 1; }
done

# Pinned so the only difference between arms is the vLLM version. Round 32
# nearly credited the patches with P2P's +11.2% by leaving this inconsistent.
export NCCL_P2P_DISABLE=1
# Round 34 measured the tuned MoE config at 0.79x prefill; keep it out.
export VLLM_TUNED_CONFIG_FOLDER=
# The whole point: AITER FA ahead of ROCM_ATTN. Without this the images are
# indistinguishable on the attention path and the sweep measures less than it
# appears to.
export VLLM_PREFER_AITER_FA=1
export TP=2
export READY_TIMEOUT=900

for tag in $IMAGES; do
    label="rd36-w8a8-${tag//./}"
    echo ""
    echo "=== $(date -u +%T) arm: vllm-mi210:$tag ==="
    export VLLM_IMAGE="vllm-mi210:$tag"
    "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 32768 2>&1 | tail -14
    echo "arm $label rc=${PIPESTATUS[0]}"

    # Proof the ASM path ran, per arm. Recorded either way -- a zero here does
    # not fail the round, it changes what the numbers mean.
    log="$BASE/logs/$label.serverlog"
    n=$(grep -c "LoadKernel" "$log" 2>/dev/null || echo 0)
    b=$(grep -ohE "Overriding with [A-Z_]+" "$log" 2>/dev/null | sort -u | tr '\n' ' ')
    echo "  ASM code objects loaded: $n   backend: ${b:-unknown}"
    [ "$n" -gt 0 ] || echo "  WARNING: no ASM loaded -- not comparable to an arm that had it"
done

echo ""
echo "=== $(date -u +%T) round 36 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
tags = [("latest", "0.23.1"), ("v0260", "0.26.0"), ("v0261rc0", "0.26.1rc0")]
rows = [("cold16k", "implied_prefill_tps_median", "prefill"),
        ("longctx", "implied_prefill_tps_median", "prefill"),
        ("longctx", "decode_tps_median", "decode"),
        ("longctx", "ttft_s_median", "ttft")]
print(f"{'workload':<9} {'metric':<8} " + "".join(f"{n:>12}" for _, n in tags) + f"{'vs 0.23':>10}")
print("-" * 62)
for wl, key, name in rows:
    vals = []
    for t, _ in tags:
        f = os.path.join(R, f"rd36-w8a8-{t}-{wl}.json")
        vals.append(json.load(open(f)).get(key) if os.path.isfile(f) else None)
    if all(v is None for v in vals):
        continue
    cells = "".join(f"{v:12.2f}" if isinstance(v, (int, float)) else f"{'-':>12}" for v in vals)
    base, last = vals[0], vals[-1]
    fac = f"{last/base:>9.3f}x" if isinstance(base, (int, float)) and isinstance(last, (int, float)) and base else "         -"
    print(f"{wl:<9} {name:<8} {cells}{fac}")
PY

echo ""
echo "READING THIS. The vs-0.23 column is 0.26.1rc0 over 0.23.1; for TTFT it"
echo "inverts, since lower is better. Check the per-arm 'ASM code objects"
echo "loaded' lines above BEFORE comparing anything -- an arm that ran"
echo "ROCM_ATTN is not measuring the same code path as one that ran AITER FA."
