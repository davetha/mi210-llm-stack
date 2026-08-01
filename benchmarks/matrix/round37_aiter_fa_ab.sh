#!/usr/bin/env bash
# What is the AITER ASM flash-attention carve-out actually worth on this box?
#
# This repo has never cleanly measured it. Two prior attempts:
#
#   benchmarks/vllm-aiter-asm-gfx90a.md  measured 1.23x -- but at 4096-token
#     prompts with concurrency >= 8, and explicitly 1.01-1.02x at concurrency 1.
#   round 32                             tried to isolate it and measured
#     NOTHING: both arms selected ROCM_ATTN and loaded zero ASM objects,
#     because the image of the day lacked prefer_aiter_fa_gfx90a.py.
#
# Then round 36 produced a number nobody was looking for. Its 0.23.1 arm ran
# ROCM_AITER_FA with 4 ASM objects loaded, against round 32's same-tagged image
# running ROCM_ATTN with none:
#
#   prefill @16k   5956.20 -> 7284.24   1.223x
#   prefill @25k   4668.94 -> 6197.00   1.327x
#   decode           51.06 ->   54.11   1.060x
#
# THAT COMPARISON IS NOT TRUSTWORTHY, which is why this round exists. It spans
# two rounds and the image was rebuilt between them -- gaining prefer_aiter_fa,
# the sharded_state guard, and a full recompile. Three changes, one number.
#
# It also contradicts our own archive. 1.22x at CONCURRENCY 1 is precisely where
# vllm-aiter-asm-gfx90a.md measured 1.01-1.02x. The reconciliation on offer is
# prompt size -- that null was at 4096 tokens and this is 16k, which is far more
# likely to saturate an MI210 -- but that is a hypothesis, and a cross-round
# delta is not the way to test it.
#
# ONE VARIABLE. VLLM_PREFER_AITER_FA is read by our patched rocm.py from
# os.environ directly, so flipping it moves ROCM_AITER_FA ahead of ROCM_ATTN and
# changes nothing else: same image, same weights, same flags, same session.
#
# Each arm asserts on the backend it actually got. An arm that intended AITER and
# logged zero LoadKernel lines is a failed arm, not a null result -- that is the
# exact mistake round 32 made.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
IMG=vllm-mi210:latest
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 37: AITER FA on/off, one image, one session ==="

export NCCL_P2P_DISABLE=1
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_IMAGE="$IMG"
export TP=2
export READY_TIMEOUT=900

for pref in 1 0; do
    name=$([ "$pref" = 1 ] && echo aiterfa || echo rocmattn)
    label="rd37-w8a8-$name"
    echo ""
    echo "=== $(date -u +%T) arm: VLLM_PREFER_AITER_FA=$pref ($name) ==="
    export VLLM_PREFER_AITER_FA=$pref
    "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 32768 2>&1 | tail -14
    rc=${PIPESTATUS[0]}
    echo "arm $label rc=$rc"

    log="$BASE/logs/$label.serverlog"
    n=$(grep -c "LoadKernel" "$log" 2>/dev/null || echo 0)
    b=$(grep -ohE "Overriding with [A-Z_]+" "$log" 2>/dev/null | sort -u | tr '\n' ' ')
    echo "  ASM objects: $n   backend: ${b:-unknown}"
    if [ "$pref" = 1 ] && [ "$n" -eq 0 ]; then
        echo "  FATAL: asked for AITER FA and got none -- this arm measures nothing."
        exit 1
    fi
    if [ "$pref" = 0 ] && [ "$n" -gt 0 ]; then
        echo "  FATAL: control arm loaded ASM -- the two arms are not distinct."
        exit 1
    fi
done

echo ""
echo "=== $(date -u +%T) round 37 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
print(f"{'workload':<9} {'metric':<8} {'ROCM_ATTN':>11} {'AITER FA':>11} {'factor':>9}")
print("-" * 52)
for wl in ("cold16k", "longctx"):
    for key, name in (("implied_prefill_tps_median", "prefill"),
                      ("decode_tps_median", "decode"),
                      ("ttft_s_median", "ttft")):
        v = {}
        for arm in ("rocmattn", "aiterfa"):
            f = os.path.join(R, f"rd37-w8a8-{arm}-{wl}.json")
            v[arm] = json.load(open(f)).get(key) if os.path.isfile(f) else None
        a, b = v["rocmattn"], v["aiterfa"]
        if a is None and b is None:
            continue
        fac = f"{b/a:>8.3f}x" if isinstance(a,(int,float)) and isinstance(b,(int,float)) and a else "        -"
        fa = f"{a:11.2f}" if isinstance(a,(int,float)) else f"{'-':>11}"
        fb = f"{b:11.2f}" if isinstance(b,(int,float)) else f"{'-':>11}"
        print(f"{wl:<9} {name:<8} {fa} {fb} {fac}")
PY

echo ""
echo "READING THIS. TTFT inverts -- lower is better. If prefill lands near 1.22x"
echo "it corroborates round 36's cross-round delta and means the concurrency-1"
echo "null in vllm-aiter-asm-gfx90a.md does not hold at 16k prompts. If it lands"
echo "near 1.0x, that delta was the rebuild and not AITER, and round 36's"
echo "version numbers stand while its AITER inference does not."
