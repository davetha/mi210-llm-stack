#!/usr/bin/env bash
# Round 50: was AITER MoE's 0.977x a dispatch-policy artifact?
#
# docs/45 measured VLLM_ROCM_USE_AITER_MOE=1 at 0.977x decode and closed the
# MoE path. That arm ran dispatch policy 0. vLLM documents the values:
#
#   0  default heuristic, tuned FOR LARGE BATCHES
#   1  always single-pass -- "may be preferred for low-concurrency decode"
#   2  always multi-pass  -- "+2-5% on Qwen3-Next" (upstream PR #39177)
#
# So the closing measurement used the large-batch heuristic on the least-batched
# workload there is. Worse, the env var had NO EFFECT at the time:
# get_moe_dispatch_policy() is @if_aiter_supported, so on gfx90a it returned
# None regardless of what was set. configs/enable_aiter_gdn_and_moe_policy_gfx90a.py
# carves it out; this round is the first time the setting can do anything here.
#
# WHAT WOULD MAKE THIS A NULL FOR A BORING REASON. This model is W8A8 -- int8
# experts -- and all 8 ported gfx90a fmoe ASM objects are `noquant` (docs/45).
# Round 43's kernel diff showed fused_moe_kernel still running with AITER MoE
# on, i.e. the ASM path never took over the expert GEMMs. If the dispatch policy
# only steers a path this model cannot reach, all three arms will match. That is
# worth knowing and is reported as such, not as "policy does not help".
#
# The same question on a model that CAN reach the ASM path (bf16 experts) is a
# separate round.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

IMG=vllm-mi210:gdnpolicy
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 50: AITER MoE dispatch policy 0/1/2 ==="

export VLLM_IMAGE="$IMG"
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export TP=2
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

run_one() {  # label  moe(0|1)  policy
    local label="$1" moe="$2" pol="$3"
    echo ""
    echo "=== $(date -u +%T) arm: $label  MOE=$moe policy=$pol ==="
    VLLM_ROCM_USE_AITER_MOE="$moe" \
    VLLM_EXTRA_ENV="-e VLLM_ROCM_AITER_MOE_DISPATCH_POLICY=$pol" \
        "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 131072 2>&1 | tail -6
    local rc=${PIPESTATUS[0]}
    local log="$LOGS/$label.serverlog"
    local asm moe_mod
    asm=$(grep -c "LoadKernel" "$log" 2>/dev/null); asm=${asm:-0}
    moe_mod=$(grep -c "module_moe_asm" "$log" 2>/dev/null); moe_mod=${moe_mod:-0}
    echo "  ASM objects: $asm   module_moe_asm: $moe_mod"
    if [ "$moe" = "1" ] && [ "$moe_mod" -eq 0 ]; then
        echo "  WARNING: MOE=1 but no module_moe_asm -- the AITER MoE path did not load"
    fi
    echo "arm $label rc=$rc"
}

run_one rd50-moeoff 0 0
run_one rd50-p0     1 0
run_one rd50-p1     1 1
run_one rd50-p2     1 2

echo ""
echo "=== $(date -u +%T) round 50 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
arms = [("rd50-moeoff", "MoE off"), ("rd50-p0", "MoE p0"),
        ("rd50-p1", "MoE p1"), ("rd50-p2", "MoE p2")]
rows = [("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
        ("longctx", "implied_prefill_tps_median", "longctx prefill"),
        ("longctx", "decode_tps_median", "longctx decode"),
        ("longctx", "ttft_s_median", "longctx ttft")]
def get(a, wl, k):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(k) if os.path.isfile(f) else None
print(f"{'metric':<17}" + "".join(f"{n:>12}" for _, n in arms))
print("-" * 65)
for wl, k, name in rows:
    cells = ""
    for a, _ in arms:
        v = get(a, wl, k)
        cells += f"{v:12.2f}" if isinstance(v, (int, float)) else f"{'-':>12}"
    print(f"{name:<17}{cells}")
base = get("rd50-moeoff", "longctx", "decode_tps_median")
if base:
    print()
    print("decode vs MoE-off:")
    for a, n in arms[1:]:
        v = get(a, "longctx", "decode_tps_median")
        print(f"  {n:<10} {v/base:.3f}x" if v else f"  {n:<10} -")
print()
print("docs/46 bar: decode needs >1.036x from a single pair of arms. docs/45")
print("measured MoE at 0.977x with policy 0 and an INERT env var; if p1/p2 now")
print("differ from p0, that number was partly a policy artifact. If all three")
print("match, the policy never reaches a path this int8 model can take.")
PY
