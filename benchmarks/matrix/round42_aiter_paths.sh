#!/usr/bin/env bash
# Round 42: the remaining AITER fast paths, one flag per arm.
#
# docs/43 opened the CK int8 GEMM and, incidentally, op REGISTRATION -- before
# that patch not one AITER custom op existed on gfx90a. A survey of all 17
# `is_*_enabled` gates then found `linear` and `mha` True and FOURTEEN
# returning None (the @if_aiter_supported -> on_mi3xx() short-circuit), while
# every underlying aiter symbol imports cleanly here: fused_moe, asm_moe_tkw1,
# topk_softmax, grouped_topk, dynamic_per_token_scaled_quant, pa_fwd_asm.
#
# configs/enable_aiter_ops_gfx90a.py carves out five of them. Each keeps its
# own VLLM_ROCM_USE_AITER_* flag defaulting OFF, so this round can move exactly
# one variable per arm and a kernel that crashes takes down only its own arm.
#
# Worth, from the round-41 decode profile (7,979 ms busy in an 8 s window):
#
#   fused_moe_kernel       1947 ms  24.4%  <- MOE arm targets this
#   + topkGating            393 ms   4.9%  <- and this
#   + moe_sum_vec           220 ms   2.8%  <- and this   = 32% of decode
#   paged_attention        1045 ms  13.1%  (no flag; see docs/18 / round 22)
#   wvSplitK                680 ms   8.5%  (NO aiter int8 path -- hipb_mm is fp8)
#
# IMPORTING IS NOT RUNNING. Every arm records which aiter JIT modules its
# server actually loaded, and the summary flags any arm whose module set is
# IDENTICAL to baseline -- because that means the flag changed nothing and the
# arm is measuring the baseline twice. That is this project's most repeated
# defect (four hardcoded-value sightings, one vacuous async A/B) and it does
# not get a sixth.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

IMAGE=vllm-mi210:aiterops
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FATAL: missing $IMAGE (build configs/Dockerfile.aiterops)"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 42: remaining AITER fast paths, one flag per arm ==="

export VLLM_IMAGE="$IMAGE"
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1   # the CK GEMM: held ON in every arm
export TP=2
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

# label                 flag suffix              how it is passed
#   baseline            -                        -
#   moe                 MOE                      dedicated var in serve script
#   rope                TRITON_ROPE              VLLM_EXTRA_ENV
#   unifattn            UNIFIED_ATTENTION        VLLM_EXTRA_ENV
#   customar            CUSTOM_AR                VLLM_EXTRA_ENV
#   tritongemm          TRITON_GEMM              VLLM_EXTRA_ENV
run_one() {  # label  [flagname]
    local label="rd42-$1" flag="${2:-}"
    echo ""
    echo "=== $(date -u +%T) arm: $label  flag=${flag:-none} ==="
    local extra=""
    if [ -n "$flag" ] && [ "$flag" = "MOE" ]; then
        export VLLM_ROCM_USE_AITER_MOE=1
    else
        export VLLM_ROCM_USE_AITER_MOE=0
        [ -n "$flag" ] && extra="-e VLLM_ROCM_USE_AITER_${flag}=1"
    fi
    VLLM_EXTRA_ENV="$extra" \
        "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 131072 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    unset VLLM_ROCM_USE_AITER_MOE

    # Which aiter JIT modules did this server actually load? That is the
    # evidence the flag reached a kernel; the flag being set is not.
    local mods
    mods=$(grep -ohE "import \[module_[a-z_0-9]+\]" "$LOGS/$label.serverlog" 2>/dev/null \
           | sort -u | tr '\n' ' ')
    echo "$mods" > "$LOGS/$label.aitermods"
    echo "  aiter modules loaded: ${mods:-<none>}"
    echo "arm $label rc=$rc"
}

run_one baseline
run_one moe        MOE
run_one rope       TRITON_ROPE
run_one unifattn   UNIFIED_ATTENTION
run_one customar   CUSTOM_AR
run_one tritongemm TRITON_GEMM

echo ""
echo "=== $(date -u +%T) round 42 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
L = "/mnt/llm-storage/bench-matrix/logs"
arms = [("baseline", "-"), ("moe", "MOE"), ("rope", "TRITON_ROPE"),
        ("unifattn", "UNIFIED_ATTENTION"), ("customar", "CUSTOM_AR"),
        ("tritongemm", "TRITON_GEMM")]

def get(a, wl, key):
    f = os.path.join(R, f"rd42-{a}-{wl}.json")
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    return d.get(key)

def mods(a):
    f = os.path.join(L, f"rd42-{a}.aitermods")
    return open(f).read().strip() if os.path.isfile(f) else ""

base_d = get("baseline", "longctx", "decode_tps_median")
base_p = get("baseline", "cold16k", "implied_prefill_tps_median")
base_mods = mods("baseline")

print(f"{'arm':<11}{'flag':<19}{'decode':>9}{'vs base':>9}{'prefill':>10}{'vs base':>9}  status")
print("-" * 82)
for a, flag in arms:
    d = get(a, "longctx", "decode_tps_median")
    p = get(a, "cold16k", "implied_prefill_tps_median")
    if d is None and p is None:
        print(f"{a:<11}{flag:<19}{'FAILED -- see logs/rd42-'+a+'.serverlog':>40}")
        continue
    dd = f"{d:9.2f}" if d else f"{'-':>9}"
    pp = f"{p:10.2f}" if p else f"{'-':>10}"
    fd = f"{d/base_d:8.3f}x" if d and base_d else f"{'-':>9}"
    fp = f"{p/base_p:8.3f}x" if p and base_p else f"{'-':>9}"
    status = ""
    if a != "baseline" and mods(a) == base_mods:
        status = "NO NEW AITER MODULE -- flag changed nothing, not a measurement"
    print(f"{a:<11}{flag:<19}{dd}{fd}{pp}{fp}  {status}")

print()
print("baseline aiter modules:", base_mods or "<none>")
for a, _ in arms[1:]:
    m = mods(a)
    if m and m != base_mods:
        new = [x for x in m.split() if x not in base_mods.split()]
        if new:
            print(f"  {a}: NEW -> {' '.join(new)}")
print()
print("READING THIS. An arm marked NO NEW AITER MODULE loaded exactly what")
print("baseline loaded, so its flag reached no kernel and its numbers are the")
print("baseline measured twice -- report it as inconclusive, never as a null.")
print("A FAILED arm is a result too: it means that aiter path does not run on")
print("gfx90a, which is worth knowing and is why the arms are isolated.")
PY
