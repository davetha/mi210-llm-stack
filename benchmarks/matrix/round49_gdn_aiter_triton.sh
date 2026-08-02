#!/usr/bin/env bash
# Round 49: AITER's Triton Gated-DeltaNet path on Qwen3-Next.
#
# HOW THIS WAS FOUND, AND WHY docs/45 MISSED IT. That survey enumerated
# `is_*_enabled` gates -- 17 -- and called the AITER surface exhausted.
# Enumerating every @if_aiter_supported method instead finds 23, and
# `are_gdn_triton_kernels_available` is one of the six the pattern skipped.
#
# WHAT IT GATES. qwen_gdn_linear_attn.py:72 reads it ONCE at import into a
# module constant, and :799 branches on it -- the docstring: "ROCm forward using
# AITER Triton fused projection+attention when available, otherwise falling back
# to the generic CUDA path." So on gfx90a every Qwen3-Next GDN layer has been
# taking the generic fallback. This is a whole forward implementation, not a
# single kernel.
#
# NOT AN FP8 DEAD END. The method body only tries to IMPORT the kernels, and all
# of them import on this card (verified):
# aiter.ops.triton.causal_conv1d_update_single_token and
# aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr. They are Triton,
# not arch-specific ASM. on_mi3xx() was the only obstacle.
#
# MODEL. t80-awq -- Qwen3NextForCausalLM, 46 GB, 48 layers, linear_conv_kernel
# present, i.e. it genuinely has the GDN layers this path serves. The 30B arm
# used everywhere else in this repo is NOT Qwen3-Next and would measure nothing.
#
# ONE VARIABLE, VIA TWO IMAGES -- deliberately, and here is why it cannot be one
# image with an env var. are_gdn_triton_kernels_available() checks
# cls._AITER_ENABLED, so the only env-var way to turn GDN off is
# VLLM_ROCM_USE_AITER=0, which would also kill AITER flash attention and turn a
# one-variable test into a two-variable one. So: control =
# vllm-mi210:aiterops (gate closed by on_mi3xx), arm = vllm-mi210:gdnpolicy
# (gate carved out). Those images differ by exactly two carve-outs, and the
# second (get_moe_dispatch_policy) is INERT here because this round runs
# VLLM_ROCM_USE_AITER_MOE=0 -- the policy is only read on the AITER MoE path.
# So GDN is the only live difference.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

for img in vllm-mi210:aiterops vllm-mi210:gdnpolicy; do
    docker image inspect "$img" >/dev/null 2>&1 || { echo "FATAL: missing $img"; exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 49: AITER Triton GDN on Qwen3-Next ==="

MODEL=$BASE/t80-awq
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=0      # keeps the policy carve-out inert
export TP=2
export READY_TIMEOUT=1800
export LONGCTX_TOKENS=27852

run_one() {  # label image
    echo ""
    echo "=== $(date -u +%T) arm: $1 ($2) ==="
    VLLM_IMAGE="$2" "$BIN/run_arm.sh" "$1" 80B awq vllm-aiter "$MODEL" \
        --max-model-len 32768 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    local log="$LOGS/$1.serverlog"
    local asm gdn
    asm=$(grep -c "LoadKernel" "$log" 2>/dev/null); asm=${asm:-0}
    # Triton kernels leave no aiter module-import line (docs/45's blind spot),
    # so look for the GDN kernel modules by name instead. Reported, not
    # asserted on: absence here is weak evidence, since Triton JIT logging is
    # not guaranteed at this verbosity.
    gdn=$(grep -ciE "gated_delta|causal_conv1d|gdn_aiter" "$log" 2>/dev/null); gdn=${gdn:-0}
    echo "  ASM objects: $asm   GDN kernel mentions: $gdn"
    echo "arm $1 rc=$rc"
}

run_one rd49-generic vllm-mi210:aiterops
run_one rd49-gdn     vllm-mi210:gdnpolicy

echo ""
echo "=== $(date -u +%T) round 49 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
def get(a, wl, k):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(k) if os.path.isfile(f) else None
rows = [("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
        ("cold16k", "ttft_s_median", "cold16k ttft"),
        ("longctx", "implied_prefill_tps_median", "longctx prefill"),
        ("longctx", "decode_tps_median", "longctx decode"),
        ("longctx", "ttft_s_median", "longctx ttft")]
print(f"{'metric':<17}{'generic':>11}{'AITER GDN':>11}{'GDN/generic':>13}")
print("-" * 52)
for wl, k, name in rows:
    a, b = get("rd49-generic", wl, k), get("rd49-gdn", wl, k)
    if a is None and b is None:
        continue
    ca = f"{a:11.2f}" if isinstance(a, (int, float)) else f"{'-':>11}"
    cb = f"{b:11.2f}" if isinstance(b, (int, float)) else f"{'-':>11}"
    fac = f"{b/a:12.3f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else f"{'-':>13}"
    print(f"{name:<17}{ca}{cb}{fac}")
for a in ("rd49-generic", "rd49-gdn"):
    print(f"  {a}: correctness = {get(a, 'longctx', 'correctness_probe_pass')}")
print()
print("docs/46 measured the decode bar at 1.036x and prefill at ~1.005x for a")
print("single pair of arms. Those were measured on the 30B; this is a different")
print("model, so treat them as indicative. GDN is a linear-attention path, so")
print("any effect should grow with context -- longctx over cold16k.")
PY
