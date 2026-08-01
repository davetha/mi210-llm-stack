#!/usr/bin/env bash
# Round 46: does AllReduce+RMSNorm fusion do anything at TP=2 on gfx90a?
#
# THE LEAD. All four allreduce ASM objects ported to gfx90a --
# all_reduce.co, allreduce_rmsnorm_N8192.co, allreduce_rmsnorm_qnt_N8192.co,
# allreduce_layernorm_N8192.co -- and docs/37 dismissed them partly because
# "there is no XGMI for it to accelerate anyway". Round 31 refuted that:
# PCIe P2P measures 26.98 GB/s and is worth +11.2% prefill. At TP=2 the
# per-layer collective is a fixed cost the CK GEMM win cannot touch, so fusing
# it into the norm is the right shape of attack.
#
# TWO THINGS HAVE TO CHANGE, which is why this is its own round:
#   1. pass_manager.py adds RocmAiterAllReduceFusionPass only when
#      rocm_aiter_ops.is_enabled() is true -- and that gate is
#      @if_aiter_supported -> on_mi3xx(), so it returns None on CDNA2 and the
#      pass is silently never added. configs/enable_aiter_master_gate_gfx90a.py
#      carves it out.
#   2. pass_config.fuse_allreduce_rms defaults to None and must be turned on
#      explicitly, which this round does via --compilation-config.
#
# THE RISK, STATED UP FRONT. Unlike the five narrow gates in docs/45,
# is_enabled() has ~22 consumers across 10 files, TWO OF THEM in
# v1/attention/backends/rocm_aiter_fa.py -- the flash attention path currently
# delivering 1.19-1.33x prefill. Today it sees None and the stack works;
# flipping it to True changes branches inside a WORKING optimization. So this
# round asserts that both existing wins survive, and treats a regression in
# either as the headline result rather than a footnote:
#
#     LoadKernel count > 0      -> AITER FA ASM still running
#     module_gemm_a8w8 present  -> CK int8 GEMM still running
#
# An arm that loses either is reported as a REGRESSION, and its throughput
# numbers are not comparable to the control's.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

for img in vllm-mi210:aiterops vllm-mi210:mastergate; do
    docker image inspect "$img" >/dev/null 2>&1 || { echo "FATAL: missing $img"; exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 46: AllReduce+RMSNorm fusion at TP=2 ==="

export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=0
export TP=2                       # allreduce only exists at TP>1
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

run_one() {  # label  image  [extra serve args...]
    local label="$1" image="$2"; shift 2
    echo ""
    echo "=== $(date -u +%T) arm: $label ($image) ==="
    VLLM_IMAGE="$image" "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter \
        "$BASE/t35-w8a8" --max-model-len 131072 "$@" 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}

    local log="$LOGS/$label.serverlog"
    local asm ck ar
    asm=$(grep -c "LoadKernel" "$log" 2>/dev/null); asm=${asm:-0}
    ck=$(grep -c "module_gemm_a8w8" "$log" 2>/dev/null); ck=${ck:-0}
    # Direct evidence the allreduce ASM engaged: aiter names the .co it loads.
    ar=$(grep -ciE "allreduce|all_reduce" "$log" 2>/dev/null); ar=${ar:-0}
    echo "  ASM code objects: $asm   CK GEMM refs: $ck   allreduce mentions: $ar"

    if [ "$asm" -eq 0 ]; then
        echo "  REGRESSION: AITER FA ASM stopped loading in this arm."
        echo "  The 1.19-1.33x prefill win is GONE here; throughput below is not"
        echo "  comparable to the control."
    fi
    if [ "$ck" -eq 0 ]; then
        echo "  REGRESSION: the CK int8 GEMM stopped loading in this arm."
        echo "  The 1.48x decode win is GONE here; numbers are not comparable."
    fi
    echo "arm $label rc=$rc"
}

run_one rd46-control vllm-mi210:aiterops
run_one rd46-arfuse  vllm-mi210:mastergate \
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'

echo ""
echo "=== $(date -u +%T) round 46 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
def get(a, wl, k):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(k) if os.path.isfile(f) else None
rows = [("cold16k", "implied_prefill_tps_median", "prefill"),
        ("cold16k", "ttft_s_median", "ttft"),
        ("longctx", "implied_prefill_tps_median", "prefill"),
        ("longctx", "decode_tps_median", "decode"),
        ("longctx", "ttft_s_median", "ttft")]
print(f"{'workload':<9} {'metric':<8} {'control':>10} {'AR-fused':>10} {'factor':>9}")
print("-" * 50)
for wl, k, name in rows:
    a, b = get("rd46-control", wl, k), get("rd46-arfuse", wl, k)
    if a is None and b is None:
        continue
    ca = f"{a:10.2f}" if isinstance(a, (int, float)) else f"{'-':>10}"
    cb = f"{b:10.2f}" if isinstance(b, (int, float)) else f"{'-':>10}"
    fac = f"{b/a:8.3f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else f"{'-':>9}"
    print(f"{wl:<9} {name:<8} {ca} {cb} {fac}")
print()
print("Allreduce fusion should show up in PREFILL first -- prefill collectives")
print("move large activation tensors and are bandwidth-bound, decode's are small")
print("and latency-bound (that is the shape round 31 measured for P2P). Judge")
print("any decode delta against the round 45 noise floor, not against zero.")
PY
