#!/usr/bin/env bash
# Round 51: the only configuration on this hardware that can reach ALL THREE
# ASM families at once.
#
# WHY NO PRIOR ARM COULD. Every vLLM benchmark in this repo runs the W8A8
# checkpoint, and that model is structurally locked out of two of the three:
#
#   fmha_v3_fwd  48/56 ported, IN USE today -- 1.19-1.33x prefill (docs/28)
#   pa            8/56 ported, needs GQA ratio 8 or 16
#   fmoe          8/839 ported, and ALL EIGHT ARE `noquant` (docs/45) -- they
#                 cannot consume int8 expert weights, which is why round 42's
#                 AITER MoE arm loaded module_moe_asm and still left
#                 fused_moe_kernel running (round 43's kernel diff).
#
# docs/35 wrote the checklist and named the model. t35-bf16 satisfies all of it,
# verified on disk before this round:
#
#   head_dim 128                    -> fmha eligible (ported: 24x hd128 +
#                                      24x hd192x128, all bf16, no fp16)
#   32 heads / 4 kv = ratio 8       -> pa eligible
#   experts unquantized bfloat16    -> fmoe eligible
#   VLLM_ROCM_USE_AITER_MOE=1       -> set by the arm below
#
# THE LOAD OBJECTION IS GONE. docs/28 discourages unquantized bf16 because of a
# 12,366 s load. t35-bf16-sharded exists -- 57 GB, 12 safetensors, rank-0 and
# rank-1, i.e. a TP=2 snapshot -- so --load-format sharded_state applies
# (docs/34). This round MUST run TP=2 to match it; the TP guard from
# configs/enable_sharded_state_tp_check.py is in the image and will refuse a
# mismatch loudly rather than half-loading, which is exactly the failure that
# patch was written for.
#
# THE MEASUREMENT. Both arms are the same image and the same sharded
# checkpoint; only VLLM_ROCM_USE_AITER_MOE moves. The interesting number is not
# only throughput but the ASM object count: if pa and fmoe engage, LoadKernel
# should rise above the 4 that fmha alone produces on the W8A8 arms. An arm
# that shows MOE=1 with no new ASM objects has not reached the new families,
# and its throughput is not evidence about them.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

IMG=vllm-mi210:gdnpolicy    # carries the MoE dispatch-policy carve-out too
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-bf16-sharded
[ -f "$MODEL/config.json" ] || { echo "FATAL: no sharded snapshot at $MODEL"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 51: bf16 full-ASM (fmha + pa + fmoe) ==="

export VLLM_IMAGE="$IMG"
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export TP=2                    # MUST match the snapshot's rank count
export READY_TIMEOUT=1800
export LONGCTX_TOKENS=27852

run_one() {  # label  moe(0|1)
    local label="$1" moe="$2"
    echo ""
    echo "=== $(date -u +%T) arm: $label  AITER_MOE=$moe ==="
    VLLM_ROCM_USE_AITER_MOE="$moe" \
        "$BIN/run_arm.sh" "$label" 35B bf16 vllm-aiter "$MODEL" \
        --max-model-len 32768 --load-format sharded_state 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}
    local log="$LOGS/$label.serverlog"
    local asm co moe_mod pa_mod
    asm=$(grep -c "LoadKernel" "$log" 2>/dev/null); asm=${asm:-0}
    # Which ASM families actually loaded -- the point of the whole round.
    co=$(grep -ohE "[a-z0-9_]+\.co" "$log" 2>/dev/null | sort -u | tr '\n' ' ')
    moe_mod=$(grep -c "module_moe_asm" "$log" 2>/dev/null); moe_mod=${moe_mod:-0}
    pa_mod=$(grep -ciE "pa_fwd|paged_attention_common" "$log" 2>/dev/null); pa_mod=${pa_mod:-0}
    echo "  ASM objects: $asm   module_moe_asm: $moe_mod   pa refs: $pa_mod"
    echo "  distinct .co loaded: ${co:-<none>}"
    echo "$co" > "$LOGS/$label.cofiles"
    echo "arm $label rc=$rc"
}

run_one rd51-moeoff 0
run_one rd51-moeon  1

echo ""
echo "=== $(date -u +%T) round 51 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
L = "/mnt/llm-storage/bench-matrix/logs"
def get(a, wl, k):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(k) if os.path.isfile(f) else None
rows = [("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
        ("cold16k", "ttft_s_median", "cold16k ttft"),
        ("longctx", "implied_prefill_tps_median", "longctx prefill"),
        ("longctx", "decode_tps_median", "longctx decode"),
        ("longctx", "ttft_s_median", "longctx ttft")]
print(f"{'metric':<17}{'MoE off':>11}{'MoE on':>11}{'on/off':>10}")
print("-" * 49)
for wl, k, name in rows:
    a, b = get("rd51-moeoff", wl, k), get("rd51-moeon", wl, k)
    if a is None and b is None:
        continue
    ca = f"{a:11.2f}" if isinstance(a, (int, float)) else f"{'-':>11}"
    cb = f"{b:11.2f}" if isinstance(b, (int, float)) else f"{'-':>11}"
    fac = f"{b/a:9.3f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else f"{'-':>10}"
    print(f"{name:<17}{ca}{cb}{fac}")
print()
for a in ("rd51-moeoff", "rd51-moeon"):
    print(f"  {a}: correctness = {get(a, 'longctx', 'correctness_probe_pass')}")
    f = os.path.join(L, f"{a}.cofiles")
    if os.path.isfile(f):
        print(f"    .co: {open(f).read().strip() or '<none>'}")
print()
print("THE ASM LINES MATTER MORE THAN THE THROUGHPUT. W8A8 arms load 4 objects,")
print("all fmha. If the MoE-on arm here shows fmoe/pa objects that the W8A8 arms")
print("never had, this is the first configuration on this card to reach them --")
print("and only then does its throughput say anything about those families.")
print("Compare decode against docs/46's 1.036x bar, and note this is a DIFFERENT")
print("checkpoint from every other arm, so cross-round comparison is invalid.")
PY
