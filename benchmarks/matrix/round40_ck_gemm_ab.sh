#!/usr/bin/env bash
# Round 40: does the AITER CK int8 GEMM beat the Triton fallback END TO END?
#
# Round 38e's profile put `scaled_mm_kernel` -- vLLM's generic Triton int8
# GEMM -- at 44% of DECODE kernel time, on a card measured 99.9% kernel-busy
# during decode. probe_a8w8_ck_gfx90a.py then measured AITER's CK GEMM at the
# real model shapes: decode median 1.662x (min 1.075x, max 1.810x), prefill
# median 1.210x, all within 3.5e-3 of an fp32 reference.
#
# Amdahl on those two numbers predicts ~1.17-1.21x end-to-end decode. This
# round finds out, because a kernel microbenchmark and a served model are not
# the same claim: real batches vary, CK's DEFAULT (untuned) instance
# selection is doing the choosing -- there are no gfx90a rows in
# a8w8_tuned_gemm.csv -- and round 34 already measured a case where a tuned
# MoE config was 0.79x, i.e. worse than none.
#
# ONE VARIABLE. Same image, same model, same pins; only
# VLLM_ROCM_USE_AITER_LINEAR moves. The image carries
# configs/enable_aiter_ck_gemm_gfx90a.py (CU-map entry + is_linear_enabled
# carve-out) and has module_gemm_a8w8 pre-built, so neither arm pays a JIT
# stall the other does not.
#
# THE ASSERTION THAT MATTERS. vLLM does NOT log which int8 kernel it picked
# -- choose_scaled_mm_linear_kernel() returns the first supported entry
# silently (kernels/linear/__init__.py:566-577). So "the flag was set" is not
# evidence the CK kernel ran. AITER does log its module imports, so
# `module_gemm_a8w8` appearing in the serverlog is the direct proof, and this
# round requires it present in the CK arm and ABSENT in the Triton arm.
# Without that pair, both arms could be Triton and the null would look like a
# result -- the failure this project has now shipped three times.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

IMAGE=vllm-mi210:v0.26.1rc0-ckgemm-warm
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FATAL: missing image $IMAGE (build it from Dockerfile.ckgemm, then warm it)"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 40: CK int8 GEMM vs Triton scaled_mm, end to end ==="

export VLLM_IMAGE="$IMAGE"
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export TP=2
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

run_one() {  # label linear(0|1)
    local label="$1" want="$2"
    echo ""
    echo "=== $(date -u +%T) arm: $label (VLLM_ROCM_USE_AITER_LINEAR=$want) ==="
    VLLM_ROCM_USE_AITER_LINEAR="$want" \
        "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 131072 2>&1 | tail -8
    local rc=${PIPESTATUS[0]}

    local ck asm
    ck=$(grep -c "module_gemm_a8w8" "$LOGS/$label.serverlog" 2>/dev/null); ck=${ck:-0}
    asm=$(grep -c "LoadKernel" "$LOGS/$label.serverlog" 2>/dev/null); asm=${asm:-0}
    echo "  module_gemm_a8w8 references: $ck   ASM code objects: $asm"

    if [ "$want" = "1" ] && [ "$ck" -eq 0 ]; then
        echo "  ARM INVALID: AITER_LINEAR=1 but the CK GEMM module never loaded."
        echo "  This arm ran the Triton kernel and is NOT a measurement of CK."
        return 1
    fi
    if [ "$want" = "0" ] && [ "$ck" -gt 0 ]; then
        echo "  ARM INVALID: AITER_LINEAR=0 but the CK GEMM module loaded anyway."
        echo "  The control is not a control; the comparison is void."
        return 1
    fi
    echo "  verified: linear backend is $([ "$want" = 1 ] && echo 'AITER CK' || echo 'Triton')"
    echo "arm $label rc=$rc"
}

run_one rd40-triton 0
run_one rd40-ck     1

echo ""
echo "=== $(date -u +%T) round 40 done ==="
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
print(f"{'workload':<9} {'metric':<8} {'Triton':>10} {'AITER CK':>10} {'CK/Triton':>11}")
print("-" * 52)
for wl, key, name in rows:
    t, c = get("rd40-triton", wl, key), get("rd40-ck", wl, key)
    if t is None and c is None:
        continue
    ct = f"{t:10.2f}" if isinstance(t, (int, float)) else f"{'-':>10}"
    cc = f"{c:10.2f}" if isinstance(c, (int, float)) else f"{'-':>10}"
    fac = f"{c/t:10.3f}x" if isinstance(t, (int, float)) and isinstance(c, (int, float)) and t else f"{'-':>11}"
    print(f"{wl:<9} {name:<8} {ct} {cc} {fac}")

d_t, d_c = get("rd40-triton", "longctx", "decode_tps_median"), get("rd40-ck", "longctx", "decode_tps_median")
if d_t and d_c:
    print()
    print(f"decode {d_c/d_t:.3f}x.  Prediction from the kernel probe + the 44% "
          "decode share was\n~1.17-1.21x; Amdahl on a microbenchmark is an "
          "upper bound, so landing below it\nis expected and landing above it "
          "means something else moved too.")
print()
print("TTFT rows invert -- lower is better, so a factor BELOW 1.0 is the win there.")
PY
