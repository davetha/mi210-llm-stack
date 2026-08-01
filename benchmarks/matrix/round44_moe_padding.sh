#!/usr/bin/env bash
# Round 44: does ROCm MoE stride padding help the INT8 expert path?
#
# vLLM pads MoE weights by 256 B and slices straight back, leaving the logical
# shape and every value unchanged and moving only the row stride, to de-alias
# memory banks. VLLM_ROCM_MOE_PADDING defaults True; upstream (vllm#14454)
# reports up to 10% on Mixtral.
#
# It is applied ONLY on the unquantized path -- `VLLM_ROCM_MOE_PADDING` appears
# in envs.py, fused_moe/oracle/unquantized.py and
# unquantized_fused_moe_method.py, and nowhere else. A W8A8 checkpoint never
# receives it. configs/enable_moe_padding_int8_rocm.py extends it to the INT8
# expert path.
#
# Worth testing because `fused_moe_kernel` is the LARGEST single decode kernel
# on this box -- 1947 ms, 24.4% of the decode window (docs/45) -- now that the
# CK GEMM has displaced the previous leader.
#
# ELIGIBILITY IS PARTIAL, BY DESIGN. The upstream guard only pads a tensor
# whose row stride is already a multiple of 512 B. At int8 for this model:
#   w13 [E, 1536, 2048] -> stride 2048 B, 2048 % 512 == 0   ELIGIBLE
#   w2  [E, 2048,  768] -> stride  768 B,  768 % 512 == 256 not eligible
# So about half the expert bytes move. A null here does NOT mean padding is
# worthless in general -- it means it is worthless at this coverage.
#
# THE ASSERTION. "The patch is in the image" is not evidence the stride
# actually changed: the pad is a view, and any later .contiguous() silently
# re-tightens it (vLLM guards exactly that on the unquantized path, at
# oracle/unquantized.py:353). So this round measures the SERVED model's real
# w13/w2 strides in-process and refuses to report a comparison unless the
# padded arm's stride actually differs from the control's.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

for img in vllm-mi210:aiterops vllm-mi210:moepad; do
    docker image inspect "$img" >/dev/null 2>&1 || { echo "FATAL: missing $img"; exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 44: INT8 MoE stride padding ==="

# --- stride evidence, before any benchmarking -------------------------------
# Load the model offline in each image and print the real strides. This is the
# check that distinguishes "padded" from "padded then quietly undone".
stride_probe() {  # $1 = image, $2 = label
    echo ""
    echo "--- stride probe: $2 ($1)"
    docker run --rm --name "probe-stride-$2" \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
      --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
      -v /mnt/llm-storage:/models -e HSA_NO_SCRATCH_RECLAIM=1 \
      -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_LINEAR=1 \
      -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      --entrypoint python3 "$1" -c '
from vllm import LLM
llm = LLM(model="/models/bench-matrix/t35-w8a8", max_model_len=4096,
          gpu_memory_utilization=0.85, tensor_parallel_size=1,
          enable_prefix_caching=False, seed=1234)
m = llm.llm_engine.model_executor.driver_worker.model_runner.model
seen = 0
for name, mod in m.named_modules():
    w13 = getattr(mod, "w13_weight", None)
    w2 = getattr(mod, "w2_weight", None)
    if w13 is None or w2 is None:
        continue
    print(f"STRIDE {name} w13 shape={tuple(w13.shape)} stride={tuple(w13.stride())} "
          f"| w2 shape={tuple(w2.shape)} stride={tuple(w2.stride())}")
    seen += 1
    if seen >= 2:
        break
if seen == 0:
    print("STRIDE none-found")
' 2>&1 | grep -E "^STRIDE" | head -3 | tee "$LOGS/rd44-$2.strides"
}

stride_probe vllm-mi210:aiterops control
stride_probe vllm-mi210:moepad   padded

if ! diff -q "$LOGS/rd44-control.strides" "$LOGS/rd44-padded.strides" >/dev/null 2>&1; then
    echo ""
    echo "STRIDES DIFFER -- the padding is live. Proceeding to the A/B."
else
    echo ""
    echo "ROUND ABORTED: served strides are IDENTICAL between the two images."
    echo "Either the pad never applied or something re-tightened it. Benchmarking"
    echo "now would compare an image against itself and report the null as a"
    echo "result. Fix the patch, do not fix the report."
    exit 1
fi

# --- the A/B ----------------------------------------------------------------
export NCCL_P2P_DISABLE=0
export VLLM_TUNED_CONFIG_FOLDER=
export VLLM_PREFER_AITER_FA=1
export VLLM_ROCM_USE_AITER_LINEAR=1
export VLLM_ROCM_USE_AITER_MOE=0     # rounds 42/43: AITER MoE is a wash-to-loss
export TP=2
export READY_TIMEOUT=1200
export LONGCTX_TOKENS=27852

run_one() {  # label image
    echo ""
    echo "=== $(date -u +%T) arm: $1  ($2) ==="
    VLLM_IMAGE="$2" "$BIN/run_arm.sh" "$1" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 131072 2>&1 | tail -8
    echo "arm $1 rc=${PIPESTATUS[0]}"
}

run_one rd44-nopad vllm-mi210:aiterops
run_one rd44-pad   vllm-mi210:moepad

echo ""
echo "=== $(date -u +%T) round 44 done ==="
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
print(f"{'workload':<9} {'metric':<8} {'no pad':>10} {'padded':>10} {'pad/nopad':>11}")
print("-" * 52)
for wl, k, name in rows:
    a, b = get("rd44-nopad", wl, k), get("rd44-pad", wl, k)
    if a is None and b is None:
        continue
    ca = f"{a:10.2f}" if isinstance(a, (int, float)) else f"{'-':>10}"
    cb = f"{b:10.2f}" if isinstance(b, (int, float)) else f"{'-':>10}"
    fac = f"{b/a:10.3f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else f"{'-':>11}"
    print(f"{wl:<9} {name:<8} {ca} {cb} {fac}")
print()
print("Only w13 is eligible (row stride 2048 B); w2 at 768 B is not, so this is")
print("partial coverage by construction. The noise floor on this rig is +/-2-3%,")
print("so anything inside that is a null, not a small win.")
PY
