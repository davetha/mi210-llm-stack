#!/usr/bin/env bash
# Does sharded_state round-trip an AWQ checkpoint? 16 GB to find out, not 116.
#
# WHAT ROUND 25 PROVED, AND WHAT IT DID NOT. t35-bf16 went from 12,366 s to
# 114.65 s -- 108x -- with prefill unchanged (8,743.1 -> 8,755.2, 0.14%). That
# validates the mechanism, but it validates it through _load_w13, the UNQUANTIZED
# expert path.
#
# The 235B AWQ dies somewhere else. Round 23:
#
#   WARNING [awq_marlin.py:316] Layer '...mlp.experts' is not supported by
#     AWQMoeMarlin. Falling back to Moe WNA16 kernels.
#   Loading safetensors shards: 8% | 2/25 [43:32<8:33:11, 1338.78s/it]
#
# That is moe_wna16_weight_loader, a different loader, with AWQ's packed int32
# weights plus per-group scales and zero-points. Whether save_sharded_state
# round-trips that packing correctly is untested, and it is the only thing
# standing between us and committing ~9 hours.
#
# WHY t35-awq IS THE RIGHT PROXY. Same quant_method ("awq", 4-bit) and the same
# 128 experts as QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ, at 16 GB against
# 116 GB and 48 layers against 94. t80-awq was rejected as a proxy: it is
# compressed-tensors, i.e. different packing.
#
# NOTE THE SCALE DEPENDENCE THIS ALSO PROBES. docs/28 records t35-awq loading in
# 115 s at TP=1, while the 235B projects to 9.3 h -- both AWQ, both 128 experts.
# That is 287x for roughly 2x the layers, which is the same super-linear
# signature as the +18%-across-six-shards trend in docs/25. So a fast baseline
# here is expected and is NOT evidence the 235B will be fast; the value of this
# round is the CORRECTNESS of the round-trip, not its speed.
#
# WHAT WOULD MAKE THIS A FAILURE. Not a slow reload -- a wrong one. AWQ stores
# int32-packed nibbles with separate scale and zero tensors, and
# save_sharded_state snapshots post-process_weights_after_loading runtime
# tensors, which on this box means WNA16 layout rather than Marlin. If any of
# that does not survive, the model will still serve and will produce garbage.
# The correctness probe in bench_matrix is therefore the primary signal here,
# ahead of any timing.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
SRC=$BASE/t35-awq
DST=$BASE/t35-awq-sharded
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 26: AWQ sharded_state round-trip ==="

# PIN BOTH PHASES TO ONE IMAGE. sharded_state snapshots model.state_dict()
# AFTER process_weights_after_loading -- runtime tensors, not checkpoint tensors
# -- so the snapshot is only valid for the build that produced it (docs/34).
# The first version of this round hardcoded the save to :pa256k while
# serve_vllm_aiter.sh defaults to :latest, two builds 13 hours apart. That made
# any correctness failure unattributable: AWQ packing and a cross-build layout
# mismatch produce the same symptom. VLLM_IMAGE is exported so run_arm.sh ->
# serve_vllm_aiter.sh uses the identical build.
IMG=rocm-vllm-aiter-gfx90a:pa256k
export VLLM_IMAGE="$IMG"
H=/mnt/llm-storage

run() {  # run <label> <model-dir> <extra serve args...>
    local label="$1" model="$2"; shift 2
    echo "--- $label ---"
    LONGCTX_TOKENS=101000 ARM_TIMEOUT=3600 READY_TIMEOUT=2400 \
        "$BIN/run_arm.sh" "$label" 35B awq4 vllm-aiter "$model" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. baseline through the normal loader ---------------------------------
# Re-measured rather than quoted: docs/28's 115 s figure is TP=1, and round 17
# showed what a mismatched baseline does to a comparison.
run t35awq-normal "$SRC" --safetensors-load-strategy=prefetch

# --- B. snapshot -----------------------------------------------------------
if [ ! -d "$DST" ] || [ -z "$(ls -A "$DST" 2>/dev/null)" ]; then
    mkdir -p "$DST"
    echo "=== $(date -u +%T) saving sharded state ==="
    t0=$(date +%s)
    timeout 7200 docker run --rm \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v "$H":/models \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 -e GPU_MAX_HW_QUEUES=4 \
        -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 \
        --entrypoint python3 "$IMG" \
        /app/vllm/examples/features/sharded_state/save_sharded_state_offline.py \
            --model "${SRC/#$H//models}" \
            --tensor-parallel-size 2 \
            --output "${DST/#$H//models}" \
        2>&1 | tail -20
    rc=${PIPESTATUS[0]}
    echo "=== save exit=$rc after $(( ($(date +%s) - t0) / 60 )) min ==="
    [ "$rc" -ne 0 ] && { echo "!! snapshot failed"; rm -rf "$DST"; exit 1; }
fi
du -sh "$DST"

# --- C. reload -------------------------------------------------------------
run t35awq-sharded "$DST" --load-format sharded_state

echo "=== $(date -u +%T) round 26 done ==="
echo
echo "======== CORRECTNESS FIRST, THEN TIMING ========"
python3 - <<'PY'
import json, glob, os
for lab in ("t35awq-normal", "t35awq-sharded"):
    for w in ("cold16k", "longctx"):
        f = f"results/{lab}-{w}.json"
        if not os.path.exists(f):
            fail = f"results/{lab}-FAILED.json"
            if os.path.exists(fail):
                print(f"  {lab:16} {w:8} FAILED: {json.load(open(fail)).get('reason')}")
            continue
        d = json.load(open(f))
        ef = d.get("engine_footprint") or {}
        print("  %-16s %-8s ok=%-5s load=%-9s prefill=%8.1f decode=%6.2f" % (
            lab, w, d.get("correctness_probe_pass"), ef.get("load_seconds", "?"),
            d.get("implied_prefill_tps_median") or 0, d.get("decode_tps_median") or 0))
PY
echo
echo "ok=False on the sharded arm means the AWQ packing did NOT survive the"
echo "round-trip -- and that is the answer that saves ~9 hours on the 235B."
