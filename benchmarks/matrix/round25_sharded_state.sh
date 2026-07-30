#!/usr/bin/env bash
# Pay the 3.4-hour bf16 load once, then measure what it costs afterwards.
#
# WHY THIS MATTERS. bf16 posted 8,743.1 t/s prefill at TTFT 1.7 s in round 19 --
# the fastest prefill number in this entire project, ahead of W8A8's 8,343. And
# docs/28 tells people to avoid it, because the same model takes 12,366 s to load
# against 60 s as W8A8. That rule exists because of the LOADER, not because of
# bf16.
#
# docs/25 item 1 is the longest investigation in this repo and all of it went
# into trying to PATCH that loader -- three attempts, all refuted. The question
# never asked was whether another backend skips it. One does, and it ships:
# sharded_state_loader.py's entire weight-load body is
#
#     param_data.copy_(tensor)
#     state_dict.pop(key)
#
# a flat copy into an already-allocated runtime parameter. No weight_loader, no
# _load_w13, no moe_wna16_weight_loader -- so the per-expert iteration behind the
# hsakmt_ioctl storm has no analogue, and neither does the super-linear
# +18%-across-six-shards term. Found by Andrei-Dr; verified against the install
# in docs/34.
#
# WHAT THIS ROUND ACTUALLY BUYS. Not a throughput number -- serving performance
# should be identical, since sharded_state changes only how weights arrive. It
# buys a REUSABLE ARTIFACT plus a measurement of the reload. If the reload is
# minutes rather than hours, bf16 stops being a curiosity and becomes the
# default for anything prefill-heavy that fits, and every future bf16 experiment
# gets cheap. That is why it is worth 3.4 hours of GPU time once.
#
# THE SAVE PHASE HOLDS THE LOCK, DELIBERATELY. Unlike a download, this needs both
# GPUs -- it loads the model through the slow path in order to snapshot the
# runtime tensors. There is no version of this that runs unlocked.
#
# CONSTRAINTS, from docs/34 and the source:
#   - Pre-sharded only; the snapshot must exist before --load-format works.
#   - Bound to the TP layout: the glob embeds rank=, so reload must use TP=2.
#   - Version-specific: it snapshots model.state_dict() AFTER
#     process_weights_after_loading, i.e. runtime tensors. Regenerate on a vLLM
#     bump or the reload will not match.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
SRC=$BASE/t35-bf16
DST=$BASE/t35-bf16-sharded
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 25: sharded_state conversion ==="

IMG=rocm-vllm-aiter-gfx90a:pa256k
H=/mnt/llm-storage

# --- Phase 1: snapshot. Expect ~3.4 h; this is the cost being amortised. ----
if [ ! -d "$DST" ] || [ -z "$(ls -A "$DST" 2>/dev/null)" ]; then
    mkdir -p "$DST"
    echo "=== $(date -u +%T) saving sharded state (expect ~3.4 h) ==="
    t0=$(date +%s)
    timeout 21600 docker run --rm \
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
        2>&1 | tail -25
    rc=${PIPESTATUS[0]}
    echo "=== save exit=$rc after $(( ($(date +%s) - t0) / 60 )) min ==="
    if [ "$rc" -ne 0 ]; then
        echo "!! snapshot failed -- not attempting the reload arm"
        rm -rf "$DST"
        exit 1
    fi
else
    echo "=== snapshot already present, skipping save ==="
fi
du -sh "$DST"

# --- Phase 2: reload, and time it ------------------------------------------
# --load-format sharded_state with the SAME TP as the snapshot. The engine
# footprint harvester in run_arm.sh picks up load_seconds, and the loader also
# prints its own "Loading weights took %.2f seconds" line.
LONGCTX_TOKENS=101000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" t35bf16-sharded 35B bf16 vllm-aiter "$DST" \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --load-format sharded_state \
    || echo "!! t35bf16-sharded failed (recorded)"

echo "=== $(date -u +%T) round 25 done ==="
echo
echo "=============== LOAD TIME: THE WHOLE POINT ==============="
echo "  baseline (normal loader, docs/28) : 12,366 s"
grep -oE "Loading weights took [0-9.]+ seconds" logs/t35bf16-sharded.serverlog 2>/dev/null | tail -1 | sed "s/^/  sharded_state                     : /"
python3 -c "
import json,glob
for f in sorted(glob.glob('results/t35bf16-sharded-*.json')):
    d=json.load(open(f))
    ef=d.get('engine_footprint') or {}
    print('  %-8s load=%ss prefill=%.1f decode=%.2f' % (d.get('workload'),
        ef.get('load_seconds','?'), d.get('implied_prefill_tps_median') or 0,
        d.get('decode_tps_median') or 0))
" 2>/dev/null
echo "  (serving throughput should MATCH round 19's 8,743.1 t/s -- sharded_state"
echo "   changes only how weights arrive, so a difference here means something"
echo "   else moved and the comparison is not clean.)"
