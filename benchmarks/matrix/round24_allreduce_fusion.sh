#!/usr/bin/env bash
# The fused allreduce+RMSNorm path: complete, ROCm-specific, and never switched on.
#
# WHAT WE FOUND. pass_manager.py registers a dedicated ROCm pass:
#
#   if self.pass_config.fuse_allreduce_rms:
#       if rocm_aiter_ops.is_enabled():
#           self.passes += [RocmAiterAllReduceFusionPass(config)]
#       else:
#           self.passes += [AllReduceFusionPass(config)]
#
# RocmAiterAllReduceFusionPass is fully wired -- it calls
# rocm_aiter_ops.initialize_aiter_allreduce(group, device) and
# get_fused_allreduce_rmsnorm_op(), and _rocm_aiter_fused_allreduce_rmsnorm_impl
# has a real body plus a registered meta impl. We already satisfy
# rocm_aiter_ops.is_enabled(). The only reason it has never run is that
# fuse_allreduce_rms defaults OFF and nothing here has ever set it.
#
# That is a different category from every other ASM miss this project has found:
# not an absent call site (pa_fwd_asm), not a declined device (fmoe), not an
# inert flag (mla). Just an unset option on a complete code path.
#
# WHY THE 70B. hidden_size = 8192, which matches allreduce_rmsnorm_N8192.co
# exactly -- the gfx90a binaries are compiled per hidden size and 8192 is the one
# we have. It is also the worst decode in the matrix (6.8 t/s, docs/28), and
# dense 70B at TP=2 is 80 layers x 2 collectives per token over PCIe with no
# XGMI. TP=2 returns only 1.28x (33.80 -> 43.40 t/s on the 30B), which is
# per-layer collective latency rather than bandwidth. Fusing the collective with
# the norm removes a launch and a full pass over the tensor per layer, on a
# decode path docs/33 argues is issue-bound. This is the most direct attack on
# that 1.28x available.
#
# NO --enforce-eager, AND IT CANNOT BE ADDED. These are torch.compile pattern
# passes; disabling graph capture disables them entirely. Note the consequence:
# allreduce fusion and the prefetch offloader are MUTUALLY EXCLUSIVE, because
# round 18 established the offloader requires eager to avoid an RCCL/graph
# capture collision. Any future config has to choose one.
#
# 64000 matches the context docs/28 quotes for this arm's 6.8 t/s decode, so the
# comparison is against a published number rather than against nothing.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/t70-w8a8
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 24: fused allreduce+RMSNorm ==="

run() {  # run <label> <extra serve args...>
    local label="$1"; shift
    echo "--- $label : $* ---"
    LONGCTX_TOKENS=64000 ARM_TIMEOUT=7200 READY_TIMEOUT=2400 \
        "$BIN/run_arm.sh" "$label" 70B w8a8 vllm-aiter "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. baseline, shipped defaults -----------------------------------------
# Re-measured here rather than quoted from docs/28, because round 17 showed how
# a contaminated or differently-flagged baseline manufactures a headline.
run t70ar-base

# --- B. fused allreduce + RMSNorm ------------------------------------------
# Compact JSON, no spaces, so it survives as a single argv element.
run t70ar-fused \
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'

# --- C. all three ROCm-aware fusions ---------------------------------------
# fuse_act_padding gates RocmAiterTritonAddRMSNormPadFusionPass and
# fuse_norm_quant gates another ROCm branch in the same block; both are
# likewise unset by default. Run together only after B isolates the
# allreduce effect on its own.
run t70ar-allfuse \
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":true,"fuse_act_padding":true,"fuse_norm_quant":true}}'

echo "=== $(date -u +%T) round 24 done ==="
echo
echo "============ DID THE FUSED ALLREDUCE KERNEL LOAD? ============"
for l in t70ar-base t70ar-fused t70ar-allfuse; do
    log="logs/$l.serverlog"
    [ -f "$log" ] || { echo "  $l: no serverlog"; continue; }
    co=$(grep -o "LoadKernel.*allreduce[^ ]*\.co" "$log" 2>/dev/null | sed "s|.*/||" | sort -u | head -3)
    if [ -n "$co" ]; then
        echo "  $l: *** ALLREDUCE ASM LOADED ***"
        echo "$co" | sed "s/^/      /"
    else
        echo "  $l: no allreduce .co"
    fi
    grep -oE "RocmAiterAllReduceFusionPass|rocm_aiter_allreduce_fusion_pass|replaced [0-9]+ patterns" "$log" 2>/dev/null \
        | sort -u | head -3 | sed "s/^/      pass: /"
done
echo
echo "--- vs docs/28 baseline: 843 prefill @15k / 6.8 decode @64k ---"
for l in t70ar-base t70ar-fused t70ar-allfuse; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-16s %-8s prefill=%8.1f decode=%7.2f' % (sys.argv[2], sys.argv[3],
    d.get('implied_prefill_tps_median') or 0, d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
    f="results/$l-FAILED.json"
    [ -f "$f" ] && python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
print('  %-16s FAILED: %s' % (sys.argv[2], d.get('reason')))" "$f" "$l"
done
