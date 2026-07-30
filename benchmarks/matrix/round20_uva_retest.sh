#!/usr/bin/env bash
# UVA offload, retested properly. The original test made the worst possible choice.
#
# WHAT WAS MEASURED, AND WHY IT WAS NOT A FAIR TEST. docs/28 records vLLM's UVA
# offload as unusable at tier 4: GLM-4.6 AWQ with --cpu-offload-gb 70, one 28k
# request running past 35 minutes at a steady 505% CPU. That run passed a byte
# budget and nothing else, and vLLM's own documentation says what that means:
#
#   cpu_offload_params: "If this set is empty, parameters are offloaded
#   NON-SELECTIVELY until the memory limit defined by cpu_offload_gb is reached."
#
# So it offloaded whatever it walked into first, until 70 GB was gone. On a model
# whose parameter iteration order starts with embeddings and early attention
# blocks, that means ATTENTION WEIGHTS ended up in host memory, read across PCIe
# inside every attention op of every layer. That is the one category that must
# never be offloaded, and the test had no way to avoid it.
#
# THE FIX IS ONE FLAG. --cpu-offload-params experts restricts UVA to the expert
# stacks -- ~168 GB of GLM-4.6's 176 GB, so the byte budget is still satisfiable
# -- and keeps attention, embeddings and norms resident. This flag exists in vLLM
# 0.23 and was never used.
#
# WHY IT MIGHT STILL WORK DESPITE HAVING NO PREFETCH. UVA is zero-copy: the GEMM
# reads the weight from pinned host memory as it runs, with no staging buffer and
# no overlap. But amortisation does not require staging -- within a single GEMM a
# weight tile is read once and reused across the whole batch dimension. At
# --max-num-batched-tokens 8192 that is thousands of tokens per fetch, the same
# arithmetic that makes the prefetch backend work. What UVA loses is the ability
# to hide the transfer behind compute, not the ability to amortise it.
#
# So the honest prediction is: slower than prefetch's 695.9 t/s, but not the
# 35-minute catastrophe -- and if it IS still catastrophic with attention
# resident, that isolates the cost to the absence of overlap rather than to
# offloading itself, which is worth knowing either way.
#
# --enforce-eager is held constant against the prefetch arms. UVA does not splice
# a private stream into graph capture the way PrefetchOffloader does, so it likely
# does not need it; keeping it fixed removes a variable from the comparison.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/glm-awq
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 20 (UVA retest, selective) ==="

# ARM_TIMEOUT is generous but NOT unbounded. The original failure ran past 35
# minutes on a single request; 5400 s per request lets a merely-slow result
# complete and be recorded, while still terminating a repeat of the catastrophe
# instead of hanging the queue overnight.
run() {  # run <label> <extra serve args...>
    local label="$1"; shift
    echo "--- $label : $* ---"
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=5400 READY_TIMEOUT=5400 \
        "$BIN/run_arm.sh" "$label" 400B awq4 vllm-aiter "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --enforce-eager \
        --offload-backend uva \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. selective UVA, experts only ----------------------------------------
# 45 GB/rank of experts on the host leaves ~21 GB of experts plus all attention
# resident -- roughly the same split as the prefetch 75% arm, so the two are
# comparable and the only difference is staging-and-overlap versus zero-copy.
run glm46-awq-uva-experts45 --cpu-offload-params experts --cpu-offload-gb 45

# --- B. less offloaded, more resident --------------------------------------
# If A is bound by PCIe traffic rather than by a fixed per-op cost, halving the
# offloaded bytes should roughly halve the penalty. If A and B are equally bad,
# the cost is not proportional to traffic and zero-copy is structurally wrong
# here regardless of how little is offloaded.
run glm46-awq-uva-experts25 --cpu-offload-params experts --cpu-offload-gb 25

echo "=== $(date -u +%T) round 20 done ==="
echo "--- UVA selective vs prefetch (prefetch 75%: 695.9 prefill / ~0.4 decode) ---"
for l in glm46-awq-uva-experts45 glm46-awq-uva-experts25; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-28s %-8s ttft=%8.1fs prefill=%8.1f decode=%6.2f' % (sys.argv[2], sys.argv[3],
    d.get('ttft_s_median') or 0, d.get('implied_prefill_tps_median') or 0,
    d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
    f="results/$l-FAILED.json"
    [ -f "$f" ] && python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
print('  %-28s FAILED: %s' % (sys.argv[2], d.get('reason')))" "$f" "$l"
done
