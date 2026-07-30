#!/usr/bin/env bash
# vLLM prefetch offload, retried with CUDA graph capture disabled.
#
# WHAT FAILED. Round 16 ran --offload-backend prefetch at 75% expert offload on
# GLM-4.6 AWQ (TP=2) and died during load:
#
#   Worker_TP1: RuntimeError: NCCL error: unhandled cuda error
#     in ncclAllReduce
#
# A control settled that this was not leftover damage from the XNACK incident 18
# minutes earlier: t35-w8a8 at TP=2 loaded normally on the same GPUs -- 130 s,
# 41.88 GiB KV cache. The offloader failed on its own merits. See docs/29.
#
# WHY --enforce-eager IS THE FIRST THING TO TRY. PrefetchOffloader's own docstring
# says it "uses static buffers and event-based stream forking for torch.compile +
# CUDA graph compatibility. Events allow the copy stream to JOIN CUDA GRAPH
# CAPTURES." So it deliberately splices a private torch.cuda.Stream into graph
# capture, and it also patches module forwards to insert custom ops
# (wait_prefetch, start_prefetch) into the captured graph.
#
# RCCL collectives inside a graph capture are the fragile part of that on ROCm.
# --enforce-eager removes graph capture entirely, which is the smallest change
# that tests the hypothesis. It costs some decode throughput -- irrelevant here,
# because the offloader is a PREFILL strategy: at -ub-equivalent batch sizes each
# paged-in expert is reused across thousands of tokens, while at batch 1 it is
# used once. Decode was never the reason to want this.
#
# THE PIN-MEMORY ARM IS THE SECOND HYPOTHESIS. should_pin_memory() defaults true,
# so at 75% offload this pins roughly 63 GB of host memory per rank, ~126 GB
# total against 499 GB. vLLM exposes VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
# for systems where pinning is harmful. Run it with eager as well, so a pass
# there isolates pinning rather than confounding the two.
#
# FAILURES ARE NOW CHEAP. run_arm.sh scans the server log for worker-fatal
# patterns during the readiness poll (docs/29), so a repeat of round 16's crash
# costs about two minutes instead of the 90-minute READY_TIMEOUT it burned. That
# is what makes running four arms here reasonable rather than reckless.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/glm-awq
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 18 (prefetch offload, eager) ==="

# 28000 tokens matches the llama.cpp GLM-4.6 arms so the numbers are comparable.
#
# ARM_ENV is read here rather than written as `VLLM_EXTRA_ENV=... run ...`. A
# `VAR=val funcname` prefix in bash does not reliably export VAR to the
# grandchild process, and serve_vllm_aiter.sh reads it from its environment --
# losing it would run the arm WITHOUT the flag and report the result as though
# it had been applied. Same trap as NGL in round 17.
run() {  # run <label> <extra serve args...>
    local label="$1"; shift
    echo "--- $label : $* ${ARM_ENV:+[env: $ARM_ENV]} ---"
    VLLM_EXTRA_ENV="${ARM_ENV:-}" \
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=5400 \
        "$BIN/run_arm.sh" "$label" 400B awq4 vllm-aiter "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --enforce-eager \
        --offload-backend prefetch \
        --offload-params experts \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. does eager alone fix it? -------------------------------------------
# 75% first, the same point that crashed in round 16, so this is a direct
# controlled comparison rather than a new configuration.
run glm46-awq-pf75-eager --offload-group-size 4 --offload-num-in-group 3

# --- B. eager + no pinned host memory --------------------------------------
ARM_ENV="-e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1"
run glm46-awq-pf75-eager-nopin --offload-group-size 4 --offload-num-in-group 3
ARM_ENV=""

# --- C. if it works, sweep how much to offload -----------------------------
# Lower offload means more experts resident, so less PCIe traffic per forward
# pass but more VRAM. These only mean anything if A or B loaded at all; if both
# failed they will fail identically and cheaply.
run glm46-awq-pf67-eager --offload-group-size 3 --offload-num-in-group 2
run glm46-awq-pf50-eager --offload-group-size 2 --offload-num-in-group 1

echo "=== $(date -u +%T) round 18 done ==="
echo "--- prefetch offload with eager (baselines: llama.cpp IQ3_XS 204.5 prefill / 8.62 decode) ---"
for l in glm46-awq-pf75-eager glm46-awq-pf75-eager-nopin glm46-awq-pf67-eager glm46-awq-pf50-eager; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-30s %-8s prefill=%8.1f decode=%6.2f' % (sys.argv[2], sys.argv[3],
    d.get('implied_prefill_tps_median') or 0, d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
    f="results/$l-FAILED.json"
    [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1])); print('  %-30s FAILED: %s' % (sys.argv[2], d.get('reason')))" "$f" "$l"
done
