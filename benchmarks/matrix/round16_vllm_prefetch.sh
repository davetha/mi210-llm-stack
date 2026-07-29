#!/usr/bin/env bash
# vLLM weight offload, the backend we never tried.
#
# THE MISTAKE THIS CORRECTS. docs/28 records "vLLM's CPU offload is not viable
# at this tier" on the strength of one measurement: GLM-4.6-AWQ with
# --cpu-offload-gb 70, one 28k request running past 35 minutes at a steady 505%
# CPU. That conclusion was drawn from ONE of vLLM 0.23's TWO offload backends,
# and it happens to be the one that cannot work over PCIe.
#
#   --cpu-offload-gb   -> UVAOffloader.  Zero-copy: the weight tensor STAYS in
#                         CPU-pinned memory and the GEMM reads it across the bus
#                         as it runs. No staging, no reuse, no overlap. Every
#                         forward pass pays full PCIe latency inside the kernel.
#
#   --offload-group-size -> PrefetchOffloader.  Weights are COPIED to a static
#                         GPU buffer pool on a separate torch.cuda.Stream, with
#                         double buffering, layer N+1 landing while layer N
#                         computes. Transfer is hidden whenever compute per
#                         layer exceeds transfer per layer.
#
# vLLM has NO managed-memory path at all -- zero hits for hipMallocManaged or
# cudaMallocManaged in the whole package -- so this is not the same mechanism as
# the llama.cpp unified-memory arms in round 15. That one is OS demand paging at
# page granularity with no prefetch; this one is explicit, tensor-granular, and
# scheduled. They are two different answers to "weights in RAM, compute on GPU"
# and both deserve a number.
#
# WHY IT SHOULD WIN HERE, AND ONLY ON PREFILL. GLM-4.6 AWQ is 176 GB, of which
# ~168 GB is expert weights (89 MoE layers x 160 experts x 3 matrices x
# 5120x1536 at 4 bits = 1.89 GB/layer). At TP=2 that is ~84 GB of experts per
# rank against a 64 GB card, so roughly half must live off-card.
#
#   Moving 50-75% of 84 GB per forward pass is 42-63 GB per rank. At the ~25
#   GB/s this box gets over PCIe that is 1.7-2.5 s of transfer per pass.
#
#   Prefill amortises it: a 15k prompt at --max-num-batched-tokens 8192 is two
#   forward passes, so ~5 s of transfer against tens of seconds of compute --
#   hideable. Decode does not: one forward per token means 2.5 s per token,
#   ~0.4 t/s, and no amount of prefetching fixes that. Expect this arm to be a
#   prefill result and a decode disaster, the same shape as round 15.
#
# --offload-params experts is the whole point of the selective form: attention
# and the dense layers stay resident, only the expert stacks commute.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/glm-awq
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting vLLM prefetch-offload sweep ==="

# 28000, not the default 262144: this is a comparison against the llama.cpp
# GLM-4.6 arms, which ran --ctx-size 32768 clamped to 28000. Same prompt or the
# numbers are not comparable.
run() {  # run <label> <extra serve args...>
    local label="$1"; shift
    echo "--- $label : $* ---"
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=5400 \
        "$BIN/run_arm.sh" "$label" 400B awq4 vllm-aiter "$MODEL" \
        --tensor-parallel-size 2 \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --offload-backend prefetch \
        --offload-params experts \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. how much to offload -----------------------------------------------
# group_size/num_in_group offloads the last num_in_group layers of every
# group_size. Start at the MOST offloaded and walk back: the aggressive end is
# the one most likely to fit at all, and a load failure at 50% tells us nothing
# if we never established that the mechanism works.
#
#   4/3 = 75% offloaded -> ~21 GB of experts resident per rank
#   3/2 = 67%           -> ~28 GB
#   2/1 = 50%           -> ~42 GB, tightest fit, most compute-resident
run glm46-awq-pf75 --offload-group-size 4 --offload-num-in-group 3
run glm46-awq-pf67 --offload-group-size 3 --offload-num-in-group 2
run glm46-awq-pf50 --offload-group-size 2 --offload-num-in-group 1

# --- B. deeper prefetch ----------------------------------------------------
# prefetch_step is how many layers ahead the copy stream runs. 1 is double
# buffering; 2 is triple, which hides more latency at the cost of another slot
# in the static buffer pool (~0.94 GB per rank per slot here). Run it at 67%,
# the middle point, so it is judged against a configuration that fits
# comfortably rather than one that may already be VRAM-bound.
run glm46-awq-pf67-step2 --offload-group-size 3 --offload-num-in-group 2 \
    --offload-prefetch-step 2

# --- C. the control, deliberately last -------------------------------------
# Re-measure UVA on THIS harness so the comparison is like-for-like rather than
# against a remembered 35-minute anecdote. It is expected to be terrible and it
# is expected to be slow to produce that result, which is exactly why it runs
# after the numbers we actually want.
echo "--- UVA control (expect this to be very slow) ---"
LONGCTX_TOKENS=28000 ARM_TIMEOUT=10800 READY_TIMEOUT=5400 \
    "$BIN/run_arm.sh" glm46-awq-uva50 400B awq4 vllm-aiter "$MODEL" \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --offload-backend uva \
    --cpu-offload-params experts \
    --cpu-offload-gb 45 \
    || echo "!! glm46-awq-uva50 failed (recorded)"

echo "=== $(date -u +%T) round 16 done ==="
