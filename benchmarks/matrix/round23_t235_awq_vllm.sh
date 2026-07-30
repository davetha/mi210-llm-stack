#!/usr/bin/env bash
# Qwen3-235B on vLLM+AITER: the same model we already benchmarked, on an engine
# that can actually reach the ASM kernels.
#
# WHY THIS MODEL. It is the only large checkpoint found that satisfies the whole
# fast-path profile in docs/35:
#
#   head_dim 128                     -> fmha_v3_fwd ASM eligible (bf16 objects)
#   64 heads / 4 KV = GQA ratio 16   -> pa ASM eligible; 16 is one of exactly
#                                       TWO ratios those kernels exist for
#   no kv_lora_rank                  -> standard attention, not MLA
#   torch_dtype bfloat16             -> matches the bf16-only fmha objects
#   116 GB AWQ-Int4, weights-only    -> fits 128 GB of VRAM, unpruned
#
# Every GLM-5.2 variant fails this profile structurally: MLA routes to the mla/
# family, which round 21 measured as never dispatching (VLLM_ROCM_USE_AITER_MLA
# 1 vs 0 gives 8630.8/93.34 against 8660.9/93.56, and vLLM logs "Using
# FLASH_ATTN MLA" either way). No quantization choice changes that.
#
# THE POINT: WE MEASURED THIS MODEL ON THE WRONG ENGINE. docs/28 records
# t235-q3kxl at 698.2 t/s prefill and 23.09 t/s decode -- on llama.cpp, which
# gets NO AITER at all (different stack, ggml-hip, its own mma.cuh). On the 30B
# the engine gap alone was 8,343 vs 3,416 t/s, i.e. 2.4x, before any ASM.
#
# THE VRAM FIT IS THE RISK. 116 GB at TP=2 is 58 GB/rank against 64 GB cards,
# and serve_vllm_aiter.sh passes --gpu-memory-utilization 0.90, which leaves only
# 57.6 GB. Arm A therefore overrides to 0.95 (argparse takes the last occurrence
# of a repeated flag). If that still OOMs, arm B adds a small prefetch offload --
# round 18 established the cost is GRADED, monotone in the offloaded fraction
# (50% -> 800.7, 67% -> 729.9, 75% -> 695.9 t/s), so 12.5% should cost little.
#
# Arm A deliberately does NOT use --enforce-eager. That flag was required for the
# prefetch offloader because it splices a private stream into CUDA-graph capture,
# but with nothing offloaded there is no such conflict, and graphs help decode.
# Arm B needs it for exactly the round-18 reason.
#
# AWQ, NOT GPTQ. docs/28 records gptq-packaged MoE as catastrophic --
# moe_wna16_weight_loader runs per expert per weight in single-threaded Python,
# and t235-gptq4 failed outright. AWQ is a different loader path and the tier-1
# and tier-2 AWQ arms loaded in 115-160 s.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
DEST=$BASE/t235-awq
cd "$BASE"

. "$BIN/wait_for_bench.sh"

# Fetch before claiming: rounds 12 and 14 each idled both GPUs for over half an
# hour by downloading inside the critical section.
if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ]; then
    echo "=== $(date -u +%T) fetching Qwen3-235B AWQ (~116 GB), lock NOT held ==="
    python3 "$BIN/fetch_model.py" QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ "$DEST" \
        --connections 1 --concurrent 4 \
        || { echo "!! fetch failed"; exit 1; }
fi

bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 23: Qwen3-235B AWQ on vLLM+AITER ==="

# 28000 matches the tier-4 arms so the longctx decode number is comparable.
# cold16k is directly comparable to every arm in docs/28, including the
# llama.cpp t235-q3kxl baseline of 698.2 t/s.
run() {  # run <label> <extra serve args...>
    local label="$1"; shift
    echo "--- $label : $* ---"
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "$label" 235B awq4 vllm-aiter "$DEST" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. fully resident, graphs on ------------------------------------------
run t235awq-resident --gpu-memory-utilization 0.95

# --- B. fallback: small offload if A did not fit ---------------------------
# group_size 8 / num_in_group 1 offloads one expert layer in eight = 12.5%.
if [ ! -f "results/t235awq-resident-cold16k.json" ]; then
    echo "=== arm A produced no result; trying 12.5% offload ==="
    run t235awq-pf12 --gpu-memory-utilization 0.92 --enforce-eager \
        --offload-backend prefetch --offload-params experts \
        --offload-group-size 8 --offload-num-in-group 1
fi

echo "=== $(date -u +%T) round 23 done ==="
echo
echo "=================== DID THE ASM ACTUALLY LOAD? ==================="
# The .co load line is the proof, not the backend-selection line. docs/28 makes
# this explicit because an earlier result in this repo credited AITER for a run
# that never loaded a single kernel.
for l in t235awq-resident t235awq-pf12; do
    log="logs/$l.serverlog"
    [ -f "$log" ] || continue
    n=$(grep -c "LoadKernel" "$log" 2>/dev/null | head -1)
    echo "  $l: $n LoadKernel lines"
    grep -o "LoadKernel.*\.co" "$log" 2>/dev/null | sed "s|.*/||" | sort -u | head -4 | sed "s/^/      /"
    grep -o "gqa_ratio.: [0-9]*" "$log" 2>/dev/null | sort -u | head -2 | sed "s/^/      JIT /"
done
echo
echo "--- vs llama.cpp GGUF baseline: 698.2 prefill @15k / 23.09 decode ---"
for l in t235awq-resident t235awq-pf12; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-20s %-8s ttft=%7.1fs prefill=%9.1f decode=%7.2f' % (sys.argv[2], sys.argv[3],
    d.get('ttft_s_median') or 0, d.get('implied_prefill_tps_median') or 0,
    d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
    f="results/$l-FAILED.json"
    [ -f "$f" ] && python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
print('  %-20s FAILED: %s' % (sys.argv[2], d.get('reason')))" "$f" "$l"
done
