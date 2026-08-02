#!/usr/bin/env bash
# Round 53: tune the CK 2-stage MoE for gfx90a -- 32% of decode, never tuned.
#
# WHY THIS IS THE BIGGEST SINGLE LINE ITEM. docs/45 put the MoE cluster at
# 2560 ms / 32% of decode. docs/49 then established what actually runs it:
# vLLM -> aiter.fused_moe -> ck_moe_stage1_fwd / ck_moe_stage2_fwd, i.e.
# Composable Kernel compiled from source, NOT the hand-written ASM tree. Every
# fmoe ASM finding in docs/45-49 is orthogonal to this code path.
#
# AND IT RUNS UNTUNED HERE. aiter/configs/tuned_fmoe.csv has 617 rows and every
# single one is cu_num=80. MI210 is 104 CUs. There is no gfx column at all --
# arch is inferred from cu_num via _LEGACY_CU_NUM_TO_GFX = {256: gfx950,
# 80: gfx942, 304: gfx942}, which has no 104 entry, so it falls back to the
# live arch. Nothing in that table can match this card.
#
# THE BLOCKER, AND WHY NO PATCH IS NEEDED. gemm_moe_tune.py:2151
# get_1stage_file_info() branches on gfx950 and gfx942 with NO else, so on
# gfx90a it falls off the end returning None and the caller's tuple unpack
# raises. That looked like it required a carve-out. It does not: the call sits
# inside gen_1stage_asm_task, which is reached only via `if _want("asm")`
# (:4023), and _want is driven by the TUNE_ONLY env var (:3998-4005). Setting
#
#     TUNE_ONLY=cktile
#
# selects gen_2stages_task ONLY -- verified to wire FmoeTuner.ck_moe_stage1_fwd_out
# and ck_moe_stage2_fwd_out, exactly the two functions vLLM calls -- and skips
# both ASM generators, so the gfx90a-crashing path is never entered. This is
# the rare case where the arch gate is sidesteppable without patching anything.
#
# SHAPES. t35-w8a8 = Qwen3-30B-A3B W8A8: model_dim 2048, moe_intermediate 768,
# 128 experts, topk 8, Silu, gate+up (g1u1=1), int8 activations and weights,
# per-token quant. At TP=2 vLLM shards the intermediate dim, so the per-rank
# inter_dim is 768/2 = 384. Both 384 and 768 are tuned: 384 is what the TP=2
# deployment actually asks for, 768 covers a TP=1 replica -- which round 54 is
# about to make relevant if DP=2 wins.
#
# TOKEN SWEEP. Decode is token=1. The ramp is included because the MoE kernel
# choice is strongly token-count dependent (that is the entire premise of
# dispatch policy 0 being "tuned for large batches", which round 50 showed does
# nothing here -- possibly BECAUSE there was no tuned config to dispatch to).
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
WORK=$BASE/tune-gfx90a
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
mkdir -p "$WORK"

{
  echo "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
  for inter in 384 768; do
    for tok in 1 2 4 8 16 32 64 128 512 2048; do
      echo "$tok,2048,$inter,128,8,ActivationType.Silu,torch.bfloat16,torch.int8,torch.int8,QuantType.per_Token,1,0"
    done
  done
} > "$WORK/fmoe_untuned_gfx90a.csv"
echo "shapes to tune: $(( $(wc -l < "$WORK/fmoe_untuned_gfx90a.csv") - 1 ))"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 53: tuning CK 2-stage MoE for gfx90a ==="

# cwd must NOT be /src/aiter -- see round 52's header. From a neutral cwd
# `import aiter` resolves to site-packages and its prebuilt modules.
docker run --rm --name bench-tune-ckmoe \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --shm-size 32g --ipc=host \
  -v "$WORK":/work \
  -e TUNE_ONLY=cktile \
  --entrypoint bash "$IMG" -c '
    set -o pipefail
    cd /tmp
    echo "TUNE_ONLY=$TUNE_ONLY  (cktile => ck_moe_stage1/2 only, no ASM generators)"
    # --mp 1 to match round 52. NOTE: the round 52 header originally claimed
    # --mp 2 deadlocks. That was a misdiagnosis -- the tuner was compiling
    # kernel candidates (ninja/hipcc/clang in the process tree), which is
    # CPU-bound and silent and leaves the GPU at 0%. --mp 2 is not known to be
    # broken. Expect a long quiet compile phase here too before any GPU work.
    # (No apostrophes in this block: it is inside a single-quoted bash -c.)
    python3 /src/aiter/csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
        -i /work/fmoe_untuned_gfx90a.csv \
        -o /work/fmoe_tuned_gfx90a.csv \
        --mp 1 \
        2>&1
  ' 2>&1 | tee "$WORK/round53.log"
rc=${PIPESTATUS[0]}
echo "tuner rc=$rc"

echo ""
echo "=== $(date -u +%T) round 53 done ==="
if [ -f "$WORK/fmoe_tuned_gfx90a.csv" ]; then
    echo "tuned rows produced: $(( $(wc -l < "$WORK/fmoe_tuned_gfx90a.csv") - 1 ))"
    echo "--- cu_num column (MUST be 104, not 80 -- 80 means it tuned for the wrong card) ---"
    tail -n +2 "$WORK/fmoe_tuned_gfx90a.csv" | cut -d, -f1 | sort -n | uniq -c
    echo "--- first rows ---"
    head -6 "$WORK/fmoe_tuned_gfx90a.csv"
else
    echo "NO TUNED CSV PRODUCED"
    echo "--- if this is the gfx90a None-unpack crash, TUNE_ONLY did not take: ---"
    grep -nE "get_1stage_file_info|cannot unpack|NoneType|Traceback" "$WORK/round53.log" 2>/dev/null | head -8
fi
echo ""
echo "Unlike round 52 this tuner has no --compare, so it does NOT self-measure."
echo "A tuned CSV here is a prerequisite, not a result: the actual win (or null)"
echo "has to come from a serving arm run against docs/46's 1.036x decode bar."
