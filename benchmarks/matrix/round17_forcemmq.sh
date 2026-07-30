#!/usr/bin/env bash
# Does llama.cpp's MMQ-vs-hipBLAS cutoff hold on gfx90a?
#
# THE FINDING THIS TESTS. llama.cpp needs no MFMA patches -- ggml/src/ggml-cuda/
# mma.cuh already emits v_mfma_i32_16x16x16i8 for CDNA2, the K=16 form, with
# gfx942's K=32 variant correctly excluded. The matrix-core surface is closed.
#
# What is NOT settled is the DISPATCH. ggml_cuda_should_use_mmq() picks between
# those MFMA kernels and dequantize-to-fp16 + hipBLAS, and its CDNA branch ends:
#
#     if (n_experts > 64 || ne11 <= 128) return true;
#     if (Q4_0|Q4_1|Q5_0|Q5_1)           return true;
#     if (ne11 <= 256 && (Q4_K|Q5_K))    return true;
#     return false;
#
# Every MoE model here clears n_experts > 64 (GLM-4.6 160, Qwen3-30B 128), so
# the expert matmuls already use MFMA and cannot change. The attention
# projections and dense layers can: at -ub 2048 their batch dimension is
# ne11 = 2048, an order of magnitude past the 256 cutoff, so they dequantize to
# fp16 and call hipBLAS on every prefill.
#
# That 256 is an upstream guess about where rocBLAS overtakes MMQ, and it was
# not measured on this box. Upstream ROCm heuristics have been wrong here twice
# -- rocWMMA flash attention is 18-26% SLOWER on gfx90a (docs/22), and the
# paged-attention 128k ceiling cost 10x decode (configs/extend_rocm_pa_256k_gfx9).
# Two for two is reason to measure, not reason to assume.
#
# EXPECT A SMALL EFFECT, POSSIBLY NEGATIVE. On a sparse MoE most of the FLOPs
# are in the experts, which this flag does not touch. If the heuristic is right,
# forcing MMQ makes attention slower and the arm loses. A loss is a publishable
# result: it closes the question instead of leaving it open.
#
# BASELINES ARE RE-RUN HERE, NOT QUOTED. The historical 195.9 t/s came from an
# arm whose exact flags are only partly recoverable from the logs -- the
# t35-q4km serverlog is gone entirely. Running baseline and forcemmq
# back-to-back in this script means both see identical flags, identical harness
# and identical machine state, so any difference is the flag. Quoting the old
# number would have risked attributing a flag change to a forgotten argument.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 17 ==="

# THE BUILD RUNS UNDER THE LOCK, DELIBERATELY, AND IT IS THE WRONG SHAPE.
# Round 12 taught the opposite lesson -- a 104 GB download held the queue head
# with the GPUs idle, so fetching moved outside the lock. A compile is not the
# same case. It saturates 48 threads, and rounds 13 and 14 are measuring CPU
# thread scaling for expert offload; building alongside them would corrupt the
# numbers they exist to produce.
#
# So this costs roughly 30 minutes of idle GPU, knowingly. The alternative --
# a second synchronisation primitive meaning "wait for CPU-sensitive work
# only" -- is how the fifth deadlock of this project got written, inside the
# helper built to prevent deadlocks. One total order, no exceptions.
IMG=llama-rocm714-forcemmq:latest
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "=== $(date -u +%T) building $IMG (~30 min, GPUs idle) ==="
    docker build -t "$IMG" -f "$BASE/configs/Dockerfile.llama-forcemmq" "$BASE/configs" \
        || { echo "!! build failed -- no arms run, nothing recorded"; exit 1; }
fi
echo "=== $(date -u +%T) build present ==="

# run <label> <image> <model-dir> <extra args...>
#
# NGL is passed through the explicit ARM_NGL variable rather than by writing
# `NGL=auto run ...`. A `VAR=val funcname` prefix in bash does not reliably
# export VAR to the grandchild process, and serve_llamacpp.sh reads NGL from
# its environment. Silently losing it would default NGL to 999, which on
# GLM-4.6 does not fail loudly -- it disables llama.cpp's own fitting and dies
# with CUBLAS_STATUS_ALLOC_FAILED, recorded as a model problem.
run() {
    local label="$1" image="$2" model="$3"; shift 3
    echo "--- $label  (image=$image, NGL=${ARM_NGL:-999}) ---"
    LLAMA_IMAGE="$image" NGL="${ARM_NGL:-999}" \
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "$label" "$TIER" "$QUANT" llamacpp "$model" "$@" \
        || echo "!! $label failed (recorded)"
}

# --- A. Qwen3-30B-A3B Q4_K_M ----------------------------------------------
# The fast pair, and the more sensitive one: Q4_K attention weights sit exactly
# on the branch this flag overrides (Q4_K is allowed MMQ only at ne11 <= 256).
TIER=35B QUANT=q4_k_m
M35=$BASE/t35-gguf-q4km
run t35-q4km-mmqbase "llama-rocm714:latest" "$M35" --ctx-size 32768 --flash-attn on
run t35-q4km-forcemmq "$IMG"                "$M35" --ctx-size 32768 --flash-attn on

# --- B. GLM-4.6 IQ3_XS ----------------------------------------------------
# NGL=auto reproduces the tier-4 auto-fit placement (135.57 GB on GPU), which
# round 11 showed beats every manual --n-cpu-moe value. Forcing 999 here would
# OOM, and pinning --n-cpu-moe would measure a placement we know is worse.
TIER=400B QUANT=iq3_xs ARM_NGL=auto
MGLM=$BASE/glm-gguf-iq3xs
run glm46-iq3xs-mmqbase  "llama-rocm714:latest" "$MGLM" --ctx-size 32768 --flash-attn on
run glm46-iq3xs-forcemmq "$IMG"                 "$MGLM" --ctx-size 32768 --flash-attn on

echo "=== $(date -u +%T) round 17 done ==="
echo "--- MMQ dispatch A/B ---"
for w in cold16k longctx; do
    for l in t35-q4km-mmqbase t35-q4km-forcemmq glm46-iq3xs-mmqbase glm46-iq3xs-forcemmq; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-28s %-8s prefill=%8.1f decode=%6.2f ok=%s' % (
    sys.argv[2], sys.argv[3], d.get('implied_prefill_tps_median') or 0,
    d.get('decode_tps_median') or 0, d.get('correctness_probe_pass')))" "$f" "$l" "$w"
    done
done
