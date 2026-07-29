#!/usr/bin/env bash
# Weights in system RAM, compute on the GPU -- via demand-paged unified memory.
#
# THE IDEA. --n-cpu-moe puts expert weights in RAM and computes them ON THE CPU:
# 24 AVX2 cores against ~204 GB/s of DDR4. The alternative is to keep the
# weights in RAM but let the GPU compute them, with the driver paging over PCIe
# on fault. That trades 204 GB/s of DDR4 for ~25 GB/s of PCIe, but buys GPU
# arithmetic instead of AVX2 arithmetic -- roughly two orders of magnitude more
# FLOPs.
#
# WHETHER THAT WINS DEPENDS ENTIRELY ON REUSE. A page fetched for the first
# token of a micro-batch stays resident for the rest of it, so the transfer
# amortises over -ub tokens. At -ub 2048 each expert's weights are moved once
# and used 2048 times. At batch 1 -- decode -- they are moved once and used
# once, which is why vLLM's UVA offload was catastrophic on a single request
# (35+ minutes, docs/24). So this is a PREFILL strategy and is expected to do
# nothing for decode.
#
# IT IS AVAILABLE HERE, which was not obvious. The GPUs default to xnack-, and
# demand paging needs xnack+:
#
#   default      gfx90a:sramecc+:xnack-     XNACK enabled: NO
#   HSA_XNACK=1  gfx90a:sramecc+:xnack+
#
# and the llama.cpp build already contains GGML_CUDA_ENABLE_UNIFIED_MEMORY and
# hipMallocManaged. Both halves exist; nothing here has ever set them.
#
# BASELINES at 15k prefill on GLM-4.6 IQ3_XS (139 GB against 135.57 GB VRAM):
#   auto-fit, ~3 GiB on CPU        196 t/s   <- the number to beat
#   --n-cpu-moe 60, CPU compute    149 t/s
#   -ngl 999 without unified mem   CUBLAS_STATUS_ALLOC_FAILED (this is the OOM
#                                  that unified memory is supposed to convert
#                                  into paging rather than failure)
#
# RISK, and why this does not touch the vLLM arms. xnack+ is a different code
# object target from xnack-. The 242 translated AITER ASM kernels were built
# for the default, so enabling XNACK globally could make them unloadable. It is
# set per-container here, on llama.cpp arms only.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/glm-gguf-iq3xs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others

echo "=== $(date -u +%T) sanity: does unified memory actually engage? ==="
docker run --rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
    -e HSA_XNACK=1 --entrypoint bash llama-rocm714:latest -c \
    'rocminfo 2>/dev/null | grep -iE "xnack" | head -2' || true

# GGML_CUDA_ENABLE_UNIFIED_MEMORY makes ggml allocate managed memory, so -ngl 999
# no longer has to fit in VRAM. -ub is swept because the whole economics is
# amortisation: bigger micro-batches mean each paged-in expert is used more.
for UB in 2048 4096 8192; do
    echo "--- unified memory, -ngl 999, -ub $UB (vs 196 auto-fit / 149 cpu-moe) ---"
    LLAMA_EXTRA_ENV="-e HSA_XNACK=1 -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1" \
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm-uvm-ub$UB" 400B iq3_xs llamacpp "$MODEL" \
        --ctx-size 32768 -ub "$UB" --flash-attn on -ngl 999 \
        || echo "!! ub=$UB failed (recorded)"
done

echo "=== $(date -u +%T) round 15 complete ==="
python3 - <<'PY'
import glob, json, os
print("  RAM-resident weights, GPU compute (paged):")
for f in sorted(glob.glob("/mnt/llm-storage/bench-matrix/results/glm-uvm-*-cold16k.json")):
    d = json.load(open(f))
    print(f"    {os.path.basename(f)[:-13]:18} ttft={d['ttft_s_median']:7.1f}s "
          f"prefill={d['implied_prefill_tps_median']:6.0f}")
print("  baselines: auto-fit 196 t/s | --n-cpu-moe 60 (CPU compute) 149 t/s")
PY
