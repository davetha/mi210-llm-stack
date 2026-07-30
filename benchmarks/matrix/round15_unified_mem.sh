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
#
# ============================================================================
# RESULT: THIS DOES NOT WORK ON THIS KERNEL, AND IT DAMAGES THE HOST.
# ============================================================================
#
# Run 2026-07-29 22:0x. The -ub 2048 arm never finished loading:
#
#   llama-server: runtime/hsa-runtime/core/runtime/runtime.cpp:2026:
#     Runtime::VMFaultHandler: Assertion `false && "GPU memory access fault."'
#
# A VM fault during weight load, i.e. the demand-paging path faulted on a
# migration and the HSA runtime aborted rather than servicing it. The abort left
# the kernel side holding an rwsem with no live owner, and amdgpu's SVM eviction
# workers piled up behind it:
#
#   INFO: task kworker/44:10 blocked for more than 122 seconds.
#   Workqueue: events svm_range_evict_svm_bo_worker [amdgpu]
#   ... blocked on an rw-semaphore, but the owner is not found.
#
# Load average reached 70 and was still climbing, on a box that also serves
# production. Killing the arm drained it (70 -> 34 -> 20) and both GPUs came
# back healthy -- rocminfo enumerated, 42-47 C idle, no reset needed -- but the
# harness did NOT stop on its own: run_arm.sh correctly recorded the failure and
# moved straight to the -ub 4096 arm, re-triggering the fault. An automated
# sweep is exactly the wrong shape for a failure mode that wedges kernel
# workers.
#
# So the answer to "can we keep weights in RAM and compute on the GPU" is: not
# by this route, on kernel 7.0.0-28 with this ROCm. The prefill-amortisation
# argument above is still sound in principle and is untested -- the fault
# happens at LOAD, before any of it is exercised. The vLLM prefetch offloader
# (round 16) reaches the same goal by explicit staged copies instead of
# demand paging, and does not involve XNACK at all.
#
# Re-running requires an explicit opt-in, because the cost of a mistake here is
# a degraded host rather than a failed benchmark.
set -uo pipefail

if [ "${ALLOW_UVM_HANG:-0}" != "1" ]; then
    cat >&2 <<'WARN'
round15 is disabled: HSA_XNACK unified memory VM-faults at load on this kernel
and leaves amdgpu svm_range_evict_svm_bo_worker threads wedged (load avg 70+,
hung-task warnings). See the header of this script. To run it anyway:

    ALLOW_UVM_HANG=1 ./bin/round15_unified_mem.sh

and watch /proc/loadavg -- kill it the moment hung-task warnings appear.
WARN
    exit 1
fi
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
