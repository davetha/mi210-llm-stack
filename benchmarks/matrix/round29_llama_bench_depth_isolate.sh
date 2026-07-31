#!/usr/bin/env bash
# Isolate the llama-bench depth fault: single GPU, smallest depth, one variable.
#
# WHAT HAPPENED. Round 28 ran llama-bench with -d 0,4096,8192,16384,32768. The
# two d=0 rows completed cleanly --
#
#   pp512  2441.41 +/- 13.86 t/s
#   tg128   116.65 +/-  0.20 t/s
#
# -- and then it died entering the first non-zero depth:
#
#   [mmhub0] no-retry page fault (src_id:0 ring:144 vmid:3 pasid:35032)
#     Faulty UTCL2 client ID: SDMA1 (0x101)
#     MAPPING_ERROR:     0x1      <- the page is NOT mapped
#     PERMISSION_FAULTS: 0x2
#     RW:                0x0      <- a read
#
# THIS IS NOT THE ROUND 15 FAULT, despite hitting the same userspace assertion.
# That one was gfxhub0 / TCP (vector L1, compute) with MAPPING_ERROR 0x0 and
# RW 0x1 -- a compute kernel writing to a RESIDENT page it lacked permission on,
# preceded by a recoverable retry fault. This is mmhub0 / SDMA1 (the DMA engine)
# reading an address with NO mapping, and no retry stage at all. Different hub,
# different client, different direction, different failure mode. It also died
# cleanly rather than wedging SVM eviction workers, which fits: there is nothing
# to fault in when MAPPING_ERROR is set.
#
# THE ADDRESS SAYS OUT-OF-BOUNDS, NOT CROSS-DEVICE. llama-bench's own dump lists
#   0x738857c00000 + 0x230fa4000  VRAM  -> ends ~0x738a88ba4000
#   0x738c0ea00000 + 0x25cd000    VRAM
# and the fault address 0x738bfcc00000 falls in the GAP between them. That is a
# read past the end of an allocation, which points at a wrong size or offset in
# the depth path rather than at a missing peer mapping.
#
# SO THIS ROUND CHANGES ONE THING AT A TIME:
#   A  single GPU, d=4096 only     -- if it still faults, multi-GPU is exonerated
#                                     and the depth path itself is at fault
#   B  single GPU, full sweep      -- only runs if A survives
#   C  both GPUs, d=4096 only      -- only runs if A survives, to test whether
#                                     the second card is what breaks it
#
# The 30B Q4_K_M is 17.28 GiB, so one 64 GB card holds it plus a 32k KV cache
# comfortably -- single-GPU costs nothing here except tensor-split parallelism.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/t35-gguf-q4km
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 29: isolating the depth fault ==="

IMG=llama-rocm714-bench:latest
H=/mnt/llm-storage
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "!! $IMG missing (round 28 builds it)"; exit 1; }

GGUF=$(find "$MODEL" -name '*.gguf' | sort | head -1)
[ -n "$GGUF" ] || { echo "!! no .gguf under $MODEL"; exit 1; }

# bench <label> <hip-visible-devices|all> <depth-list>
bench() {
    local label="$1" dev="$2" depths="$3"
    local devflag=()
    [ "$dev" != "all" ] && devflag=(-e HIP_VISIBLE_DEVICES="$dev")
    echo
    echo "############ $label  (GPUs=$dev, -d $depths) ############"
    timeout 3600 docker run --rm \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v "$H":/models \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e GPU_MAX_HW_QUEUES=4 \
        "${devflag[@]}" \
        "$IMG" \
            -m "${GGUF/#$H//models}" \
            -p 512 -n 128 -d "$depths" \
            -ngl 999 -ub 2048 -fa 1 -r 3 \
        > "logs/rd29-$label.out" 2>&1
    local rc=$?
    if grep -qE "Memory access fault|VMFaultHandler" "logs/rd29-$label.out"; then
        echo "  RESULT: FAULTED (exit $rc)"
        grep -vE "^LOAD " "logs/rd29-$label.out" | grep -E "^\| qwen|Memory access fault" | head -8
        return 1
    fi
    echo "  RESULT: survived (exit $rc)"
    grep -E "^\| (model|qwen|---)" "logs/rd29-$label.out" | head -14
    return 0
}

# --- A. one GPU, smallest non-zero depth -----------------------------------
# The decisive test. Round 28 proved d=0 works on two GPUs; if d=4096 faults on
# ONE GPU then neither the second card nor the depth magnitude is responsible,
# and the depth code path is.
if bench single-d4096 0 4096; then
    # --- B. one GPU, the sweep that was originally asked for ---------------
    bench single-sweep 0 0,4096,8192,16384,32768 || true
    # --- C. two GPUs, smallest depth ---------------------------------------
    bench dual-d4096 all 4096 || true
else
    echo
    echo "A faulted -- skipping B and C. Multi-GPU is exonerated: the fault"
    echo "reproduces on a single card at the smallest non-zero depth, so it is"
    echo "the depth path, not tensor-split or scale."
fi

echo
echo "=== $(date -u +%T) round 29 done ==="
echo "per-arm output: logs/rd29-*.out"
echo
echo "For reference, round 28's surviving rows (2 GPUs, d=0):"
echo "  pp512  2441.41 t/s     tg128  116.65 t/s"
