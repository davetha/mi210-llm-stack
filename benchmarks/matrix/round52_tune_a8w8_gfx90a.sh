#!/usr/bin/env bash
# Round 52: tune the CK int8 GEMM for gfx90a -- the kernel that already wins.
#
# WHY THIS IS THE TOP LEAD. docs/43 measured aiter.gemm_a8w8_CK at 1.480x decode
# and it has been deployed ever since. What was never checked is whether it runs
# a TUNED config on this card. It does not:
#
#   aiter/configs/a8w8_tuned_gemm.csv   keyed on (gfx, cu_num, M, N, K, q_dtype_w)
#       26 rows gfx942
#      553 rows gfx950
#        0 rows gfx90a          <-- us
#
# So the 1.48x was measured with the kernel falling back to a default heuristic.
# The tuner calls get_gfx() and writes an arch column, so running it here
# produces gfx90a rows that did not previously exist. This is the same class of
# omission as the GFX_CU_NUM_MAP gfx90a:104 gap that was part of the original
# 1.48x fix -- a lookup table that silently does not match this card.
#
# SHAPES. t35-w8a8 is Qwen3-30B-A3B: hidden 2048, 32 heads x 128 = 4096 q,
# 4 kv heads x 128 = 512 each, TP=2. Per rank:
#     qkv_proj   N = (4096+512+512)/2 = 2560,  K = 2048
#     o_proj     N = 2048,                     K = 4096/2 = 2048
# These are exactly the two GEMMs docs/43 profiled (Triton emitted 40 and 16
# workgroups on a 104-CU card at M=1). The MoE expert GEMMs do NOT come through
# here -- they go via ck_moe_stage1/2 and are tuned from untuned_fmoe.csv, which
# is round 53.
#
# M SWEEP. Decode is M=1. Everything above is prefill/chunked-prefill. We tune
# the decode-critical range plus enough of the ramp to see whether the win is
# M-dependent.
#
# MEASUREMENT COMES FREE. --compare runs the production operator benchmark
# BEFORE and AFTER tuning and prints both, so this round self-measures without a
# separate serving arm. --update_improved keeps only shapes that actually got
# faster, so a null round cannot silently poison the deployed config.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
WORK=$BASE/tune-gfx90a
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
mkdir -p "$WORK"

# ---- the shapes ----------------------------------------------------------
# M sweep of 6 values per shape (12 shapes, ~1032 candidates). It was trimmed
# from 10 values / 1720 candidates in response to a hang that turned out not to
# be one -- see the --mp note below -- so this narrower sweep is a cost choice,
# not a necessity. Every candidate is separately JIT-compiled, so the sweep
# width sets the compile time. 1 and 2 are the decode regime that actually
# matters here; 8 and 32 cover small batches; 128 and 2048 anchor the prefill
# end so an M-dependent result stays visible.
cat > "$WORK/a8w8_untuned_gfx90a.csv" <<'CSV'
M,N,K,q_dtype_w
1,2560,2048,torch.int8
2,2560,2048,torch.int8
8,2560,2048,torch.int8
32,2560,2048,torch.int8
128,2560,2048,torch.int8
2048,2560,2048,torch.int8
1,2048,2048,torch.int8
2,2048,2048,torch.int8
8,2048,2048,torch.int8
32,2048,2048,torch.int8
128,2048,2048,torch.int8
2048,2048,2048,torch.int8
CSV
echo "shapes to tune: $(( $(wc -l < "$WORK/a8w8_untuned_gfx90a.csv") - 1 ))"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 52: tuning CK a8w8 int8 GEMM for gfx90a ==="

# --shape_grouped puts all candidates for one shape on a single GPU, removing
# cross-GPU timing variance -- this rig's decode noise floor is 1.036% (docs/46)
# and tuning decisions are made on much smaller deltas than that, so timing
# hygiene matters more here than in a serving arm.
CFG=${CFG:-/home/davec/eypc/mi210-llm-stack/configs}
[ -f "$WORK/skip_fp8_tune_instances_gfx90a.py" ] || \
  cp "$CFG/skip_fp8_tune_instances_gfx90a.py" "$WORK/" 2>/dev/null || \
  echo "WARNING: FP8 skip patch not found; build will be ~2x longer"

docker run --rm --name bench-tune-a8w8 \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --shm-size 16g \
  -v "$WORK":/work \
  -v /var/cache/mi210-ccache:/ccache \
  -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=100G \
  --entrypoint bash "$IMG" -c '
    set -o pipefail
    # Drop the FP8 tuning instances before anything builds. gen_instances.py
    # emits BOTH abI8 and abF8 for every kernel -- 83 each, and gfx90a has no
    # FP8 MFMA, so the abF8 half can never run let alone win. It is not merely
    # half the build: the first attempt at this round spent 37+ minutes with
    # ninja down to ONE straggler,
    #   a8w8_rowwise_256x256x256x128_..._intrawave_v3_abF8_dF32_eB16.cpp
    # at 2073 s of CPU on that single translation unit, while all 165 others
    # had finished. The int8 half alone had compiled in about five minutes.
    # ROOT MATTERS. There are four copies of gen_instances.py in this image and
    # the one that runs is the aiter_meta copy under site-packages, NOT
    # /src/aiter/csrc. Patching /src alone reports "patched" and changes
    # nothing -- verified by the instance mix still showing 83 abF8 afterwards.
    # Patch both so this works regardless of which aiter is imported.
    P=/work/skip_fp8_tune_instances_gfx90a.py
    if [ -f "$P" ]; then
        ok=0
        for r in /opt/python/lib/python3.14/site-packages/aiter_meta /src/aiter; do
            if [ -f "$r/csrc/ck_gemm_a8w8/gen_instances.py" ]; then
                python3 "$P" --root "$r" && python3 "$P" --root "$r" --assert-patched && ok=1
            fi
        done
        [ "$ok" = "1" ] || { echo "FATAL: FP8 skip did not apply to any root"; exit 1; }
    else
        echo "WARNING: proceeding WITHOUT the FP8 skip -- expect a long tail"
    fi
    # CWD MATTERS, and getting it wrong cost this round its first attempt.
    # `cd /src/aiter` makes `import aiter` resolve to the SOURCE tree, whose
    # jit build dir has no gemm_a8w8_manifest.h -- that header is generated,
    # and with zero gfx90a rows in the tuned CSV there is nothing to generate
    # it from, so the build dies with "gemm_a8w8_manifest.h file not found".
    # From a neutral cwd, `import aiter` resolves to site-packages, which ships
    # a PREBUILT module_gemm_a8w8.so and its manifest -- that prebuilt module
    # is what the deployed 1.48x actually runs on. Verified from here:
    #   aiter -> /opt/python/lib/python3.14/site-packages/aiter
    #   gfx   -> gfx90a,  cu_num -> 104
    cd /tmp
    # --mp 1 here is a leftover from a MISDIAGNOSIS, recorded so nobody repeats
    # it. The first attempt (--mp 2 --shape_grouped) went quiet for 26 minutes
    # after "Distributing 20 task groups across 2 GPUs" with both GPUs at 0%,
    # and that was read as a multiprocessing deadlock -- the parent sat in
    # futex_do_wait, a worker in hrtimer_nanosleep. It was NOT hung. The
    # process tree also contained ninja, hipcc and clang-23: the tuner was
    # JIT-COMPILING ~1720 CK kernel candidates. That phase is CPU-bound, emits
    # no progress output, and necessarily leaves the GPU idle until
    # benchmarking starts. Confirmed on the rerun: object-file count climbing
    # and load average ~38 under `ninja -j 38`.
    #
    # HOW TO TELL THE DIFFERENCE, since a silent GPU-idle tuner looks alarming:
    #   docker exec <c> ps -eo pid,stat,wchan:16,comm   # ninja/hipcc => building
    #   find <jit build dir> -name '*.o' | wc -l        # should be climbing
    # A real hang has neither compiler processes nor growing object files.
    #
    # --mp 1 is kept only because re-testing parallelism costs more than it
    # saves; --mp 2 was never shown to be broken.
    python3 /src/aiter/csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
        -i /work/a8w8_untuned_gfx90a.csv \
        -o /work/a8w8_tuned_gfx90a.csv \
        -o2 /work/a8w8_profile_gfx90a.csv \
        --mp 1 --compare --update_improved \
        --min_improvement_pct 1.0 \
        2>&1
  ' 2>&1 | tee "$WORK/round52.log"
rc=${PIPESTATUS[0]}
echo "tuner rc=$rc"

echo ""
echo "=== $(date -u +%T) round 52 done ==="
if [ -f "$WORK/a8w8_tuned_gfx90a.csv" ]; then
    echo "tuned rows produced: $(( $(wc -l < "$WORK/a8w8_tuned_gfx90a.csv") - 1 ))"
    echo "--- arch column check (MUST say gfx90a, else the run was mis-targeted) ---"
    tail -n +2 "$WORK/a8w8_tuned_gfx90a.csv" | cut -d, -f1 | sort | uniq -c
    echo "--- tuned result ---"
    column -s, -t < "$WORK/a8w8_tuned_gfx90a.csv" 2>/dev/null | head -25
else
    echo "NO TUNED CSV PRODUCED -- see $WORK/round52.log"
fi
echo ""
echo "The --compare block in the log above is the actual result: it benchmarks"
echo "the production operator before and after tuning. A row that did not"
echo "improve by >=1% was deliberately NOT written to the tuned CSV, so an"
echo "empty or short output here is a real null, not a failure."
