#!/usr/bin/env bash
# Third pass at the arms that keep dying on harness problems rather than on
# anything about the hardware.
#
# Two distinct causes, both now fixed:
#
# 1. PORT RACE. `docker rm -f` returns before the daemon releases the port
#    mapping, so back-to-back arms collided:
#      Bind for 0.0.0.0:8100 failed: port is already allocated
#    The arm was recorded as "server would not start" with a ZERO-BYTE server
#    log, which reads exactly like a model that cannot load. That is how the
#    80B W8A8 decode arm was lost -- it never created a container at all.
#    run_arm.sh now waits for the port and sweeps stale holders.
#
# 2. WRONG SWEEP RANGE. --n-cpu-moe 30 and 45 OOM'd:
#      allocating 70673.58 MiB on device 1: cudaMalloc failed: out of memory
#    -ngl 999 disables auto-fit, so everything except N expert layers goes to
#    GPU. GLM-4.6 IQ3_XS is 139 GB against 135.57 GB of VRAM, so leaving only
#    30-45 expert layers on CPU asks for more than fits. The viable range
#    starts around 60, not below it.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

echo "=== $(date -u +%T) waiting for earlier work ==="
while ps -eo cmd | grep -qE "[b]in/round2_followup.sh|[b]in/round3_prefill.sh|[b]in/round4_spec.sh|[f]etch_model.py" \
   || docker ps --format '{{.Names}}' | grep -q '^bench-'; do
    sleep 120
done
echo "=== $(date -u +%T) starting ==="

# ---------------------------------------------------------------------------
# R1  80B W8A8 decode -- third attempt.
#
# Attempt 1 produced decode_tps=null because the prompt asked for one word and
# the Instruct build answered in 3 tokens, below MIN_TOKENS_FOR_DECODE_RATE.
# Attempt 2 died on the port race. bench_matrix.py now asks for a count to 60
# when max_tokens is large enough to measure decode.
#
# This is the missing half of the tier-2 comparison. Prefill already favours
# W8A8 at 7,253 vs 6,679 t/s; the W8A16 build decodes 51.3 t/s at 101k.
# ---------------------------------------------------------------------------
echo "--- R1: 80B W8A8 decode ---"
LONGCTX_TOKENS=110000 ARM_TIMEOUT=7200 \
    "$BIN/run_arm.sh" t80-w8a8-decode 80B w8a8 vllm-aiter "$BASE/t80-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    || echo "!! R1 failed (recorded)"

# ---------------------------------------------------------------------------
# R2  --n-cpu-moe sweep over the range that actually fits.
#
# Anchor: auto-fit placed 135.57 of 139 GB on GPU and gave 196 t/s prefill /
# 77.5 s TTFT at 15k, and 8.51 t/s decode at 25.8k.
# Known point: N=60 gives 149 t/s prefill / 101.5 s TTFT -- 31% worse prefill
# for moving 60 expert layers off the GPU.
#
# docs/26 predicts decode degrades roughly LINEARLY in pinned-layer count.
# Three more points across 60..92 test that. If the curve is not close to
# linear, the bandwidth model in that doc is wrong and it must say so.
# ---------------------------------------------------------------------------
for N in 70 80 92; do
    echo "--- R2: n-cpu-moe=$N (anchor auto-fit 196 t/s; N=60 gave 149) ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm46-iq3xs-ncmoe$N" 400B iq3_xs llamacpp "$BASE/glm-gguf-iq3xs" \
        --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$N" \
        || echo "!! N=$N failed (recorded)"
done

# ---------------------------------------------------------------------------
# R3  GLM-4.6 UD-IQ2_M, once the resumed download completes.
#
# The first attempt failed on a truncated file -- "tensor
# 'blk.28.ffn_up_exps.weight' data is not within the file bounds" -- because
# the fetch was interrupted at 81 of 122.6 GB and left an .aria2 control file.
# Guarded on size so it skips rather than repeating that failure.
# ---------------------------------------------------------------------------
SZ=$(du -sb glm-gguf-ud-iq2m 2>/dev/null | cut -f1)
if [ "${SZ:-0}" -gt 120000000000 ] && ! ls glm-gguf-ud-iq2m/UD-IQ2_M/*.aria2 >/dev/null 2>&1; then
    echo "--- R3: GLM-4.6 UD-IQ2_M, fully GPU-resident test ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" glm46-udiq2m 400B ud_iq2_m llamacpp "$BASE/glm-gguf-ud-iq2m" \
        --ctx-size 32768 -ub 2048 --flash-attn on \
        || echo "!! R3 failed (recorded)"
else
    echo "R3 skipped: UD-IQ2_M incomplete (${SZ:-0} bytes, or .aria2 present)"
fi

echo "=== $(date -u +%T) round 5 complete ==="
