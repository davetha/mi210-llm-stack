#!/usr/bin/env bash
# Tier 3 (235B) -- the last cell in the matrix with no number at all.
#
# WHY IT WAS EMPTY. Not scheduling. All three vLLM-servable checkpoints fail
# differently, and the one with correct packaging is the one that will not fit:
#
#   Qwen/Qwen3-235B-A22B-GPTQ-Int4          117 GB   gptq   ~7h load, measured
#   QuantTrio/...-AWQ                       124 GB   awq    same loader; 62 GB/card leaves no KV
#   RedHatAI/...quantized.w8a8              239.5 GB compressed-tensors, will not fit
#
# llama.cpp sidesteps the loader entirely -- GGUF has no per-expert Python
# weight loader, which is the thing that makes the vLLM path cost hours
# (docs/25 item 1).
#
# QUANT CHOICE. unsloth UD-Q3_K_XL is 104.2 GB against 135.57 GB of usable
# VRAM, so it fits with ~31 GB left for KV. The alternative, IQ4_XS at 125.5 GB,
# fits with only ~10 GB -- and tier 4 already showed that squeezing under the
# line buys little (UD-IQ2_M beat IQ3_XS by 6.1% on prefill), so the comfortable
# fit with real KV headroom is the better first data point.
#
# EXPECTATION, stated up front so the result can contradict it. Qwen3-235B-A22B
# runs ~22B active parameters against GLM-4.6's ~32B and Qwen3-Next-80B's 3B.
# Prefill scales with active params, so somewhere between GLM's ~200 t/s and the
# 80B's 4,668 t/s -- nearer the low end. If it lands far outside that, the
# active-parameter model of prefill (docs/26) needs revisiting.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
DEST=$BASE/t235-gguf-q3kxl
cd "$BASE"

. "$BIN/wait_for_bench.sh"

# FETCH BEFORE CLAIMING. The first version of this claimed the bench lock and
# THEN downloaded 104 GB inside the critical section. When the CDN stalled --
# two sockets stuck in SYN-SENT, zero bytes for minutes -- this script sat at
# the head of the FIFO holding the GPUs idle while four ready rounds queued
# behind a download that was making no progress.
#
# Nothing about a download needs the GPUs. Only the arm does. So the fetch runs
# unlocked, concurrently with whatever else is benchmarking, and the claim is
# taken afterwards.
#
# aria2 runs with --continue=true, so an interrupted fetch resumes rather than
# restarting; the .aria2 control-file guard below is what catches a fetch that
# resumed but did not finish.
if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ] \
   || ls "$DEST"/**/*.aria2 >/dev/null 2>&1 || ls "$DEST"/*.aria2 >/dev/null 2>&1; then
    echo "=== $(date -u +%T) fetching UD-Q3_K_XL (104.2 GB), lock NOT held ==="
    python3 "$BIN/fetch_model.py" unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF "$DEST" \
        --include UD-Q3_K_XL --connections 1 --concurrent 4 \
        || { echo "!! fetch failed"; exit 1; }
fi

bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting ==="
# A truncated GGUF reports the right size and then dies with "data is not within
# the file bounds" -- that cost three attempts at tier 4. Refuse to start if a
# control file is still present.
if ls "$DEST"/**/*.aria2 >/dev/null 2>&1 || ls "$DEST"/*.aria2 >/dev/null 2>&1; then
    echo "!! .aria2 control file present -- download incomplete, not starting"
    exit 1
fi
du -sh "$DEST"

# NGL=auto so llama.cpp fits it itself. Forcing -ngl 999 is what OOM'd the
# tier-4 arm at 122.6 GB. LONGCTX_TOKENS matches --ctx-size (run_arm.sh would
# clamp anyway, but be explicit).
echo "--- 235B UD-Q3_K_XL, auto-fit ---"
NGL=auto LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=5400 \
    "$BIN/run_arm.sh" t235-q3kxl 235B ud_q3_k_xl llamacpp "$DEST" \
    --ctx-size 32768 -ub 2048 --flash-attn on \
    || echo "!! failed (recorded)"

echo "--- placement ---"
grep -aoE "offloaded [0-9]+/[0-9]+ layers to GPU|CPU_Mapped model buffer size = *[0-9.]+ MiB" \
    "$BASE/logs/t235-q3kxl.serverlog" 2>/dev/null | head -4
echo "=== $(date -u +%T) round 12 complete ==="
