#!/usr/bin/env bash
# GLM-4.6 UD-IQ2_M, fourth attempt. Three previous failures, three different
# causes, none of them the experiment:
#
#   1. download interrupted at 81 of 122.6 GB -> "tensor
#      'blk.28.ffn_up_exps.weight' data is not within the file bounds"
#   2. aria2 RESUMED that file to the correct length with wrong content, so the
#      size guard passed and it failed identically. Purged and refetched clean.
#   3. serve_llamacpp.sh defaults NGL=999 unless told otherwise, which disables
#      auto-fit and forces all 122.6 GB onto the cards -> CUBLAS_STATUS_ALLOC_
#      FAILED. The IQ3_XS run that worked used NGL=auto; this one inherited the
#      default because I passed placement flags on the run_arm line instead.
#
# THE QUESTION. IQ3_XS is 139 GB against 135.57 GB of usable VRAM, so auto-fit
# left ~3 GiB on the CPU -- and prefill collapsed to 181-196 t/s while decode
# stayed a healthy 8.51 t/s. That asymmetry says a small CPU-resident fraction
# serialises prefill. UD-IQ2_M is 122.6 GB, which should fit with room for KV.
#
# If prefill jumps from ~196 to something in the thousands, the 3 GiB
# explanation is confirmed and the tier-4 lesson is "get under the VRAM line at
# any quantization cost". If prefill stays near 196, that explanation is wrong.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting ==="

# NGL=auto is the whole point: let llama.cpp fit the model itself and report
# where it put things, rather than forcing 999 and OOMing.
echo "--- UD-IQ2_M with auto-fit (anchor: IQ3_XS 196 t/s prefill, 8.51 t/s decode) ---"
NGL=auto ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" glm46-udiq2m 400B ud_iq2_m llamacpp "$BASE/glm-gguf-ud-iq2m" \
    --ctx-size 32768 -ub 2048 --flash-attn on \
    || echo "!! failed (recorded)"

echo "--- where did auto-fit put it? ---"
grep -aoE "offloaded [0-9]+/[0-9]+ layers to GPU|CPU_Mapped model buffer size = *[0-9.]+ MiB|ROCm[01] model buffer size = *[0-9.]+ MiB" \
    "$BASE/logs/glm46-udiq2m.serverlog" 2>/dev/null | head -6
echo "=== $(date -u +%T) round 7 complete ==="
