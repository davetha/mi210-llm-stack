#!/usr/bin/env bash
# Round 55: why is AWQ-Int4 SLOWER than W8A8 when it moves half the bytes?
#
# THE ANOMALY. docs/24 measured, on the same model family:
#     t35-awq   AWQ-Int4, 16 GB on disk   5.07 s
#     t35-w8a8  int8,     30 GB on disk   3.20 s
# Decode is memory-bandwidth bound. Halving the weight bytes should make it
# FASTER. It is 1.58x slower. Something in the int4 path costs more than the
# bandwidth it saves, and nobody has looked at what.
#
# THE HYPOTHESIS, from docs/39 item 1b. The W4A16 Triton kernel spends its time
# UNPACKING rather than multiplying. docs/30 established instructions-per-MFMA
# as the diagnostic for this machine, which is issue-bound rather than
# FLOP-bound: int8, fp16 and bf16 MFMA all cap at 181 TOPS on gfx90a, so a
# kernel that issues many scalar/vector ops per matrix op is throwing away the
# only resource that is scarce.
#
# WHAT THE ANSWER MEANS, decided BEFORE the measurement so it cannot be
# rationalised afterwards:
#   ratio ~100:1  -> the unpack dominates. The format is fine and the KERNEL is
#                    the problem. This confirms docs/39 item 1b and makes
#                    W4A16-fp16 with a better unpack the target -- QServe's
#                    "three logical operations per four weights" or QQQ's
#                    per-channel shift-by-4 (1 v_lshlrev_b32 per pair).
#   ratio ~10:1   -> the kernel is already tight and int4's loss is elsewhere:
#                    group-wise scale loads, tiling, or occupancy. docs/39
#                    item 1b is then MIS-AIMED and should be closed.
#
# METHOD. Triton compiles to AMDGCN and caches it. Serve the AWQ checkpoint,
# drive enough decode to force the kernel to be compiled and autotuned at real
# shapes, then read the cached .amdgcn directly and count. No profiler needed,
# no sampling error -- this is a static count of what was actually emitted.
#
# NOTE ON WHAT IS BEING COMPARED. t35-w8a8 does NOT go through Triton: the
# enable_aiter_ck_gemm_gfx90a.py carve-out routes it to CK's gemm_a8w8. So this
# round measures the int4 kernel on its own terms against the docs/30 bar; it
# is not a like-for-like Triton-vs-Triton diff, and is not presented as one.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
WORK=$BASE/awq-probe
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-awq
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }
rm -rf "$WORK"; mkdir -p "$WORK/triton-cache"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 55: AWQ-Int4 instructions-per-MFMA ==="

cleanup() { docker rm -f rd55-awq >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

# TRITON_CACHE_DIR must be on the mounted volume or the cache dies with the
# container and there is nothing left to disassemble.
TP=1 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 \
VLLM_EXTRA_ENV="-e TRITON_CACHE_DIR=/work/triton-cache -v $WORK:/work" \
    "$BIN/serve_vllm_aiter.sh" "$MODEL" rd55-awq 8105 --max-model-len 32768 >/dev/null

t=0
until curl -sf http://127.0.0.1:8105/health >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -q '^rd55-awq$'; then
        echo "FATAL: server exited"; docker logs rd55-awq 2>&1 | tail -25; exit 1
    fi
    [ $t -ge 1800 ] && { echo "FATAL: not ready in 1800s"; exit 1; }
    sleep 10; t=$((t+10))
done
echo "server ready after ${t}s"

# Drive real decode so the GEMM kernels compile and autotune at production
# shapes. A single short request is not enough -- Triton autotuning picks a
# config per shape, and we want the one decode actually uses.
echo "=== $(date -u +%T) driving decode to force kernel compilation ==="
for i in 1 2 3; do
    curl -s http://127.0.0.1:8105/v1/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"bench","prompt":"Explain paged attention in detail.","max_tokens":128,"temperature":0}' \
      >/dev/null 2>&1
done
echo "  done"
cleanup

echo ""
echo "=== $(date -u +%T) analysing cached AMDGCN ==="
docker run --rm -v "$WORK":/work --entrypoint bash "$IMG" -c '
n=$(find /work/triton-cache -name "*.amdgcn" 2>/dev/null | wc -l)
echo "cached AMDGCN kernels: $n"
if [ "$n" -eq 0 ]; then
    echo "NO KERNELS CACHED -- TRITON_CACHE_DIR did not take, or vLLM used no"
    echo "Triton GEMM for this checkpoint. Both are real findings; check which:"
    find /work -maxdepth 3 -type d | head
    exit 0
fi
printf "%-52s %8s %8s %9s\n" "kernel" "instrs" "mfma" "ins/mfma"
printf "%-52s %8s %8s %9s\n" "----------------------------------------------------" "--------" "--------" "---------"
# COUNTING NOTE, learned the hard way. `grep -c` EXITS 1 when the count is
# zero, so the idiomatic-looking `$(grep -c ... || echo 0)` yields the string
# "0\n0" for any kernel with no MFMA -- and then `[ "$mfma" -gt 0 ]` dies with
# "integer expression expected" for every kernel, printing an empty table. Pipe
# into `wc -l` instead, which always exits 0.
find /work/triton-cache -name "*.amdgcn" | while read -r f; do
    k=$(basename "$f" .amdgcn)
    # real instructions: indented mnemonics, excluding directives and labels
    ins=$(grep -E "^[[:space:]]+[a-z][a-z0-9_]*" "$f" 2>/dev/null | wc -l)
    mfma=$(grep -E "v_mfma" "$f" 2>/dev/null | wc -l)
    [ "$mfma" -gt 0 ] || continue
    awk -v k="${k:0:52}" -v a="$ins" -v b="$mfma" \
        "BEGIN{printf \"%-52s %8d %8d %9.1f\n\", k, a, b, a/b}"
done | sort -k4 -rn | head -25
echo ""
echo "=== the int4 unpack instruction mix in GEMM kernels ==="
for f in $(find /work/triton-cache -name "*.amdgcn"); do
    grep -qE "v_mfma" "$f" || continue
    echo "--- $(basename $f .amdgcn) ---"
    grep -ohE "^[[:space:]]+v_(lshrrev|lshlrev|and_or|and_b32|bfe|perm|cvt)[a-z0-9_]*" "$f" \
        | tr -d "[:blank:]" | sort | uniq -c | sort -rn | head -8
done | head -40
'

echo ""
echo "=== $(date -u +%T) round 55 done ==="
cat <<'EOF'
READING THIS. docs/30 uses instructions-per-MFMA as the issue-pressure metric
for gfx90a, which is issue-bound not FLOP-bound. A GEMM kernel at ~100:1 is
spending ~99% of its issue slots on something other than matrix math, and for
an int4 kernel that something is almost certainly the dequant/unpack -- the
shift/mask/permute histogram above says so directly.

If the ratio is high, docs/39 item 1b is CONFIRMED and the fix is the unpack,
not the format: W4A16-fp16 keeps the same MFMA rate, gets a better mantissa
than W4A8-int8, and skips per-token activation quantisation entirely.
If the ratio is ~10:1, item 1b is mis-aimed and should be closed.
EOF
