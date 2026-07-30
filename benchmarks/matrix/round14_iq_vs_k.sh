#!/usr/bin/env bash
# IQ-quants vs K-quants for CPU-OFFLOADED prefill.
#
# THE HYPOTHESIS, and I think it is the biggest available win on the CPU side.
#
# I-quants (IQ3_XS, IQ2_M) reach their size by using codebook lookups. On a GPU
# that is nearly free -- the table lives in fast memory and the ALUs are idle
# waiting on bandwidth anyway. On a CPU it is not: every dot product does table
# indirection that AVX2 cannot vectorise well, and the EPYC 74F3 has AVX2 only,
# no AVX-512. K-quants (Q2_K, Q3_K, Q4_K) use plain scaled integer blocks that
# map directly onto AVX2 integer FMA.
#
# So the tradeoff INVERTS when experts move to CPU. Every tier-4 measurement so
# far used I-quants -- IQ3_XS and UD-IQ2_M -- chosen for size on the assumption
# that GPU residency was what mattered. With --n-cpu-moe 60 forcing real CPU
# work, a physically LARGER K-quant may prefill considerably faster.
#
# This is a config change rather than a code patch, which is the point: it is
# testable today and, if it holds, it changes the tier-4 recommendation in
# docs/28 from "smallest I-quant that fits" to "K-quant, sized to the placement".
#
# FALSIFIABLE. If Q2_K_XL prefills no better than IQ3_XS at the same placement,
# the codebook cost is not material here and I-quants stay the right choice --
# in which case the remaining CPU lever is thread placement (round 13) and
# nothing else.
#
# Baselines at --n-cpu-moe 60, both already measured:
#   IQ3_XS   149 t/s prefill, 101.5 s TTFT
#   (auto-fit, ~0 CPU layers: 196 t/s -- the ceiling this cannot beat)
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
DEST=$BASE/glm-gguf-q2kxl
NC=60
cd "$BASE"

. "$BIN/wait_for_bench.sh"

# FETCH BEFORE CLAIMING -- same correction as round 12, which had the identical
# bug and cost 35 minutes of idle GPU when its CDN connection stalled while it
# sat at the head of the queue.
#
# This script cost another 40 minutes for the same reason, and it did so AFTER
# round 12 was fixed. The lesson is not "round 12 had a bug"; it is that the
# claim-then-fetch shape was copied into every round script that downloads
# anything, and fixing the one that failed visibly left the copies in place.
# Rounds 15, 16 and 17 were checked and fetch nothing, so this is the last one.
#
# A download needs no GPU. Only the arm does.
if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ] \
   || ls "$DEST"/**/*.aria2 >/dev/null 2>&1 || ls "$DEST"/*.aria2 >/dev/null 2>&1; then
    echo "=== $(date -u +%T) fetching GLM-4.6 UD-Q2_K_XL (134.7 GB), lock NOT held ==="
    python3 "$BIN/fetch_model.py" unsloth/GLM-4.6-GGUF "$DEST" \
        --include UD-Q2_K_XL --connections 1 --concurrent 4 \
        || { echo "!! fetch failed"; exit 1; }
fi

bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting ==="
if ls "$DEST"/**/*.aria2 >/dev/null 2>&1 || ls "$DEST"/*.aria2 >/dev/null 2>&1; then
    echo "!! .aria2 present -- download incomplete, refusing to run"
    exit 1
fi
du -sh "$DEST"

# Same placement as the IQ3_XS baseline so the quant type is the only variable.
echo "--- K-quant at n-cpu-moe=$NC (vs IQ3_XS 149 t/s) ---"
LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" "glm-q2kxl-ncmoe$NC" 400B ud_q2_k_xl llamacpp "$DEST" \
    --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$NC" \
    || echo "!! failed (recorded)"

# And with auto-fit, to see whether the K-quant also changes the GPU-resident
# number -- it should NOT, which is a useful control on the whole hypothesis.
echo "--- K-quant with auto-fit (control; IQ3_XS got 196 t/s) ---"
NGL=auto LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" glm-q2kxl-autofit 400B ud_q2_k_xl llamacpp "$DEST" \
    --ctx-size 32768 -ub 2048 --flash-attn on \
    || echo "!! failed (recorded)"

echo "=== $(date -u +%T) round 14 complete ==="
python3 - <<'PY'
import glob, json, os
print("  IQ vs K at the same placement:")
for pat in ("glm46-iq3xs-ncmoe60", "glm-q2kxl-ncmoe60", "glm46-iq3xs", "glm-q2kxl-autofit"):
    f = f"/mnt/llm-storage/bench-matrix/results/{pat}-cold16k.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"    {pat:24} ttft={d['ttft_s_median']:7.1f}s prefill={d['implied_prefill_tps_median']:6.0f}")
PY
