#!/usr/bin/env bash
# Finish the --n-cpu-moe sweep. Its decode half never ran, and decode is the
# half that actually tests the bandwidth model in docs/26.
#
# WHY IT FAILED. The arms serve with --ctx-size 32768, but the longctx workload
# defaults to 262144 tokens, so every rep got:
#
#   HTTP 400: request (241846 tokens) exceeds the available context size (32768)
#
# All four arms were recorded as failures that looked like placement or OOM
# problems. The anchor run set LONGCTX_TOKENS=28000 explicitly; I dropped it
# when writing the sweep. run_arm.sh now clamps to 85% of --ctx-size when
# LONGCTX_TOKENS is unset, so this cannot recur silently -- but this script sets
# it anyway, to match the anchor exactly rather than approximately.
#
# WHAT IT TESTS. docs/26 predicted decode degrades roughly linearly in pinned
# layer count; the prefill half already refuted that (auto-fit -> 60 costs 24%,
# 60 -> 92 costs another 4.7%, flat thereafter). Decode is where the DDR4
# bandwidth argument actually applies, so it is the fair test of the model, and
# the one number the doc is still missing.
#
# Anchor: auto-fit, 135.57 of 139 GB on GPU -> 8.51 t/s decode at 25,792 tokens.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting ==="

for N in 60 70 80 92; do
    echo "--- n-cpu-moe=$N  (anchor: auto-fit 8.51 t/s decode @25.8k) ---"
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm46-iq3xs-ncmoe$N" 400B iq3_xs llamacpp "$BASE/glm-gguf-iq3xs" \
        --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$N" \
        || echo "!! N=$N failed (recorded)"
done

# Same omission killed the UD-IQ2_M decode number, so complete tier 4 while the
# model is on disk. Auto-fit, matching the run that produced its prefill figure.
echo "--- UD-IQ2_M decode (auto-fit; prefill was 208 t/s vs IQ3_XS 196) ---"
NGL=auto LONGCTX_TOKENS=28000 ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" glm46-udiq2m 400B ud_iq2_m llamacpp "$BASE/glm-gguf-ud-iq2m" \
    --ctx-size 32768 -ub 2048 --flash-attn on \
    || echo "!! udiq2m failed (recorded)"

echo "=== $(date -u +%T) round 11 complete ==="
python3 - <<'PY'
import glob, json, os
print("  --n-cpu-moe decode sweep:")
for f in sorted(glob.glob("/mnt/llm-storage/bench-matrix/results/glm46-*ncmoe*-longctx.json")):
    d = json.load(open(f))
    print(f"    {os.path.basename(f):44} decode={d.get('decode_tps_median')}")
PY
