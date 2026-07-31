#!/usr/bin/env bash
# Generate the MI210 fused_moe config that vLLM does not ship.
#
# vLLM ships 317 tuned fused_moe configs. None is for an MI210, so gfx90a tile
# selection falls to heuristics, and docs/33 measured what that costs: decode
# shapes select 16x16x16bf16_1k at 191-196 MACs per issued instruction while
# larger shapes select 32x32x8bf16_1k at 485-497, with ~450-500 instructions of
# fixed overhead either way. Decode sits on the wrong side of a 2.5x spread.
#
# THE FILENAME THIS MUST PRODUCE ----------------------------------------------
#
#   E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
#
# WITH the dtype tag. An earlier revision of this header argued the opposite at
# length and was wrong; it is worth recording why, because the reasoning looked
# solid.
#
# It read fused_moe.py:1595, saw _get_config_dtype_str called with
# use_fp8_w8a8 / use_int8_w8a16 / use_int4_w4a16 and NO use_int8_w8a8 parameter,
# checked config.py:36-66 for a branch that did not exist, and concluded a W8A8
# MoE falls through to `return None` and therefore looks up an untagged file.
# Every one of those observations is true. The conclusion was still wrong,
# because the compressed-tensors W8A8 serving path sets use_int8_w8a16=True --
# so the "no use_int8_w8a8 parameter" fact means the opposite of what it looked
# like.
#
# Round 34 settled it empirically, and only because it asserted on the load
# before reading any number. vLLM logged:
#
#   Using default MoE config. Performance might be sub-optimal! Config file not
#   found at /models/bench-matrix/moe-configs-mi210/E=128,N=384,
#   device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
#
# So docs/25 item 3b, tune_moe_w8a8_v4.sh's header and the docstring of
# configs/fix_benchmark_moe_int8_w8a16.py were right all along, and ~7.5
# GPU-hours were spent producing a well-formed file with the wrong name. It
# cannot be salvaged by renaming: `--dtype auto` benchmarked a different kernel,
# and a config generated under the wrong scheme is worse than none.
#
# The genuine defect in the v1/v2 attempts remains `--tp-size 1`, which gives
# N=768 instead of N=384.
#
# This round therefore needs configs/fix_benchmark_moe_int8_w8a16.py applied to
# benchmark_moe.py first -- without it the int8_w8a16 path dies in const_data_ptr
# before measuring anything, which is what blocked this for weeks.
#
# N=384 because moe_intermediate_size is 768 and the serving arm is TP=2.
# E=128 from num_experts. Both read from the checkpoint, asserted below.
#
# ONE BATCH SIZE PER INVOCATION, MERGED ---------------------------------------
#
# An earlier revision of this header argued the opposite -- pass the whole list
# in one invocation, since benchmark_moe.py:996 builds {batch_size: config}
# across the list and save_configs writes ONE file, whereas v1 invoked it per
# size and each run overwrote the last.
#
# Both halves of that are true and the conclusion was still wrong, because
# save_configs runs once at the END. When a Ray worker died partway through
# (see the loop below) the three completed batch sizes went with it. v1's real
# mistake was not the per-size invocation, it was failing to MERGE between them
# -- which is exactly what tune_moe_merge.sh in this directory exists to do.
#
# So: one size per invocation, merged after each, and reruns resume.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
OUT=$BASE/moe-configs-mi210
RAW=$BASE/logs/rd33-tuner.log
WANT="E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json"
IMG=vllm-mi210:latest
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 33: fused_moe tuning for MI210 ==="

# Assert the shape before spending GPU-hours on the wrong one.
python3 - <<'PY' || exit 1
import json, sys
c = json.load(open("/mnt/llm-storage/bench-matrix/t35-w8a8/config.json"))
e, inter = c["num_experts"], c["moe_intermediate_size"]
n = inter // 2
print(f"  num_experts={e}  moe_intermediate_size={inter}  -> N at TP=2 = {n}")
if (e, n) != (128, 384):
    print(f"FATAL: expected E=128,N=384, got E={e},N={n}", file=sys.stderr)
    sys.exit(1)
PY

# NOT wiped. The tuner crashes at unpredictable points, so a rerun must resume
# rather than restart -- otherwise the per-size merge below buys nothing across
# invocations. Sizes already present in the merged file are skipped.
mkdir -p "$OUT"

# bench- prefix so wait_for_bench.sh can see it. The previous tuning scripts
# named this container `moe-tune`, which is invisible to that check -- the same
# defect that let `nccl-probe` hold 41.88 GiB while the next round started arms
# on top of it (docs/29).
docker rm -f bench-moe-tune >/dev/null 2>&1 || true
echo "=== $(date -u +%T) tuning: 8 batch sizes, one invocation each, merged ==="
echo "    target: $WANT"
# ONE BATCH SIZE PER INVOCATION, MERGED AFTER EACH -- because the tuner dies.
#
# The first attempt passed all 12 sizes in one invocation. It completed 1, 2 and
# 4, then a Ray BenchmarkWorker vanished mid-run:
#
#   The actor is dead because its worker process has died.
#   Worker exit type: SYSTEM_ERROR ... connection error code 2. End of file.
#
# Ray's message lists the OOM killer first, and that is NOT what happened here:
# dmesg shows no OOM, the host had 461 GB free, and there was no amdgpu fault or
# HIP error. The worker simply died. Since save_configs() is called once at the
# END of the whole invocation, all three completed batch sizes were lost too.
#
# So the failure mode to design against is not the timeout -- it is a crash at
# an unpredictable point. tune_moe_merge.sh already established the fix for the
# adjacent problem (the tuner rewrites the whole file each run, so tuning one
# size at a time overwrites the last): snapshot after each run and merge.
# Applying it here makes a crash cost ONE batch size instead of everything.
#
# THE BATCH SIZES ARE CHOSEN, not merely fewer than the 19-size default.
# docs/33's finding is that DECODE sits on the wrong side of a 2.5x tile spread,
# and decode is small-M: 1-64 covers single-stream and modest concurrency, which
# is every decode measurement in this repo. 128-2048 covers chunked prefill at
# -ub 2048. Dropping 3072/4096 costs nothing we serve.
#
# A partial config is not a broken one, which is what makes this safe:
# get_moe_configs returns a dict keyed by M and try_get_optimal_moe_config picks
# the nearest entry, so any size we fail to tune falls back to exactly the
# heuristics it uses today.
MERGED="$OUT/$WANT"
declare -a DONE_SIZES=() FAILED_SIZES=()

for bs in 1 2 4 8 16 32 64 128; do
    if [ -f "$MERGED" ] && python3 -c "
import json,sys; sys.exit(0 if '$bs' in json.load(open('$MERGED')) else 1)" 2>/dev/null; then
        echo "--- batch_size=$bs already tuned, skipping ---"
        DONE_SIZES+=("$bs")
        continue
    fi
    echo ""
    echo "--- $(date -u +%T) batch_size=$bs ---"
    rm -rf "$OUT/staging"; mkdir -p "$OUT/staging"
    docker rm -f bench-moe-tune >/dev/null 2>&1 || true
    # 90 min per size. The first attempt used 45 and hit exit 124 on every size
    # from 8 upward -- six consecutive timeouts, ~4.5 GPU-hours for nothing.
    timeout 5400 docker run --rm --name bench-moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 "$IMG" \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
        --tune --tp-size 2 --dtype int8_w8a16 --seed 1234 --batch-size "$bs" \
        --save-dir /models/bench-matrix/moe-configs-mi210/staging \
        >> "$RAW" 2>&1
    rc=$?

    if [ $rc -ne 0 ] || [ ! -f "$OUT/staging/$WANT" ]; then
        echo "  batch_size=$bs FAILED (exit $rc) -- continuing"
        FAILED_SIZES+=("$bs")
        continue
    fi

    # Merge this size into the accumulating file. Done in python rather than by
    # copying, because the tuner writes the WHOLE file each run containing only
    # the size it just tuned -- a copy would discard every earlier size, which
    # is the exact bug tune_moe_merge.sh was written for.
    python3 - "$MERGED" "$OUT/staging/$WANT" <<'PY'
import json, os, sys
merged_path, new_path = sys.argv[1], sys.argv[2]
merged = {}
if os.path.isfile(merged_path):
    with open(merged_path) as fh:
        merged = json.load(fh)
with open(new_path) as fh:
    merged.update(json.load(fh))
with open(merged_path, "w") as fh:
    json.dump(merged, fh, indent=4)
    fh.write("\n")
PY
    DONE_SIZES+=("$bs")
    echo "  batch_size=$bs OK  (merged: $(python3 -c "
import json;d=json.load(open('$MERGED'))
print(','.join(sorted((k for k in d if k.isdigit()), key=int)))"))"
done

rm -rf "$OUT/staging"
echo ""
echo "tuned OK  : ${DONE_SIZES[*]:-none}"
echo "tuned FAIL: ${FAILED_SIZES[*]:-none}"
rc=0
[ ${#DONE_SIZES[@]} -gt 0 ] || rc=1

echo ""
echo "=== produced ==="
ls -la "$OUT" 2>/dev/null || echo "(nothing)"

# Gate on the EXACT filename. A tuned config generated under the wrong shape or
# the wrong dtype tag is worse than none: it is silently ignored at serve time,
# so the arm reports "no improvement" and the hours look like a negative result.
if [ ! -f "$OUT/$WANT" ]; then
    echo ""
    echo "FAIL: expected $WANT"
    echo "Tuner output tail:"
    tail -25 "$RAW"
    exit 1
fi

echo ""
echo "OK: $WANT"
python3 - <<PY
import json
d = json.load(open("$OUT/$WANT"))
ks = sorted((int(k) for k in d if k.isdigit()))
print(f"  batch sizes tuned: {len(ks)} -> {ks}")
print(f"  sample (M={ks[0]}): {d[str(ks[0])]}")
PY

echo ""
echo "NOT INSTALLED. Copy into the image and A/B it before trusting it:"
echo "  configs dir: /opt/python/lib/python3.14/site-packages/vllm/"
echo "               model_executor/layers/fused_moe/configs/"
echo "A tuned config is a claim about this box; it needs a measurement, not a"
echo "filename check. Run the W8A8 TP=2 arm with and without it."
