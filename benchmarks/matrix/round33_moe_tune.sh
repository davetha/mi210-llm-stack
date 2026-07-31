#!/usr/bin/env bash
# Generate the MI210 fused_moe config that vLLM does not ship.
#
# vLLM ships 317 tuned fused_moe configs. None is for an MI210, so gfx90a tile
# selection falls to heuristics, and docs/33 measured what that costs: decode
# shapes select 16x16x16bf16_1k at 191-196 MACs per issued instruction while
# larger shapes select 32x32x8bf16_1k at 485-497, with ~450-500 instructions of
# fixed overhead either way. Decode sits on the wrong side of a 2.5x spread.
#
# THE FILENAME THIS MUST PRODUCE, and a correction to the repo -----------------
#
# docs/25 item 3b, the header of tune_moe_w8a8_v4.sh, and the docstring of
# configs/fix_benchmark_moe_int8_w8a16.py all treat `--dtype auto` as defective
# because it "omits the tag", and treat `dtype=int8_w8a16` as the filename to
# aim for. For the W8A8 checkpoint we actually serve, that is backwards.
#
# Read the lookup rather than assuming it. fused_moe.py:1595 calls
# _get_config_dtype_str with use_fp8_w8a8, use_int8_w8a16 and use_int4_w4a16 --
# there is NO use_int8_w8a8 parameter, and config.py:36-66 has no branch for it.
# A W8A8 MoE therefore falls through to `return None`, and get_config_file_name
# (fused_moe.py:1006) renders `dtype_selector = ""`. The file the runtime opens
# at serve time is:
#
#   E=128,N=384,device_name=AMD_Instinct_MI210.json      <- no dtype tag
#
# So an `int8_w8a16`-tagged file would be tuned for hours and then never read by
# the W8A8 arm. `--dtype auto` was right all along; the only genuine defect in
# the v1/v2 attempts was `--tp-size 1`, which gives N=768 instead of N=384.
#
# int8_w8a16 tuning is still worth doing later for the tier-2 W8A16 checkpoints
# (cyankiwi AWQ-8bit), and configs/fix_benchmark_moe_int8_w8a16.py is what
# unblocks it -- but that is a different file and a different arm.
#
# N=384 because moe_intermediate_size is 768 and the serving arm is TP=2.
# E=128 from num_experts. Both read from the checkpoint, asserted below.
#
# ONE INVOCATION, NOT ONE PER BATCH -------------------------------------------
#
# benchmark_moe.py:996 builds {batch_size: config} across the whole list and
# save_configs writes ONE file. v1 invoked it once per batch size, so each run
# overwrote the last and only the final batch survived -- ~3 GPU-hours for one
# usable entry. Passing no --batch-size uses the full default list (line 902)
# and produces a complete config in a single pass.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
OUT=$BASE/moe-configs-mi210
RAW=$BASE/logs/rd33-tuner.log
WANT="E=128,N=384,device_name=AMD_Instinct_MI210.json"
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

rm -rf "$OUT"; mkdir -p "$OUT"

# bench- prefix so wait_for_bench.sh can see it. The previous tuning scripts
# named this container `moe-tune`, which is invisible to that check -- the same
# defect that let `nccl-probe` hold 41.88 GiB while the next round started arms
# on top of it (docs/29).
docker rm -f bench-moe-tune >/dev/null 2>&1 || true
echo "=== $(date -u +%T) tuning (full batch list, one invocation) ==="
echo "    target: $WANT"
timeout 21600 docker run --rm --name bench-moe-tune \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
    --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
    -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
    -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
    --entrypoint python3 "$IMG" \
    /bin2/benchmark_moe.py --model /models/bench-matrix/t35-w8a8 \
    --tune --tp-size 2 --dtype auto --seed 1234 \
    --save-dir /models/bench-matrix/moe-configs-mi210 > "$RAW" 2>&1
rc=$?
echo "tuner exit: $rc  (124 = hit the 6h timeout)"

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
ks = sorted(int(k) for k in d)
print(f"  batch sizes tuned: {len(ks)} -> {ks}")
print(f"  sample (M={ks[0]}): {d[str(ks[0])]}")
PY

echo ""
echo "NOT INSTALLED. Copy into the image and A/B it before trusting it:"
echo "  configs dir: /opt/python/lib/python3.14/site-packages/vllm/"
echo "               model_executor/layers/fused_moe/configs/"
echo "A tuned config is a claim about this box; it needs a measurement, not a"
echo "filename check. Run the W8A8 TP=2 arm with and without it."
