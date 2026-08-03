#!/usr/bin/env bash
# Round 77: tune fused_moe for int4_w4a16 on MI210, one batch size at a time.
#
# WHY. vLLM ships 317 tuned fused_moe configs and NONE for an MI210, so the
# W4A16 MoE runs on heuristics. The server says so directly:
#
#   Using default MoE config. Performance might be sub-optimal! Config file not
#   found at E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json
#
# That kernel is where a sparse MoE spends nearly all its compute, and round 75
# put ~82% of vLLM's decode time in latency/serialization rather than weight
# traffic -- the shape of an untuned kernel.
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID. docs/41 tuned this same kernel for
# int8_w8a16 and the tuned config made everything SLOWER:
#
#     cold 16k prefill  5,945.95 -> 4,672.43   0.786x
#     longctx prefill   4,667.07 -> 3,824.84   0.820x
#     longctx decode       50.88 ->    47.05   0.925x
#
# Not because tuning does not work, but because only M=1..32 were tuned and
# vLLM's nearest-M matching has NO FALLBACK -- prefill shapes at M>32 matched
# a config tuned for a decode shape. A PARTIAL CONFIG IS WORSE THAN NONE.
#
# So this round tunes upward through the prefill range, and the A/B that
# consumes it must gate on prefill not regressing.
#
# ONE BATCH SIZE PER INVOCATION, MERGED BETWEEN. benchmark_moe.py builds
# {batch_size: config} across the whole list and writes ONE file at the end, so
# a crash anywhere discards everything -- docs/41 lost three completed sizes to
# a Ray worker death. But the symmetric trap is that per-size invocations
# OVERWRITE the file, so each run must be merged into the accumulated config.
# This script merges after every size and keeps a per-size backup.
#
# MEASURED COST ON THIS SHAPE (E=512, N=256, int4_w4a16, TP=2): M=1 took ~14
# minutes over 8,000 candidate configurations. docs/41's shape (E=128, N=384)
# scaled roughly 7x from M=1 to M=32, so budget hours, not minutes, and expect
# the large sizes to dominate. Each size is timed and logged so the curve is
# known rather than guessed.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
OUT=$BASE/moe-configs-int4
ACC=$OUT/accumulated.json
IMG=${IMG:-vllm-mi210:v0.26.1rc0-ckgemm-warm}
MODEL=${MODEL:-$BASE/t80-awq}
CFG_NAME="E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json"

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

SIZES=${SIZES:-"2 4 8 16 32 64 128 256 512 1024 2048"}
PER_SIZE_TIMEOUT=${PER_SIZE_TIMEOUT:-10800}
mkdir -p "$OUT" "$OUT/per-size"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 77: MoE tuning int4_w4a16 E=512 N=256 ==="
echo "    sizes: $SIZES   per-size cap: ${PER_SIZE_TIMEOUT}s"
echo "    NOTE: M=1 already tuned (~14 min) and is seeded into the accumulator."

# Seed the accumulator from whatever is already tuned, so a resumed run does
# not discard earlier sizes.
if [ ! -f "$ACC" ] && [ -f "$OUT/$CFG_NAME" ]; then
    cp "$OUT/$CFG_NAME" "$ACC"
    cp "$OUT/$CFG_NAME" "$OUT/per-size/m1.json"
    echo "    seeded accumulator from existing M=1 config"
fi

merge_into_acc() {  # fresh_file
    python3 - "$ACC" "$1" <<'PY'
import json, os, sys
acc_p, new_p = sys.argv[1], sys.argv[2]
acc = json.load(open(acc_p)) if os.path.isfile(acc_p) else {}
new = json.load(open(new_p))
for k, v in new.items():
    # triton_version is metadata, not a batch size; keep the newest.
    acc[k] = v
# Sort numeric keys so the file is readable; keep non-numeric first.
meta = {k: v for k, v in acc.items() if not k.isdigit()}
nums = {k: acc[k] for k in sorted((k for k in acc if k.isdigit()), key=int)}
json.dump({**meta, **nums}, open(acc_p, "w"), indent=4)
print("  accumulated sizes:", ",".join(sorted((k for k in acc if k.isdigit()), key=int)))
PY
}

for BS in $SIZES; do
    echo ""
    echo "--- $(date -u +%T) tuning M=$BS ---"
    start=$(date +%s)
    docker rm -f probe-moetune >/dev/null 2>&1
    # Tune into a scratch dir so a failed run cannot clobber the accumulator.
    scratch=$OUT/scratch-m$BS
    rm -rf "$scratch"; mkdir -p "$scratch"
    timeout "$PER_SIZE_TIMEOUT" docker run --rm --name probe-moetune \
      --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
      --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
      -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
      -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
      --entrypoint python3 "$IMG" \
      /bin2/benchmark_moe.py --model "${MODEL/#$BASE//models/bench-matrix}" \
        --tune --tp-size 2 --dtype int4_w4a16 --seed 1234 --batch-size "$BS" \
        --save-dir "/models/bench-matrix/moe-configs-int4/scratch-m$BS" \
        > "$LOGS/moetune-m$BS.log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))
    docker rm -f probe-moetune >/dev/null 2>&1

    fresh="$scratch/$CFG_NAME"
    if [ -f "$fresh" ]; then
        cp "$fresh" "$OUT/per-size/m$BS.json"
        printf "  M=%-5s OK in %5ds  " "$BS" "$dur"
        merge_into_acc "$fresh"
    else
        printf "  M=%-5s FAILED after %5ds " "$BS" "$dur"
        [ "$rc" -eq 124 ] && echo "(TIMEOUT at ${PER_SIZE_TIMEOUT}s)" || echo "(rc=$rc, see $LOGS/moetune-m$BS.log)"
        # A missing size is not fatal to the run, but it IS fatal to deploying
        # the config -- nearest-M matching will send that shape somewhere wrong.
        echo "     ^ this gap must be recorded before the config is deployed"
    fi
    rm -rf "$scratch"
done

echo ""
echo "=== $(date -u +%T) round 77 done ==="
if [ -f "$ACC" ]; then
    cp "$ACC" "$OUT/$CFG_NAME"
    echo "final config written to $OUT/$CFG_NAME"
    python3 - "$ACC" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
sizes = sorted((int(k) for k in c if k.isdigit()))
print("tuned batch sizes:", sizes)
print()
print(f"{'M':>6}{'BLK_M':>7}{'BLK_N':>7}{'BLK_K':>7}{'GRP_M':>7}{'warps':>7}{'stages':>8}{'w/eu':>6}")
for m in sizes:
    v = c[str(m)]
    print(f"{m:>6}{v.get('BLOCK_SIZE_M','-'):>7}{v.get('BLOCK_SIZE_N','-'):>7}"
          f"{v.get('BLOCK_SIZE_K','-'):>7}{v.get('GROUP_SIZE_M','-'):>7}"
          f"{v.get('num_warps','-'):>7}{v.get('num_stages','-'):>8}{v.get('waves_per_eu','-'):>6}")
print()
gaps = [m for m in (1,2,4,8,16,32,64,128,256,512,1024,2048) if m not in sizes]
if gaps:
    print("MISSING SIZES:", gaps)
    print("vLLM's nearest-M matching has NO FALLBACK -- a shape landing on a")
    print("missing size gets the nearest TUNED config, which is how docs/41's")
    print("partial config produced 0.786x prefill. Do NOT deploy with gaps in")
    print("the prefill range without an A/B that gates on prefill.")
else:
    print("No gaps across the standard batch-size ladder.")
PY
else
    echo "no accumulated config produced"
fi
