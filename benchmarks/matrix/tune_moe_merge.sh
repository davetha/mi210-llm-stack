#!/usr/bin/env bash
# Tune fused_moe across batch sizes WITHOUT losing previous results.
#
# THE BUG THIS FIXES. benchmark_moe.py --tune --save-dir writes the whole
# config file each run, containing only the batch sizes that run tuned. So
# tuning one batch size at a time overwrites the last one. After tuning 1, 2,
# 4 and 8 across two rounds, the surviving file contained batch 4 alone:
#
#   { "triton_version": "3.7.1", "4": { ... } }
#
# Batch 1 and 2 were overwritten; batch 8 hit the 90-minute timeout and never
# wrote. Four tuning runs, one batch size kept -- and vLLM only applies a tuned
# config for batch sizes present in the file, so the rest silently fell back to
# generic heuristics.
#
# Fix: snapshot the config after each run and merge, so the file accumulates.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CFG=$BASE/moe-configs
MERGED=$BASE/moe-configs-merged
BATCHES="${BATCHES:-1 2 4 8 16 32}"
TIMEOUT="${TUNE_TIMEOUT:-5400}"
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting merge-aware MoE tuning ==="

mkdir -p "$CFG" "$MERGED"

for BS in $BATCHES; do
    echo "### batch-size $BS  $(date -u +%T)"
    docker rm -f moe-tune >/dev/null 2>&1 || true
    timeout "$TIMEOUT" docker run --rm --name moe-tune \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
        -v /mnt/llm-storage:/models -v "$BIN":/bin2 \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e NCCL_P2P_DISABLE=1 \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest \
        /bin2/benchmark_moe.py --model /models/bench-matrix/t35-awq \
        --tune --tp-size 1 --dtype auto --seed 1234 --batch-size "$BS" \
        --save-dir /models/bench-matrix/moe-configs 2>&1 | tail -3
    rc=$?
    [ "$rc" -ne 0 ] && echo "### batch $BS did not finish (rc=$rc) -- merging whatever landed"

    # Merge immediately, before the next run can overwrite it.
    python3 - "$CFG" "$MERGED" <<'PY'
import glob, json, os, sys
src, dst = sys.argv[1], sys.argv[2]
for f in glob.glob(os.path.join(src, "*.json")):
    name = os.path.basename(f)
    out = os.path.join(dst, name)
    merged = {}
    if os.path.exists(out):
        merged = json.load(open(out))
    new = json.load(open(f))
    added = [k for k in new if k.isdigit() and k not in merged]
    merged.update(new)
    json.dump(merged, open(out, "w"), indent=4)
    keys = sorted(int(k) for k in merged if k.isdigit())
    print(f"   merged {name}: added {added or 'nothing new'} -> now {keys}")
PY
done

echo "### final merged configs:"
python3 - <<'PY'
import glob, json
for f in glob.glob("/mnt/llm-storage/bench-matrix/moe-configs-merged/*.json"):
    d = json.load(open(f))
    print("  ", f.split("/")[-1], sorted(int(k) for k in d if k.isdigit()))
PY

# Re-benchmark W8A8 decode against the merged config. This is the point of the
# whole exercise: W8A8 lost decode to bf16 (43.4 vs 62.6) and to W8A16 at tier 2
# (45.19 vs 51.34), and an untuned expert GEMM at small batch is the leading
# explanation. VLLM_TUNED_CONFIG_FOLDER is forwarded by serve_vllm_aiter.sh.
echo "=== $(date -u +%T) re-benchmarking W8A8 TP=2 with merged config ==="
echo "    baseline to beat: 43.40 t/s decode @101k, 7,278 t/s prefill"
VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs-merged \
LONGCTX_TOKENS=110000 ARM_TIMEOUT=7200 \
    "$BIN/run_arm.sh" t35-w8a8-tp2-tuned 35B w8a8-tuned vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching \
    || echo "!! tuned arm failed (recorded)"

echo "--- PROOF the config was loaded (absence means tuning bought nothing) ---"
grep -aiE "tuned config|moe-configs-merged|Using configuration from" \
    "$BASE/logs/t35-w8a8-tp2-tuned.serverlog" 2>/dev/null | head -3 \
    || echo "  NO tuned-config line found -- treat any delta as unexplained"

echo "=== $(date -u +%T) merge-aware tuning complete ==="
