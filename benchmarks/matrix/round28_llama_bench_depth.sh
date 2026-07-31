#!/usr/bin/env bash
# llama-bench depth sweep: how much does decode cost as the KV cache grows?
#
# WHAT THIS MEASURES THAT NOTHING ELSE HERE DOES. Every arm in docs/28 reports
# throughput at one or two fixed context lengths. This sweeps depth directly --
# the model is fed n_depth tokens of context first, then prefill and decode are
# timed against that already-populated KV cache. The result is a curve rather
# than two points, which is the shape needed to separate "decode is slow" from
# "decode gets slow as KV grows".
#
# That matters for the open question in docs/25 item 1c. Decode there sits ~3.1x
# off a bandwidth bound that INCLUDES the KV term, and the resized calculation
# assumes KV traffic scales linearly with context. A depth sweep tests that
# assumption directly: if the tokens/s curve falls off faster than 1/depth, the
# cost is not just KV bytes.
#
# FLAG NOTE. The request was `--depth 0 4096 8192 16384 32768`. llama-bench spells
# it `-d` / `--n-depth` and takes a COMMA-separated list (llama-bench.cpp:593,
# parsed with split and joined with ","), so the space-separated form would parse
# as one depth plus four stray positional arguments. Corrected here.
#
# WHY Qwen3-30B Q4_K_M. Same model family as round 27's vllm bench serve arms, so
# the two harnesses can be read against each other. 18 GB fits VRAM with room for
# a 32k KV cache, which the deepest point needs.
#
# -ub 2048 is not a default: docs/28 measured it as the optimum on this hardware,
# with 4096 and 8192 both measurably worse, and round 13 confirmed it still holds
# when experts are CPU-resident. Pinning it keeps this comparable to every other
# llama.cpp number in the repo.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODEL=$BASE/t35-gguf-q4km
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 28: llama-bench depth sweep ==="

IMG=llama-rocm714-bench:latest
H=/mnt/llm-storage

# The build is CPU-heavy, so it runs under the lock rather than alongside a
# latency benchmark. Round 27 measures TTFT at concurrency 1, where a parallel
# compile would show up directly in the numbers.
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "=== $(date -u +%T) building $IMG (one target, objects cached) ==="
    docker build -t "$IMG" -f "$BASE/configs/Dockerfile.llama-bench" "$BASE/configs" \
        || { echo "!! build failed -- no sweep run"; exit 1; }
fi

GGUF=$(find "$MODEL" -name '*.gguf' | sort | head -1)
[ -n "$GGUF" ] || { echo "!! no .gguf under $MODEL"; exit 1; }
echo "model: $GGUF"

# -p 512 / -n 128 are llama-bench defaults, kept so these rows are comparable to
# the figures other people publish. The sweep is over -d.
timeout 7200 docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
    --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
    -v "$H":/models \
    -e HSA_NO_SCRATCH_RECLAIM=1 -e GPU_MAX_HW_QUEUES=4 \
    "$IMG" \
        -m "${GGUF/#$H//models}" \
        -p 512 \
        -n 128 \
        -d 0,4096,8192,16384,32768 \
        -ngl 999 \
        -ub 2048 \
        -fa 1 \
        -r 3 \
    2>&1 | tee "logs/rd28-llama-bench.out" | tail -40

echo
echo "=== $(date -u +%T) round 28 done ==="
echo "full output: logs/rd28-llama-bench.out"
echo
echo "READING THIS: pp512 rows are PREFILL at that depth, tg128 rows are DECODE."
echo "The interesting quantity is how tg128 falls from d=0 to d=32768 -- that is"
echo "the KV-growth cost, and docs/25 item 1c assumes it scales linearly."
