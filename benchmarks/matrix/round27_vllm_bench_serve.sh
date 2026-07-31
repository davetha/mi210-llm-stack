#!/usr/bin/env bash
# Qwen3-30B through vLLM's own `vllm bench serve`, for comparable numbers.
#
# WHY A SECOND HARNESS. bench_matrix.py answers "how fast is this configuration"
# and reports end-to-end TTFT plus an implied prefill rate. It does NOT report
# TPOT, inter-token latency, or percentiles, and its numbers are not directly
# comparable to what anyone else publishes. `vllm bench serve` is the tool the
# rest of the ecosystem quotes, so running it here makes these results legible
# outside this repo -- and gives a cross-check on our own harness.
#
# THE CONFIGURATION MIRRORS THE REFERENCE OUTPUT that prompted this: 10 requests,
# concurrency 1, 1024 input and 1024 output tokens each (10,240 of each in
# total). Concurrency 1 is the important part -- it measures LATENCY, not
# throughput, which is the opposite of what most of docs/28 measures. Expect
# numbers that look "worse" than our 8,755 t/s prefill figure for exactly that
# reason: a single request cannot fill the machine.
#
# --ignore-eos is set so every request generates exactly 1024 tokens. Without it
# the model stops early, output-token counts vary per request, and TPOT is
# computed over a different denominator each run.
#
# THREE MODELS, ONE OF WHICH IS NEW. t35-bf16-sharded exists because of round 25:
# it is the same weights as t35-bf16 but loads in 114.65 s instead of 12,366 s,
# and it posted the fastest prefill and decode in the matrix (8,755.2 / 65.23).
# Before round 25 it would have cost 3.4 hours just to include it here.
#
# NO SPECULATIVE DECODING. The reference output has an acceptance-rate section
# because it was run with a draft model. Omitted deliberately: docs/25 measured
# speculation LOSING on this box even at 100% draft acceptance (6.85 -> 6.01
# t/s), because decode is ~3x off its bandwidth bound and issue-bound, so
# verifying N+1 tokens costs ~N+1x rather than ~1x. Adding it would produce a
# section of numbers that describe a configuration nobody should run here.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
OUT=$BASE/results/bench-serve
cd "$BASE"
mkdir -p "$OUT"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 27: vllm bench serve on Qwen3-30B ==="

IMG=rocm-vllm-aiter-gfx90a:pa256k
export VLLM_IMAGE="$IMG"
H=/mnt/llm-storage
PORT=8100

# serve <name> <model-dir> <extra serve args...>
bench_one() {
    local label="$1" model="$2"; shift 2
    local name="bench-$label"
    echo
    echo "############ $label ############"
    docker rm -f "$name" >/dev/null 2>&1 || true
    # Wait for the port to actually free -- docker rm returns before the daemon
    # releases the mapping, which silently cost an arm earlier in this project.
    for _ in $(seq 1 30); do
        docker ps -q --filter "publish=$PORT" 2>/dev/null | grep -q . || break
        sleep 2
    done

    VLLM_PREFER_AITER_FA=1 "$BIN/serve_vllm_aiter.sh" "$model" "$name" "$PORT" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        "$@" || { echo "!! $label: server would not start"; return 1; }

    echo "waiting for $label to become ready..."
    local n=0
    until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
        docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q . \
            || { echo "!! $label: container exited during load"; return 1; }
        n=$((n+1)); [ "$n" -gt 360 ] && { echo "!! $label: not ready"; return 1; }
        sleep 10
    done
    echo "ready after $((n*10))s"

    # Runs on the HOST against the served port, so it never competes with the
    # engine for GPU memory the way an in-container client would.
    #
    # --model is the TOKENIZER source and must be a real path: `vllm bench serve`
    # loads it through transformers, so passing the served alias makes it try to
    # fetch a HF repo literally named "bench". --served-model-name carries the
    # alias the API actually expects. /models is mounted for the same reason.
    #
    # Output is tee'd rather than filtered. The first version piped straight into
    # `sed -n '/Serving Benchmark Result/,/^====/p'`, which silently discarded the
    # OSError above and left three arms looking like they had simply produced
    # nothing.
    docker run --rm --network host \
        -v "$OUT":/out -v "$H":/models \
        --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat \
            --endpoint /v1/chat/completions \
            --base-url "http://127.0.0.1:$PORT" \
            --model "${model/#$H//models}" \
            --served-model-name bench \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 1024 \
            --num-prompts 10 \
            --max-concurrency 1 \
            --ignore-eos \
            --percentile-metrics ttft,tpot,itl,e2el \
            --save-result --result-filename "/out/$label.json" \
        2>&1 | tee "logs/benchserve-$label.out" | sed -n '/Serving Benchmark Result/,/^====/p'
    grep -qF "Serving Benchmark Result" "logs/benchserve-$label.out" \
        || { echo "!! $label: bench produced no result -- tail follows"; \
             tail -12 "logs/benchserve-$label.out"; }

    docker logs "$name" > "logs/benchserve-$label.serverlog" 2>&1 || true
    docker rm -f "$name" >/dev/null 2>&1 || true
}

# W8A8 -- the docs/28 "best all-round" pick, and the baseline most comparable to
# what other people run.
bench_one t35-w8a8 "$BASE/t35-w8a8"

# bf16 via the round-25 snapshot: fastest prefill and decode measured here.
if [ -d "$BASE/t35-bf16-sharded" ]; then
    bench_one t35-bf16-sharded "$BASE/t35-bf16-sharded" --load-format sharded_state
else
    echo "!! t35-bf16-sharded absent -- skipping (round 25 produces it)"
fi

# AWQ-Int4 -- the memory pick, and the format round 26 is validating.
bench_one t35-awq "$BASE/t35-awq"

echo
echo "=== $(date -u +%T) round 27 done ==="
echo "raw JSON in $OUT/"
python3 - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("/mnt/llm-storage/bench-matrix/results/bench-serve/*.json")):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[:-5],
                 d.get("output_throughput"), d.get("total_token_throughput"),
                 d.get("median_ttft_ms"), d.get("p99_ttft_ms"),
                 d.get("median_tpot_ms"), d.get("median_itl_ms")))
if rows:
    print("%-20s %9s %9s %9s %9s %9s %9s" % (
        "model", "out tok/s", "tot tok/s", "TTFT p50", "TTFT p99", "TPOT p50", "ITL p50"))
    for r in rows:
        print("%-20s %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f" % tuple(
            (x if x is not None else 0) for x in r))
PY
echo
echo "NOTE: concurrency 1 measures LATENCY. These output-token rates are not"
echo "comparable to docs/28's prefill figures, which saturate the batch."
