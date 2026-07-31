#!/usr/bin/env bash
# llama-benchy depth sweep to 150k, against both engines.
#
# THE TOOL. eugr/llama-benchy -- "llama-bench style benchmarking tool for all
# backends", but unlike llama-bench it is a CLIENT: it drives an
# OpenAI-compatible endpoint rather than loading the model itself. That is why it
# has --concurrency and --latency-mode, and it is also why it may sidestep the
# fault round 28 hit. llama-bench primes depth internally through its own KV
# path; llama-benchy just sends a long prompt over HTTP, which is the ordinary
# serving path. Round 29 is isolating that fault separately.
#
# --latency-mode generation measures a single-token generation and SUBTRACTS it
# from subsequent timings, so the reported numbers exclude client and API
# overhead. Worth knowing when comparing against docs/28, whose figures are
# end-to-end and therefore include it.
#
# WHY BOTH ENGINES, AND WHY 150k IS THE INTERESTING PART. The requested depths
# run to 150000, which is ABOVE vLLM's stock 131072 ceiling. docs/28 records that
# exceeding it on stock vLLM bakes the Triton fallback into every request and
# costs 10x decode; configs/extend_rocm_pa_256k_gfx9.py fixes exactly that, and
# rocm-vllm-aiter-gfx90a:pa256k carries the fix. So the deep end of this sweep is
# a direct test of that patch, not just a throughput measurement.
#
# llama.cpp has no such ceiling, which makes it the control: if vLLM tracks
# llama.cpp past 131072, the patch is working.
#
# CONTEXT SIZING. Deepest point is 150000 + 128 generated, so both servers get
# 160000 to leave headroom. Qwen3-30B-A3B is 32 heads / 4 KV heads at head_dim
# 128 over 48 layers, i.e. ~96 KiB/token of KV, so 150k costs ~14.4 GB on top of
# a 17.28 GiB model -- comfortable on two cards.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
OUT=$BASE/results/benchy
cd "$BASE"
mkdir -p "$OUT"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 30: llama-benchy depth sweep to 150k ==="

PORT=8100
H=/mnt/llm-storage
DEPTHS="4096 8132 16000 30000 60000 90000 120000 150000"
CLIENT=rocm-vllm-aiter-gfx90a:pa256k   # has python + uv; used only as a client

# Install once into a named volume so the two arms do not each re-download.
docker volume create benchy-venv >/dev/null 2>&1 || true
echo "=== installing llama-benchy ==="
docker run --rm -v benchy-venv:/venv --entrypoint bash "$CLIENT" -c '
    set -e
    [ -x /venv/bin/llama-benchy ] && { echo "already installed"; exit 0; }
    python3 -m venv /venv
    /venv/bin/pip install -q --upgrade pip
    /venv/bin/pip install -q llama-benchy
    /venv/bin/llama-benchy --help >/dev/null
    echo "installed"
' || { echo "!! could not install llama-benchy"; exit 1; }

# benchy <label> <ready-url>
benchy() {
    local label="$1"
    echo
    echo "############ llama-benchy: $label ############"
    # shellcheck disable=SC2086
    timeout 10800 docker run --rm --network host \
        -v benchy-venv:/venv -v "$OUT":/out \
        --entrypoint /venv/bin/llama-benchy "$CLIENT" \
            --base-url "http://127.0.0.1:$PORT/v1" \
            --model bench \
            --depth $DEPTHS \
            --pp 2048 \
            --tg 128 \
            --concurrency 1 \
            --latency-mode generation \
        2>&1 | tee "logs/rd30-benchy-$label.out" | tail -45
    grep -qiE "depth|tg128|tok/s|t/s" "logs/rd30-benchy-$label.out" \
        || { echo "!! $label: no results -- tail follows"; tail -15 "logs/rd30-benchy-$label.out"; }
}

wait_ready() {
    local name="$1" n=0
    until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
        docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q . \
            || { echo "!! $name exited during load"; return 1; }
        n=$((n+1)); [ "$n" -gt 300 ] && { echo "!! $name not ready"; return 1; }
        sleep 10
    done
    echo "ready after $((n*10))s"
}

free_port() {
    for _ in $(seq 1 30); do
        docker ps -q --filter "publish=$PORT" 2>/dev/null | grep -q . || break
        sleep 2
    done
}

# --- A. llama.cpp: the control, no 131072 ceiling --------------------------
docker rm -f bench-benchy-llamacpp >/dev/null 2>&1 || true; free_port
if NGL=999 "$BIN/serve_llamacpp.sh" "$BASE/t35-gguf-q4km" bench-benchy-llamacpp "$PORT" \
        --ctx-size 160000 --flash-attn on; then
    wait_ready bench-benchy-llamacpp && benchy llamacpp
    docker logs bench-benchy-llamacpp > logs/rd30-llamacpp.serverlog 2>&1 || true
fi
docker rm -f bench-benchy-llamacpp >/dev/null 2>&1 || true; free_port

# --- B. vLLM at 160k: exercises the 256k paged-attention patch -------------
if VLLM_IMAGE=rocm-vllm-aiter-gfx90a:pa256k VLLM_PREFER_AITER_FA=1 \
        "$BIN/serve_vllm_aiter.sh" "$BASE/t35-w8a8" bench-benchy-vllm "$PORT" \
        --tensor-parallel-size 2 --max-model-len 160000 \
        --max-num-batched-tokens 8192 --no-enable-prefix-caching; then
    wait_ready bench-benchy-vllm && benchy vllm
    docker logs bench-benchy-vllm > logs/rd30-vllm.serverlog 2>&1 || true
    # Proof the patched kernel was used rather than the Triton fallback.
    echo "--- paged-attention path actually taken ---"
    grep -oE "npar_loops.: [0-9]+|Overriding with [A-Z_]+|ROCM_AITER_FA" \
        logs/rd30-vllm.serverlog 2>/dev/null | sort -u | head -4
fi
docker rm -f bench-benchy-vllm >/dev/null 2>&1 || true

echo
echo "=== $(date -u +%T) round 30 done ==="
echo "raw output: logs/rd30-benchy-*.out"
echo
echo "READING THIS: the deep end is the point. Depths above 131072 are where"
echo "stock vLLM falls back to Triton and loses ~10x decode (docs/28); if the"
echo "vLLM curve tracks llama.cpp at 150000, the 256k patch is doing its job."
