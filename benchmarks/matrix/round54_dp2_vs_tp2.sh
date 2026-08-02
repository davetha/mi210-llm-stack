#!/usr/bin/env bash
# Round 54: DP=2 vs TP=2. The cheapest untested item in the backlog.
#
# THE MEASURED PROBLEM. TP=2 returns 1.28x, not 2x, from doubling both
# bandwidth and compute:
#     t35-w8a8 TP=1  33.80 tok/s decode @101k   (results/t35-w8a8-longctx.json)
#     t35-w8a8 TP=2  43.40 tok/s                (docs/25 item 1c)
# There is no xGMI on this box, so a decode step is ~2 host-staged allreduces
# per layer. Sub-linear scaling of that shape is per-layer collective LATENCY,
# not bandwidth -- which is why docs/39 item 9 proposed sidestepping the
# collective entirely rather than trying to make it faster.
#
# WHY DP HAS NEVER BEEN TRIED HERE. docs/04 measured tensor-parallel (2342 t/s)
# against EXPERT-parallel (811 t/s) and correctly concluded TP wins for this
# fabric. That result got generalised into "TP is the answer", but data
# parallelism was never in that comparison -- docs/39 notes DP "appears nowhere
# in this repo". Two independent single-card engines pay ZERO inter-GPU traffic.
#
# WHAT IS BEING COMPARED, AND WHY IT MUST BE THROUGHPUT.
#   arm A  one server, TP=2, both cards, model sharded
#   arm B  two servers, TP=1, one card each, full model on each
# DP cannot improve single-request latency -- one request still runs on one
# card. It can only improve AGGREGATE throughput. So single-stream decode
# (what run_arm.sh measures) would show DP LOSING, and that would be a
# meaningless result. This round drives concurrent load through
# `vllm bench serve` and compares total output tok/s at matched total
# concurrency. Latency percentiles are reported alongside precisely so the
# tradeoff is visible rather than hidden.
#
# THE HONEST COSTS OF DP, stated up front so a win is not oversold:
#   - halves KV cache per replica (each card holds the FULL weights, ~30 GB of
#     a 64 GB card, leaving far less for KV than a TP=2 rank has)
#   - does nothing for single-request latency or TTFT
#   - only applies to models that fit one card; TP stays forced above that
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
RES=$BASE/results
cd "$BASE"

IMG=vllm-mi210:gdnpolicy
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

# Load shape. Held identical across both arms.
IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
TOTAL_CONC=${TOTAL_CONC:-16}        # total in flight across the whole system
TOTAL_PROMPTS=${TOTAL_PROMPTS:-128}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 54: DP=2 vs TP=2 ==="
echo "load: in=$IN_LEN out=$OUT_LEN total_concurrency=$TOTAL_CONC prompts=$TOTAL_PROMPTS"

cleanup() { docker rm -f rd54-tp2 rd54-dp0 rd54-dp1 >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {  # port  name
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "  $name ready after ${t}s"; return 0
        fi
        if ! docker ps --format '{{.Names}}' | grep -q "^$name$"; then
            echo "  FATAL: $name exited during startup"; docker logs "$name" 2>&1 | tail -20; return 1
        fi
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name not ready within ${READY_TIMEOUT}s"; return 1
}

# `vllm bench serve` needs a real tokenizer path for --model. --max-concurrency
# bounds in-flight requests; --num-prompts is the total issued.
run_bench() {  # port  conc  prompts  outfile
    local port="$1" conc="$2" prompts="$3" out="$4"
    docker run --rm --network host \
      -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$port" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random \
        --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts "$prompts" --max-concurrency "$conc" \
        --ignore-eos --seed 1234 \
        2>&1 | tee "$out"
}

# ---------------- arm A: TP=2 -------------------------------------------
echo ""
echo "=== $(date -u +%T) arm A: TP=2, one server, both cards ==="
TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    "$BIN/serve_vllm_aiter.sh" "$MODEL" rd54-tp2 8101 --max-model-len 32768 >/dev/null
wait_ready 8101 rd54-tp2 || { echo "arm A FAILED"; exit 1; }
run_bench 8101 "$TOTAL_CONC" "$TOTAL_PROMPTS" "$LOGS/rd54-tp2.bench"
docker rm -f rd54-tp2 >/dev/null 2>&1

# ---------------- arm B: DP=2 -------------------------------------------
# HIP and rocm-smi enumerate these cards in OPPOSITE orders (serve_vllm.sh:25).
# That does not matter for TP=2 which takes both, but it matters here: these
# two replicas must land on DIFFERENT physical cards. Verified after startup
# by checking each container sees exactly 1 GPU.
echo ""
echo "=== $(date -u +%T) arm B: DP=2, two servers, one card each ==="
for i in 0 1; do
    TP=1 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 \
    VLLM_EXTRA_ENV="-e HIP_VISIBLE_DEVICES=$i" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "rd54-dp$i" "810$((2+i))" \
        --max-model-len 32768 >/dev/null
done
for i in 0 1; do wait_ready "810$((2+i))" "rd54-dp$i" || { echo "arm B FAILED"; exit 1; }; done

echo "--- confirming each replica has exactly one GPU ---"
for i in 0 1; do
    n=$(docker exec "rd54-dp$i" python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo "?")
    echo "  rd54-dp$i sees $n GPU(s)  (MUST be 1; 2 means HIP_VISIBLE_DEVICES did not take)"
done

# Half the concurrency and half the prompts to each replica, so the SYSTEM sees
# the same total offered load as arm A. Both run concurrently -- running them
# sequentially would measure two single-replica runs, not data parallelism.
HALF_C=$(( TOTAL_CONC / 2 )); HALF_P=$(( TOTAL_PROMPTS / 2 ))
echo "  each replica: concurrency=$HALF_C prompts=$HALF_P"
run_bench 8102 "$HALF_C" "$HALF_P" "$LOGS/rd54-dp0.bench" &
p0=$!
run_bench 8103 "$HALF_C" "$HALF_P" "$LOGS/rd54-dp1.bench" &
p1=$!
wait $p0; wait $p1
cleanup

echo ""
echo "=== $(date -u +%T) round 54 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
def parse(f):
    p = os.path.join(L, f)
    if not os.path.isfile(p): return None
    s = open(p, errors="replace").read()
    def g(pat):
        m = re.search(pat, s)
        return float(m.group(1)) if m else None
    return {
        "out_tps":  g(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
        "tot_tps":  g(r"Total Token throughput \(tok/s\):\s*([\d.]+)"),
        "req_tps":  g(r"Request throughput \(req/s\):\s*([\d.]+)"),
        "ttft_med": g(r"Median TTFT \(ms\):\s*([\d.]+)"),
        "tpot_med": g(r"Median TPOT \(ms\):\s*([\d.]+)"),
        "p99_ttft": g(r"P99 TTFT \(ms\):\s*([\d.]+)"),
    }
tp2 = parse("rd54-tp2.bench")
d0, d1 = parse("rd54-dp0.bench"), parse("rd54-dp1.bench")
if not tp2 or not d0 or not d1:
    print("MISSING RESULTS -- inspect logs/rd54-*.bench"); raise SystemExit
def add(a, b, k):
    return (a[k] + b[k]) if a.get(k) is not None and b.get(k) is not None else None
dp_out = add(d0, d1, "out_tps"); dp_tot = add(d0, d1, "tot_tps"); dp_req = add(d0, d1, "req_tps")
print(f"{'metric':<26}{'TP=2':>12}{'DP=2 (sum)':>13}{'DP/TP':>9}")
print("-" * 60)
for name, a, b in [("output tok/s", tp2["out_tps"], dp_out),
                   ("total tok/s",  tp2["tot_tps"], dp_tot),
                   ("request/s",    tp2["req_tps"], dp_req)]:
    if a and b: print(f"{name:<26}{a:12.2f}{b:13.2f}{b/a:8.3f}x")
print()
print("latency -- DP should be WORSE here if it is winning on throughput:")
for name, k in [("median TTFT ms", "ttft_med"), ("P99 TTFT ms", "p99_ttft"),
                ("median TPOT ms", "tpot_med")]:
    a = tp2.get(k); b0, b1 = d0.get(k), d1.get(k)
    if a and b0 and b1:
        print(f"  {name:<18} TP=2 {a:8.2f}   DP replicas {b0:8.2f} / {b1:8.2f}")
print()
print("READING THIS. The throughput rows are the result; DP=2 is summed across")
print("both replicas because the system served that many tokens. A DP/TP above")
print("1.0 means sidestepping the collective beat sharding. Latency rows are")
print("shown because DP cannot help single-request latency by construction --")
print("if TTFT/TPOT degrade, that is the real cost of the throughput gain and")
print("must be weighed before deploying. Also note DP halves KV per replica,")
print("so this comparison holds only at this context length.")
PY
