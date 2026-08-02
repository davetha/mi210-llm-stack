#!/usr/bin/env bash
# Round 61: DP=2 and ASM paged attention together. The interaction is favourable.
#
# TWO MEASURED RESULTS THAT HAVE NEVER BEEN COMBINED.
#   docs/50 round 54: DP=2 beat TP=2 by 1.068x on aggregate throughput, at the
#     cost of 50% worse median TTFT. Two single-card engines pay ZERO of the
#     ~96 host-staged collectives per decode step that TP=2 pays at ~86 us each.
#   docs/51 round 57: AITER's ASM paged attention beats the HIP fallback above
#     ~128 heads (1.013x at conc 16, 1.008x at conc 32) and loses below it. The
#     gate is num_seqs * num_heads > 2 * cu_num = 208.
#
# WHY THEY COMPOUND RATHER THAN JUST ADD. At TP=2 each rank owns 32/2 = 16
# attention heads, so ASM needs num_seqs >= 14. At TP=1 -- which is what each DP
# replica runs -- each replica owns all **32** heads, so the same threshold is
# reached at num_seqs >= 7. Going data-parallel therefore makes the ASM kernel
# eligible at HALF the batch size, on top of removing the collectives.
#
# Round 54 ran without VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT, so its DP replicas
# could not reach paged_attention_common at all (rocm_aiter_fa.py:1283 gates on
# it). Its 1.068x is therefore a DP-only number with the ASM path switched off.
#
# ARMS. Three, all measuring AGGREGATE throughput at matched total offered load:
#   tp2       one TP=2 server, shuffle on   -- the deployed shape
#   dp2       two TP=1 servers, shuffle OFF -- reproduces round 54
#   dp2asm    two TP=1 servers, shuffle ON  -- the untested combination
# dp2 is included rather than assumed so the ASM contribution is separable from
# the DP contribution; without it, dp2asm vs tp2 would confound the two.
#
# THE HONEST COSTS, restated from docs/50 so a win is not oversold: DP cannot
# improve single-request latency, and it halves KV cache per replica. This is a
# throughput answer only.
#
# CONCURRENCY 16 AND 32. Not 1 -- docs/51 measured a ~7.6% second-arm penalty
# there in this harness. And DP replicas each see half the total, so total 16
# means 8 per replica = 256 heads at TP=1, comfortably over the 208 threshold;
# total 32 means 16 each = 512. Both DP points clear the ASM gate, which is the
# entire premise of the round.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"16 32"}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 61: DP=2 + ASM paged attention ==="

cleanup() { docker rm -f rd61-tp2 rd61-dp0 rd61-dp1 >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED:"; docker logs "$name" 2>&1 | tail -25; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

bench_one() {  # port conc prompts outfile
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$1" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts "$3" --max-concurrency "$2" \
        --ignore-eos --seed 1234 > "$4" 2>&1
}

co_of() { grep -ohE "pa_[a-z0-9_]*\.co" "$1" 2>/dev/null | sort -u | tr '\n' ' '; }

# ---- arm 1: TP=2 -------------------------------------------------------
echo ""
echo "=== $(date -u +%T) arm rd61-tp2 (TP=2, shuffle ON) ==="
TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1" \
    "$BIN/serve_vllm_aiter.sh" "$MODEL" rd61-tp2 8130 --max-model-len 32768 >/dev/null
wait_ready 8130 rd61-tp2 || exit 1
for c in $CONCS; do
    echo "--- $(date -u +%T) tp2 @ total conc $c ---"
    bench_one 8130 "$c" $(( c * 8 )) "$LOGS/rd61-tp2-c$c.bench"
    grep -E "Output token throughput|Median TTFT|Median TPOT" "$LOGS/rd61-tp2-c$c.bench" | sed 's/^/    /'
done
docker logs rd61-tp2 > "$LOGS/rd61-tp2.container" 2>&1 || true
echo "  pa .co: $(co_of "$LOGS/rd61-tp2.container")"
docker rm -f rd61-tp2 >/dev/null 2>&1

# ---- arms 2 and 3: DP=2, shuffle off then on ---------------------------
run_dp() {  # tag shuffle(0|1)
    local tag="$1" shuf="$2"
    echo ""
    echo "=== $(date -u +%T) arm rd61-$tag (DP=2, shuffle=$shuf) ==="
    for i in 0 1; do
        TP=1 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 \
        VLLM_EXTRA_ENV="-e HIP_VISIBLE_DEVICES=$i -e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=$shuf" \
            "$BIN/serve_vllm_aiter.sh" "$MODEL" "rd61-dp$i" "813$((1+i))" \
            --max-model-len 32768 >/dev/null
    done
    for i in 0 1; do wait_ready "813$((1+i))" "rd61-dp$i" || return 1; done
    for i in 0 1; do
        n=$(docker exec "rd61-dp$i" python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo "?")
        echo "  rd61-dp$i sees $n GPU(s)  (MUST be 1)"
    done
    for c in $CONCS; do
        local half=$(( c / 2 )) hp=$(( c * 8 / 2 ))
        echo "--- $(date -u +%T) $tag @ total conc $c ($half per replica) ---"
        bench_one 8131 "$half" "$hp" "$LOGS/rd61-$tag-r0-c$c.bench" &
        local p0=$!
        bench_one 8132 "$half" "$hp" "$LOGS/rd61-$tag-r1-c$c.bench" &
        local p1=$!
        wait $p0; wait $p1
        for r in 0 1; do
            grep -E "Output token throughput" "$LOGS/rd61-$tag-r$r-c$c.bench" | sed "s/^/    replica$r /"
        done
    done
    for i in 0 1; do
        docker logs "rd61-dp$i" > "$LOGS/rd61-$tag-dp$i.container" 2>&1 || true
        echo "  dp$i pa .co: $(co_of "$LOGS/rd61-$tag-dp$i.container")"
    done
    docker rm -f rd61-dp0 rd61-dp1 >/dev/null 2>&1
}

run_dp dp2    0 || echo "(dp2 arm failed)"
run_dp dp2asm 1 || echo "(dp2asm arm failed)"

echo ""
echo "=== $(date -u +%T) round 61 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "16 32").split()]
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
def rd(p):
    if not os.path.isfile(p): return None
    m = re.search(TPUT, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
print(f"{'total conc':>11}{'TP=2':>10}{'DP=2':>10}{'DP+ASM':>10}"
      f"{'DP/TP':>9}{'DP+ASM/TP':>11}{'ASM gain':>10}")
print("-" * 71)
for c in CONCS:
    tp = rd(f"{L}/rd61-tp2-c{c}.bench")
    dp = sum(filter(None, [rd(f"{L}/rd61-dp2-r{r}-c{c}.bench") for r in (0,1)])) or None
    da = sum(filter(None, [rd(f"{L}/rd61-dp2asm-r{r}-c{c}.bench") for r in (0,1)])) or None
    f = lambda v: f"{v:10.2f}" if v else f"{'-':>10}"
    r1 = f"{dp/tp:8.3f}x" if tp and dp else f"{'-':>9}"
    r2 = f"{da/tp:10.3f}x" if tp and da else f"{'-':>11}"
    r3 = f"{da/dp:9.3f}x" if dp and da else f"{'-':>10}"
    print(f"{c:>11}{f(tp)}{f(dp)}{f(da)}{r1}{r2}{r3}")
print()
for tag in ("dp2", "dp2asm"):
    for i in (0, 1):
        p = f"{L}/rd61-{tag}-dp{i}.container"
        s = ""
        if os.path.isfile(p):
            import subprocess
            s = " ".join(sorted(set(re.findall(r"pa_[a-z0-9_]*\.co", open(p, errors='replace').read()))))
        print(f"  {tag} dp{i}: pa .co = {s or '<none>'}")
print()
print("THE 'ASM gain' COLUMN IS THE POINT -- it isolates ASM paged attention")
print("from data parallelism, which round 54 could not do because it never set")
print("shuffle_kv and therefore never reached the ASM path at all.")
print()
print("VALIDITY: dp2 replicas must show NO pa .co; dp2asm replicas must show")
print("pa_bf16_noquant_gqa8. At TP=1 each replica owns all 32 heads, so the")
print("208-head gate is met from num_seqs >= 7 -- both concurrency points")
print("clear it. If dp2asm shows no .co, the ASM path did not engage and the")
print("ASM-gain column is measuring nothing.")
print()
print("DP is a THROUGHPUT answer only: it cannot help single-request latency")
print("and it halves KV per replica. docs/50 measured 50% worse median TTFT.")
PY
