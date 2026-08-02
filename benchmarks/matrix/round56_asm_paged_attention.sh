#!/usr/bin/env bash
# Round 56: AITER ASM paged attention. No patch needed -- docs/37 was wrong.
#
# WHAT docs/37 sec5 SAYS, AND WHY IT IS WRONG. It records that pa_fwd_asm "has
# no call site in vLLM" and that wiring one is real work. Both halves are false
# in this build:
#
#   vllm/_aiter_ops.py:2912          def pa_fwd_asm(...)          -- wrapper
#   vllm/_aiter_ops.py:2947          def paged_attention_common(...)
#   rocm_aiter_fa.py:1318            rocm_aiter_ops.paged_attention_common(...)
#                                                                 -- LIVE CALL
#
# The wrapper even carries an explicit note that it is deliberately NOT
# decorated with @if_aiter_supported, "to allow explicit backend selection via
# attention_config to work even when VLLM_ROCM_USE_AITER=0". So unlike almost
# everything else this project has had to carve out, ASM paged attention has NO
# ARCHITECTURE GATE AT ALL.
#
# AND THE KERNELS ARE NOT ORPHANS. Unlike the 8 fmoe objects (docs/49), the
# gfx90a pa config table is populated -- hsa/gfx90a/pa/pa_asm.csv has 8 rows,
# including exactly what this model needs:
#     bf16,bf16,8,...,pa_bf16_noquant_gqa8_1tg_4w.co
# t35-w8a8 is head_dim 128, 32 heads / 4 kv = GQA ratio 8, bf16. Eligible.
#
# SO WHY HAS IT NEVER RUN? Two gates, neither of them architectural:
#
#   1. rocm_aiter_fa.py:1283 reaches the call only under
#      `elif rocm_aiter_ops.is_shuffle_kv_cache_enabled():`
#      which is envs.VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT -- a plain bool env var
#      defaulting to False, with no arch check anywhere near it.
#
#   2. aiter/ops/attention.py _should_use_asm_kernel() requires
#          head_size == 128                            (we pass)
#          num_seqs * num_heads > 2 * cu_num           (we usually do NOT)
#      cu_num on MI210 is 104, so the threshold is 208. At TP=2 this model has
#      32/2 = 16 heads per rank, so ASM engages only at **num_seqs >= 14**.
#
# THAT SECOND GATE IS WHY EVERY PREVIOUS ROUND MISSED THIS. Every decode
# measurement in this repo is batch-1 single-stream: total_heads = 16, versus a
# threshold of 208. Thirteen times too small. paged_attention_ll4mi -- the 13.1%
# of decode in docs/45 -- is the HIP fallback being selected by a heuristic,
# not a missing kernel.
#
# THE EXPERIMENT. Run at concurrency 32 so num_seqs*num_heads = 512, comfortably
# clear of 208 even if the batch dips, and A/B only the env var. Concurrency 16
# would sit at 256 -- above the line but close enough that a partially-filled
# batch could silently drop back to HIP mid-run and average the two.
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
CONC=${CONC:-32}
PROMPTS=${PROMPTS:-256}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 56: ASM paged attention (shuffle-kv gate) ==="
echo "concurrency $CONC -> total_heads ~= $((CONC*16)) vs ASM threshold 208"

cleanup() { docker rm -f rd56-hip rd56-asm >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || { echo "  FATAL: $name exited"; docker logs "$name" 2>&1 | tail -20; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

run_arm() {  # name port shuffle(0|1)
    local name="$1" port="$2" shuf="$3"
    echo ""
    echo "=== $(date -u +%T) arm $name  SHUFFLE_KV_CACHE_LAYOUT=$shuf ==="
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=$shuf" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" --max-model-len 32768 >/dev/null
    wait_ready "$port" "$name" || return 1
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$port" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts "$PROMPTS" --max-concurrency "$CONC" \
        --ignore-eos --seed 1234 2>&1 | tee "$LOGS/$name.bench"
    # Proof of engagement. Without this the throughput number is unattributable:
    # the ASM arm could have silently fallen back to HIP via either gate.
    local co
    co=$(docker logs "$name" 2>&1 | grep -ohE "pa_[a-z0-9_]*\.co" | sort -u | tr '\n' ' ')
    echo "  pa .co objects loaded: ${co:-<NONE>}"
    docker logs "$name" 2>&1 | grep -c "LoadKernel" | sed 's/^/  LoadKernel count: /'
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm rd56-hip 8106 0
run_arm rd56-asm 8107 1

echo ""
echo "=== $(date -u +%T) round 56 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
def parse(f):
    p = os.path.join(L, f)
    if not os.path.isfile(p): return None
    s = open(p, errors="replace").read()
    g = lambda pat: (float(m.group(1)) if (m := re.search(pat, s)) else None)
    return {"out": g(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
            "tot": g(r"Total Token throughput \(tok/s\):\s*([\d.]+)"),
            "ttft": g(r"Median TTFT \(ms\):\s*([\d.]+)"),
            "tpot": g(r"Median TPOT \(ms\):\s*([\d.]+)")}
a, b = parse("rd56-hip.bench"), parse("rd56-asm.bench")
if not a or not b:
    print("MISSING RESULTS -- inspect logs/rd56-*.bench"); raise SystemExit
print(f"{'metric':<22}{'HIP':>12}{'ASM':>12}{'ASM/HIP':>10}")
print("-"*56)
for name, k in [("output tok/s","out"),("total tok/s","tot"),
                ("median TTFT ms","ttft"),("median TPOT ms","tpot")]:
    x, y = a[k], b[k]
    if x and y: print(f"{name:<22}{x:12.2f}{y:12.2f}{y/x:9.3f}x")
print()
print("THE .co LINE ABOVE DECIDES WHETHER THIS ROUND MEANS ANYTHING. If the ASM")
print("arm loaded no pa_*.co, one of the two gates still declined and the")
print("throughput delta is measuring something else. Expect")
print("pa_bf16_noquant_gqa8_1tg_4w.co on the ASM arm and nothing on the HIP arm.")
print("On throughput rows higher is better; on TTFT/TPOT lower is better.")
PY
