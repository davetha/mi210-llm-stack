#!/usr/bin/env bash
# Round 74: the CK int8 GEMM on the 80B, on an image where it actually engages.
#
# WHY ROUND 73 MEASURED NOTHING. It reported CK GEMM at 1.014x on the 80B
# against docs/43's 1.480x on the 30B, and the honest reading of that gap was
# "the flag did not engage". It did not. The server log said:
#
#     Selected TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8
#
# with VLLM_ROCM_USE_AITER_LINEAR=1 set and present in the container env. Both
# arms were Triton; the 1.4% was noise between two identical configurations.
#
# THE CAUSE WAS IN THE ROUND 73 SCRIPT, NOT THE STACK. round73 set IMG= for the
# BENCH CLIENT container and never set VLLM_IMAGE, which is what
# serve_vllm_aiter.sh reads for the SERVER:
#
#     IMAGE="${VLLM_IMAGE:-rocm-vllm-aiter-gfx90a:latest}"
#
# So the servers ran the default image (f511959632eb) while round 73's own
# header recorded that the CK carve-outs had been verified in
# vllm-mi210:gdnpolicy (37191e92384b) -- a different image that was never
# serving. The pre-flight check was real and inspected the wrong container.
#
# This is the fourth instance in this project of an arm-launch path silently
# running something other than what the round claimed (VLLM_IMAGE in round 31,
# NCCL_P2P_DISABLE in 32, VLLM_PREFER_AITER_FA in 37) -- and serve_vllm_aiter.sh
# carries a comment warning about exactly this. The lesson that keeps not taking:
# verifying a FLAG is set proves nothing about which KERNEL ran.
#
# WHAT IS DIFFERENT HERE. Directly confirmed before writing this round, by
# reading the selection line out of a running server rather than inferring it:
#
#   rocm-vllm-aiter-gfx90a:latest      -> TritonInt8ScaledMMLinearKernel
#   vllm-mi210:v0.26.1rc0-ckgemm-warm  -> AiterInt8ScaledMMLinearKernel   (30B)
#   vllm-mi210:v0.26.1rc0-ckgemm-warm  -> AiterInt8ScaledMMLinearKernel   (80B)
#
# The 80B loads and selects the CK kernel on the docs/43 image. So the win is
# reachable for this model and round 73 simply never tested it.
#
# THE ASSERTION THIS ROUND ADDS. Each arm greps the server log for the kernel it
# actually selected and ABORTS if it does not match what the arm intends. A
# round that cannot prove which kernel ran does not get to report a ratio.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
SERVER_IMG=${SERVER_IMG:-vllm-mi210:v0.26.1rc0-ckgemm-warm}
CLIENT_IMG=${CLIENT_IMG:-$SERVER_IMG}
MODEL=$BASE/t80-w8a8

docker image inspect "$SERVER_IMG" >/dev/null 2>&1 || { echo "FATAL: missing $SERVER_IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

MAXLEN=${MAXLEN:-163840}
DEEP=${DEEP:-130000}
DEEP_OUT=${DEEP_OUT:-128}
AGG_CTX=${AGG_CTX:-16384}
AGG_CONC=${AGG_CONC:-8}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

[ $(( DEEP + DEEP_OUT + 1024 )) -le "$MAXLEN" ] || { echo "FATAL: deep ctx exceeds max-model-len"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 74: CK int8 GEMM on the 80B, correct image ==="
echo "    server image: $SERVER_IMG"
echo "    target: llama.cpp Coder-Next 80B Q4_K_M f16 KV = 62.89 tok/s at depth 130,000"
echo "    round 73 on the wrong image: Triton 44.62, 'CK' 45.27 (both Triton)"

cleanup() { docker rm -f rd74-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag  aiter_linear  expected_kernel
    local tag="$1" lin="$2" want="$3" t=0
    echo ""
    echo "=== $(date -u +%T) arm $tag  AITER_LINEAR=$lin  expecting $want ==="
    docker rm -f rd74-srv >/dev/null 2>&1
    VLLM_IMAGE="$SERVER_IMG" \
      TP=2 \
      VLLM_ROCM_USE_AITER_LINEAR="$lin" \
      VLLM_PREFER_AITER_FA=1 \
      NCCL_P2P_DISABLE=0 \
      "$BIN/serve_vllm_aiter.sh" "$MODEL" rd74-srv 8188 --max-model-len "$MAXLEN" >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8188/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd74-srv$' || {
            echo "  EXITED -- log at $LOGS/rd74-$tag.startup-fail"
            docker logs rd74-srv > "$LOGS/rd74-$tag.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|not support|Unsupported|out of memory" "$LOGS/rd74-$tag.startup-fail" \
              | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib|triton_kernels|AVX2" | head -6
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"

    # THE ASSERTION. Read the kernel out of the server, do not infer it from the
    # flag. Round 73 inferred and was wrong for its entire runtime.
    local got
    got=$(docker logs rd74-srv 2>&1 | grep -aoE "Selected [A-Za-z0-9]+ScaledMMLinearKernel" | tail -1 | awk '{print $2}')
    echo "  selected kernel: ${got:-<none logged>}"
    if [ "$got" != "$want" ]; then
        echo "  ABORTING ARM: expected $want, got ${got:-nothing}."
        echo "  A ratio measured against the wrong kernel is worse than no ratio."
        docker rm -f rd74-srv >/dev/null 2>&1
        return 1
    fi
    docker logs rd74-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'

    bench() {  # label inlen outlen conc prompts outfile
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$CLIENT_IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8188 \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$2" --random-output-len "$3" \
            --num-prompts "$5" --max-concurrency "$4" \
            --ignore-eos --seed 1234 > "$6" 2>&1
        local failed
        failed=$(grep -aoE "Failed requests: +[0-9]+" "$6" | grep -aoE "[0-9]+$" | head -1)
        printf "  %-22s " "$1"
        if [ -n "${failed:-}" ] && [ "$failed" -gt 0 ]; then echo "FAILED ($failed rejected)"; return; fi
        printf "TPOT %7s ms   agg %8s tok/s\n" \
          "$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$6" | grep -aoE "[0-9.]+$" | head -1)" \
          "$(grep -aoE "Output token throughput \(tok\/s\): +[0-9.]+" "$6" | grep -aoE "[0-9.]+$" | head -1)"
    }

    bench "deep d=$DEEP conc1"          "$DEEP"    "$DEEP_OUT" 1           3                    "$LOGS/rd74-$tag-deep.bench"
    bench "agg  d=$AGG_CTX c$AGG_CONC"  "$AGG_CTX" 128         "$AGG_CONC" $(( AGG_CONC * 4 )) "$LOGS/rd74-$tag-agg.bench"
    docker rm -f rd74-srv >/dev/null 2>&1
}

run_arm triton 0 TritonInt8ScaledMMLinearKernel || echo "(triton arm failed/aborted)"
run_arm ck     1 AiterInt8ScaledMMLinearKernel  || echo "(ck arm failed/aborted)"

echo ""
echo "=== $(date -u +%T) round 74 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
LLAMA = 62.89   # rounds 70/71: Coder-Next 80B Q4_K_M, f16 KV, depth 130,000
def v(tag, kind, pat):
    p = os.path.join(L, f"rd74-{tag}-{kind}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    x = float(m.group(1));  return x if x > 0 else None
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"

print(f"{'arm':<26}{'deep tok/s':>12}{'vs llama':>10}{'TPOT ms':>10}{'agg tok/s':>12}")
print("-" * 70)
got = {}
for tag, label in [("triton", "Triton int8 GEMM"), ("ck", "AITER CK int8 GEMM")]:
    t = v(tag, "deep", TPOT)
    ps = 1000.0 / t if t else None
    agg = v(tag, "agg", TPUT)
    got[tag] = ps
    dash = "-"
    print(f"{label:<26}"
          f"{(f'{ps:12.2f}' if ps else f'{dash:>12}')}"
          f"{(f'{ps/LLAMA:9.3f}x' if ps else f'{dash:>10}')}"
          f"{(f'{t:10.2f}' if t else f'{dash:>10}')}"
          f"{(f'{agg:12.2f}' if agg else f'{dash:>12}')}")
print(f"{'llama.cpp Q4_K_M f16 KV':<26}{LLAMA:12.2f}{1.0:9.3f}x")

tri, ck = got.get("triton"), got.get("ck")
print()
if tri and ck:
    print(f"CK GEMM vs Triton on the 80B: {ck/tri:.3f}x   (docs/43: 1.480x on the 30B)")
    print(f"CK GEMM vs llama.cpp:         {ck/LLAMA:.3f}x")
    print()
    if ck >= LLAMA:
        print("vLLM MATCHES OR BEATS the production stack on single-stream decode at")
        print("130K -- while also prefilling ~2.85x faster (7,249 vs 2,546 tok/s).")
        print("The stack choice stops being a tradeoff on this model.")
    else:
        print(f"vLLM still trails llama.cpp by {LLAMA/ck:.2f}x on single-stream decode at")
        print("130K. Its case rests on prefill (~2.85x) and the aggregate column,")
        print("not on decode.")
else:
    print("An arm did not produce a number. If it ABORTED on the kernel assertion,")
    print("that is the round working as intended -- round 73 reported a ratio it")
    print("had not earned because it never checked.")
PY
