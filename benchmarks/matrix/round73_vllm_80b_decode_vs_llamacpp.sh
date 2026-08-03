#!/usr/bin/env bash
# Round 73: can vLLM's 80B decode reach llama.cpp's, with every known win on?
#
# THE TARGET, MEASURED THE SAME WAY. Round 70/71 put llama.cpp on the production
# Coder-Next 80B Q4_K_M with f16 KV at:
#
#     depth 130,000   ->   62.89 tok/s   single stream
#
# docs/24's vLLM 80B decode numbers are 45.19 tok/s (W8A8) and 51.34 (W8A16) at
# depth ~101k. Those were measured 2026-07-28. The CK int8 GEMM landed
# 2026-08-01 (docs/43) and is worth 1.48x decode on the 30B. It has NEVER been
# run against the 80B. 45.19 x 1.48 = 66.9, which would clear llama.cpp -- but
# that is arithmetic across two models and two context lengths, not a
# measurement. This round measures it.
#
# WHY W8A8 AND NOT AWQ. The CK GEMM is an a8w8 kernel: it needs 8-bit weights
# AND 8-bit activations. t80-w8a8 config.json confirms both are num_bits 8, so
# it is genuine W8A8, not the W8A16 that docs/26 warns names can hide. AWQ and
# W8A16 checkpoints dequantize to 16-bit activations and cannot reach this
# kernel at all, so they are not candidates for the win under test.
#
# THE DP=2 ARM DOES NOT EXIST, AND THAT IS A RESULT. docs/52 measured DP=2 plus
# shuffled KV layout at 1.118x throughput -- the best throughput result in the
# project. It CANNOT apply here. DP replicates the whole model per rank, and
# t80-w8a8 is 77 GB against 64 GB per card. Measured, not assumed: `du -sh` on
# the checkpoint. So the 80B is TP-only and one of the two big wins is
# structurally unavailable to it. Recording this because "apply the DP win to
# the 80B" is the obvious next thought and it is dead on arrival.
#
# THE FAILURE MODE THIS ROUND IS BUILT AGAINST. serve_vllm_aiter.sh defaults
# VLLM_ROCM_USE_AITER_LINEAR to 0 and its own comment records that hardcoding
# such a flag has silently produced two identical arms THREE times in this
# project (rounds 31, 32, 37), each reported as a measurement. Both CK GEMM
# blockers were verified present in vllm-mi210:gdnpolicy before writing this:
#
#   GFX_CU_NUM_MAP has "gfx90a": 104          (blocker 1, the codegen)
#   _aiter_ops.py:1640 is_linear_enabled has  (blocker 2, the selection gate)
#     the gfx90a carve-out
#
# and the summary below REFUSES to report a speedup if the triton and ck arms
# come back within 1%, because that is what a dead flag looks like.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t80-w8a8

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

MAXLEN=${MAXLEN:-163840}
DEEP=${DEEP:-130000}          # matched to llama.cpp's 62.89 tok/s point
DEEP_OUT=${DEEP_OUT:-128}
AGG_CTX=${AGG_CTX:-16384}
AGG_CONC=${AGG_CONC:-8}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

[ $(( DEEP + DEEP_OUT + 1024 )) -le "$MAXLEN" ] || {
    echo "FATAL: deep ctx $DEEP + $DEEP_OUT output + template > max-model-len $MAXLEN"
    echo "       (this is the round 66 defect -- fix before running)"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 73: vLLM 80B W8A8 decode vs llama.cpp ==="
echo "    target: llama.cpp Coder-Next 80B Q4_K_M f16 KV = 62.89 tok/s at depth 130,000"
echo "    deep: ${DEEP}+${DEEP_OUT} conc 1   aggregate: ${AGG_CTX} conc ${AGG_CONC}"

cleanup() { docker rm -f rd73-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag  aiter_linear  extra_env
    local tag="$1" lin="$2" extra="${3:-}" t=0
    echo ""
    echo "=== $(date -u +%T) arm $tag  AITER_LINEAR=$lin  extra='${extra:-none}' ==="
    docker rm -f rd73-srv >/dev/null 2>&1
    TP=2 \
      VLLM_ROCM_USE_AITER_LINEAR="$lin" \
      VLLM_PREFER_AITER_FA=1 \
      NCCL_P2P_DISABLE=0 \
      VLLM_EXTRA_ENV="$extra" \
      "$BIN/serve_vllm_aiter.sh" "$MODEL" rd73-srv 8183 --max-model-len "$MAXLEN" >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8183/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd73-srv$' || {
            echo "  EXITED -- log kept at $LOGS/rd73-$tag.startup-fail"
            docker logs rd73-srv > "$LOGS/rd73-$tag.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|assert|not support|Unsupported|out of memory" \
              "$LOGS/rd73-$tag.startup-fail" \
              | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib" | head -6
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"
    docker logs rd73-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'
    # vLLM does not log which int8 kernel it picked (docs/43), so capture what
    # IS observable and let the triton-vs-ck delta be the real evidence.
    docker logs rd73-srv 2>&1 | grep -aoiE "Using [A-Za-z]+ Int8 MoE backend|Selected [A-Za-z]+ScaledMM[A-Za-z]*" \
      | sort -u | head -3 | sed 's/^/  kernel: /'

    bench() {  # label inlen outlen conc prompts outfile
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8183 \
            --model "$MODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$2" --random-output-len "$3" \
            --num-prompts "$5" --max-concurrency "$4" \
            --ignore-eos --seed 1234 > "$6" 2>&1
        local failed
        failed=$(grep -aoE "Failed requests: +[0-9]+" "$6" | grep -aoE "[0-9]+$" | head -1)
        printf "  %-22s " "$1"
        if [ -n "${failed:-}" ] && [ "$failed" -gt 0 ]; then
            echo "FAILED ($failed rejected) -- see $6"; return
        fi
        printf "TPOT %7s ms   out %9s tok/s\n" \
          "$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$6" | grep -aoE "[0-9.]+$" | head -1)" \
          "$(grep -aoE "Output token throughput \(tok\/s\): +[0-9.]+" "$6" | grep -aoE "[0-9.]+$" | head -1)"
    }

    bench "deep d=$DEEP conc1"  "$DEEP"    "$DEEP_OUT" 1           3                    "$LOGS/rd73-$tag-deep.bench"
    bench "agg  d=$AGG_CTX c$AGG_CONC" "$AGG_CTX" 128         "$AGG_CONC" $(( AGG_CONC * 4 )) "$LOGS/rd73-$tag-agg.bench"
    docker rm -f rd73-srv >/dev/null 2>&1
}

run_arm triton  0 ""                                          || echo "(triton arm failed)"
run_arm ck      1 ""                                          || echo "(ck arm failed)"
run_arm ckshuf  1 "-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1"    || echo "(ck+shuffle arm failed)"

echo ""
echo "=== $(date -u +%T) round 73 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
ARMS = [("triton", "Triton int8 GEMM"), ("ck", "AITER CK int8 GEMM"),
        ("ckshuf", "CK GEMM + shuffled KV")]
LLAMA_DEEP = 62.89   # round 70/71, Coder-Next 80B Q4_K_M, f16 KV, depth 130,000

def val(tag, kind, pat):
    p = os.path.join(L, f"rd73-{tag}-{kind}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    v = float(m.group(1))
    return v if v > 0 else None

TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"

print(f"{'arm':<24}{'deep tok/s':>12}{'vs llama':>10}{'deep TPOT':>12}{'agg tok/s':>12}")
print("-" * 70)
rows = {}
for tag, label in ARMS:
    t = val(tag, "deep", TPOT)
    per_stream = 1000.0 / t if t else None      # conc 1 -> tok/s of the single stream
    agg = val(tag, "agg", TPUT)
    rows[tag] = per_stream
    print(f"{label:<24}"
          f"{(f'{per_stream:12.2f}' if per_stream else f'{chr(45):>12}')}"
          f"{(f'{per_stream/LLAMA_DEEP:9.3f}x' if per_stream else f'{chr(45):>10}')}"
          f"{(f'{t:12.2f}' if t else f'{chr(45):>12}')}"
          f"{(f'{agg:12.2f}' if agg else f'{chr(45):>12}')}")

print()
print(f"llama.cpp reference at the SAME depth ({130000} tokens, f16 KV):")
print(f"{'Coder-Next 80B Q4_K_M':<24}{LLAMA_DEEP:12.2f}{1.0:9.3f}x")

# A dead flag looks exactly like a null result. Refuse to report one as the other.
tri, ck = rows.get("triton"), rows.get("ck")
print()
if tri and ck:
    d = abs(ck - tri) / tri
    if d < 0.01:
        print(f"!! triton and ck arms agree to {d*100:.2f}% -- treat this as a DEAD FLAG,")
        print("   not as 'the CK GEMM does not help'. serve_vllm_aiter.sh has produced")
        print("   two identical arms three times in this project. Verify the image has")
        print("   both carve-outs before believing any null here.")
    else:
        print(f"CK GEMM vs Triton on the 80B: {ck/tri:.3f}x decode "
              f"(docs/43 measured 1.480x on the 30B).")
print()
print("WHAT THIS DECIDES. If the CK arm clears 62.89 tok/s, vLLM matches the")
print("production stack on single-stream decode at 130K while ALSO prefilling")
print("~2.85x faster (7,249 vs 2,546 tok/s, docs/24 + round 72) -- and the")
print("stack choice stops being a tradeoff. If it does not clear it, llama.cpp")
print("keeps single-stream decode and long context, and vLLM's case rests on")
print("prefill and the aggregate column above.")
print()
print("NOT MEASURED HERE: DP=2. It is structurally impossible for this model --")
print("DP replicates the full 77 GB checkpoint per rank against 64 GB cards, so")
print("docs/52's 1.118x cannot be applied to the 80B at all.")
PY
