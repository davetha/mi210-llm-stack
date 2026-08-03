#!/usr/bin/env bash
# Round 75: is there a Q4_K_M-sized checkpoint that keeps vLLM's prefill?
#
# THE QUESTION. Round 74 left a clean split: vLLM prefills the 80B ~2.85x faster
# than llama.cpp (7,249 vs 2,546 tok/s) but decodes 1.35x slower (47.44 vs
# 62.89). If a checkpoint existed with llama.cpp's weight footprint AND vLLM's
# prefill, the split would collapse and there would be one right answer.
#
# t80-awq IS THAT CANDIDATE, AND IT HAS NEVER BEEN RUN AT TP=2.
#
#     t80-w8a8   77 GB   W8A8   (8-bit weights, 8-bit activations)
#     t80-awq    46 GB   W4A16  (4-bit group-wise weights, NULL activations)
#
# 46 GB is essentially Q4_K_M's 45 GB. docs/24 measured this checkpoint only at
# TP=1: prefill 4,487 tok/s, decode 45.9 tok/s. That TP=1 decode already matches
# W8A8's TP=2 number (45.19), which is the reason to look harder.
#
# THE TWO COMPETING PREDICTIONS, WRITTEN DOWN FIRST.
#
#   W4A16 should DECODE better. Qwen3-Next-80B-A3B activates ~3B params per
#   token, so weight bytes per decode step are ~3.0 GB at 8-bit against ~1.5 GB
#   at 4-bit. If decode is limited by weight traffic, 4-bit wins outright.
#
#   W4A16 should PREFILL worse. "NULL activations" means the weights are
#   dequantized to bf16 and the GEMM runs at 16-bit -- dequant overhead paid
#   with no arithmetic benefit, at exactly the large-M shapes where W8A8's int8
#   MFMA should shine. docs/24's TP=1 rows disagree with the strength of this
#   prediction (4,487 vs 4,739, only 5% apart), which is itself worth resolving.
#
# WHY THE ANSWER IS NOT OBVIOUS FROM THE FIRST PREDICTION. docs/50's roofline
# says decode on this box is LATENCY-bound, not bandwidth-bound: 113 GB/s of
# 1,170 available (10%) and 15.5 VALU ops/cycle of ~104 (15%). If that holds at
# 80B scale, halving weight bytes buys little and the 4-bit decode advantage
# will not appear. That roofline was measured on the 30B and has never been
# repeated at 80B, so this round is also a test of whether the diagnosis
# transfers.
#
# HOLDING THE IMAGE FIXED, DELIBERATELY. Round 74 measured the SAME Triton
# kernel on the same model at 47.44 tok/s where round 73 measured 44.62 -- 6%
# apart, differing only in container image. That is larger than most of the
# flags chased in docs/50-53. Both arms here run
# vllm-mi210:v0.26.1rc0-ckgemm-warm, the image round 74 used and the only one
# confirmed to both load the 80B and select the AITER CK kernel.
#
# PREFILL IS MEASURED WITH output-len 1 so decode cannot contaminate TTFT, and
# reported as input_len/TTFT rather than as a throughput field that mixes both.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
SERVER_IMG=${SERVER_IMG:-vllm-mi210:v0.26.1rc0-ckgemm-warm}
CLIENT_IMG=${CLIENT_IMG:-$SERVER_IMG}

docker image inspect "$SERVER_IMG" >/dev/null 2>&1 || { echo "FATAL: missing $SERVER_IMG"; exit 1; }

MAXLEN=${MAXLEN:-163840}
PRE_CTX=${PRE_CTX:-16384}
DEEP=${DEEP:-130000}
DEEP_OUT=${DEEP_OUT:-128}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

[ $(( DEEP + DEEP_OUT + 1024 )) -le "$MAXLEN" ] || { echo "FATAL: deep ctx exceeds max-model-len"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 75: W4A16 (46 GB) vs W8A8 (77 GB) at TP=2 ==="
echo "    image: $SERVER_IMG"
echo "    references -- llama.cpp Q4_K_M f16 KV: prefill 2,546 t/s, decode 62.89 tok/s @130k"

cleanup() { docker rm -f rd75-srv >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag model_subdir
    local tag="$1" sub="$2" t=0
    local model="$BASE/$sub"
    [ -f "$model/config.json" ] || { echo "  no model at $model"; return 1; }
    echo ""
    echo "=== $(date -u +%T) arm $tag  ($sub, $(du -sh "$model" 2>/dev/null | cut -f1)) ==="
    docker rm -f rd75-srv >/dev/null 2>&1
    VLLM_IMAGE="$SERVER_IMG" TP=2 \
      VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
      "$BIN/serve_vllm_aiter.sh" "$model" rd75-srv 8189 --max-model-len "$MAXLEN" >/dev/null
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        curl -sf http://127.0.0.1:8189/health >/dev/null 2>&1 && break
        docker ps --format '{{.Names}}' | grep -q '^rd75-srv$' || {
            echo "  EXITED -- log at $LOGS/rd75-$tag.startup-fail"
            docker logs rd75-srv > "$LOGS/rd75-$tag.startup-fail" 2>&1 || true
            grep -aiE "Error|Exception|not support|Unsupported|out of memory" "$LOGS/rd75-$tag.startup-fail" \
              | grep -avE "core_client|launch_core_engines|wait_for_engine_startup|contextlib|triton_kernels|AVX2" | head -6
            return 1; }
        sleep 10; t=$((t+10))
    done
    [ "$t" -lt "$READY_TIMEOUT" ] || { echo "  timeout"; return 1; }
    echo "  ready (${t}s)"
    # Which GEMM actually ran -- round 73 reported a ratio it had not earned
    # because it never checked. Not an assertion here (the two arms are
    # DIFFERENT quantizations and legitimately select different kernels), but
    # it is recorded so the comparison can be read honestly.
    docker logs rd75-srv 2>&1 | grep -aoE "Selected [A-Za-z0-9]+(ScaledMM|MPLinear|Marlin)[A-Za-z]*" | sort -u | head -3 | sed 's/^/  kernel: /'
    docker logs rd75-srv 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'

    # PREFILL: output-len 1 so TTFT is prefill and nothing else.
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$CLIENT_IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url http://127.0.0.1:8189 --model "$model" --served-model-name bench \
        --dataset-name random --random-input-len "$PRE_CTX" --random-output-len 1 \
        --num-prompts 4 --max-concurrency 1 --ignore-eos --seed 1234 \
        > "$LOGS/rd75-$tag-prefill.bench" 2>&1
    # DECODE: matched to llama.cpp's 130,000-token point.
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$CLIENT_IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url http://127.0.0.1:8189 --model "$model" --served-model-name bench \
        --dataset-name random --random-input-len "$DEEP" --random-output-len "$DEEP_OUT" \
        --num-prompts 3 --max-concurrency 1 --ignore-eos --seed 1234 \
        > "$LOGS/rd75-$tag-deep.bench" 2>&1

    PRE_CTX="$PRE_CTX" python3 - "$LOGS/rd75-$tag-prefill.bench" "$LOGS/rd75-$tag-deep.bench" <<'PY'
import os, re, sys
def g(p, pat):
    try: s = open(p, errors="replace").read()
    except OSError: return None
    m = re.search(pat, s)
    if not m: return None
    v = float(m.group(1)); return v if v > 0 else None
def failed(p):
    try: s = open(p, errors="replace").read()
    except OSError: return 0
    m = re.search(r"Failed requests: +(\d+)", s)
    return int(m.group(1)) if m else 0
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
pre, deep = sys.argv[1], sys.argv[2]
n = int(os.environ["PRE_CTX"])
t = g(pre, TTFT)
if failed(pre) or not t: print("  prefill  FAILED")
else: print(f"  prefill  TTFT {t:8.1f} ms   {n/(t/1000.0):9.1f} tok/s")
d = g(deep, TPOT)
if failed(deep) or not d: print("  decode   FAILED")
else: print(f"  decode   TPOT {d:8.2f} ms   {1000.0/d:9.2f} tok/s  @130k")
PY
    docker rm -f rd75-srv >/dev/null 2>&1
}

run_arm awq  t80-awq  || echo "(W4A16 arm failed)"
run_arm w8a8 t80-w8a8 || echo "(W8A8 arm failed)"

echo ""
echo "=== $(date -u +%T) round 75 done ==="
PRE_CTX="$PRE_CTX" python3 - <<'PY'
import os, re
L = "/mnt/llm-storage/bench-matrix/logs"
N = int(os.environ["PRE_CTX"])
def g(tag, kind, pat):
    p = os.path.join(L, f"rd75-{tag}-{kind}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    v = float(m.group(1)); return v if v > 0 else None
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
LLAMA_PRE, LLAMA_DEC = 2546.0, 62.89

rows = []
for tag, label, gb in [("awq", "W4A16 (t80-awq)", "46 GB"), ("w8a8", "W8A8 (t80-w8a8)", "77 GB")]:
    t, d = g(tag, "prefill", TTFT), g(tag, "deep", TPOT)
    rows.append((label, gb, N/(t/1000.0) if t else None, 1000.0/d if d else None))
rows.append(("llama.cpp Q4_K_M f16KV", "45 GB", LLAMA_PRE, LLAMA_DEC))

dash = "-"
print(f"{'config':<24}{'weights':>9}{'prefill t/s':>13}{'decode t/s':>12}{'vs llama pre':>14}{'vs llama dec':>14}")
print("-" * 86)
for label, gb, p, d in rows:
    print(f"{label:<24}{gb:>9}"
          f"{(f'{p:13.1f}' if p else f'{dash:>13}')}"
          f"{(f'{d:12.2f}' if d else f'{dash:>12}')}"
          f"{(f'{p/LLAMA_PRE:13.2f}x' if p else f'{dash:>14}')}"
          f"{(f'{d/LLAMA_DEC:13.2f}x' if d else f'{dash:>14}')}")

aw_p, aw_d = rows[0][2], rows[0][3]
w8_p, w8_d = rows[1][2], rows[1][3]
print()
if aw_p and w8_p and aw_d and w8_d:
    print(f"W4A16 vs W8A8:  prefill {aw_p/w8_p:.3f}x   decode {aw_d/w8_d:.3f}x")
    print()
    # The MAGNITUDE carries the information, not the sign. Halving weight bytes
    # in a weight-bandwidth-bound decode would buy ~2x; anything much smaller
    # means weight traffic is a minor term. Solve for the split between
    # weight-traffic time W and everything else O, given (W+O)/(W/2+O) = r:
    #     O = W * (2 - r) / (2 * (r - 1))
    r = aw_d / w8_d
    if r > 1.001:
        o_over_w = (2 - r) / (2 * (r - 1))
        frac = 1.0 / (1.0 + o_over_w)          # weight traffic as share of decode
        ceiling = w8_d * (1.0 + o_over_w) / o_over_w   # decode if weights were FREE
        print(f"Halving weight bytes bought {r:.3f}x. Weight-bandwidth-bound decode")
        print(f"would buy ~2x, so weight traffic is only ~{frac*100:.0f}% of decode time;")
        print(f"the other ~{(1-frac)*100:.0f}% is latency/serialization (docs/50).")
        print(f"CEILING: with FREE weights, decode reaches ~{ceiling:.1f} tok/s.")
        if ceiling < LLAMA_DEC:
            print(f"That is still short of llama.cpp's {LLAMA_DEC:.2f}, so NO vLLM checkpoint")
            print("closes the decode gap by getting smaller. The gap is kernel")
            print("efficiency, not weight bytes.")
    else:
        print("DECODE DID NOT IMPROVE despite halving weight bytes per token. That")
        print("CONFIRMS docs/50's latency-bound diagnosis at 80B scale: the card is")
        print("not waiting on weight traffic, so cheaper weights buy nothing. It")
        print("also means no smaller checkpoint will close the decode gap.")
    print()
    if aw_p >= w8_p * 0.95 and aw_d >= LLAMA_DEC:
        print("BEST OF BOTH: llama.cpp's footprint, vLLM's prefill, llama.cpp's decode.")
        print("The stack split from docs/55 collapses and this is the checkpoint.")
    elif aw_p >= w8_p * 0.95:
        print("W4A16 keeps vLLM's prefill at 40% of the weight memory -- worth having")
        print("even though decode still trails llama.cpp.")
    else:
        print("W4A16 gives up prefill, which was vLLM's whole advantage. The docs/55")
        print("split stands: prefill -> vLLM W8A8, decode and long context -> llama.cpp.")
PY
