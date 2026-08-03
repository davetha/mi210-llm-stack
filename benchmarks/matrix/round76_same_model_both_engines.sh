#!/usr/bin/env bash
# Round 76: vLLM vs llama.cpp on the SAME base model, finally.
#
# WHAT WAS WRONG WITH EVERY COMPARISON BEFORE THIS. docs/55 reports vLLM
# prefilling 2.77-3.07x faster and llama.cpp decoding 1.22x faster, and every
# one of those numbers puts two DIFFERENT models against each other:
#
#     vLLM       t80-awq / t80-w8a8   Qwen3-Next-80B-A3B-THINKING
#     llama.cpp  coder-next-q4        Qwen3-Coder-Next-80B-abliterated
#
# Same family, both 80B-A3B, both ~4-bit on the llama.cpp side -- but not the
# same weights. Every caveat line in docs/54 and docs/55 says so, which is
# honest and does not make the comparison valid. An "engine" difference that
# also changes the model is measuring both at once.
#
# THE MATCHED PAIR, ON DISK THE WHOLE TIME.
#
#     vLLM       t80-awq         Qwen3-Next-80B-A3B-Thinking  W4A16    46.0 GB
#     llama.cpp  t80-gguf-q4km   Qwen3-Next-80B-A3B-Thinking  Q4_K_M   45.4 GB
#
# Same base model, comparable bit width, footprints within 1.3%. This is the
# comparison the previous four rounds were reaching for.
#
# WHAT COULD CHANGE. The standing claims are prefill 2.77x to vLLM and decode
# 1.22x to llama.cpp. If those hold on matched weights, they are engine
# properties. If either moves materially, it was partly a model difference and
# docs/55's headline needs rewriting. Both outcomes are worth having; only one
# of them is currently assumed.
#
# HARNESS LESSONS FOLDED IN, EACH FROM A ROUND THAT LOST DATA TO IT:
#   round 69 - llama-bench writes its CSV at exit, so one invocation covering
#              several depths loses everything if the deepest crashes. Every
#              (engine, depth) cell here is its own process.
#   round 71 - `docker rm -f` returns before the driver has reclaimed VRAM, and
#              a cell started into that window read 6.68 tok/s where the true
#              value was 52.72, then hard-faulted on a retry. Cells wait for
#              VRAM to drain first.
#   round 66 - prompt + generation must fit inside the context window or every
#              request is rejected and reported as 0.00.
#   round 73 - VLLM_IMAGE is what serve_vllm_aiter.sh reads for the SERVER;
#              setting only the client image ran three arms on the wrong build.
#              Pinned explicitly here.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs

LCPP_IMG=${LCPP_IMG:-llama-rocm714-bench:latest}
GGUF=${GGUF:-$BASE/t80-gguf-q4km/Qwen_Qwen3-Next-80B-A3B-Thinking-Q4_K_M.gguf}
VLLM_SRV_IMG=${VLLM_SRV_IMG:-vllm-mi210:v0.26.1rc0-ckgemm-warm}
VMODEL=${VMODEL:-$BASE/t80-awq}

docker image inspect "$LCPP_IMG"     >/dev/null 2>&1 || { echo "FATAL: missing $LCPP_IMG"; exit 1; }
docker image inspect "$VLLM_SRV_IMG" >/dev/null 2>&1 || { echo "FATAL: missing $VLLM_SRV_IMG"; exit 1; }
[ -f "$GGUF" ]              || { echo "FATAL: no gguf at $GGUF"; exit 1; }
[ -f "$VMODEL/config.json" ] || { echo "FATAL: no model at $VMODEL"; exit 1; }

DEPTHS=${DEPTHS:-"8192 32768 65536 130000"}
PRE_CTX=${PRE_CTX:-16384}
NGEN=${NGEN:-32}
MAXLEN=${MAXLEN:-163840}
CELL_TIMEOUT=${CELL_TIMEOUT:-2400}
READY_TIMEOUT=${READY_TIMEOUT:-2400}

for d in $DEPTHS; do
    [ $(( d + NGEN + 1024 )) -le "$MAXLEN" ] || { echo "FATAL: depth $d + gen > $MAXLEN"; exit 1; }
done

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 76: same model, both engines ==="
echo "    model:  Qwen3-Next-80B-A3B-Thinking"
echo "    vLLM:   $(basename "$VMODEL")        W4A16  $(du -sh "$VMODEL" 2>/dev/null | cut -f1)"
echo "    llama:  $(basename "$GGUF")  Q4_K_M $(du -sh "$GGUF" 2>/dev/null | cut -f1)"
echo "    depths: $DEPTHS   prefill: $PRE_CTX   n-gen: $NGEN"

cleanup() { docker rm -f rd76-lcpp rd76-vllm >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

vram_mib() {
    local tot=0 u
    for d in /sys/class/drm/card*/device; do
        [ -f "$d/mem_info_vram_used" ] || continue
        [ "$(cat "$d/device" 2>/dev/null)" = "0x740f" ] || continue
        u=$(cat "$d/mem_info_vram_used"); tot=$(( tot + u / 1048576 ))
    done
    echo "$tot"
}
settle() {  # round 71: never start a cell into a teardown
    local t=0 used
    while [ $t -lt 180 ]; do
        used=$(vram_mib); [ "$used" -lt 512 ] && { sleep 5; return 0; }
        sleep 5; t=$((t+5))
    done
    echo "    WARNING: VRAM still ${used} MiB after ${t}s -- cell may race a teardown"
}

########## llama.cpp: one process per cell ##########
echo ""
echo "=== $(date -u +%T) llama.cpp  (Q4_K_M, f16 KV) ==="
lcpp_cell() {  # extra_args... ; writes csv to $OUT
    docker rm -f rd76-lcpp >/dev/null 2>&1
    settle
    timeout "$CELL_TIMEOUT" docker run --rm --name rd76-lcpp --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v /mnt/llm-storage:/mnt/llm-storage \
      --entrypoint /src/build/bin/llama-bench "$LCPP_IMG" \
      -m "$GGUF" -ngl 999 -fa on -ctk f16 -ctv f16 -r 1 -o csv "$@" > "$OUT" 2>"${OUT%.csv}.err"
    local rc=$?
    docker rm -f rd76-lcpp >/dev/null 2>&1
    return $rc
}
csv_ts() { python3 - "$1" <<'PY'
import csv, sys
try:
    r = list(csv.DictReader(open(sys.argv[1], errors="replace")))
    print(f"{float(r[-1]['avg_ts']):.2f}" if r else "")
except Exception: print("")
PY
}

OUT="$LOGS/rd76-lcpp-prefill.csv"; lcpp_cell -p "$PRE_CTX" -n 0 || true
v=$(csv_ts "$OUT"); printf "  prefill p=%-7s %10s tok/s\n" "$PRE_CTX" "${v:-FAILED}"
for d in $DEPTHS; do
    OUT="$LOGS/rd76-lcpp-d$d.csv"; lcpp_cell -p 0 -n "$NGEN" -d "$d" || true
    v=$(csv_ts "$OUT"); printf "  decode  d=%-7s %10s tok/s\n" "$d" "${v:-FAILED}"
done

########## vLLM: one server, benches per depth ##########
echo ""
echo "=== $(date -u +%T) vLLM  (W4A16) ==="
settle
VLLM_IMAGE="$VLLM_SRV_IMG" TP=2 \
  VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
  "$BIN/serve_vllm_aiter.sh" "$VMODEL" rd76-vllm 8190 --max-model-len "$MAXLEN" >/dev/null
t=0
while [ "$t" -lt "$READY_TIMEOUT" ]; do
    curl -sf http://127.0.0.1:8190/health >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q '^rd76-vllm$' || {
        echo "  vLLM EXITED -- log at $LOGS/rd76-vllm.startup-fail"
        docker logs rd76-vllm > "$LOGS/rd76-vllm.startup-fail" 2>&1 || true
        grep -aiE "Error|Exception|not support|out of memory" "$LOGS/rd76-vllm.startup-fail" \
          | grep -avE "core_client|launch_core_engines|contextlib|triton_kernels|AVX2" | head -5
        break; }
    sleep 10; t=$((t+10))
done
if curl -sf http://127.0.0.1:8190/health >/dev/null 2>&1; then
    echo "  ready (${t}s)"
    docker logs rd76-vllm 2>&1 | grep -aoE "Selected [A-Za-z0-9]+(ScaledMM|MPLinear|Marlin)[A-Za-z]*" | sort -u | head -2 | sed 's/^/  kernel: /'
    docker logs rd76-vllm 2>&1 | grep -aoiE "GPU KV cache size: [0-9,]+ tokens" | tail -1 | sed 's/^/  /'

    vbench() {  # inlen outlen prompts outfile
        docker run --rm --network host -v "$BASE":"$BASE" \
          --entrypoint /opt/python/bin/vllm "$VLLM_SRV_IMG" bench serve \
            --backend openai-chat --endpoint /v1/chat/completions \
            --base-url http://127.0.0.1:8190 --model "$VMODEL" --served-model-name bench \
            --dataset-name random --random-input-len "$1" --random-output-len "$2" \
            --num-prompts "$3" --max-concurrency 1 --ignore-eos --seed 1234 > "$4" 2>&1
        grep -aoE "Failed requests: +[0-9]+" "$4" | grep -aoE "[0-9]+$" | head -1
    }
    f=$(vbench "$PRE_CTX" 1 4 "$LOGS/rd76-vllm-prefill.bench")
    tt=$(grep -aoE "Median TTFT \(ms\): +[0-9.]+" "$LOGS/rd76-vllm-prefill.bench" | grep -aoE "[0-9.]+$" | head -1)
    if [ "${f:-0}" -gt 0 ] || [ -z "$tt" ]; then printf "  prefill p=%-7s %10s\n" "$PRE_CTX" "FAILED"
    else printf "  prefill p=%-7s %10.2f tok/s\n" "$PRE_CTX" "$(python3 -c "print($PRE_CTX/($tt/1000.0))")"; fi
    for d in $DEPTHS; do
        f=$(vbench "$d" "$NGEN" 3 "$LOGS/rd76-vllm-d$d.bench")
        tp=$(grep -aoE "Median TPOT \(ms\): +[0-9.]+" "$LOGS/rd76-vllm-d$d.bench" | grep -aoE "[0-9.]+$" | head -1)
        if [ "${f:-0}" -gt 0 ] || [ -z "$tp" ]; then printf "  decode  d=%-7s %10s\n" "$d" "FAILED"
        else printf "  decode  d=%-7s %10.2f tok/s\n" "$d" "$(python3 -c "print(1000/$tp)")"; fi
    done
fi
docker rm -f rd76-vllm >/dev/null 2>&1

echo ""
echo "=== $(date -u +%T) round 76 done ==="
DEPTHS="$DEPTHS" PRE_CTX="$PRE_CTX" python3 - <<'PY'
import csv, os, re
L = "/mnt/llm-storage/bench-matrix/logs"
DEPTHS = [int(d) for d in os.environ["DEPTHS"].split()]
N = int(os.environ["PRE_CTX"])

def lcpp(p):
    p = os.path.join(L, p)
    if not os.path.isfile(p): return None
    try:
        r = list(csv.DictReader(open(p, errors="replace")))
        return float(r[-1]["avg_ts"]) if r else None
    except Exception: return None
def vll(p, pat):
    p = os.path.join(L, p)
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    if not m: return None
    v = float(m.group(1)); return v if v > 0 else None
TTFT = r"Median TTFT \(ms\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"

lp = lcpp("rd76-lcpp-prefill.csv")
vt = vll("rd76-vllm-prefill.bench", TTFT)
vp = N / (vt / 1000.0) if vt else None
dash = "-"

print("SAME BASE MODEL: Qwen3-Next-80B-A3B-Thinking")
print("  vLLM      t80-awq        W4A16   46.0 GB")
print("  llama.cpp t80-gguf-q4km  Q4_K_M  45.4 GB")
print()
print(f"{'metric':<20}{'llama.cpp':>12}{'vLLM':>12}{'vLLM/llama':>12}")
print("-" * 56)
print(f"{'prefill @'+str(N):<20}"
      f"{(f'{lp:12.1f}' if lp else f'{dash:>12}')}"
      f"{(f'{vp:12.1f}' if vp else f'{dash:>12}')}"
      f"{(f'{vp/lp:11.2f}x' if lp and vp else f'{dash:>12}')}")
for d in DEPTHS:
    a = lcpp(f"rd76-lcpp-d{d}.csv")
    t = vll(f"rd76-vllm-d{d}.bench", TPOT)
    b = 1000.0 / t if t else None
    print(f"{'decode @'+str(d):<20}"
          f"{(f'{a:12.2f}' if a else f'{dash:>12}')}"
          f"{(f'{b:12.2f}' if b else f'{dash:>12}')}"
          f"{(f'{b/a:11.2f}x' if a and b else f'{dash:>12}')}")

print()
print("THE CLAIMS UNDER TEST, from docs/55 -- both measured on MISMATCHED models:")
print("    prefill  vLLM 2.77x faster      decode  llama.cpp 1.22x faster")
print("If the ratios above land near those, they are ENGINE properties and")
print("docs/55 stands. If either moves materially, part of what was attributed")
print("to the engine was the model, and docs/55's headline needs rewriting.")
print()
print("Note the llama.cpp arm is Q4_K_M with f16 KV -- the configuration round")
print("70 showed beats q8_0 KV at every depth, NOT what production runs today.")
PY
