#!/usr/bin/env bash
# Round 57: where is the real ASM-vs-HIP paged-attention crossover on CDNA2?
#
# THE QUESTION. aiter/ops/attention.py:_should_use_asm_kernel() ends with
#     return num_seqs * num_heads > 2 * cu_num
# which is 208 on a 104-CU MI210. At TP=2 this model has 16 heads per rank, so
# AITER's ASM paged attention engages only at num_seqs >= 14. Below that vLLM
# runs paged_attention_ll4mi -- 13.1% of decode per docs/45.
#
# WHY THE CONSTANT IS SUSPECT. Scaling the threshold by CU count is defensible.
# The factor 2 is not obviously portable: it encodes how the ASM kernel's
# occupancy compares to the HIP fallback's, which is an ARCHITECTURE property,
# not a CU-count one. AMD calibrated it on CDNA3.
#
# WHAT WE ALREADY MEASURED. docs/50 round 56 ran num_seqs 32 -- 512 heads, 2.5x
# over the threshold -- and the ASM path WON: 1.033x throughput, 0.975x TTFT,
# 0.957x TPOT, with pa_bf16_noquant_gqa8_1tg_4w.co confirmed loaded. So the
# kernel is good on this card. The region 1..13 has never been probed, and that
# is where single-stream and light-concurrency serving actually lives -- every
# other decode measurement in this repo is batch-1.
#
# THE DESIGN, AND WHY IT IS TWO SERVERS AND NOT TWELVE. The force setting is
# read per dispatch from the environment, but concurrency is a property of the
# LOAD, not the server. So one server per arm, six concurrency points each:
# 2 server starts instead of 12. Both arms set
# VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 because without it rocm_aiter_fa.py:1283
# never reaches paged_attention_common at all and both arms would silently be
# HIP -- which would produce a beautiful, meaningless null.
#
# READING THE RESULT. The interesting number is where ASM/HIP crosses 1.0.
#   crossover well below 208 -> the stock heuristic costs us on mid-size batches
#   crossover near 208       -> AMD's constant transfers to CDNA2; close it
#   ASM wins at num_seqs 1   -> 13.1% of decode has been on the wrong kernel
# A loss at low batch is a real answer too: the heuristic exists because
# low-occupancy ASM dispatch is presumably worse, and that may simply be true.
#
# PIGGYBACKED: clocks/power/temperature under sustained load. Everything
# measured so far was at IDLE (42 W, sclk parked at 800 MHz vs a 1700 MHz
# boost), which rules out a permanent handicap but says nothing about throttling
# during a long run -- and these are passive cards. GPU0's HBM also idles 14 C
# hotter than GPU1's, unexplained. Sampled here for free.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
CFGDIR=$BASE/tune-gfx90a

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }
HOOK=$CFGDIR/asm_pa_threshold_gfx90a.py
[ -f "$HOOK" ] || { echo "FATAL: missing $HOOK (copy configs/asm_pa_threshold_gfx90a.py there)"; exit 1; }

IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"1 2 4 8 16 32"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 57: ASM paged-attention crossover sweep ==="
echo "concurrency points: $CONCS   (stock heuristic engages ASM at num_seqs >= 14)"

cleanup() { docker rm -f rd57-asm rd57-hip probe-clocks >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

# ---- clock/thermal sampler (item 9), runs for the whole round ----------
docker run -d --name probe-clocks \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --entrypoint bash "$IMG" -c '
    while true; do
      printf "%s " "$(date -u +%T)"
      rocm-smi --showclocks --showtemp --showpower 2>/dev/null \
        | grep -oE "sclk clock level: [0-9]+: \(([0-9]+)Mhz\)|Temperature \(Sensor junction\) \(C\): [0-9.]+|Average Graphics Package Power \(W\): [0-9.]+" \
        | grep -oE "[0-9.]+(Mhz)?" | tr "\n" " "
      echo
      sleep 20
    done' >/dev/null 2>&1

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || { echo "  FATAL: $name exited"; docker logs "$name" 2>&1 | tail -20; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

start_arm() {  # name port force
    local name="$1" port="$2" force="$3"
    echo ""
    echo "=== $(date -u +%T) starting $name  AITER_PA_ASM_FORCE=$force ==="
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 -e AITER_PA_ASM_FORCE=$force -v $CFGDIR:/cfg" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" --max-model-len 32768 >/dev/null
    wait_ready "$port" "$name" || return 1
    local co
    co=$(docker logs "$name" 2>&1 | grep -ohE "pa_[a-z0-9_]*\.co" | sort -u | tr '\n' ' ')
    echo "  pa .co at startup: ${co:-<none yet -- kernels load on first decode>}"
}

bench_at() {  # port conc outfile
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$1" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts $(( $2 * 8 )) --max-concurrency "$2" \
        --ignore-eos --seed 1234 2>&1 | tee "$3" | grep -E "Output token throughput|Median TPOT|Median TTFT"
}

# THE PATCH IS APPLIED INSIDE THE SERVER CONTAINER, not baked into the image,
# so a failure to apply is loud rather than silent. serve_vllm_aiter.sh has no
# pre-start hook, so patch the image once into a derived tag instead.
echo "=== $(date -u +%T) building patched image tag ==="
docker rm -f probe-pahook >/dev/null 2>&1 || true
docker run --name probe-pahook -v "$CFGDIR":/cfg --entrypoint bash "$IMG" -c '
    python3 /cfg/asm_pa_threshold_gfx90a.py && \
    python3 /cfg/asm_pa_threshold_gfx90a.py --assert-patched' || {
        echo "FATAL: PA hook did not apply"; docker rm -f probe-pahook >/dev/null 2>&1; exit 1; }
docker commit probe-pahook vllm-mi210:pahook >/dev/null
docker rm -f probe-pahook >/dev/null 2>&1
echo "  built vllm-mi210:pahook"
export VLLM_IMAGE=vllm-mi210:pahook
IMG=vllm-mi210:pahook

for arm in asm hip; do
    if [ "$arm" = "asm" ]; then force=1; port=8110; else force=0; port=8111; fi
    start_arm "rd57-$arm" "$port" "$force" || { echo "arm $arm FAILED"; exit 1; }
    for c in $CONCS; do
        echo "--- $(date -u +%T) rd57-$arm @ concurrency $c ---"
        bench_at "$port" "$c" "$LOGS/rd57-$arm-c$c.bench"
    done
    co=$(docker logs "rd57-$arm" 2>&1 | grep -ohE "pa_[a-z0-9_]*\.co" | sort -u | tr '\n' ' ')
    echo "  pa .co loaded by rd57-$arm: ${co:-<NONE>}"
    echo "$co" > "$LOGS/rd57-$arm.cofiles"
    docker rm -f "rd57-$arm" >/dev/null 2>&1
done

echo ""
echo "=== $(date -u +%T) clock/thermal samples under load ==="
docker logs probe-clocks 2>&1 | tail -20
docker rm -f probe-clocks >/dev/null 2>&1

echo ""
echo "=== $(date -u +%T) round 57 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "1 2 4 8 16 32").split()]
def val(arm, c, pat):
    p = os.path.join(L, f"rd57-{arm}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
print(f"{'conc':>5}{'heads':>7}{'HIP tok/s':>11}{'ASM tok/s':>11}{'ASM/HIP':>9}"
      f"{'HIP TPOT':>10}{'ASM TPOT':>10}{'TPOT r':>8}")
print("-" * 71)
for c in CONCS:
    h, a = val("hip", c, TPUT), val("asm", c, TPUT)
    ht, at = val("hip", c, TPOT), val("asm", c, TPOT)
    if h is None and a is None: continue
    heads = c * 16   # 32 heads / TP=2
    r  = f"{a/h:8.3f}x" if h and a else f"{'-':>9}"
    rt = f"{at/ht:7.3f}x" if ht and at else f"{'-':>8}"
    print(f"{c:>5}{heads:>7}"
          f"{(f'{h:11.2f}' if h else f'{chr(45):>11}')}"
          f"{(f'{a:11.2f}' if a else f'{chr(45):>11}')}{r}"
          f"{(f'{ht:10.2f}' if ht else f'{chr(45):>10}')}"
          f"{(f'{at:10.2f}' if at else f'{chr(45):>10}')}{rt}")
print()
for arm in ("asm", "hip"):
    f = os.path.join(L, f"rd57-{arm}.cofiles")
    s = open(f).read().strip() if os.path.isfile(f) else ""
    print(f"  rd57-{arm}: pa .co = {s or '<NONE>'}")
print()
print("THE .co LINES DECIDE VALIDITY. rd57-asm must show pa_bf16_noquant_gqa8;")
print("rd57-hip must show none. If the hip arm loaded pa objects the force=0")
print("override failed and every ratio above is comparing ASM to ASM.")
print()
print("'heads' is num_seqs*num_heads, against the stock threshold of 208 --")
print("so rows up to conc 8 are ones the stock heuristic DECLINES. If ASM/HIP")
print("exceeds 1.0 there, the stock constant is costing us. docs/46 puts the")
print("decode noise bar at 1.036x for a single pair of arms; treat smaller")
print("deltas as directional, and prefer the TPOT column, which is the metric")
print("most directly attributable to a decode-path kernel swap.")
PY
