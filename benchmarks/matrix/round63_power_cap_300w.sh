#!/usr/bin/env bash
# Round 63: raise the power cap 200W -> 300W. Hardware change, monitored.
#
# THE FINDING THIS TESTS. docs/51 sampled clocks every 20 s through round 57 and
# found the cards pinned at EXACTLY 199-200 W in every sample while clocking
# 1235-1375 MHz against a 1700 MHz boost, at 67-71 C junction. Flat power with
# thermal headroom is a POWER CAP, not a thermal limit:
#
#     card2  power1_cap = 200W   power1_cap_max = 300W
#     card3  power1_cap = 200W   power1_cap_max = 300W
#
# ~25% of clock left unused. Every software lead in docs/50-52 fought for 1-3%.
#
# WHY IT WAS NOT DONE THEN. These are PASSIVE MI210s -- no fans of their own,
# cooled entirely by chassis fans driven by gpu-fan-control.service. Raising the
# cap by 50% raises thermal load by roughly the same, and overheating degrades
# hardware slowly rather than erroring loudly. The owner has since confirmed the
# cooling headroom exists, so this proceeds -- but as a MEASURED, MONITORED,
# REVERSIBLE change rather than a setting quietly flipped.
#
# WHAT SHOULD MOVE, AND WHAT SHOULD NOT. Prefill is compute-bound, so clock
# feeds it directly and it should gain. Decode is memory-bound and mclk is
# ALREADY at its 1600 MHz maximum (docs/51), so decode should gain little or
# nothing. A round that shows both moving equally is suspicious and probably
# measuring something else. Both are therefore driven separately:
#     prefill-weighted   8192 in / 32 out
#     decode-weighted     512 in / 512 out
#
# SAFETY, IN ORDER OF PRECEDENCE:
#   1. A background sampler reads BOTH junction and HBM temp every 10 s and
#      restores 200 W the moment either exceeds its threshold. HBM is the
#      tighter limit on this hardware (crit 94 C vs junction 100 C) and the
#      cards are asymmetric -- docs/51 saw GPU0 HBM idling 14 C above GPU1 --
#      so watching junction alone would have let memory reach within 4 C of
#      critical before reacting.
#   2. An EXIT/INT/TERM trap restores 200 W on any abnormal termination,
#      including a kill from outside.
#   3. Steps are incremental -- 200, 250, 300 -- so a thermal problem shows up
#      at 250 W with margin left, rather than at full power.
#   4. The cap is verified as actually applied after each write; sysfs accepts
#      values it then clamps, so writing is not the same as taking effect.
#
# power1_cap is sysfs and does NOT survive a reboot, which makes the whole
# change inherently reversible. Persisting it is a separate, deliberate act.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=vllm-mi210:gdnpolicy
MODEL=$BASE/t35-w8a8

# TWO thresholds, because HBM is the tighter constraint -- verified on this
# hardware, not assumed:
#     temp2 junction  crit 100C  emergency 105C
#     temp3 mem       crit  94C  emergency  99C
# Memory trips 6C earlier than junction, and docs/51 recorded GPU0 HBM idling
# 14C hotter than GPU1 (55 vs 41), so the hotter card's memory is what actually
# limits this box. An earlier draft of this round watched junction only and
# would have run HBM to within 4C of its critical point before reacting.
ABORT_TEMP=${ABORT_TEMP:-90}       # junction, 10C below crit
ABORT_MEM=${ABORT_MEM:-85}         # memory,    9C below crit
STEPS=${STEPS:-"200 250 300"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
CONC=${CONC:-8}

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

# Only 0x740f (MI210). The box has another AMD device with a 190 W cap that
# must not be touched.
HWMONS=()
for d in /sys/class/drm/card*/device; do
    [ -e "$d/vendor" ] || continue
    [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] || continue
    [ "$(cat "$d/device" 2>/dev/null)" = "0x740f" ] || continue
    hw=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
    [ -n "$hw" ] && HWMONS+=("$hw")
done
[ "${#HWMONS[@]}" -eq 2 ] || { echo "FATAL: expected 2 MI210s, found ${#HWMONS[@]}"; exit 1; }
echo "MI210 hwmon nodes: ${HWMONS[*]}"

ORIG=()
for hw in "${HWMONS[@]}"; do ORIG+=("$(cat "$hw/power1_cap")"); done
echo "original caps: $(( ${ORIG[0]} / 1000000 ))W $(( ${ORIG[1]} / 1000000 ))W"

# VERIFY, DO NOT ASSERT. The first run of this round printed "caps restored"
# from an unconditional echo while the cards were still sitting at 300 W --
# restore_caps never actually ran, and nothing checked. A safety claim that is
# printed rather than verified is worse than no claim, because it stops anyone
# looking. This version reads the caps back and says loudly if they are wrong.
restore_caps() {
    local i ok=1 got
    for i in 0 1; do
        sudo tee "${HWMONS[$i]}/power1_cap" >/dev/null <<< "${ORIG[$i]}" 2>/dev/null || ok=0
    done
    sleep 1
    for i in 0 1; do
        got=$(cat "${HWMONS[$i]}/power1_cap" 2>/dev/null || echo 0)
        if [ "$got" != "${ORIG[$i]}" ]; then
            ok=0
            echo "  !!! CAP NOT RESTORED on ${HWMONS[$i]}: reads $(( ${got:-0} / 1000000 ))W, wanted $(( ${ORIG[$i]} / 1000000 ))W"
        fi
    done
    if [ "$ok" = "1" ]; then
        echo "  caps VERIFIED restored to $(( ${ORIG[0]} / 1000000 ))W"
    else
        echo "  !!! CAPS LEFT MODIFIED. Restore by hand:"
        for i in 0 1; do
            echo "        echo ${ORIG[$i]} | sudo tee ${HWMONS[$i]}/power1_cap"
        done
    fi
}
cleanup() {
    docker rm -f rd63-srv probe-rd63-temp >/dev/null 2>&1 || true
    restore_caps
}
trap cleanup EXIT INT TERM

set_cap() {  # watts -> echoes actual applied watts
    local w=$1 uw=$(( $1 * 1000000 )) ok=1
    for hw in "${HWMONS[@]}"; do
        sudo tee "$hw/power1_cap" >/dev/null <<< "$uw" 2>/dev/null || ok=0
    done
    sleep 1
    local got
    for hw in "${HWMONS[@]}"; do
        got=$(( $(cat "$hw/power1_cap") / 1000000 ))
        # sysfs silently CLAMPS out-of-range values, so a successful write is
        # not evidence the cap took effect. Verify.
        [ "$got" = "$w" ] || { echo "  WARNING: asked ${w}W, cap reads ${got}W"; ok=0; }
    done
    [ "$ok" = "1" ] && echo "  cap set to ${w}W on both cards"
}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 63: power cap sweep, abort at junction ${ABORT_TEMP}C / mem ${ABORT_MEM}C ==="

# Watchdog: restores the cap and kills this run if junction temp runs away.
( while true; do
    for hw in "${HWMONS[@]}"; do
        tj=$(( $(cat "$hw/temp2_input" 2>/dev/null || echo 0) / 1000 ))   # junction
        tm=$(( $(cat "$hw/temp3_input" 2>/dev/null || echo 0) / 1000 ))   # memory
        why=""
        [ "$tj" -gt "$ABORT_TEMP" ] 2>/dev/null && why="junction ${tj}C > ${ABORT_TEMP}C"
        [ "$tm" -gt "$ABORT_MEM"  ] 2>/dev/null && why="memory ${tm}C > ${ABORT_MEM}C (HBM crit is 94C)"
        if [ -n "$why" ]; then
            echo "!!! ABORT: $why -- restoring caps" | tee -a "$LOGS/round63-abort.log"
            for i in 0 1; do sudo tee "${HWMONS[$i]}/power1_cap" >/dev/null <<< "${ORIG[$i]}" 2>/dev/null; done
            pkill -f "[r]ound63_power" 2>/dev/null
            exit 1
        fi
    done
    sleep 10
done ) &
WATCHDOG=$!

wait_ready() {
    local t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:8140/health" >/dev/null 2>&1 && { echo "  server ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q '^rd63-srv$' || { echo "  server EXITED"; docker logs rd63-srv 2>&1 | tail -20; return 1; }
        sleep 10; t=$((t+10))
    done
    return 1
}

sample_hw() {  # label -> junction temps, sclk, power at this instant
    local lbl="$1" out=""
    for hw in "${HWMONS[@]}"; do
        local t p
        t=$(( $(cat "$hw/temp2_input" 2>/dev/null || echo 0) / 1000 ))
        m=$(( $(cat "$hw/temp3_input" 2>/dev/null || echo 0) / 1000 ))
        p=$(( $(cat "$hw/power1_average" 2>/dev/null || echo 0) / 1000000 ))
        out+="j${t}C/m${m}C/${p}W "
    done
    local sclk
    sclk=$(docker exec rd63-srv rocm-smi --showclocks 2>/dev/null | grep -oE "sclk clock level: [0-9]+: \([0-9]+Mhz\)" | grep -oE "[0-9]+Mhz" | tr '\n' ' ')
    echo "    [$lbl] $out sclk: ${sclk:-?}"
}

bench() {  # inlen outlen outfile label
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url http://127.0.0.1:8140 \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$1" --random-output-len "$2" \
        --num-prompts $(( CONC * 8 )) --max-concurrency "$CONC" \
        --ignore-eos --seed 1234 > "$3" 2>&1 &
    local bp=$!
    sleep 25; sample_hw "$4 mid-run"
    wait $bp
    grep -E "Output token throughput|Total Token throughput|Median TTFT" "$3" | sed 's/^/      /'
}

for W in $STEPS; do
    echo ""
    echo "=== $(date -u +%T) ===== power cap ${W}W ====="
    set_cap "$W"
    docker rm -f rd63-srv >/dev/null 2>&1
    TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
    VLLM_EXTRA_ENV="-e VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1" \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" rd63-srv 8140 --max-model-len 32768 >/dev/null
    wait_ready || { echo "  server failed at ${W}W"; continue; }
    sample_hw "idle"
    echo "  -- prefill-weighted (8192 in / 32 out) -- clock should matter here"
    bench 8192 32  "$LOGS/rd63-${W}w-prefill.bench" "prefill"
    echo "  -- decode-weighted (512 in / 512 out) -- mclk already maxed, expect little"
    bench 512 512  "$LOGS/rd63-${W}w-decode.bench" "decode"
    docker rm -f rd63-srv >/dev/null 2>&1
done

kill $WATCHDOG 2>/dev/null || true

echo ""
echo "=== $(date -u +%T) round 63 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
STEPS = [int(w) for w in os.environ.get("STEPS", "200 250 300").split()]
def v(w, kind, pat):
    p = os.path.join(L, f"rd63-{w}w-{kind}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
TOT = r"Total Token throughput \(tok/s\):\s*([\d.]+)"
OUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
base_p, base_d = v(STEPS[0], "prefill", TOT), v(STEPS[0], "decode", OUT)
print(f"{'cap':>6}{'prefill tot tok/s':>20}{'vs 200W':>10}{'decode out tok/s':>19}{'vs 200W':>10}")
print("-" * 65)
for w in STEPS:
    p, d = v(w, "prefill", TOT), v(w, "decode", OUT)
    rp = f"{p/base_p:9.3f}x" if p and base_p else f"{'-':>10}"
    rd_ = f"{d/base_d:9.3f}x" if d and base_d else f"{'-':>10}"
    print(f"{w:>5}W{(f'{p:20.2f}' if p else f'{chr(45):>20}')}{rp}"
          f"{(f'{d:19.2f}' if d else f'{chr(45):>19}')}{rd_}")
print()
print("EXPECTED SHAPE: prefill gains (compute-bound, clock feeds it directly),")
print("decode gains little (memory-bound, and mclk was already at its 1600 MHz")
print("maximum before any of this). If DECODE moves as much as prefill, be")
print("suspicious -- that is not what raising a power cap should do, and the")
print("round is probably measuring run-to-run variance instead.")
print()
print("Check the sclk samples above: if clocks did NOT rise with the cap, the")
print("cards were not power-limited after all and the throughput rows are noise.")
PY
echo ""
echo "The EXIT trap now restores the caps AND READS THEM BACK. If you do not see"
echo "a 'caps VERIFIED restored' line below, the cards are still modified and"
echo "the exact commands to fix that will have been printed instead."
echo "power1_cap is sysfs and does not survive a reboot, so a permanent setting"
echo "needs a deliberate unit file -- see configs/gpu-power-cap.service."
