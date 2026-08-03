#!/usr/bin/env bash
# Round 71: how fast is llama.cpp at 262K, 524K and 1M tokens of depth?
#
# WHY THIS ROUND EXISTS. The 1M question was investigated entirely on vLLM
# (rounds 64-68) and answered "technically reachable, practically unusable":
# bf16 caps at 902,160 KV tokens and cannot hold 1M at all, while
# turboquant_4bit_nc holds 2.79M but decays to 8.0 tok/s at 130K and
# extrapolates to ~1 tok/s at 1M.
#
# That framing was wrong because it never asked the production stack. Direct
# VRAM measurement on llama.cpp with the production Coder-Next 80B Q4_K_M:
#
#     -c 65536    49,572 MiB      -c 262144   55,710 MiB
#     -c 196608   53,664 MiB      -c 1048576  80,267 MiB   <- SERVES
#
# of 131,040 MiB total. A 1M context fits with 50 GB to spare, UNCOMPRESSED,
# on an 80B model. vLLM could not fit 1M on a 35B.
#
# THE REASON THE NUMBERS ARE SO SMALL. Qwen3-Coder-Next is a hybrid GDN
# architecture: most layers are gated-delta-net linear attention carrying a
# constant-size recurrent state, and only a minority carry a growing KV cache.
# Weights are 46,235 MiB of the 49,572 at 64K, so KV is ~32 KB/token -- about
# 32 GB at 1M. This is also why quantizing KV saves so little here: q8_0 saved
# 509 MiB at 64K, roughly 1% of VRAM, for up to 57% of decode speed (round 70).
#
# WHAT IS STILL UNKNOWN, AND WHY IT DECIDES THE QUESTION. Fitting is not
# serving. Round 70 measured f16 decode decaying 76.41 -> 62.89 tok/s from depth
# 8,192 to 130,000 -- a gentle 0.823x across a 16x increase. Extrapolating that
# to 1M is exactly the kind of move rounds 65 and 69 punished: round 65's -3.8%
# was an empty-cache artifact, and turboquant's own decay ACCELERATED rather
# than flattening. A curve measured over 8K-130K does not license a claim about
# 1M. So this round measures it.
#
# ISOLATION, PER ROUND 70. Each depth is its own process. Round 69 asked
# llama-bench for several depths in one invocation, crashed at the deepest, and
# lost the completed shallow rows because the CSV is written at exit. At these
# depths a single cell can take many minutes, so losing the set would be
# expensive.
#
# EXPECT THIS TO BE SLOW. Each cell prefills its full depth before timing
# generation. A 1M-token prefill is the dominant cost and the timeout is sized
# for it, not for the 32 generated tokens.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=llama-rocm714-bench:latest
MODEL=${MODEL:-/mnt/llm-storage/coder-next-q4/Huihui-Qwen3-Coder-Next-abliterated.i1-Q4_K_M.gguf}

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

DEPTHS=${DEPTHS:-"262144 524288 1000000"}
NGEN=${NGEN:-32}
REPS=${REPS:-1}
CELL_TIMEOUT=${CELL_TIMEOUT:-3600}
CTK=${CTK:-f16}
CTV=${CTV:-f16}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 71: f16 KV decode at 262K / 524K / 1M depth ==="
echo "    ctk=$CTK ctv=$CTV  n-gen=$NGEN  reps=$REPS  cell timeout=${CELL_TIMEOUT}s"

cleanup() { docker rm -f probe-rd71 >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

vram_total_mib() {
    local tot=0 u
    for d in /sys/class/drm/card*/device; do
        [ -f "$d/mem_info_vram_used" ] || continue
        [ "$(cat "$d/device" 2>/dev/null)" = "0x740f" ] || continue
        u=$(cat "$d/mem_info_vram_used"); tot=$(( tot + u / 1048576 ))
    done
    echo "$tot"
}

# Wait for the GPUs to actually be free before starting a cell.
#
# THIS IS NOT COSMETIC -- IT IS THE DIFFERENCE BETWEEN 52.72 AND A CRASH. The
# first run of this round measured depth 262144 at 6.68 tok/s, an eighth of what
# the curve predicts and SLOWER than the 524288 point above it. Re-running it
# immediately after the 1M cell produced a hard GPU memory access fault. Run a
# third time from an idle GPU, the same configuration returned 52.72 tok/s --
# exactly on the curve between 130000 (62.89) and 524288 (40.55).
#
# Both bad readings started moments after tearing down a container holding tens
# of GB of VRAM. `docker rm -f` returns before the driver has finished
# reclaiming, so the next process races the teardown. Temperatures were 47 C and
# power 42 W when checked, so this is not thermal.
#
# A non-monotonic point is a gift: decode CANNOT get faster with more KV, so the
# curve announced its own bad sample. A single anomalous number at the deepest
# point measured -- with nothing above it to contradict -- would have been
# indistinguishable from a real finding.
settle_gpu() {
    local t=0 used
    while [ $t -lt 180 ]; do
        used=$(vram_total_mib)
        [ "$used" -lt 512 ] && { [ $t -gt 0 ] && echo "    (GPU idle after ${t}s)"; sleep 5; return 0; }
        sleep 5; t=$((t+5))
    done
    echo "    WARNING: VRAM still ${used} MiB after ${t}s -- this cell may be racing a teardown"
}

for d in $DEPTHS; do
    out="$LOGS/rd71-d$d.csv"; err="$LOGS/rd71-d$d.err"
    echo ""
    echo "--- $(date -u +%T) depth $d ---"
    docker rm -f probe-rd71 >/dev/null 2>&1
    settle_gpu
    # -c must exceed the depth plus the generation, or the run is rejected the
    # way round 66's 131072 point was.
    ctx=$(( d + NGEN + 1024 ))
    timeout "$CELL_TIMEOUT" docker run --rm --name probe-rd71 --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v /mnt/llm-storage:/mnt/llm-storage \
      --entrypoint /src/build/bin/llama-bench "$IMG" \
      -m "$MODEL" -ngl 999 -fa on -ctk "$CTK" -ctv "$CTV" \
      -p 0 -n "$NGEN" -d "$d" -r "$REPS" -o csv \
      > "$out" 2> "$err" &
    bench_pid=$!

    # Sample peak VRAM while the cell runs -- the capacity half of the answer.
    peak=0
    while kill -0 "$bench_pid" 2>/dev/null; do
        v=$(vram_total_mib); [ "$v" -gt "$peak" ] && peak=$v
        sleep 15
    done
    wait "$bench_pid"; rc=$?
    docker rm -f probe-rd71 >/dev/null 2>&1

    ts=$(python3 - "$out" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1], errors="replace")))
    print(f"{float(rows[-1]['avg_ts']):.2f}" if rows else "")
except Exception:
    print("")
PY
)
    if [ -n "$ts" ]; then
        printf "  depth %-8s %8s tok/s   peak VRAM %s MiB of 131040\n" "$d" "$ts" "$peak"
    else
        why="no rows"
        [ "$rc" -eq 124 ] && why="TIMEOUT after ${CELL_TIMEOUT}s"
        grep -aqi "memory access fault" "$err" && why="GPU MEMORY ACCESS FAULT"
        grep -aqiE "out of memory|failed to allocate|cannot allocate" "$err" && why="OUT OF MEMORY"
        printf "  depth %-8s FAILED -- %s   (peak VRAM %s MiB)\n" "$d" "$why" "$peak"
    fi
done

echo ""
echo "=== $(date -u +%T) round 71 done ==="
DEPTHS="$DEPTHS" python3 - <<'PY'
import csv, os
L = "/mnt/llm-storage/bench-matrix/logs"
DEPTHS = [int(d) for d in os.environ["DEPTHS"].split()]
# Round 70, same model, same f16/f16 KV.
KNOWN = [(8192, 76.41), (32768, 71.95), (65536, 69.16), (98304, 64.99), (130000, 62.89)]

def ts(d):
    p = os.path.join(L, f"rd71-d{d}.csv")
    if not os.path.isfile(p): return None
    try:
        rows = list(csv.DictReader(open(p, errors="replace")))
        return float(rows[-1]["avg_ts"]) if rows else None
    except Exception:
        return None

rows = [(d, v) for d, v in KNOWN] + [(d, ts(d)) for d in DEPTHS]
base = KNOWN[0][1]
print(f"{'depth':>9}{'tok/s':>10}{'vs d=8192':>12}   source")
print("-" * 48)
for d, v in rows:
    src = "round 70" if any(d == k for k, _ in KNOWN) else "round 71"
    if v is None:
        print(f"{d:>9}{'-':>10}{'-':>12}   {src} (failed)")
    else:
        print(f"{d:>9}{v:10.2f}{v/base:11.3f}x   {src}")

deep = [(d, ts(d)) for d in DEPTHS if ts(d)]
if deep:
    d1, v1 = deep[-1]
    print()
    print(f"AT {d1} TOKENS OF DEPTH: {v1:.2f} tok/s, {v1/base:.3f}x of the d=8192 rate.")
    print()
    print("THE COMPARISON THAT MATTERS. vLLM on this box cannot hold a 1M context")
    print("at all -- bf16 caps at 902,160 KV tokens on a 35B model. Its only")
    print("configuration that fits 1M is turboquant_4bit_nc, which measured 8.0")
    print("tok/s at 130K and extrapolates near 1 tok/s at 1M (docs/54).")
    print("llama.cpp is doing this UNCOMPRESSED, on an 80B, at f16 precision.")
else:
    print()
    print("No deep cell returned a row. A failure here is a real result: it means")
    print("the context ALLOCATES (measured: 80,267 MiB at -c 1048576) but cannot")
    print("be driven at that depth. Read the .err files before concluding which.")
PY
