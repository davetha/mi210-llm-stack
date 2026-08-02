#!/usr/bin/env bash
# Round 70: llama.cpp KV quant vs depth, one invocation per cell.
#
# WHY ROUND 69 PRODUCED NOTHING, AND WHAT IT COST. Round 69 asked llama-bench
# for all three depths in ONE invocation (-d 8192,65536,130000). All four arms
# died with:
#     Memory access fault by GPU node-1 ... GPU memory access fault.
# and because llama-bench writes its CSV at the END, the crash at the largest
# depth destroyed the completed rows for the smaller ones. Four arms ran for
# twenty-three minutes and produced zero usable numbers.
#
# The failure is NOT about the cache type. q4_0/q4_1 uses far less KV than
# f16/f16 and faulted identically, so it is neither KV type nor KV capacity. A
# direct probe confirms the mechanism:
#     -ctk q8_0 -ctv q8_0 -d 8192 -n 32 -r 1   ->   71.28 tok/s, clean exit
# So `-d` works and the model loads; something at a LARGER depth faults.
#
# THE FIX IS ISOLATION, NOT A SMALLER SWEEP. Every (arm, depth) pair runs as its
# own process writing its own CSV. A fault in one cell costs exactly that cell.
# This is the same lesson as round 66 -- a harness that loses good data when one
# point fails will eventually lose the data you cared about.
#
# THE DEPTH LADDER IS ALSO THE ANSWER TO "HOW FAR CAN llama.cpp GO". Because
# each cell is independent, the ladder finds the depth ceiling for free: the
# highest depth that returns a row is where this stack stops working, and that
# is a capacity result worth having on its own. Round 69 could not report it.
#
# WHAT THIS ROUND IS FOR. docs/54 records vLLM's turboquant decaying from 1.475x
# TPOT at 8192 to 5.214x at 130000, and deliberately leaves the llama.cpp
# comparison OPEN, because round 65's -3.8% was measured with a ~20-token prompt
# against an essentially empty cache. This round closes it by measuring
# llama.cpp the same way vLLM was measured: decode speed with N tokens of KV
# resident. If the KV-quant ratios HOLD as depth grows, the production stack has
# the long-context KV compression vLLM lacks. If they decay similarly, neither
# stack compresses KV usefully at depth.
#
# EARLY EVIDENCE THAT DEPTH MATTERS HERE TOO: round 65 measured q8_0/q8_0 at
# 76.81 tok/s with an empty cache; the probe above measured 71.28 tok/s at depth
# 8192. That is 7.2% lost to depth alone, before any comparison between KV
# types -- which is exactly why round 65's number could not be compared to
# vLLM's.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=llama-rocm714-bench:latest
MODEL=${MODEL:-/mnt/llm-storage/coder-next-q4/Huihui-Qwen3-Coder-Next-abliterated.i1-Q4_K_M.gguf}

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

DEPTHS=${DEPTHS:-"8192 32768 65536 98304 130000"}
NGEN=${NGEN:-32}
REPS=${REPS:-1}
CELL_TIMEOUT=${CELL_TIMEOUT:-1500}

# tag:ctk:ctv -- q8q8 first so the production baseline exists before anything is
# compared against it.
ARMS=${ARMS:-"q8q8:q8_0:q8_0 q8q4:q8_0:q4_1 q4q4:q4_0:q4_1 f16:f16:f16"}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 70: llama.cpp KV quant x depth, isolated cells ==="
echo "    depths: $DEPTHS   n-gen: $NGEN   reps: $REPS"

cleanup() { docker rm -f probe-rd70 >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

cell() {  # tag ctk ctv depth
    local tag="$1" ctk="$2" ctv="$3" d="$4"
    local out="$LOGS/rd70-$tag-d$d.csv" err="$LOGS/rd70-$tag-d$d.err"
    docker rm -f probe-rd70 >/dev/null 2>&1
    timeout "$CELL_TIMEOUT" docker run --rm --name probe-rd70 --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v /mnt/llm-storage:/mnt/llm-storage \
      --entrypoint /src/build/bin/llama-bench "$IMG" \
      -m "$MODEL" -ngl 999 -fa on -ctk "$ctk" -ctv "$ctv" \
      -p 0 -n "$NGEN" -d "$d" -r "$REPS" -o csv \
      > "$out" 2> "$err"
    local rc=$?
    docker rm -f probe-rd70 >/dev/null 2>&1

    local ts
    ts=$(python3 - "$out" <<'PY'
import csv, sys
try:
    rows = [r for r in csv.DictReader(open(sys.argv[1], errors="replace"))]
    print(f"{float(rows[-1]['avg_ts']):.2f}" if rows else "")
except Exception:
    print("")
PY
)
    if [ -n "$ts" ]; then
        printf "  %-6s d=%-7s %8s tok/s\n" "$tag" "$d" "$ts"
        return 0
    fi
    # Distinguish the three ways a cell can produce nothing. Round 69 reported
    # all of them as the same thing.
    local why="no rows"
    [ "$rc" -eq 124 ] && why="TIMEOUT after ${CELL_TIMEOUT}s"
    grep -aqi "memory access fault" "$err" && why="GPU MEMORY ACCESS FAULT"
    grep -aqiE "out of memory|failed to allocate|cannot allocate" "$err" && why="OUT OF MEMORY"
    printf "  %-6s d=%-7s FAILED -- %s\n" "$tag" "$d" "$why"
    return 1
}

for spec in $ARMS; do
    tag=${spec%%:*}; rest=${spec#*:}; ctk=${rest%%:*}; ctv=${rest#*:}
    echo ""
    echo "=== $(date -u +%T) arm $tag  -ctk $ctk -ctv $ctv ==="
    for d in $DEPTHS; do
        cell "$tag" "$ctk" "$ctv" "$d" || true
    done
done

echo ""
echo "=== $(date -u +%T) round 70 done ==="
DEPTHS="$DEPTHS" ARMS="$ARMS" python3 - <<'PY'
import csv, os
L = "/mnt/llm-storage/bench-matrix/logs"
DEPTHS = [int(d) for d in os.environ["DEPTHS"].split()]
SPECS = [s.split(":") for s in os.environ["ARMS"].split()]
LABEL = {"q8q8": "q8_0/q8_0 (prod)", "q8q4": "q8_0/q4_1",
         "q4q4": "q4_0/q4_1", "f16": "f16/f16"}

def ts(tag, d):
    p = os.path.join(L, f"rd70-{tag}-d{d}.csv")
    if not os.path.isfile(p): return None
    try:
        rows = list(csv.DictReader(open(p, errors="replace")))
        return float(rows[-1]["avg_ts"]) if rows else None
    except Exception:
        return None

base = SPECS[0][0]
print(f"decode tok/s at KV depth   (ratio vs {LABEL.get(base, base)} at the same depth)")
hdr = f"{'arm':<18}"
for d in DEPTHS: hdr += f"{'d='+str(d):>13}{'ratio':>8}"
print(hdr); print("-" * len(hdr))
for tag, _, _ in SPECS:
    row = f"{LABEL.get(tag, tag):<18}"
    for d in DEPTHS:
        x, b = ts(tag, d), ts(base, d)
        row += f"{x:13.2f}" if x else f"{chr(45):>13}"
        row += f"{x/b:7.3f}x" if x and b else f"{chr(45):>8}"
    print(row)

# The decay of the baseline itself is a result: it is what depth costs before
# any quantization question is asked.
b0 = ts(base, DEPTHS[0])
if b0:
    print()
    print(f"{LABEL.get(base, base)} vs its own shallowest point (d={DEPTHS[0]}):")
    for d in DEPTHS:
        x = ts(base, d)
        if x: print(f"  d={d:<7} {x:8.2f} tok/s   {x/b0:.3f}x")
    print("  (round 65 measured 76.81 tok/s for this config with an ESSENTIALLY")
    print("   EMPTY cache -- the gap to d=8192 is what depth alone costs.)")

print()
print("THE COMPARISON. vLLM turboquant_4bit_nc vs its own bf16 baseline, TPOT")
print("ratio, same box (docs/54, round 67):")
print("     d=8192 -> 1.475x     d=65536 -> 3.449x     d=130000 -> 5.214x")
print("As decode SPEED that is 0.678x, 0.290x, 0.192x of baseline.")
print("Compare the ratio columns above directly against those three numbers.")
print("A KV quant whose ratio HOLDS as depth grows is a usable long-context")
print("option and would make the production stack the answer to the 1M")
print("question; one that decays like turboquant is not.")
print()
print("BLANK CELLS ARE DATA. Each cell ran as its own process, so a blank is a")
print("real failure at that depth and not collateral from another cell. The")
print("deepest non-blank cell per row is that config's working ceiling.")
PY
