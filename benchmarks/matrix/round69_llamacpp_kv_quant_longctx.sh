#!/usr/bin/env bash
# Round 69: does llama.cpp's KV quantization hold up at long context?
#
# WHY THIS ROUND EXISTS -- IT FIXES A COMPARISON I ALMOST PUBLISHED. Round 65
# measured llama.cpp KV quant at -3.8% and rounds 64/67 measured vLLM turboquant
# at -49% rising to -80%. Writing "llama.cpp does it for a twentieth of the
# price" was tempting and would have been wrong, because the two numbers come
# from different regimes:
#
#   round 65   --ctx-size 32768   but the probe prompt is ~20 tokens, n_predict 96
#              -> the KV cache was essentially EMPTY. 32768 is the ALLOCATION.
#   round 64   --random-input-len 8192
#   round 67   contexts 8192 / 65536 / 130000
#
# Rounds 66/67 are precisely the demonstration that this distinction matters:
# turboquant cost 1.475x TPOT at 8192 and 5.214x at 130000. A method that looks
# cheap at zero depth can be ruinous at depth. Round 65's number is therefore
# evidence about SHORT-context serving only, and comparing it to vLLM's
# long-context collapse compares three different regimes.
#
# WHAT llama-bench -d DOES, AND WHY IT IS THE RIGHT TOOL. `-d N` prefills N
# tokens of KV and THEN measures generation, reporting tg tok/s at that depth.
# That is the same quantity as vLLM's TPOT-vs-context curve, inverted: decode
# speed with N tokens of KV resident and being re-read every step. Using the
# same shape of measurement on both stacks is what makes the comparison legal.
#
# THE ARMS, AND WHY f16 IS INCLUDED EVEN THOUGH PRODUCTION NEVER RUNS IT.
#   f16  / f16    16.0/16.0 bpw   the true uncompressed baseline
#   q8_0 / q8_0    8.5/8.5        what llama-swap-config.yaml runs today
#   q8_0 / q4_1    8.5/5.0        what launch-235b.sh / launch-mimo.sh run
#   q4_0 / q4_1    4.5/5.0        more aggressive, still mainline
# Production's "baseline" is already q8_0, so measuring only against q8_0 hides
# what quantization has ALREADY cost. f16 may fail to allocate at 130000 -- that
# failure is a capacity result and is reported, not hidden.
#
# -fa on IS MANDATORY. Round 65's first attempt omitted it and both q4_1 arms
# died; quantized V requires flash attention in llama.cpp. Production sets it.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
IMG=llama-rocm714-bench:latest
MODEL=${MODEL:-/mnt/llm-storage/coder-next-q4/Huihui-Qwen3-Coder-Next-abliterated.i1-Q4_K_M.gguf}

docker image inspect "$IMG" >/dev/null 2>&1 || { echo "FATAL: missing $IMG"; exit 1; }
[ -f "$MODEL" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

DEPTHS=${DEPTHS:-"8192,65536,130000"}
NGEN=${NGEN:-64}
REPS=${REPS:-2}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 69: llama.cpp KV quant vs context depth ==="
echo "    model:  $MODEL"
echo "    depths: $DEPTHS   n-gen: $NGEN   reps: $REPS"

cleanup() { docker rm -f rd69-bench >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

run_arm() {  # tag ctk ctv
    local tag="$1" ctk="$2" ctv="$3"
    echo ""
    echo "=== $(date -u +%T) arm $tag  -ctk $ctk -ctv $ctv ==="
    docker rm -f rd69-bench >/dev/null 2>&1
    # -p 0 so no separate prompt-processing row is generated; -d supplies the KV
    # depth and -n the generation being timed.
    docker run --rm --name rd69-bench --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v /mnt/llm-storage:/mnt/llm-storage \
      --entrypoint /src/build/bin/llama-bench "$IMG" \
      -m "$MODEL" -ngl 999 -fa on -ctk "$ctk" -ctv "$ctv" \
      -p 0 -n "$NGEN" -d "$DEPTHS" -r "$REPS" -o csv \
      > "$LOGS/rd69-$tag.csv" 2> "$LOGS/rd69-$tag.err"

    if [ ! -s "$LOGS/rd69-$tag.csv" ]; then
        echo "  ARM $tag PRODUCED NO ROWS -- actual error below."
        echo "  (Do NOT assume it is the cache type; round 65's first run died on a"
        echo "   bad model path while claiming an unsupported cache type.)"
        grep -aiE "error|unsupported|unknown|invalid|failed|out of memory|alloc" \
          "$LOGS/rd69-$tag.err" | head -6 | sed 's/^/    /'
        return 1
    fi
    python3 - "$LOGS/rd69-$tag.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], errors="replace")))
for r in rows:
    d = r.get("n_depth", "?")
    ts = float(r.get("avg_ts", 0) or 0)
    sd = float(r.get("stddev_ts", 0) or 0)
    print(f"    depth {d:>7}  {ts:8.2f} tok/s  (+/- {sd:.2f})")
PY
}

run_arm f16   f16  f16  || echo "  (f16 arm failed -- at 130000 this is plausibly a real capacity limit)"
run_arm q8q8  q8_0 q8_0 || echo "  (q8_0/q8_0 arm failed)"
run_arm q8q4  q8_0 q4_1 || echo "  (q8_0/q4_1 arm failed)"
run_arm q4q4  q4_0 q4_1 || echo "  (q4_0/q4_1 arm failed)"

echo ""
echo "=== $(date -u +%T) round 69 done ==="
python3 - <<'PY'
import csv, os
L = "/mnt/llm-storage/bench-matrix/logs"
ARMS = [("f16", "f16 / f16"), ("q8q8", "q8_0 / q8_0 (prod)"),
        ("q8q4", "q8_0 / q4_1"), ("q4q4", "q4_0 / q4_1")]

data, depths = {}, []
for tag, _ in ARMS:
    p = os.path.join(L, f"rd69-{tag}.csv")
    if not os.path.isfile(p) or os.path.getsize(p) == 0: continue
    for r in csv.DictReader(open(p, errors="replace")):
        try:
            d, ts = int(r["n_depth"]), float(r["avg_ts"])
        except (KeyError, ValueError, TypeError):
            continue
        data[(tag, d)] = ts
        if d not in depths: depths.append(d)
depths.sort()

if not depths:
    print("NO USABLE ROWS -- read the .err files in", L)
    raise SystemExit

base = "f16" if any(("f16", d) in data for d in depths) else "q8q8"
print(f"decode tok/s at KV depth (ratios vs {dict(ARMS)[base]})")
hdr = f"{'arm':<22}"
for d in depths: hdr += f"{'d='+str(d):>12}{'ratio':>9}"
print(hdr); print("-" * len(hdr))
for tag, label in ARMS:
    row = f"{label:<22}"
    for d in depths:
        x, b = data.get((tag, d)), data.get((base, d))
        row += f"{x:12.2f}" if x else f"{chr(45):>12}"
        row += f"{x/b:8.3f}x" if x and b else f"{chr(45):>9}"
    print(row)

print()
print("THE COMPARISON THIS ROUND EXISTS TO MAKE. vLLM turboquant_4bit_nc, same")
print("box, measured as TPOT ratio vs its own bf16 baseline (round 67):")
print("     8192 -> 1.475x    65536 -> 3.449x    130000 -> 5.214x")
print("i.e. decode speed fell to 68%, 29%, then 19% of baseline as depth grew.")
print("Convert the ratios above the same way: a KV quant that HOLDS its ratio")
print("as depth grows is a usable long-context option; one that decays like")
print("turboquant is not, no matter how good it looks at depth 0.")
print()
print("Round 65 measured -3.8% for q8_0/q4_1 with an essentially EMPTY cache")
print("(~20-token prompt). If the d=8192 column here reproduces roughly that,")
print("round 65 was measuring overhead; if the deeper columns hold, llama.cpp")
print("has the long-context KV compression that vLLM lacks -- which would make")
print("the production stack, not vLLM, the answer to the 1M question.")
PY
