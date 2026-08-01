#!/usr/bin/env bash
# Round 38: three cheap suspects for the ~3x decode gap, plus the profile that
# has never been run.
#
# docs/39 re-derives batch-1 decode as ~3.1x off its bandwidth bound and scopes
# a rocprofv3 decomposition that "has never been done". Before (and alongside)
# that decomposition, three config-only suspects have zero measurements in this
# repo:
#
#   B  CLOCKS. docs/30 concludes decode is issue-bound, and issue rate scales
#      with SCLK. The DPM table on these cards is 500/800/1700 MHz and they idle
#      at 800. If power management holds a middle level during decode's small-
#      kernel steady state, that fraction of the 3x is free. Zero mentions of
#      clocks or power anywhere in this repo before this round. Every arm also
#      runs a 1 Hz sysfs sampler so the baseline arm ANSWERS what DPM does,
#      independent of whether pinning moves the numbers.
#
#   C  ASYNC SCHEDULING. --async-scheduling exists in the 0.26 images built
#      this week and appears nowhere in the repo. It overlaps CPU scheduling
#      with GPU execution -- exactly the launch/gap overhead the rocprofv3
#      decomposition quantifies.
#
#   D  CAPTURE GEOMETRY. docs/39 item 2: graph capture bakes partition geometry
#      from --max-model-len, requests run far below it, and whether the paged-
#      attention reduction early-exits per sequence is UNVERIFIED (docs/23).
#      Arm A serves at 131072 (npar_loops geometry for 512 partitions), arm D
#      at 32768 (128 partitions), and BOTH serve the same 27,852-token longctx
#      requests -- LONGCTX_TOKENS is pinned round-wide precisely so run_arm's
#      per-arm clamp cannot hand the two arms different workloads and turn the
#      geometry question into a workload difference.
#
# Then phase 2 runs the docs/39 decomposition itself: rocprofv3 kernel-trace on
# TP=1 offline decode, union-of-intervals kernel coverage vs wall clock over the
# pure-decode tail. Coverage near 1 means the residual is in-kernel (docs/30's
# claim); a large gap fraction reopens launch overhead.
#
# ONE VARIABLE PER ARM. All arms: vllm-mi210:v0.26.1rc0, 30B W8A8, TP=2, P2P
# on, AITER FA on, tuned MoE config off. Arm A is the shared baseline.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 38: decode-gap probes (clocks / async-sched / capture geometry) + rocprofv3 ==="

docker image inspect vllm-mi210:v0.26.1rc0 >/dev/null 2>&1 \
  || { echo "FATAL: missing image vllm-mi210:v0.26.1rc0"; exit 1; }

# ---------------------------------------------------------------- clocks ----
# Discover the MI210s by their DPM table rather than hardcoding card indices,
# which change across reboots. An MI210's pp_dpm_sclk lists 1700Mhz; the iGPU
# and anything else does not.
CARD_DEVS=""
for d in /sys/class/drm/card*/device; do
    [ -f "$d/pp_dpm_sclk" ] && grep -q "1700Mhz" "$d/pp_dpm_sclk" && CARD_DEVS="$CARD_DEVS $d"
done
n_cards=$(echo $CARD_DEVS | wc -w)
if [ "$n_cards" -ne 2 ]; then
    echo "FATAL: expected 2 MI210 sysfs nodes, found $n_cards ($CARD_DEVS)"
    exit 1
fi
echo "MI210 sysfs nodes:$CARD_DEVS"

set_perf() {  # high | auto -- applied to both cards, verified by readback
    local want="$1" got
    for d in $CARD_DEVS; do
        echo "$want" | sudo -n tee "$d/power_dpm_force_performance_level" >/dev/null || return 1
        got=$(cat "$d/power_dpm_force_performance_level")
        [ "$got" = "$want" ] || { echo "  perf level readback: wanted $want got $got on $d"; return 1; }
    done
    return 0
}
restore_clocks() { set_perf auto || echo "WARNING: could not restore perf level to auto"; }

SAMPLER_PID=""
start_sampler() {  # $1 = outfile. Lines: epoch card sclk_mhz power_uW
    (
        while :; do
            ts=$(date +%s)
            for d in $CARD_DEVS; do
                mhz=$(awk '/\*/{gsub(/Mhz/,"",$2); print $2}' "$d/pp_dpm_sclk" 2>/dev/null | head -1)
                pw=$(cat "$d"/hwmon/hwmon*/power1_average 2>/dev/null | head -1)
                echo "$ts $(basename "$(dirname "$d")") ${mhz:-na} ${pw:-na}"
            done
            sleep 1
        done
    ) >> "$1" &
    SAMPLER_PID=$!
}
stop_sampler() {
    [ -n "$SAMPLER_PID" ] && { kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; }
    SAMPLER_PID=""
}
trap 'stop_sampler; restore_clocks' EXIT

# ------------------------------------------------------------- constants ----
export VLLM_IMAGE=vllm-mi210:v0.26.1rc0
export NCCL_P2P_DISABLE=0          # round 31: +11.2% prefill, now the default
export VLLM_TUNED_CONFIG_FOLDER=   # round 34: tuned config is 0.79x, keep out
export VLLM_PREFER_AITER_FA=1      # round 37: 1.19-1.33x, and asserted below
export TP=2
export READY_TIMEOUT=900
# Identical requests for every arm regardless of --max-model-len; see header.
export LONGCTX_TOKENS=27852

run_one() {  # label clocks(auto|high) extra-serve-args...
    local label="$1" clocks="$2"; shift 2
    echo ""
    echo "=== $(date -u +%T) arm: $label (clocks=$clocks) ==="
    if [ "$clocks" = "high" ]; then
        if ! set_perf high; then
            # An arm that silently ran at auto would be recorded as "pinning
            # does nothing". Refuse to produce that number.
            echo "ARM SKIPPED: $label -- could not pin perf level high"
            restore_clocks
            return 1
        fi
    fi
    start_sampler "$LOGS/$label.clocks"
    "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" "$@" 2>&1 | tail -14
    local rc=${PIPESTATUS[0]}
    stop_sampler
    [ "$clocks" = "high" ] && restore_clocks
    # grep -c prints its count even when it exits 1 on zero matches, so an
    # `|| echo 0` here would emit "0" TWICE for a match-free log. Default only
    # the missing-file case, where stdout really is empty.
    local n
    n=$(grep -c "LoadKernel" "$LOGS/$label.serverlog" 2>/dev/null)
    n=${n:-0}
    echo "arm $label rc=$rc  ASM code objects loaded: $n"
    [ "$n" -gt 0 ] || echo "  WARNING: no ASM loaded -- not comparable to an arm that had it"
}

run_one rd38-base    auto --max-model-len 131072
run_one rd38-clkhigh high --max-model-len 131072
run_one rd38-async   auto --max-model-len 131072 --async-scheduling
run_one rd38-len32k  auto --max-model-len 32768

# ---------------------------------------------- phase 2: rocprofv3, TP=1 ----
# The docs/39 decomposition. TP=1 removes collectives so kernel coverage vs
# wall is launch-vs-in-kernel with no NCCL term. In-process engine
# (VLLM_ENABLE_V1_MULTIPROCESSING=0) so one process produces one trace.
# Best-effort: a profile failure is reported loudly and does not eat the four
# arms above.
echo ""
echo "=== $(date -u +%T) phase 2: rocprofv3 decode decomposition (TP=1) ==="
PROF=$LOGS/rd38-prof
rm -rf "$PROF"; mkdir -p "$PROF"
docker run --rm --name probe-rd38-prof \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 32G \
  -v /mnt/llm-storage:/models -v "$PROF":/prof \
  -e HSA_NO_SCRATCH_RECLAIM=1 -e GPU_MAX_HW_QUEUES=4 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MHA=1 \
  -e VLLM_PREFER_AITER_FA=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  --entrypoint bash vllm-mi210:v0.26.1rc0 -c '
set -e
SDK=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel
cat > /tmp/decode_trace.py <<'"'"'PY'"'"'
import time
from vllm import LLM, SamplingParams

llm = LLM(model="/models/bench-matrix/t35-w8a8", max_model_len=32768,
          gpu_memory_utilization=0.90, tensor_parallel_size=1,
          enable_prefix_caching=False, seed=1234)
tok = llm.get_tokenizer()
ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 1400)[:8000]
prompt = tok.decode(ids)
# 1024, not 512: this is an A3B MoE at TP=1, decode may run 60-90 tok/s, and
# the analysis below takes an 8 s pure-decode tail. 512 tokens could finish in
# ~7 s and let prefill leak into the window.
sp = SamplingParams(max_tokens=1024, temperature=0.0, ignore_eos=True)
t0 = time.time()
out = llm.generate([prompt], sp)
t1 = time.time()
n = len(out[0].outputs[0].token_ids)
print(f"PROFILE_MARK decode_tokens={n} wall_s={t1-t0:.2f}")
PY
"$SDK/bin/rocprofv3" --kernel-trace --hip-trace --output-format csv \
    -d /prof -o rd38 -- python3 /tmp/decode_trace.py
' 2>&1 | tail -8
# PIPESTATUS, not $?: $? is tail's exit status, which is 0 even when the
# profile died. run_arm.sh documents the identical trap.
prof_rc=${PIPESTATUS[0]}
[ "$prof_rc" -eq 0 ] || echo "PROFILE FAILED (rc=$prof_rc) -- phase 2 has no result; the four arms above stand on their own"

# ----------------------------------------------------------------- summary --
echo ""
echo "=== $(date -u +%T) round 38 done ==="
python3 - <<'PY'
import csv, glob, json, os
BASE = "/mnt/llm-storage/bench-matrix"
R, L = f"{BASE}/results", f"{BASE}/logs"

arms = ["rd38-base", "rd38-clkhigh", "rd38-async", "rd38-len32k"]
rows = [("cold16k", "implied_prefill_tps_median", "prefill"),
        ("cold16k", "ttft_s_median", "ttft"),
        ("longctx", "implied_prefill_tps_median", "prefill"),
        ("longctx", "decode_tps_median", "decode"),
        ("longctx", "ttft_s_median", "ttft")]

print(f"{'workload':<9} {'metric':<8}" + "".join(f"{a[5:]:>10}" for a in arms))
print("-" * 57)
for wl, key, name in rows:
    vals = []
    for a in arms:
        f = os.path.join(R, f"{a}-{wl}.json")
        vals.append(json.load(open(f)).get(key) if os.path.isfile(f) else None)
    if all(v is None for v in vals):
        continue
    cells = "".join(f"{v:10.2f}" if isinstance(v, (int, float)) else f"{'-':>10}" for v in vals)
    print(f"{wl:<9} {name:<8}{cells}")

print()
print("factors vs base (longctx decode / cold16k prefill):")
def get(a, wl, key):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(key) if os.path.isfile(f) else None
for a in arms[1:]:
    d0, d1 = get("rd38-base", "longctx", "decode_tps_median"), get(a, "longctx", "decode_tps_median")
    p0, p1 = get("rd38-base", "cold16k", "implied_prefill_tps_median"), get(a, "cold16k", "implied_prefill_tps_median")
    dd = f"{d1/d0:.3f}x" if d0 and d1 else "-"
    pp = f"{p1/p0:.3f}x" if p0 and p1 else "-"
    print(f"  {a[5:]:<8} decode {dd:>8}   prefill {pp:>8}")

print()
print("DPM behaviour per arm (1 Hz sysfs samples, both cards pooled):")
for a in arms:
    f = os.path.join(L, f"{a}.clocks")
    if not os.path.isfile(f):
        print(f"  {a[5:]:<8} no clock log"); continue
    mhz, pw = [], []
    for line in open(f):
        p = line.split()
        if len(p) == 4:
            if p[2].isdigit(): mhz.append(int(p[2]))
            if p[3].isdigit(): pw.append(int(p[3]))
    if not mhz:
        print(f"  {a[5:]:<8} no parseable samples"); continue
    from collections import Counter
    c = Counter(mhz); tot = len(mhz)
    dist = "  ".join(f"{k}MHz:{100*v//tot}%" for k, v in sorted(c.items()))
    pmed = sorted(pw)[len(pw)//2] / 1e6 if pw else float("nan")
    print(f"  {a[5:]:<8} {dist}   median power {pmed:.0f} W  ({tot} samples)")

print()
print("rocprofv3 decode decomposition (union kernel coverage over pure-decode tail):")
traces = sorted(glob.glob(f"{L}/rd38-prof/**/*kernel_trace*.csv", recursive=True))
if not traces:
    print("  NO TRACE FILES -- profile phase failed; see console above")
else:
    iv, names = [], {}
    for t in traces:
        with open(t) as fh:
            rd = csv.DictReader(fh)
            cols = rd.fieldnames or []
            sc = next((c for c in cols if "Start_Timestamp" in c), None)
            ec = next((c for c in cols if "End_Timestamp" in c), None)
            kc = next((c for c in cols if "Kernel_Name" in c), None)
            if not (sc and ec):
                print(f"  {t}: unrecognised columns {cols[:6]}..."); continue
            for row in rd:
                try:
                    s, e = int(row[sc]), int(row[ec])
                except (KeyError, ValueError):
                    continue
                iv.append((s, e))
                k = row.get(kc, "?") if kc else "?"
                names[k] = names.get(k, 0) + (e - s)
    if not iv:
        print("  trace files present but no kernel records parsed")
    else:
        iv.sort()
        t_end = max(e for _, e in iv)
        W = 8_000_000_000  # 8 s tail: prefill ~2 s, decode ~15 s, so this is pure decode
        w0 = t_end - W
        clipped = [(max(s, w0), e) for s, e in iv if e > w0]
        merged, cs, ce = [], None, None
        for s, e in sorted(clipped):
            if cs is None: cs, ce = s, e
            elif s <= ce: ce = max(ce, e)
            else: merged.append((cs, ce)); cs, ce = s, e
        if cs is not None: merged.append((cs, ce))
        busy = sum(e - s for s, e in merged)
        nk = len(clipped)
        print(f"  window 8.0 s, kernels {nk}, coverage {100*busy/W:.1f}%  "
              f"(gap fraction {100*(1-busy/W):.1f}% = launch/CPU-bound share)")
        print(f"  kernel launches/s in window: {nk/8.0:,.0f}")
        print("  top kernels by total time (whole trace):")
        for k, ns in sorted(names.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {ns/1e6:10.1f} ms  {k[:90]}")
PY

echo ""
echo "READING THIS. (1) clkhigh vs base: any decode gain is DPM downclocking,"
echo "and the DPM table above shows what auto actually did. (2) async vs base:"
echo "scheduling overlap. (3) len32k vs base: SAME 27,852-token requests, only"
echo "the captured --max-model-len geometry differs -- a len32k win means part"
echo "of the decode gap is self-inflicted capture config (docs/39 item 2)."
echo "(4) coverage near 100% puts the residual in-kernel, corroborating"
echo "docs/30; a large gap fraction reopens launch overhead and makes"
echo "--async-scheduling the interesting arm."
