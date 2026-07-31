#!/usr/bin/env bash
# Does the tuned MI210 fused_moe config actually make anything faster?
#
# Round 33 produced a PARTIAL config: batch sizes 1, 2 and 4 only. Everything
# from 8 up hit the 45-minute per-size cap (exit 124), six times in a row, after
# ~7.5 GPU-hours. That is not a failure to work around here -- it is the input
# to this round, and it shapes what this round can and cannot show.
#
# WHAT THE PARTIAL CONFIG BUYS US, and why it is a good experiment anyway.
# get_moe_configs returns a dict keyed by M and try_get_optimal_moe_config
# selects the nearest entry, so only shapes near M=1..4 are affected. This
# harness decodes single-stream, i.e. M=1, and prefills chunked at -ub 2048,
# i.e. M~2048 -- which falls back to the same heuristic in BOTH arms.
#
# So the round has a control built into it:
#
#   decode   -- M=1, tuned in arm B          -> may move
#   prefill  -- M~2048, untuned in both arms -> must NOT move
#
# A result where both move is not a win, it is evidence of drift, and should be
# rejected rather than reported.
#
# WHAT THE TUNED VALUES ALREADY SUGGEST. All three entries chose
# matrix_instr_nonkdim=16 -- the same 16x16x16 MFMA tile docs/33 identified as
# what decode "gets stuck with" at 191-196 MACs/instruction, against 485-497 for
# the 32x32x8 tile larger shapes select. The tuner varied BLOCK_SIZE_N (32, 64,
# 128) and num_warps (2, 4, 8) instead. That is the expected answer on
# reflection: at M=1 a 32-row tile wastes 31/32 of the M dimension. So docs/33's
# "decode is on the wrong side of a 2.5x spread" may describe an unreachable
# ceiling rather than available headroom, and a null result here would be
# evidence FOR that reading rather than a failed experiment.
#
# PROVING THE CONFIG WAS LOADED. get_moe_configs logs exactly one of:
#
#   Using configuration from <path> for MoE layer.        <- tuned
#   Using default MoE config. Performance might be sub-optimal!   <- not tuned
#
# Both arms are asserted against those lines. install_moe_config.sh names the
# hazard precisely: without the check, "the tuned arm would have reported 'no
# improvement' from a config that was never loaded -- a false negative
# indistinguishable from a real one."
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
CFG=$BASE/moe-configs-mi210
WANT="E=128,N=384,device_name=AMD_Instinct_MI210.json"
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 34: tuned fused_moe config A/B ==="

[ -f "$CFG/$WANT" ] || { echo "FATAL: no tuned config at $CFG/$WANT"; exit 1; }
python3 - <<PY || exit 1
import json, sys
d = json.load(open("$CFG/$WANT"))
ks = sorted((k for k in d if k.isdigit()), key=int)
print(f"  tuned batch sizes: {ks}")
if not ks:
    print("FATAL: config has no batch-size entries", file=sys.stderr); sys.exit(1)
PY

# Pinned on both sides. Round 31 measured P2P at +11.2% prefill and round 32
# nearly attributed it to the patches; this round must not repeat that.
export NCCL_P2P_DISABLE=1
export TP=2
export VLLM_IMAGE=vllm-mi210:latest
export READY_TIMEOUT=900

for arm in stock tuned; do
    label="rd34-w8a8-$arm"
    echo ""
    echo "=== $(date -u +%T) arm: $label ==="
    if [ "$arm" = "tuned" ]; then
        # The DIRECTORY, not the file: get_moe_configs does
        # os.path.join(VLLM_TUNED_CONFIG_FOLDER, json_file_name).
        # /mnt/llm-storage is mounted at /models inside the container.
        export VLLM_TUNED_CONFIG_FOLDER=/models/bench-matrix/moe-configs-mi210
    else
        export VLLM_TUNED_CONFIG_FOLDER=
    fi
    "$BIN/run_arm.sh" "$label" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 32768 2>&1 | tail -16
    echo "arm $label rc=${PIPESTATUS[0]}"

    # Assert the arm loaded what it was supposed to, BEFORE any number is read.
    log="$BASE/logs/$label.serverlog"
    if [ "$arm" = "tuned" ]; then
        if grep -q "Using configuration from .*MI210.*for MoE layer" "$log" 2>/dev/null; then
            echo "  VERIFIED: tuned config loaded"
        else
            echo "  FATAL: tuned arm did NOT load the config -- result is meaningless."
            grep -m1 "Using default MoE config" "$log" 2>/dev/null && echo "  (it used the default)"
            exit 1
        fi
    else
        if grep -q "Using default MoE config" "$log" 2>/dev/null; then
            echo "  VERIFIED: control used the default config"
        else
            echo "  WARNING: control did not log the default-config line; check"
            grep -m1 "Using configuration from" "$log" 2>/dev/null
        fi
    fi
done

echo ""
echo "=== $(date -u +%T) round 34 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
print(f"{'workload':<9} {'metric':<8} {'stock':>10} {'tuned':>10} {'factor':>9}")
print("-" * 50)
for w in ("cold16k", "longctx"):
    for metric, key in (("prefill", "implied_prefill_tps_median"),
                        ("decode", "decode_tps_median")):
        vals = []
        for arm in ("stock", "tuned"):
            f = os.path.join(R, f"rd34-w8a8-{arm}-{w}.json")
            vals.append(json.load(open(f)).get(key) if os.path.isfile(f) else None)
        s, t = vals
        if s is None and t is None:
            continue
        fac = f"{t / s:>8.3f}x" if isinstance(s, (int, float)) and isinstance(t, (int, float)) and s else "        -"
        fs = f"{s:10.2f}" if isinstance(s, (int, float)) else f"{str(s):>10}"
        ft = f"{t:10.2f}" if isinstance(t, (int, float)) else f"{str(t):>10}"
        print(f"{w:<9} {metric:<8} {fs} {ft} {fac}")
PY

echo ""
echo "READING THIS. Only M=1,2,4 are tuned, so ONLY decode can legitimately"
echo "move. Prefill runs at M~2048 and falls back to the same heuristic in both"
echo "arms -- it is the control. If prefill moves too, reject the run as drift"
echo "rather than reporting a win."
