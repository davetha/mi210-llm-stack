#!/usr/bin/env bash
# What do the patches actually buy? Stock rocm/vllm against vllm-mi210, same session.
#
# WHY THIS EXISTS. docs/37 section 3 currently assembles its before/after from
# numbers measured in different rounds, on different days, on two different
# images -- the 256k pair from t35-awq-{128,256}kcfg-ctx110k.json, the AITER
# figures from benchmarks/vllm-aiter-asm-gfx90a.md, the loader figures from
# round 25. Each is sound on its own and none of them were taken side by side.
# This round measures every claim in one sitting, on one box, with one variable.
#
# THE ONE VARIABLE, AND THE TRAP IN IT. run_arm.sh routes engine `vllm` to
# serve_vllm.sh (stock rocm/vllm image, no AITER flags) and engine `vllm-aiter`
# to serve_vllm_aiter.sh (our image, AITER attention on). That is exactly the
# comparison wanted -- stock stack against patched stack -- but the two scripts
# disagreed on NCCL_P2P_DISABLE after round 31 flipped the aiter default to 0.
# Left alone, every patched arm would have quietly collected P2P's +11.2%
# prefill on top of whatever the patches did. Both are pinned to 1 below, so
# this round measures PATCHES ONLY; the P2P gain is a separate, already-measured
# axis in docs/37 section 3.4 and must not be double-counted.
#
# THE ARMS, and what each isolates:
#
#   1/2  W8A8 TP=2 32k   -- configs/enable_int8_moe_rocm.py.
#        Expect the stock arm to FAIL at load with
#          NotImplementedError: No Int8 MoE backend supports the deployment
#        because vLLM 0.23 gates INT8 MoE behind current_platform.is_cuda().
#        A FAILED arm is the result here, not an error: the format docs/37
#        recommends as best all-round does not run at all on stock.
#        (Fixed upstream in 0.26 by `or current_platform.is_rocm()`, so this
#        particular win has a shelf life -- worth recording before it expires.)
#
#   3/4  AWQ TP=1 --max-model-len 262144  -- extend_rocm_pa_256k_gfx9.py.
#        The largest single number in the repo. Stock evaluates the gfx9 gate at
#        CUDA-graph capture time against the CONFIGURED max, so 262144 > 131072
#        bakes the Triton fallback into every request. Expect ~10x decode.
#
#   5/6  AWQ TP=1 32k    -- the AITER ASM attention carve-out, alone.
#        Expect LITTLE OR NOTHING, and that is the honest result:
#        benchmarks/vllm-aiter-asm-gfx90a.md measured 1.23x only at 4096-token
#        prompts with concurrency >= 8, and 1.01-1.02x at concurrency 1, which
#        is what this harness runs. Recording a null here is the point -- docs/37
#        should not imply the ASM win applies to single-stream serving.
#
# Arms 1 and 2 use TP=2; the rest TP=1, matching how docs/28 measured each.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 32: stock vs patched, one session ==="

PATCHED=vllm-mi210:latest
STOCK=rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0

for img in "$PATCHED" "$STOCK"; do
    docker image inspect "$img" >/dev/null 2>&1 \
      || { echo "FATAL: image missing: $img"; exit 1; }
done
for m in t35-w8a8 t35-awq; do
    [ -d "$BASE/$m" ] || { echo "FATAL: model missing: $BASE/$m"; exit 1; }
done

# Pin P2P off on BOTH sides. See the header.
export NCCL_P2P_DISABLE=1
# Stock W8A8 fails in well under a minute; nothing here legitimately loads slowly.
export READY_TIMEOUT=900

run() {   # run <short-label> <engine> <tp> <model> [extra serve args...]
    local short="$1" engine="$2" tp="$3" model="$4"; shift 4
    # run_arm.sh writes results/<label>-<workload>.json, so the rd32- prefix has
    # to be part of the label it receives -- not added afterwards by the summary.
    local label="rd32-$short"
    local quant="${short%%-*}"
    echo ""
    echo "=== $(date -u +%T) arm: $label  (engine=$engine tp=$tp) ==="
    # VLLM_IMAGE is only read by serve_vllm_aiter.sh; serve_vllm.sh hardcodes
    # the stock image. Exporting it unconditionally is harmless and keeps the
    # patched arms from silently landing on serve_vllm_aiter.sh's own default
    # tag (rocm-vllm-aiter-gfx90a:latest), which is a DIFFERENT build -- the
    # mistake that made round 26 unattributable.
    TP="$tp" VLLM_IMAGE="$PATCHED" "$BIN/run_arm.sh" \
        "$label" 35B "$quant" "$engine" "$BASE/$model" "$@" 2>&1 | tail -18
    # PIPESTATUS[0], not $? -- after a pipe, $? is tail's status and every arm
    # reports rc=0 including the ones that died.
    echo "arm $label rc=${PIPESTATUS[0]}"
}

run w8a8-stock    vllm       2 t35-w8a8 --max-model-len 32768
run w8a8-patched  vllm-aiter 2 t35-w8a8 --max-model-len 32768

run awq256k-stock    vllm       1 t35-awq --max-model-len 262144
run awq256k-patched  vllm-aiter 1 t35-awq --max-model-len 262144

run awq32k-stock    vllm       1 t35-awq --max-model-len 32768
run awq32k-patched  vllm-aiter 1 t35-awq --max-model-len 32768

echo ""
echo "=== $(date -u +%T) round 32 done ==="
echo ""
python3 - <<'PY'
import glob, json, os

BASE = "/mnt/llm-storage/bench-matrix/results"
PAIRS = [
    ("int8 MoE gate",      "w8a8",    "cold16k"),
    ("256k paged attn",    "awq256k", "longctx"),
    ("AITER ASM attn",     "awq32k",  "longctx"),
    ("AITER ASM attn",     "awq32k",  "cold16k"),
]

def load(label, workload):
    f = os.path.join(BASE, f"rd32-{label}-{workload}.json")
    if os.path.isfile(f):
        with open(f) as fh:
            return json.load(fh)
    if glob.glob(os.path.join(BASE, f"rd32-{label}-FAILED.json")):
        return "FAILED"
    return None

def fmt(v, w=10):
    return f"{v:>{w}.1f}" if isinstance(v, (int, float)) else f"{str(v):>{w}}"

print(f"{'patch':<18} {'workload':<9} {'metric':<8} {'stock':>10} {'patched':>10} {'factor':>9}")
print("-" * 70)
for name, label, workload in PAIRS:
    s, p = load(f"{label}-stock", workload), load(f"{label}-patched", workload)
    for metric, key in (("prefill", "implied_prefill_tps_median"),
                        ("decode", "decode_tps_median")):
        sv = s if isinstance(s, str) or s is None else s.get(key)
        pv = p if isinstance(p, str) or p is None else p.get(key)
        if sv is None and pv is None:
            continue
        if isinstance(sv, (int, float)) and isinstance(pv, (int, float)) and sv:
            factor = f"{pv / sv:>8.2f}x"
        elif sv == "FAILED":
            factor = "  NO LOAD"
        else:
            factor = "        -"
        print(f"{name:<18} {workload:<9} {metric:<8} {fmt(sv)} {fmt(pv)} {factor}")
PY

echo ""
echo "READING THIS. A FAILED stock arm is a result, not a harness error --"
echo "check results/rd32-*-FAILED.json for the reason before treating it as one."
echo "And expect the AITER rows to show ~1.0x: this harness runs single-stream,"
echo "where benchmarks/vllm-aiter-asm-gfx90a.md already measured 1.01-1.02x."
