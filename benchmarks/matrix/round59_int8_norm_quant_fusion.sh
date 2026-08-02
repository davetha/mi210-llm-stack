#!/usr/bin/env bash
# Round 59: turn on int8 rmsnorm+quant fusion. One flag, 6.2% of decode.
#
# THE LEAD docs/46 PREPARED AND NEVER PULLED. That document called
# `dynamic_scaled_int8_quant` (498 ms, 6.2% of the decode window) "the most
# concrete unexamined lead", then established every piece needed to attack it:
#
#   int8 fused rms+quant kernel in vLLM      does not exist
#   rms_quant_fusion.py FUSED_OPS/QUANT_OPS  FP8 and FP4 only, zero int8
#   rocm_aiter_fusion.py                     the string "int8" appears 0 times
#   BUT AITER HAS THE KERNEL:
#     vllm/_aiter_ops.py:723 _rocm_aiter_rmsnorm_fused_dynamic_quant_impl
#       assert quant_dtype in [torch.int8, FP8_DTYPE]
#       -> aiter.rmsnorm2d_fwd_with_dynamicquant
#     registered as rocm_aiter_rmsnorm_fused_dynamic_quant
#   AND THE PASS EXISTS:
#     if self.pass_config.fuse_norm_quant:
#         if rocm_aiter_ops.is_enabled():
#             self.passes += [RocmAiterRMSNormQuantFusionPass(config)]
#
# The blocker was `is_enabled` being @if_aiter_supported -> on_mi3xx(), so the
# pass was never added on CDNA2. configs/enable_aiter_master_gate_gfx90a.py
# carves that out, and round 46 VALIDATED IT SAFE: both arms kept 4 ASM objects
# and the CK GEMM, so the 1.19-1.33x prefill and 1.48x decode wins survive the
# carve-out. The feared blast radius (22 consumers, two in rocm_aiter_fa.py)
# did not materialise -- those two are capability advertisements, not dispatch.
#
# AND THEN ROUND 46 MEASURED THE OTHER FLAG. It ran
#   --compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'
# which was a null, and closed on shape grounds (the ported allreduce objects
# are allreduce_rmsnorm_N8192.co; this model's hidden size is 2048). It never
# set fuse_norm_quant. So the int8 fusion has been one flag away since.
#
# WHY IT SHOULD PAY, AND WHY THE SHAPE OF THE WIN MATTERS. docs/30 established
# this machine is ISSUE-BOUND, not FLOP-bound. Today a W8A8 model runs rmsnorm
# as one full pass over the hidden state and int8 quant as a second -- 48 layers
# x 2 norms = 96 redundant passes per token. Fusing does not make a kernel
# faster; it DELETES a kernel. That is the only category that reliably wins when
# issue rate is the scarce resource, and it is why this outranks swapping
# vLLM's quant kernel for AITER's (which would buy a marginally faster pass).
#
# An FP8 model already gets this fusion. A W8A8 model pays for it. That
# asymmetry is the whole finding.
#
# WHAT WOULD MAKE THIS A NULL FOR A BORING REASON. The pass has to actually
# MATCH. vLLM's own rms_quant_fusion.py has no int8 patterns, so the match must
# come from RocmAiterRMSNormQuantFusionPass. If the arm shows no fusion applied,
# that is the result -- reported as "the ROCm pass does not match our graph",
# not as "fusion does not help".
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs

CONTROL_IMG=${CONTROL_IMG:-vllm-mi210:aiterops}     # is_enabled NOT carved out
FUSED_IMG=${FUSED_IMG:-vllm-mi210:mastergate}       # is_enabled carved out (round 46)
for img in "$CONTROL_IMG" "$FUSED_IMG"; do
    docker image inspect "$img" >/dev/null 2>&1 || { echo "FATAL: missing $img"; exit 1; }
done
MODEL=$BASE/t35-w8a8
[ -f "$MODEL/config.json" ] || { echo "FATAL: no model at $MODEL"; exit 1; }

IN_LEN=${IN_LEN:-4096}
OUT_LEN=${OUT_LEN:-256}
CONCS=${CONCS:-"1 8 32"}
READY_TIMEOUT=${READY_TIMEOUT:-1800}

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 59: int8 rmsnorm+quant fusion ==="

cleanup() { docker rm -f rd59-ctl rd59-fused >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

wait_ready() {
    local port="$1" name="$2" t=0
    while [ $t -lt "$READY_TIMEOUT" ]; do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  $name ready (${t}s)"; return 0; }
        docker ps --format '{{.Names}}' | grep -q "^$name$" || {
            echo "  $name EXITED:"; docker logs "$name" 2>&1 | tail -25; return 1; }
        sleep 10; t=$((t+10))
    done
    echo "  FATAL: $name timeout"; return 1
}

bench_at() {  # port conc outfile
    docker run --rm --network host -v "$BASE":"$BASE" \
      --entrypoint /opt/python/bin/vllm "$FUSED_IMG" bench serve \
        --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$1" \
        --model "$MODEL" --served-model-name bench \
        --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --num-prompts $(( $2 * 8 )) --max-concurrency "$2" \
        --ignore-eos --seed 1234 2>&1 | tee "$3" \
      | grep -E "Output token throughput|Median TPOT|Median TTFT"
}

run_arm() {  # name port image extra-args...
    local name="$1" port="$2" img="$3"; shift 3
    echo ""
    echo "=== $(date -u +%T) arm $name ($img) $* ==="
    VLLM_IMAGE="$img" TP=2 VLLM_ROCM_USE_AITER_LINEAR=1 VLLM_PREFER_AITER_FA=1 NCCL_P2P_DISABLE=0 \
        "$BIN/serve_vllm_aiter.sh" "$MODEL" "$name" "$port" \
        --max-model-len 32768 "$@" >/dev/null
    wait_ready "$port" "$name" || return 1

    for c in $CONCS; do
        echo "--- $(date -u +%T) $name @ concurrency $c ---"
        bench_at "$port" "$c" "$LOGS/$name-c$c.bench"
    done

    # THE GUARD RUNS AFTER THE WORKLOAD, NOT BEFORE. AITER FA loads its ASM
    # code objects LAZILY on the first forward pass, so checking straight after
    # /health returns 200 reports LoadKernel=0 on a perfectly healthy server.
    # The first version of this script did exactly that and printed
    # "REGRESSION: AITER FA ASM is gone" for the control arm; re-checking after
    # inference showed the expected 4 objects
    # (fwd_hd128_bf16_causal_rtna_group.co, fwd_hd128_bf16_rtna_group.co).
    #
    # Round 46 established these two checks as the guard on carving out
    # is_enabled(); an arm that genuinely loses either is a REGRESSION and its
    # throughput is not comparable to the control's.
    local asm ck fuse
    asm=$(docker logs "$name" 2>&1 | grep -c "LoadKernel"); asm=${asm:-0}
    ck=$(docker logs "$name" 2>&1 | grep -c "module_gemm_a8w8"); ck=${ck:-0}
    # Did the fusion pass actually fire? This decides whether the round means
    # anything at all.
    fuse=$(docker logs "$name" 2>&1 | grep -ciE "RocmAiterRMSNormQuantFusion|rmsnorm_fused_dynamic_quant|fuse_norm_quant"); fuse=${fuse:-0}
    echo "  ASM objects: $asm   CK GEMM refs: $ck   fusion-pass mentions: $fuse"
    [ "$asm" -eq 0 ] && echo "  REGRESSION: AITER FA ASM is gone"
    [ "$ck"  -eq 0 ] && echo "  REGRESSION: CK int8 GEMM is gone"
    docker logs "$name" 2>&1 | grep -ohE "fwd_hd[0-9x]+_bf16[a-z_]*\.co" | sort -u | head -4 | sed 's/^/    /'
    docker logs "$name" 2>&1 | grep -aiE "RocmAiterRMSNormQuantFusion|replaced .* pattern|fusion.*match" | head -5 | sed 's/^/    /'
    echo "$fuse" > "$LOGS/$name.fusecount"
    docker rm -f "$name" >/dev/null 2>&1
}

run_arm rd59-ctl   8114 "$CONTROL_IMG" || { echo "CONTROL FAILED"; exit 1; }
run_arm rd59-fused 8115 "$FUSED_IMG" \
    --compilation-config '{"pass_config":{"fuse_norm_quant":true}}' \
    || echo "(fused arm failed -- see above)"

echo ""
echo "=== $(date -u +%T) round 59 done ==="
python3 - <<'PY'
import re, os
L = "/mnt/llm-storage/bench-matrix/logs"
CONCS = [int(c) for c in os.environ.get("CONCS", "1 8 32").split()]
def val(arm, c, pat):
    p = os.path.join(L, f"{arm}-c{c}.bench")
    if not os.path.isfile(p): return None
    m = re.search(pat, open(p, errors="replace").read())
    return float(m.group(1)) if m else None
TPUT = r"Output token throughput \(tok/s\):\s*([\d.]+)"
TPOT = r"Median TPOT \(ms\):\s*([\d.]+)"
print(f"{'conc':>5}{'ctl tok/s':>11}{'fused tok/s':>13}{'ratio':>9}"
      f"{'ctl TPOT':>10}{'fus TPOT':>10}{'TPOT r':>8}")
print("-" * 66)
for c in CONCS:
    a, b = val("rd59-ctl", c, TPUT), val("rd59-fused", c, TPUT)
    at, bt = val("rd59-ctl", c, TPOT), val("rd59-fused", c, TPOT)
    if a is None and b is None: continue
    r  = f"{b/a:8.3f}x" if a and b else f"{'-':>9}"
    rt = f"{bt/at:7.3f}x" if at and bt else f"{'-':>8}"
    print(f"{c:>5}"
          f"{(f'{a:11.2f}' if a else f'{chr(45):>11}')}"
          f"{(f'{b:13.2f}' if b else f'{chr(45):>13}')}{r}"
          f"{(f'{at:10.2f}' if at else f'{chr(45):>10}')}"
          f"{(f'{bt:10.2f}' if bt else f'{chr(45):>10}')}{rt}")
print()
f = os.path.join(L, "rd59-fused.fusecount")
n = open(f).read().strip() if os.path.isfile(f) else "?"
print(f"  fusion-pass mentions in the fused arm: {n}")
print()
print("THIS ROUND IS ONLY MEANINGFUL IF THE PASS FIRED. vLLM's own")
print("rms_quant_fusion.py has ZERO int8 patterns, so any match must come from")
print("RocmAiterRMSNormQuantFusionPass. A fused arm with no fusion evidence is")
print("reported as 'the ROCm pass did not match our graph' -- NOT as 'fusion")
print("does not help'. Those are different findings and only one of them is")
print("about performance.")
print()
print("docs/46 measured this rig's decode bar at 1.036x for a single pair of")
print("arms. The prize here is one deleted pass out of 96 per token, so expect")
print("a small number even on success; TPOT is the more attributable column.")
PY
