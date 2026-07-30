#!/usr/bin/env bash
# Turn on the two AITER components this stack has always disabled.
#
# WHAT WAS ASSUMED. serve_vllm_aiter.sh has always passed
# VLLM_ROCM_USE_AITER_LINEAR=0 and VLLM_ROCM_USE_AITER_MOE=0, on the strength of
# a research note in benchmarks/matrix/research/ that marks both
# "NO-EFFECT -- fails gate", citing envs.py:130 and :132.
#
# THAT CITATION IS THE DECLARATION, NOT THE GATE. envs.py only defines the
# variables. Read where they are actually consumed in the installed vLLM 0.23:
#
#   _aiter_ops.py refresh_env_variables():
#       cls._LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
#       cls._FMOE_ENABLED   = envs.VLLM_ROCM_USE_AITER_MOE
#
# assigned straight from the environment with NO architecture check. And the
# unquantized MoE oracle is decisive in both directions:
#
#   fused_moe/oracle/unquantized.py:277
#       if is_set(USE_AITER) or is_set(USE_AITER_MOE):
#           if not USE_AITER or not USE_AITER_MOE:
#               AVAILABLE_BACKENDS.remove(UnquantizedMoeBackend.AITER)   <-- us
#           else:
#               backend = UnquantizedMoeBackend.AITER; return ...        <-- override
#
# So setting MOE=0 ACTIVELY REMOVES the AITER MoE backend. We have been switching
# it off by hand, not watching a gate reject it. This is the fourth time in this
# project that "unreachable on gfx90a" turned out to be a misread gate --
# fmha_v3_fwd (80/80 exact), pa_fwd_asm (48/48 exact), and VLLM_PREFER_AITER_FA
# (which no longer exists) were the first three.
#
# WHY THE BF16 ARM IS THE POINT. That oracle is unquantized.py -- bf16/fp16 MoE
# only. Which is exactly what the 8 gfx90a fmoe ASM objects serve: every one is
# noquantFp16 or noquantBf16 (docs/33). fmoe measures ~1,129 MACs per issued
# instruction, the highest of anything installed on this box, against 39-99 for
# the compiled paths. A bf16 MoE model with AITER_MOE=1 is the only configuration
# that can reach it.
#
# THE BF16 ARM IS EXPENSIVE AND RUNS LAST. Qwen3-30B bf16 loads in ~12,366 s
# (docs/28) -- 3.4 hours, the hsakmt_ioctl path. The W8A8 arms load in 60 s, so
# they run first and cheaply establish whether the flags do anything at all on
# the int8 path, where fmoe ASM cannot apply.
#
# EXPECT THE W8A8 ARMS TO BE FLAT OR WORSE. No int8 fmoe ASM exists for gfx90a --
# docs/16 lists fmoe_2stages (INT8) as patched, but the installed gfx90a tree has
# no fmoe_2stages directory at all (docs/33). So AITER MoE on W8A8 would fall to a
# CK or Triton fused_moe, which is not obviously better than stock. A loss there
# is still a result: it bounds the flags to the bf16 tier.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 19 (AITER component matrix) ==="

# ARM_ENV is read inside run(), not written as `VAR=val run ...`: a bash
# assignment prefixing a FUNCTION call does not reliably export to the
# grandchild, and serve_vllm_aiter.sh reads these from its environment. Losing
# them would run the arm with the flags absent and report it as though they were
# applied -- the exact failure this round exists to correct.
run() {  # run <label> <model-dir> <tier> <quant> <extra serve args...>
    local label="$1" model="$2" tier="$3" quant="$4"; shift 4
    echo "--- $label  [env: ${ARM_ENV:-none}] ---"
    VLLM_EXTRA_ENV="${ARM_ENV:-}" \
    ARM_TIMEOUT=7200 READY_TIMEOUT="${RT:-2400}" \
        "$BIN/run_arm.sh" "$label" "$tier" "$quant" vllm-aiter "$model" \
        --tensor-parallel-size 2 \
        --max-model-len 131072 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        "$@" \
        || echo "!! $label failed (recorded)"
}

M35W=$BASE/t35-w8a8

# --- A. W8A8 int8 path: does either flag move anything? --------------------
# Baseline reproduces the shipped configuration (both flags 0) on this harness,
# rather than quoting the historical 7,278 t/s, so the comparison is matched.
ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=0"
run t35w8a8-aiter-base "$M35W" 35B w8a8

ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_USE_AITER_MOE=0"
run t35w8a8-aiter-linear "$M35W" 35B w8a8

ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=1"
run t35w8a8-aiter-moe "$M35W" 35B w8a8

ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_USE_AITER_MOE=1"
run t35w8a8-aiter-all "$M35W" 35B w8a8

# --- B. bf16: the only path that can reach fmoe ASM ------------------------
# READY_TIMEOUT is raised to 5 hours because this loader legitimately takes 3.4.
# --safetensors-load-strategy=prefetch is mandatory on btrfs (docs/25) and is not
# automatic for local filesystems.
# WHETHER fmoe ASM ACTUALLY RAN IS THE RESULT. Throughput alone cannot tell a
# 1,129-MACs/ins ASM kernel from the CK/Triton fused_moe it falls back to, and
# this project has published that exact mistake twice (docs/14, docs/16). The
# per-arm evidence file run_arm.sh now writes is the discriminator; read it and
# say so out loud, so the round log carries the verdict next to the number.
fmoe_verdict() {
    local label="$1"
    python3 - "$BASE/results/$label-asm.json" "$label" <<'VERDICT' || true
import json, sys
path, label = sys.argv[1:3]
try:
    d = json.load(open(path))
except Exception:
    print("  VERDICT %s: no ASM evidence file -- attribution UNKNOWN" % label)
    sys.exit(0)
if not d.get("serverlog_readable", True):
    print("  VERDICT %s: serverlog unreadable -- attribution UNKNOWN" % label)
elif d.get("families", {}).get("fmoe"):
    print("  VERDICT %s: fmoe ASM LOADED (%d objects) -- attributable"
          % (label, d["families"]["fmoe"]))
elif d.get("any_asm"):
    print("  VERDICT %s: ASM loaded but NO fmoe (%s) -- MoE ran the fallback"
          % (label, ", ".join(sorted(d.get("families", {})))))
else:
    print("  VERDICT %s: no ASM loaded at all -- MoE ran the fallback" % label)
VERDICT
}

M35B=$BASE/t35-bf16
if [ -d "$M35B" ]; then
    ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=0 -e VLLM_ROCM_USE_AITER_MOE=0"
    RT=18000 run t35bf16-aiter-base "$M35B" 35B bf16 \
        --safetensors-load-strategy=prefetch
    fmoe_verdict t35bf16-aiter-base

    ARM_ENV="-e VLLM_ROCM_USE_AITER_LINEAR=1 -e VLLM_ROCM_USE_AITER_MOE=1"
    RT=18000 run t35bf16-aiter-all "$M35B" 35B bf16 \
        --safetensors-load-strategy=prefetch
    fmoe_verdict t35bf16-aiter-all
else
    echo "!! $M35B absent -- skipping the bf16 arms, which are the fmoe test"
fi

echo "=== $(date -u +%T) round 19 done ==="
echo "--- AITER component matrix ---"
for l in t35w8a8-aiter-base t35w8a8-aiter-linear t35w8a8-aiter-moe t35w8a8-aiter-all \
         t35bf16-aiter-base t35bf16-aiter-all; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-24s %-8s prefill=%9.1f decode=%7.2f' % (sys.argv[2], sys.argv[3],
    d.get('implied_prefill_tps_median') or 0, d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
    f="results/$l-FAILED.json"
    [ -f "$f" ] && python3 -c "
import json,sys; d=json.load(open(sys.argv[1]))
print('  %-24s FAILED: %s' % (sys.argv[2], d.get('reason')))" "$f" "$l"
done
echo
echo "VERIFY fmoe ASM actually loaded on the bf16-all arm -- the .co line is the"
echo "proof, not the backend-selection line:"
echo "  grep -i 'LoadKernel.*fmoe' logs/t35bf16-aiter-all.serverlog"
