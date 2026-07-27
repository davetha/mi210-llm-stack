#!/usr/bin/env bash
# Build a vLLM image that can actually reach AITER's gfx90a ASM kernels.
#
#   ./build_vllm_aiter_gfx90a.sh [output-tag]
#
# Stock rocm/vllm cannot use AITER on an MI210, and the failure is silent. The
# engine logs a perfectly normal-looking
#
#   Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN','TRITON_ATTN']
#
# and serves from the generic ROCm path. AITER never appears in the candidate
# list, so setting VLLM_ROCM_USE_AITER=1 changes nothing. Measured consequence
# on Qwen3-30B-A3B AWQ: 15.1k prefill at ~2,660 tok/s, about 17 TFLOP/s
# effective against a 181 TFLOP/s bf16 peak.
#
# Three separate things are broken, at three layers, and all three must be
# fixed or the stack stays on the slow path:
#
#   1. AITER ships NO gfx90a code objects at all -- only gfx942, gfx950,
#      gfx1250. repatch_gfx942_to_gfx90a.py disassembles each gfx942 kernel,
#      substitutes the CDNA2 mnemonics, RE-ASSEMBLES for gfx90a, and emits only
#      the ones that genuinely translate (242 of 1,422). Portability is proven
#      by the assembler, not assumed.
#   2. AITER's own dispatch refuses gfx90a in ~11 places, including a negated
#      arch test in mha_fwd.cu that returns -1 before kernel lookup.
#      enable_gfx90a_asm_paths.py opens exactly those.
#   3. vLLM's gate is on_mi3xx() = gfx942|gfx950.
#      enable_vllm_aiter_gfx90a.py widens ONLY the two attention checks, so
#      AITER's GEMM/MoE/FP8 paths stay unreachable on a chip with no FP8 ALU.
#
# Run this on the GPU host with the repo's configs/ directory mounted.
set -euo pipefail

TAG="${1:-rocm-vllm-aiter-gfx90a:latest}"
BASE="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"
CFG="${CFG_DIR:-/mnt/llm-storage/bench-matrix/configs}"
BUILDER="vllm-aiter-build-$$"

echo "=== base : $BASE"
echo "=== out  : $TAG"
echo "=== cfgs : $CFG"

docker rm -f "$BUILDER" >/dev/null 2>&1 || true
docker run -d --name "$BUILDER" \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
  -v "$CFG":/cfg \
  -v /var/cache/mi210-ccache:/ccache \
  -e CCACHE_DIR=/ccache -e CCACHE_MAXSIZE=100G -e CCACHE_DEPEND=1 \
  --entrypoint sleep "$BASE" infinity

trap 'docker rm -f "$BUILDER" >/dev/null 2>&1 || true' EXIT

SITE=/opt/python/lib/python3.14/site-packages

echo ""
echo "=== 0. sanity: does this image have the gfx942 kernels to translate from?"
docker exec "$BUILDER" sh -c "ls $SITE/aiter_meta/hsa/ 2>/dev/null || ls $SITE/aiter/hsa/ 2>/dev/null" \
  || { echo "FATAL: no aiter hsa/ directory in this image; nothing to repatch." >&2; exit 1; }

echo ""
echo "=== 1. translate gfx942 ASM -> gfx90a (expect TALLY OK=242 NOTPORT=1180)"
docker exec "$BUILDER" sh -c "
  set -e
  HSA=\$(ls -d $SITE/aiter_meta/hsa 2>/dev/null || ls -d $SITE/aiter/hsa)
  python3 /cfg/repatch_gfx942_to_gfx90a.py \"\$HSA/gfx942\" /tmp/gfx90a 2>&1 | tail -5
  # Install alongside the stock arches. The loader picks by arch name, so the
  # gfx942 objects are left untouched rather than replaced.
  rm -rf \"\$HSA/gfx90a\"
  cp -r /tmp/gfx90a \"\$HSA/gfx90a\"
  echo \"installed \$(find \"\$HSA/gfx90a\" -name '*.co' | wc -l) gfx90a code objects\"
"

echo ""
echo "=== 2. open AITER's gfx90a dispatch paths"
docker exec "$BUILDER" python3 /cfg/enable_gfx90a_asm_paths.py

echo ""
echo "=== 3. let vLLM route attention to AITER"
docker exec "$BUILDER" python3 /cfg/enable_vllm_aiter_gfx90a.py

echo ""
echo "=== 4. verify all three layers report patched"
docker exec "$BUILDER" python3 /cfg/enable_vllm_aiter_gfx90a.py --check
docker exec "$BUILDER" sh -c "
  HSA=\$(ls -d $SITE/aiter_meta/hsa 2>/dev/null || ls -d $SITE/aiter/hsa)
  n=\$(find \"\$HSA/gfx90a\" -name '*.co' 2>/dev/null | wc -l)
  echo \"gfx90a code objects present: \$n\"
  [ \"\$n\" -gt 0 ] || { echo 'FATAL: no gfx90a code objects installed' >&2; exit 1; }
"

echo ""
echo "=== 5. commit"
docker commit "$BUILDER" "$TAG" >/dev/null
echo "committed $TAG"
echo ""
echo "Serve with VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1 and confirm the"
echo "log does NOT say \"Overriding with ROCM_ATTN\". With AITER_LOG_LEVEL=info"
echo "the runtime prints the code object it loads for each ASM call -- that line"
echo "is the only proof the ASM path ran rather than a silent fallback."
