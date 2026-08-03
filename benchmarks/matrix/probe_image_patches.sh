#!/usr/bin/env bash
# Report which gfx90a patches are present in each vLLM image.
#
# WHY. Round 73 ran three arms on the wrong image and reported a ratio it had
# not earned. The round's own header recorded that the CK GEMM carve-outs had
# been verified in vllm-mi210:gdnpolicy -- an image that was never serving,
# because serve_vllm_aiter.sh reads VLLM_IMAGE for the server while the round
# had only set the benchmark CLIENT image. There was no way to ask an image
# what was in it, so the pre-flight check verified the wrong container and
# looked correct.
#
# WHAT THIS IS AND IS NOT. These are STATIC markers -- a patch's source is
# present. That is enough to catch "wrong image", which is the failure this
# exists to prevent. It is NOT proof the fast path engages at runtime: docs/55
# records an image carrying the CK carve-outs whose workers still selected
# TritonInt8ScaledMMLinearKernel. A round must still assert on the kernel it
# reads back from a running server. Static manifest answers "could this image
# possibly do X"; the runtime assertion answers "did it".
set -uo pipefail

IMAGES=${IMAGES:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -E 'vllm-mi210|rocm-vllm-aiter' | sort)}
R=/opt/python/lib/python3.14/site-packages

probe_one() {  # image
    local img="$1"
    docker run --rm --entrypoint /bin/bash "$img" -c '
R=/opt/python/lib/python3.14/site-packages
say() { printf "%s=%s;" "$1" "$2"; }

# configs/enable_aiter_ck_gemm_gfx90a.py, blocker 1: the codegen CU table.
grep -q "\"gfx90a\"" $R/aiter/jit/utils/build_targets.py 2>/dev/null \
  && say ck_gemm_cu_map yes || say ck_gemm_cu_map no

# configs/enable_aiter_ck_gemm_gfx90a.py, blocker 2: the selection gate.
grep -q "gfx90a carve-out" $R/vllm/_aiter_ops.py 2>/dev/null \
  && grep -A8 "def is_linear_enabled" $R/vllm/_aiter_ops.py 2>/dev/null | grep -q "is_aiter_attention_supported" \
  && say ck_gemm_linear_gate yes || say ck_gemm_linear_gate no

# configs/enable_vllm_aiter_gfx90a.py -- AITER attention dispatch.
grep -rq "is_aiter_attention_supported" $R/vllm/_aiter_ops.py 2>/dev/null \
  && say aiter_attention yes || say aiter_attention no

# configs/enable_aiter_gdn_and_moe_policy_gfx90a.py
grep -q "enable_aiter_gdn_and_moe_policy_gfx90a" $R/vllm/_aiter_ops.py 2>/dev/null \
  && say gdn_moe_policy yes || say gdn_moe_policy no

# configs/extend_rocm_pa_256k_gfx9.py -- npar_loops past the 128k gate.
grep -rq "npar_loops" $R/aiter_meta/csrc/cpp_itfs/pa/ 2>/dev/null \
  && say pa_256k yes || say pa_256k no

# configs/enable_moe_padding_int8_rocm.py -- measured 0.961x, should be ABSENT.
grep -rq "_mi210_pad_moe_weight" $R/vllm/model_executor/layers/quantization/ 2>/dev/null \
  && say moe_padding_int8 yes || say moe_padding_int8 no

# Version, for cross-referencing rounds.
v=$(/opt/python/bin/python -c "import vllm;print(vllm.__version__)" 2>/dev/null | tail -1)
say vllm "${v:-unknown}"
echo
' 2>/dev/null
}

echo "image                                        ck_map  ck_gate  aiter_attn  gdn_moe  pa256k  moe_pad  vllm"
echo "---------------------------------------------------------------------------------------------------------"
for img in $IMAGES; do
    id=$(docker inspect --format '{{.Id}}' "$img" 2>/dev/null | cut -c8-19)
    out=$(probe_one "$img")
    get() { echo "$out" | tr ';' '\n' | grep "^$1=" | cut -d= -f2; }
    printf "%-44s %-7s %-8s %-11s %-8s %-7s %-8s %s\n" \
        "$img" "$(get ck_gemm_cu_map)" "$(get ck_gemm_linear_gate)" "$(get aiter_attention)" \
        "$(get gdn_moe_policy)" "$(get pa_256k)" "$(get moe_padding_int8)" "$(get vllm)"
    echo "    id=$id"
done

echo ""
echo "STATIC markers only. An image showing ck_map=yes and ck_gate=yes CAN reach"
echo "the AITER CK int8 GEMM; whether it DOES is a runtime question -- docs/55"
echo "records one that carried both and still selected the Triton kernel in its"
echo "workers. Rounds must read the selected kernel back from a running server."
echo ""
echo "serve_vllm_aiter.sh reads VLLM_IMAGE for the SERVER and defaults to"
echo "rocm-vllm-aiter-gfx90a:latest. Setting only a client image leaves the"
echo "server on the default -- the round 73 defect."
