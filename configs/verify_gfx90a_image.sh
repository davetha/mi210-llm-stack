#!/usr/bin/env bash
# Prove that a vllm-mi210 image can actually reach the AITER ASM kernels on the
# card in front of it. Run ON A MACHINE WITH MI210s:
#
#   docker run --rm --device /dev/kfd --device /dev/dri \
#     --security-opt seccomp=unconfined --group-add video \
#     --entrypoint /usr/local/bin/verify-gfx90a vllm-mi210:latest
#
# Dockerfile.vllm-mi210 already verifies, at build time, that every patch landed
# in the files. That is necessary and not sufficient: the patches open a path,
# and this script is what establishes that something travels down it. The
# distinction matters because a half-patched image does not error -- it serves
# correct results slowly, from the Triton fallback, which is how this project
# published bad numbers twice before.
#
# Exit 0 only if every check passes. Nothing here is advisory.
set -uo pipefail

SITE=/opt/python/lib/python3.14/site-packages
fail=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=1; }

echo "=== 1. the card is visible and is gfx90a ==="
arch="$("${SITE}/_rocm_sdk_devel/bin/rocminfo" 2>/dev/null | grep -om1 'gfx[0-9a-z]*')"
if [ "${arch}" = "gfx90a" ]; then
    pass "rocminfo reports ${arch}"
else
    bad "expected gfx90a, rocminfo reports '${arch:-nothing}'. Pass --device /dev/kfd --device /dev/dri."
    # Everything below needs the GPU; there is no point continuing.
    echo; echo "ABORTED: no usable gfx90a device."; exit 1
fi

echo
echo "=== 2. aiter imports and agrees about the architecture ==="
# GPU_ARCHS is deliberately NOT set here. The build sets it so the patch steps
# can run on a machine with no card, but at runtime we want the real detection
# path exercised -- that is the one vLLM's dispatch actually consults.
if out="$(env -u GPU_ARCHS python3 -c '
import aiter
from aiter.jit.utils.chip_info import get_gfx
print(get_gfx())
' 2>&1)"; then
    got="$(printf '%s' "${out}" | tail -1)"
    if [ "${got}" = "gfx90a" ]; then
        pass "import aiter OK, get_gfx() = ${got}"
    else
        bad "import aiter OK but get_gfx() = '${got}', expected gfx90a"
    fi
else
    bad "import aiter failed:"; printf '%s\n' "${out}" | tail -12
fi

echo
echo "=== 3. the fmha_v3_fwd ASM kernel loads AND is numerically correct ==="
# Two claims, deliberately checked together in one process:
#
#   (a) it RAN -- with AITER_LOG_LEVEL=info the runtime prints the .co file it
#       loads per ASM call. That log line is the only direct evidence the ASM
#       kernel executed rather than falling back to CK or Triton, which it does
#       silently and which no throughput number distinguishes reliably.
#   (b) it was RIGHT -- compared against torch SDPA. A fast wrong answer is
#       worse than the slow right one, and binary-patched kernels are exactly
#       the place to be paranoid about that.
#
# bf16 only: of the 56 fmha_v3_fwd kernels, the 48 bf16 ones are portable to
# CDNA2 and the 8 FP8 ones are not, because gfx90a has no FP8 ALU.
env -u GPU_ARCHS AITER_LOG_LEVEL=info python3 - <<'PY' 2>&1 | tee /tmp/fmha_verify.log
import sys
import torch
import torch.nn.functional as F
import aiter
from aiter import dtypes

torch.manual_seed(0)
b, sq, sk, hq, hkv, hd = 2, 1024, 1024, 8, 1, 128   # GQA ratio 8, hdim 128
q = torch.empty(b, sq, hq, hd, dtype=dtypes.bf16, device="cuda").uniform_(-1, 1)
k = torch.empty(b, sk, hkv, hd, dtype=dtypes.bf16, device="cuda").uniform_(-1, 1)
v = torch.empty(b, sk, hkv, hd, dtype=dtypes.bf16, device="cuda").uniform_(-1, 1)
scale = hd ** -0.5

out = aiter.flash_attn_func(q, k, v, softmax_scale=scale, causal=True)

ref = F.scaled_dot_product_attention(
    q.transpose(1, 2).float(),
    k.transpose(1, 2).float().expand(b, hkv, sk, hd).repeat_interleave(hq // hkv, dim=1),
    v.transpose(1, 2).float().expand(b, hkv, sk, hd).repeat_interleave(hq // hkv, dim=1),
    is_causal=True, scale=scale,
).transpose(1, 2).to(out.dtype)

err = (out.float() - ref.float()).abs().max().item()
print(f"VERIFY max_abs_err={err:.5f}")
# bf16 has ~3 decimal digits; 1e-2 is loose enough for accumulation order to
# differ and tight enough that a wrong kernel cannot pass.
sys.exit(0 if err < 1e-2 else 1)
PY
numeric=${PIPESTATUS[0]}

if [ "${numeric}" -eq 0 ] && grep -q 'VERIFY max_abs_err' /tmp/fmha_verify.log; then
    pass "flash_attn_func matches SDPA ($(grep -o 'max_abs_err=[0-9.]*' /tmp/fmha_verify.log))"
else
    bad "flash_attn_func numeric check failed"; tail -15 /tmp/fmha_verify.log
fi

if grep -qE 'gfx90a.*\.co|\.co.*gfx90a' /tmp/fmha_verify.log; then
    # Anchor on the LoadKernel line. A bare `grep -oE '[^ /]*\.co'` over the
    # whole log picks up ".copy()" from a pandas deprecation warning that is
    # printed earlier, and reports the code object as ".co".
    co="$(grep 'LoadKernel:' /tmp/fmha_verify.log | grep -oE '[^/]+\.co' | head -1)"
    pass "loaded a gfx90a ASM code object: ${co:-<name not parsed>}"
else
    # This is the check that catches the failure this whole image exists to
    # prevent: correct results, produced by the fallback, at a third of the speed.
    bad "NO gfx90a .co load logged -- the result was correct but the ASM kernel did NOT run."
    echo "        (this is the silent-fallback failure; the image is not delivering AITER ASM)"
fi

echo
echo "=== 4. vLLM selects an AITER attention backend ==="
# The tell is a negative: on an unpatched image vLLM logs
# "Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN','TRITON_ATTN']"
# and AITER never enters the candidate list at all.
if out="$(env -u GPU_ARCHS python3 -c '
from vllm._aiter_ops import is_aiter_attention_supported
print("attention_supported:", bool(is_aiter_attention_supported()))
' 2>&1)"; then
    if printf '%s' "${out}" | grep -q 'attention_supported: True'; then
        pass "vllm._aiter_ops.is_aiter_attention_supported() is True"
    else
        bad "is_aiter_attention_supported() is not True: ${out}"
    fi
else
    bad "could not query vLLM's AITER gate:"; printf '%s\n' "${out}" | tail -8
fi

# ...and that it can actually be CHOSEN, which is a separate question.
#
# This check exists because the first build of this image passed everything
# above and still never ran an ASM kernel. _get_backend_priorities() appends
# ROCM_ATTN unconditionally and returns the first valid entry, so AITER sits in
# the candidate list forever:
#
#   Overriding with ROCM_ATTN out of potential backends:
#       ['ROCM_ATTN', 'ROCM_AITER_FA', 'TRITON_ATTN']
#
# Round 32 measured that arm at zero LoadKernel lines. Candidacy is not
# selection, and only the ordering distinguishes them.
if grep -q '_prefer_aiter_fa' \
     "${SITE}/vllm/platforms/rocm.py" 2>/dev/null; then
    pass "ROCM_AITER_FA can outrank ROCM_ATTN (serve with VLLM_PREFER_AITER_FA=1)"
else
    bad "AITER FA is admitted but can never be SELECTED -- ROCM_ATTN is appended"
    echo "        first unconditionally, so no ASM kernel will ever load."
    echo "        Apply configs/prefer_aiter_fa_gfx90a.py."
fi

echo
echo "=== 5. paged attention accepts npar_loops > 8 (the 256k patch) ==="
# 8 * 64 * 256 = 131,072 is the stock ceiling. A max_seq_len above it must NOT
# raise "Unsupported npar_loops"; on a stock build it does, loudly, which is
# what makes this checkable at all rather than a silent correctness question.
if out="$(env -u GPU_ARCHS python3 -c '
from vllm.platforms.rocm import RocmPlatform
import inspect, vllm.platforms.rocm as m
src = inspect.getsource(m)
assert "256 * 1024" in src, "gfx9 gate still capped at 128k"
print("gate: 256k")
' 2>&1)"; then
    pass "gfx9 custom-PA gate raised to 256k"
else
    bad "256k gate not present:"; printf '%s\n' "${out}" | tail -6
fi

echo
echo "=== 6. vLLM can inspect a model architecture ==="
#
# Added after round 36. Both source-built images passed every check above and
# then failed EVERY arm with
#
#   Model architectures ['Qwen3MoeForCausalLM'] failed to be inspected
#
# The cause was a tensorizer downgrade breaking an import, but the point here is
# the gap it exposed: everything above tests kernels and gates, and none of it
# loads a model. vLLM inspects architectures in a subprocess and DISCARDS its
# stderr, so the serve-time error names neither the package nor the real
# exception. Running that subprocess directly is the cheapest way to find out --
# it needs no weights and no GPU time.
# Run the registry module exactly as vLLM does, and judge it on HOW it fails.
#
# An earlier version of this check called ModelRegistry.resolve_model_cls() with
# one argument. That signature needs a model_config, so the check raised
# TypeError and reported two perfectly good images as broken -- a false alarm is
# as bad as a missed one here.
#
# This runs the real subprocess with no stdin. A healthy image gets all its
# imports done and THEN dies on the empty pickle:
#
#   EOFError: Ran out of input          <- expected, means imports succeeded
#
# A broken optional dependency instead dies during import, which is what
# tensorizer 2.10.1 did, and is what makes every serve fail with the unhelpful
# "Model architectures [...] failed to be inspected".
reg_out="$(python3 -m vllm.model_executor.models.registry </dev/null 2>&1)"
if printf '%s' "${reg_out}" | grep -q "EOFError: Ran out of input"; then
    pass "model registry subprocess imports cleanly (reached stdin read)"
elif printf '%s' "${reg_out}" | grep -qE "ImportError|ModuleNotFoundError|KeyError|AttributeError"; then
    bad "registry subprocess fails during IMPORT -- every serve will fail with"
    echo "        \"Model architectures [...] failed to be inspected\", which names"
    echo "        neither the package nor the error. The real cause:"
    printf '%s\n' "${reg_out}" | grep -E "ImportError|ModuleNotFoundError|KeyError|AttributeError" | tail -3 | sed 's/^/          /'
else
    bad "registry subprocess failed in an unrecognised way:"
    printf '%s\n' "${reg_out}" | tail -5 | sed 's/^/          /'
fi

echo
echo "=== 7. the AITER CK int8 GEMM is reachable (registered AND selectable) ==="
# Added after round 40, which shipped an image where checks 1-6 all passed and
# serving still died at the first qkv_proj with
#
#   AttributeError: '_OpNamespace' 'vllm' object has no attribute
#                   'rocm_aiter_w8a8_gemm'
#
# register_ops_once() carries @if_aiter_supported -> on_mi3xx(), so on CDNA2
# NOT ONE aiter custom op is registered. Nothing above catches it: the ASM
# attention path calls aiter directly rather than through torch.ops.vllm, so
# every attention check passes on an image where the linear path cannot work.
#
# Both halves are checked because they fail independently -- the op can exist
# while the gate refuses to select it, and vice versa. The flags are set here
# deliberately: is_linear_enabled() reads them, and without them a perfectly
# good image reports False.
if out="$(env -u GPU_ARCHS VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_LINEAR=1 \
    python3 -c '
import torch
import vllm._aiter_ops as ao
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel as K,
)
print("registered:", hasattr(torch.ops.vllm, "rocm_aiter_w8a8_gemm"))
print("gate:", bool(ao.rocm_aiter_ops.is_linear_enabled()))
print("selectable:", bool(K.is_supported()[0]))
' 2>&1)"; then
    if printf '%s' "${out}" | grep -q 'registered: True' \
    && printf '%s' "${out}" | grep -q 'gate: True' \
    && printf '%s' "${out}" | grep -q 'selectable: True'; then
        pass "CK int8 GEMM op registered, gated on, and selectable"
    else
        bad "CK int8 GEMM is NOT reachable -- serving will die at the first"
        echo "        quantized linear, or silently run the Triton fallback at"
        echo "        ~0.68x decode. Apply configs/enable_aiter_ck_gemm_gfx90a.py"
        echo "        (after enable_vllm_aiter_gfx90a.py). Got:"
        printf '%s\n' "${out}" | tail -6 | sed 's/^/          /'
    fi
else
    bad "could not query the CK GEMM path:"; printf '%s\n' "${out}" | tail -8
fi

echo
if [ "${fail}" -eq 0 ]; then
    echo "ALL CHECKS PASSED -- this image reaches AITER ASM on gfx90a."
else
    echo "VERIFICATION FAILED. Do NOT benchmark this image; it will produce"
    echo "correct-looking numbers from a fallback path."
fi
exit "${fail}"
