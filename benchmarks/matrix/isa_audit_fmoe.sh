#!/usr/bin/env bash
# Reproduces every instruction-level claim in docs/49. Static analysis only --
# no GPU is touched, no container claims a device, so this is safe to run at any
# time and does not need the bench lock.
#
# Run inside the vLLM image, which carries the full ROCm LLVM toolchain:
#   docker run --rm -v $PWD/isa_audit_fmoe.sh:/tmp/a.sh --entrypoint bash \
#       vllm-mi210:gdnpolicy /tmp/a.sh
set -uo pipefail

LLVM=/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin
OBJD=$LLVM/llvm-objdump
MC=$LLVM/llvm-mc
H=/opt/python/lib/python3.14/site-packages/aiter_meta/hsa
A=/opt/python/lib/python3.14/site-packages/aiter

T=$H/gfx942/fmoe/silu/fmoe_bf16_noquantBf16_g1u0_atm_inlv_silu_1tg_32x512.co
G=$H/gfx90a/fmoe/silu/fmoe_fp16_noquantFp16_g1u0_atm_inlv_silu_1tg_32x512.co

echo "############ 1. the two kernels are the same kernel ############"
for f in "$T" "$G"; do [ -f "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done
$OBJD -d --mcpu=gfx942 "$T" 2>/dev/null | awk '{print $1}' | grep -E '^[vsbdfg]' > /tmp/a.txt
$OBJD -d --mcpu=gfx90a "$G" 2>/dev/null | awk '{print $1}' | grep -E '^[vsbdfg]' > /tmp/b.txt
echo "gfx942 bf16 stream length : $(wc -l < /tmp/a.txt)"
echo "gfx90a fp16 stream length : $(wc -l < /tmp/b.txt)"
echo "differing positions       : $(paste /tmp/a.txt /tmp/b.txt | awk '$1!=$2' | wc -l)"
echo "--- the COMPLETE substitution list (expect exactly two) ---"
paste /tmp/a.txt /tmp/b.txt | awk '$1!=$2{print $1" -> "$2}' | sort | uniq -c | sort -rn

echo
echo "############ 2. operand-level identity of the atomic ############"
echo "-- gfx942:"; $OBJD -d --mcpu=gfx942 "$T" 2>/dev/null | grep 'global_atomic_pk_add' | head -2
echo "-- gfx90a:"; $OBJD -d --mcpu=gfx90a "$G" 2>/dev/null | grep 'global_atomic_pk_add' | head -2

echo
echo "############ 3. assembler verdicts ############"
try() {
    printf '%s\n' "$2" > /tmp/t.s
    if out=$($MC -arch=amdgcn -mcpu="$1" -show-encoding /tmp/t.s 2>&1); then
        printf "  %-7s ACCEPTS  %s\n" "$1" "$(echo "$out" | grep -oE 'encoding: \[.*\]' | head -1)"
    else
        printf "  %-7s REJECTS  %s\n" "$1" "$(echo "$out" | grep -oE 'error:.*' | head -1)"
    fi
}
echo "the MFMA, in each architecture's own spelling -- note gfx942 accepts BOTH"
echo "and assembles them identically, i.e. it is one operation with two names:"
try gfx90a "v_mfma_f32_16x16x16_bf16 v[128:131], a[0:1], v[64:65], v[128:131]"
try gfx942 "v_mfma_f32_16x16x16_bf16 v[128:131], a[0:1], v[64:65], v[128:131]"
try gfx90a "v_mfma_f32_16x16x16bf16_1k v[128:131], a[0:1], v[64:65], v[128:131]"
try gfx942 "v_mfma_f32_16x16x16bf16_1k v[128:131], a[0:1], v[64:65], v[128:131]"
echo "the one real silicon gap:"
try gfx90a "global_atomic_pk_add_bf16 v0, v[2:3], v1, off"
echo "and FP8, for contrast:"
try gfx90a "v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:5], v[6:7], v[0:3]"
try gfx942 "v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:5], v[6:7], v[0:3]"

echo
echo "############ 4. every ported gfx90a fmoe object does bf16 MFMA ############"
for f in $H/gfx90a/fmoe/silu/*.co $H/gfx90a/fmoe/gelu/*.co; do
    [ -f "$f" ] || continue
    printf "  %-58s %s\n" "$(basename "$f")" \
        "$($OBJD -d --mcpu=gfx90a "$f" 2>/dev/null | grep -oE 'v_mfma[a-z0-9_]*' | sort -u | tr '\n' ' ')"
done

echo
echo "############ 5. what the live dispatcher asks for vs what exists ############"
echo "--- asm_moe.py g1u0 branch, keyed on INPUT dtype ---"
grep -n -A4 'input_dtype == "__half"\|input_dtype == "__hip_bfloat16"' \
    /src/aiter/csrc/cpp_itfs/moe/asm_moe.py 2>/dev/null | grep -E '\.co"|input_dtype' | head
echo "--- presence on gfx90a ---"
for n in fmoe_f16.co fmoe_b16.co; do
    printf "  %-16s %s\n" "$n" "$([ -f "$H/gfx90a/$n" ] && echo PRESENT || echo MISSING)"
done
printf "  %-16s %s objects\n" "int8 subGU" "$(ls $H/gfx90a/fmoe/silu/fmoe_int8_g1u0_subGU_* 2>/dev/null | wc -l)"

echo
echo "############ 6. the 8 noquant objects have no consumer ############"
echo -n "  .so files containing the string 'noquant' : "
find /opt/python/lib/python3.14/site-packages /src/aiter -name '*.so' 2>/dev/null \
    | xargs -r -I{} sh -c 'strings "{}" 2>/dev/null | grep -q noquant && echo "{}"' | wc -l
echo -n "  cfg_fmoe_*noquant* refs in C++ tree       : "
grep -rn 'noquant' --include=*.cu --include=*.cpp --include=*.hpp --include=*.h \
    /src/aiter 2>/dev/null | grep -v build/lib | wc -l
echo -n "  generated asm_fmoe_configs.hpp present    : "
find / -name 'asm_fmoe_configs*' 2>/dev/null | head -1 | grep -q . && echo YES || echo NO

echo
echo "############ 7. the load-bearing arch assert, and its reason ############"
grep -n -B1 -A3 'FLAT fmoe asm kernels require' $A/fused_moe.py 2>/dev/null
grep -n 'global_atomic_pk_add_bf16' $A/fused_moe.py 2>/dev/null

echo
echo "############ 8. what vLLM actually calls (NOT the ASM tree) ############"
grep -n 'ck_moe_stage1\|ck_moe_stage2_fwd' $A/fused_moe.py 2>/dev/null | head -5

echo
echo "=== expected, per docs/49 ==="
cat <<'EOF'
  1. both streams 3833 long; 1072 differing; exactly two substitutions
     (1024x MFMA rename, 48x bf16->fp16 atomic)
  3. gfx90a REJECTS v_mfma_f32_16x16x16_bf16 but ACCEPTS the _1k spelling;
     gfx942 accepts both and emits the SAME encoding for each.
     gfx90a REJECTS global_atomic_pk_add_bf16 and the fp8 MFMA.
  4. all 8 objects show v_mfma_f32_16x16x16bf16_1k -- bf16 compute
  5. fmoe_f16.co PRESENT, fmoe_b16.co MISSING, 0 int8 subGU objects
  6. zero .so hits, zero C++ refs, no generated header -> orphaned objects
  8. ck_moe_stage1/ck_moe_stage2_fwd -> Composable Kernel, source-compiled
EOF
