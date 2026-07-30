#!/usr/bin/env python3
"""Fix benchmark_moe.py's int8_w8a16 path, which declares int8 ACTIVATIONS.

THE BUG. Tuning an int8_w8a16 MoE dies before it measures anything:

    RuntimeError: const_data_ptr,
      torch/include/torch/csrc/stable/tensor_inl.h:69, expected scalar type

docs/25 item 3b recorded this as "a dtype mismatch inside the tuner's own
int8_w8a16 tensor construction, on torch 2.11's stable ABI. Not a configuration
error and not fixable from the invocation." The first half is right and the
second is too, but the mismatch is not in the tensor construction -- the weights
are built correctly as torch.int8. It is in the quant config:

    elif use_int8_w8a16:
        quant_dtype = torch.int8          # <- declares int8 ACTIVATIONS

quant_dtype is the ACTIVATION quantization type, not the weight type. From
FusedMoEQuantConfig.make's own docstring (config.py:517):

    - quant_dtype: Optional quantization type. None if activations are
      unquantized or quantized prior to calling.

w8a16 means int8 weights and SIXTEEN-BIT activations. Setting quant_dtype to
torch.int8 configures W8A8. The kernel then reads the activation tensor
expecting int8, is handed bf16, and throws in const_data_ptr.

The weight side has its own parameter, already used a few lines below for int4:

    weight_dtype="int4" if use_int4_w4a16 else None

So the fix is to leave activations unquantized and declare the weights int8.

WHY IT MATTERS HERE. vLLM ships tuned fused_moe configs for MI300X, MI308X,
MI325X, MI350X and MI355X, and none for MI210, so gfx90a tile selection falls to
heuristics. Counting the shipping fused_moe_kernel on this box shows the cost:
decode shapes select 16x16x16bf16_1k at 191-196 MACs per issued instruction
while larger shapes select 32x32x8bf16_1k at 485-497. Fixed per-kernel overhead
is ~450-500 instructions either way, so the large tiles amortise it over 4x the
MACs and decode sits on the wrong side of a 2.5x spread. This tuner is the
supported route to an MI210 config, and it has never run.

HOW THIS COULD BE WRONG. The RuntimeError is reported from the stable-ABI layer,
so a torch 2.11 regression could in principle produce the same message from a
genuinely different cause; this patch is justified by the docstring contract
rather than by a stack trace naming quant_dtype. If the tuner still fails after
this, the remaining error is a second bug and not evidence that this one was
imaginary. And a config generated from a mis-declared kernel would be worse than
none, so verify the tuner reports int8_w8a16 in the filename it writes.

Usage:
    python3 fix_benchmark_moe_int8_w8a16.py --path /path/to/benchmark_moe.py
    python3 fix_benchmark_moe_int8_w8a16.py --check --path ...
    python3 fix_benchmark_moe_int8_w8a16.py --revert --path ...
"""
import argparse
import sys

OLD = """        elif use_int8_w8a16:
            quant_dtype = torch.int8
"""
NEW = """        elif use_int8_w8a16:
            # w8a16 is int8 WEIGHTS with 16-bit activations. quant_dtype is the
            # activation type and must stay None here; the weight side is
            # declared through weight_dtype below. Setting quant_dtype=torch.int8
            # configures W8A8 and makes the kernel read bf16 activations as int8.
            quant_dtype = None
"""
OLD_W = '            weight_dtype="int4" if use_int4_w4a16 else None,\n'
NEW_W = '            weight_dtype=_moe_weight_dtype(use_int4_w4a16, use_int8_w8a16),\n'
HELPER = '''

def _moe_weight_dtype(use_int4_w4a16: bool, use_int8_w8a16: bool):
    """Weight dtype for FusedMoEQuantConfig.make, separate from activations."""
    if use_int4_w4a16:
        return "int4"
    if use_int8_w8a16:
        return torch.int8
    return None
'''
ANCHOR = "\ndef main(" 


def apply(text):
    if OLD not in text:
        return None, "quant_dtype block not found (source differs from expected)"
    if OLD_W not in text:
        return None, "weight_dtype line not found (source differs from expected)"
    if ANCHOR not in text:
        return None, "no 'def main(' anchor to insert the helper before"
    text = text.replace(OLD, NEW, 1).replace(OLD_W, NEW_W, 1)
    text = text.replace(ANCHOR, HELPER + ANCHOR, 1)
    return text, None


def revert(text):
    if NEW not in text:
        return None, "patch not present"
    text = text.replace(NEW, OLD, 1).replace(NEW_W, OLD_W, 1).replace(HELPER, "", 1)
    return text, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    src = open(a.path).read()
    applied = NEW in src and NEW_W in src

    if a.check:
        print("PATCHED" if applied else "NOT PATCHED", "--", a.path)
        return 0 if applied else 1

    if a.revert:
        if not applied:
            print("not patched, nothing to revert"); return 1
        out, err = revert(src)
        if err:
            print("refusing:", err); return 1
        open(a.path, "w").write(out); print("reverted", a.path); return 0

    if applied:
        print("already patched"); return 0
    out, err = apply(src)
    if err:
        # Refuse rather than force. A forced patch here silently produces a
        # kernel selected under the wrong scheme, which is worse than the
        # crash it replaces.
        print("refusing:", err); return 1
    open(a.path, "w").write(out); print("patched", a.path); return 0


if __name__ == "__main__":
    sys.exit(main())
