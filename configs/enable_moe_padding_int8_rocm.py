#!/usr/bin/env python3
"""Extend vLLM's ROCm MoE stride padding to the INT8 expert path.

WHAT IT IS. vLLM has a ROCm-specific MoE weight optimization that pads a
weight by 256 bytes and immediately slices it back:

    num_pad = 256 // weight.element_size()
    weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]

The logical shape and every value are unchanged; only the row **stride** moves,
which de-aliases memory channels/banks. Upstream (vllm-project/vllm#14454)
reports up to 10% on Mixtral. `VLLM_ROCM_MOE_PADDING` defaults to **True**
(`envs.py:146`).

THE GAP. Grep the tree for `VLLM_ROCM_MOE_PADDING` and it appears in exactly
three files: `envs.py`, `fused_moe/oracle/unquantized.py`, and
`unquantized_fused_moe_method.py`. **It is nowhere in the quantized path.** A
W8A8 checkpoint never receives it — and not because something undoes it later
(vLLM explicitly guards that at `oracle/unquantized.py:353`, "Skip
.contiguous(): it would undo the ROCm MoE weight padding"). It is simply never
applied.

That matters here because `fused_moe_kernel` is the **largest single decode
kernel** on this box — 1947 ms, 24.4% of decode window (`docs/45`) — after the
CK GEMM removed the previous leader.

ELIGIBILITY. The upstream guard only pads when the row stride is already a
multiple of 512 bytes. For Qwen3-30B-A3B W8A8 at int8 (element_size 1):

    w13  [E, 2*768, 2048]  row stride 2048 B  ->  2048 % 512 == 0   ELIGIBLE
    w2   [E, 2048,  768]   row stride  768 B  ->   768 % 512 == 256 not eligible

So roughly half the expert weight bytes are affected. The condition is checked
per tensor at runtime, so a model whose shapes do not qualify simply gets the
current behaviour.

WHY THIS IS SAFE. Pad-then-slice returns a view with the original logical
shape and identical values; nothing downstream sees a different tensor. The
one real hazard is a later `.contiguous()` re-tightening the stride and
silently undoing it — which is why this script does NOT claim a win, and why
the round that uses it asserts on measured tensor strides rather than on the
patch having been applied.

NOT A CLAIM. No measurement backs this yet on gfx90a. Upstream's 10% is
Mixtral, unquantized, TP=8, batch 64 — a very different regime from batch-1
decode at TP=2. This script exists so the A/B can be run.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_TARGET = (
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "compressed_tensors_moe/compressed_tensors_moe_w8a8_int8.py"
)

_ANCHOR = """        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
"""

_PATCHED = '''        # gfx90a: ROCm MoE stride padding, which vLLM applies only on the
        # unquantized path (configs/enable_moe_padding_int8_rocm.py). Pads by
        # 256 B and slices straight back, so the logical shape and all values
        # are unchanged and only the row stride moves -- the point is
        # de-aliasing memory banks, not changing the math.
        w13 = _mi210_pad_moe_weight(w13)
        w2 = _mi210_pad_moe_weight(w2)
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
'''

# Inserted once at module scope. Mirrors _maybe_pad_weight in
# unquantized_fused_moe_method.py, minus the EPLB check -- EPLB needs
# contiguous weights for its rearrangement, and this path does not implement
# EPLB, but the guard is kept defensively via the stride test below.
_HELPER = '''

def _mi210_pad_moe_weight(weight: "torch.Tensor") -> "torch.Tensor":
    """Bump the row stride by 256 B to de-alias memory banks.

    Added by configs/enable_moe_padding_int8_rocm.py. Upstream ships this for
    unquantized MoE only (VLLM_ROCM_MOE_PADDING, default True); the INT8 expert
    path never receives it. Logical shape and values are unchanged.
    """
    import torch.nn.functional as _F

    from vllm import envs as _envs
    from vllm.platforms import current_platform as _plat

    if not (
        getattr(_envs, "VLLM_ROCM_MOE_PADDING", False)
        and _plat.is_rocm()
        and weight.stride(-1) == 1
        and (weight.stride(-2) * weight.element_size()) % 512 == 0
    ):
        return weight
    num_pad = 256 // weight.element_size()
    return _F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
'''

_HELPER_MARK = "def _mi210_pad_moe_weight("
_ANCHOR_FOR_HELPER = "logger = init_logger(__name__)\n"


def _path(site: pathlib.Path) -> pathlib.Path:
    p = site / _TARGET
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(site: pathlib.Path) -> int:
    s = _path(site).read_text()
    helper = _HELPER_MARK in s
    # Test ONLY for the inserted call. Do NOT also assert the anchor is absent:
    # _ANCHOR (the two replace_parameter lines) is a SUBSTRING of _PATCHED,
    # since the patch inserts before them rather than replacing them. An
    # "anchor is gone" test therefore fails on a correctly patched file --
    # which is exactly what it did on the first run of the round-trip test.
    call = "_mi210_pad_moe_weight(w13)" in s and "_mi210_pad_moe_weight(w2)" in s
    print(f"  helper inserted     : {'PATCHED' if helper else 'not patched'}")
    print(f"  call site rewired   : {'PATCHED' if call else 'not patched'}")
    return 0 if (helper and call) else 1


def apply(site: pathlib.Path) -> int:
    p = _path(site)
    s = p.read_text()

    if _HELPER_MARK in s and "_mi210_pad_moe_weight(w13)" in s:
        print("  already patched; leaving it alone")
        return 0

    n = s.count(_ANCHOR)
    if n != 1:
        sys.exit(
            f"FATAL: replace_parameter anchor matched {n} times, expected 1. "
            "Upstream changed process_weights_after_loading; re-derive this "
            "patch rather than forcing it."
        )
    m = s.count(_ANCHOR_FOR_HELPER)
    if m != 1:
        sys.exit(
            f"FATAL: logger anchor matched {m} times, expected 1 -- cannot "
            "place the helper deterministically."
        )

    s = s.replace(_ANCHOR_FOR_HELPER, _ANCHOR_FOR_HELPER + _HELPER, 1)
    s = s.replace(_ANCHOR, _PATCHED, 1)
    p.write_text(s)
    print(f"  patched {p}")
    print("  NOTE: inert unless VLLM_ROCM_MOE_PADDING is true (default) and the")
    print("  tensor's row stride is already a multiple of 512 bytes.")
    return 0


def revert(site: pathlib.Path) -> int:
    p = _path(site)
    s = p.read_text()
    if _PATCHED not in s and _HELPER not in s:
        print("  not patched; nothing to revert")
        return 0
    before = s
    s = s.replace(_PATCHED, _ANCHOR, 1).replace(_HELPER, "", 1)
    if s == before:
        print("  COULD NOT revert: the file carries a variant written by a "
              "different revision of this script. Remove it by hand.")
        return 1
    p.write_text(s)
    print(f"  reverted {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="/opt/python/lib/python3.14/site-packages")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--assert-patched", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    site = pathlib.Path(a.site)
    if a.check:
        return check(site)
    if a.assert_patched:
        rc = check(site)
        if rc:
            print("ASSERTION FAILED: INT8 MoE padding patch is not applied.")
        return rc
    if a.revert:
        return revert(site)
    return apply(site)


if __name__ == "__main__":
    sys.exit(main())
