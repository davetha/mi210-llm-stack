#!/usr/bin/env python3
"""Open the remaining AITER capability gates on gfx90a, one flag each.

WHY. `enable_aiter_ck_gemm_gfx90a.py` fixed op REGISTRATION globally -- before
it, `register_ops_once()` was `on_mi3xx()`-gated and not one AITER custom op
existed on CDNA2. With that done, every other AITER capability is blocked by
exactly one thing: its own `is_*_enabled()` gate, also `on_mi3xx()`-gated.

A survey of all 17 gates on this card found `linear` and `mha` True (the two
already carved out) and **14 returning None** -- the decorator short-circuit.
Separately, every underlying aiter symbol imports cleanly here:
`aiter.fused_moe.fused_moe`, `asm_moe_tkw1`, `topk_softmax`, `grouped_topk`,
`dynamic_per_token_scaled_quant`, `pa_fwd_asm`, `hipb_mm` -- all OK on gfx90a.

Ranked against the round-41 decode profile (7,979 ms busy in an 8 s window),
the gated kernels are worth:

    fused_moe_kernel        1947 ms   24.4%   <- largest single kernel
    paged_attention_ll4mi   1045 ms   13.1%
    wvSplitK (2 variants)    680 ms    8.5%
    dynamic_scaled_int8_q    498 ms    6.2%
    topkGating               393 ms    4.9%
    moe_sum_vec              220 ms    2.8%

WHAT THIS DOES NOT DO. It does not turn anything on. Each capability keeps its
own `VLLM_ROCM_USE_AITER_*` env flag, all of which default to off, so this
patch alone changes no behaviour -- it only makes the flags *capable* of taking
effect. That is deliberate: one image, then one A/B per flag, so a kernel that
crashes or regresses is isolated to its own arm instead of taking the image
down. Same reason the CK GEMM shipped inert.

DELIBERATELY EXCLUDED, with reasons:

  is_linear_hipbmm_enabled -- its op is `rocm_aiter_hipb_mm_fp8`, an FP8 path,
      and gfx90a has no FP8 ALU. It also carries a SECOND `on_mi3xx()` inside
      its own body, so the decorator swap would not be enough anyway. This is
      why wvSplitK (8.5% of decode) has no AITER int8 replacement.
  is_mla_enabled -- MLA is DeepSeek-family only; no model here uses it.
  is_fp4bmm / is_fp8bmm / is_asm_fp4_gemm_dynamic_quant / is_linear_fp8 --
      FP8/FP4 compute, absent on CDNA2. `docs/27`.

ORDER. Must run after `enable_vllm_aiter_gfx90a.py` (for the
`is_aiter_attention_supported` predicate) and after
`enable_aiter_ck_gemm_gfx90a.py` (for op registration, without which every
carve-out here is a no-op that still cannot call anything). Both are asserted.

IMPORTING IS NOT RUNNING. Every symbol resolving proves the plumbing exists,
not that the kernel executes on this card -- `docs/37` records `fmoe` as
declining the device, and this project has twice published fallback numbers as
real ones. Each flag needs a served A/B with a `module_*` / LoadKernel
assertion before any number from it is believed.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Each entry: (gate name, the one-line body it returns).
# All five share the same shape -- decorator, classmethod, single return -- so
# the anchors are generated rather than hand-written, which keeps them honest
# if upstream reformats one of them (the match-count assert then fires).
_GATES: list[tuple[str, str]] = [
    ("is_fused_moe_enabled", "return cls._AITER_ENABLED and cls._FMOE_ENABLED"),
    ("is_triton_rotary_embed_enabled",
     "return cls._AITER_ENABLED and cls._TRITON_ROTARY_EMBED"),
    ("is_triton_unified_attn_enabled",
     "return cls._AITER_ENABLED and cls._TRITON_UNIFIED_ATTN_ENABLED"),
    ("is_custom_all_reduce_enabled",
     "return cls._AITER_ENABLED and cls._CUSTOM_ALL_REDUCE_ENABLED"),
    ("is_triton_gemm_enabled",
     "return cls._AITER_ENABLED and cls._TRITON_UNQUANT_GEMM"),
]

_PREREQ_PREDICATE = "def is_aiter_attention_supported() -> bool:"
_PREREQ_REGISTRATION = "if not is_aiter_attention_supported():\n            return\n        global _OPS_REGISTERED"


def _anchor(name: str, body: str) -> str:
    return (
        "    @classmethod\n"
        "    @if_aiter_supported\n"
        f"    def {name}(cls) -> bool:\n"
        f"        {body}\n"
    )


def _patched(name: str, body: str) -> str:
    return (
        "    @classmethod\n"
        f"    def {name}(cls) -> bool:\n"
        "        # gfx90a carve-out -- configs/enable_aiter_ops_gfx90a.py.\n"
        "        # Still governed by this capability's own VLLM_ROCM_USE_AITER_*\n"
        "        # flag, which defaults off; this only makes the flag capable of\n"
        "        # taking effect.\n"
        "        if not is_aiter_attention_supported():\n"
        "            return False\n"
        f"        {body}\n"
    )


def _ops_path(site: pathlib.Path) -> pathlib.Path:
    p = site / "vllm" / "_aiter_ops.py"
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(site: pathlib.Path) -> int:
    s = _ops_path(site).read_text()
    bad = 0
    for name, body in _GATES:
        done = _patched(name, body) in s and _anchor(name, body) not in s
        print(f"  {name:34}: {'PATCHED' if done else 'not patched'}")
        bad += 0 if done else 1
    return 0 if bad == 0 else 1


def apply(site: pathlib.Path) -> int:
    ops = _ops_path(site)
    s = ops.read_text()

    if _PREREQ_PREDICATE not in s:
        sys.exit(
            "FATAL: is_aiter_attention_supported() missing. Run "
            "configs/enable_vllm_aiter_gfx90a.py first."
        )
    if _PREREQ_REGISTRATION not in s:
        sys.exit(
            "FATAL: register_ops_once() is not carved out. Run "
            "configs/enable_aiter_ck_gemm_gfx90a.py first -- without it no "
            "AITER custom op is registered on gfx90a and every gate opened "
            "here would still have nothing to call."
        )

    for name, body in _GATES:
        anchor, patched = _anchor(name, body), _patched(name, body)
        if patched in s:
            print(f"  {name} already carved out")
            continue
        n = s.count(anchor)
        if n != 1:
            sys.exit(
                f"FATAL: {name} anchor matched {n} times, expected 1. Upstream "
                "changed the method; re-derive this patch rather than forcing it."
            )
        s = s.replace(anchor, patched)
        print(f"  patched {name}")
    ops.write_text(s)
    return 0


def revert(site: pathlib.Path) -> int:
    ops = _ops_path(site)
    s = ops.read_text()
    touched = False
    for name, body in _GATES:
        anchor, patched = _anchor(name, body), _patched(name, body)
        if patched not in s:
            print(f"  {name} not patched; nothing to revert")
            continue
        s = s.replace(patched, anchor)
        touched = True
        print(f"  reverted {name}")
    if touched:
        ops.write_text(s)
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
            print("ASSERTION FAILED: AITER capability gates are not all carved out.")
        return rc
    if a.revert:
        return revert(site)
    return apply(site)


if __name__ == "__main__":
    sys.exit(main())
