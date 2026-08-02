#!/usr/bin/env python3
"""Carve out the two AITER capability queries that are not `is_*_enabled`.

HOW THESE WERE MISSED. `docs/45` surveyed the AITER surface by enumerating
`is_*_enabled` gates -- 17 of them -- and concluded the surface was exhausted.
That pattern is incomplete. Enumerating every `@if_aiter_supported`-decorated
method finds **23**, and the six that do not match `is_*_enabled` were never
examined:

    are_gdn_triton_kernels_available   <- carved out here
    fused_moe_supports_gate_mode       no consumers in vLLM; dead
    fuse_sigmoid_in_kernel             shared-experts router only; no shared
                                       experts in any model here
    get_moe_dispatch_policy            <- carved out here
    is_enabled                         carved out by
                                       enable_aiter_master_gate_gfx90a.py
    register_ops_once                  carved out by
                                       enable_aiter_ck_gemm_gfx90a.py
    topk_softmax_supports_fused_sigmoid  no consumers in vLLM; dead

So "exhausted" was wrong, and the way it was wrong is instructive: a survey is
only as complete as its search pattern.

--------------------------------------------------------------------------
1. are_gdn_triton_kernels_available -- Gated DeltaNet, an entire forward path
--------------------------------------------------------------------------

`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:72` reads it once at
import:

    GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()

and at :799 it selects between two implementations -- the docstring is explicit:
*"ROCm forward using AITER Triton fused projection+attention when available,
otherwise falling back to the generic CUDA path."* On gfx90a the decorator
returns None, so every Qwen3-Next GDN layer takes the generic fallback.

This is not an FP8 dead end. The method's own body only tries to IMPORT the
kernels, and all of them import cleanly on this card (verified):
`aiter.ops.triton.causal_conv1d_update_single_token`,
`aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr`. They are Triton,
not architecture-specific ASM. The only thing stopping them is `on_mi3xx()`.

Relevant to real models here: both `t80-awq` and `t80-awq8` are
`Qwen3NextForCausalLM`, and production's `thinking-80b` is the same family.

--------------------------------------------------------------------------
2. get_moe_dispatch_policy -- and round 42 measured MoE with the wrong one
--------------------------------------------------------------------------

Feeds `moe_sorting_dispatch_policy` in AITER's fused MoE call. vLLM documents
the values:

    0  default heuristic, tuned FOR LARGE BATCHES
    1  always single-pass -- "may be preferred for low-concurrency decode"
    2  always multi-pass  -- "+2-5% on Qwen3-Next" (upstream PR #39177)

The getter is `@if_aiter_supported`, so on gfx90a it returns None and whatever
AITER defaults to is used regardless of the env var.

`docs/45` recorded AITER MoE at **0.977x** on batch-1 decode -- measured with
policy 0, the large-batch heuristic, on the least-batched workload there is.
That number may be partly the policy mismatch rather than a verdict on the MoE
kernels. Policies 1 and 2 have never been tried.

--------------------------------------------------------------------------

INERT ON ITS OWN. Neither carve-out turns anything on. GDN still requires
VLLM_ROCM_USE_AITER=1 (its body checks `cls._AITER_ENABLED`), and the dispatch
policy still reads VLLM_ROCM_AITER_MOE_DISPATCH_POLICY, which defaults to 0 and
only takes effect when VLLM_ROCM_USE_AITER_MOE=1.

ORDER. Run after `enable_vllm_aiter_gfx90a.py` (for the predicate) and after
`enable_aiter_ck_gemm_gfx90a.py` (for op registration). Both are asserted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_GDN_ANCHOR = """    @classmethod
    @if_aiter_supported
    def are_gdn_triton_kernels_available(cls) -> bool:
"""
_GDN_PATCHED = """    @classmethod
    def are_gdn_triton_kernels_available(cls) -> bool:
        # gfx90a carve-out -- configs/enable_aiter_gdn_and_moe_policy_gfx90a.py.
        # The body below only tries to IMPORT the Triton kernels and they all
        # import on CDNA2, so the stock on_mi3xx() decorator was the only thing
        # forcing Qwen3-Next GDN layers onto the generic fallback path.
        # Still gated on cls._AITER_ENABLED by the body itself.
"""

_POLICY_ANCHOR = """    @classmethod
    @if_aiter_supported
    def get_moe_dispatch_policy(cls) -> int:
"""
_POLICY_PATCHED = """    @classmethod
    def get_moe_dispatch_policy(cls) -> int:
        # gfx90a carve-out -- configs/enable_aiter_gdn_and_moe_policy_gfx90a.py.
        # Without this the decorator returns None on CDNA2 and
        # VLLM_ROCM_AITER_MOE_DISPATCH_POLICY has no effect at all, so round
        # 42's AITER MoE arm ran policy 0 -- the large-batch heuristic -- on a
        # batch-1 decode workload.
"""

_PREREQ_PREDICATE = "def is_aiter_attention_supported() -> bool:"
_PREREQ_REGISTRATION = (
    "if not is_aiter_attention_supported():\n            return\n        global _OPS_REGISTERED"
)

_PAIRS = [
    ("are_gdn_triton_kernels_available", _GDN_ANCHOR, _GDN_PATCHED),
    ("get_moe_dispatch_policy", _POLICY_ANCHOR, _POLICY_PATCHED),
]


def _ops(site: pathlib.Path) -> pathlib.Path:
    p = site / "vllm" / "_aiter_ops.py"
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(site: pathlib.Path) -> int:
    s = _ops(site).read_text()
    bad = 0
    for name, _, patched in _PAIRS:
        done = patched in s
        print(f"  {name:34}: {'PATCHED' if done else 'not patched'}")
        bad += 0 if done else 1
    return 0 if bad == 0 else 1


def apply(site: pathlib.Path) -> int:
    p = _ops(site)
    s = p.read_text()
    if _PREREQ_PREDICATE not in s:
        sys.exit("FATAL: run configs/enable_vllm_aiter_gfx90a.py first.")
    if _PREREQ_REGISTRATION not in s:
        sys.exit("FATAL: run configs/enable_aiter_ck_gemm_gfx90a.py first -- "
                 "without op registration these carve-outs have nothing to call.")
    for name, anchor, patched in _PAIRS:
        if patched in s:
            print(f"  {name} already carved out")
            continue
        n = s.count(anchor)
        if n != 1:
            sys.exit(f"FATAL: {name} anchor matched {n} times, expected 1. "
                     "Upstream changed the method; re-derive this patch.")
        s = s.replace(anchor, patched)
        print(f"  patched {name}")
    p.write_text(s)
    return 0


def revert(site: pathlib.Path) -> int:
    p = _ops(site)
    s = p.read_text()
    touched = False
    for name, anchor, patched in _PAIRS:
        if patched not in s:
            print(f"  {name} not patched; nothing to revert")
            continue
        s = s.replace(patched, anchor)
        touched = True
        print(f"  reverted {name}")
    if touched:
        p.write_text(s)
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
            print("ASSERTION FAILED: GDN / MoE-policy carve-outs not applied.")
        return rc
    if a.revert:
        return revert(site)
    return apply(site)


if __name__ == "__main__":
    sys.exit(main())
