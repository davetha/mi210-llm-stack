#!/usr/bin/env python3
"""Carve out `rocm_aiter_ops.is_enabled()` on gfx90a — the MASTER gate.

WHY IT MATTERS. `pass_manager.py:162-168` adds vLLM's ROCm fusion passes only
when `rocm_aiter_ops.is_enabled()` is true:

    if self.pass_config.fuse_norm_quant:
        if rocm_aiter_ops.is_enabled():
            self.passes += [RocmAiterRMSNormQuantFusionPass(config)]
    if self.pass_config.fuse_allreduce_rms:
        if rocm_aiter_ops.is_enabled():
            self.passes += [RocmAiterAllReduceFusionPass(config)]

`is_enabled` is `@if_aiter_supported` -> `on_mi3xx()`, so on CDNA2 it returns
None and **four fusion passes are silently never added**:

    RocmAiterRMSNormQuantFusionPass        (fuse_norm_quant)
    RocmAiterAllReduceFusionPass           (fuse_allreduce_rms)
    RocmAiterTritonAddRMSNormPadFusionPass (fuse_act_padding)
    RocmAiterSiluMulFp8GroupQuantFusionPass(fuse_act_quant)

The interesting one is **AllReduce+RMSNorm fusion**. All four allreduce ASM
objects ported to gfx90a (`all_reduce.co`, `allreduce_rmsnorm_N8192.co`,
`allreduce_rmsnorm_qnt_N8192.co`, `allreduce_layernorm_N8192.co`), and
`docs/37` dismissed them partly because "there is no XGMI for it to accelerate
anyway" — which round 31 refuted by measuring PCIe P2P at 26.98 GB/s and
+11.2% prefill. At TP=2 the per-layer collective is a fixed cost the CK GEMM
win cannot touch, and fusing it with the norm is the right shape of attack.

The other three are FP8-oriented and expected to be inert here; `docs/45`
established that `rocm_aiter_fusion.py` contains **zero** int8 patterns and
that `MatcherQuantFP8` is the only quant matcher vLLM has, so int8 RMSNorm+quant
fusion cannot happen without new pattern code regardless of this gate.

BLAST RADIUS — READ THIS BEFORE USING IT. Unlike the five narrow gates in
`enable_aiter_ops_gfx90a.py`, `is_enabled()` has ~22 consumers across 10 files,
including **two in `v1/attention/backends/rocm_aiter_fa.py`** — the flash
attention path that currently delivers 1.19-1.33x prefill. Today it sees
`None` and the stack works; flipping it to True changes branches inside a
WORKING optimization.

Therefore any round using this patch MUST assert that the existing wins still
engage:
  * AITER FA ASM still loads  -> LoadKernel count > 0 in the serverlog
  * the CK int8 GEMM still runs -> `module_gemm_a8w8` in the serverlog
A regression in either is the expected failure mode and is the reason this is
a separate script rather than another entry in the five-gate patch.

Also inert on its own: the passes it unblocks are still governed by their
`pass_config` flags (`fuse_allreduce_rms` etc.), which are independently
default-off in this deployment.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ANCHOR = """    @classmethod
    @if_aiter_supported
    def is_enabled(cls) -> bool:
        return cls._AITER_ENABLED
"""

_PATCHED = """    @classmethod
    def is_enabled(cls) -> bool:
        # gfx90a carve-out -- configs/enable_aiter_master_gate_gfx90a.py.
        # The stock decorator resolves to on_mi3xx() and excludes CDNA2, which
        # silently drops four ROCm fusion passes in pass_manager.py. Still
        # governed by VLLM_ROCM_USE_AITER and by each pass's own pass_config
        # flag, so this alone turns nothing on.
        if not is_aiter_attention_supported():
            return False
        return cls._AITER_ENABLED
"""

_PREREQ = "def is_aiter_attention_supported() -> bool:"


def _ops(site: pathlib.Path) -> pathlib.Path:
    p = site / "vllm" / "_aiter_ops.py"
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(site: pathlib.Path) -> int:
    s = _ops(site).read_text()
    done = _PATCHED in s and _ANCHOR not in s
    print(f"  is_enabled carve-out: {'PATCHED' if done else 'not patched'}")
    return 0 if done else 1


def apply(site: pathlib.Path) -> int:
    p = _ops(site)
    s = p.read_text()
    if _PREREQ not in s:
        sys.exit(
            "FATAL: is_aiter_attention_supported() missing. Run "
            "configs/enable_vllm_aiter_gfx90a.py first."
        )
    if _PATCHED in s:
        print("  already carved out")
        return 0
    n = s.count(_ANCHOR)
    if n != 1:
        sys.exit(
            f"FATAL: is_enabled anchor matched {n} times, expected 1. Upstream "
            "changed the method; re-derive rather than forcing it."
        )
    p.write_text(s.replace(_ANCHOR, _PATCHED))
    print(f"  patched {p}: is_enabled carve-out")
    print("  REMINDER: ~22 consumers including rocm_aiter_fa.py. Any round using")
    print("  this MUST assert LoadKernel > 0 and module_gemm_a8w8 present.")
    return 0


def revert(site: pathlib.Path) -> int:
    p = _ops(site)
    s = p.read_text()
    if _PATCHED not in s:
        print("  not patched; nothing to revert")
        return 0
    p.write_text(s.replace(_PATCHED, _ANCHOR))
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
            print("ASSERTION FAILED: is_enabled is not carved out.")
        return rc
    if a.revert:
        return revert(site)
    return apply(site)


if __name__ == "__main__":
    sys.exit(main())
