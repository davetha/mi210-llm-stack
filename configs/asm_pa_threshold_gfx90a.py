#!/usr/bin/env python3
"""Make AITER's ASM paged-attention batch heuristic overridable, to map its
real crossover on CDNA2.

WHAT THIS GATES. `aiter/ops/attention.py:_should_use_asm_kernel()` decides
between AITER's hand-written ASM paged attention and the HIP fallback
(`paged_attention_ll4mi`, **13.1% of decode** per `docs/45`). Its final line is:

    total_heads = num_seqs * num_heads
    return total_heads > 2 * cu_num

On a 104-CU MI210 that threshold is **208**. At TP=2 this model has 16 heads per
rank, so ASM engages only at `num_seqs >= 14`.

WHY THE CONSTANT IS SUSPECT. Scaling by CU count is reasonable; the factor `2`
is not obviously portable. It encodes how the ASM kernel's occupancy compares to
the HIP fallback's, and that ratio is an ARCHITECTURE property, not a CU-count
property. AMD calibrated it on CDNA3. Nothing suggests it transfers to CDNA2,
and nobody has checked.

WHAT WE ALREADY KNOW. `docs/50` round 56 measured the ASM path at `num_seqs 32`
(512 heads, 2.5x over the threshold) and it **won**: 1.033x throughput, 0.975x
TTFT, 0.957x TPOT, with `pa_bf16_noquant_gqa8_1tg_4w.co` confirmed loaded. So
the kernel is good on this card. The unmapped region is `num_seqs` 1..13, which
is exactly where single-stream and light-concurrency serving lives -- and every
decode measurement in this repo is batch-1, so it has never been probed.

WHAT THIS PATCH DOES. Adds an env-var escape hatch and nothing else:

    AITER_PA_ASM_FORCE=1   always use ASM   (subject to the head_size check)
    AITER_PA_ASM_FORCE=0   always use HIP
    unset                  stock heuristic, unchanged

PLACEMENT MATTERS. The hook goes AFTER the `head_size != 128` early-return and
before everything else. That check is a genuine capability limit -- the ported
gfx90a `pa` objects are all head_dim 128 -- not a heuristic, so forcing past it
would ask for a kernel that does not exist. The two quantized-KV early returns
(`high_precision == 2`, `kv_cache_tensor_dtype == torch.int8`) sit BELOW the
hook, which means `AITER_PA_ASM_FORCE=0` also overrides them; that is
deliberate, so the HIP arm is a true control even under quantized KV.

INERT BY DEFAULT. With the variable unset the function is byte-for-byte
equivalent to stock. This is an instrument, not a change in behaviour.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_ANCHOR = """    # ASM kernel only supports head_size == 128; all other head sizes use HIP.
    if head_size != 128:
        return False
"""

_PATCHED = """    # ASM kernel only supports head_size == 128; all other head sizes use HIP.
    if head_size != 128:
        return False

    # gfx90a experiment hook -- configs/asm_pa_threshold_gfx90a.py.
    # AITER_PA_ASM_FORCE=1 -> always ASM, =0 -> always HIP, unset -> stock.
    # Deliberately placed AFTER the head_size capability check (the ported
    # gfx90a pa objects are head_dim 128 only) and BEFORE the quantized-KV
    # early returns, so FORCE=0 yields a true HIP control in every case.
    # `os` is not imported at module scope in this file; keep it local.
    import os as _os

    _force = _os.environ.get("AITER_PA_ASM_FORCE")
    if _force == "1":
        return True
    if _force == "0":
        return False
"""

_REL = "aiter/ops/attention.py"


def _target(root: pathlib.Path) -> pathlib.Path:
    p = root / _REL
    if not p.is_file():
        sys.exit(f"FATAL: {p} does not exist")
    return p


def check(root: pathlib.Path) -> int:
    s = _target(root).read_text()
    done = _PATCHED in s
    print(f"  asm_pa_threshold_hook: {'PATCHED' if done else 'not patched'}")
    if not done and _ANCHOR not in s:
        print("  WARNING: anchor missing too -- upstream changed the function.")
    return 0 if done else 1


def apply(root: pathlib.Path) -> int:
    p = _target(root)
    s = p.read_text()
    if _PATCHED in s:
        print("  already patched")
        return 0
    n = s.count(_ANCHOR)
    if n != 1:
        sys.exit(f"FATAL: anchor matched {n} times, expected 1. "
                 "Upstream changed _should_use_asm_kernel; re-derive this patch.")
    p.write_text(s.replace(_ANCHOR, _PATCHED))
    print("  patched: AITER_PA_ASM_FORCE now overrides the batch heuristic")
    return 0


def revert(root: pathlib.Path) -> int:
    p = _target(root)
    s = p.read_text()
    if _PATCHED not in s:
        print("  not patched; nothing to revert")
        return 0
    p.write_text(s.replace(_PATCHED, _ANCHOR))
    print("  reverted")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/opt/python/lib/python3.14/site-packages",
                    help="site-packages root containing aiter/")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--assert-patched", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    if a.check:
        return check(root)
    if a.assert_patched:
        rc = check(root)
        if rc:
            print("ASSERTION FAILED: ASM PA threshold hook not applied.")
        return rc
    if a.revert:
        return revert(root)
    return apply(root)


if __name__ == "__main__":
    sys.exit(main())
