"""Let vLLM dispatch to AITER on gfx90a (MI210 / CDNA2).

`enable_gfx90a_asm_paths.py` unlocks AITER's own ASM fast paths. This script
fixes the layer above it: vLLM refuses to route anything to AITER on an MI210,
so those kernels are unreachable from the serving stack no matter which
`VLLM_ROCM_USE_AITER_*` flags are set.

The block is not the documented one. `_aiter_ops.py` states in prose that its
check functions verify "(1) platform is ROCm, (2) device arch is gfx9, and
(3) aiter library is installed", and gfx90a is gfx9. But the code implementing
that check calls `on_mi3xx()`, which is

    _ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])

so it excludes CDNA2. Because `if_aiter_supported` returns `None` -- not
`False` -- when the check fails, every `is_*_enabled()` query answers falsy and
vLLM silently drops ROCM_AITER_FA and ROCM_AITER_UNIFIED_ATTN from the backend
candidate list. The failure is invisible: the engine logs a normal-looking
"Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN',
'TRITON_ATTN']" and serves happily from the Triton path, with AITER never
touched. Setting VLLM_ROCM_USE_AITER=1 changes nothing at all.

Sites patched
-------------
`vllm/_aiter_ops.py`, `is_aiter_found_and_supported()` -- the master gate that
every `@if_aiter_supported` check function funnels through.

`vllm/v1/attention/backends/rocm_aiter_fa.py`,
`AiterFlashAttentionBackend.supports_compute_capability()` -- a second,
independent `on_mi3xx()` that rejects the backend during validation even once
the master gate admits it.

Both become `on_gfx9()`, which is what the surrounding documentation already
claims the condition is.

Deliberately NOT patched
------------------------
`_aiter_ops.py:is_linear_hipbmm_enabled` and
`model_executor/kernels/linear/scaled_mm/pytorch.py` keep their `on_mi3xx()`
checks. Both guard FP8 scaled-GEMM paths, and CDNA2 has no FP8 ALU, so widening
them would route gfx90a into kernels it cannot execute. `on_mi3xx()` is the
correct condition there; it is only wrong as a stand-in for "supports AITER".

This does not by itself select an AITER backend -- it only makes one selectable.
See benchmarks/vllm-aiter-asm-gfx90a.md for the env vars that pick it, and note
that `VLLM_ATTENTION_BACKEND` is not a recognised variable in vLLM 0.23.x;
backend choice goes through `--attention-config`.

    python enable_vllm_aiter_gfx90a.py [--revert] [--check]
"""
import argparse
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
AITER_OPS = f"{SITE}/vllm/_aiter_ops.py"
AITER_FA = f"{SITE}/vllm/v1/attention/backends/rocm_aiter_fa.py"

# (path, unpatched, patched, expected occurrences)
PATCHES = [
    (
        AITER_OPS,
        "        from vllm.platforms.rocm import on_mi3xx\n\n        return on_mi3xx()\n",
        "        from vllm.platforms.rocm import on_gfx9\n\n        return on_gfx9()\n",
        1,
    ),
    (
        AITER_FA,
        "        from vllm.platforms.rocm import on_mi3xx\n",
        "        from vllm.platforms.rocm import on_gfx9\n",
        1,
    ),
    (
        AITER_FA,
        "        return on_mi3xx()\n",
        "        return on_gfx9()\n",
        1,
    ),
]


def apply(revert: bool = False, check: bool = False) -> int:
    failures = 0
    for path, old, new, want in PATCHES:
        if revert:
            old, new = new, old
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError as exc:
            print(f" ERROR   {path}: {exc}")
            failures += 1
            continue

        n_unpatched = text.count(old)
        n_patched = text.count(new)
        name = path.rsplit("/", 1)[-1]

        if check:
            state = "unpatched" if n_unpatched else "patched"
            print(f" {state:>9}  {name}  (want {want}, "
                  f"found {n_unpatched} unpatched / {n_patched} patched)")
            if n_unpatched + n_patched != want:
                failures += 1
            continue

        if n_patched == want and n_unpatched == 0:
            print(f" already   {name}")
            continue
        if n_unpatched != want:
            print(f" ERROR    {name}: expected {want} occurrences, found "
                  f"{n_unpatched}. Upstream code moved -- reread the source.")
            failures += 1
            continue

        with open(path, "w") as fh:
            fh.write(text.replace(old, new))
        print(f" patched   {name}")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true",
                    help="restore the on_mi3xx() gates")
    ap.add_argument("--check", action="store_true",
                    help="report state without writing")
    args = ap.parse_args()

    failures = apply(revert=args.revert, check=args.check)
    if failures:
        print(f"\n{failures} site(s) did not match as expected.")
        return 1
    print("\nOK. Remember: .pyc caches are keyed on mtime, so the next import "
          "picks this up automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
