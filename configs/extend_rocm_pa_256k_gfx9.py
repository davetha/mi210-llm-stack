"""Extend ROCm custom paged attention past 128k on gfx9, removing the decode cliff.

`docs/23` measures the symptom: setting `--max-model-len` above 128k costs
**10x decode on every request**, including short ones. The mechanism is that the
gfx9 gate is evaluated at CUDA-graph capture time against the *configured* max,
so a 256k server bakes the Triton fallback into the graph and replays it forever.

`docs/25` listed three candidate fixes and warned that the 128k ceiling
"probably exists for a reason -- likely partition-count or temp-buffer sizing",
and that raising it blindly could produce silently wrong attention above 128k.

**It is neither of those, and it is not silent.** The reason is written in the
source, in `csrc/rocm/attention.cu`:

    const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
    // reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
    // (partition size) = 128K context length
    switch (npar_loops) {
      case 1: LAUNCH_CUSTOM_REDUCTION(1); break;
      ...
      case 8: LAUNCH_CUSTOM_REDUCTION(8); break;
      default:
        TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
    }

8 x 64 x 256 = 131,072 = exactly 128K. The ceiling is a **dispatch table with
eight entries** -- a set of missing template instantiations, nothing more.

Three things follow, and they change the risk assessment completely:

1. **The failure mode is loud.** Raising the Python gate without extending the
   switch does not corrupt attention; it hits `TORCH_CHECK(false, ...)` and
   aborts the request with the offending `npar_loops` value. That is the
   opposite of the silent-garbage risk `docs/25` feared, and it means the two
   halves of this patch can be applied and tested independently.

2. **NPAR_LOOPS=16 is already proven to compile and run.** The RDNA launcher
   (`paged_attention_custom_launcher_navi`) instantiates the same
   `paged_attention_ll4mi_reduce_kernel` template at 1..16 today. Note this does
   NOT mean RDNA reaches 256k -- its warps are 32 wide, so 16 x 32 x 256 is also
   131,072. Both architectures cap at 128K; they just arrive there differently.
   What it establishes is that the template body is valid at 16.

3. **The resource cost is small and checkable.** NPAR_LOOPS sizes exactly one
   shared array and three register arrays:

       __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];
       int   valid_partition[NPAR_LOOPS];
       float reg_max_logit[NPAR_LOOPS];
       float rescaled_exp_sum[NPAR_LOOPS];

   At NPAR_LOOPS=16, WARP_SIZE=64: shared memory goes 2,048 -> 4,096 bytes
   against gfx90a's 64 KB LDS per workgroup, and the register arrays go 24 ->
   48 VGPRs against 256 architected VGPRs. Neither is close to a limit, though
   the extra VGPRs may cost some occupancy -- which is a performance question to
   measure, not a correctness one.

So this patch does two things:

  * adds `case 9` .. `case 16` to the **gfx9** launcher's reduction switch,
    raising its ceiling to 16 x 64 x 256 = 262,144 = 256K;
  * raises the `max_seq_len <= 128 * 1024` term in
    `use_rocm_custom_paged_attention` to `256 * 1024`, for the `_ON_GFX9`
    branch only.

The RDNA branch of that gate is deliberately left alone. Its kernel genuinely
tops out at 128K for the warp-width reason above, so raising it would convert a
working fallback into a `TORCH_CHECK` abort.

WHAT THIS DOES NOT ESTABLISH
----------------------------
That the results above 128k are *numerically correct*. The instantiation is
valid and the buffers are sized right, but no reference comparison has been run
at 200k+. `docs/25` is right that this needs numeric verification and not just a
throughput number, and that verification is the acceptance criterion here --
a faster wrong answer is worse than the slow right one we have.

Run `tests/test_rocm_pa_256k.py` before trusting any long-context result from a
patched build.

REBUILD REQUIRED
----------------
This edits a `.cu` file, so unlike the pure-Python patches in this directory it
needs vLLM's ROCm extension rebuilt. With ccache primed that is minutes, not
the full image build -- see `guides/setup-ccache-docker.md`.

    python3 configs/extend_rocm_pa_256k_gfx9.py --vllm-src /workspace/vllm
    python3 configs/extend_rocm_pa_256k_gfx9.py --check    # verify, no writes
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The eight-case switch, as it appears in the gfx9 launcher. Anchored on the
# comment because that string is unique in the file -- matching on `case 8:`
# alone would also hit the RDNA launcher, which must NOT be extended.
_CU_ANCHOR = """  // reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
  // (partition size) = 128K context length
"""

_CU_REPLACEMENT = """  // reduction kernel supports upto 16 NPAR_loops * 64 (warp_size) * 256
  // (partition size) = 256K context length
  //
  // PATCHED (mi210-llm-stack): upstream stops at 8, which is exactly 128K and
  // is what forces long-context decode onto the Triton fallback. Cases 9..16
  // are added below. The template body is unchanged and is already instantiated
  // at 16 by the RDNA launcher; at WARP_SIZE=64 this costs 4 KB of LDS and
  // ~48 VGPRs. See configs/extend_rocm_pa_256k_gfx9.py.
"""

# Inserted immediately before the `default:` arm of the gfx9 switch.
_NEW_CASES = "".join(
    f"    case {n}:\n      LAUNCH_CUSTOM_REDUCTION({n});\n      break;\n"
    for n in range(9, 17)
)

_PY_OLD = "            and max_seq_len <= 128 * 1024\n            and sinks is None"
_PY_NEW = (
    "            # PATCHED (mi210-llm-stack): 256K, matching the reduction\n"
    "            # switch extended to npar_loops=16 in csrc/rocm/attention.cu.\n"
    "            # 16 * 64 (warp) * 256 (partition) = 262,144.\n"
    "            # The _ON_GFX1X branch below is deliberately NOT raised: its\n"
    "            # warps are 32 wide, so its 16 cases also top out at 128K.\n"
    "            and max_seq_len <= 256 * 1024\n"
    "            and sinks is None"
)


def already_patched(text: str) -> bool:
    return "PATCHED (mi210-llm-stack)" in text


def patch_cu(text: str) -> str:
    if already_patched(text):
        raise SystemExit("attention.cu already patched -- refusing to double-apply")
    if _CU_ANCHOR not in text:
        raise SystemExit(
            "anchor comment not found in attention.cu; upstream changed the "
            "reduction dispatch. Re-read the switch before forcing this."
        )
    text = text.replace(_CU_ANCHOR, _CU_REPLACEMENT, 1)

    # Extend only the FIRST switch after the anchor -- that is the gfx9 one.
    # rindex-style search would land on the RDNA launcher instead.
    start = text.index(_CU_REPLACEMENT)
    tail = text[start:]
    m = re.search(r"\n(    default:\n      TORCH_CHECK\(false, \"Unsupported npar_loops: \", npar_loops\);)", tail)
    if not m:
        raise SystemExit("could not locate the gfx9 switch default arm")
    at = start + m.start() + 1
    return text[:at] + _NEW_CASES + text[at:]


def patch_py(text: str) -> str:
    if "PATCHED (mi210-llm-stack)" in text:
        raise SystemExit("rocm.py already patched -- refusing to double-apply")
    # Both branches contain `max_seq_len <= 128 * 1024`, but only the gfx9 one
    # is followed immediately by `and sinks is None` -- gfx1x interposes
    # `alibi_slopes` and `kv_cache_dtype` checks first. So this two-line anchor
    # already selects gfx9 uniquely, and the count assertion is what proves it
    # rather than a positional `replace(..., 1)` that would silently pick the
    # wrong branch if upstream reordered the terms.
    n = text.count(_PY_OLD)
    if n != 1:
        raise SystemExit(
            f"expected the gfx9 128k anchor exactly once in rocm.py, found {n}. "
            "Upstream reordered the gate terms; re-read both branches before "
            "forcing this -- patching the gfx1x branch would turn a working "
            "fallback into a TORCH_CHECK abort."
        )
    return text.replace(_PY_OLD, _PY_NEW, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-src", default="/workspace/vllm",
                    help="vLLM source checkout (for csrc/rocm/attention.cu)")
    ap.add_argument("--site-packages", default=None,
                    help="installed vllm package dir (for platforms/rocm.py); "
                         "defaults to the importable one")
    ap.add_argument("--check", action="store_true", help="report state, write nothing")
    args = ap.parse_args()

    cu = Path(args.vllm_src) / "csrc" / "rocm" / "attention.cu"
    if args.site_packages:
        py = Path(args.site_packages) / "platforms" / "rocm.py"
    else:
        import vllm  # noqa: PLC0415
        # __file__ is Optional on namespace packages; a vllm without one is not
        # a thing we can patch, and saying so beats crashing on None.
        vllm_file = getattr(vllm, "__file__", None)
        if vllm_file is None:
            print("vllm has no __file__ (namespace package?); pass --site-packages",
                  file=sys.stderr)
            return 2
        py = Path(vllm_file).parent / "platforms" / "rocm.py"

    for p in (cu, py):
        if not p.is_file():
            print(f"MISSING: {p}", file=sys.stderr)
            return 2

    if args.check:
        for p in (cu, py):
            state = "PATCHED" if "PATCHED (mi210-llm-stack)" in p.read_text() else "stock"
            print(f"{state:8} {p}")
        return 0

    # Patch each file independently. They are separate halves -- the switch and
    # the gate -- and a rerun after one succeeded should finish the job rather
    # than abort on the half already done.
    if already_patched(cu.read_text()):
        print(f"already patched {cu}")
    else:
        cu.write_text(patch_cu(cu.read_text()))
        print(f"patched {cu}  (reduction switch -> npar_loops 16, 256K)")

    if already_patched(py.read_text()):
        print(f"already patched {py}")
    else:
        py.write_text(patch_py(py.read_text()))
        print(f"patched {py}  (gfx9 gate -> 256 * 1024; gfx1x left at 128k)")

    print("\nREBUILD the ROCm extension, then run tests/test_rocm_pa_256k.py.")
    print("A throughput win with no numeric check is NOT an acceptance signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
