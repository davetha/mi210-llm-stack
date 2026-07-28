"""Let vLLM's Triton INT8 MoE run on ROCm CDNA, instead of refusing outright.

Serving an INT8 W8A8 MoE checkpoint on an MI210 fails at startup:

    NotImplementedError: No Int8 MoE backend supports the deployment configuration.

There is exactly one Int8 MoE backend in vLLM -- TRITON -> TritonExperts
(`fused_moe/oracle/int8.py`) -- and it refuses. The refusal is not a hardware
limit. In `fused_moe/experts/triton_moe.py`:

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        # INT8 requires at least 7.5 (Turing).
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        )

`current_platform.is_cuda()` is **False on ROCm**. Note the device check
immediately above it in the same class uses `is_cuda_alike()`, which *does*
accept ROCm:

    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike() or current_platform.is_xpu()

So the kernel is declared runnable on ROCm, and then its INT8 scheme is gated
behind a CUDA-only predicate carrying a Turing compute-capability comment that
has no meaning on AMD. The effect is that INT8 MoE is unavailable on every AMD
GPU, including ones with native INT8 matrix hardware.

CDNA2 (gfx90a, MI210/MI250) has `v_mfma_i32_16x16x16i8` and peaks at **181
TOPS INT8 -- the same as its bf16 peak**. That makes INT8 the natural
quantization for this hardware: it halves weight-memory traffic against bf16
with no arithmetic penalty. CDNA3 (gfx942) has INT8 MFMA too, at K=32.

So the patch admits ROCm on gfx9 specifically, rather than all of ROCm:

    device_supports_int8 = (
        (current_platform.is_cuda()
         and current_platform.has_device_capability((7, 5)))
        or (current_platform.is_rocm() and on_gfx9())
    )

`on_gfx9()` rather than a blanket `is_rocm()` because INT8 MFMA is a CDNA/gfx9
feature; RDNA parts reach this code path too and were never in scope here. If
someone benchmarks INT8 MoE on RDNA and it works, widening this is a one-line
change -- but it should be a measured decision, not an assumption.

**This patch enables a code path; it does not prove the path is correct.** The
Triton INT8 MoE kernel has presumably never run on AMD, since this gate has
always refused it. Verify numerically before trusting throughput from it --
`benchmarks/matrix/bench_matrix.py` runs a correctness probe before timing for
exactly this reason, and a fast-but-wrong backend must be reported as broken
rather than as a result.

    python enable_int8_moe_rocm.py [--revert] [--check]
"""
import argparse
import sys

SITE = "/opt/python/lib/python3.14/site-packages"
TRITON_MOE = f"{SITE}/vllm/model_executor/layers/fused_moe/experts/triton_moe.py"

_ANCHOR = """        # INT8 requires at least 7.5 (Turing).
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        )
"""

_PATCHED = """        # INT8 requires at least 7.5 (Turing) on NVIDIA.
        #
        # On ROCm this used to be `current_platform.is_cuda()`, which is False,
        # so INT8 MoE was refused on every AMD GPU and serving an INT8 W8A8 MoE
        # checkpoint died with "No Int8 MoE backend supports the deployment
        # configuration". That gate was a CUDA compute-capability test applied
        # to hardware it does not describe -- note _supports_current_device()
        # in this same class already accepts ROCm via is_cuda_alike().
        #
        # CDNA has native INT8 matrix hardware: gfx90a provides
        # v_mfma_i32_16x16x16i8 and peaks at 181 TOPS INT8, equal to its bf16
        # peak, which makes INT8 the natural quantization on CDNA2 -- half the
        # weight traffic of bf16 at the same arithmetic throughput.
        #
        # Scoped to gfx9 rather than all of ROCm because INT8 MFMA is a
        # CDNA/gfx9 feature; widening to RDNA should be a measured decision.
        _rocm_int8 = False
        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx9

            _rocm_int8 = on_gfx9()
        device_supports_int8 = (
            current_platform.is_cuda()
            and current_platform.has_device_capability((7, 5))
        ) or _rocm_int8
"""

PATCHES = [(TRITON_MOE, _ANCHOR, _PATCHED, 1)]


def apply(revert: bool = False, check: bool = False) -> int:
    failures = 0
    for path, old, new, want in PATCHES:
        if revert:
            old, new = new, old
        try:
            text = open(path).read()
        except OSError as exc:
            print(f" ERROR    {path}: {exc}")
            failures += 1
            continue
        name = path.rsplit("/", 1)[-1]
        n_new, n_old = text.count(new), text.count(old)

        if check:
            print(f" {'patched' if n_new else 'unpatched':>9}  {name} "
                  f"(unpatched {n_old} / patched {n_new})")
            if n_old + n_new != want:
                failures += 1
            continue
        if n_new == want and n_old == 0:
            print(f" already    {name}")
            continue
        if n_old != want:
            print(f" ERROR      {name}: expected {want} occurrence(s), found "
                  f"{n_old}. Upstream moved -- reread the source, do not force.")
            failures += 1
            continue
        open(path, "w").write(text.replace(old, new, want))
        print(f" patched    {name}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    failures = apply(revert=args.revert, check=args.check)
    if failures:
        print(f"\n{failures} site(s) did not match as expected.")
        return 1
    if not args.check:
        print("\nOK. This enables a path that has probably never run on AMD -- "
              "verify numerically before quoting any throughput from it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
