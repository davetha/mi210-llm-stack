# rocWMMA FlashAttention on gfx90a: the matrix cores were never idle

**Date**: 2026-07-27
**Hardware**: 2x AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each
**Software**: ROCm 7.14.0, HIP 7.14.60850, llama.cpp `67b9b0e`, rocWMMA 2.0.0 and 2.2.1
**Model**: `Huihui-Qwen3-Coder-Next-abliterated.i1-Q4_K_M` (80B MoE, GGUF)

`docs/01-gfx90a-architecture-constraints.md` says to build llama.cpp with
`-DGGML_HIP_ROCWMMA_FATTN=OFF`, and the production `llama-rocm714:latest` image
was built that way. The open question was whether that flag was leaving the
MI210's matrix cores unused during attention — if ggml's FlashAttention were
running on plain vector ALUs, turning rocWMMA ON should have been a large
prefill win.

**That premise is wrong, and it is worth being blunt about why, because "we
tried it and it was slower" invites someone to retry it with different tuning.
There is nothing to tune.**

llama.cpp already uses the matrix cores for attention on gfx90a, through its
`fattn-mma-f16` path. `GGML_HIP_ROCWMMA_FATTN` does not *enable* MFMA. It
**selects a different kernel** — the older `fattn-wmma-f16` — which emits the
same `v_mfma_f32_16x16x16f16` instruction with worse blocking. The flag is a
choice between two MFMA kernels, and rocWMMA picks the worse one. Prefill is
18-26% slower as a result.

So this question is closed, not merely answered unfavourably. The only thing
that would reopen it is upstream rewriting `fattn-wmma-f16` itself.

## The measurement

Both arms ran on the same idle MI210 (GPU 1), same model, same flags
(`-ngl 999 -c 262144 -np 8 -fa on -ctk q8_0 -ctv q8_0 -ub 2048 --jinja`),
with unique UUID-seeded prompts so nothing could hit a prefix cache.

| | rocWMMA OFF (production) | rocWMMA 2.2.1 ON | delta |
|---|---|---|---|
| 16k prefill | 9,125 / 9,282 ms | 11,186 / 11,082 ms | **+21% time** |
| 16k prefill rate | 1,756 / 1,767 tok/s | 1,433 / 1,446 tok/s | **-18.3%** |
| 24k prefill | 14,253 / 14,328 ms | 19,126 / 19,216 ms | **+34% time** |
| 24k prefill rate | 1,715 / 1,711 tok/s | 1,275 / 1,271 tok/s | **-25.7%** |
| generation (256 tok) | 74.22 tok/s | 74.68 tok/s | +0.6% (noise) |
| 16k correctness | `ACKNOWLEDGED` | `ACKNOWLEDGED` | both correct |

rocWMMA is not *broken* here — output is correct. It is simply slower, and the
penalty grows with context length, which is the opposite of what you want.

The result was reproduced **three times by two people on independent runs**,
which is why it is stated this firmly:

| run | 16k | 24k |
|---|---|---|
| this investigation | −18.3% | −25.7% |
| second run, separate cards | −19.8% | −26.3% |
| third run, separate harness | −19.8% | −26.3% |

All arms passed the `ACKNOWLEDGED` check, so no arm was fast because it was
computing garbage.

Generation is unaffected because decode uses the vector kernel
(`BEST_FATTN_KERNEL_VEC`), which neither path touches.

## Why: both paths emit the same instruction

The premise was that rocWMMA maps to MFMA and the OFF build therefore does not.
That is false. Disassembling the **production** `libggml-hip.so` — the one built
with `GGML_HIP_ROCWMMA_FATTN=OFF` — shows its FlashAttention kernels are already
packed with matrix instructions:

```
$ python3 configs/scan_fatbin_mfma.py /src/build/bin/libggml-hip.so.0.17.0 flash_attn_ext_f16
...
co089.elf: 12 matching kernels, 2592 MFMA
      2592  v_mfma_f32_16x16x16f16
co093.elf: 16 matching kernels, 2832 MFMA
      2832  v_mfma_f32_16x16x16f16

total MFMA in kernels matching 'flash_attn_ext_f16': 22848
```

**22,848 MFMA instructions in the attention kernels of the build that supposedly
has matrix acceleration disabled.** The kernels carrying them are the
`fattn-mma-f16` instances:

```
T void flash_attn_ext_f16<112, 112, 16, 4, false, false>(...)
T void flash_attn_ext_f16<128, 128, 16, 4, false, false>(...)
```

The rocWMMA build's kernel emits **the identical instruction**:

```
   2008 v_mfma_f32_16x16x16f16
```

So the choice is not "matrix cores vs vector ALUs". Both paths use
`v_mfma_f32_16x16x16f16`. The flag only decides *which MFMA kernel* runs, and
the one rocWMMA selects is worse on this shape.

The CDNA2 pass-count concern (`V_MFMA_F32_16X16X16*` is 8-pass on gfx90a vs
4-pass on gfx942, needing 11 wait states rather than 7) is real but is not the
differentiator: both arms issue the same 8-pass instruction, so it cancels out
of the comparison.

## Why: the flag changes routing, not capability

`ggml_cuda_get_best_fattn_kernel()` in `ggml/src/ggml-cuda/fattn.cu` tests the
rocWMMA predicate *before* the AMD MFMA predicate:

```c
// line ~503 — checked first
if (ggml_cuda_should_use_wmma_fattn(cc) && ...) {
    return BEST_FATTN_KERNEL_WMMA_F16;   // legacy Volta-era WMMA kernel
}

// line ~511 — only reached when the above is false
if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && ...) {
    return BEST_FATTN_KERNEL_MMA_F16;    // modern MFMA kernel
}
```

`amd_mfma_available()` returns true for **all** CDNA including gfx90a. So with
rocWMMA OFF, gfx90a falls through to the modern `fattn-mma-f16` path — the good
one. Turning rocWMMA ON *diverts* prefill to `fattn-wmma-f16`, llama.cpp's
older, Volta-derived kernel. Upstream's ordering is deliberate.

## The trap: `-DGGML_HIP_ROCWMMA_FATTN=ON` can be a silent no-op

Building with `-DGGML_HIP_ROCWMMA_FATTN=ON` against the rocWMMA that ships with
ROCm 7.1 (**version 2.0.0**) changes nothing at all on gfx90a. Measured, it is
identical to the OFF build within noise (16k: 1,740/1,760 tok/s; 24k:
1,710/1,708 tok/s).

`ggml/src/ggml-cuda/fattn-wmma-f16.cuh` blacklists exactly 2.0.0 on CDNA, via
**two independent gates** keyed on the same version test:

```c
// gate 1, line ~10 — controls whether the kernel BODY is compiled at all
#if defined(CDNA) && (ROCWMMA_VERSION_MAJOR < 2 || ROCWMMA_VERSION_MINOR > 0 || ROCWMMA_VERSION_PATCH > 0)
#define GGML_USE_WMMA_FATTN
#elif defined(CDNA)
#warning "rocwmma fattn on CDNA is broken on rocwmma v2.0.0, expect degraded performance"
#endif

// gate 2, line ~36 — controls whether the host DISPATCHES to that kernel
static bool ggml_cuda_should_use_wmma_fattn(const int cc) { ... }
```

With 2.0.0 both gates stay shut, `rocwmma.hpp` is never even included, and the
build is byte-for-byte equivalent in behaviour to OFF. **The flag reports as ON
in `CMakeCache.txt` while doing nothing** — check the rocWMMA version, not the
CMake flag.

Lifting only gate 2 (dispatch) without gate 1 (codegen) produces a binary that
dispatches to a kernel with no device code:

```
fattn-wmma-f16.cu:514: ERROR: HIP kernel flash_attn_ext_f16 has no device code compatible with HIP arch 1300
rocdevice.cpp:3580: Callback: Queue aborting with error : HSA_STATUS_ERROR_EXCEPTION
```

(server exits 139). Both gates must move together.

## Getting a rocWMMA that actually works

Ubuntu's `librocwmma-dev` (7.1.0-0ubuntu1, rocWMMA 2.0.0) is **incomplete** — it
ships five top-level headers and omits the entire `internal/` directory that
`rocwmma.hpp` includes on its first line:

```
/usr/include/rocwmma/rocwmma.hpp:29:10: fatal error: 'internal/accessors.hpp' file not found
```

It is only usable because `ggml-cuda/vendors/hip.h` includes just
`<rocwmma/rocwmma-version.hpp>`, the one lightweight header the package does
ship. Any build that genuinely enables the path will fail to compile.

The working configuration is rocWMMA **2.2.1** from source, which supports
gfx90a explicitly (`ROCWMMA_ARCH_GFX90A` in `internal/config.hpp`) and passes
upstream's version guard legitimately (`MINOR > 0`), so **no llama.cpp source
patch is needed**:

```bash
git clone --depth 1 --branch develop https://github.com/ROCm/rocWMMA.git
rm -rf /usr/include/rocwmma
cp -r rocWMMA/library/include/rocwmma /usr/include/rocwmma
sed -e 's/@rocwmma_VERSION_MAJOR@/2/' -e 's/@rocwmma_VERSION_MINOR@/2/' \
    -e 's/@rocwmma_VERSION_PATCH@/1/' \
    rocWMMA/library/include/rocwmma/internal/rocwmma-version.hpp.in \
  > /usr/include/rocwmma/internal/rocwmma-version.hpp
cp /usr/include/rocwmma/internal/rocwmma-version.hpp /usr/include/rocwmma/rocwmma-version.hpp
```

That build compiles clean, runs correctly, and loses. No rocWMMA source changes
were needed or made.

## Conclusion

Keep `-DGGML_HIP_ROCWMMA_FATTN=OFF` on gfx90a. `llama-coder-80b` stays on
`llama-rocm714:latest`.

The reason to keep it off is not the one in
`docs/01-gfx90a-architecture-constraints.md` (which claims the fragments do not
map and FA "crashes or silently produces wrong results" — with rocWMMA 2.2.1 it
does neither). The real reason is that ggml already reaches the matrix cores on
CDNA2 through `fattn-mma-f16`, and rocWMMA can only replace that with a slower
kernel that uses the same instruction.

Corollary worth remembering: **a build flag that appears ON in `CMakeCache.txt`
is not evidence it did anything.** Verify at the instruction level.

## Reproducing

Benchmark harness: `benchmarks/bench_rocwmma_fattn.py`, one arm per build,
each against an otherwise idle card:

```bash
python3 bench_rocwmma_fattn.py http://127.0.0.1:8093 baseline
```

Instruction-level check: `configs/scan_fatbin_mfma.py <libggml-hip.so> flash_attn_ext_f16`.

Each arm needs an otherwise-idle card: on a two-card host, pin the two builds to
different GPUs rather than running them back to back on one, and do not
benchmark while another model is resident on the same card.

The production chat stack **was** interrupted during this work, twice — once
unintentionally (a second container was started on a port already in use by the
benchmark, and the model it displaced was not brought back), and once
deliberately with Dave's authorisation to free both cards. Neither interruption
affected the measurements, but the earlier draft of this section claimed the
stack was never stopped, which was wrong. If you repeat this, assume you will
take the chat down and plan the restart, rather than assuming you can bench
alongside it.

The image built for this investigation, `llama-rocm714-wmma:latest`, **has since
been deleted**, along with its build container. Do not go looking for it. The
rocWMMA 2.2.1 vendoring recipe in "Getting a rocWMMA that actually works" above
is self-contained and is now the only copy — rebuild from that if this ever
needs revisiting. It should not: see the conclusion.
