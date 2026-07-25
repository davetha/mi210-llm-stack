# KTransformers POC Results on MI210

POC detail behind the numbers in [`benchmarks/`](../benchmarks/README.md). KTransformers (the CPU+GPU hybrid MoE server) was evaluated as a potential replacement for the llama.cpp CPU-split MoE path. **Verdict: cannot serve any model on this hardware — three independent blockers.**

---

## What worked

| Component | Result |
|---|---|
| `kt-kernel` 0.6.4 build | ✅ Compiles cleanly (ROCm 7.14, gfx90a) |
| GPU HIP matmul | ✅ Works (~60 iters/s for 4096³ bf16) |

The HIP matmul sanity check passes — the GPU kernels that *do* ship in `kt-kernel` run correctly on MI210. So the build toolchain and the basic GPU compute path are fine.

---

## Three independent blockers

### Blocker 1 — `sgl-kernel` is CUDA-only (server import fails)

KTransformers' server component imports `sgl-kernel`. The upstream `sgl-kernel` package is **CUDA-only** — it has no ROCm / HIP build path. Importing it on a gfx90a box fails at the shared-object load.

This is the same `sgl-kernel` that SGLang needs, but the situation here is worse: for SGLang we could patch `setup_rocm.py` to add gfx90a as a build target (see [`docs/05-sglang-on-gfx90a.md`](05-sglang-on-gfx90a.md)) and build a working ROCm wheel. KTransformers consumes `sgl-kernel` as a prebuilt dependency and does not expose the same build hook, so the same patch strategy doesn't apply cleanly.

### Blocker 2 — EPYC 74F3 is AVX2-only (CPU expert kernels need AVX-512 / AMX)

KTransformers' CPU-side expert kernels (the whole point of the project — offload MoE experts to host RAM and compute them on CPU) require **AVX-512** or **AMX**. The EPYC 74F3 is a Zen3 Milan part: it has **AVX2, but not AVX-512 and not AMX**.

This is the fundamental architectural mismatch. KTransformers' CPU MoE path is designed around Intel Xeon / AMD Zen4+ servers with wide SIMD. On a Zen3 box the expert kernels either don't compile or fall back to a scalar path that is far too slow to be useful.

| ISA | EPYC 74F3 (Zen3) | KTransformers needs |
|---|---|---|
| AVX2 | ✅ | ✅ |
| AVX-512 | ❌ | ✅ |
| AMX | ❌ | ✅ (preferred) |

### Blocker 3 — DeepSeek-V2-Lite geometry rejected (`1408 % 256 ≠ 0`)

Even setting aside the above two, KTransformers rejects the model geometry of DeepSeek-V2-Lite. The GPU kernel that handles the attention/expert dispatch requires the hidden dimension to be a multiple of 256; DeepSeek-V2-Lite has a dimension of **1408**, and `1408 % 256 = 128 ≠ 0`. The model is rejected at load.

This is a hard kernel-alignment requirement, not a tunable.

---

## Recommendation: keep using llama.cpp

All three blockers are independent — fixing any one of them still leaves the model unservable:

| Path | Blocker |
|---|---|
| Try the server | sgl-kernel CUDA-only |
| Patch sgl-kernel + run server | EPYC lacks AVX-512/AMX |
| Try a different model with aligned geometry | still needs the server + AVX-512 |

The current production stack (llama.cpp with `-ot` CPU split, served via llama-swap) avoids all three:

- **No sgl-kernel dependency** — llama.cpp is self-contained.
- **AVX2 is enough** — llama.cpp's CPU kernels are tuned for AVX2 and work fine on Zen3.
- **No geometry constraint** — llama.cpp has no 256-multiple requirement on the hidden dim.

For the mimo (230B MoE) production workload, llama.cpp remains the only viable engine on this hardware. See [`docs/04-moe-engine-survey.md`](04-moe-engine-survey.md) for the full engine comparison and [`docs/08-platform-gaps-gfx90a.md`](08-platform-gaps-gfx90a.md) for the platform-gap context.
