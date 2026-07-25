# Build & Run TurboQuant Triton on gfx90a

How to run the wave64-safe Triton TurboQuant test on the MI210. This is the
GEMM-based WHT implementation that avoids the wave64 corruption in the CUDA/HIP
kernels — see [`docs/02-turboquant-analysis.md`](../docs/02-turboquant-analysis.md).

## Prerequisites

The test needs **torch + triton** in a ROCm container. On `big` we use the
`sglang-gfx90a:test` image (torch 2.11, triton 3.7.1, ROCm 7.14):

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g \
  -v /path/to/turboquant-triton-amd:/work \
  --entrypoint bash sglang-gfx90a:test
```

## Install deps

```bash
pip install torch triton
```

No other dependencies — the Lloyd-Max solver uses a hand-rolled trapezoidal
integrator (no scipy).

## Run

```bash
cd /work
python3 test_turboquant_triton.py
```

The script auto-detects the first MI210 and runs both 3-bit (8 centroids) and
4-bit (16 centroids) quantize → dequantize round-trips at `head_dim=128`,
comparing the GPU Triton kernel against a pure-PyTorch CPU reference.

## Expected output

```
  3-bit Triton: Cosine = 0.983800  PASS
  4-bit Triton: Cosine = 0.995500  PASS

  Overall: ALL PASS
```

| Bits | Cosine similarity | Threshold | Verdict |
|------|------------------:|----------:|---------|
| 3-bit | 0.9838 | 0.95 | ✅ PASS |
| 4-bit | 0.9955 | 0.99 | ✅ PASS |

GPU (Triton on gfx90a) and CPU (PyTorch reference) match to `max_diff < 0.01`.

## Source

The test script is also mirrored in [`tests/test_turboquant_triton.py`](../tests/test_turboquant_triton.py) and in the standalone repo [`davetha/turboquant-triton-amd`](https://github.com/davetha/turboquant-triton-amd).

## Why this works where the HIP kernels don't

The Triton implementation replaces the warp-shuffle WHT butterfly with a single
GEMM (`y = x @ H`, where `H` is the precomputed Hadamard matrix). GEMMs are
block-level — no wave-level intrinsics to get wrong. Triton compiles each kernel
to wave64-native code automatically. See [`ALGORITHM.md`](https://github.com/davetha/turboquant-triton-amd/blob/main/ALGORITHM.md) for the full math.
