# Build FlashAttention (CK backend) on gfx90a

Standalone ROCm FlashAttention 2.8.3 works on the MI210 via the **Composable Kernel (CK) backend** — the same FA that llama.cpp's rocWMMA path can't use (CDNA3+ only). This builds the library standalone, useful for vLLM / SGLang / custom kernels.

## Prerequisites

Use the `sglang-gfx90a:test` Docker image (torch 2.11, ROCm 7.14, CK headers):

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g \
  --entrypoint bash sglang-gfx90a:test
```

## Clone — use `origin/main`, NOT a pinned tag

```bash
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
```

> **CRITICAL:** You MUST build from **`origin/main`** (commit `b3ae4966` or later).
> Do **NOT** pin to `3cea2fb` — that tag's CK backend is too old for ROCm 7.14
> and the build fails on missing CK ops.

## Build

```bash
GPU_ARCHS=gfx90a MAX_JOBS=32 python3 setup.py install
```

| Flag | Value | Why |
|------|-------|-----|
| `GPU_ARCHS` | `gfx90a` | Compile only for MI210. |
| `MAX_JOBS` | **32** (not 8) | The machine has 48 cores and 499 GB RAM — the default `MAX_JOBS=8` is needlessly slow. |

### Build time

~60 minutes. The CK backend compiles **2926 kernel objects** (one per shape × dtype × layout combination). This is a one-time cost; subsequent incremental builds are fast if you set up ccache (see [`setup-ccache-docker.md`](setup-ccache-docker.md)).

## Verification

```python
import torch
from flash_attn import flash_attn_func

# causal flash attention, head_dim=64 (mimo's dimension)
B, S, H, D = 2, 512, 8, 64
q = torch.randn(B, S, H, D, dtype=torch.float16, device='cuda')
k = torch.randn_like(q)
v = torch.randn_like(q)

out = flash_attn_func(q, k, v, causal=True)
ref = torch.nn.functional.scaled_dot_product_attention(
    q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
).transpose(1,2)

max_diff = (out - ref).abs().max().item()
print(f"max_diff = {max_diff:.6f}")
assert max_diff < 0.01, f"FA output mismatch: {max_diff}"
print("FlashAttention on gfx90a: PASS")
```

Expected: `max_diff < 0.01`, `PASS`.

## Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CK op not found` at build | pinned old tag (`3cea2fb`) | use `origin/main` |
| OOM during build | `MAX_JOBS` too high for RAM | reduce to `MAX_JOBS=16` |
| `gfx90a not in GPU_ARCHS` | stale CMake cache | `rm -rf build/` and reconfigure |
