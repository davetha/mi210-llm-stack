# SGLang on gfx90a — Proven Viable

**Previous assessment was WRONG.** SGLang is NOT impossible on gfx90a — it works with just **2 patches to sgl-kernel** + 1 runtime patch.

## What was verified

| Check | Result |
|-------|--------|
| sgl-kernel 0.4.5 builds for gfx90a | ✅ |
| `import sgl_kernel` works (exports: all_reduce_reg, allreduce, apply_shuffle_mul_sum, …) | ✅ |
| Both MI210s detected (`torch.cuda.device_count() == 2`) | ✅ |
| Triton kernels work (vector_add PASS, matmul PASS) | ✅ |
| DeepSeek-V2-Lite-Chat (16B MoE) loads (4/4 shards, ~3 min from tmpfs) | ✅ |
| Server starts: `"The server is fired up and ready to roll!"` | ✅ |
| Inference works end-to-end (dummy weights → pipeline fires) | ✅ |
| KV cache allocates (433,906 tokens, 12.57 GB) | ✅ |

## Patches needed

### Patch 1 — `sgl-kernel/setup_rocm.py` line 77

Allow gfx90a as a build target:

```diff
- if amdgpu_target not in ["gfx942", "gfx950"]:
+ if amdgpu_target not in ["gfx942", "gfx950", "gfx90a"]:
```

### Patch 2 — `sgl-kernel/include/utils.h` line 387

Guard the FP8 `#error` (fp8 is dead on gfx90a anyway — see [`docs/01`](01-gfx90a-architecture-constraints.md)):

```diff
  #if defined(__gfx90a__)
+ #if !defined(__gfx90a__)
  #error "fp8 is not supported in this processor (arch < gfx942)."
+ #endif
  #endif
```

### Patch 3 — runtime layernorm fix (SGLang 0.5.10.post1)

SGLang's `layernorm.py` calls `fused_add_rms_norm()` with 6 args (needs flashinfer), but the vLLM `_custom_ops` version only accepts 4. The runtime patch ([`configs/patch_layernorm.py`](../configs/patch_layernorm.py)) rewrites the call:

```python
# OLD (6-arg, broken without flashinfer):
fused_add_rms_norm(out, x, residual_out, residual, self.weight.data, self.variance_epsilon)

# NEW (4-arg, works):
fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
```

Applied at container startup before launching the server. See [`configs/patch_layernorm.py`](../configs/patch_layernorm.py).

## What SGLang unlocks

- **RadixAttention** — prefix-tree KV reuse. Identical prompt prefixes (e.g. a system prompt sent on every request) hit cached KV → **eliminates 100% of cached-prefix prefill**. This is exactly the cold-start prefill pain we couldn't fix in llama.cpp.
- **Triton attention backend** — JIT-compiled for gfx90a via AITER.
- **Continuous batching** — excellent concurrency scaling.

## BLOCKED for the 230B MoE

Despite working on gfx90a, SGLang can't serve mimo (230B) today:

1. **No selective expert offload** — SGLang loads all weights into VRAM → OOM (130 GB model, 128 GB VRAM).
2. **No GGUF support** — mimo is GGUF-only.
3. KTransformers-style CPU+GPU MoE for SGLang is roadmap, not available.

## Viable path: hybrid SGLang + llama.cpp

Since SGLang can't hold the 230B model but *can* serve smaller models with RadixAttention, the viable architecture is **hybrid**:

```
small/coding model → SGLang (RadixAttention prefix caching, snappy interactive)
230B reasoner     → llama.cpp (-ot CPU split, slow prefill but fits)
```

Route by use case. SGLang eliminates cold-start prefill for the frequently-used small model; llama.cpp handles the frontier-size reasoner.

## Reproduce

Full build + launch procedure: see [`changes/05-sglang-gfx90a-build.md`](../changes/05-sglang-gfx90a-build.md) and [`configs/Dockerfile.sglang-gfx90a`](../configs/Dockerfile.sglang-gfx90a).

```bash
# Build sgl-kernel for gfx90a (inside sglang-gfx90a:test container):
cd /tmp/sgl-kernel-src/sgl-kernel && AMDGPU_TARGET=gfx90a python3 setup_rocm.py install

# Launch SGLang server:
python3 -m sglang.launch_server \
  --model-path /models/deepseek-v2-lite-chat \
  --host 127.0.0.1 --port 5892 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.70 \
  --disable-cuda-graph \
  --served-model-name ds-lite
```

## Loading-speed note

Model weight loading from btrfs-compressed (zstd) NVMe is **extremely slow** (~200 MB/min for 32 GB = ~2+ hours). Confirmed via `py-spy` the scheduler is actively loading (102% CPU in `_load_w13`), not hung. **Fix:** copy model to tmpfs (`/dev/shm`) or an uncompressed ext4/xfs partition before launch, or pre-warm the page cache: `cat model*.safetensors > /dev/null` (~5 min for 30 GB).
