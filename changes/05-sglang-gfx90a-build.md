# Change 05 — SGLang on gfx90a (Docker Build + Patches)

SGLang is **NOT impossible on gfx90a** — it works with 2 patches to sgl-kernel
+ 1 runtime patch. This documents the full build.

See [`docs/05-sglang-on-gfx90a.md`](../docs/05-sglang-on-gfx90a.md) for the analysis.

## Dockerfile → [`configs/Dockerfile.sglang-gfx90a`](../configs/Dockerfile.sglang-gfx90a)

Base image: `llama-vllm025:gfx90a` (pre-existing, working vLLM 0.25.2 on gfx90a).

```dockerfile
FROM llama-vllm025:gfx90a

# Rust toolchain (for outlines_core build)
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Install SGLang with --no-deps, then manually install everything EXCEPT flashinfer
RUN pip install --no-cache-dir --no-deps sglang && \
    pip install --no-cache-dir \
      "IPython" "orjson" "python-multipart" "pybind11" \
      "outlines>=0.2.1,<0.2.4" "outlines_core>=0.2.9" \
      "llguidance>=0.7.11" "xgrammar" \
      "torchao" "transformers>=4.51.0" \
      "modelscope" "tiktoken>=0.6.0" \
      "sentencepiece" "tenacity" \
      "pydantic>=2.0.0" "fastapi" "uvicorn" "uvloop" \
      "aiohttp" "openai>=1.0" "anthropic" \
      "pillow" "pyzmq>=25.0.0" "psutil" "prometheus-client" \
      "numpy<2.5.0,>=2.0.0" "einops" \
      "compressed-tensors" \
      "tabulate" "packaging" "requests" "tqdm"

CMD ["/bin/bash"]
```

Resulting image: `sglang-gfx90a:test`.

## sgl-kernel patches (2 files)

Clone sparse:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/sgl-project/sglang.git
cd sglang && git sparse-checkout set sgl-kernel
```

### Patch 1 — `setup_rocm.py` line 77

```diff
- if amdgpu_target not in ["gfx942", "gfx950"]:
+ if amdgpu_target not in ["gfx942", "gfx950", "gfx90a"]:
```

### Patch 2 — `include/utils.h` line 387

```diff
  #if defined(__gfx90a__)
+ #if !defined(__gfx90a__)
  #error "fp8 is not supported in this processor (arch < gfx942)."
+ #endif
  #endif
```

(fp8 is dead on gfx90a anyway — see [`docs/01`](../docs/01-gfx90a-architecture-constraints.md).)

## Build sgl-kernel

```bash
cd /sgl-kernel && AMDGPU_TARGET=gfx90a python3 setup_rocm.py install
```

Takes ~2 minutes. Produces `sglang_kernel-0.4.5-py3.14-linux-x86_64.egg`.

## Runtime layernorm patch → [`configs/patch_layernorm.py`](../configs/patch_layernorm.py)

SGLang 0.5.10.post1's `layernorm.py` calls `fused_add_rms_norm()` with 6 args
(needs flashinfer), but the vLLM `_custom_ops` version only accepts 4. Applied at
container startup via `python3 /patch.py` before launching the server.

## Server launch

```bash
docker run -d --name sglang-server --network host --init \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g --cap-add SYS_PTRACE \
  -v /dev/shm:/host-shm \
  -v /tmp/sgl-kernel-src/sgl-kernel:/sgl-kernel-src \
  -v /mnt/llm-storage/patch_layernorm.py:/patch.py \
  -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint bash \
  sglang-gfx90a:test \
  -c 'cd /sgl-kernel-src && python3 setup_rocm.py install >/dev/null 2>&1 && \
      python3 /patch.py && \
      python3 -m sglang.launch_server \
        --model-path /host-shm/ds-lite \
        --host 127.0.0.1 --port 5892 \
        --attention-backend triton \
        --moe-runner-backend triton \
        --trust-remote-code \
        --mem-fraction-static 0.70 \
        --served-model-name ds-lite \
        --disable-cuda-graph \
        --skip-server-warmup'
```

## Verified working

- sgl-kernel 0.4.5 builds and imports ✅
- Triton kernels work (vector_add PASS, matmul PASS) ✅
- Both MI210s detected ✅
- DeepSeek-V2-Lite-Chat loads (4/4 shards, ~3 min from tmpfs) ✅
- Server starts, inference fires end-to-end ✅
- KV cache allocates (433,906 tokens, 12.57 GB) ✅

## Loading-speed note

Model weight loading from btrfs-compressed (zstd) NVMe is **extremely slow**
(~200 MB/min). Fix: copy model to `/dev/shm` (tmpfs) before launch, or pre-warm
the page cache: `cat model*.safetensors > /dev/null`.
