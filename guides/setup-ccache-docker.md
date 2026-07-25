# Set Up ccache in Docker for Fast Incremental GPU Builds

GPU kernel compilation is the slowest part of building llama.cpp / FlashAttention
on gfx90a — the CK backend alone is **2926 kernel objects**. ccache makes
incremental rebuilds return from cache instantly. This is the single biggest
build-time win after `MAX_JOBS`.

## 1. Install ccache in the Docker image

Add to your Dockerfile (or run inside the container):

```dockerfile
RUN apt-get update && apt-get install -y ccache
```

## 2. Configure CMake to use ccache as a compiler launcher

Add these flags to your `cmake -B build` command:

```bash
cmake -B build \
  -DGPU_TARGETS=gfx90a -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
```

| Flag | What it does |
|------|-------------|
| `CMAKE_CXX_COMPILER_LAUNCHER=ccache` | Wraps the C++ compiler with ccache. |
| `CMAKE_HIP_COMPILER_LAUNCHER=ccache` | Wraps the HIP (`hipcc`) compiler with ccache — **this is the important one** for GPU kernels. |

> If your CMake version doesn't recognize `CMAKE_HIP_COMPILER_LAUNCHER`, set
> `CCACHE_CPP2=yes` and prefix `hipcc` with `ccache` via `CMAKE_HIP_COMPILER=ccache\ hipcc`.

## 3. Mount a persistent cache directory

The cache must survive container restarts to be useful. Mount a host directory:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  -v /mnt/llm-storage/ccache:/root/.ccache \
  -v /mnt/llm-storage/turbo-build/src:/build/src \
  -w /build/src/build \
  --entrypoint bash \
  llama-rocm714:latest \
  -c 'cmake --build . --target ggml-hip llama-cli -- -j$(nproc)'
```

On `big` the cache lives at `/mnt/llm-storage/ccache` (the 1.9 TB btrfs NVMe volume).

## Impact

| Build type | Without ccache | With ccache (warm) |
|------------|---------------|--------------------|
| Full clean build | ~60 min | ~60 min (cold) |
| Incremental (1 file changed) | ~3–8 min | **<30 s** |
| Incremental (after `git stash`/`pop`) | ~60 min | **<2 min** |

The GPU kernel `.o` files return from cache on the hash of (source + compiler flags + `GPU_TARGETS`). Changing only a non-kernel source file means all 2926 CK objects are cache hits.

## 4. Verify ccache is working

```bash
ccache --show-stats
# after a build:
ccache --show-stats | grep "Cacheable calls"
```

You should see `Hits:` increasing on the second build.

## 5. Cache size tuning

The default max is 5 GB. For GPU builds, bump it:

```bash
ccache --max-size 20G
# or set in the container env:
ENV CCACHE_MAXSIZE=20G
```
