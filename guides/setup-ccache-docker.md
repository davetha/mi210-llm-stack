# Shared ccache for GPU builds on `big`

GPU kernel compilation dominates every rebuild in this stack. Containers get
recreated constantly, so the cache has to live on the **host** and be mounted
in, or it dies with the container.

Measured on `big` (2026-07-27) building llama.cpp `llama-server` for gfx90a,
419 cacheable compilations, `-j40`:

| Build | Wall time | ccache hits |
|---|---|---|
| Cold (empty cache) | 109 s | 0 / 419 (0%) |
| Full rebuild, fresh build dir | **25 s** | **405 / 419 (96.7%)** |
| Rebuild after a header genuinely changed | 50 s | 293 / 419 (69.9%) |

The 96.7% case is the one that matters: deleting the build directory entirely
and reconfiguring from scratch costs 25 s instead of 109 s, a **4.4x** speedup.
The 13 misses are translation units that embed the build path.

The third row is the correctness check, not a failure. Installing a different
rocWMMA version changed `rocwmma-version.hpp`, which every `ggml-cuda`
translation unit includes; ccache correctly invalidated exactly those 126 and
reused the other 293. A cache that had "hit" 100% there would have been
silently serving stale objects.

## Where it lives

```
/var/cache/mi210-ccache      # host, on / (507 GB free), mode 1777
```

`/mnt/llm-storage` is the alternative. When this was written that NVMe volume
was at 96% (82 GB free), which made the choice obvious; after a model cleanup on
2026-07-27 it sits at 8% (1.8 TB free), so the capacity argument no longer
decides it. Keep the cache on `/` anyway — it is small, it is not model data,
and separating build artifacts from weights means a `du` on either one still
means something.

## Build containers (llama.cpp and friends)

Mount the host directory and set the environment at `docker run` time:

```bash
docker run -d --name my-build --entrypoint /bin/bash \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
  -v /var/cache/mi210-ccache:/ccache \
  -v /mnt/llm-storage:/models \
  -e CCACHE_DIR=/ccache \
  -e CCACHE_MAXSIZE=100G \
  -e CCACHE_DEPEND=1 \
  -e CCACHE_SLOPPINESS=locale,time_macros,include_file_ctime,include_file_mtime \
  llama-rocm714:latest -c "sleep infinity"

docker exec my-build bash -lc "apt-get update -qq && apt-get install -y ccache"
```

> `llama-rocm714:latest` has `ENTRYPOINT ["/src/build/bin/llama-server"]`, so
> `--entrypoint /bin/bash` is required or the container exits with
> `error: invalid argument: sleep`.

`CCACHE_DEPEND=1` is the setting that earns its keep. Depend mode hashes the
recorded dependency list instead of re-running the preprocessor, which matters
enormously given Composable Kernel's header weight. All 405 hits above were
direct-mode hits, zero preprocessed.

### CMake

Pass all three launchers. The HIP one is the important one — the `.cu` files are
compiled as HIP, so `CMAKE_CXX_COMPILER_LAUNCHER` alone misses every GPU kernel:

```bash
cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx90a -DAMDGPU_TARGETS=gfx90a \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
```

Confirm it took effect before trusting a build time:

```bash
grep -c ccache build/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/build.make   # 141
```

## aiter's JIT needs the binary shadowed, not the PATH changed

aiter resolves its compiler to an **absolute path**
(`aiter/jit/core.py` -> `executable_path("hipcc")`), so putting ccache earlier
in `PATH` does nothing. The compiler binary itself has to be shadowed:

```bash
mv /opt/rocm/bin/hipcc /opt/rocm/bin/hipcc.real
cp hipcc-ccache-shim /opt/rocm/bin/hipcc     # see configs/hipcc-ccache-shim
chmod 755 /opt/rocm/bin/hipcc
```

The shim `exec`s `ccache /opt/rocm/bin/hipcc.real "$@"` and sets cache defaults
that the environment can still override.

**Verify before you walk away, and keep a rollback ready** — a broken shim
silently breaks every future JIT compile in that container:

```bash
echo 'int main(){return 0;}' > /tmp/t.hip
/opt/rocm/bin/hipcc -x hip --offload-arch=gfx90a -c /tmp/t.hip -o /tmp/t.o
# on failure: rm /opt/rocm/bin/hipcc && mv /opt/rocm/bin/hipcc.real /opt/rocm/bin/hipcc
```

Do **not** compare the object bytes against the real compiler's and conclude the
shim is corrupting output — `hipcc` is nondeterministic and produces different
bytes on every run even when invoked twice identically. Compare sizes and check
that a repeat compile is a ccache hit instead.

### `fa-build` is a special case

`fa-build` holds the AITER ASM work and must not be restarted, and Docker cannot
add a bind mount to a running container. Its only host-backed path is the
existing `/mnt/llm-storage:/models` mount, so its cache is at
`/models/ccache-aiter` (host `/mnt/llm-storage/ccache-aiter`), capped at 40 GB —
on the other volume, against the general rule above. It still survives container
churn, which is the point.

That split was originally a space worry, since `/mnt/llm-storage` was nearly
full. It is not any more (8% used as of 2026-07-27), so this is now purely a
tidiness wart rather than a risk.

It also runs ccache **4.9.1** (from its own Ubuntu base) versus 4.12.3
elsewhere, which is a second reason to keep its cache directory separate.

Next time `fa-build` is legitimately recreated, give it
`-v /var/cache/mi210-ccache:/ccache` and drop the separate directory.

## Verifying the cache actually works

A silently-missing ccache is worse than none, because it hides its own cost.
Always check hits rather than assuming:

```bash
ccache -z                      # zero stats before the build
cmake --build build -j40
ccache -s | grep -E "Cacheable|Hits:|Misses:"
```

Expect near-zero hits on the first build and a high hit rate on the second. If
the second build is still all misses, the launcher never reached the compiler —
check `build.make` as above.
