# Change 06 — vLLM TP=2 works on 2× MI210 (Docker #2942 fix)

**vLLM tensor-parallel across both MI210s works.** This was previously believed to be a hardware limitation (P2P/peer-DMA is dead on CDNA2 — see [`docs/01`](../docs/01-gfx90a-architecture-constraints.md)). It is **not**. The root cause was Docker container resource limits, and the fix is purely Docker configuration.

See [`docs/06-vllm-poc-results.md`](../docs/06-vllm-poc-results.md) for the full results.

## Results

| Model | Load Time | Decode (tok/s) | Output |
|---|---|---|---|
| facebook/opt-1.3b | 29.6 s | 10.6 | Correct |
| DeepSeek-V2-Lite (16B MoE) | ~58 min | **21.7** | Correct |

The 16B MoE serves correctly across both cards. **MoE + tensor-parallel works on MI210.**

## Root cause: Docker resource limits (issue #2942)

The TP=2 failure mode (workers deadlocking during the NCCL/RCCL collective init) looked like a hardware P2P problem but was actually the container starving the collective of the resources it needs:

1. **Shared memory too small** — default 64 MB `/dev/shm`. RCCL uses `/dev/shm` for the all-reduce; 64 MB → deadlock.
2. **`nproc` ulimit** — default cap kills worker processes during collective bootstrap.
3. **`memlock` ulimit** — RCCL needs to pin shared-memory pages.
4. **`pids-limit` / seccomp** — default seccomp profile blocks `shm_*` syscalls RCCL uses.

None of this is GPU-architecture-specific. It's the same class of Docker "works on a well-provisioned box, fails in a stock container" issue.

## The fix — exact `docker run` flags

```bash
docker run ... \
  --device /dev/dri \                     # whole directory, NOT a single renderD### node
  --group-add 991 \                        # numeric GID, NOT the "video" name
  --shm-size=16G \
  --ulimit nproc=unlimited:unlimited \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE
```

### The two non-obvious gotchas

These two are the ones that cost the most debugging time:

#### 1. `--device /dev/dri` (whole dir), NOT a single render node

```bash
# WRONG — only sees one card:
--device /dev/dri/renderD128

# RIGHT — sees both cards:
--device /dev/dri
```

On a 2-GPU box the second card is `/dev/dri/renderD129`. Passing only `renderD128` makes the second MI210 invisible to the container, and TP=2 fails to find it. Pass the whole `/dev/dri` directory.

#### 2. `--group-add 991` (numeric GID), NOT `--group-add video`

```bash
# WRONG — wrong group:
--group-add video      # resolves to GID 44 INSIDE the container

# RIGHT — the host video group's actual GID:
--group-add 991
```

The host's `video` group is **GID 991**, but inside a stock Ubuntu container `video` resolves to **GID 44**. Passing `--group-add video` adds the container's GID 44, which has no permission on the host render nodes → permission denied on the second GPU. Always pass the **numeric GID**.

### The resource-limit flags (standard collective hygiene)

| Flag | Why |
|---|---|
| `--shm-size=16G` | RCCL all-reduce backing store. Default 64 MB deadlocks. |
| `--ulimit nproc=unlimited:unlimited` | Don't kill workers during bootstrap. |
| `--ulimit memlock=-1:-1` | Let RCCL pin shared-memory pages. |
| `--ulimit stack=67108864:67108864` | 64 MB stack for the collective's recursion. |
| `--security-opt seccomp=unconfined` | Default seccomp blocks `shm_*` syscalls RCCL uses. |
| `--cap-add SYS_PTRACE` | Lets workers inspect each other during collective init. |

## Why TP=2 works despite no P2P

The original "TP=2 is impossible on CDNA2" assumption came from `hipDeviceEnablePeerAccess(...)` failing (peer-DMA needs xGMI, which the MI210 lacks). But **tensor-parallel doesn't require peer-DMA** — the all-reduce just bounces through host RAM over PCIe instead. That's slower than xGMI but fully functional. The collective completes; the model serves correctly; the cost is PCIe latency on every all-reduce step, not a hard failure.

## DeepSeek-V2-Lite load time (~58 min)

The ~58-minute load is **not** a TP=2 regression — it's single-threaded weight loading from btrfs-compressed (zstd) NVMe, the same pathology documented in [`docs/05-sglang-on-gfx90a.md`](../docs/05-sglang-on-gfx90a.md) (~200 MB/min). Fix: stage the model on `/dev/shm` (tmpfs) or uncompressed ext4/xfs before launch.

## What this unlocks

- **MoE TP=2 on MI210 is real.** The 16B DeepSeek-V2-Lite served correctly at 21.7 tok/s across both cards.
- The path to larger models (full DeepSeek-V2/V3 split across both 64 GB cards) is open — gated by load time and KV-cache sizing, not a hardware wall.
- The Docker fix is portable: any ROCm container doing multi-GPU collectives should use these same flags.
