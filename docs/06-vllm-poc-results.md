# vLLM POC Results — single GPU + multi-GPU on MI210

Full POC detail behind the numbers in [`benchmarks/`](../benchmarks/README.md). Two vLLM builds were tested: **0.25.2.dev0** (single-GPU) and **0.23.1.dev1** (multi-GPU / TP=2). Both on the same 2× MI210 box.

---

## Single GPU (vLLM 0.25.2.dev0, one MI210)

### Baseline — DeepSeek-V2-Lite (16B MoE)

| Setting | Value |
|---|---|
| Tensor parallel | 1 |
| Dtype | BF16 |
| Attention | eager |
| TTFT | 0.06–0.38 s |
| **Decode** | **25.0 tok/s** |
| Prefill | ~1,600 tok/s |
| Output | Correct |

Solid baseline: the 16B MoE runs correctly and usefully fast on a single MI210.

### `cpu_offload_gb` — proven no-op on CDNA2

| `cpu_offload_gb` | TTFT | Decode (tok/s) | KV-cache token budget |
|---|---|---|---|
| 0 (baseline) | 0.06–0.38 s | 25.0 | 453,440 |
| 20 | 0.06 s | 23.9 | **453,440** (identical) |
| 30 | 0.05 s | 23.3 | **453,440** (identical) |

The KV-cache token budget did not change at all between offload=0 / 20 / 30 — `453,440` tokens in every run. If the offload were actually moving 20–30 GB to host memory, the token budget would have grown substantially. It did not.

The tiny decode-speed dip (25.0 → 23.9 → 23.3) is consistent with the extra vLLM bookkeeping overhead, **not** with real PCIe streaming: if 20 GB were genuinely streaming over the bus every decode step, decode would cap near **~15 tok/s**, not stay at ~24.

### Prefetch offloader (explicit H2D copy path)

The `PrefetchOffloader` is vLLM's explicit-host-to-device-copy fallback (as opposed to UVA zero-copy). Ran with it enabled:

| Setting | TTFT | Decode (tok/s) |
|---|---|---|
| prefetch offload | 0.06 s | 23.5 |

Same ~24 tok/s, same effective no-op behavior. This is actually **informative, not broken**: the explicit-H2D path *is* the CDNA2-compatible one (UVA zero-copy requires CDNA3). See [`changes/07-vllm-cpu-offload-analysis.md`](../changes/07-vllm-cpu-offload-analysis.md).

---

## Multi-GPU TP=2 (vLLM 0.23.1.dev1) ✅ WORKS

**The headline result:** tensor-parallel across both MI210s works. This was previously believed to be a hardware limitation. It is not — it was a Docker configuration issue (issue #2942). See [`changes/06-vllm-tp2-success.md`](../changes/06-vllm-tp2-success.md).

### Results

| Model | Load Time | Decode (tok/s) | Output | Notes |
|---|---|---|---|---|
| facebook/opt-1.3b | 29.6 s | 10.6 | Correct | First TP=2 success on MI210 |
| DeepSeek-V2-Lite (16B MoE) | ~58 min | 21.7 | Correct | **MoE works with TP=2** |

The 16B MoE model is served correctly across both cards at **21.7 tok/s** — MoE tensor parallelism works on MI210, which was the open question.

### The Docker #2942 fix that made TP=2 work

The fix is **purely Docker resource limits**, nothing hardware and nothing in the model code. The exact `docker run` flags:

```bash
docker run ... \
  --device /dev/dri \                     # whole directory, NOT a single renderD### node
  --group-add 991 \                        # numeric GID, NOT the "video" name (that's GID 44 in the container)
  --shm-size=16G \
  --ulimit nproc=unlimited:unlimited \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE
```

| Flag | Why |
|---|---|
| `--device /dev/dri` | The whole dir. Passing a single `/dev/dri/renderD128` misses the sibling device on a 2-GPU box — the second card is invisible and TP=2 fails to find it. |
| `--group-add 991` | Numeric GID. The host `video` group is GID 991, but **inside the container `video` resolves to GID 44**. Passing `--group-add video` adds the wrong group and you get permission denied on the second render node. |
| `--shm-size=16G` | NCCL/RCCL (the TP=2 all-reduce) uses `/dev/shm` for the collective. Default 64 MB → deadlock. |
| `--ulimit nproc=unlimited` | Default nproc cap kills worker processes during the all-reduce bootstrap. |
| `--ulimit memlock=-1` | Same — RCCL needs to pin shared-memory pages. |
| `--ulimit stack=67108864` | 64 MB stack; large enough for the collective's recursion. |
| `--security-opt seccomp=unconfined` | Default seccomp profile blocks some `shm_*` syscalls RCCL uses. |
| `--cap-add SYS_PTRACE` | Lets the workers inspect each other during the collective init. |

### The slow load (~58 min) and why

DeepSeek-V2-Lite took **~58 minutes** to load under TP=2. This is **not** a TP=2 regression — it's single-threaded weight loading. vLLM loads model weights on one thread per rank, and the safetensors are read sequentially from a btrfs-compressed (zstd) NVMe layer. This is the same slow-loading pathology documented in [`docs/05-sglang-on-gfx90a.md`](05-sglang-on-gfx90a.md) (~200 MB/min from zstd-compressed storage).

**Fix already known:** copy the model to `/dev/shm` (tmpfs) or an uncompressed ext4/xfs partition before launch. Not a GPU problem.

### PrefetchOffloader discovery

While diagnosing the offload no-op, the `PrefetchOffloader` was identified as the **explicit H2D copy** codepath — the one that *should* work on CDNA2 (unlike UVA zero-copy, which needs CDNA3). The finding: the infrastructure for host-side weight offload on MI210 exists in vLLM; the UVA shortcut is what's CDNA3-only. Setting `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` forces the explicit-H2D fallback. Details in [`changes/07-vllm-cpu-offload-analysis.md`](../changes/07-vllm-cpu-offload-analysis.md).

---

## What this unlocks

- **MoE TP=2 on MI210 is real.** The 16B DeepSeek-V2-Lite served correctly across both cards.
- The original assumption that TP=2 was a hardware limit (P2P/peer-DMA dead on CDNA2 — see [`docs/01`](01-gfx90a-architecture-constraints.md)) was **wrong**. Tensor-parallel all-reduce just bounces through host RAM over PCIe, which is slow but functional.
- The path to serving larger models (e.g. a full DeepSeek-V2 / V3 split across both 64 GB cards) is now open, gated only by load time and KV-cache sizing — not by a hard hardware wall.
