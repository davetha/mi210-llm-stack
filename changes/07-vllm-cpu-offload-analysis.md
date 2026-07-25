# Change 07 — vLLM `cpu_offload_gb` is a no-op on CDNA2 (UVA requires CDNA3)

`vLLM`'s `--cpu-offload-gb` flag **silently does nothing** on the MI210 (gfx90a / CDNA2). The UVA zero-copy path it relies on requires **CDNA3 (MI300)**. The flag is accepted, no error is raised, but no memory is moved off the GPU. This documents the proof and the CDNA2-compatible workaround.

See [`docs/06-vllm-poc-results.md`](../docs/06-vllm-poc-results.md) for the full single-GPU POC.

## Proof: the offload is a no-op

Measured on DeepSeek-V2-Lite (16B MoE), single MI210, vLLM 0.25.2.dev0:

| `cpu_offload_gb` | KV-cache token budget | Decode (tok/s) |
|---|---|---|
| 0 (baseline) | **453,440** | 25.0 |
| 10 | **453,440** | — |
| 20 | **453,440** | 23.9 |
| 30 | **453,440** | 23.3 |

The KV-cache token budget is **byte-for-byte identical** at offload=0/10/20/30 — `453,440` tokens every time. If the offload were actually relocating 10/20/30 GB to host memory, the token budget would have grown substantially (more headroom = more tokens). It did not move at all.

### The decode-speed sanity check

The small decode dip (25.0 → 23.9 → 23.3 tok/s) is consistent with **vLLM bookkeeping overhead**, not real offload. The arithmetic:

- If 20 GB were genuinely streaming across PCIe 4.0 ×16 every decode step, the bus would cap decode at roughly **~15 tok/s** for this model — not stay at ~24.
- Decode stayed at ~24 tok/s → no 20 GB stream → no offload happening.

The flag runs, allocates nothing, moves nothing. Silent no-op.

## Root cause: UVA zero-copy needs CDNA3

`cpu_offload_gb` uses vLLM's **UVA (Unified Virtual Addressing) zero-copy** offload path: it maps a host-RAM buffer into the GPU's address space so the GPU can read it directly without an explicit copy. That mapping requires hardware support for **system-shared virtual addressing with coherent host access** — a CDNA3 feature.

| Generation | UVA zero-copy host buffer | `cpu_offload_gb` |
|---|---|---|
| CDNA2 (gfx90a / MI210) | ❌ not supported | **silent no-op** |
| CDNA3 (gfx942 / MI300X) | ✅ supported | works |

On CDNA2 the mapping call succeeds at the runtime layer but the GPU cannot actually dereference the host pointer coherently, so vLLM's allocator falls back to placing everything in VRAM. The requested offload GB are simply ignored. No error, no warning, no moved memory.

## The CDNA2-compatible path: explicit H2D copy

The good news: vLLM **does** have a CDNA2-compatible offload mechanism — the `PrefetchOffloader`, which uses **explicit host-to-device `memcpy`** instead of UVA zero-copy. This is the same philosophy llama.cpp uses for its `-ot` CPU split (see [`docs/08-platform-gaps-gfx90a.md`](../docs/08-platform-gaps-gfx90a.md)).

### Force the non-UVA fallback

```bash
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1
```

This env var tells vLLM to skip the UVA path and use the explicit-H2D-copy `PrefetchOffloader` instead. On CDNA2 this is the path that *actually moves weights* — every offloaded layer is explicitly copied host→device before use and the transfer is real (visible in profiling, and decode speed reflects the PCIe cost).

### PrefetchOffloader discovery

While diagnosing the no-op, the `PrefetchOffloader` was confirmed to use **explicit H2D copies** — the architecture that works on CDNA2. The infrastructure for host-side weight offload on MI210 exists in vLLM; it's the UVA *shortcut* that's CDNA3-only, not the entire offload feature.

## Recommendation for CDNA2

1. **Never rely on `--cpu-offload-gb` alone on MI210** — it silently does nothing.
2. **Set `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`** to force the explicit-H2D `PrefetchOffloader` if you need host-side weight staging.
3. For models that exceed VRAM, the production-proven path remains **llama.cpp `-ot` CPU split** (see [`configs/launch-mimo.sh`](../configs/launch-mimo.sh)) — it has always used explicit `memcpy` and works correctly on CDNA2.

## Why this matters

The silent no-op is worse than a hard error: the operator sees `cpu_offload_gb=30` accepted, assumes 30 GB has been offloaded, sizes the model accordingly, and then OOMs at runtime with no indication that the offload never happened. The fix is awareness + the env var.
