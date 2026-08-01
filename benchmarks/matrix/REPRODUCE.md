# Reproducing the MI210 quantization matrix

Every number in `benchmarks/matrix/results/` comes from the commands below.
Copy-pasteable, with the exact image tags, environment, and arguments used.

If you only read one section, read **[Things that silently invalidate a
run](#things-that-silently-invalidate-a-run)**. Four of the results in this
project's history were wrong because of items on that list, and none of them
looked wrong at the time.

---

## Hardware and software

| | |
|---|---|
| GPUs | 2x AMD Instinct MI210 (gfx90a / CDNA2), 64 GB HBM2e each, **no xGMI** (PCIe only) |
| CPU / RAM | AMD EPYC 74F3 24-core, 499 GB |
| Model storage | `/mnt/llm-storage`, 1.9 TB btrfs, `compress-force=zstd:3`, `noatime` |
| ROCm | 7.14.60850 |
| vLLM | `0.23.1.dev1+g9ddef7117.d20260715` |
| PyTorch | 2.11.0+rocm7.14.0 |
| AITER | `0.1.19` (built from source; the stock image ships 0.1.13) |
| triton | **3.7.1** — see the segfault note below |
| Python | 3.14.6 |

**Base images**

```
rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0   # vLLM base
llama-rocm714:latest                                                       # llama.cpp
rocm-vllm-aiter-gfx90a:latest                                              # built below
```

## 0. Device index trap — read before pinning anything

On this host HIP and rocm-smi enumerate the two cards in **opposite order**:

| rocm-smi | PCIe | HIP |
|---|---|---|
| `GPU[0]` | `87:00.0` | `HIP_VISIBLE_DEVICES=1` |
| `GPU[1]` | `C3:00.0` | `HIP_VISIBLE_DEVICES=0` |

Reading `rocm-smi` to find an idle card and then pinning `HIP_VISIBLE_DEVICES`
to that number lands on the **wrong** card. Confirm with `rocminfo` BDFID:

```bash
rocminfo | grep -E "Marketing Name|BDFID"      # BDFID 34560 = 0x8700, 49920 = 0xC300
```

## 1. Shared ccache

```bash
sudo mkdir -p /var/cache/mi210-ccache && sudo chmod 1777 /var/cache/mi210-ccache
```

Mounted into every build container as `/ccache` with
`CCACHE_DIR=/ccache CCACHE_MAXSIZE=100G CCACHE_DEPEND=1`. `CCACHE_DEPEND=1` is
the setting that earns its keep — preprocessor-mode hashing on Composable
Kernel headers costs a large fraction of the compile it is avoiding. Measured:
109 s -> 25 s on a from-scratch llama.cpp rebuild, 405/419 hits.

Full detail: `guides/setup-ccache-docker.md`.

## 2. Build the AITER-enabled vLLM image

```bash
./benchmarks/matrix/build_vllm_aiter_gfx90a.sh rocm-vllm-aiter-gfx90a:latest v0.1.19
```

This does four things that must **all** succeed, because a half-patched image
does not error — it just runs slow:

1. Installs AITER v0.1.19 from source (`--recursive`; it vendors Composable
   Kernel as a submodule), then **reinstalls `triton==3.7.1`**. Installing
   AITER drags triton down to 3.7.0 as an undeclared transitive dependency, and
   3.7.0 **segfaults** on `import aiter` inside `triton/knobs.py` — exit 139,
   no Python traceback. `import triton` alone still works at 3.7.0, so a naive
   check clears it.
2. Translates gfx942 ASM code objects to gfx90a — **242 of 1,425** survive.
   Portability is proven by re-assembling each instruction for gfx90a, never
   assumed.
3. Opens AITER's own gfx90a dispatch paths (~16 sites).
4. Widens vLLM's `on_mi3xx()` attention gate to `on_gfx9()` (6 sites).

Then apply the performance patches, **in this order**:

```bash
docker exec <container> python3 configs/prefer_aiter_fa_gfx90a.py     # ROCM_AITER_FA selectable
docker exec <container> python3 configs/enable_int8_moe_rocm.py       # INT8 MoE on CDNA
docker exec <container> python3 configs/enable_aiter_ck_gemm_gfx90a.py  # CK int8 GEMM -- 1.48x decode
docker commit <container> rocm-vllm-aiter-gfx90a:latest
```

`prefer_aiter_fa_gfx90a.py` is required for AITER to be **chosen**, not merely
admitted — vLLM's backend list appends `ROCM_ATTN` first unconditionally and
takes the first valid entry.

`enable_aiter_ck_gemm_gfx90a.py` **must run after** the gate-widening step
above, and asserts that rather than writing a broken file. It removes three
independent blockers (`docs/43`): aiter's `GFX_CU_NUM_MAP` has no gfx90a entry
so instance codegen raises and the build dies later on a missing generated
header; `is_linear_enabled()` is `on_mi3xx()`-gated; and `register_ops_once()`
is `on_mi3xx()`-gated too, which means **no AITER custom op is registered at
all** on CDNA2 — the attention path never hit that because it calls aiter
directly rather than through `torch.ops.vllm`.

Verify the whole image before benchmarking it, on a machine with the cards:

```bash
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --entrypoint /usr/local/bin/verify-gfx90a <image>
```

Check 7 is the one that matters for the GEMM: checks 1–6 all passed on an image
where serving died at the first `qkv_proj`, because they exercise the attention
path only.

## 3. Fetch models

```bash
python3 benchmarks/matrix/fetch_model.py <hf-repo> <dest> [--include PATTERN] [--dry-run]
```

Always `--dry-run` an unfamiliar repo first; the printed total decides whether
it fits. Downloads whole files concurrently and **never splits a single file** —
Hugging Face's Xet CDN signs each redirect for a specific byte range, so
`aria2 --split=16` gets HTTP 403 on 15 of 16 connections while appearing to
progress.

Tier-1 set used here:

```bash
B=/mnt/llm-storage/bench-matrix
python3 fetch_model.py Qwen/Qwen3-30B-A3B-Thinking-2507                   $B/t35-bf16
python3 fetch_model.py QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ          $B/t35-awq
python3 fetch_model.py ramblingpolymath/Qwen3-30B-A3B-thinking-2507-W8A8  $B/t35-w8a8
python3 fetch_model.py Qwen/Qwen3-30B-A3B-Thinking-2507-FP8               $B/t35-fp8
python3 fetch_model.py unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF $B/t35-gguf-q4km --include Q4_K_M
python3 fetch_model.py unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF $B/t35-gguf-q8   --include Q8_0
```

## 4. Environment

```bash
source benchmarks/matrix/env/gfx90a-common.env
```

The load-bearing entries, with verdicts:

| Variable | Value | Verdict on gfx90a |
|---|---|---|
| `HSA_NO_SCRATCH_RECLAIM` | `1` | **Required.** AMD documents 5-10x latency win on MI200. Not the default. |
| `NCCL_P2P_DISABLE` | `0` | **CORRECTED 2026-07-31.** This table previously said `1`, "required for TP, no xGMI here". The premise is true and the conclusion is false: the driver advertises a PCIe P2P link, a device-to-device copy measures **26.98 GB/s** against 14.16 staged through host memory, and enabling it is worth **+11.2% prefill / −10.3% TTFT** at TP=2 (round 31, `docs/40`). Set it back to `1` only if collective setup hangs, and say so in the run label. |
| `GPU_MAX_HW_QUEUES` | `4` | Stated explicitly so it is recorded, not inherited. |
| `VLLM_PREFER_AITER_FA` | `1` | Makes AITER FA **selected** rather than merely available. Worth **1.19–1.33× prefill** (round 37). Candidacy is not selection — `_get_backend_priorities()` appends `ROCM_ATTN` first. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | **The largest single win: 1.48× decode** at TP=2, 1.71× at TP=1 (round 40, `docs/43`). Routes int8 GEMM to AITER's CK kernel instead of vLLM's Triton fallback, which emits only 40 workgroups for qkv_proj on a 104-CU card at M=1. Requires `enable_aiter_ck_gemm_gfx90a.py`; **inert without it**. |
| `VLLM_ROCM_USE_AITER_MOE` | `0` | Measured **0.977×** — it runs (`module_moe_asm` loads) but is a mild regression. `docs/45`. |
| `VLLM_ROCM_USE_AITER` / `_MHA` | `1` | Only effective with the patches above. |
| `VLLM_ROCM_USE_AITER_{TRITON_ROPE,UNIFIED_ATTENTION,CUSTOM_AR,TRITON_GEMM}` | *unset* | All four **cannot engage** on this stack, each for a different reason. `docs/45`. |
| `HIP_FORCE_DEV_KERNARG` | *unset* | NO-EFFECT — gated on `isGfx94x`, already defaults to 1. |
| `VLLM_USE_TRITON_FLASH_ATTN` | *unset* | REMOVED from current vLLM. |
| `VLLM_USE_V1` | *unset* | REMOVED; V1 is the only architecture. |
| Quick Reduce | *unset* | MI300-ONLY — rejects gfx90a **and** assumes xGMI. |

## 5. Run an arm

```bash
TP=1 LONGCTX_TOKENS=110000 READY_TIMEOUT=6000 \
  benchmarks/matrix/run_arm.sh t35-awq 35B awq vllm-aiter $B/t35-awq \
  --max-model-len 131072 --safetensors-load-strategy=prefetch
```

llama.cpp arms:

```bash
TP=1 LONGCTX_TOKENS=250000 \
  benchmarks/matrix/run_arm.sh t35-q8 35B q8_0 llamacpp $B/t35-gguf-q8 \
  --ctx-size 262144
```

Or the whole tier: `benchmarks/matrix/sweep_tier35.sh`.

Fixed settings across all runs: `--seed 1234`, `temperature 0`,
`PYTHONHASHSEED=0`, `--gpu-memory-utilization 0.90`,
`--no-enable-prefix-caching`. llama.cpp uses `-ub 2048` (measured optimum;
4096 and 8192 are both worse) and does **not** quantize the KV cache.

## Known-benign warnings

These appear on every run and are **not** problems:

```
ERROR [config.py:29] Failed to import Triton kernels. Please make sure your
triton version is compatible. Error: No module named 'triton_kernels.matmul_ogs'
```

A side-effect of reinstalling `triton==3.7.1` (required — 3.7.0 segfaults on
`import aiter`), which drops the separate `triton_kernels` companion package.
It gates the **mxfp4** path only, which nothing here uses. It is logged at ERROR
level, appears five times at startup, and is harmless: the W8A8 arm that
produced the fastest result in the matrix logged it too.

```
[aiter] Current hipcc not support: -mllvm -amdgpu-coerce-illegal-types=1, skip it
```

AITER probing for a compiler flag this hipcc lacks, then proceeding without it.

## Things that silently invalidate a run

Each of these produced a wrong number in this project before being caught.

**Prefix caching.** vLLM V1 enables it by default; llama.cpp keeps one per
slot. A repeated prompt returns near-zero TTFT that reads as a spectacular
result. Defended three ways: `--no-enable-prefix-caching`, UUID seeding in the
prompt's **first** tokens (a unique suffix still shares a cacheable prefix),
and `--verify-cold`, which fails the run if a later repetition beats the first
by more than 2x.

**`--max-model-len` above 131072.** Costs **10x decode on every request**,
including short ones, with nothing at request time to indicate it. The gate is
evaluated at CUDA-graph capture against `max_model_len`, so the Triton fallback
gets baked into the graph. See `docs/23-vllm-gfx90a-cudagraph-decode-cliff.md`.

**A "fast" number with no proof the fast path ran.** Backend *selection* is not
evidence. Require the `.co` load lines with `AITER_LOG_LEVEL=info`:

```
fwd_hd128_bf16_causal_rtna_group.co
fwd_hd128_bf16_rtna_group.co
```

Docs 16 and 17 in this repo published fallback timings as ASM results because
nothing checked.

**Reasoning models and short generations.** Qwen3-*-Thinking emits a `<think>`
block and needs ~250 tokens before answering, so a correctness check on a
`max_tokens=8` run always fails on a healthy model. Correctness is a **separate
probe** with its own budget. Relatedly, llama.cpp emits thinking tokens as
`delta.reasoning_content` while vLLM leaves them in `delta.content` — reading
only `content` makes llama.cpp streams look completely empty.

**Decode rates from short generations.** Below ~32 tokens a "rate" is scheduler
jitter. The harness reports `n/a` rather than a quotable number.

**VRAM sampled from `rocm-smi`.** Measures `--gpu-memory-utilization`, not model
size — every arm reports ~90% of the card. Footprint comes from vLLM's own
accounting (`Model loading took X GiB`), harvested by `run_arm.sh`.

**Partial downloads.** A multi-connection download that aborts leaves holes in
the middle of the file. Resuming by size fills from the end, producing a
full-sized file with zeroed gaps that passes a size check, mmaps fine, and
emits garbage. `fetch_model.py` purges any incomplete file lacking an aria2
control file rather than resuming it.

**Benchmarking two models on one card.** They do not share; the second fails
with a free-memory error that reads like a config problem. All GPU work is
serialized through `followup_chain.sh`.

## Building the results table

```bash
python3 benchmarks/matrix/summarize_results.py
```

Reads every JSON in `results/`. The table is generated, never transcribed.
