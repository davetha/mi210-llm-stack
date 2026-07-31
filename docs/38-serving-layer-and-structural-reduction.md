# The serving layer: the loader fix, the caching blind spot, and structural reduction

**Date**: 2026-07-30 · Companion to `docs/25` item 1, `changes/04`, `docs/05`.
Everything platform-specific below was checked against the **installed** vLLM on
the box (`0.23.1.dev1+g9ddef7117`), not against upstream docs.

Tags: **(M)** measured/read from the install · **(V)** verified from primary
source · **(I)** inference with arithmetic shown.

---

## 1. The 7-hour load has a shipped fix: `--load-format sharded_state` (M)

`docs/25` item 1 spent the investigation trying to *patch* the loader. The
unexamined question was whether a different loader backend skips it entirely.
One does.

`model_executor/model_loader/sharded_state_loader.py`, read from the installed
tree — the entire weight-load body is:

```python
state_dict = self._filter_subtensors(model.state_dict())
for key, tensor in self.iterate_over_files(filepaths):
    param_data = state_dict[key].data
    ...
    param_data.copy_(tensor)      # line 154
    state_dict.pop(key)
```

**No `weight_loader` call. No `_load_w13`. No `moe_wna16_weight_loader`.** It
walks the rank's pre-sharded files and does one flat `copy_` per already-existing
runtime parameter. The per-expert iteration where `py-spy --native` caught
8/8 samples in `hsakmt_ioctl` **cannot occur on this path** — there is no
per-expert fresh-mmap source to register.

Workflow: run once with the normal loader, dump a snapshot with
`save_sharded_state.py --tensor-parallel-size 2`, thereafter serve with
`--load-format sharded_state --tensor-parallel-size 2`. **7 hours becomes a
one-time cost.**

Constraints, all read from the source rather than assumed:

- **Pre-sharded only.** `raise ValueError("... only pre-sharded checkpoints are
  currently supported!")` — the snapshot must be generated first.
- **Bound to the layout.** The glob pattern embeds `rank=`, so reload must use
  the identical TP. No any-to-any resharding.
- **Engine-version-specific.** It snapshots `model.state_dict()` *after*
  `process_weights_after_loading`, i.e. runtime tensors. Regenerate after a vLLM
  bump.
- `iterate_over_files` will use `runai_safetensors_weights_iterator` if
  `load_format == "runai_streamer_sharded"`, so the two compose if the plain
  reader ever becomes the limit.

**Verification that it worked:** grep the load log for `_load_w13` /
`moe_wna16_weight_loader` frames — they should be absent — and check the
`"Loading weights took %.2f seconds"` line the loader emits directly.

### Upstream context worth having (V)

vLLM **PR #46766** independently root-causes the same mechanism on the
fused-MoE path and names it: `expert_data.copy_(loaded_weight)` with a
**non-contiguous CPU** source, "about 3-4 seconds per weight." Their fix is
`.contiguous()` on the **CPU side before** the copy — note this is *not* what
`configs/fast_moe_expert_load.py` did (that made the already-contiguous narrowed
*source* contiguous, which `docs/25` correctly refuted twice over). Their claim is
unverified on ROCm and conflicts with the `--native` ioctl evidence, so it should
not be trusted here — but it confirms the path is upstream-known and
quant-method-agnostic.

**PR #49459** ("Reload layout-identical weights directly") generalises the
`sharded_state` idea but is `needs-rebase` and excludes compressed-tensors/WNA16.

### Killed as loader fixes (V)

| option | why not |
|---|---|
| **fastsafetensors** | ROCm-supported (nogds only; README: no GDS alternative on ROCm) and genuinely faster file→CPU — but that path was already measured at **4.63 GB/s** and is not the bottleneck. Issue **#48644**: on a Qwen3-35B-A3B MoE it drives the layerwise path to allocate **>40 GB** of temp buffers and loads out of order. It still funnels every expert through the same `weight_loader`. |
| **runai_streamer** (non-sharded) | ROCm-capable, but same per-expert loader. `runai_streamer_sharded` overlaps `sharded_state` without the clean copy-into-runtime guarantee. |
| **tensorizer** | CUDA/S3-oriented, no verified ROCm path, still per-tensor deserialize. |
| **ServerlessLLM** (**2401.14351**, OSDI'24) | 3.6–8.2× loading gains are **I/O-path** (sequential chunk reads vs random mmap) — the axis already proven irrelevant here. Loader is coupled to their serving system, not a vLLM `--load-format`. Same for **Tangram** (**2512.01357**). |

### Worth one cheap probe alongside it (V + I)

AMD's HIP docs give the arena pattern explicitly: registered memory performs like
`hipHostMalloc` *after* setup, so the cost is the per-call page-lock+map, and the
documented remedy for many small buffers is to register **one large region up
front and carve it up**. PyTorch exposes that as config:

```
PYTORCH_HIP_ALLOC_CONF=pinned_reserve_segment_size_mb:<N>,pinned_use_hip_host_register:True,pinned_num_register_threads:8
```

`pinned_reserve_segment_size_mb` is literally "reserve one large pinned segment
and sub-allocate small requests from it to reduce expensive device library
calls." **(I)** It only helps if the copy goes through PyTorch's pinned caching
allocator, and the fused-MoE path copies from a raw mmap'd slice, so it may not
intercept — but it is an env var, not a patch.

---

## 2. The caching blind spot, and why it is worth more here than on a GPU-only box

### The measurement methodology structurally hid this (M)

`benchmarks/matrix/serve_vllm.sh:75` passes `--no-enable-prefix-caching`, and
`bench_matrix.py` seeds the UUID into the prompt's **first** tokens specifically
to defeat prefix matching. Both are *correct* for measuring quantization. The
consequence is that **production caching behaviour has never been characterised**.

`configs/launch-mimo.sh` also runs `-np 1` — a single slot. With open-webui in
front, any conversation that is not the warm system prefix evicts it, so
cross-conversation reuse in production is approximately zero.

### The CPU-expert question, answered — and the answer is favourable (I, high confidence)

The obvious fear: prefix reuse only saves attention KV, so the 7.4 s of CPU
expert GEMM re-runs anyway, making caching much less valuable here than it looks.

**That is false, and the reason matters.** A KV cache entry *is the output of
having run those token positions through every layer* — attention and the expert
FFNs included. On a prefix hit, those positions are not forwarded through the
network at all. The 7.4 s is a **per-token-processed prefill cost**; cached tokens
are not processed, so their expert cost disappears with them. There is no separate
"expert cache" and none is needed.

**So prefix caching is worth *more* on this box than on a GPU-only one**, exactly
because each skipped token was going to cost DDR4 expert GEMM rather than cheap
HBM attention. This is now the highest-EV unexplored item in the serving layer.

### The blocker is real and already documented (M)

`changes/04:58-59`:

> **Restored KV does NOT enable prefill skip.** Despite `sim_best=1.000` and
> `f_keep=0.922` (perfect LCP match), all 106 tokens are still reprocessed (4.4s).

So `--slot-save-path` currently buys **persistence without the compute saving** —
the thing that would actually pay. `changes/04:74` already identifies the fix as
upstream work on the prefill-skip-on-restore path. **Re-check current llama.cpp
master against this**; the infrastructure is already staged for it and it converts
a persistence hack into a real win.

In-session prefix reuse *does* work (the measured 8×), which is why the
TTL 1800 → 86400 bump was the right pragmatic call.

### What the automatic versions give that the hand-rolled one doesn't (V)

- **RadixAttention** (**2312.07104**, SGLang, Dec 2023): LRU **radix tree** of KV,
  prefix-match on arrival, cache-aware scheduling that sorts the queue by matched
  prefix length. Reported **up to 5×** throughput, hit rates 50–99%. Code open.
- **vLLM APC**: block-level content hashing, `(parent_hash, block_tokens, extras)`,
  only full blocks cached. Host-side hashing over standard paged attention, so it
  runs on gfx90a.
- Both are **prefill-only** wins. Neither touches decode.

The hand-built version is a single-session, single-prefix, manual subset of
RadixAttention — further hobbled by the restore bug above. Given SGLang cannot
hold the 230B MoE (`docs/05`: no selective expert offload, no GGUF), the realistic
path is the hybrid already sketched there: **SGLang + RadixAttention for the small
models, llama.cpp for the 230B once restore-skip works.**

---

## 3. KV spill to the 499 GB of DDR4 — shipped, and proven on ROCm with a 230B MoE (V)

**LMCache** was benchmarked on **MiniMax-M2.5, a 230 GB FP8 MoE, on 2× MI300X,
vLLM 0.19.0 + LMCache built from source for ROCm**, against 739 real multi-turn
traces, comparing no-cache vs vLLM HBM prefix cache vs **LMCache CPU-DRAM
offload**. That is close to this exact scenario.

- Integration: `LMCacheConnectorV1` in-process, or
  `--kv-offloading-backend lmcache --kv-offloading-size <GiB>`.
- **Gotcha from the same benchmark: `PYTHONHASHSEED=0` is mandatory**, or you get
  a 0% hit rate on bit-identical prompts. `env/gfx90a-common.env` already sets
  `PYTHONHASHSEED=0` for determinism — that happens to be a prerequisite here.
- **(I)** MI300X ≠ MI210, but DRAM offload is engine-level block movement with no
  FP8 ALU dependency, so gfx90a should be fine. Expect to build from source; no
  gfx90a wheel confirmed.

This answers the capacity question directly: **yes, long contexts can exceed VRAM
and survive model eviction, with shipped code, today.** With 499 GB the KV tier is
effectively unbounded for this workload.

**And CacheBlend rides on it** — **2405.16444**, EuroSys'25 **Best Paper**: reuse
precomputed KV for chunks *regardless of position*, then selectively recompute
~10–15% of tokens to repair cross-attention between independently cached chunks.
**TTFT −2.2–3.3×, throughput +2.8–5×, no quality loss.** It is a first-class
LMCache feature, which makes it the only Axis-2 method with a ROCm-viable path.
For open-webui RAG, retrieved chunks stop being a fresh full prefill — and per §2
that means they stop paying CPU expert GEMM too.

**Killed:** Prompt Cache (**2311.04934**) needs author-defined schemas, research
prototype. EPIC (**2410.15332**, ICML'25) is stronger on paper but research code
only. Mooncake (**2407.00079**), Llumnix, dLoRA — all multi-node architectures for
a single 2-GPU box.

---

## 4. Structural reduction: fewer experts, fewer layers

Given the `docs/33` finding that per-token cost is dominated by in-kernel
instruction work, removing *work* rather than *bytes* is well-aimed for the first
time.

### REAP expert pruning is the one that hits the measured bottleneck (V + I)

**REAP** (**2510.13999**, Oct 2025, rev 2026-05): Router-weighted Expert
Activation Pruning, **one-shot, calibration-only, no retraining**.
**Near-lossless at 50% expert pruning** on Qwen3-Coder-480B and Kimi-K2, tested
20B–1T SMoE. Finds pruning beats merging for generative tasks. Code exists.

**Why this is the right one here (I):** per `docs/31`, CPU-side prefill traffic is
`(n/c) · E · bytes_per_expert`. REAP halves **E** — the term that dominates — and
it halves the CPU-DDR4-resident set at the same time. Nothing else on this list
touches that term.

### Reducing top-k is cheaper but hits a different phase (I)

Routing top-6 instead of top-10 is a config change. But per `docs/31`'s saturation
arithmetic, at `-ub 2048` prefill already touches **all ~512 experts per chunk**
(saturation at ~51 tokens), so **top-k reduction barely changes prefill** — it
cuts **batch-1 decode**, where 10→6 is a real 40% reduction in expert bytes read.
So REAP and top-k attack different phases and compose.

**Caveat, and it is not academic given `deephat` and the coding models (V):** the
result that half-top-k holds for **greedy** decoding explicitly **degrades
multi-sample pass@n** — the extra experts matter for sampling diversity. Also, I
could not pin that finding to a verifiable arXiv ID (the surfaced one was
future-dated), so treat it as indicated, not confirmed, and validate locally.
Related and properly cited: "Harder Tasks Need More Experts" (**2403.07652**),
dynamic-k by cumulative routing mass, code exists.

### Depth pruning — real, but the literature's own latency number is discouraging (V)

- **ShortGPT** (**2403.03853**): Block Influence = 1 − cos(input, output) per
  layer. Llama2-7B at ~27% layers removed: MMLU 45.39 → 43.96. Llama2-13B at
  ~25%: 55.00 → 54.69. Training-free core result.
- **SLEB** (**2402.09025**, ICML'24): iterative perplexity-verified block removal,
  **20% of blocks, no retraining**, perplexity maintained. Code exists.
- **Gromov et al.** (**2403.17887**, ICLR'25): angular distance picks a contiguous
  block; up to ~half of layers — **but only after QLoRA healing**, so not
  training-free at that depth.
- **Sheared-LLaMA** (**2310.06694**) needs 50B tokens of continued pretraining.
  Disqualified.

**The honest number:** ShortGPT measured only **1.16× throughput for a 25% depth
cut** — well below the ~1.33× a linear argument gives. **(I)** That was on
CUDA/GPTQ, i.e. a more bandwidth-bound regime; on an instruction-bound decode path
removing 25% of layers removes ~25% of in-kernel work, so conversion *should* be
closer to linear here. But **no paper measures depth pruning on an issue-bound
accelerator**, so that is a prediction. Depth removal also cuts KV proportionally
(KV is per-layer) — structural, though none of these papers headline it.

**All speculation-class methods stay dead:** LayerSkip (**2404.16710**),
Draft & Verify, CALM, SkipDecode, Mixture-of-Depths. Self-speculation re-pays the
verification cost, which is precisely what the 100%-draft-acceptance result
already disproved. MoD/CALM additionally need an adaptive-depth training recipe.

---

## 5. Two things I proposed that the machine refuted

Recorded because both were plausible and both are wrong.

### `fp8_e5m2` KV cache — dead twice over (M + V)

The idea: `docs/21` measured e4m3 decode at **117 VALU ops per 4 values** against
**11 for e5m2** and **11 for int8**. So an e5m2 KV cache should be ~10× cheaper to
decode than the e4m3 everyone defaults to.

Wrong for two independent reasons:

1. **It is not selectable on ROCm.** `config/cache.py:77-78`, read from the
   install: *"CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU)
   supports fp8 (=fp8_e4m3)."* There is no int8 KV option in vLLM at all.
2. **The measurement was from the wrong kernel.** The 117-vs-11 figure is the
   **weight-decode path inside the FP8 block GEMM**. The KV cache path is a
   different kernel — ROCm paged attention reads FP8 KV via
   `fp8::vec_conversion`/`scaled_convert`. Extrapolating a GEMM instruction count
   to an attention kernel was unjustified.

And there is no tradeoff to weigh anyway: **e4m3 is also the more accurate KV
format.** K/V activations are softmax-bounded so e5m2's extra range is wasted while
its 2 mantissa bits cost real accuracy (**2502.01070** benchmarks e4m3 beating
e5m2 on MMLU across models). One thing worth carrying forward from that search:
FP8-KV damage **concentrates in long-context retrieval and is invisible to
MMLU/GSM8K** — a documented case went 91% → 13% on a 128k needle test from an
FP32-accumulation bug. That is directly relevant to the 256k llama.cpp path and to
`-ctk q8_0 -ctv q4_1`: whatever KV quant is running should be validated with a
long-context retrieval probe, not a short benchmark.

### Full-graph capture as the launch-overhead fix — already on (M)

Checked the install: default `cudagraph_mode` at O2/O3 is **`FULL_AND_PIECEWISE`**,
and both `rocm_attn.py` and `triton_attn.py` declare
`AttentionCGSupport.ALWAYS = 3`. So **decode is already fully graph-captured** and
per-token launches are already amortized.

I also expected a confound: `rocm_aiter_fa.py` declares `UNIFORM_BATCH = 2`, which
looked like it would silently downgrade the graph mode when the AITER FA patch is
active, contaminating the AITER on/off A/B. Tracing the logic in
`config/compilation.py:1339-1400`, it does not —
`FULL_AND_PIECEWISE.mixed_mode()` is `PIECEWISE`, not `FULL`, so the downgrade
branch never fires, and the decode-mode check only triggers on `NEVER`.
**No confound; the +12.8% AITER figure is clean.**

The useful consequence: launch overhead is *not* where the decode gap lives, which
independently corroborates `docs/30`/`docs/33` — the residual is in-kernel
instruction issue. The currently-running `-eager` arm is the exact control that
settles it: much worse than the graph-captured baseline means graphs are working
and the residual is in-kernel; roughly equal means graph capture is not helping
and launch cost is back in play.

---

## Ranked additions to the plan

| # | Action | Cost | Needs GPU? |
|---|---|---|---|
| A | **`--load-format sharded_state`** snapshot at TP=2 — turns the 7-hour load into a one-time cost | one long run, then config | yes |
| B | **Re-check llama.cpp master for restore-prefill-skip** (`changes/04:74`) — converts session persistence into an actual compute saving | build + one probe | briefly |
| C | **Characterise prefix caching in production** — worth more here than on a GPU box, and never measured | one bench arm with APC **on** | yes |
| D | **REAP 50% expert prune** on the 230B — halves `E` in `(n/c)·E·bytes` and the CPU-resident set | offline calibration | partly |
| E | **LMCache CPU-DRAM KV offload** (+ `PYTHONHASHSEED=0`) — spill KV to the 499 GB | build from source | yes |
| F | **CacheBlend via LMCache** for open-webui RAG | after E | yes |
| G | **top-k 10→6** — decode-only win, validate greedy *and* pass@n | config | yes |
| H | `PYTORCH_HIP_ALLOC_CONF` pinned-arena probe on a stock bf16 load | env var | yes |
| I | **SLEB/ShortGPT 20% depth** on a dense arm — would be the first published-style measurement on an issue-bound accelerator | offline + arm | partly |

A and B are the two highest-value items: one removes a 7-hour tax that currently
constrains which checkpoints are even usable, the other unlocks a caching win that
the benchmark methodology has been hiding by design.
