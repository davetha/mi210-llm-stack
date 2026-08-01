# Where the wins actually land, how big the noise is, and why W8A8 pays a tax

**Date**: 2026-08-01 · Three tracks run together. Track 1 needed no GPU;
track 2 is `benchmarks/matrix/round45_noise_floor.sh`; track 3 is source
analysis of vLLM 0.26.1rc0 with no measurement attached and none claimed.

Three questions, and the first one reframes the other two.

---

## Track 1 — the optimization work does not touch production

Every measured win in this repo (`docs/40`–`docs/45`) is a **vLLM** result:
AITER ASM flash attention, the CK int8 GEMM at 1.48× decode, PCIe P2P, 256k
paged attention. All of it assumes vLLM serving a W8A8 checkpoint.

**Production on this box does not run vLLM.** `/mnt/llm-storage/llama-swap-config.yaml`
serves everything through `llama-server` from `llama-rocm714-rpc:latest`, on
GGUF weights, over RPC:

| model | checkpoint | engine |
|---|---|---|
| `coder` | Qwen3-Coder-Next-abliterated **Q4_K_M** | llama.cpp |
| `deephat` | DeepHat-V1-7B **Q4_K_M** | llama.cpp |
| `thinking-80b` | Qwen3-Next-80B-A3B-Thinking **Q5_K_M** | llama.cpp |

`grep -ci vllm` on that config returns **0**. litellm routes `coder` →
`:8092` and `deephat` → `:8091`, both llama-swap-managed llama.cpp servers.

So this is not a deployment gap that a config edit closes — it is an **engine
difference**. The vLLM work is research on a stack production does not use.
Three honest consequences:

1. **None of the measured speedups are being realised in production today.**
   A 1.48× that nothing serves from is worth zero until something serves from
   it.
2. **Moving production to vLLM is not a flag change.** It needs W8A8 (or
   equivalent) checkpoints rather than GGUF, and would forfeit whatever
   llama.cpp-specific tuning the current setup has (`-ctk q8_0 -ctv q8_0`,
   RPC-split across cards, `-np` concurrency).
3. **Or the vLLM results stay what they honestly are** — a characterisation of
   what CDNA2 can do under vLLM, useful for anyone choosing an engine, and the
   basis for the upstream reports this project has generated.

No recommendation is made here between (2) and (3); that is a deployment
decision, not a measurement. What is recorded is that the question had never
been asked, and the answer was not the assumed one.

---

## Track 3 — int8 activation quant has no fusion path anywhere in vLLM

`docs/45` left `dynamic_scaled_int8_quant` (498 ms, **6.2% of the decode
window**) as the most concrete unexamined lead, on the suspicion that
`fuse_norm_quant` was fp8-only. It is worse than that, and the suspicion is
confirmed at every layer:

| layer | finding |
|---|---|
| an int8 fused rms+quant kernel in vLLM | **does not exist** — no match in `csrc/` or `vllm/` |
| `rms_quant_fusion.py` `QUANT_OPS` / `FUSED_OPS` | **FP8 and FP4 only**; entries are `kFp8StaticTensorSym`, `kFp8DynamicTokenSym`, `rms_norm_static_fp8_quant`, … — zero int8 |
| `rocm_aiter_fusion.py` | the string `int8` appears **zero times** in the whole file |
| quant matcher classes in vLLM | **`MatcherQuantFP8` is the only one** |
| any int8 fusion pattern under `vllm/compilation/` | **none** |

**But AITER has the kernel.** `_rocm_aiter_rmsnorm_fused_dynamic_quant_impl`
(`vllm/_aiter_ops.py:723`) asserts

```python
assert quant_dtype in [torch.int8, FP8_DTYPE]
```

and calls `aiter.rmsnorm2d_fwd_with_dynamicquant`. vLLM already registers it as
`rocm_aiter_rmsnorm_fused_dynamic_quant`.

So the shape of the gap is precise: **a W8A8 model pays a separate activation
quantization pass that an FP8 model does not**, not because the fused kernel is
missing but because vLLM's entire fusion infrastructure is FP8-only. On this
box that is 6.2% of decode, every token, structurally.

**What exploiting it would take** — and this is feature work, not a gate flip:
a `MatcherQuantInt8` class (none exists), int8 pattern registration in the
AITER fusion pass, numeric verification, and the master gate below. Ceiling is
the 6.2% minus whatever the fused kernel itself costs, so call it low single
digits. It is also a clean upstream contribution if anyone wants one.

---

## The sixth gate: `is_enabled()` silently drops four fusion passes

`docs/45` surveyed 17 `is_*_enabled` gates and carved out five. It missed the
one with the widest reach. `pass_manager.py:162-171`:

```python
if self.pass_config.fuse_norm_quant:
    if rocm_aiter_ops.is_enabled():
        self.passes += [RocmAiterRMSNormQuantFusionPass(config)]
if self.pass_config.fuse_allreduce_rms:
    if rocm_aiter_ops.is_enabled():
        self.passes += [RocmAiterAllReduceFusionPass(config)]
```

`is_enabled` is `@if_aiter_supported` → `on_mi3xx()`, so on CDNA2 it returns
`None` and **four ROCm fusion passes are never added**:
`RocmAiterRMSNormQuantFusionPass`, `RocmAiterAllReduceFusionPass`,
`RocmAiterTritonAddRMSNormPadFusionPass`,
`RocmAiterSiluMulFp8GroupQuantFusionPass`.

Three are FP8-oriented and, per track 3, cannot help an int8 model regardless.
**The fourth is the interesting one.** All four allreduce ASM objects ported to
gfx90a — `all_reduce.co`, `allreduce_rmsnorm_N8192.co`,
`allreduce_rmsnorm_qnt_N8192.co`, `allreduce_layernorm_N8192.co` — and
`docs/37` dismissed them partly because *"there is no XGMI for it to accelerate
anyway"*, an argument round 31 refuted by measuring PCIe P2P at 26.98 GB/s and
+11.2% prefill. At TP=2 the per-layer collective is a fixed cost the CK GEMM
cannot touch, and fusing it with the norm is the right shape of attack.

`configs/enable_aiter_master_gate_gfx90a.py` carves it out. It is deliberately
a **separate script** from the five-gate patch, because the blast radius is
not comparable: `is_enabled()` has ~22 consumers across 10 files, including
**two in `v1/attention/backends/rocm_aiter_fa.py`** — the flash attention path
currently delivering 1.19–1.33× prefill. Today it sees `None` and the stack
works; flipping it to `True` changes branches inside a working optimization.

Any round using it must assert that the existing wins survive: `LoadKernel`
count > 0 (FA ASM) and `module_gemm_a8w8` present (CK GEMM). **Untested as of
this document.**

---

## Track 2 — the noise floor, measured

Five runs of the **identical** configuration through **separate server
launches** — which is what an A/B arm actually is, since each arm relaunches.
`bench_matrix.py` already medians 3 reps inside one server, so this measures
the variance that survives that.

| metric | n | mean | stdev | CV | spread | **95% bar** |
|---|---:|---:|---:|---:|---:|---:|
| **decode t/s** | 5 | 82.66 | 1.073 | **1.30%** | 3.16% | **1.036×** |
| longctx prefill | 5 | 6979.13 | 11.35 | 0.16% | 0.38% | 1.005× |
| longctx ttft | 5 | 3.69 | 0.002 | 0.06% | 0.15% | 1.002× |
| cold16k prefill | 5 | 8562.13 | 6.28 | 0.07% | 0.16% | 1.002× |
| cold16k ttft | 5 | 1.77 | 0.004 | 0.25% | 0.61% | 1.007× |

The last column is the smallest ratio a **single-arm A/B** must exceed to mean
anything at 95% confidence, given two independent launches (sd of a difference
is √2·σ).

**Decode is ~8× noisier than prefill.** That single fact explains the shape of
this project's results: every prefill finding has replicated across rounds,
while several decode findings have wobbled.

### Retroactive audit — what survives

| result | metric | ratio | verdict |
|---|---|---:|---|
| CK int8 GEMM | decode | 1.480 | **SURVIVES** |
| async scheduling | decode | 1.110 | **SURVIVES** |
| AITER FA (decode) | decode | 1.076 | **SURVIVES** |
| AITER FA @25k | prefill | 1.332 | **SURVIVES** |
| AITER FA @16k | prefill | 1.190 | **SURVIVES** |
| PCIe P2P | prefill | 1.112 | **SURVIVES** |
| MoE stride padding | decode | 1.041 | BORDERLINE |
| **vLLM 0.23.1 → 0.26.1rc0** | decode | **1.035** | **INCONCLUSIVE — inside noise** |
| clock pinning | decode | 1.033 | inconclusive (was reported null — consistent) |
| AITER MoE | decode | 1.024 | inconclusive (was called a mild regression) |
| capture geometry 131k/32k | decode | 1.014 | inconclusive (was reported null — consistent) |

**Three corrections follow, and one vindication.**

1. **The version-climb number is retracted as a measurement.** `docs/42` and
   `benchmarks/README.md` both carry "vLLM 0.23.1 → 0.26.1rc0: 1.035× decode".
   The bar is 1.036×. It is inside the noise and must be read as *no
   demonstrated decode difference*, not as a small gain. (Round 39's
   **sync-path** comparison at 0.992× was always described as flat, and that
   reading is unaffected.)
2. **AITER MoE's 0.977× is inconclusive**, not a mild regression. `docs/45`
   hedged it as "sitting at the ±2–3% noise floor", which was the right
   instinct; it is now quantified. The *reason* MoE is closed — the flag never
   reaches the ASM kernels, round 43's kernel diff — does not depend on the
   throughput number at all, so the closure stands.
3. **Every null reported as a null is confirmed as one.** Clock pinning and
   capture geometry both land inside the interval, exactly as they were
   reported.
4. **Round 44's refusal to call stride padding a regression was correct.** At
   1.041× it only just clears the bar, and its control was the 85.16 outlier —
   which now sits outside this five-run cluster entirely (81.38–83.99). Had it
   been called a regression, this round would be retracting that too.

### The rule going forward

A decode A/B on this rig needs **>1.036×** to be reportable from a single pair
of arms. Below that, either run repetitions or report it as inconclusive.
Prefill and TTFT need only ~1.005×, which is why they carry the load in every
solid result here.

---

## Binary audit: the ported tree is 80% dead weight, and the scoreboard was wrong

A full re-enumeration of `aiter_meta/hsa/{gfx90a,gfx942}` (1425 gfx942 objects,
242 ported) turned up three families this repo had never listed, and one of
them is the largest single entry in the tree.

| family | gfx942 | gfx90a | reachable? |
|---|---:|---:|---|
| `fmoe` | 839 | 8 | measured 0.977× — closed |
| `fmoe_2stages` | 186 | 0 | — |
| **`fmha_v3_bwd`** | **156** | **138** | **NO — training only** |
| `pa` | 56 | 8 | see below |
| `fmha_v3_fwd` | 56 | 48 | **YES — in use** |
| loose root | 45 | 7 | 4 allreduce + 3 orphans |
| `mla` | 26 | 11 | no MLA model |
| `topksoftmax` | 22 | 22 | no (`docs/45`) |
| `bf16gemm` | 22 | 0 | hardware-blocked, below |
| `i8gemm` | 9 | 0 | — |
| `fp8gemm_blockscale` | 6 | 0 | no FP8 on CDNA2 |
| `topk_per_row_{decode,prefill}` | 2 | 0 | gfx940+ ISA |

**`fmha_v3_bwd` is 138 of the 242 ported objects — 57% of everything the
repatcher produced — and it is the flash-attention *backward* pass.** Its only
caller is `aiter/ops/mha.py:1287`; `fmha_v3_bwd`, `mha_bwd` and
`flash_attn_backward` appear **zero times** in vLLM, which never runs a
backward pass. It is doubly dead: `enable_gfx90a_asm_paths.py` patches only the
forward arch gates, so even a training workload could not reach them.

**So the honest score is 48 useful of 242 ported, not 242.** Every prior
statement of the form "242 objects ported" overstates the useful yield by 5×.
`docs/37` already said only `fmha_v3_fwd`'s 48 are reachable; this quantifies
what the rest actually is.

### `pa` is wired after all — and still cannot fire here

`docs/37` recorded `pa` as having "no call site". That is wrong as stated. The
live route is `v1/attention/backends/rocm_aiter_fa.py:1318` →
`rocm_aiter_ops.paged_attention_common()` → `aiter/ops/attention.py` →
`pa_fwd_asm`. (`rocm_aiter_ops.pa_fwd_asm` itself has no direct callers, which
is probably what produced the original claim.)

Three gates stand in front of it, and two are closed for us permanently:

1. `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` (default `False`) — the call sits under
   `is_shuffle_kv_cache_enabled()`. This one is just a flag.
2. `aiter/ops/attention.py:172` — *"ASM kernel only supports head_size == 128"*,
   hard `return False` otherwise. The **production** model
   (Qwen3-Next-80B, `head_dim: 256`) can never reach it. The 30B bench model
   (`head_dim 128`, 32 heads) does qualify.
3. A dispatch heuristic `num_seqs * num_heads > 2 * cu_num` → **> 208** on a
   104-CU MI210. Batch-1 decode gives `1 × 32 = 32`. **It cannot fire in the
   regime every benchmark here measures**, only at concurrency ≳ 7.

So `pa` is closed for production by head_dim and closed for our decode
benchmark by the heuristic. It would only ever appear in a high-concurrency
throughput test on a head_dim-128 model — a different experiment than any run
in this repo.

### `bf16gemm` 0/22 is hardware-blocked — close it permanently

Every `bf16gemm` kernel is a split-K GEMM whose cross-workgroup reduction
epilogue uses `global_atomic_pk_add_bf16`. Verified directly:

```
$ echo 'global_atomic_pk_add_bf16 v0, v[0:1], v2, off' | llvm-mc -mcpu=gfx90a
error: instruction not supported on this GPU (gfx90a)
```

CDNA2 has `global_atomic_pk_add_f16` and `global_atomic_add_f32` but not the
packed bf16 atomic, which arrived in gfx940. Emulating it with a 32-bit CAS
loop changes the instruction count, which a byte-length-preserving repatcher
cannot do. This matches `docs/25` §3's finding for the 539 blocked kernels.

### The repatcher's rejections are correct — 1183 fully accounted

| reason | count |
|---|---:|
| fp8/bf8 MFMA or convert | 650 |
| int8 MFMA (gfx942 wide forms) | 482 |
| fails to assemble for gfx90a | 51 |
| **total** | **1183** (+242 = 1425 ✓) |

Top blockers by object count: `v_mfma_f32_16x16x32_fp8_fp8` (640),
`v_cvt_pk_fp8_f32` (531), `v_mfma_i32_16x16x32_i8` (480). Of the 51 assembler
failures, 37 are `global_atomic_pk_add_bf16` and 2 are gfx940+ scalar ops.

**No category was excluded too aggressively in practice.** Two latent
imprecisions were found and both unlock exactly zero kernels today: the
`_i8\b` arm would also match `v_dot4_i32_i8`, which *does* assemble on gfx90a
(no object in the tree uses it), and the probe rejects gfx942 cache modifiers
`sc0/sc1/nt` where gfx90a spells them `glc/slc` (every affected file is
independently blocked by the bf16 atomic). The `smfmac` and `_xf32` arms never
fire at all — zero occurrences in the tree.

An encoding-drift check of all 242 ported objects — re-encoding every
*non-substituted* instruction and comparing bytes — found no real drift. The
byte-patch approach is sound.

### Also confirmed independently

The 8 ported `fmoe` objects are all **fp16**: `fmoe_bf16_noquant_g1u0_silu.csv`
on gfx90a is header-only, every bf16 row pruned. So the ASM MoE path does not
exist for a bf16 model on this card at all — an independent confirmation of the
`docs/45` closure that does not rely on the 0.977× measurement.

`pa_a16w16_b16.co` and `pa_a16w16_f16.co` are orphans: referenced nowhere in
`aiter`, `aiter_meta` or `vllm` outside the wheel's own file manifest.

**Verdict: no unexploited ASM opportunity remains.** The largest unlisted
family is inference-irrelevant, the only structurally live new path is blocked
by head_dim and by a concurrency heuristic, and the rejection logic is correct.

---

## Round 46: the master gate works, and the allreduce lead closes three gates deep

`round46_allreduce_fusion.sh`, TP=2, control (`aiterops`) vs `mastergate` with
`--compilation-config '{"pass_config":{"fuse_allreduce_rms":true}}'`.

| workload | metric | control | AR-fused | factor | bar |
|---|---|---:|---:|---:|---:|
| longctx | decode | 82.56 | 83.19 | 1.008× | 1.036× |
| longctx | prefill | 6968.98 | 6987.25 | 1.003× | 1.005× |
| cold16k | prefill | 8555.47 | 8567.64 | 1.001× | 1.002× |
| cold16k | ttft | 1.78 | 1.77 | 0.998× | 1.007× |

**Every metric is inside the round-45 noise floor.** And the control's 82.56
decode lands on round 45's five-run mean of 82.66 — a sixth independent launch
falling inside the interval, which validates that measurement too.

### The good news: no regression

Both arms show **4 ASM code objects** and **CK GEMM present**. The stated risk —
that carving out `is_enabled()` would disturb the working flash-attention path
through its two consumers in `rocm_aiter_fa.py` — **did not materialise**. Those
two call sites are `fused_rope_kvcache_supported()` and
`fused_qk_norm_rope_kvcache_supported()`, capability *advertisements* for
fusions whose own `pass_config` flags default off, not the ASM dispatch. The
1.19–1.33× prefill and 1.48× decode wins are untouched.

### The gate works — and the fusion still refuses

The carve-out did exactly what it was supposed to. From the fused arm's log:

```
Enabled custom fusions: norm_quant, allreduce_rms, mla_dual_rms_norm
```

Those passes are now added where previously `pass_manager.py` skipped them
entirely. Then:

```
AITER allreduce fusions are disabled because AITER Custom All Reduce is not
enabled. Set VLLM_ROCM_USE_AITER_CUSTOM_AR=1
AllReduce fusion pass is disabled.
```

So this arm is **under-configured, not a disproof** — predicted before the run
from `config/vllm.py:1924`, which sizes the fusion via
`is_custom_all_reduce_enabled()`.

### Why it is not worth chasing further

Setting that flag is not sufficient, and two independent facts close the lead:

1. **Custom all-reduce is MI300-only at the platform layer.**
   `platforms/rocm.py:900`: `use_custom_allreduce()` returns
   `any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])` — gfx90a excluded by
   design — which force-sets `disable_custom_all_reduce = True`
   (`config/parallel.py:1017`). Reaching the fusion needs a **third** carve-out,
   into collective code, where a defect produces wrong reductions rather than a
   slow kernel. That is a different risk class from every gate opened so far.
2. **The ASM objects are shape-specialised and do not match this model.** The
   ported files are `allreduce_rmsnorm_**N8192**.co`,
   `allreduce_rmsnorm_qnt_N8192.co`, `allreduce_layernorm_N8192.co`. This
   model's hidden size is **2048**. Even with all three gates open, the N8192
   kernels have no shape to match here.

Point 2 stands regardless of point 1, so the lead closes on shape grounds
alone. `docs/37`'s original dismissal reached the right answer via the wrong
argument ("no XGMI", refuted by round 31); the correct reasons are
MI300-gating in the platform layer and N8192 specialisation.

**What survives:** `configs/enable_aiter_master_gate_gfx90a.py` is validated as
safe — it enables three fusion passes with no regression to either existing
win — and is available if a future model or fusion needs it. It is **not**
wired into `Dockerfile.vllm-mi210`, because on this model it changes nothing
measurable.
