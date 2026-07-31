# The shareable build — what the patches buy, and what to run

**Date**: 2026-07-31 · **Hardware**: 2× MI210 (gfx90a / CDNA2, 64 GB HBM2e each),
EPYC 74F3, 499 GB DDR4, ROCm 7.14, PCIe only (no XGMI)

Everything this repo established about running vLLM on CDNA2, reduced to one
buildable image and one set of recommendations. Build instructions in
`configs/Dockerfile.vllm-mi210`; the numbers below are all measured on this box.

---

## 1. Why a Dockerfile and not a fork

The obvious move at this point is to fork vLLM, fork AITER, and carry the
patches as branches. That is the wrong shape for what we actually have.

**The payload is two patches that matter.** Of the 242 `.co` objects the
repatcher produces, only `fmha_v3_fwd`'s 48 are reachable — `pa`, `mla`, `fmoe`,
`topksoftmax` and `allreduce_*` are each unreachable for a *different* reason
(§5). A fork would be 190 dead binaries and four small source edits.

**Patch scripts fail louder than forks do.** Every script here asserts on its
expected match count and refuses to double-apply. When upstream moves the code,
the build dies with "expected the gfx9 128k anchor exactly once in rocm.py,
found 0 — upstream reordered the gate terms". The equivalent event in a fork is
a merge conflict at best, and at worst a silent semantic drift in a file that
still applies cleanly. Given that **every failure mode in this project has been
a silent one** — correct output from a fallback path, at a fraction of the
speed — loud beats convenient.

**vLLM 0.23 is moving fast.** Rebasing a fork against a 0.x project with weekly
releases is real, recurring work in exchange for nothing the patch scripts don't
already do.

**The `.co` files are generated artifacts.** `repatch_gfx942_to_gfx90a.py`
disassembles each gfx942 kernel, substitutes CDNA2 mnemonics and *re-assembles*
for gfx90a, emitting only what the assembler accepts. Portability is proven, not
asserted. Checking 242 binaries into git would replace a proof with a snapshot.

**What should go upstream, separately and on its own merits:**

| finding | why it is upstream's problem, not ours |
|---|---|
| `pass_manager.py:158` `NameError` | Architecture-neutral. `AllReduceFusionPass` is referenced on ROCm but imported only under `is_cuda()`; ROCm is `is_cuda_alike()` but not `is_cuda()`. **Latent, not a live crash** — see below. |
| `save_sharded_state` corrupts AWQ | Silently. Verified twice (`docs/28`). Round-trips bf16 correctly, so the failure is format-specific and undocumented. |
| `llama-bench` multi-GPU depth fault | `-d 4096` on two cards faults in the depth-priming path; `llama-server` is fine to 150k on the same cards (`docs/36`). |

> **Severity correction on the `NameError`,** from outside review by
> [Andrei-Dr](https://github.com/Andrei-Dr). An earlier draft of this section
> said any ROCm host with `fuse_allreduce_rms=True` crashes instead of
> degrading. **The auto path cannot reach it.** `config/vllm.py:140` gates
> auto-enabling on `rocm_aiter_ops.is_enabled() and tensor_parallel_size > 1` —
> the same condition that selects `RocmAiterAllReduceFusionPass` — so the
> `else` branch holding the undefined name is unreachable under auto-config,
> and the CUDA pass is dead code on ROCm by construction. It is reachable only
> through an explicit `-O` / `--compilation-config` override. Still worth
> reporting; **not** worth reporting as "ROCm crashes by default", which is
> what our arms actually did because they set the flag explicitly.

---

## 2. The image

```bash
docker build -f configs/Dockerfile.vllm-mi210 -t vllm-mi210:latest .

# then, on a machine with the cards:
docker run --rm --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --entrypoint /usr/local/bin/verify-gfx90a vllm-mi210:latest
```

It replaces the two ad-hoc images this project had been benchmarking against
(`rocm-vllm-aiter-gfx90a:latest` and `:pa256k`, built 13 h apart by committing
from a running container). That split is not cosmetic — it produced one
unattributable result in round 26, because the serve script defaulted to
`:latest` while the save step hardcoded `:pa256k`. One image, one build, no
drift.

**The build needs no GPU.** Every step — the AITER install, the repatcher, the
file edits, the `_rocm_C` rebuild — is CPU-only, with `GPU_ARCHS=gfx90a`
standing in for device detection. This matters more than it sounds: it means the
image can be built in CI, on a laptop, by anyone.

**Verification is split, deliberately.** The build verifies what it can — that
every patch landed, that ~242 code objects exist, that the rebuilt `_rocm_C` is
not byte-identical to stock, that no stale `module_fmha_v3_fwd*.so` shipped
(a stale JIT module whose kernarg layout predates the `.co` files is exactly how
`pa_fwd_asm` appeared broken for weeks — `docs/18`). It cannot verify that
anything *runs*: `import aiter` calls `get_gfx_runtime()`, which shells to
`rocminfo` and — unlike the codegen path — ignores `GPU_ARCHS`. So
`verify-gfx90a` does the rest on a card, including the check that matters most:
that a **gfx90a `.co` was actually loaded**, not just that the answer was right.

> A correct answer proves nothing here. The fallback paths are correct. They are
> just slow, and no throughput number distinguishes them reliably from a
> half-patched image. The `.co` load line does.

---

## 3. Before / after — what each patch actually buys

### 3.1 256k paged attention — the largest single win

Stock vLLM's ROCm custom paged attention dispatches its reduction through an
**eight-entry switch**: 8 NPAR_loops × 64 (warp) × 256 (partition) = 131,072,
exactly 128K. Above it, the kernel is rejected at *CUDA-graph capture time*
against the **configured** max, so a 256k server bakes the Triton fallback into
every request — including short ones.

**Measured in one session, round 32.** Stock `rocm/vllm` against `vllm-mi210`,
same box, same AWQ-Int4 checkpoint, both at `--max-model-len 262144`, prompt
**205,291 tokens**, 3 reps per arm:

| metric | stock | patched | factor |
|---|---:|---:|---:|
| prefill @16k | 2,661.1 | 2,636.8 | 0.99× |
| prefill @205k | 515.5 | 512.9 | 0.99× |
| TTFT @205k | 398.4 s | 400.5 s | 1.01× |
| **decode @205k** | **0.89** | **11.62** | **13.08×** |

**Four control metrics inside 1%, one metric 13×.** That isolation is the whole
value of the round: it establishes the two images are otherwise
indistinguishable, so the decode result cannot be attributed to anything else.
And it matches the mechanism exactly — the patch adds cases 9–16 to the
*reduction* dispatch, which only the decode path uses. Prefill never touches it,
and prefill did not move. Both arms passed the `ACKNOWLEDGED` correctness probe.

At 205,291 tokens, `npar_loops = ceil(ceil(205291/256)/64) = 13` — a case that
exists only because of this patch.

> Earlier revisions of this section cited **17.18 vs 1.73 t/s (9.9×)** from
> `results/t35-awq-{128,256}kcfg-ctx110k.json`. That pair is real but was
> measured at a different configuration in a different round; the table above
> supersedes it. `docs/23` separately measures 0.7485 t/s at 241k over two reps.

The patch adds cases 9–16, raising the ceiling to 16 × 64 × 256 = 262,144.
Validated in production rather than in a unit test:

```
ROCM_AITER_FA
npar_loops': 10
```

`npar_loops = ceil(ceil(160000/256)/64) = 10` — **a value stock vLLM cannot
produce**, since its switch stops at 8. So this is direct evidence the custom
kernel ran, not an inference from throughput. And the curve behaves:

| depth | decode t/s |
|---:|---:|
| 120,000 | 42.15 |
| **150,000** | **39.43** |

A 6% decline straight through the 131,072 boundary, where stock takes a ~10×
cliff. Full curves in `docs/36`.

> **Not established**: numerical correctness above ~200k. The template is valid
> and the buffers are sized right, but no reference comparison has been run at
> 250k. Run `tests/test_rocm_pa_256k.py` before trusting a result up there.

### 3.1b The int8 MoE gate — the only binary result here

Stock `rocm/vllm` **cannot serve W8A8 at all** on gfx90a. Round 32, same session:

| arm | result |
|---|---|
| W8A8 TP=2, **stock** | ✗ `NotImplementedError: No Int8 MoE backend supports the deployment configuration` |
| W8A8 TP=2, **patched** | ✅ 5,956 t/s prefill @16k · 4,669 @25k · **51.06 t/s decode** |

Not a percentage — a yes/no. The format §4 recommends as the sensible default
does not load on an unpatched image.

**This win has a shelf life.** AMD fixed it upstream in
[`1053e248f0`](https://github.com/vllm-project/vllm/commit/1053e248f0) (PR
#46765, `[ROCm][Quantization][5/N]`, 2026-07-27), broadening the gate to
`or current_platform.is_rocm()` — wider than our gfx9-scoped version. It landed
as a side effect of a Quark refactor, twelve days *after* the
`rocm7.14.0_cdna_..._vllm_0.23.0` image was published, so it is in no released
`rocm/vllm` build yet. Drop `configs/enable_int8_moe_rocm.py` when AMD ships an
image carrying it.

### 3.2 AITER ASM flash attention — real, but narrow

Before, on a stock image, the engine logs something that looks completely
healthy:

```
Overriding with ROCM_ATTN out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN']
```

AITER is **not in the candidate list at all**, so `VLLM_ROCM_USE_AITER=1` does
nothing whatsoever. After:

```
Overriding with ROCM_AITER_FA out of potential backends:
  ['ROCM_AITER_FA', 'ROCM_ATTN', 'TRITON_ATTN']
```

and `fwd_hd128_bf16_rtna_group.co` loads 44× per arm.

> **Getting AITER into the candidate list is not the same as getting it used,
> and this repo confused the two.** `_get_backend_priorities()` appends
> `ROCM_ATTN` **unconditionally first**, so `ROCM_AITER_FA` is reached only when
> `ROCM_ATTN` is invalid — which it essentially never is. Selection requires
> `configs/prefer_aiter_fa_gfx90a.py` **and** `VLLM_PREFER_AITER_FA=1`.
>
> The first build of `vllm-mi210` omitted that patch, and nothing caught it:
> the build passed, `verify-gfx90a` passed all five checks, and
> `is_aiter_attention_supported()` answered `True` — while **zero ASM kernels
> ran**. Round 32 found it by counting `LoadKernel` lines. Both the Dockerfile
> and the verifier now check the ordering, not just candidacy.
>
> **So round 32 cannot measure this patch.** Its `awq32k` arms are `ROCM_ATTN`
> vs `ROCM_ATTN` — 0 `LoadKernel` in both — and came out at 0.99–1.01×, which
> is evidence that the two images are otherwise identical and **no evidence at
> all about AITER**. The numbers below remain the best measurement of the ASM
> path and come from `benchmarks/vllm-aiter-asm-gfx90a.md`, on an image where it
> was actually selected.

Qwen3-30B-A3B, output tokens/s:

| prompt | concurrency | stock | ASM | factor |
|---:|---:|---:|---:|---:|
| 128 | 1 / 8 / 32 | — | — | **1.00–1.02×** (nothing) |
| 4,096 | 1 | 27.0 | 27.4 | 1.01× |
| 4,096 | 8 | 50.7 | **62.4** | **1.23×** |
| 4,096 | 32 | 64.9 | **79.9** | **1.23×** |

At concurrency 32 that is also **1.25× better TPOT** (394 → 315 ms) and **14%
lower TTFT** (12.5 → 10.7 s).

**Be honest about the shape of this win.** It is 1.23× on long prompts under
concurrency and *nothing at all* otherwise — a 128-token prefill is too small to
occupy an MI210, and single-stream decode is bound by streaming 28 GB of weights,
not by the attention kernel. Essentially all of the gain is prefill: the ASM
*decode* kernel, measured at 1.72× in isolation, is worth about 1% end-to-end,
which is inside this harness's noise. In isolation the kernel is genuinely fast —
1.13–1.86× vs PyTorch SDPA, **89.9 TFLOP/s, 50% of the card's bf16 peak** — but
serving is not a microbenchmark.

### 3.3 The repatcher

242 of 1,425 gfx942 code objects translate to gfx90a; 1,183 do not (fp8/int8/xf32
MFMA that CDNA2 has no encoding for). That is the tally on aiter **v0.1.19**;
`repatch_gfx942_to_gfx90a.py`'s docstring records 242/1,180 against 0.1.17, so
three kernels were added upstream and none of them ported. The count of portable
objects has not moved. Of the 242, **only `fmha_v3_fwd`'s 48
objects are reachable from vLLM** — see §5. The other 194 are installed because
the repatcher is a whole-tree tool and their presence costs nothing, not because
they do anything.

### 3.4 PCIe peer-to-peer — a premise that was wrong for the whole project

`env/gfx90a-common.env` carried `NCCL_P2P_DISABLE=1` from the beginning,
justified by *"the two MI210s have NO xGMI bridge and cannot peer-to-peer"*. The
first half is true. **The second does not follow from it, and it is false.**
Found by outside review from [Andrei-Dr](https://github.com/Andrei-Dr).

The driver says so — `p2p_links/0/properties` reports type 2
(`HSA_IOLINKTYPE_PCIEXPRESS`; XGMI would be 11), 32000 MB/s, `flags 3` =
Override + NonCoherent, so bit 4 `NoPeerToPeerDMA` is **clear**. And the runtime
agrees:

| | |
|---|---:|
| `can_device_access_peer(0,1)` | **True** (and reciprocal) |
| cuda:0 → cuda:1 | **26.98 GB/s** (84% of PCIe 4.0 x16) |
| via pinned host, 2 hops | 14.16 GB/s |

A host-staged copy makes two trips over the same link and cannot exceed about
half the link rate, so 26.98 GB/s is not reachable that way — the copy peers.
Every TP=2 allreduce ever measured in this project took the 14 GB/s path.

The A/B (round 31, 3 reps per arm, Qwen3-30B-A3B W8A8 at TP=2):

| workload | metric | P2P off | P2P on | delta |
|---|---|---:|---:|---:|
| cold 16k | prefill t/s | 7,279.2 | **8,093.0** | **+11.2%** |
| cold 16k | TTFT s | 2.085 | **1.869** | **−10.3%** |
| longctx | prefill t/s | 6,190.8 | **6,760.7** | **+9.2%** |
| longctx | TTFT s | 4.162 | **3.811** | **−8.4%** |
| longctx | decode t/s | 54.06 | 54.76 | +1.3% — noise |

**Prefill gains, decode does not**, which is the expected shape rather than a
disappointment: prefill allreduces move large activation tensors and are
bandwidth-bound; decode allreduces move small buffers and are latency-bound.
Andrei explicitly predicted this could happen — *"P2P could be real and still
not move decode"* — and it did.

The control validates the harness: its cold-16k prefill is **7,279.2** against
`docs/28`'s published **7,278**, and it loaded in 59.4 s against 60 s.

It also refutes one of the risks he raised against his own finding. `ACSCtl`
does read `ReqRedir+ CmpltRedir+` on both upstream bridges, so peer traffic is
redirected through the root complex rather than routed bridge-to-bridge — but
the 26.98 GB/s **already includes that redirect**. `pcie_acs_override=downstream`
remains a further, untested lever.

**The default is now `NCCL_P2P_DISABLE=0`.** The RCCL setup stall that motivated
the flag was real and is still unexplained; it did not recur across the four
TP=2 collective setups run here. That is evidence, not proof — revert with
`NCCL_P2P_DISABLE=1` and label the run.

### 3.5 Not patches, but the same size of win — configuration

| change | before | after | factor |
|---|---:|---:|---:|
| bf16 load via `--load-format sharded_state`, **per restart** | 12,366 s | **114.65 s** | **108×** |
| `GGML_CUDA_FORCE_MMQ` (llama.cpp), VRAM-saturated | 204.5 | 214.3 | 1.048× |
| `GGML_CUDA_FORCE_MMQ`, otherwise | — | — | ~1.00× (and −1.9% in places) |

`sharded_state` is the single largest number in this document and it required no
patch at all — it is a stock flag whose flat `param_data.copy_()` bypasses the
per-tensor `weight_loader` entirely. Throughput after conversion is unchanged
(8,743 → 8,755 t/s), so this is purely load time. See `docs/34`.

> **The 108× is per restart, not free.** Creating the snapshot runs the model
> through the *same* slow loader once, so the honest statement is
> **12,366 s every start → 12,366 s once, then 114.65 s every start.** It pays
> for itself on the second start and the ratio is real, but the slow load does
> not disappear — it is paid once and amortised. It also costs a second full
> copy of the weights on disk, is bound to the `--tensor-parallel-size` it was
> written at (the file glob embeds `rank=`), and snapshots post-
> `process_weights_after_loading` runtime tensors, so it must be regenerated
> after a vLLM bump.
>
> **What the pair is not confounded by.** Both endpoints are the same
> `Loading weights took` line, same model (28.51 GiB/rank), same rank, same
> TP=2, same box — and both reproduce: the stock loader measured 12,365.46 s
> and 12,357.66 s on separate days (0.06% apart), `sharded_state` 114.65 s and
> 116.72 s (1.8% apart). Page cache is the obvious suspect, since 57 GiB fits
> easily in 499 GB of RAM, but **neither endpoint is I/O-bound**: `docs/25`
> measured storage delivering this model in 13.19 s at 4.63 GB/s, and 114.65 s
> for 57 GiB is ~0.5 GB/s — already 9× slower than the disk. A caching effect
> cannot open a 108× gap between two loads that are both limited by something
> other than reading bytes.

`FORCE_MMQ` needs its own image and buys ~5% in one narrow case. **We are not
shipping it.** It is documented in `configs/Dockerfile.llama-forcemmq` for
anyone who wants it. Note that the first A/B read +65% prefill / 5.8× decode —
that run was contaminated by a container that escaped the FIFO lock and left
42 GiB of KV cache resident (`docs/29`). The clean re-run is the table above.

---

## 4. What to run — good, better, best

There is no single ranking, because **best prefill and best decode come from
different engines, different quantizations and opposite placement strategies**.
Pick by workload.

### The short answer

| you want | run | prefill | decode |
|---|---|---:|---:|
| **Best all-round** | `RedHatAI/*-quantized.w8a8`, vLLM TP=2 | 7,278 | 43.4 |
| **Fastest, if you have the VRAM** | bf16 + `--load-format sharded_state` | **8,755** | **65.2** |
| **Best decode at 80B** | `cyankiwi/*-AWQ-8bit` (W8A16), TP=2 | 6,679 | **51.3** |
| **Best prefill at 80B** | `RedHatAI/*.w8a8`, TP=2 | **7,253** | 45.2 |
| **Smallest that still flies** | AWQ-Int4, TP=1 | 3,002 | 17.2 |
| **Long context (>128k)** | GGUF Q8_0 / Q4_K_M, llama.cpp | 3,326 | 29.1 @230k |
| **357B, long prompt short answer** | AWQ-Int4 + prefetch offload + `--enforce-eager` | **800.7** | 0.62 |
| **357B, anything that generates** | GGUF UD-Q2_K_XL, llama.cpp | 208.5 | **11.52** |

### Good — it works

**AWQ-Int4** at 30B. 15.7 GiB on one card, 3,002 prefill, and 2.9× faster than
FP8 on the same model. The memory pick, and the only sensible single-card option
below 80B.

Why it is only "good": int4 weights are **dequantized to bf16 before the MFMA**.
CDNA2 has no int4 matrix instruction, so the quantization buys memory, not math,
and you pay the dequant on every pass. It also degrades badly with scale — see
the WNA16 trap in §5.

### Better — the default

**W8A8 `compressed-tensors`** (`input_activations` type `int`), vLLM TP=2. This
is what to run if you are not sure. 60 s load, 7,278 prefill, 43.4 decode at 30B;
7,253 / 45.2 at 80B.

Why: int8 is the **one quantized format that reaches CDNA2's matrix cores
natively**. `v_mfma_i32_16x16x16i8` exists; there is no fp8 or fp4 equivalent.
Both weights and activations stay int8 into the GEMM, so unlike int4 there is no
dequantization tax.

> **W8A8 requires `configs/enable_int8_moe_rocm.py`, and the first build of this
> image shipped without it.** Loading a W8A8 MoE on an unpatched image dies at
> startup with `NotImplementedError: No Int8 MoE backend supports the deployment
> configuration` — vLLM's only Int8 MoE backend gates the scheme behind
> `current_platform.is_cuda() and has_device_capability((7,5))`, a CUDA-only
> predicate three lines below a device check that uses `is_cuda_alike()` and
> accepts ROCm. The patch is now applied at step 4b and gated twice.
>
> **A correction, and a lesson about how it was found.** This paragraph first
> claimed that *none* of the images on the box carried the patch, and that
> `docs/28`'s W8A8 numbers therefore came from an image that no longer existed.
> That was wrong. `:latest` and `:pa256k` both carry it. The claim came from
> grepping for `is_rocm() and on_gfx9()` — a string that appears in
> `enable_int8_moe_rocm.py`'s **docstring**, as a simplified illustration, and
> never in the code the script actually writes. The real replacement is a
> multi-line block that imports `on_gfx9` inside an `is_rocm()` guard and
> assigns `_rocm_int8`.
>
> The same wrong string was used as the build gate, which then failed a build in
> which the patch had applied correctly. **Verify against the emitted text, not
> against the documentation of the emitted text.** Only the `vllm-mi210` image
> genuinely lacked the patch; that gap was real and is fixed.

**W8A16** (`compressed-tensors` with `input_activations: null`) is the same
weights with bf16 activations: 14% better decode at 80B, 8.5% worse prefill.
Long generations favour it; long prompts favour W8A8.

### Best — if the workload allows it

**bf16 with `--load-format sharded_state`.** 8,755 prefill and 65.2 decode at
30B — 20% and 50% ahead of W8A8 respectively — because nothing is quantized and
nothing is dequantized.

Two conditions. It needs the memory (28.5 GiB/rank at 30B against W8A8's 14.7),
and it needs the one-time `save_sharded_state` conversion, without which load is
**12,366 s**. With it, 114.65 s. Do not attempt the conversion on AWQ weights —
it produces a silently wrong snapshot (§5).

**GGUF K-quants on llama.cpp**, for long context and for anything with expert
layers on CPU. llama.cpp wins decode at every depth measured (46.4 vs 39.4 t/s at
150k) and is the only option past ~160k. Use **K-quants, not I-quants**: I-quants
lose 36% decode once experts are CPU-resident.

### Why — the four mechanisms behind all of it

**1. Active parameters dominate everything else.** Llama-3.3-70B dense W8A8:
**843 t/s** prefill. Qwen3-Next-80B MoE W8A8: **7,253 t/s**. A *larger* model,
8.6× faster, because the MoE activates 3B parameters per token and the dense
model activates 70B. Both are correct W8A8 at `head_dim = 128`, and the dense one
actually loads ASM kernels the 80B cannot. If you are choosing between a dense
model and a sparse MoE of similar size, the MoE wins by roughly the ratio of
active parameters, and no kernel work closes that gap.

**2. The loader decides load time, not the format.** bf16's 12,366 s and AWQ-at-235B's
9.3 h are the *same bug class* — a per-tensor Python `weight_loader` — and
`sharded_state`'s flat copy fixes both. Format and load time are independent
axes; treat them that way when reading a checkpoint.

**3. CDNA2 has int8 MFMA and nothing below it.** This single fact explains the
whole "never" list: fp8, fp4 and signed int4 have no matrix instruction on this
chip. FP8 does not merely fail to help — it emulated `e4m3 → fp16` in software at
**7,106 of 11,997 instructions** against 64 MFMA, which is why it measured 1,047
t/s against AWQ's 3,002.

**4. Fast paths are a multiplier, not a substitute.** The eligibility profile is
narrow — `head_dim` 128 or 192, bf16, GQA ratio exactly 8 or 16 (`docs/35`) — and
Qwen3-Next-80B satisfies **none** of it (`head_dim = 256`) while being the
fastest thing in the matrix. Choose on architecture and active parameters first;
treat fast-path eligibility as a bonus worth ~1.23× under concurrency.

---

## 5. What does not work — and the specific reason for each

The negatives are the more useful half of this document, because every one of
them cost real time to establish.

### AITER families that are installed but unreachable

| family | objects | why it never runs |
|---|---:|---|
| `fmha_v3_fwd` | 48 | ✅ **runs** — the only one |
| `pa` | 8 | **No call site.** `pa_fwd_asm` appears 6× in vLLM 0.23.1, none of them a call. The kernels are fine: 48/48 configs numerically exact, 0 failures on re-run. |
| `mla` | 11 | **Flag inert.** `VLLM_ROCM_USE_AITER_MLA` 1 vs 0 → 8,630.8/93.34 vs 8,660.9/93.56. vLLM logs `Using FLASH_ATTN MLA` either way. |
| `fmoe` | 8 | **The kernel declines the device**, and all 8 gfx90a objects are `noquant{Fp16,Bf16}` — they cannot consume quantized weights regardless. |
| `topksoftmax` | 22 | Untested, and not worth testing: MoE routing is <1% of MoE work. |
| `allreduce_*` | 4 | Blocked by the upstream `NameError` (§1), and there is no XGMI for it to accelerate anyway. |

All of them share one upstream root cause — `is_aiter_found_and_supported()` →
`on_mi3xx()` = `gfx942|gfx950` — which is why `enable_vllm_aiter_gfx90a.py`
widens **only** the two attention checks. Widening the master gate would admit
gfx90a to `AiterFp8BlockScaledMMKernel`, an FP8 GEMM on a chip with no FP8 ALU.

### Formats and settings to avoid outright

| don't | why | evidence |
|---|---|---|
| **FP8, any model** | No FP8 ALU. Every weight upcasts to bf16; you pay dequant and get nothing. | 1,047 vs AWQ's 3,002 — **2.9× slower** |
| **NVFP4 / MXFP4** | No FP4 ALU at all. | `docs/27` |
| **W4A8, W4A4** | No ROCm kernel takes int8 activations, *and* none accepts signed int4. | kernel-by-kernel refusal, `docs/27` |
| **`compressed-tensors` 4-bit, `type:"int"`** | The ROCm mixed-precision kernel accepts `uint4b8`/`uint4`, not signed `int4`. AWQ and GPTQ emit the unsigned form and work. | `Quant type int4 not supported` at load |
| **Any MoE that falls back to WNA16** — incl. **AWQ at 235B** | `moe_wna16_weight_loader` runs per expert, per weight, single-threaded Python. Not a `gptq` property: AWQ lands here once Marlin declines the experts, and Marlin is CUDA-only here. | **1,339 s/shard × 25 ≈ 9.3 h**; both arms timed out at 3,600 s |
| **`save_sharded_state` on AWQ** | Produces a silently wrong snapshot. bf16 round-trips correctly. | reproduced twice |
| **Speculative decoding** | Decode sits **6.4× off its bandwidth bound**, so verifying N+1 tokens costs ~N+1×. Loses at 100% acceptance. | 6.85 → 6.01 t/s |
| **`--cpu-offload-gb` (UVA) at 357B** | Zero-copy reads across PCIe, no overlap. One 28k request ran **35+ min** at 505% CPU. | `docs/28` |
| **Prefetch offload without `--enforce-eager`** | The offloader splices a private stream into CUDA-graph capture; RCCL collectives inside that capture die. | `NCCL error: unhandled cuda error` at load |
| **`HSA_XNACK=1` unified memory** | **Damages the host.** VM fault on a resident managed page, leaving an rwsem with no live owner; amdgpu SVM eviction workers stacked up and load average hit **70 and climbing**. | `docs/29` |

> `HSA_XNACK` is the only entry here that is dangerous rather than merely slow.
> The harness recorded the failure and advanced to the next arm, re-triggering
> it. Any script that touches it should be gated behind an explicit opt-in;
> ours is (`ALLOW_UVM_HANG=1`).

---

## 6. Reading a checkpoint before you download it

Four fields, none of them in the repo name:

1. **`kv_lora_rank` in `config.json`** — check this **first**. If present the
   model is MLA and `num_key_value_heads` is vestigial. Reading GLM-5.2's
   `64 == 64` as MHA produced a **44× KV-size error** (123 GiB vs 2.74 GiB at
   32k) in an earlier draft of `docs/35`.
2. **`head_dim`** — 128 or 192 for ASM eligibility; 256 (Qwen3-Next) reaches no
   ROCm fast path at all.
3. **`num_attention_heads / num_key_value_heads`** — exactly 8 or 16 for `pa`
   coverage, which is moot today (no call site) but is the filter to apply if
   that changes.
4. **The quantization block** — `type` (`int` vs `uint4b8`), and
   `input_activations` (`null` → W8A16, `int` → W8A8).

---

## 7. Open, and worth someone's time

- **Numeric verification of paged attention above 200k.** The 150k result is
  solid; 250k is instantiated but unproven.
- **`PYTORCH_HIP_ALLOC_CONF` pinning** (`pinned_reserve_segment_size_mb`,
  `pinned_use_hip_host_register`, `pinned_num_register_threads`) against the
  prefetch offloader — untested, and the no-pinning arm already showed 652.5 vs
  695.9, so pinning is worth ~7% and might be worth more when tuned.
- **The three upstream reports** in §1, none of which are filed.
- **`fmoe` on unquantized bf16 experts.** It declines the device today, but it is
  the highest MACs/instruction kernel on this box (1,129 vs a compiled
  `fused_moe` median of 195, `docs/33`). If the device check is a gate rather
  than a capability statement, this is the largest remaining win.
