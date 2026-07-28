# Optimization backlog for gfx90a

Leads found while building the quantization matrix, not yet pursued. Each entry
states the evidence, the proposed change, and — importantly — how it could be
**wrong**, because three plausible-looking leads in this project turned out to
be correct upstream behaviour and patching them would have made things worse.

Ordered by expected value, not by effort.

---

## 1. vLLM's per-expert MoE loader — attempted, REFUTED

**Evidence for the cost is solid.** Loading Qwen3-30B-A3B bf16 at TP=2 takes
697 s/shard and Qwen3-235B GPTQ-Int4 takes 810 s/shard, while the same files
read at 3.0 GB/s under `dd`. `py-spy` put every sample in `_load_w13`.

### The decisive comparison: same model, same TP, same box, 207×

Round 2 produced the control this investigation needed for weeks. Load rates
across every arm, normalised per byte:

| arm | TP | GiB/rank | load (s) | **GiB/s** |
|---|---:|---:|---:|---:|
| t80-w8a8 (MoE, compressed-tensors) | 2 | 38.31 | 142.8 | **0.2683** |
| **t35-w8a8-tp2** (MoE, compressed-tensors) | **2** | 14.66 | **59.7** | **0.2455** |
| t35-w8a8 (MoE, compressed-tensors) | 1 | 29.17 | 122.8 | 0.2376 |
| t70-w8a8 (**dense**, compressed-tensors) | 2 | 33.88 | 420.5 | 0.0806 |
| **t35-bf16** (MoE, no quant config) | **2** | 28.51 | **12,365.8** | **0.0023** |

**The same 30B model at the same TP=2 loads in 59.7 s as W8A8 and 12,365.8 s as
bf16 — 207×.** Per byte it is 107× slower.

This eliminates every remaining structural explanation in one shot:

- **Not TP=2.** The TP=2 W8A8 arm is the *fastest* loader in the entire table.
- **Not the model.** Same architecture, same expert count, same files on disk.
- **Not model size.** bf16 moves *fewer* GiB per rank than the 80B arm that
  loads in 143 s.
- **Not storage, filesystem, or prefetch.** Already established, and now
  doubly so — the fast arms read from the same btrfs volume.

**It is the bf16 code path specifically.** `compressed-tensors` checkpoints
route through a different weight loader and never enter the path where
`py-spy --native` finds `hsakmt_ioctl`. Whatever the ioctl storm is,
compressed-tensors sidesteps it completely.

One secondary observation worth keeping: the **dense** compressed-tensors arm
(t70) runs at 0.0806 GiB/s, ~3× slower per byte than the MoE compressed-tensors
arms. That is a much smaller effect and a separate question — it does not go
through `_load_w13` at all, having no experts — but it says the fast path is not
uniformly fast either.

**Practical consequence, and it is the actionable one:** bf16 is the fastest
*inference* configuration measured at TP=2, and it costs **3.4 hours to load
against 60 seconds**. For anything that restarts more than about twice a month,
that trade is not close. `docs/26` treats this as a selection rule.

### Storage is definitively not the bottleneck

Worth stating flatly, because this repo got it wrong once — an early diagnosis
blamed btrfs zstd compression for slow loads, and it was wrong. `dd` refuted it
at 3.0 GB/s, but `dd` on one file is weak evidence. Here is the strong version,
measured on the real model files during an actual load:

```
Prefetching checkpoint files: 80% (8/8)
Prefetching checkpoint files into page cache finished in 13.19s
...
Loading safetensors checkpoint shards: 31% | 5/16 [1:06:42<2:29:31, 815.57s/it]
```

**61.08 GB (measured, `du -sb`) read in 13.19 s = 4.63 GB/s** — off
`/mnt/llm-storage`, which is btrfs with `compress=zstd:5`. The loader then
spends **815 s per shard**, ~3.6 hours total.

Storage delivers the entire model in thirteen seconds; the loader spends three
and a half hours. That is a **~1000× gap**, and it closes the question:
compression, btrfs and I/O are all irrelevant to this cost. It is CPU in Python.

Two practical notes that follow:

- **Faster storage would not help.** Any effort spent on NVMe layout, disabling
  compression, or mount tuning is aimed at 0.1% of the problem.
- **But `--safetensors-load-strategy=prefetch` is still mandatory on btrfs**,
  for a different reason — see below.

### `--safetensors-load-strategy=prefetch` is required on btrfs, and is not automatic

Correcting an earlier claim in this document that prefetch is "on by default and
needs no flag". **It is not.** The auto-detection covers network filesystems
only:

```python
is_net_fs = fs_type in ("nfs", "nfs4", "lustre")
...
if safetensors_load_strategy is None:
    if is_net_fs and fits_in_ram:
        should_prefetch = True
    elif not is_net_fs and fits_in_ram:
        logger.info_once(
            "Auto-prefetch is disabled because the filesystem (%s) is not a "
            "recognized network FS (NFS/Lustre). If you want to force "
            "prefetching, start vLLM with --safetensors-load-strategy=prefetch.",
            fs_name,
        )
```

btrfs, ext4 and xfs all fall into the second branch and get **no prefetch**
unless asked. vLLM even names the filesystem in the log:

```
Filesystem type for checkpoints: BTRFS. Checkpoint size: 56.87 GiB.
Available RAM: 455.36 GiB.
```

**This stack already passes the flag** (`bf16_final.sh`, `followup_chain.sh`,
`rerun_glm_cold.sh`, `rerun_t80_cold.sh` and others), which is why the 13.19 s
prefetch above happened at all. Anyone reproducing from this repo on a local
filesystem should keep it. Without it, upstream
[#40988](https://github.com/vllm-project/vllm/issues/40988) reports the failure
mode on ext4: mmap-driven random reads during weight materialization — ~13,000
scattered reads per rank — leaving some workers hung past 60 minutes while
others finish in 6–9. That is a **hang from random-read tail latency**, a
different symptom from the steady CPU-bound crawl documented here.

### The shards get *slower*, which rules out I/O completely

The strongest evidence yet that this is not storage, and a new lead in its own
right. Successive shards of the same model, same run:

| shard | s/it |
|---:|---:|
| 1/16 | 697.08 |
| 2/16 | 773.67 |
| 3/16 | 794.97 |
| 4/16 | 808.96 |
| 5/16 | 815.57 |
| 6/16 | 819.98 |

**Monotonically increasing, +18% across six shards.** If disk were the
bottleneck this would trend the other way as page cache warms. Instead the cost
grows with the number of tensors already resident — which is the signature of
work that scales with what has been loaded, not with what is being read.

That is a concrete, previously unnoticed lead: something in the load path is
super-linear in loaded-tensor count. If it is O(n²), it explains why the problem
is catastrophic rather than merely slow at 235B (810 s/shard × 48 shards ≈ 7
hours) while being tolerable at 30B.

**Late-shard profile: same function, not a different one.** `py-spy` on the
running worker at shard 6 lands on the identical frame the early samples did:

```
_load_w13 (fused_moe/layer.py:771)          <- expert_data.copy_(loaded_weight)
_load_model_weight_or_group_weight_scale (layer.py:623)
weight_loader (layer.py:1159)
load_weights (models/qwen3_moe.py:627)
```

Line 771 is the host-to-device copy itself. Note this build **already carries
the `contiguous()` fix** — the refuted patch is in the image — and is still at
815 s/shard, which is consistent with that refutation rather than a new result.

### The number that makes this the top loader lead

**Effective host-to-device throughput is single-digit MB/s.**

56.87 GiB checkpoint, TP=2, so ~28.4 GiB materialised per rank. At 815 s/shard ×
16 shards ≈ 13,040 s, that is **~2.2 MB/s per rank**. Against a PCIe 4.0 x16 link
that should sustain ~25 GB/s, this is **four orders of magnitude off**.

No amount of I/O tuning explains a 10,000× shortfall on a bus that is not being
used. Whatever `copy_` is doing here, it is not DMA — it is the per-element
gather the code comment warns about, or a page-fault-per-access walk of the
mmap'd source, or both. The `contiguous()` fix targeted exactly this and did
nothing, which means the strided-source hypothesis in its docstring is wrong or
incomplete.

### The probe ran, and it refutes the copy hypothesis — but is itself confounded

`benchmarks/matrix/bin/probe_loader{,2,3}.py`. Three rounds, and the honest
summary is that **line 771 is not slow, and I do not yet know what is.**

**Round 1 — the copy is fast.** With a preallocated destination and a
materialised source:

| operation | rate |
|---|---:|
| `dst.copy_(strided slice)` — *line 771* | **21,713 MB/s** |
| `dst.copy_(contiguous src)` | 22,067 MB/s |
| pinned H2D (ceiling) | 23,425 MB/s |

Line 771 runs at 93% of the pinned-memory ceiling. It is **not** a per-element
gather. Two incidental findings: `narrow(dim=0, ...)` on a row-major 2-D tensor
is **already contiguous**, so the refuted `contiguous()` patch was a no-op for a
second independent reason; and the strided and contiguous paths are within 2% of
each other, so the docstring's stated rationale is wrong.

**Round 2 — the slowdown reproduces.** Iterating real expert tensors:
**1.6 MB/s effective**, against ~2.2 MB/s observed in the live load. Same
phenomenon. 100% of the time in `torch.empty(device) + copy_`, ~1001 ms/tensor.

**Round 3 — and it falls apart.** Splitting allocation from copy:

| operation | ms/tensor |
|---|---:|
| `torch.empty(device)` only | 0.01 |
| `copy_` into preallocated buffers | **1001.29** |
| `empty + copy_` **interleaved** | **0.08** |
| pooled buffer reuse | 0.07 |

**The interleaved case — strictly more work — is 12,000× faster than the
isolated copy.** That is not a real cost structure. Two tells:

1. The slow figure is **1001.29 ms and 1000.98 ms** across independent runs:
   essentially exactly 1.000 s. Costs that scale with data do not land on a
   round number twice.
2. Whichever operation runs **first** is slow; everything after is fast.

**The confound is stated rather than explained away: a bf16 arm was loading on
both cards throughout.** A ~1 s stall on first GPU touch under contention is a
far better fit than any property of `copy_`. So round 3 measures the test
conditions, not the loader.

**What survives:** line 771 sustains ~21.7 GB/s when unobstructed, so the
"strided copy degrades to a gather" story in the code comment is **wrong**, and
the ~2.2 MB/s effective rate of the real load remains unexplained.

### ROOT CAUSE: it is blocked in a kernel-driver ioctl

`py-spy --native` on the live TP=2 load. **Eight consecutive samples, all
identical:**

```
ioctl (libc.so.6)
hsakmt_ioctl (libhsa-runtime64.so.1)
_load_w13 (vllm/model_executor/layers/fused_moe/layer.py:771)
```

The worker is blocked in an **HSA thunk ioctl into the amdgpu/KFD driver** — not
in memcpy, not in DMA, not in page faults. Every prior profile in this document
was taken *without* `--native`, which is why they all "proved" line 771 was the
cost: a pure-Python profile cannot see inside a HIP call and attributes the
entire wait to the last Python frame.

**This is the single most important lesson from the whole loader investigation.**
Four hypotheses were eliminated — storage, strided copies, allocation, and
`copy_` itself — and each was eliminated by measurement while the *actual* cause
sat one stack frame below where the profiler could see.

**Mechanism this implies.** Each expert tensor is a **fresh mmap'd host region**,
so every `copy_` must register/map that region with the driver before it can DMA
from it. Registration is an ioctl with a large fixed cost, and the ~3.2 MB
payload is irrelevant beside it. That single mechanism accounts for every
observation that previously looked contradictory:

| Observation | Explained by |
|---|---|
| Line 771 hits 21.7 GB/s in probe 1 | source was materialised, already registered |
| First touch slow, reuse fast (probe 3) | registration happens once |
| ~1000 ms constants regardless of size | fixed ioctl cost, not bandwidth |
| Shard times rising 697 → 822 s | registration table growing |
| Pinned memory fastest in probe 1 | pinned is registered at allocation |
| `contiguous()` patch did nothing | never addressed registration |

**Candidate fix, and it is cheap:** copy through **one reused pinned staging
buffer** instead of DMA'ing from each fresh mmap'd slice — registration once
rather than 18,432 times. Probe 3's pooled-reuse case already hit 0.07 ms/tensor
by accident.

`benchmarks/matrix/probe_loader4.py` tests exactly this (direct vs staged vs
registration-warm), and `round2.sh` E10 runs it on an idle box. **Not yet
measured** — the mechanism is strongly evidenced but the fix is not.

**Confidence: high** that the block is in driver ioctls (8/8 native samples),
**medium** that it is specifically memory registration, **unmeasured** on the
fix.

**The fix did not work.** `configs/fast_moe_expert_load.py` makes each narrowed
slice contiguous before the host-to-device copy. Measured on 235B GPTQ-Int4 at
TP=2, patch verified present in the container:

| | s/shard |
|---|---:|
| without patch | 810.49 |
| **with patch** | **810.81** |

Identical. Source-side contiguity is not the cost.

**Why it did nothing: I patched the wrong function.** `py-spy` on the GPTQ-Int8
arm during its 20-minute load puts every sample here:

```
moe_wna16_weight_loader (quantization/moe_wna16.py:445)
load_weights            (models/qwen3_moe.py:627)
```

Not `_load_w13` in `fused_moe/layer.py`, which is what the patch modifies.
**GPTQ never reaches that function.** It routes through the WNA16 quantization
loader instead, which runs per expert per weight and does a
`loaded_weight.to(device)` followed by a format conversion (`convert_awq_tensor`
or the gptq equivalent) on each one. Same structural problem — thousands of
small host-to-device transfers plus per-tensor Python work — in a different
place.

The original `py-spy` evidence was from the **30B bf16** run, which genuinely
does hit `_load_w13`. Generalising from that one profile to "the MoE loader" was
the error: the loader depends on the quantization method, which is also why
`compressed-tensors` loads fast while bf16 and GPTQ crawl.

So there are (at least) two slow loaders, not one, and a fix for either does
nothing for the other. `_load_w13`'s destination-side scatter remains an open
suspect for the bf16 path specifically.

### The cost is quant-method specific, not TP specific

| Model | TP | `quant_method` | Load rate |
|---|---:|---|---:|
| 30B bf16 | 2 | none (bf16) | **697 s/shard** |
| 235B GPTQ-Int4 | 2 | `gptq` | **810 s/shard** |
| 80B AWQ | 1 | `compressed-tensors` | 1.10 s/it |
| GLM-4.6 AWQ | **2** | `compressed-tensors` | **9.11 s/it** |

**TP>1 is not sufficient to cause it.** GLM-4.6 runs TP=2 and loads two orders of
magnitude faster. The slow arms are bf16 and `gptq`; `compressed-tensors` is fast
at either TP, so it evidently does not reach the `_load_w13` path `py-spy`
caught.

I briefly read GLM's fast load as confirming the patch. It does not — that
container was created from the image tag *before* the patched layers were
committed (`e27dbf262daa` vs `85de3e47a14e`), and `grep -c
loaded_weight.is_contiguous()` inside it returns 0. It was fast without the fix.

**Practical consequence, independent of any fix:** a 235B at TP=2 on this
hardware costs ~7 hours to load. That is not a benchmark artifact, it is the
deployment reality, and it makes vLLM impractical for that tier here.

**The deployment-side workaround is now the recommendation** and it does not
require patching anything: choose `compressed-tensors` packaging. `docs/26`
turns this into a checkpoint-selection rule. That does not close the lead — the
bf16 and `gptq` loaders are still slow — but it removes the urgency, since no
recommended configuration goes through either of them.

---

## 1b. A Triton W4A8-int kernel for ROCm — now the highest-value open lead

**Evidence.** `docs/27`. int8 is the only sub-16-bit matrix dtype gfx90a has
(verified by assembler: every `i4`, `fp8` and `smfmac` form is rejected), so
W8A8 is currently the *only* format that reaches quantized arithmetic. W4A8-int8
would halve weight bytes again while keeping `v_mfma_i32_16x16x16i8` — the right
direction on a chip where INT8 is bandwidth-bound, not arithmetic-bound.

Both halves of the software already exist: vLLM ships
`compressed_tensors_w4a8_int.py`, and llm-compressor produces the checkpoints.

**What blocks it.** Every kernel in the mixed-precision registry that accepts
`act_type == torch.int8` is unreachable here — `marlin.py` is CUDA PTX,
`dynamic_4bit.py` is ARM/KleidiAI. Every ROCm-reachable kernel
(`triton_w4a16`, `rdna_hybrid_w4a16`, `exllama`) requires fp16/bf16 activations.

**Proposed.** Write a Triton w4a8-int kernel — unpack int4 → int8 in-register,
dynamic per-token activation quantization, `tl.dot` with int8 operands into an
int32 accumulator — and register it in the mixed-precision registry. Triton on
ROCm does emit `v_mfma_i32_16x16x16i8` for int8 `tl.dot`, so the hardware path is
reachable from Triton.

**How this could be wrong.** This is a **new kernel, not a gate patch**, and the
distinction matters: the Int8 MoE fix worked because a working kernel sat behind
a bad `is_cuda()` test. Here no kernel is being wrongly excluded. Also unmeasured:
int4 unpacking and per-token activation quantization both cost something, and on
a bandwidth-bound chip the win could be smaller than the arithmetic suggests.

**Confidence: high** that the instruction path exists, **unknown** on payoff.
`benchmarks/matrix/round2.sh` E4 probes what vLLM currently does with a
W4A8-int8 checkpoint on gfx90a, which is the cheap first step.

---

## 1c. Speculative decoding — MEASURED, and it hurt both ways

Prompted by a look at ROCmFPX (RDNA-only, so its formats do not apply here), the
one transferable idea was multi-token prediction: scheduling rather than kernels,
therefore ISA-independent, and aimed at this box's weakest axis.

**It made decode worse in every configuration tested.**

| arm | prefill @15k | decode @101k | vs baseline |
|---|---:|---:|---|
| 80B W8A16, no speculation | 6,679 | **51.34** | — |
| 80B W8A16 + native MTP, 1 token | 5,510 | 43.09 | **−16%** |
| 80B W8A16 + native MTP, 2 tokens | 6,296 | 37.30 | **−27%** |
| 30B W8A8 TP=2, no speculation | 7,278 | **43.40** | — |
| 30B W8A8 TP=2 + n-gram | 7,224 | 32.76 | **−24%** |

MTP also cost prefill — 6,679 → 5,510 at one speculative token.

**Read the n-gram row narrowly.** The benchmark prompt is random filler plus a
counting instruction, which is close to a worst case for prompt-lookup: there is
nothing in the prompt to speculate from. That was stated in the script before the
run, and the result is consistent with it. It says "not helped by this workload",
not "does not work".

**The MTP rows are the surprising ones and do not have that excuse.** Counting
from 1 to 60 is about as predictable as generation gets, so acceptance should be
high, and more speculative tokens should help rather than hurt. Getting *worse*
at 2 tokens than at 1 points at per-step overhead dominating whatever acceptance
buys — plausibly the same untuned-MoE-kernel problem that makes W8A8 lose decode
twice over (`docs/24`), since each speculative step runs the MoE again.

**Not closed, but demoted.** Worth one re-test after the MI210 `fused_moe` config
is in place, because if speculation is being taxed by an untuned expert GEMM then
tuning changes the sign. Until then, do not enable it here.

**Confidence: high** that it hurts as currently configured, **low** that this
generalises past this hardware and this prompt shape.

---

## 2. The 128k graph-capture gate — unlock fast decode above 128k

**Evidence.** `docs/23`. `gpu/model_states/default.py:152-156`:

```python
if for_capture:
    max_seq_len = self.max_model_len
else:
    max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
```

Capture evaluates the gfx9 gate (`max_seq_len <= 128*1024`) against the
*configured* max, so a 256k server bakes the Triton fallback into the graph and
replays it for **every** request. Measured cost: 10x decode.

**ROOT CAUSE FOUND — and it is neither of the things this entry feared.** It is
not temp-buffer sizing and it is not a correctness limit. From
`csrc/rocm/attention.cu`:

```c
const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
// reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
// (partition size) = 128K context length
switch (npar_loops) {
  case 1: LAUNCH_CUSTOM_REDUCTION(1); break;
  ...
  case 8: LAUNCH_CUSTOM_REDUCTION(8); break;
  default: TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
}
```

8 × 64 × 256 = 131,072 = exactly 128K. **The ceiling is a dispatch table with
eight entries** — a set of missing template instantiations.

Three consequences, all of which lower the risk this entry assigned:

1. **The failure mode is loud, not silent.** Raising the Python gate without
   extending the switch aborts on `TORCH_CHECK` with the offending value. The
   silent-garbage scenario feared above cannot occur through this path.
2. **`NPAR_LOOPS=16` already compiles and runs.** The RDNA launcher instantiates
   the same reduce-kernel template at 1..16 today. (That does *not* mean RDNA
   reaches 256k — its warps are 32 wide, so 16 × 32 × 256 is also 131,072. Both
   architectures cap at 128K by different arithmetic. What it establishes is
   that the template body is valid at 16.)
3. **The resource cost is small.** `NPAR_LOOPS` sizes one shared array and three
   register arrays. At 16 with `WARP_SIZE=64`: LDS 2,048 → 4,096 bytes against
   64 KB available; register arrays 24 → 48 VGPRs against 256. Occupancy may
   drop slightly — a performance question, not a correctness one.

**Patch: `configs/extend_rocm_pa_256k_gfx9.py`.** Adds cases 9–16 to the *gfx9*
launcher only and raises that branch's gate to `256 * 1024`. The RDNA branch is
deliberately left at 128k, where its kernel genuinely stops. Verified to apply
cleanly against upstream `attention.cu`; requires rebuilding the ROCm extension.

### Numerics: VERIFIED

`tests/test_rocm_pa_256k.py` compares the custom kernel against Triton — the
fallback it replaces — both driven through the real
`chunked_prefill_paged_decode` entry point, with an explicit assertion that the
gate *selected* the custom path before any comparison is made.

| seq_len | `npar_loops` | stock build | patched build |
|---:|---:|---|---|
| 131,072 | 8 | **PASS** `rel_err 6.62e-03` | **PASS** `rel_err 6.62e-03` |
| 139,264 | 9 | FAIL — gate declined | **PASS** `4.15e-03` |
| 196,608 | 12 | FAIL — gate declined | **PASS** `4.26e-03` |
| 262,144 | 16 | FAIL — gate declined | **PASS** `5.29e-03` |
| 266,240 | 17 | PASS — declined | **PASS — still declined** |

Three things this establishes, none of which the argument above could:

1. **The control is identical on both builds** (`6.62e-03`), so the patch does
   not perturb the path that already worked.
2. **The test discriminates.** Run against stock, exactly the three patched
   lengths fail. A test that passed on both builds would be measuring nothing.
3. **The ceiling still exists.** 266,240 is declined on the patched build too,
   so the guard was extended rather than removed — which was the dangerous
   outcome, since it would have passed every other case.

Errors of 4–7e-3 are bf16 rounding between two independent kernels, not drift;
a dropped partition group is an order-1 error and would be unmissable.

Two test-design notes worth keeping, both learned by getting it wrong first:

- **A hand-written fp32 reference was the wrong choice.** It disagreed at every
  length *including `npar_loops = 1`*, because the KV layout is 5-D K
  `[blocks, kv_heads, head_size/x, block_size, x]` and 4-D V — which is **not**
  what `RocmAttentionBackend.get_kv_cache_shape` declares. Comparing two kernels
  that consume the same tensors removes that whole class of error.
- **The gate assertion is load-bearing.** Without it the test has a silent-pass
  mode: if the gate declines, both arms run Triton, agree perfectly, and a
  broken patch looks flawless. That is the `ROCM_AITER_FA` failure again —
  backend admitted, never chosen, credited anyway.

**Confidence: high** on the mechanism, the patch, and the numerics.
**Still unmeasured: the actual decode win.** Build tagged
`rocm-vllm-aiter-gfx90a:pa256k`.

---

## 3. `global_atomic_pk_add_bf16` emulation — CLOSED, not substitutable

**539 of 1,422 blocked kernels** need this one instruction — the largest single
category, bigger than FP8 and gfx942-INT8 combined. Worth checking properly, and
the answer is definitive.

Real syntax taken from a shipped kernel rather than guessed
(`bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co`):

```
global_atomic_pk_add_bf16 v4, v76, s[16:17]   // DD488000 00104C04
```

Assembler verdicts:

| Instruction | gfx942 | gfx90a |
|---|---|---|
| `global_atomic_pk_add_bf16` | **8 bytes** | **rejected** |
| `global_atomic_cmpswap` | — | 8 bytes |
| `global_load_dword` | — | 8 bytes |
| `v_add_f32_e32` | — | 4 bytes |
| `v_cmp_eq_u32_e32` | — | 4 bytes |

So gfx90a *has* every primitive a CAS loop needs. The emulation is expressible:
load the 32-bit container, unpack two bf16 lanes, add, repack, `cmpswap`,
compare and retry.

**It still cannot be done as a substitution.** The minimal loop is roughly
load (8) + unpack (~8-16) + two adds (8) + repack (~8-16) + cmpswap (8) +
compare-and-branch (8) — **60-80 bytes replacing 8**. `repatch` requires
identical encoding length for a reason that is not fussiness: these are
prebuilt code objects with no linker involved, so growing an instruction shifts
every subsequent address and silently breaks every branch target and relocation
in the object. An 8-to-80-byte expansion is relocation, not substitution.

The two routes that remain are both worse than the fallback they would replace:

- **Trampoline out to a helper.** Requires free space, a call/return convention
  inside a hand-written ASM kernel that assumes full register control, and
  patching the branch. Very high risk of silent corruption in exactly the code
  paths where correctness is hardest to observe.
- **Recompile from source.** These are prebuilt ASM blobs. There is no source.

And even if it worked, the performance argument is against it: these atomics
are cross-workgroup reduction points, so a CAS retry loop under contention could
easily be slower than the CK path already running.

**Verdict: closed.** The 242/1,422 ceiling is not conservative — it is the real
boundary for instruction-level translation, and this category is the reason.

---

## 4. Per-tier MoE tuning

**Evidence.** vLLM ships tuned `fused_moe` configs for MI300X, MI308X, MI325X,
MI350X, MI355X, R9700 and A100 — and **none for MI210**. `get_moe_wna16_block_config`
short-circuits to tuned `BLOCK_SIZE_N/K` when present and otherwise uses fixed
heuristics.

**Proposed.** Tune per tier. Configs are keyed `E={E},N={N},device_name=...`, so
the tier-1 config (E=128, N=768) does **not** help the 80B, 235B or GLM-4.6,
which have different expert geometries.

**How this could be wrong.** The full sweep proved unbounded — 704 → 2,660 →
4,990 → 7,810 candidates, growing per batch size, ~1 s each on a serialized GPU.
Bound it by batch size and wall clock, or it eats a day per tier. Also: tuning
targets the WNA16 path, and W8A8 already beats it, so the payoff may be small on
exactly the format that matters.

**Confidence: high** that it helps something, **low** that it changes the
recommendation.

---

## 5. llama.cpp MMQ / `v_dot4_i32_i8` — CLOSED, already in use

**Answered by disassembly, not benchmark.** Extracting all 45 gfx90a offload
bundles from the shipped `libggml-hip.so` and disassembling 632,972 instructions:

| Instruction | Count |
|---|---:|
| **`v_dot4c_i32_i8`** | **11,008** |
| `v_mfma_f32_16x16x16f16` | 1,200 |
| `v_mfma_i32_16x16x16i8` | 544 |

llama.cpp is **already** reaching INT8 dot-product hardware, heavily. The MMQ
path is not dormant and `GGML_CUDA_FORCE_MMQ` has nothing to switch on — the
kernels that would be forced are the ones already running.

This is the same shape as the rocWMMA result in `docs/22`: the "obvious"
optimization was already in place, and the flag would have changed which
already-good kernel ran rather than turning hardware on. Suspecting that first,
and checking by disassembly rather than by A/B, cost minutes instead of a
benchmark cycle.

**One open sub-question, deliberately not pursued.** `v_dot4c_i32_i8`
outnumbers `v_mfma_i32_16x16x16i8` 20:1, so ggml prefers the dot-product form
over INT8 MFMA for quantized matmul. Whether MFMA would be faster for the large
GEMM shapes in prefill is unanswered — but it is a change to ggml's kernel
selection, not a flag, and llama.cpp already wins prefill at tier 1 and 2. Low
priority absent evidence that these specific shapes are underperforming.

---

## 6. Runtime activation quantization for popular formats — DEPRIORITIZED

**The mechanism is now confirmed.** w8a16 — int8 weights, bf16 activations —
dequantizes to a bf16 GEMM instead of reaching `v_mfma_i32_16x16x16i8`. The
discriminator is a single field, `input_activations` in the
`compressed-tensors` config group: populated means W8A8, `null` means
weight-only. `docs/26`.

**The premise behind the lead was wrong, though.** It assumed W8A8 checkpoints
are rare, so the only route to the INT8 path for a popular model was to
synthesize activation scales at runtime. In fact `RedHatAI` alone publishes ~90
`quantized.w8a8` checkpoints, all `compressed-tensors` `int-quantized` with
dynamic per-token activations — covering Qwen3, Qwen3-Next, Llama 3.x, Gemma 3,
Mistral, Phi-4, GLM-4.6 and the DeepSeek-R1 distills. **Downloading the right
file is strictly better than patching around the wrong one**, and for anything
genuinely unpublished, `llm-compressor` produces one offline.

**Not closed, just demoted.** The remaining case is a model with *no* W8A8 and
*no* bf16 weights to quantize from. Rare. And the objection still stands: the
scales in a w8a16 file were fitted for a bf16 GEMM, so runtime activation
quantization changes numerics and needs an accuracy check, not just a throughput
measurement.

**Confidence: high** on the mechanism, **low** that the work is worth doing.

---

## 7. AITER has no `hd256` fmha ASM — for any architecture

**Evidence.** The tier-2 80B arms select AITER and then load **only `torch.co`** —
no ASM code objects at all — while the tier-1 30B arms load
`fwd_hd128_bf16_causal_rtna_group.co`. Qwen3-Next has `head_dim = 256`.
Enumerating head dimensions in AITER's fmha ASM set:

```
=== head dims available in the gfx942 fmha ASM set ===
hd128 hd192
--- and which of those survived translation to gfx90a ---
hd128 hd192
```

**This is an upstream gap, not a translation gap.** There is no `hd256` kernel
to port. `repatch` cannot help; the 242/1,422 ceiling is not the limit here.

**Consequence, already reflected in `docs/24` and `docs/26`:** AITER ASM
attention is worth **+12.8% on `head_dim = 128` models and exactly 0% on
`head_dim = 256` models**. The flags still cost nothing, so leave them on, but do
not attribute 80B-class results to them.

**Proposed, if it ever matters.** Writing an hd256 fmha kernel from scratch is a
different order of work from translating one, and the payoff is bounded by the
+12.8% seen at hd128. Not worth starting unless an hd256 model becomes the
primary workload.

**How this could be wrong.** The enumeration covers the *fmha* ASM set. Other
AITER ASM families (GEMM, MoE) are indexed differently and were not checked for
head-dim coupling — but they are not head-dim parameterised, so this is unlikely
to hide anything.

**Confidence: high.** Verified by listing shipped code objects on both targets,
and corroborated by which `.co` files each arm actually loads at runtime.

---

## 8. AITER MLA ASM

**Evidence.** 11 portable MLA kernels sit behind a gate at
`asm_mla.cu:863` (see `docs/19`). vLLM gates MLA to gfx950, so reaching them
needs the same treatment as `enable_vllm_aiter_gfx90a.py` gave attention.

**A prior claim here was retracted, and it matters.** `docs/14` originally
reported MLA ASM *working* on gfx90a at "3M tok/s prefill, 0.090 ms/step decode,
3x faster than Triton". That doc is now SUPERSEDED for two reasons, both traps:

- The binary patch was **wrong**. It swapped `D3E1 → D3CD` — bf16 MFMA to
  **f16** MFMA. gfx90a *does* have BF16 MFMA: `v_mfma_f32_16x16x16bf16_1k`,
  opcode **D3E7**. D3CD is a perfectly valid instruction that computes the
  wrong thing, so the kernel ran at full speed and produced garbage. Those
  scripts are quarantined in `configs/attic/`.
- The throughput was **mis-attributed**. `mha.py` and `mla.py` gate their ASM
  paths to gfx942/gfx950, so those runs measured the CK or Triton fallback.
  The numbers were real; the attribution was not.

So MLA is genuinely *unenabled*, not "already done" — do not read `docs/14` as a
completed result. The 11 portable kernels are what `repatch` proved translatable
by re-assembling, which is a different and much stronger claim than an opcode
swap.

**How this could be wrong.** No MLA model is in the current matrix — DeepSeek-V3
class. Zero value until one is being served; the 235B and GLM arms are GQA.

**Confidence: high** that the kernels translate, **low** priority absent an MLA
workload, **zero** in any performance figure predating the port matrix.

---

## Closed — do not retry

| Lead | Verdict |
|---|---|
| rocWMMA FlashAttention | **Slower, 18-26%.** The matrix cores were never idle; the flag diverts to an older kernel using the same MFMA with worse blocking. `docs/22`. |
| Unblock Marlin MoE on ROCm | `check_moe_marlin_supports_layer` returns False on ROCm **correctly** — Marlin is CUDA PTX. |
| Enable `should_moe_wna16_use_cuda` | Gate is **correct**: `torch.ops._moe_C` has no wna16 op in the ROCm build. Forcing it crashes. |
| Widen the AITER master gate | Admits gfx90a to FP8 GEMM on a chip with no FP8 ALU. Attention-only is deliberate. |
| FP8 on CDNA2 | 2.9x slower than AWQ-Int4. No ALU; every weight upcasts to bf16. |
| Opcode-swap binary patching (`D3E1 → D3CD`) | **Silent garbage.** D3CD is f16 MFMA; gfx90a's bf16 is D3E7. Superseded by `repatch_gfx942_to_gfx90a.py`, which re-assembles and lets the assembler prove portability. Quarantined in `configs/attic/`. |
| Any ASM performance figure predating the port matrix | **Mis-attributed.** `mha.py`/`mla.py` gated ASM to gfx942/gfx950, so gfx90a runs measured the CK or Triton fallback while being reported as ASM. |

The first three all *looked* like the INT8 bug — an `is_cuda()`/`is_rocm()` check
standing between AMD hardware and a working kernel. Two were correct and one was
a genuine bug. The difference was always checkable in a couple of minutes:
**does the kernel actually exist for this target?**
