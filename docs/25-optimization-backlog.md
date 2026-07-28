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

**Proposed.** Three options, cheapest first:
1. Evaluate the capture-time gate against a representative runtime length.
2. Raise the gfx9 threshold and verify correctness at long context.
3. Make the captured graph's attention dispatch metadata-driven rather than baked.

**How this could be wrong.** The 128k ceiling probably exists for a reason —
likely partition-count or temp-buffer sizing in the custom paged-attention
kernel. Option 2 without a correctness check could produce silently wrong
attention above 128k. Any of these needs numeric verification at 200k+, not just
a throughput measurement.

**Confidence: high** on the mechanism, **unverified** end to end. The decisive
experiment is in `docs/23`.

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
