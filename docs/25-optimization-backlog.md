# Optimization backlog for gfx90a

Leads found while building the quantization matrix, not yet pursued. Each entry
states the evidence, the proposed change, and — importantly — how it could be
**wrong**, because three plausible-looking leads in this project turned out to
be correct upstream behaviour and patching them would have made things worse.

Ordered by expected value, not by effort.

---

## 1. vLLM's per-expert MoE loader: ~3 hours becomes minutes

**Evidence.** Loading Qwen3-30B-A3B bf16 at TP=2 takes **697 s per shard**, ~3
hours for 61 GB, while the same file reads at **3.0 GB/s** (measured with `dd`).
`py-spy` on the pinned worker:

```
_load_w13 (vllm/model_executor/layers/fused_moe/layer.py:762)
weight_loader (layer.py:1138)
load_weights (models/qwen3_moe.py:627)
```

The hot line is:

```python
loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
...
expert_data.copy_(loaded_weight)
```

`narrow()` produces a **non-contiguous** view, and the copy runs once per expert
per layer — 128 × 48 = 6,144 host→device copies of strided memory. The narrowing
only happens when `tp_size > 1`.

**Proposed.** Make the source contiguous before the copy (`.contiguous()` on the
narrowed view), or batch expert copies into one transfer per layer. Either turns
6,144 strided PCIe transfers into far fewer contiguous ones.

**How this could be wrong.** `.contiguous()` allocates — on a 61 GB model that
could OOM host RAM if done carelessly, and the current code may be strided
deliberately to keep peak memory low. Measure peak RSS alongside load time.

**Confidence: high** that the cost is real and here. **Medium** that the fix is
this simple.

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

## 3. `global_atomic_pk_add_bf16` emulation — the largest blocked kernel class

**Evidence.** Of 1,422 gfx942 AITER kernels, **539 are blocked solely by
`global_atomic_pk_add_bf16`**, which CDNA2 lacks. That is by far the biggest
single category in `docs/19` — larger than FP8 and gfx942-INT8 combined.

**Proposed.** The operation is an atomic packed bf16 add. It can be emulated
with a `global_atomic_cmpswap` loop on the 32-bit container: load, unpack, add
two bf16 lanes, pack, CAS, retry on failure. If that sequence assembles for
gfx90a and matches encoding length constraints, `repatch_gfx942_to_gfx90a.py`
could grow a substitution rule that expands one instruction into a loop.

**How this could be wrong.** This is the least certain entry here.
- The repatcher currently requires **same-length** encodings, and a CAS loop is
  many instructions replacing one. That means relocating code, not substituting
  it — a fundamentally harder transform than everything done so far.
- Atomics are used for cross-workgroup reduction; a CAS loop changes the
  performance profile enormously under contention and could be slower than the
  CK fallback it replaces.
- Correctness under concurrency is exactly where silent wrongness hides.

**Confidence: low** on feasibility, **high** on payoff if feasible. Prototype in
isolation with a contention stress test before touching the repatcher.

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

## 5. llama.cpp MMQ / `v_dot4_i32_i8`

**Evidence.** Untested. llama.cpp's MMQ path uses AMD dot-product intrinsics for
quantized matmul; `GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS` select
between MMQ and dequant+BLAS. Given INT8 arithmetic is free on this chip
(181 TOPS = bf16 peak), the MMQ path should be favoured — but nothing here has
confirmed which path the current build takes for Q4_K_M/Q8_0.

**Proposed.** Disassemble the shipped `libggml-hip.so` for `v_dot4_i32_i8` /
MFMA-int8 the same way `docs/22` did for attention, then A/B the flag.

**How this could be wrong.** llama.cpp already routes CDNA sensibly in the
attention case — `docs/22` found the matrix cores were *already* in use and the
"obvious" flag made things 18-26% **slower**. Assume the same until disassembly
says otherwise.

**Confidence: medium.** Cheap to check, and the check is disassembly, not a
benchmark.

---

## 6. Runtime activation quantization for popular formats

**Evidence.** Pending the 8-bit format comparison now queued. If GPTQ-Int8 and
AWQ-8bit lose to W8A8 as predicted, the cause is w8a16 — int8 weights, bf16
activations — dequantizing to a bf16 GEMM instead of reaching
`v_mfma_i32_16x16x16i8`.

**Proposed.** vLLM supports dynamic per-token activation quantization. If a
w8a16 checkpoint can be served with activations quantized at runtime, popular
formats reach the INT8 path without republishing weights.

**How this could be wrong.** Dynamic activation quantization costs a quantize
pass per token and changes numerics. It may not pay for itself, and accuracy
needs checking, not just speed. **Blocked on the comparison result** — do not
start until the mechanism is confirmed.

**Confidence: unknown**, deliberately. This entry exists so the follow-up is not
forgotten, not because it is likely.

---

## 7. AITER MLA ASM

**Evidence.** 11 portable MLA kernels sit behind a gate at
`asm_mla.cu:863` (see `docs/19`). vLLM gates MLA to gfx950, so reaching them
needs the same treatment as `enable_vllm_aiter_gfx90a.py` gave attention.

**How this could be wrong.** No MLA model is in the current matrix — DeepSeek-V3
class. Zero value until one is being served, and the 235B/GLM arms are GQA.

**Confidence: high** it works, **low** priority absent an MLA workload.

---

## Closed — do not retry

| Lead | Verdict |
|---|---|
| rocWMMA FlashAttention | **Slower, 18-26%.** The matrix cores were never idle; the flag diverts to an older kernel using the same MFMA with worse blocking. `docs/22`. |
| Unblock Marlin MoE on ROCm | `check_moe_marlin_supports_layer` returns False on ROCm **correctly** — Marlin is CUDA PTX. |
| Enable `should_moe_wna16_use_cuda` | Gate is **correct**: `torch.ops._moe_C` has no wna16 op in the ROCm build. Forcing it crashes. |
| Widen the AITER master gate | Admits gfx90a to FP8 GEMM on a chip with no FP8 ALU. Attention-only is deliberate. |
| FP8 on CDNA2 | 2.9x slower than AWQ-Int4. No ALU; every weight upcasts to bf16. |

The first three all *looked* like the INT8 bug — an `is_cuda()`/`is_rocm()` check
standing between AMD hardware and a working kernel. Two were correct and one was
a genuine bug. The difference was always checkable in a couple of minutes:
**does the kernel actually exist for this target?**
