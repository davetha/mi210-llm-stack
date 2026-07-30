# Published algorithms that emit data, not kernels — zero gfx90a risk

**Date**: 2026-07-30 · **Hardware**: 2× MI210 (gfx90a / CDNA2), wave64, no FP8,
no int4 MFMA, no sparsity

Every method here produces **configuration or weights offline**. None requires a
new kernel, none touches wave64 shuffle semantics, and none depends on FP8, int4
MFMA or SMFMAC. That makes them the lowest-risk items available on this hardware —
the opposite risk profile from everything in `docs/19`–`docs/22`.

---

## 1. Three per-layer KV allocation rules, for infra that already exists

`changes/01` added `-ctk-cpu` / `-ctv-cpu` — **per-layer KV cache types**. That is
a rare capability: the KV-compression literature almost universally assumes a
*uniform* budget, because uniform is all mainline engines expose. Three papers
publish non-uniform allocation rules and emit nothing but numbers:

| method | arXiv · date · code | what it emits | reported |
|---|---|---|---|
| **DuoAttention** | **2410.10819** · 2024-10-14 · `mit-han-lab/duo-attention` | per-**head** classification: *retrieval* (full KV) vs *streaming* (sink + recent window only) | **1.63× prefill (GQA), 1.67× memory** |
| **PyramidKV** | **2406.02069** · 2024-06-04 (v4 2025-05-15) · code | per-**layer** budget, monotonically **decreasing with depth** | 12% of KV retains full accuracy |
| **MoA** | **2406.14909** · 2024-06-21 (v3 2025-11-24) · `thu-nics/MoA` | per-**head and per-layer** sliding-window lengths from training-free calibration | — |

**DuoAttention is the best fit** and it is worth saying why precisely: streaming
heads stop re-reading grown KV entirely, so the saving lands on traffic rather
than on FLOPs — which is the axis that matters on this box. It is also
per-head-heterogeneous, and the `-ctk-cpu`/`-ctv-cpu` machinery is already the
right shape to carry that.

**The combination is what is novel.** Composing per-head retrieval/streaming
(DuoAttention) × depth-decreasing budget (PyramidKV) × calibrated per-layer window
(MoA) **over per-layer KV cache *types*** — not just budgets — is not something
anyone has shipped, because nobody else exposes the types. It costs calibration
runs and a config generator, no kernel work.

**How this could be wrong:** all three were calibrated on dense models with
uniform KV precision. Interaction with `q8_0`/`q4_1` KV and with KIVI2 is
unstudied — a head classified "retrieval" at fp16 may not tolerate 2-bit. Validate
per-arm rather than assuming composition.

---

## 2. Absorb the rotations that are free

- **SpinQuant** (**2405.16406** · 2024-05-26, ICLR'25 · `facebookresearch/SpinQuant`)
  and **QuaRot** (**2404.00456** · 2024-04-01 · `spcl/QuaRot`): **R1 and R2 absorb
  entirely into adjacent weight matrices — zero inference cost.** `SpinQuant_no-had`
  (W4A8) is *fully* offline-absorbed. Only R3/R4 are online, and SpinQuant needs 2
  per block where QuaRot needs 4.

**This should be adopted regardless of everything else.** It is what makes 3–4 bit
weights viable at all, and at runtime it costs nothing — the rotated weights are
just weights.

**The online rotations, and a framing that is genuinely inverted here.** `docs/02`
already has a **wave64-safe Triton GEMM-based Walsh-Hadamard** (cosine 0.9838 at
3-bit, 0.9955 at 4-bit), built because TurboQuant's GPU path uses warp-32
shuffle/ballot. Everyone else deliberately *avoids* the GEMM form and hand-fuses
rotations as elementwise butterflies — specifically to keep tensor cores free for
the GEMM that matters. On CDNA2 the matrix cores are at parity with everything
else (181 across int8/f16/bf16) and idle at decode, so **rotation-as-MFMA-GEMM is
the natural form here.** No paper frames it as an advantage because on every other
chip it is a pessimization.

**Caveat that limits this:** per `docs/30` the scarce resource is *instruction
issue and kernel launches*, not FLOPs. An extra rotation GEMM must be **fused into
the adjacent kernel**, not added as its own launch, or it feeds the exact term
that causes the decode gap.

---

## 3. Low-rank error correction — a GEMM, not a gather

The distinction that decides everything on this chip: is the correction term a
**matmul** (cheap here) or a **table lookup / gather** (expensive here, because it
adds bytes and random access to an issue-bound kernel)?

| method | arXiv · date · code | correction form | verdict |
|---|---|---|---|
| **LQER / L²QER** | **2402.02446** · 2024-02-04, ICML'24 · `ChengZhang-98/lqer` | **low-rank GEMM**; the paper explicitly avoids scatter/gather so the kernel stays a regular blocked GEMM | **best fit** |
| **QERA** | **2410.06040** · 2024-10-08 · code | low-rank GEMM, **closed-form optimal rank** | good |
| **ZeroQuant-V2 / LoRC** | **2303.08302** · 2023-03-15, AAAI'24 · DeepSpeed | low-rank GEMM (SVD of the error) | good |
| **ResQ** | **2412.14363** · 2024-12-18 · code | 8-bit low-rank subspace (1/8 hidden) + rotation | good |
| CALDERA | **2405.18886** · 2024-05-29 · `pilancilab/caldera` | low-rank **+ QuIP# E8-lattice backbone** | mixed — backbone is a lookup |
| AQLM / QuIP# / QTIP / GPTVQ | **2401.06118** / **2402.04396** / **2406.11235** | codebook / trellis **lookup** | **avoid** — see below |

**The arithmetic that makes this attractive here.** A rank-32 correction on an
8192×8192 layer is:

- **0.78% of the FLOPs** (`2·8192·32·2 / (2·8192²)`)
- **+1.56% of the bytes** (`2·8192·32·2 / (8192²·1)` against int8 weights)

**And the question the literature does not answer:** *does 3-bit + rank-32 match
4-bit accuracy, and at what rank?* LQER and QERA report W4A8 near-lossless with
small `r`; **none publish a 3-bit-matches-4-bit result at a stated rank.** That is
an open experiment, and it is the right shape for this box because it wins by
**reducing bytes** — the only currency CDNA2 trades in — at ~zero added
arithmetic.

**Why the codebook family is wrong here despite better accuracy:** dequant is a
gather into a codebook, then a float GEMM. It never reaches
`v_mfma_i32_16x16x16i8`, it optimises FLOPs (irrelevant — 181 = 181), and it adds
random access to a kernel that `docs/30` shows is already issue-bound. QTIP's own
paper notes the LUT needs >10× GPU cache. Great papers, wrong chip.

---

## 4. Decode-phase KV tools that look relevant and are not

Recording these so they are not picked up later, because they are the obvious
search results for "offloaded KV":

| method | arXiv · date | why it does not apply |
|---|---|---|
| **ShadowKV** | **2410.21465** · 2024-10-28 | low-rank K on GPU + V offloaded, on-the-fly reconstruction. 3.04× on A100 — **decode phase**. Does not reduce prefill. |
| **InfiniGen** | **2406.19707** · 2024-06-28, OSDI'24 | KV in host DDR, GPU rehearses to pick essential entries and prefetches only those. Exactly the right *shape* — but it is a **decode-time speculator**; during prefill you are writing KV, not selecting from a full cache. |
| **Quest** | **2406.10774** · 2024-06-16, ICML'24 | page min/max metadata + top-k page selection. Decode, GPU-resident KV, reduces **FLOPs** not bytes. |
| **SnapKV** | **2404.14469** · 2024-04-22 | runs **full** prefill attention then compresses. No prefill benefit at all. |
| **SeerAttention** | **2410.13276** · 2024-10-17 | requires **training** a gate via self-distillation, plus a block-sparse FA kernel. |

The deeper reason all three of the first group miss: they reduce **KV bandwidth**,
and per the `docs/25` addendum decode here is not KV-bandwidth-bound — it is
~3× off a bound that already includes KV, i.e. bound by per-token instruction
issue. Right idea, wrong bound.

---

## Suggested order for this document's items

| # | Item | Cost | Risk |
|---|---|---|---|
| 1 | **SpinQuant absorbed R1/R2** on an existing W8A8/W4A16 checkpoint | offline, hours | very low — rotated weights are just weights |
| 2 | **DuoAttention** head classification → `-ctk-cpu`/`-ctv-cpu` per-head/per-layer config | calibration run + config generator | low — data only |
| 3 | **PyramidKV depth rule + MoA windows** layered on top | calibration | low, but validate the composition |
| 4 | **3-bit + rank-32 LQER** vs plain 4-bit — PPL *and* decode tok/s | offline quantize + one serving arm | medium — open in the literature |

All four are runnable while the GPUs are busy with other benchmarks, except item 4's
serving measurement.
