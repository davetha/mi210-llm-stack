# vLLM cannot reach llama.cpp's decode on the 80B — and `docs/24` is stale

> **READ ROUND 76 FIRST.** Rounds 73–75 below compare vLLM on
> Qwen3-Next-80B-**Thinking** against llama.cpp on Qwen3-**Coder**-Next-80B —
> two different models. On matched weights the prefill claim survives (2.70×)
> but **the decode gap narrows from 1.22× to 1.11× at 130K**, and vLLM's decode
> curve turns out to be far flatter with depth. Every decode ratio before the
> round 76 section is measured across a model difference and should not be
> quoted. The round 74/75 *within-vLLM* comparisons (CK GEMM, W4A16 vs W8A8)
> are unaffected — those hold the model fixed.

Rounds 73–74. The question: **can vLLM's 80B decode be brought up to llama.cpp's, with every known win enabled?**

Rounds 70–71 measured llama.cpp on the production Coder-Next 80B Q4_K_M with f16 KV at **62.89 tok/s at depth 130,000**. `docs/24` measured vLLM's 80B decode at 45.19 tok/s (W8A8) and 51.34 (W8A16) at ~101k, but those numbers predate the CK int8 GEMM ([`docs/43`](43-ck-int8-gemm-gfx90a.md), 1.480× on the 30B). The arithmetic 45.19 × 1.48 = 66.9 suggested vLLM would clear it.

**It does not.** The CK GEMM does not transfer to the 80B.

## The measurement

Round 74, TP=2, W8A8, depth 130,000, single stream. Both arms **assert** on the kernel actually selected, read out of the server log:

| arm | kernel (asserted) | decode | vs llama.cpp | TPOT | agg @16k c8 |
|---|---|---:|---:|---:|---:|
| Triton int8 GEMM | `TritonInt8ScaledMMLinearKernel` | **47.44 t/s** | 0.754× | 21.08 ms | 49.03 |
| AITER CK int8 GEMM | `AiterInt8ScaledMMLinearKernel` | 46.60 t/s | 0.741× | 21.46 ms | **37.51** |
| llama.cpp Q4_K_M, f16 KV | — | **62.89 t/s** | 1.000× | — | — |

**CK GEMM on the 80B: 0.982×** — a slight regression, against 1.480× on the 30B. It also costs **23% of aggregate throughput** at concurrency 8.

vLLM trails by **1.35×** on single-stream decode, and no remaining lever closes it:

- **DP=2 is structurally impossible.** `docs/52` measured DP=2 + shuffled KV at 1.118×, the best throughput result in the project. DP replicates the full model per rank, and `t80-w8a8` is 77 GB against 64 GB cards. Measured with `du -sh`, not assumed.
- **Shuffled KV layout** is worth ~2% (round 73: 44.62 → 46.19).
- **The CK GEMM is negative here.**

### Why it does not transfer — hypothesis, not measured

The 80B is Qwen3-Next, a hybrid GDN architecture: most layers are gated-delta-net linear attention, and per `docs/24`'s own correction the MoE experts run **W8A16**, not the a8w8 path. That leaves the dense a8w8 GEMM a much smaller share of decode than in the 30B, which is a standard MoE transformer. `docs/43` also records that `a8w8_tuned_gemm.csv` has **zero gfx90a rows**, so the CK kernel runs untuned instances.

Both are plausible and neither is measured. Recorded as a hypothesis so it is not mistaken for a finding.

## `docs/24`'s decode comparison is stale, and this does not fix it

`docs/24` concluded "bf16 wins decode at equal TP by 44%" and attributed the W8A8 loss to a poorly-tuned Triton int8 MoE kernel — pointing at the fix that became `docs/43`. It has never been updated to say whether that fix changed the conclusion.

For the **80B**, this document answers it: the CK GEMM does not help, so `docs/24`'s ranking stands there. For the **30B**, it remains open — `docs/43`'s 1.480× was measured at 27,852 tokens against a different baseline than `docs/24`'s decode@101k, and no head-to-head with CK enabled on both arms has been run.

## The stack choice, stated from measurements

**Superseded by round 76** — the table that was here compared two different models and two non-optimal checkpoints. The matched-model version, `t80-awq` (W4A16) against `t80-gguf-q4km` (Q4_K_M), both Qwen3-Next-80B-A3B-Thinking:

| | llama.cpp Q4_K_M, f16 KV | vLLM W4A16 |
|---|---:|---:|
| prefill @16k | 2,610 t/s | **7,059 t/s (2.70×)** |
| decode @8k | **72.76 t/s (1.27×)** | 57.34 t/s |
| decode @130k | **61.41 t/s (1.11×)** | 55.31 t/s |
| decode decay 8k→130k | −15.6% | **−3.5%** |
| weights | **45.4 GB** | 46.0 GB |
| KV capacity | 1M measured ([`docs/54`](54-kv-compression-on-vllm-and-llamacpp.md)) | 2,334,585 tokens |
| long-context stability | **intermittent GPU faults** (see below) | none observed |

**Prefill-dominated work** — long prompts, RAG, agentic tool loops, large code context — favours vLLM by a wide margin, and that margin is an engine property that survived the model correction. At 16k tokens it is 2.3 s against 6.3 s to first token.

**Decode-dominated work** favours llama.cpp, but by **1.11× at 130K rather than the 1.22× claimed above**, and the advantage shrinks as context grows. At short context it is a real 1.27×.

vLLM's structural advantages are batching — its aggregate column has no llama.cpp equivalent — and stability at long depth. llama.cpp's are short-context decode and 40% smaller weights at the same quant level.

## Round 75: W4A16 is the better vLLM checkpoint, and it bounds the decode question

Two open questions after round 74: *why* is vLLM's decode slower, and does a checkpoint exist with llama.cpp's footprint but vLLM's prefill? `t80-awq` answers both — 46 GB of W4A16 (4-bit group-wise weights, null activations) against `t80-w8a8`'s 77 GB, same image, TP=2.

| config | weights | prefill @16k | decode @130k | KV capacity |
|---|---:|---:|---:|---:|
| **W4A16** (`t80-awq`) | **46 GB** | **7,063 t/s** | **51.57 t/s** | **2,334,585 tok** |
| W8A8 (`t80-w8a8`) | 77 GB | 7,806 t/s | 46.90 t/s | 961,015 tok |
| llama.cpp Q4_K_M, f16 KV | 45 GB | 2,546 t/s | 62.89 t/s | — |

**W4A16 vs W8A8: prefill 0.905×, decode 1.100×.** It gives up 9.5% of prefill and gains 40% of weight memory, 10% of decode, and **2.43× the KV capacity** — while still prefilling **2.77× faster than llama.cpp**.

On every axis except that 9.5%, W4A16 is the better vLLM configuration for this model. It also fits a 1M context, which corrects an over-general claim in [`docs/54`](54-kv-compression-on-vllm-and-llamacpp.md).

The prediction that W4A16 would lose prefill badly — dequant to bf16 at large M, no int8 MFMA benefit — was directionally right and much too strong. 9.5%, not a collapse.

### The decode gap is 82% latency, and no checkpoint fixes it

This round was written to test a binary: if decode improved with 4-bit weights, decode is bandwidth-bound and `docs/50`'s latency-bound roofline (measured on the 30B) does not transfer to the 80B. Decode improved, and the script printed exactly that.

**The binary was too coarse, and the magnitude says the opposite.** Halving weight bytes per decode step — ~3.0 GB to ~1.5 GB at ~3B active params — bought only **1.10×**. Bandwidth-bound decode would have bought nearly 2×. Solving for the split between weight-traffic time `W` and everything else `O`:

```
(W + O) / (W/2 + O) = 1.10   ->   O = 4.5 · W
```

Weight traffic is **~18% of decode time**. The remaining 82% is latency and serialization, which **confirms** `docs/50` at 80B scale rather than refuting it.

That yields a hard bound. With *free* weights, `O` alone gives `5.5/4.5 = 1.222×` over W8A8 — about **57.3 tok/s**, still short of llama.cpp's 62.89.

**No vLLM checkpoint closes the decode gap by getting smaller.** The gap is kernel efficiency, not weight bytes, and W4A16 has already collected most of what weight size has to give.

A pre-registered binary is still worth writing — it stops a post-hoc story. But a sign test on a quantity whose *magnitude* carries the information will mislead, and this one did. The script's printed conclusion is wrong and is superseded by this section.

### Caveat

`t80-awq` and `t80-w8a8` are Qwen3-Next-80B-A3B-Thinking; llama.cpp's is Qwen3-Coder-Next-80B-abliterated. Same family, both A3B, not identical models.

## Round 76: the same model on both engines, and it moves the answer

Every comparison above puts two **different models** against each other:

```
vLLM       t80-awq / t80-w8a8   Qwen3-Next-80B-A3B-THINKING
llama.cpp  coder-next-q4        Qwen3-Coder-Next-80B-abliterated
```

Same family, both 80B-A3B, and every caveat line says so — which is honest and does not make the comparison valid. An "engine" difference that also changes the model measures both at once.

The matched pair was on disk the whole time: `t80-awq` (W4A16, 46.0 GB) and `t80-gguf-q4km` (Q4_K_M, 45.4 GB), both **Qwen3-Next-80B-A3B-Thinking**, footprints within 1.3%. Both arms generate 32 tokens.

| metric | llama.cpp Q4_K_M, f16 KV | vLLM W4A16 | vLLM/llama |
|---|---:|---:|---:|
| prefill @16,384 | 2,610 t/s | **7,059 t/s** | **2.70×** |
| decode @8,192 | **72.76** | 57.34 | 0.79× |
| decode @32,768 | **70.91** | 55.96 | 0.79× |
| decode @65,536 | **68.02** | 55.71 | 0.82× |
| decode @130,000 | **61.41** | 55.31 | 0.90× |

**Prefill survives the correction; decode does not.**

- **Prefill 2.70×** against the 2.77× claimed on mismatched models. That is an engine property.
- **Decode: llama.cpp leads 1.27× at 8K, narrowing to 1.11× at 130K** — not the **1.22×** this document claimed above. Part of that gap was the model.

### The shape matters more than the ratio

vLLM's decode is far flatter with depth than llama.cpp's:

| | 8,192 → 130,000 |
|---|---:|
| vLLM W4A16 | 57.34 → 55.31 (**−3.5%**) |
| llama.cpp Q4_K_M | 72.76 → 61.41 (**−15.6%**) |

The curves are converging. Linear extrapolation of the two slopes crosses near **~210,000 tokens**, beyond which vLLM would decode faster. That is extrapolation and should not be quoted as a result — round 71 shows llama.cpp's curve *steepening* past 130K (62.89 → 52.72 → 40.55 → 27.47 at 130K/262K/524K/1M) rather than staying linear, which would move the crossover earlier. Neither engine has been measured past 130K on matched weights.

The flatness is consistent with round 75's decomposition: vLLM's decode is ~82% latency and serialization, and that component does not grow with KV depth. llama.cpp's steeper decline suggests a larger share of its decode time *is* KV-dependent — it starts faster and gives ground as context grows.

### Prefill repeatability

Round 76 measured vLLM W4A16 prefill at 7,059.1 t/s against round 75's 7,063.3 — **0.06% apart**, on separate server starts. The rig is sound; the variance in this project comes from images and GPU state, not the measurement.

## llama.cpp faults intermittently at long depth

The 65,536 cell in round 76 first returned **21.07 tok/s** — non-monotonic against the 61.41 measured at *twice* the depth, so the curve announced its own bad sample again. Re-running it from a fully idle GPU (10 MiB VRAM) produced a hard **GPU memory access fault**. Two further attempts returned **68.08** and **67.95** tok/s, 0.2% apart, which is the true value and sits smoothly on the curve.

Tallying the long-depth llama.cpp runs across rounds 69, 71 and 76: roughly **half** either faulted outright or returned degraded numbers, and the failures are not explained by GPU state — this one recurred from a fully idle card.

**This is a production reliability concern, not a benchmarking nuisance.** The engine that wins single-stream decode is also the one that intermittently dies at long context on this hardware. It is unquantified — no controlled repeat-count exists — and it deserves its own round before llama.cpp is trusted at 130K+ in production.

## Method note: round 73 measured nothing, and the bug was in the round

Round 73 reported CK GEMM at 1.014× on the 80B. That was two Triton arms.

`round73` set `IMG=` for the **benchmark client** container and never set `VLLM_IMAGE`, which is what `serve_vllm_aiter.sh` reads for the **server**:

```bash
IMAGE="${VLLM_IMAGE:-rocm-vllm-aiter-gfx90a:latest}"
```

So every server ran the default image, while round 73's header recorded that the CK carve-outs had been verified in `vllm-mi210:gdnpolicy` — a different image that was never serving. The pre-flight check was performed correctly on the wrong container.

Confirmed afterwards by reading the selection line out of running servers:

| image | selects |
|---|---|
| `rocm-vllm-aiter-gfx90a:latest` | `TritonInt8ScaledMMLinearKernel` |
| `vllm-mi210:v0.26.1rc0-ckgemm-warm` | `AiterInt8ScaledMMLinearKernel` |

This is the **fourth** instance in this project of an arm-launch path silently running something other than what the round claimed — `VLLM_IMAGE` in round 31, `NCCL_P2P_DISABLE` in round 32, `VLLM_PREFER_AITER_FA` in round 37 — and `serve_vllm_aiter.sh` carries a comment warning about exactly this class of error.

**The lesson that keeps not taking: verifying that a FLAG is set proves nothing about which KERNEL ran.** Round 74 therefore greps the server log for the selected kernel and aborts the arm if it does not match what the arm intends. A round that cannot prove which kernel ran does not get to report a ratio.

The 1.4% that round 73 reported was also below the threshold of its own dead-flag guard, which only warned under 1%. A guard calibrated to noise will pass anything that looks like a small real effect.

### Image choice moves decode more than most flags

Round 74's Triton arm measured 47.44 tok/s where round 73's measured 44.62 — the same kernel on the same model, **6% apart**, differing only in image. Several of the flags chased across `docs/50`–`docs/53` were worth less than that. Any A/B that does not hold the image fixed is measuring the image.
