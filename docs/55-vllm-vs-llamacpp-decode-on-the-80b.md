# vLLM cannot reach llama.cpp's decode on the 80B — and `docs/24` is stale

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

| | llama.cpp Q4_K_M | vLLM W8A8 |
|---|---:|---:|
| prefill @~16k | 2,546 t/s | **7,249 t/s** (2.85×) |
| decode, single stream @130k | **62.89 t/s** (1.35×) | 47.44 t/s |
| weights | **45 GB** | 77 GB |
| max context | **1M, measured** | 130k tested; 1.4M KV tokens |

**Prefill-dominated work** — long prompts, RAG, agentic tool loops, large code context — favours vLLM, and by a wide margin. At 16k tokens that is 2.3 s against 6.4 s to first token.

**Decode-dominated work** — chat, long conversations, sustained generation — favours llama.cpp, which also holds a 1M context (see [`docs/54`](54-kv-compression-on-vllm-and-llamacpp.md)) and uses 40% less weight memory.

vLLM's remaining structural advantage is batching: its aggregate column is a real number where llama.cpp has no comparable story. That is the case for vLLM, not decode.

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
