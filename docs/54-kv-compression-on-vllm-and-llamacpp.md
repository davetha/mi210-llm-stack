# KV Cache Compression: What Actually Works on gfx90a

Rounds 64–68. The question behind all of them: **can this box serve a 256K or 1M context?**

Everything in [`docs/50`](50-the-five-leads-measured.md)–[`docs/53`](53-the-power-cap.md) fought over 1–3% of throughput. This is a different kind of question. A few percent of decode speed changes a benchmark; KV capacity changes whether a request can be served at all.

## The capacity wall, stated first

At TP=2 on the W8A8 model, the server reports its KV budget at startup. Measured, twice, from the server's own log:

| KV dtype | GPU KV cache | Longest single context |
|---|---:|---|
| `bf16` (today) | **902,160 tokens** | fits 256K comfortably; **cannot fit 1M** |
| `turboquant_4bit_nc` | **2,790,992 tokens** | 3.09× — fits a 1M context with room to spare |

So the 1M case is not a tuning problem. bf16 physically cannot hold a 1M-token context on two MI210s, and no amount of scheduling changes that. Compression is the only lever, which is why it was worth measuring properly rather than assuming.

The 3.09× measured against a claimed 3.8× is close enough to call the flag working — the gap is block-allocation rounding and the non-KV portion of the reservation, not a failed setting.

## At short context, vLLM compression costs about half your throughput

Round 64, W8A8 model, 8192-token inputs, 256 output tokens, zero failed requests on every arm. "Half" turns out to be the *best* case — see the context curve below:

| arm | conc 8 tok/s | conc 32 tok/s | vs bf16 @32 | conc 32 TPOT |
|---|---:|---:|---:|---:|
| `bf16` | 147.96 | 232.97 | — | 127.48 ms |
| `turboquant_4bit_nc` | 87.01 | 119.16 | **0.511×** | 254.16 ms |
| `turboquant_k8v4` | 87.62 | 119.45 | 0.513× | 253.74 ms |

All three correctness probes returned coherent output. This is a performance finding, not a numerics failure.

### The two turboquant rows are the important part

`k8v4` compresses 2.6×. `4bit_nc` compresses 3.8×. They move **different amounts of KV** and they perform **identically** — 119.45 vs 119.16 tok/s, 0.2% apart, which is inside run-to-run noise.

If the penalty came from moving KV bytes, the arm moving fewer bytes would have won. It didn't. **The cost is the backend, not the bandwidth.**

That single comparison is what turns this from "quantization is expensive here" into "*this implementation* is expensive here" — and those have very different consequences.

`turboquant_k8v4` also started cleanly, which was not expected: round 58b killed fp8 KV on this card with `AssertionError: Unsupported dtype: torch.float8_e4m3fn`. TurboQuant's fp8 key path does not hit that assertion. Worth recording, because it means the round 58b failure was narrower than "fp8 KV is dead on gfx90a".

## Longer context makes it worse, not better

The obvious hypothesis after round 64: 8192-token inputs are where KV is small, so turboquant's cost is mostly fixed overhead with little KV to amortize it against. Decode re-reads the entire KV every token, so KV bytes scale with context while a fixed overhead does not. If so, the penalty should shrink as context grows, and somewhere there is a crossover.

Round 66/67 tested that directly — one request at a time, prompt of *N* tokens, 64 decode steps, TPOT as the metric, because TPOT is per-token decode latency and that is what KV traffic drives.

The hypothesis was wrong, and not in a marginal way. Round 67, full curve:

| context | bf16 TPOT | turboquant TPOT | ratio | bf16 TTFT | turboquant TTFT |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 17.02 ms | 25.11 ms | **1.475×** | 853 ms | 869 ms |
| 65,536 | 20.58 ms | 70.99 ms | **3.449×** | 15,991 ms | 16,500 ms |
| 130,000 | 23.87 ms | **124.45 ms** | **5.214×** | 50,105 ms | 52,525 ms |

Round 66 measured the 8,192 point independently, from a separate process and a separate server start, and got 16.90 / 25.48 ms against round 67's 17.02 / 25.11 — 0.7% and 1.5% apart. Both runs also reported byte-identical KV budgets (902,160 and 2,790,992). The rig repeats, so the curve below is signal.

The slope — ms of TPOT added per 1,000 tokens of context — is the number that settles it:

| span | bf16 | turboquant | ratio |
|---|---:|---:|---:|
| 8,192 → 65,536 | 0.0621 | 0.8001 | 12.89× |
| 65,536 → 130,000 | 0.0510 | 0.8293 | **16.25×** |

bf16's slope *falls* as context grows — it amortizes. TurboQuant's holds flat at ~0.83 and the gap widens. A kernel paying a fixed overhead would converge toward the baseline; this one diverges from it.

That is the signature of a decode kernel less efficient *per KV byte*, not one paying a startup cost. Combined with the k8v4-vs-4bit_nc result — where byte count didn't matter at all — the picture is a Triton decode path whose per-token work scales worse with sequence length than AITER FA's does.

**Long context is where turboquant is worst.** That is exactly backwards from the only reason to want it.

At 130,000 tokens, read as per-stream generation speed: **41.9 tok/s on bf16 against 8.0 tok/s on turboquant.** Extrapolating the measured 0.83 ms/1k slope to a 1M context lands near ~1 tok/s. The capacity to hold 1M exists; the speed to use it does not.

### Prefill is *not* what it costs — a correction

I expected turboquant to forfeit AITER FA's 1.19–1.33× prefill win ([`docs/45`](45-aiter-fast-path-survey.md)), since it is its own attention backend and does not appear in AITER FA's `supported_kv_cache_dtypes`.

The TTFT column says otherwise. Across a 16× span of context the two arms are within **1.9%, 3.2% and 4.8%** — prefill is essentially untouched. Whatever the arms do differently, it is not paying for prefill.

So the bill is narrower than predicted but far larger in total: the entire cost is in decode, and decode is where it compounds with context.

## The other vLLM KV quantizer is worse, and that closes the question

TurboQuant is not vLLM's only KV quantization. `int4/int8/fp8_per_token_head` are handled in `v1/attention/backends/triton_attn.py` — the generic Triton backend, a **different kernel** from turboquant's. Round 64's result does not transfer to it in either direction, so round 68 measured it rather than assuming.

| metric @ 65,536 | `bf16` | `turboquant_4bit_nc` | `int8_per_token_head` |
|---|---:|---:|---:|
| TPOT | 20.45 ms | 70.99 ms (3.45×) | 61.72 ms (**3.02×**) |
| TTFT | 15,454 ms | 16,500 ms (1.03×) | **105,890 ms (6.85×)** |
| KV capacity | 902,160 | 2,790,992 (3.09×) | 1,749,648 (1.94×) |

Its decode slope is slightly better than turboquant's — 0.685 vs 0.800 ms per 1,000 tokens — and still eleven times bf16's 0.062. But it inverts turboquant's one virtue: where turboquant left prefill alone, this spends **106 seconds** prefilling a 64K prompt against bf16's 15.5. And it compresses less. TurboQuant dominates it on capacity and prefill; it wins only a sliver of decode.

`int8_per_token_head` at 1.94× would clear 1M on capacity alone. The prefill cost makes that academic.

### `int4_per_token_head` is not actually available

Enumerating `vllm.config.cache.CacheDType` returns sixteen values including `int4_per_token_head`. The server's argparse accepts **fifteen**, and that is not one of them:

```
vllm serve: error: argument --kv-cache-dtype: invalid choice: 'int4_per_token_head'
```

The type alias and the CLI disagree. Enumerating the type overstates what can actually be selected — a reminder that a name in a config type is not a supported configuration until something accepts it. (The same caution as `docs/50`'s: a gap in a config table is a hypothesis, not a finding — and so is an entry in one.)

**With both vLLM KV quantizers measured and both unaffordable, the conclusion is not about a flag choice. vLLM has no usable KV compression on gfx90a today.**

## llama.cpp looked twenty times cheaper, and wasn't

Round 65, on the **production** Coder-Next Q4_K_M model, same box:

| KV config | bits/value (K/V) | decode tok/s | vs baseline |
|---|---|---:|---:|
| `q8_0` / `q8_0` (production today) | 8.5 / 8.5 | 76.81 | — |
| `q8_0` / `q4_1` | 8.5 / 5.0 | **73.88** | **0.962×** (−3.8%) |
| `q4_0` / `q4_1` | 4.5 / 5.0 | 72.12 | 0.939× (−6.1%) |

All probes coherent. Prompt processing moved 64.30 → 62.25 → 61.86 tok/s, a 3.8% spread across the whole sweep.

### This is NOT yet a fair comparison to the vLLM numbers

The tempting headline — "llama.cpp does it for −3.8% where vLLM charges −49%" — is not supported by these two experiments, and the reason is the whole point of rounds 66/67.

Round 65 ran `--ctx-size 32768`, but that is the *allocation*, not the load. The actual probe prompt is about twenty tokens with `n_predict 96`. So that −3.8% was measured against an essentially **empty KV cache**. vLLM's −49% was measured at 8,192 real input tokens, and its penalty at near-zero context would have been smaller too.

Rounds 66/67 established that KV-quant cost is strongly context-dependent — turboquant went 1.48× → 5.21× across a 16× context span. Comparing a llama.cpp number taken at ~0 tokens of KV against a vLLM number taken at 8K, and then against one taken at 130K, compares three different regimes.

What round 65 licenses is therefore very narrow: at a prompt of roughly twenty tokens, `q8_0`/`q4_1` costs 3.8% and `q4_0`/`q4_1` costs 6.1%, with coherent output. That is a statement about overhead at zero depth, not about serving.

### Round 70: the full depth matrix, and production is on the wrong setting

Round 70 re-measured with `llama-bench -d`, which prefills *N* tokens of KV and then times generation — the same quantity as vLLM's TPOT-vs-context curve. Every (arm, depth) pair ran as its own process, so a blank is a real failure at that depth and not collateral from another cell.

Decode tok/s, production Coder-Next 80B Q4_K_M, ratios against `q8_0`/`q8_0` at the same depth:

| arm | 8,192 | 32,768 | 65,536 | 98,304 | 130,000 |
|---|---:|---:|---:|---:|---:|
| `q8_0`/`q8_0` (production) | 71.03 | 61.79 | 52.87 | 45.78 | 40.17 |
| `q8_0`/`q4_1` | 35.07 (0.494×) | 15.11 (0.245×) | timeout | timeout | timeout |
| `q4_0`/`q4_1` | 39.88 (0.561×) | 17.25 (0.279×) | timeout | timeout | timeout |
| **`f16`/`f16`** | **76.41 (1.076×)** | **71.95 (1.164×)** | **69.16 (1.308×)** | **64.99 (1.420×)** | **62.89 (1.566×)** |

Three findings, in ascending order of how much they matter.

**1. `q4_1` V-cache does not merely cost — it stops working.** 51% at depth 8,192, 75% at 32,768, and beyond that it cannot produce 32 tokens within 25 minutes. For scale, `q8_0`/`q8_0` completed all five depths in 3.4 minutes total. The −3.8% from round 65 was measuring overhead against an empty cache and nothing else.

**2. The entire penalty is the V cache; quantizing K is free.** `q8_0`/`q4_1` and `q4_0`/`q4_1` differ *only* in K precision, and the 4-bit-K arm is slightly **faster** (39.88 vs 35.07). Both collapse identically because both carry `q4_1` V. vLLM shows the same asymmetry — `turboquant_k8v4` (fp8 K) and `turboquant_4bit_nc` (4-bit K) landed within 0.2% of each other. Two independent engines, same conclusion: **K is cheap to quantize, V is not.**

**3. `f16` beats production's `q8_0`/`q8_0` at every depth, and the gap widens.** 1.076× at 8,192 rising to **1.566× at 130,000**. Production is running a KV quantization that *costs* 57% of decode speed at long context.

That third result inverts the premise the setting rests on. Quantized KV is supposed to trade a little compute for less memory traffic. On gfx90a the dequant work exceeds the bandwidth saved, and the deficit compounds with depth. Even `q8_0` — the mildest option available — loses.

`f16` also decays far more gracefully: **0.823×** across a 16× depth increase, against `q8_0`/`q8_0`'s 0.565×.

The cross-stack conclusion is now consistent across two engines and five KV configurations: **on gfx90a, quantizing the V cache costs at least half of decode at real depth, and the cost compounds. Uncompressed KV is both faster and better-behaved.**

### The production recommendation

`llama-swap-config.yaml` passes `-ctk q8_0 -ctv q8_0` for three models. **Switch them to `-ctk f16 -ctv f16`.** Expected: 1.08–1.31× decode across the configured context sizes, at zero quality cost — f16 is the exact baseline, not an approximation.

The constraint is VRAM: f16 KV is ~1.88× the size of `q8_0`. Two facts bound the risk:

- Round 70 ran f16 to **depth 130,000** on the 48.5 GB model without allocation trouble, so it fits comfortably at that scale.
- Of the configured contexts, four are `-c 65536` and two are `-c 32768` — all well inside what was demonstrated.

The two outliers, `-c 196608` and `-c 262144`, are **not** covered by this measurement and should be tested before switching. Those are the cases where `q8_0` K may still earn its place — and note finding 2 above: `-ctk q8_0 -ctv f16` is likely the right compromise there, since K quantization is nearly free and V quantization is what hurts. That combination was not measured and is the obvious next round.

### What uncompressed llama.cpp does at depth

**62.89 tok/s at 130,000 tokens of depth, on an 80B model.** vLLM bf16 manages 41.9 tok/s at the same depth on a **35B**. The production stack is doing more than twice the model at 1.5× the speed.

That is the real path to long context on this box: uncompressed KV plus enough VRAM, on either stack. Not compression.

## Rounds 71–72: the box serves a 1M context, uncompressed, at 27.5 tok/s

Every conclusion above was drawn on vLLM, and vLLM said 1M was out of reach. The production stack was never asked. Direct VRAM measurement on llama.cpp with the production Coder-Next 80B Q4_K_M, both cards summed, of 131,040 MiB total:

| context | total VRAM | free |
|---:|---:|---:|
| 65,536 | 49,572 MiB | 78 GB |
| 196,608 | 53,664 MiB | 77 GB |
| 262,144 | 55,710 MiB | 75 GB |
| **1,048,576** | **80,267 MiB** | **50 GB** |

**A 1M context allocates with 50 GB to spare** — uncompressed, f16 KV, on an 80B model. vLLM could not fit 1M on a *35B*.

### Why KV is so small here, and why that undercuts the whole compression premise

Weights are 46,235 MiB of the 49,572 at 64K, so KV runs about **32 KB/token** — roughly 32 GB at 1M. Qwen3-Coder-Next is a **hybrid GDN** architecture: most layers are gated-delta-net linear attention carrying a constant-size recurrent state, and only a minority carry a growing KV cache.

That reframes round 70's result. Measured at 64K:

| KV config | total VRAM | saved vs f16 | decode cost at 130K |
|---|---:|---:|---:|
| `f16`/`f16` | 49,572 MiB | — | — |
| `q8_0`/`q8_0` (production) | 49,063 MiB | **509 MiB (1.0%)** | **−36%** |
| `q4_1`/`q4_1` | 48,454 MiB | 1,118 MiB (2.3%) | unusable past 32K |

Production is trading **1% of VRAM for 36% of decode speed.** Compressing a cache that is already a small fraction of residency cannot pay, whatever the compression ratio.

This is architecture-specific and worth stating: on a dense full-attention model, KV would be a much larger share of VRAM and the memory case for quantizing would be real. The *speed* penalty measured in round 70 would still apply, but the trade would not be this lopsided.

### Fitting is not serving — the depth curve to 1M

| depth | tok/s | vs d=8,192 | round |
|---:|---:|---:|---|
| 8,192 | 76.41 | 1.000× | 70 |
| 32,768 | 71.95 | 0.942× | 70 |
| 65,536 | 69.16 | 0.905× | 70 |
| 98,304 | 64.99 | 0.851× | 70 |
| 130,000 | 62.89 | 0.823× | 70 |
| 262,144 | 52.72 | 0.690× | 72 |
| 524,288 | 40.55 | 0.531× | 71 |
| **1,000,000** | **27.47** | **0.360×** | 71 |

**27.47 tok/s at a million tokens of depth.** For comparison, vLLM's only 1M-capable configuration extrapolates to roughly 1 tok/s, and it is lossy. This one is exact.

The decay is gradual and shows no cliff: 0.823× at 130K, 0.690× at 262K, 0.531× at 524K, 0.360× at 1M — each doubling of depth costs progressively less than a proportional share.

### The 262,144 point is why this table has a round-72 row

The first measurement at 262,144 returned **6.68 tok/s** — an eighth of the curve's prediction, and *slower than the 524,288 point above it*. Re-running it produced a hard **GPU memory access fault**. Run a third time from an idle GPU, the same configuration returned **52.72 tok/s**, exactly on the curve.

Both bad readings began moments after tearing down a container holding tens of GB of VRAM. `docker rm -f` returns before the driver has finished reclaiming, so the next process races the teardown. GPUs were at 47 °C junction and 42 W when checked, so this was not thermal.

**The non-monotonicity is what saved it.** Decode cannot get *faster* with more KV, so the curve announced its own bad sample. Had 262,144 been the deepest point measured, with nothing above it to contradict, 6.68 tok/s would have been indistinguishable from a real finding — and "llama.cpp falls off a cliff past 130K" is exactly the kind of conclusion that gets written down.

Round 71's script now waits for VRAM to drop below 512 MiB before each cell, and warns rather than proceeding silently if it does not.

## KIVI does not apply here, for two separate reasons

Both were worth checking because KIVI2 is *our own* implementation ([`changes/02`](../changes/02-kivi2-quant-type.md)) and the natural thing to reach for.

**1. vLLM has no KIVI.** The server accepts fifteen `--kv-cache-dtype` values — `auto`, `float16`, `bfloat16`, five `fp8*` variants, four `turboquant_*`, `int8_per_token_head`, `fp8_per_token_head`, and `nvfp4`. No KIVI, and nothing that could be aliased to it. Adding it would mean writing and upstreaming a new backend, not setting a flag.

**2. KIVI2 is CPU-only, so it would not help even on llama.cpp.** All nine files the type touches are host-side; no CUDA or HIP backend defines a KIVI2 dequant kernel, so a `GGML_TYPE_KIVI2` tensor has no reader on the device. Production runs `-ngl 999`.

`docs/03` and `changes/02` both claimed KIVI2 "works on GPU too". **That claim was wrong and both are now corrected.** "Pure scalar C" means architecture-agnostic, not GPU-capable — the two got conflated. The tell was sitting in the usage example the whole time: the per-layer form pairs `-ctk-cpu kivi2` with `-ctk f16`, because the GPU layers have to be fp16.

KIVI2 remains correct and useful for what it was built for — CPU-pinned layers of a 230B MoE re-reading KV from DDR4 at ~7.4 s/chunk. That is a real bottleneck. It is not the bottleneck a fully GPU-resident deployment has.

## There is no CPU KV offload in vLLM

Worth stating plainly, because it is the obvious escape from the capacity wall and it does not exist:

- `--cpu-offload-gb` offloads **weights**, not KV.
- `--swap-space` holds KV for **preempted** sequences — swapped out when a request is descheduled, swapped back before it resumes. It does not extend the working set of a running request.
- Prefix caching and LMCache are L1/L2 **reuse** layers. `get_num_new_matched_tokens()` returns tokens matched *beyond* local hits; they avoid recomputing a prefix, they do not let active KV live in host RAM.

Active KV must be VRAM-resident. There is no `-ngl`-style dial. Given the 902,160-token bf16 ceiling, compression is the only path to 1M on this hardware.

## Where this leaves the 1M question

**The box serves a 1M context today, at 27.5 tok/s, on llama.cpp, uncompressed.** That is the answer, and none of the work that produced it involved compression.

| want | on vLLM | on llama.cpp |
|---|---|---|
| ≤130K, max speed | `bf16` — 902K tokens is plenty | **switch to `f16`/`f16`** — 1.08–1.57× over the `q8_0` in use today |
| 196K–262K | `bf16` covers it | `f16`, measured: 55,710 MiB at 262K, 52.72 tok/s |
| **1M** | **cannot fit it** — bf16 caps at 902,160 tokens | **`f16`, 80,267 MiB with 50 GB spare, 27.47 tok/s** |

Three things this investigation got backwards before it got them right.

**1. The question was framed on the wrong stack.** Rounds 64–68 established that 1M on vLLM is technically reachable and practically unusable — 2.79M KV tokens via `turboquant_4bit_nc`, decaying to 8.0 tok/s at 130K and extrapolating near 1 tok/s at 1M. All true. But vLLM was never the only option, and the stack production actually runs was not asked until round 71. It fits 1M with 38% of VRAM unused.

**2. Compression was assumed to be the lever.** Every arm measured refutes it — `turboquant_4bit_nc`, `turboquant_k8v4`, `int8_per_token_head`, `q8_0`/`q4_1`, `q4_0`/`q4_1`. Five configurations, two engines, two independent flash-attention implementations, and the cheapest still costs half of decode at depth. On gfx90a the dequant work exceeds the bandwidth saved, and the deficit compounds with depth.

**3. The memory savings were never checked.** This is the one that should have come first. Quantizing KV on this model saves **509 MiB — 1% of VRAM** — because a hybrid GDN architecture keeps most layers on constant-size recurrent state. Every round spent measuring what compression *costs* was measuring the price of something that was never buying much. One `mem_info_vram_used` read, available from the start, would have reordered the entire investigation.

The capacity number is what made this seductive: "3.09× more KV tokens" reads like a win in isolation. It took the depth curve to show the cost and the VRAM reading to show the benefit was ~1%.

### Still open

`bf16` on vLLM holds 902,160 KV tokens against the 1,000,000 a 1M context needs — **10.8% short, not 3× short** — with `--gpu-memory-utilization` at 0.90. That gap is plausibly closable, and unlike everything else here it would keep vLLM's aggregate-throughput advantage. Unmeasured.

`-ctk q8_0 -ctv f16` is also unmeasured. Round 70 showed quantizing K is free and V is what costs, so it should capture most of the (small) memory saving at near-zero speed penalty. Only worth pursuing on a model where KV is a meaningful share of VRAM — which this one is not.

## Method note

Round 66 reported `0.00` for its 131,072-token row and I initially read it as a measurement. It wasn't — the server ran `--max-model-len 131072` while the benchmark asked for `--random-input-len 131072 --random-output-len 64`, so prompt plus generation exceeded the window and all four requests were rejected on both arms. A zero that means "nothing ran" looks exactly like a zero that means "infinitely slow".

Round 67 reserves headroom for the generation and the chat template, refuses to start if the arithmetic doesn't clear, and prints the failed-request count *before* any timing so an empty run announces itself. This is the same lesson as [`docs/50`](50-the-five-leads-measured.md)'s: a number produced by a broken run is worse than no number, because it gets tabulated.

Round 69 then made a different version of the same mistake. It asked `llama-bench` for all three depths in **one invocation**, and all four arms died with a GPU memory access fault. Because `llama-bench` writes its CSV at exit, the crash at the deepest point destroyed the completed rows for the shallower ones — four arms, twenty-three minutes, zero usable numbers. Worse, the fault invited the conclusion that llama.cpp had hit a capacity ceiling at 130K.

It had not. Round 70 ran every (arm, depth) pair as its own process, and `q8_0`/`q8_0` completed 130,000 cleanly at 40.17 tok/s. The ceiling was an artifact of batching the cells together. **A harness that discards good data when one point fails will eventually discard the data the round existed to collect** — and the failure will look like a finding about the hardware.

Round 70 also separates the three ways a cell can return nothing (timeout, GPU fault, OOM) instead of reporting them all as "no rows", because round 69 reported a harness defect in language that sounded like a hardware limit.

### The correction this document exists to record

Round 65's `-3.8%` was the most quotable number in this investigation and it was nearly published as a cross-stack comparison. It was measured with a ~20-token prompt against an essentially empty KV cache. The same configuration at depth 8,192 costs **51%**, and at 32,768 costs **75%**.

The error would not have been in the measurement — round 65's number is reproducible and correct for what it measured. It would have been in the comparison: setting a zero-depth llama.cpp number beside an 8K-depth vLLM number and reporting the ratio as if the two were the same experiment. Rounds 66 and 67 had *already* established that KV-quant cost is strongly depth-dependent, which is precisely what made the comparison illegal.

The general form: **before comparing two numbers, check that they were produced by the same shape of experiment.** A flag that is set (`--ctx-size 32768`) is not the same as a condition that obtains (32,768 tokens actually resident).
