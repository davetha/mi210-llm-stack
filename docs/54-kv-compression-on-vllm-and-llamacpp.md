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

## llama.cpp does the same job for a twentieth of the price

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

What round 65 actually licenses is narrower and still useful: **on the production stack, at short context, `q8_0`/`q4_1` costs 3.8% and `q4_0`/`q4_1` costs 6.1%, with coherent output.** That covers ordinary chat traffic, where prompts are short and this is the regime production runs in.

It does **not** establish that llama.cpp's KV quant holds up at 130K, which is exactly where vLLM's fell apart. Round 69 measures that; until it reports, the cross-stack comparison is open. Recording it as settled here would repeat `docs/50`'s mistake in a new place.

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

| want | on vLLM | on llama.cpp |
|---|---|---|
| ≤256K context, max speed | `bf16` — 902K tokens is plenty | `q8_0`/`q8_0` today |
| ≤256K, more concurrent streams | `bf16`; capacity is not binding | **`q8_0`/`q4_1`, −3.8%** — recommended |
| 1M context | `turboquant_4bit_nc` fits it, at ~1 tok/s | untested at this length |

The honest summary: **1M context on vLLM is technically reachable on this box and practically unusable.** The capacity is real and measured — 2.79M KV tokens. But turboquant's decode penalty grows with context rather than amortizing, and by 130K it is already down to 8.0 tok/s per stream. A 1M context that generates at ~1 tok/s is not a serving configuration.

This is worth stating flatly because the capacity number alone reads like a win. Reporting "3.09× more KV" without the slope would have been true and thoroughly misleading.

Two things follow. **256K is comfortable today** — bf16 holds 902K tokens, needs no compression, and is the fast path. **1M needs different hardware or a different KV implementation**, not a different flag. The measured gap between llama.cpp's −3.8% and vLLM's −80% at 130K is an implementation gap, so a better vLLM KV kernel is possible in principle; it just doesn't exist today.

The useful near-term result is round 65: llama.cpp gives up ~4% for a large V-cache reduction, and that one is close to free.

## Method note

Round 66 reported `0.00` for its 131,072-token row and I initially read it as a measurement. It wasn't — the server ran `--max-model-len 131072` while the benchmark asked for `--random-input-len 131072 --random-output-len 64`, so prompt plus generation exceeded the window and all four requests were rejected on both arms. A zero that means "nothing ran" looks exactly like a zero that means "infinitely slow".

Round 67 reserves headroom for the generation and the chat template, refuses to start if the arithmetic doesn't clear, and prints the failed-request count *before* any timing so an empty run announces itself. This is the same lesson as [`docs/50`](50-the-five-leads-measured.md)'s: a number produced by a broken run is worse than no number, because it gets tabulated.
