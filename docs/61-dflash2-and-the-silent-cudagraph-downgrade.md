# DFlash2 on gfx90a — and the silent cudagraph downgrade that hid 60% of decode

2026-08-21. Two results, one of which is worth more than the model port itself.

## 1. DFlash2 runs on CDNA2, and it is the fastest drafter measured here

[DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) is a
block-diffusion drafter: one pass proposes a whole block of candidate tokens, a
lightweight selector traces a coherent path through them, and verification is
lossless. vLLM support lives in PR #52816, which is **not** in any released
build — the `dflash` name appears in the config enum, but the
`DFlash2DraftModel` architecture is unregistered, so serving fails at config
validation. (A method name in an enum is not support.)

Port: 14 commits cherry-picked onto our mi210 integration branch (fork branch
`dflash2-int`), two trivial registry conflicts, image built with the usual
AITER layering. The draft model is 3.8 GB, 5 layers.

**It works on gfx90a**: output exact on a greedy ground-truth probe, and draft
acceptance measured **434/440 = 98.6%** on structured text.

## 2. The finding that mattered more: vLLM was silently halving decode

`vllm/config/vllm.py::_maybe_override_dynamic_sd_cudagraph_mode` overrides
`cudagraph_mode` to **PIECEWISE** whenever speculative decoding is dynamic and
the v2 model runner is off. It logs a warning and continues. On 2× MI210 that
override cost **60% of decode throughput**.

```
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Verify from the log, never by assumption:
`Capturing CUDA graphs (FULL)` is right; `PIECEWISE` means the downgrade fired.

| config (Qwen3.8-27B W8A8, TP2, AITER on) | 2K | 41K | 101K |
|---|---:|---:|---:|
| MTP n2 (previous production) | 76.0 | 19.4 | 9.1 |
| DFlash2, graphs auto → PIECEWISE | 105.6 | 27.7 | 13.0 |
| DFlash2, eager (no graphs) | 86.5 | — | — |
| **DFlash2 + FULL_DECODE_ONLY** | **168.2** | **30.6** | **13.6** |

**2.21× / 1.58× / 1.49×** over the previous production config, at the full 256K
context window.

### A retraction worth keeping

An earlier reading of this same data blamed `--max-model-len`: 64K measured 168
tok/s, 256K measured 105, so the context cap looked like the lever, and a 64K
production cap was briefly deployed. **Context length was innocent** — the
short-context boots simply happened to retain FULL graphs. 256K runs at the
same 168 tok/s once the mode is forced. The capture-mode line was in the logs
the whole time; nobody read it.

### AITER adds nothing on top of DFlash2

105.6 with AITER vs 105.5 without, at equal settings — inside noise, where the
same flag was worth **2.3×** under MTP drafting. Left enabled (it does no harm),
but the credit here belongs entirely to the drafter. Hypothesis, untested:
verifying a block of 8 moves the GEMM shapes off the M values where AITER's
int8 path wins on this card.

## 3. Three operational rules this cost a day to learn

1. **ROCm releases VRAM well after `docker rm -f` returns.** Relaunch within
   ~10 s and profiling OOMs against a card reporting *0 bytes free* while
   PyTorch accounts only for its own allocation. Wait for no `VLLM::`
   processes + 12 s, then confirm `rocm-smi --showmemuse` reads 0%. Three
   separate "findings" in this campaign were this and nothing else.
2. **Boots take 6–9.5 minutes** on these images. Poll at least 12 minutes
   before calling a server wedged, and grep for `startup complete`.
3. **Pegged CPU is not evidence of compiling.** vLLM workers busy-wait in
   `shm_broadcast` at ~200% while entirely idle. `py-spy dump` (with
   `--cap-add SYS_PTRACE` on the container) settles it in seconds — here it
   showed every thread idle and EngineCore already in its post-init loop,
   retiring a day-long phantom "torch.compile infinite loop".

## Serving config in production

Image `:20260821-aiter`; `--speculative-config '{"method":"dflash","model":"/models/qwen38-dflash2","num_speculative_tokens":8}'`;
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`; AITER on; TP2;
gmu 0.72; `--max-model-len 262144`; `--max-num-batched-tokens 8192`;
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Launcher:
`big:/home/dave/launch-qwen38.sh`. Untried: `VLLM_USE_V2_MODEL_RUNNER=1`, the
path the vLLM warning itself names, may allow graphs on prefill too.

## 4. The headline is best-case text — and the real ceiling is elsewhere

The 168 tok/s above was measured generating a **numbered list**, the most
drafter-friendly text there is. Same server, same 41K context, only the
requested text changed:

| generation at 41K | DFlash2 n=8 | no speculation |
|---|---:|---:|
| predictable (numbered list) | 30.6 | 6.2 |
| unpredictable (prose) | 6.9 | 6.2 |

DFlash2 is worth **4.9× on predictable text and +11% on prose**. It never loses
(lossless, wasted drafts are cheap), so it stays on — but no single number
describes this box without stating the text profile. Real agent traffic showed
17–25% draft acceptance versus 98.6% on the list probe.

**The actual ceiling is base decode at long context**: 33.9 tok/s at 2K falls to
6.2 at 41K with no speculation at all. Bandwidth cannot explain it — KV at 41K
is ~2.8 GB (~2 ms) against ~160 ms per step. The logs give the cause:

```
Setting attention block size to 784 tokens to ensure that attention page size
  is >= mamba page size
Cannot use ROCm custom paged attention kernel, falling back to Triton
```

Qwen3.8-27B is **hybrid** (GDN/mamba + full attention). vLLM pads the attention
page size to match mamba's, and stride-padded hybrid layouts are deliberately
routed to the Triton decode path (`chunked_prefill_paged_decode.py`,
`has_native_kv_cache_layout` → `use_custom = False`). It is **not** a head_dim
gate and **not** switchable: `VLLM_ATTENTION_BACKEND=ROCM_AITER_FA` is ignored
("Overriding with ROCM_ATTN") and measures identical to the digit.

This is now the biggest optimization target on this hardware — worth an
estimated 3–5× at agent-sized contexts, and it needs code: teach the ROCm
custom paged-attention kernel the stride-padded hybrid layout, or align the
hybrid page sizes so no padding is needed.

> **Correction (2026-08-22).** The size estimate held — production went 3.0× at
> 41K and 3.6× at 101K — but the diagnosis above is wrong in two ways, and
> acting on it as written wastes the effort.
>
> The layout gate is **not** what forces Triton. Relaxing it is measurably
> inert: `block_size 784 > 64` trips a separate veto in
> `use_rocm_custom_paged_attention` first, because the free kernel was
> *numerically wrong* above block_size 64 on gfx90a. Nor is the "stride padding"
> padding — it is an exact 2× K/V interleave, and the HIP kernel does take
> `kv_block_stride`/`kv_head_stride`, contrary to what the Python wrapper's
> signature suggests.
>
> More importantly, **this whole paragraph is about a kernel a spec-decode
> deployment never reaches.** With DFlash2 running, `query_len = 9 > 1` routes
> every decode step to `context_attention_fwd` instead. The fix that actually
> moved production was partitioning *that* kernel's cached-context scan.
> See [docs/62](62-attention-partitioning-and-the-kernel-nobody-was-running.md).

## 5. What the client sends dominates all of it

| opencode configuration | prompt tokens | decode |
|---|---:|---:|
| as configured (12 MCP servers) | 69,516 | 5.2–6.4 |
| 5 heavy MCP servers disabled | 50,774 | 7.1 |
| `--pure` (no plugins/MCP) | 17,478 | 18.1 |

A one-line question costs **69.5K prompt tokens**, ~75% of it MCP tool schemas.
Prefix caching hides the prefill on repeat turns (TTFT 64.6 s → 2.4 s), but the
KV is re-read every decode step, so the tool payload sets the decode rate for
the entire session. MCP hygiene is worth ~3× — more than any server flag
currently available.
