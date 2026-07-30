# Which models can actually reach the ASM kernels — the support matrix

**Date**: 2026-07-30 · **Hardware**: 2× MI210 (gfx90a / CDNA2)
**Source**: enumerated from `aiter_meta/hsa/gfx90a` in the installed vLLM image.

`docs/28` tells you how to read a checkpoint's *quantization*. This is the other
half: which **architectures** can reach the hand-written AITER ASM that
`docs/33` measures at 539–1,058 MACs per issued instruction, against 195 for the
best compiled path.

The filters are narrower than the docs implied, and one of them was undocumented.

---

## The enumerated support matrix

### Prefill — `fmha_v3_fwd`, 48 objects

| head_dim | objects | dtype |
|---:|---:|---|
| **128** | 24 | bf16 |
| **192** | 24 | bf16 |

`head_dim = 192` is supported, confirming `docs/28`. **There are no fp16
variants** — every one of the 48 is `_bf16_`. That was not previously recorded.
It is a constraint on the *attention compute dtype*, not on weight quantization:
W8A8 and AWQ-Int4 checkpoints both run bf16 activations and are eligible.

### Decode — `pa`, 8 objects. This is the restrictive filter.

```
pa_bf16_noquant_gqa8_1tg_4w.co         pa_fp16_noquant_gqa8_1tg_4w.co
pa_bf16_noquant_gqa16_1tg_4w.co        pa_fp16_noquant_gqa16_1tg_4w.co
pa_bf16_..gqa8.._mtp_msk0.co / msk1    (+ the fp16 MTP pair)
```

**ASM paged attention exists only for GQA ratio 8 and 16.** Nothing else. No
ratio 4, 12, or 1.

This explains something already in the logs that went unnoticed: GLM-4.6 is
96 heads / 8 KV = **ratio 12**, so the prefetch-offload arm JIT-compiled a HIP
`pa_v1` template (`gqa_ratio=12, head_size=128`) instead of loading an ASM `.co`.
It was never eligible, and the 695.9 t/s in `docs/28` is with ASM **prefill**
only.

### MoE experts — `fmoe`, 8 objects

All `noquant{Fp16,Bf16}`, all `g1u0`, all `32x512` (`docs/33`). So: **unquantized
bf16/fp16 experts only**, plus `VLLM_ROCM_USE_AITER_MOE=1`, which this stack had
been setting to `0` by hand.

### MLA — `mla`, 11 objects. Status **open**, being measured.

`docs/19` records that `mla.py` gates ASM on gfx942/gfx950. Reading the installed
source does not support that:

- the `gfx942`/`gfx950`/`gfx1250` checks in `aiter/mla.py` are **perf-tuning**
  branches (`wg_per_split`, `num_warps`, block sizes), not availability gates
- `aiter/jit/core.py:882` reads `if (get_gfx() not in ("gfx942", "gfx90a")` —
  gfx90a explicitly inside an allowed set
- 11 objects exist for gfx90a, including
  `mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co`, matching the
  `aiter.mla_decode_stage1_asm_fwd` call site

That is the same shape as three claims already overturned here — `fmha_v3_fwd`
(80/80 exact), `pa_fwd_asm` (48/48 exact), `VLLM_ROCM_USE_AITER_MOE` (never
gated). It is equally possible the objects are present and simply never selected.
`round21_mla_asm_probe.sh` settles it with DeepSeek-V2-Lite (16B, MLA, ~31 GB) —
which has **16 query heads**, matching the `QH16` objects, and which `docs/15`
already listed as a never-completed validation step.

---

## The target profile

To reach every ASM path a model must satisfy **all** of:

| axis | requirement | why |
|---|---|---|
| `head_dim` | **128 or 192** | `fmha_v3_fwd` coverage |
| GQA ratio (`heads / kv_heads`) | **exactly 8 or 16** | `pa` coverage |
| attention dtype | **bf16** | no fp16 `fmha_v3_fwd` objects |
| MoE expert weights | **unquantized bf16/fp16** | `fmoe` is `noquant` only |
| env | `VLLM_ROCM_USE_AITER_MOE=1` | otherwise the backend is removed |

**You already own a model that satisfies all five**: Qwen3-30B-A3B is
`head_dim=128`, 32/4 = **ratio 8**, MoE, and `t35-bf16` is on disk. Round 19's
bf16 arms are the first configuration that can exercise fmha + pa + fmoe ASM
simultaneously.

Note the tension between rows 3–4 and `docs/28`: unquantized bf16 is exactly what
that doc discourages because of the 12,366 s load. `docs/34`'s
`--load-format sharded_state` removes that objection.

---

## Reading a config for KV compression — and the trap

Three fields decide it, no download required:

```bash
curl -s https://huggingface.co/ORG/MODEL/raw/main/config.json | python3 -c '
import json,sys; c=json.load(sys.stdin)
if "kv_lora_rank" in c:                       # CHECK THIS FIRST
    kb = c["num_hidden_layers"]*(c["kv_lora_rank"]+c.get("qk_rope_head_dim",0))*2/1024
    print(f"MLA: kv_lora_rank={c[\"kv_lora_rank\"]}  KV={kb:.0f} KiB/token")
else:
    h,kv = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    d = c.get("head_dim", c["hidden_size"]//h)
    kb = 2*c["num_hidden_layers"]*kv*d*2/1024
    print(f"{\"MHA (no compression)\" if h==kv else f\"GQA ratio {h/kv:g}\"}  KV={kb:.0f} KiB/token")'
```

**`kv_lora_rank` must be checked before the head counts.** When MLA is present,
`num_key_value_heads` is vestigial — it can equal `num_attention_heads` while the
actual cache is a compressed latent tens of times smaller.

That is not hypothetical. Reading GLM-5.2's config by eye, `num_attention_heads:
64` and `num_key_value_heads: 64` were taken to mean MHA with no compression,
giving ~3.8 MB/token and ~123 GiB of KV at 32k — cited as disqualifying. The
config also contains `kv_lora_rank: 512` and `qk_rope_head_dim: 64`, so the real
figure is:

```
78 × (512 + 64) × 2 = 87.75 KiB/token  →  2.74 GiB @ 32k
```

**A 44× error, in the pessimistic direction, from reading the fields in the wrong
order.** The script above gets it right; the eyeball did not.

---

## Worked example: GLM-5.2

`QuantTrio/GLM-5.2-Int8` — 754B parameters, **705 GiB**.

Quantization is close to ideal for this box: `compressed-tensors` (the fast
loader, not `gptq`), `type: "int"` 8-bit weights (not FP8, which has no ALU
here), `input_activations: null` → W8A16, the format that posted the best tier-2
decode. MTP head intact.

It still cannot run: **705 GiB against 128 GB VRAM + 499 GB DDR4 = 627 GB
total**, short by ~130 GB before KV, activations and the OS. There is no offload
strategy for bytes with nowhere to live.

And the ASM picture would not resemble GLM-4.6's even if it fit:

| axis | GLM-5.2 | eligible? |
|---|---|---|
| attention | **MLA** (`kv_lora_rank: 512`) | `mla/` family — status open |
| GQA ratio | 64/64 = 1 | ✗ no `pa` ASM |
| sparse attention | **IndexShare** — see below | unknown on ROCm |

### What "indexer" means

Full attention has every query attend to every cached position. Sparse attention
attends to a selected subset; the **indexer** is the module that selects it.
GLM-5.2's config:

```
index_n_heads: 32   index_head_dim: 128   index_topk: 2048
index_topk_freq: 4  indexer_types: [full, full, full, shared, shared, ...]
```

A lightweight 32-head / 128-dim scorer ranks cached positions against the current
query and keeps the top **2,048**; real attention runs over those only — 2,048
instead of 1,000,000 at 1M context, hence the claimed 2.9× FLOP reduction.
**IndexShare** is `index_topk_freq: 4`: compute the selection once and reuse it
across four layers.

It reduces FLOPs and KV *reads*, **not KV storage** — the full cache is still
required to select from. Here that is fine, because MLA already made storage
small.

### Variants, filtered against CDNA2

**Dead**: every FP8 build (`zai-org/GLM-5.2-FP8`, baseten, Bahushruth), every
NVFP4 (`nvidia`, 381B), MXFP4/MXFP8 hybrids, `W4AFP8` (FP8 activations), EXL3
(wrong engine). No FP8 or FP4 ALU on gfx90a.

**Plausible**:

| repo | approx size | note |
|---|---:|---|
| `pipenetwork/GLM-5.2-REAP50-Q3_K_M-GGUF` | ~190 GB | REAP expert pruning to 381B |
| `pipenetwork/GLM-5.2-REAP50-Q2_K-GGUF` | ~150 GB | close to GLM-4.6 IQ3_XS's 135 GB |
| `QuantTrio/GLM-5.2-Int4-Int8Mix` | ~400–450 GB | fits RAM with prefetch offload |
| `sokann/GLM-5.2-GGUF-2.244bpw` | ~211 GB | aggressive |

**The GGUF options get no AITER at all.** llama.cpp is a different stack —
ggml-hip with its own `mma.cuh` — and the AITER `.co` objects live in the vLLM
image. This is structural, not a gate. GGUF still reaches the matrix cores
through llama.cpp's own correct CDNA2 MFMA (`docs/33`), just not through ASM.

So the ordering is: settle the MLA question with the 31 GB probe before spending
400 GB on a GLM-5.2 that may reach no ASM path at all.
