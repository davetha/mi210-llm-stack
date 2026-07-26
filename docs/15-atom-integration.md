# ATOM Framework Integration on MI210

**Date**: 2026-07-26
**Status**: ✅ Installed (v0.1.5), Llama 3 8B test pending

## Install

```bash
# In fa-build container:
cd /build
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git checkout v0.1.5                 # CRITICAL — see "Version compat" below
pip install --no-deps .
```

Location: `/build/ATOM` (git checkout v0.1.5, commit b0071c5)
Package: `/opt/python/lib/python3.14/site-packages/atom/`

## Version Compatibility Notes

### Why v0.1.5 not main HEAD

ATOM main HEAD adds three imports in `atom/model_ops/attention_mla.py` that
AITER 0.1.13 does not have:
```python
from aiter.ops.triton.attention.mla import mla_decode_fwd as triton_shuffle_mla_decode_fwd
from aiter.ops.triton.kv_cache import cat_and_cache_mla as triton_cat_and_cache_mla
from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
```

These modules landed in upstream AITER around June 2026, after our pinned
rev `b32deb267`. Because `base_attention.py:16` does
`from .attention_mla import MLAModules` at top level, every model import
(including Llama) fails on main HEAD.

v0.1.5 (stable, 2026-06-22) is the newest tag whose attention_mla.py only
imports modules that already exist in AITER 0.1.13.

### Why --no-deps

- `transformers==5.12.1` is pinned by ATOM, but container has 5.14.0 (vLLM needs)
- PyPI `zmq` is a placeholder ("You are probably looking for pyzmq")
- Container has pyzmq 27.1.0 which provides the `zmq` module

## Verified

```
python -c "import atom"
# atom dist version : 0.1.5
# aiter dist version: 0.1.13.post2.dev1+gb32deb267
# get_gfx()         : 'gfx90a'
```

```
python -m atom.entrypoints.openai_server --help
# Works. Flags: --model, -tp, --kv_cache_dtype {bf16,fp8}, --level, --method
```

```
python -c "from atom.models.llama import LlamaForCausalLM"
# OK — Llama 3 8B model class imports cleanly
```

## gfx90a-Specific Issues

**None.** No arch gating, no `raise`/`assert` on `get_gfx()` anywhere in v0.1.5.
FP8 KV cache paths exist and depend on AITER ops working on gfx90a at runtime.

## CLI Quick Reference

There is **no `atom` binary** — ATOM uses Python module entrypoints:

```bash
# OpenAI-compatible server
python -m atom.entrypoints.openai_server --model <hf-model> -tp 2

# One-shot inference
python -m atom.examples.simple_inference --model <hf-model>
```

## AITER Version Compatibility Matrix

| ATOM Version | AITER 0.1.13 Compat | Notes |
|--------------|---------------------|-------|
| v0.1.5 (installed) | ✅ Yes | All symbols verified present |
| main HEAD | ❌ No | Missing 3 newer aiter modules |
| Optional segmented MLA kernels | ⚠️ Absent | ATOM handles via try/except |

## Known Issues

1. **fmoe_b16.co ILLEGAL_INSTRUCTION**: The patched BF16 MoE dispatcher
   executes 1024 swapped MFMA ops but traps mid-kernel. Cause under
   investigation (likely an unidentified unsupported opcode variant).
   Impact: MoE models (DeepSeek, MiMo) blocked until fixed. Dense models
   (Llama 3) unaffected.

2. **transformers version mismatch**: We have 5.14.0, ATOM wants 5.12.1.
   `--no-deps` install avoids forcing downgrade (would break vLLM). Top-level
   `import atom` works on 5.14. Latent risk if model loader uses 5.12-only API.

3. **moe_sorting_ck works** (CK JIT kernel) but was crashing when called with
   wrong buffer sizes. Fixed by using AITER's own `moe_sorting_ck` helper which
   allocates correct buffers (M × model_dim).

## Next Steps

1. Try loading Llama 3 8B Instruct (dense, BF16, fits 64GB)
2. If that works, validate DeepSeek-V2 Lite 16B (MoE + MLA)
3. Debug fmoe_b16.co ILLEGAL_INSTRUCTION for production MoE models
