# 🎉 AiterBackend WORKS on MI210

> ⚠️ **SUPERSEDED** — corrected: the 'hybrid ASM prefill' attribution -- no ASM ran.
>
> Performance figures attributed to ASM kernels here were **mis-attributed**.
> `mha.py` and `mla.py` gate their ASM paths on gfx942/gfx950, so on gfx90a
> these measured the CK or Triton fallback. The throughput is real; the
> attribution is not.
>
> Current status: [`../docs/19-aiter-operator-port-matrix.md`](../docs/19-aiter-operator-port-matrix.md).
> Kept as a record of the investigation, including the dead ends.

**Date**: 2026-07-27
**Status**: ✅ End-to-end text generation via ATOM AiterBackend on gfx90a

## Working Configuration

```bash
ATOM_LOADER_USE_THREADPOOL=0 python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 \
  --max-tokens 10 \
  --enforce-eager \
  --level 0
```

No ATOM_USE_UNIFIED_ATTN needed! Default AiterBackend works!

## Patches Applied

1. **dispatch_backend** (attention_mha.py): Decode routes to `paged_attention_triton`
   instead of `paged_attention_asm` (avoids pa_fwd_asm numerical cascade fault)

2. **paged_attention_triton** (attention_mha.py): Always uses `unified_attention`
   instead of `pa_decode_gluon` (which doesn't support gfx90a)

3. **Binary patches**: 1,251 .co files patched gfx942→gfx90a
4. **CK headers**: Extracted from official v0.1.17 wheel
5. **sitecustomize.py**: Triton pre-import for LLVM crash fix

## Architecture

- **Prefill**: ASM flash_attn_varlen_func (AITER native, fast)
- **Decode**: unified_attention (Triton JIT, native gfx90a)

This hybrid gives ASM speed for prefill + Triton correctness for decode.

## Performance

- TTFT: 0.857s
- TPOT: 0.029s (34.5 tok/s decode per request)
- 4 concurrent requests, all completed successfully
- Zero memory faults, zero assertions

## Sample Output

```
"introduce yourself" → "<think>\nOkay, the user asked me to introduce"
"list all prime numbers within 100" → "<think>\nOkay, so I need to list all"
"1+2+3=?" → "<think>\nOkay, so I need to solve "
"如何在一个月内增肌10公斤" → "<think>\n嗯，用户问的是如何在一个月"
```
