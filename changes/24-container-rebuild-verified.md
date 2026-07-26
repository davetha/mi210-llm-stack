# Container Rebuild + Triton Backend Verified

**Date**: 2026-07-27
**Status**: ✅ ATOM inference working on MI210 after full container rebuild

## Rebuild Summary

Container `fa-build` was destroyed by docker prune. Rebuilt from scratch:
1. Base: `rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`
2. AITER v0.1.17: built from source (`pip install --no-build-isolation .`)
3. CK headers: extracted from official v0.1.17 wheel (93 include + 7646 CK files)
4. ATOM v0.1.5: `pip install --no-deps .`
5. sitecustomize.py: Triton pre-import
6. Binary patches: 1,251 .co files re-applied

## Verified Working Config

```bash
ATOM_USE_UNIFIED_ATTN=1 ATOM_LOADER_USE_THREADPOOL=0 \
python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 \
  --max-tokens 5 \
  --block-size 64 \
  --enforce-eager \
  --level 0
```

Result: EXIT=0, coherent text, zero faults.
