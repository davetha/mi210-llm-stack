# Change Log: vLLM GGUF Plugin — DeepSeek MoE Blocked

## Date: 2026-07-25

## Summary
Tested vLLM GGUF plugin (vllm-gguf-plugin 0.0.4) on gfx90a. Plugin installs and registers correctly, but **DeepSeek-V2 MoE GGUF models cannot be loaded** due to missing expert fusion in the plugin's weight adapter.

## What Works
- Plugin installs cleanly on Python 3.14 / ROCm 7.14 / vLLM 0.25.2
- Auto-registers via `vllm.general_plugins` entry point
- GGUF config parsing succeeds (deepseek_v2 architecture recognized)
- Engine core initializes
- Device init starts
- This is NOT an AMD/gfx90a issue — it's a plugin architecture gap

## What's Blocked
```
RuntimeError: Failed to map GGUF parameters (26):
['model.layers.1.mlp.experts.gate_up_proj', ...]
```

## Root Cause
- GGUF stores experts as separate stacked tensors: `blk.N.ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`
- vLLM's DeepseekV2 expects fused: `model.layers.N.mlp.experts.gate_up_proj`
- Plugin has MoE fusion adapters for: qwen2_moe, qwen3_moe, olmoe, block_sparse_moe
- **NO deepseek_v2 fusion case exists** in the plugin

## To Fix
Contribute a `deepseek_v2` case to `vllm_gguf_plugin/weights_adapter/default.py` that:
1. Maps `ffn_gate_exps` + `ffn_up_exps` → fused `experts.gate_up_proj`
2. Maps `ffn_down_exps` → `experts.down_proj`
3. Maps `ffn_*_shexp` → `shared_experts.gate_up_proj` / `down_proj`

## Alternatives
1. Use HF safetensors format (vLLM loads natively, no GGUF needed)
2. Use llama.cpp (native GGUF support, faster kernels)
3. Contribute the fusion adapter upstream
