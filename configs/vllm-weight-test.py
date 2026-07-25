import os, sys
os.environ["VLLM_USE_AITER"] = "0"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "1"
os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx90a"

print("=== vLLM Weight Loading Test ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"vLLM path: {os.path.dirname(__import__('vllm').__file__)}", flush=True)

# Check model files exist
model_path = "/models/Qwen3-235B-A22B-GPTQ-Int4"
config_file = os.path.join(model_path, "config.json")
if os.path.exists(config_file):
    print(f"Config found: {config_file}", flush=True)
    import json
    with open(config_file) as f:
        config = json.load(f)
    print(f"Model: {config.get('architectures', 'unknown')}", flush=True)
    print(f"Quant: {config.get('quantization_config', {}).get('quant_method', 'unknown')}", flush=True)
    print(f"Hidden: {config.get('hidden_size', '?')}", flush=True)
else:
    print(f"ERROR: No config at {config_file}", flush=True)
    sys.exit(1)

# Count safetensors files
import glob
safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
print(f"Safetensors files: {len(safetensors)}", flush=True)
total_size = sum(os.path.getsize(f) for f in safetensors) / (1024**3)
print(f"Total size: {total_size:.1f} GB", flush=True)

# Try loading
print("\n=== Attempting model load ===", flush=True)
try:
    from vllm import LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        quantization="gptq",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
    )
    print("SUCCESS: Model loaded!", flush=True)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
sys.exit(0)
