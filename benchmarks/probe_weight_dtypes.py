"""Report the dtype actually stored in VRAM for each parameter class.

vLLM's "Model loading took N GiB" line already implies whether an FP8
checkpoint was dequantized at load time, but that is an inference from a
number. This reads the tensors themselves, which is not.
"""
import os
import sys
from collections import defaultdict

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from vllm import LLM

model = sys.argv[1]
llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.35,
          max_model_len=2048, enforce_eager=True, disable_log_stats=True,
          attention_config={"backend": "ROCM_AITER_FA"})

# The engine object graph moves between versions; try the known paths.
runner = None
for path in (
    lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner,
    lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner,
    lambda: llm.llm_engine.model_executor.driver_worker.model_runner,
):
    try:
        runner = path()
        break
    except AttributeError:
        continue
if runner is None:
    print("DTYPE_PROBE_FAIL: could not reach model_runner", file=sys.stderr)
    raise SystemExit(1)

net = runner.model
by_dtype = defaultdict(lambda: [0, 0])  # dtype -> [tensor count, bytes]
for name, p in net.named_parameters():
    slot = by_dtype[str(p.dtype)]
    slot[0] += 1
    slot[1] += p.numel() * p.element_size()
for name, b in net.named_buffers():
    slot = by_dtype[str(b.dtype)]
    slot[0] += 1
    slot[1] += b.numel() * b.element_size()

print("=== stored parameter dtypes ===")
total = 0
for dt, (n, nbytes) in sorted(by_dtype.items(), key=lambda kv: -kv[1][1]):
    total += nbytes
    print(f"  {dt:<24} {n:>5} tensors  {nbytes / 2**30:8.2f} GiB")
print(f"  {'TOTAL':<24} {'':>5}           {total / 2**30:8.2f} GiB")

# Name a couple of concrete transformer linears so the result is not just an
# aggregate: these are the layers that would be dequantized at load time.
print("=== sample projection weights ===")
shown = 0
for name, p in net.named_parameters():
    if name.endswith("weight") and any(
        k in name for k in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    ):
        print(f"  {name:<55} {str(p.dtype):<22} {tuple(p.shape)}")
        shown += 1
        if shown >= 4:
            break
print(f"torch.cuda.memory_allocated = {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
