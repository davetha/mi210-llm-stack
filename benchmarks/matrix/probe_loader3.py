#!/usr/bin/env python3
"""Probe 2 put 100% of the time in `torch.empty(device) + copy_`, ~1001 ms per
tensor, while probe 1 measured copy_ alone into a PREALLOCATED dst at 21.7 GB/s.
The difference is the allocation. This splits them.

CONFOUND TO RULE OUT: a bf16 arm is loading on both cards right now. If VRAM is
near-full, torch.empty may be falling through the caching allocator to hipMalloc
or triggering a defrag/synchronize -- which would make this an artifact of test
conditions rather than a property of the loader. Free VRAM is reported so the
result can be read correctly either way.
"""
import glob, os, sys, time
import torch
from safetensors import safe_open

MODEL = "/models/bench-matrix/t35-bf16"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
dev = torch.device("cuda")
torch.zeros(1, device=dev)

free, total = torch.cuda.mem_get_info()
print(f"VRAM on this device: {free/1e9:.1f} GB free of {total/1e9:.1f} GB "
      f"({100*free/total:.0f}% free)")
print("NOTE: a bf16 arm is loading concurrently; read allocation timings with that in mind.")
print()

path = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))[0]
with safe_open(path, framework="pt", device="cpu") as fh:
    keys = [k for k in fh.keys() if "experts" in k][:N]
    tensors = [fh.get_tensor(k) for k in keys]

sl = []
for w in tensors:
    d = 0 if w.ndim == 2 else 1
    sl.append(w.narrow(d, 0, w.shape[d] // 2))
mb = sum(s.numel() * s.element_size() for s in sl) / 1e6

def timed(label, fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fn(); torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{label:<46}{dt:>9.3f}s{dt/len(sl)*1000:>12.2f} ms/tensor{mb/dt:>11.1f} MB/s")
    return dt

print(f"{'operation':<46}{'total':>10}{'per-tensor':>15}{'rate':>16}")
print("-" * 87)

# A: allocate only
def alloc_only():
    global _keep
    _keep = [torch.empty(s.shape, dtype=s.dtype, device=dev) for s in sl]
a = timed("torch.empty(device) only", alloc_only)

# B: copy into the buffers just allocated
def copy_only():
    for d, s in zip(_keep, sl): d.copy_(s)
b = timed("copy_ into preallocated buffers", copy_only)

# C: the loader's actual pattern, alloc+copy interleaved
def alloc_copy():
    for s in sl:
        d = torch.empty(s.shape, dtype=s.dtype, device=dev)
        d.copy_(s)
c = timed("empty + copy_ interleaved  <- loader pattern", alloc_copy)

# D: same, but reusing one buffer per distinct shape (what a fix would do)
def pooled():
    pool = {}
    for s in sl:
        k = (tuple(s.shape), s.dtype)
        if k not in pool: pool[k] = torch.empty(s.shape, dtype=s.dtype, device=dev)
        pool[k].copy_(s)
timed("pooled buffer reuse (candidate fix)", pooled)

print()
print(f"allocation is {a/max(b,1e-9):.0f}x the cost of the copy it feeds")
