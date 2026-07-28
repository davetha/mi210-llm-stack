#!/usr/bin/env python3
"""The first probe showed line 771 running at 21.7 GB/s -- so the copy is NOT
the cost, and py-spy pointing there is misleading. But that probe called
get_tensor() OUTSIDE the timing loop, which pre-materialises the tensor and
pre-faults every page. The real loader iterates weights lazily.

This measures the whole per-tensor sequence the loader actually performs, and
attributes the time to get_tensor vs narrow vs copy_ separately.
"""
import glob, os, sys, time
import torch
from safetensors import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/models/bench-matrix/t35-bf16"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
dev = torch.device("cuda")
torch.zeros(1, device=dev)          # pay HIP init once, outside the measurement

path = files[0]
with safe_open(path, framework="pt", device="cpu") as fh:
    keys = [k for k in fh.keys() if "experts" in k][:N]
print(f"file   : {os.path.basename(path)}")
print(f"tensors: {len(keys)} expert weights")

t_get = t_narrow = t_copy = 0.0
total_bytes = 0
TP = 2
t_wall = time.perf_counter()
with safe_open(path, framework="pt", device="cpu") as fh:
    for k in keys:
        t0 = time.perf_counter()
        w = fh.get_tensor(k)
        t1 = time.perf_counter()

        shard_dim = 0 if w.ndim == 2 else 1
        per_rank = w.shape[shard_dim] // TP
        s = w.narrow(shard_dim, 0, per_rank)
        t2 = time.perf_counter()

        dst = torch.empty(s.shape, dtype=s.dtype, device=dev)
        dst.copy_(s)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_get += t1 - t0; t_narrow += t2 - t1; t_copy += t3 - t2
        total_bytes += s.numel() * s.element_size()
wall = time.perf_counter() - t_wall
mb = total_bytes / 1e6

print()
print(f"{'stage':<34}{'total(s)':>10}{'per-tensor(ms)':>16}{'% of wall':>11}")
print("-" * 71)
for name, v in [("safetensors get_tensor", t_get),
                ("narrow (view, should be ~0)", t_narrow),
                ("empty + copy_ to device", t_copy)]:
    print(f"{name:<34}{v:>10.3f}{v/len(keys)*1000:>16.3f}{100*v/wall:>10.1f}%")
print("-" * 71)
print(f"{'WALL':<34}{wall:>10.3f}{wall/len(keys)*1000:>16.3f}{100:>10.1f}%")
print()
print(f"moved {mb:.1f} MB in {wall:.2f}s  ->  {mb/wall:.1f} MB/s effective")
print(f"observed in the real load: ~2.2 MB/s per rank")
