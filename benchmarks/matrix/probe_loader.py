#!/usr/bin/env python3
"""Separate the two costs that _load_w13's single line conflates.

py-spy puts every sample on `expert_data.copy_(loaded_weight)`, but that call
blocks on BOTH the mmap'd CPU read of the source AND the host-to-device
transfer. A profile cannot tell them apart; timing them separately can.

Uses a real MoE expert tensor from the checkpoint, sliced exactly the way
_load_w13 slices it (narrow on the shard dim for TP=2, rank 0).
"""
import glob, json, os, sys, time
import torch
from safetensors import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/models/bench-matrix/t35-bf16"
files = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))
assert files, f"no safetensors in {MODEL}"

# Find a routed-expert weight -- the tensors _load_w13 actually moves.
target = None
for f in files:
    with safe_open(f, framework="pt", device="cpu") as fh:
        for k in fh.keys():
            if "experts" in k and ("gate_proj" in k or "w13" in k or "up_proj" in k):
                target = (f, k); break
    if target: break
assert target, "no expert weight found"
path, key = target
print(f"file : {os.path.basename(path)}")
print(f"key  : {key}")

dev = torch.device("cuda")
TP = 2

def t(fn, n=3):
    # First call faults pages in; report it separately from steady state.
    torch.cuda.synchronize()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return first, (time.perf_counter() - t0) / n

with safe_open(path, framework="pt", device="cpu") as fh:
    w = fh.get_tensor(key)
    mb = w.numel() * w.element_size() / 1e6
    print(f"shape: {tuple(w.shape)}  dtype={w.dtype}  {mb:.1f} MB")
    print()

    shard_dim = 0 if w.ndim == 2 else 1
    per_rank = w.shape[shard_dim] // TP
    sliced = w.narrow(shard_dim, 0, per_rank)
    print(f"narrow(dim={shard_dim}, 0, {per_rank}) -> contiguous={sliced.is_contiguous()}")
    smb = sliced.numel() * sliced.element_size() / 1e6
    print(f"slice: {tuple(sliced.shape)}  {smb:.1f} MB")
    print()

    dst = torch.empty(sliced.shape, dtype=sliced.dtype, device=dev)

    print(f"{'operation':<44}{'first(s)':>10}{'steady(s)':>11}{'MB/s':>10}")
    print("-" * 75)
    for name, fn in [
        ("CPU .contiguous() on the narrowed slice", lambda: sliced.contiguous()),
        ("CPU clone of the slice",                  lambda: sliced.clone()),
        (".to(device) from strided slice",          lambda: sliced.to(dev)),
        (".to(device) from contiguous copy",        (lambda c=sliced.contiguous(): (lambda: c.to(dev)))()),
        ("dst.copy_(strided slice)   <- line 771",  lambda: dst.copy_(sliced)),
        ("dst.copy_(contiguous src)",               (lambda c=sliced.contiguous(): (lambda: dst.copy_(c)))()),
        ("pinned H2D (ceiling for this size)",      (lambda p=sliced.contiguous().pin_memory(): (lambda: dst.copy_(p, non_blocking=False)))()),
    ]:
        first, steady = t(fn)
        print(f"{name:<44}{first:>10.4f}{steady:>11.4f}{smb/steady:>10.1f}")
