#!/usr/bin/env python3
"""Test the hypothesis that the loader is bound by per-transfer HSA memory
registration, not by data movement.

py-spy --native on the live TP=2 bf16 load, 8 consecutive samples, all identical:

    ioctl (libc.so.6)
    hsakmt_ioctl (libhsa-runtime64.so.1)
    _load_w13 (vllm/model_executor/layers/fused_moe/layer.py:771)

The worker is blocked in a kernel-driver ioctl, not in memcpy or DMA. A
pure-Python profile cannot see this -- it attributes the whole wait to the last
Python frame, which is why every earlier sample "proved" line 771 was the cost.

Mechanism this predicts: each expert tensor is a FRESH mmap'd host region, so
every copy_ makes ROCm register/map that region with the driver before it can
DMA from it. Registration is an ioctl with a large fixed cost; the 3.2 MB of
payload is irrelevant beside it. Consistent with everything observed --
first-touch slow and reuse fast, monotonically rising shard times as the
registration table grows, ~1 s constants independent of size, and pinned memory
being fastest in probe 1 because it is already registered.

If that is right, copying through ONE reused pinned staging buffer should be
dramatically faster than DMA'ing from each fresh mmap'd slice, because the
registration happens once instead of 18,432 times.

    A  direct from fresh mmap'd slice   <- what vLLM does today
    B  through a reused pinned buffer   <- the candidate fix
    C  direct from the SAME slice twice <- isolates registration from transfer

Run on an idle box. Round 3 of these probes was confounded by a concurrent load.
"""
import glob, os, sys, time
import torch
from safetensors import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/models/bench-matrix/t35-bf16"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
dev = torch.device("cuda")
torch.zeros(1, device=dev)

free, total = torch.cuda.mem_get_info()
print(f"VRAM: {free/1e9:.1f} GB free of {total/1e9:.1f} GB")
if free / total < 0.80:
    print("WARNING: <80% VRAM free -- something else is running. Results will be")
    print("         contended, which is exactly what invalidated probe round 3.")
print()

path = sorted(glob.glob(os.path.join(MODEL, "*.safetensors")))[0]
with safe_open(path, framework="pt", device="cpu") as fh:
    keys = [k for k in fh.keys() if "experts" in k][:N]

def slices(fh):
    """Fresh mmap-backed slices, sliced the way _load_w13 slices them."""
    out = []
    for k in keys:
        w = fh.get_tensor(k)
        d = 0 if w.ndim == 2 else 1
        out.append(w.narrow(d, 0, w.shape[d] // 2))
    return out

with safe_open(path, framework="pt", device="cpu") as fh:
    sl = slices(fh)
    mb = sum(s.numel() * s.element_size() for s in sl) / 1e6
    shape, dtype = sl[0].shape, sl[0].dtype
    uniform = all(s.shape == shape and s.dtype == dtype for s in sl)
    print(f"{len(sl)} tensors, {mb:.1f} MB total, uniform shape={uniform}")
    print()

    def run(label, fn):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"{label:<48}{dt:>9.3f}s{dt/len(sl)*1000:>11.2f} ms{mb/dt:>11.1f} MB/s")
        return dt

    print(f"{'method':<48}{'total':>10}{'per-tensor':>13}{'rate':>13}")
    print("-" * 84)

    dsts = [torch.empty(s.shape, dtype=s.dtype, device=dev) for s in sl]

    a = run("A  direct from fresh mmap slice  <- today",
            lambda: [d.copy_(s) for d, s in zip(dsts, sl)])

    b = None
    if uniform:
        stage = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        def through_pinned():
            for d, s in zip(dsts, sl):
                stage.copy_(s)          # CPU->CPU into already-registered memory
                d.copy_(stage)          # DMA from a buffer registered ONCE
        b = run("B  via reused pinned staging  <- candidate fix", through_pinned)

    one = sl[0]; d0 = dsts[0]
    run("C  same slice repeatedly (registration warm)",
        lambda: [d0.copy_(one) for _ in sl])

    print()
    if b:
        print(f"B is {a/b:.1f}x faster than A" if b < a else
              f"B is {b/a:.1f}x SLOWER than A -- hypothesis refuted")
