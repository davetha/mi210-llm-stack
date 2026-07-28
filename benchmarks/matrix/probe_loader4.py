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
    B  through a reused pinned buffer   <- candidate fix, needs a vLLM patch
    C  direct from the SAME slice twice <- isolates registration from transfer
    D  from heap memory (eager load)    <- candidate fix, NEEDS NO PATCH

D is the interesting one. vLLM already ships a load strategy that avoids mmap
entirely -- `--safetensors-load-strategy=eager`:

    with open(st_file, "rb") as f:
        state_dict = load(f.read())     # whole shard into a heap bytes object

Those tensors are backed by ordinary heap memory in one large allocation, not by
a per-tensor mmap view. If the registration hypothesis holds, that means one
registration per shard instead of one per tensor -- and the fix is a flag rather
than a patch. Costs a few GB of RAM per shard, which is nothing against 499 GB.

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

# D: heap-backed tensors, exactly what --safetensors-load-strategy=eager yields.
# Deserialised OUTSIDE the timing loop so this measures the transfer, not the
# read -- the same mistake probe 1 made in the other direction.
from safetensors.torch import load as st_load

with open(path, "rb") as fh_raw:
    eager_sd = st_load(fh_raw.read())
eager = []
for k in keys:
    w = eager_sd[k]
    dd = 0 if w.ndim == 2 else 1
    eager.append(w.narrow(dd, 0, w.shape[dd] // 2))
emb = sum(s.numel() * s.element_size() for s in eager) / 1e6
edsts = [torch.empty(s.shape, dtype=s.dtype, device=dev) for s in eager]

torch.cuda.synchronize(); _t0 = time.perf_counter()
for _d, _s in zip(edsts, eager):
    _d.copy_(_s)
torch.cuda.synchronize()
d = time.perf_counter() - _t0
print(f"{'D  from heap memory (eager load)  <- FLAG FIX':<48}"
      f"{d:>9.3f}s{d/len(eager)*1000:>11.2f} ms{emb/d:>11.1f} MB/s")

print()
if b:
    print(f"B (pinned staging) vs A: {a/b:.1f}x" if b < a else
          f"B is {b/a:.1f}x SLOWER than A -- pinned-staging hypothesis refuted")
print(f"D (eager/heap)     vs A: {a/d:.1f}x" if d < a else
      f"D is {d/a:.1f}x SLOWER than A -- mmap-registration hypothesis REFUTED")
print()
print("If D is much faster than A, the fix is --safetensors-load-strategy=eager")
print("and needs no code change at all.")
