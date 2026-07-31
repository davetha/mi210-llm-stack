"""Measure whether device-to-device copies between the two MI210s actually peer.

rocm-bandwidth-test is not in the rocm/vllm image, so this substitutes the two
pieces of evidence that matter: what the runtime claims (can_device_access_peer)
and what a timed copy delivers.

The interpretation rests on a threshold, not on the raw number. A peer DMA over
PCIe 4.0 x16 tops out near 32 GB/s. A copy STAGED THROUGH HOST MEMORY does two
transfers over the same link and lands near half that, so a sustained rate above
~20 GB/s is only reachable if the copy is peering. Below that the result is
ambiguous and says so, rather than being reported as a negative.
"""
import torch, time

n = torch.cuda.device_count()
print(f"devices: {n}")
for i in range(n):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
if n < 2:
    raise SystemExit("need 2 devices")

print("\n=== runtime peer capability ===")
for i in range(n):
    for j in range(n):
        if i != j:
            print(f"  can_device_access_peer({i},{j}) = "
                  f"{torch.cuda.can_device_access_peer(i, j)}")

# 512 MiB: large enough that per-launch overhead is noise, small enough to sit
# comfortably beside anything else that might be resident.
nbytes = 512 * 1024 * 1024
a = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
b = torch.empty(nbytes, dtype=torch.uint8, device="cuda:1")

def bw(dst, src, reps=20):
    for _ in range(3):                      # warm up; the first copy pays setup
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dst.copy_(src)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return nbytes * reps / dt / 1e9

print("\n=== measured device-to-device ===")
d2d_01 = bw(b, a)
d2d_10 = bw(a, b)
print(f"  cuda:0 -> cuda:1 : {d2d_01:7.2f} GB/s")
print(f"  cuda:1 -> cuda:0 : {d2d_10:7.2f} GB/s")

# Host round-trip, for scale: this is what an allreduce pays per hop when P2P is
# off. Pinned, so it is the FAST version of the staged path, not a strawman.
h = torch.empty(nbytes, dtype=torch.uint8, device="cpu").pin_memory()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    h.copy_(a)
    b.copy_(h)
torch.cuda.synchronize()
staged = nbytes * 10 / (time.perf_counter() - t0) / 1e9
print(f"  via pinned host  : {staged:7.2f} GB/s  (effective, 2 hops)")

best = max(d2d_01, d2d_10)
print(f"\n=== verdict ===")
if best > 20:
    print(f"  {best:.1f} GB/s exceeds the ~16 GB/s a host-staged copy can reach")
    print("  on this link, so the copy IS peering. P2P is functional.")
elif best > staged * 1.3:
    print(f"  {best:.1f} GB/s beats the staged path ({staged:.1f}) but does not")
    print("  clear the 20 GB/s bar. Consistent with peering through the ACS")
    print("  root-complex redirect. Suggestive, NOT conclusive.")
else:
    print(f"  {best:.1f} GB/s is not meaningfully above the staged path")
    print(f"  ({staged:.1f} GB/s). No evidence of a working peer transport.")
