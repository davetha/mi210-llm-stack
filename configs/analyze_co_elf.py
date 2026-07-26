#!/usr/bin/env python3
"""Compile reference gfx90a code object and patch MLA .co"""
import struct, subprocess, os, shutil

HIPCC = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc"

# Step 1: Write test kernel
with open("/tmp/test_mla_patch.cu", "w") as f:
    f.write('#include <hip/hip_runtime.h>\n__global__ void test_kernel(float* out) { *out = 1.0f; }\n')

# Step 2: Compile for both architectures
for arch in ["gfx90a:sramecc+:xnack-", "gfx942:sramecc+:xnack-"]:
    out = f"/tmp/test_{arch.split(':')[0]}.o"
    result = subprocess.run(
        [HIPCC, "-fgpu-rdc", f"--offload-arch={arch}", "-c", "/tmp/test_mla_patch.cu", "-o", out],
        capture_output=True, text=True
    )
    print(f"Compile {arch}: exit={result.returncode}")
    if result.stderr:
        print(f"  stderr: {result.stderr[:200]}")

# Step 3: Compare e_flags
print("\n=== e_flags comparison ===")
files = {
    "gfx90a test .o": "/tmp/test_gfx90a.o",
    "gfx942 test .o": "/tmp/test_gfx942.o",
    "gfx942 MLA .co": "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_a16w16_qh16_m16x4_n16x1_coex0_mask1.co",
}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"  {name:25s}: FILE NOT FOUND")
        continue
    with open(path, "rb") as f:
        data = f.read(64)
    flags = struct.unpack_from("<I", data, 48)[0]
    mach = flags & 0xFF
    print(f"  {name:25s}: e_flags=0x{flags:08x}, mach=0x{mach:02x}={mach}")

# Step 4: Patch MLA .co for gfx90a
print("\n=== Patching MLA .co ===")
co_path = files["gfx942 MLA .co"]
ref_90a = files["gfx90a test .o"]

if not os.path.exists(ref_90a):
    print("ERROR: No reference gfx90a .o file")
    exit(1)

with open(co_path, "rb") as f:
    co_data = bytearray(f.read())

with open(ref_90a, "rb") as f:
    ref_data = f.read(64)

orig_flags = struct.unpack_from("<I", co_data, 48)[0]
gfx90a_flags = struct.unpack_from("<I", ref_data, 48)[0]
gfx90a_mach = gfx90a_flags & 0xFF

print(f"  Original .co: e_flags=0x{orig_flags:08x}, mach=0x{orig_flags & 0xFF:02x}")
print(f"  gfx90a ref:   e_flags=0x{gfx90a_flags:08x}, mach=0x{gfx90a_mach:02x}")

# Strategy 1: Just replace the mach field (lower 8 bits)
patched_mach = (orig_flags & ~0xFF) | gfx90a_mach
# Strategy 2: Replace full e_flags with gfx90a's value
patched_full = gfx90a_flags
# Strategy 3: Replace mach and set sramecc/xnack to match gfx90a hardware
patched_hw = gfx90a_mach  # just the mach, clear all feature bits

for desc, flags_val in [("mach_only", patched_mach), ("full", patched_full), ("mach_clear_features", patched_hw)]:
    patched_data = bytearray(co_data)
    struct.pack_into("<I", patched_data, 48, flags_val)
    out_path = f"/tmp/mla_gfx90a_{desc}.co"
    with open(out_path, "wb") as f:
        f.write(patched_data)
    # Verify
    with open(out_path, "rb") as f:
        v = f.read(64)
    vf = struct.unpack_from("<I", v, 48)[0]
    print(f"  {desc:25s}: 0x{vf:08x} -> {out_path}")

# Step 5: Create gfx90a MLA directory and copy patched files
print("\n=== Setting up gfx90a MLA directory ===")
gfx90a_mla_dir = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla"
os.makedirs(gfx90a_mla_dir, exist_ok=True)

# Copy all gfx942 MLA .co files with patched headers
gfx942_mla_dir = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla"
count = 0
for fname in os.listdir(gfx942_mla_dir):
    if not fname.endswith(".co"):
        continue
    src = os.path.join(gfx942_mla_dir, fname)
    dst = os.path.join(gfx90a_mla_dir, fname)
    with open(src, "rb") as f:
        d = bytearray(f.read())
    # Patch: replace mach field
    struct.pack_into("<I", d, 48, patched_mach)
    with open(dst, "wb") as f:
        f.write(d)
    count += 1
print(f"  Copied and patched {count} MLA .co files to {gfx90a_mla_dir}")

# Also check for config CSV files
csv_dir = os.path.join(gfx942_mla_dir)
csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
if csvs:
    print(f"  Found config CSVs: {csvs}")
    for csv in csvs:
        shutil.copy2(os.path.join(gfx942_mla_dir, csv), os.path.join(gfx90a_mla_dir, csv))
        print(f"    Copied {csv}")

print(f"\nDone! gfx90a MLA directory ready at {gfx90a_mla_dir}")
