#!/usr/bin/env python3
"""Patch MLA .co files with correct gfx90a e_flags (mach=0x3f)"""
import struct, os, shutil

GFX942_MACH = 0x4c
GFX90A_MACH = 0x3f

src_dir = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla"
dst_dir = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla"

os.makedirs(dst_dir, exist_ok=True)

count = 0
for fname in os.listdir(src_dir):
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)

    if fname.endswith(".co"):
        with open(src, "rb") as f:
            data = bytearray(f.read())

        orig_flags = struct.unpack_from("<I", data, 48)[0]
        orig_mach = orig_flags & 0xFF

        # Replace mach: 0x4c (gfx942) -> 0x3f (gfx90a)
        # Keep all feature bits (sramecc, xnack)
        new_flags = (orig_flags & ~0xFF) | GFX90A_MACH

        struct.pack_into("<I", data, 48, new_flags)

        with open(dst, "wb") as f:
            f.write(data)

        verify = struct.unpack_from("<I", data, 48)[0]
        count += 1
        if count <= 3:
            print(f"  {fname}: 0x{orig_flags:08x} (mach=0x{orig_mach:02x}) -> 0x{verify:08x} (mach=0x{GFX90A_MACH:02x})")
    elif fname.endswith(".csv"):
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")

print(f"\nPatched {count} .co files with mach=0x{GFX90A_MACH:02x} (gfx90a)")

# Clear ALL JIT caches for MLA
for cache_dir in [
    "/opt/python/lib/python3.14/site-packages/aiter/jit/build/module_mla_asm",
    "/opt/python/lib/python3.14/site-packages/aiter/jit/build/module_mla_metadata",
    "/opt/python/lib/python3.14/site-packages/aiter/jit/build/module_attention_asm",
]:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared: {cache_dir}")

for cache_so in [
    "/opt/python/lib/python3.14/site-packages/aiter/jit/module_mla_asm.so",
    "/opt/python/lib/python3.14/site-packages/aiter/jit/module_mla_metadata.so",
    "/opt/python/lib/python3.14/site-packages/aiter/jit/module_attention_asm.so",
]:
    if os.path.exists(cache_so):
        os.remove(cache_so)
        print(f"Removed: {cache_so}")

print("\nReady to test!")
