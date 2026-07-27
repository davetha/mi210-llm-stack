#!/usr/bin/env python3
"""
Patch ALL MLA .co files from gfx942 to gfx90a using minimal approach:
1. e_flags: mach 0x4c → 0x3f
2. MFMA opcode: D3E1 → D3CD (only, keep AccVGPR)
3. vgpr_count: 512 → 256 (uint16 msgpack)
"""
import struct, os, shutil

SRC_DIR = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla"
DST_DIR = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla"

os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    src = os.path.join(SRC_DIR, fname)
    dst = os.path.join(DST_DIR, fname)

    if fname.endswith(".csv"):
        shutil.copy2(src, dst)
        print(f"  Copied {fname}")
        continue

    if not fname.endswith(".co"):
        continue

    with open(src, "rb") as f:
        data = bytearray(f.read())

    # Layer 1: e_flags
    orig = struct.unpack_from("<I", data, 48)[0]
    struct.pack_into("<I", data, 48, (orig & ~0xFF) | 0x3f)

    # Layer 2: MFMA opcode only
    text_off = 0x1000
    # Find text section size from ELF
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    text_size = 0
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_flags = struct.unpack_from("<Q", data, sh + 8)[0]
        if sh_flags & 0x4:
            text_off = struct.unpack_from("<Q", data, sh + 24)[0]
            text_size = struct.unpack_from("<Q", data, sh + 32)[0]
            break

    mfma_count = 0
    for off in range(text_off, text_off + text_size - 7, 4):
        w0 = struct.unpack_from("<I", data, off)[0]
        if ((w0 >> 16) & 0xFFFF) == 0xD3E1:
            new_w0 = (w0 & 0x0000FFFF) | (0xD3CD << 16)
            struct.pack_into("<I", data, off, new_w0)
            mfma_count += 1

    # Layer 3: vgpr_count (uint16 msgpack)
    key = b".vgpr_count"
    mk = bytes([0xA0 | len(key)]) + key
    pos = data.find(mk)
    vgpr_old = "?"
    if pos >= 0:
        val_start = pos + len(mk)
        fmt = data[val_start]
        if fmt == 0xCD:
            vgpr_old = (data[val_start+1] << 8) | data[val_start+2]
            data[val_start + 1] = 0x01  # 256 >> 8
            data[val_start + 2] = 0x00  # 256 & 0xFF

    print(f"  {fname}: {mfma_count} MFMA, vgpr {vgpr_old}→256, text=0x{text_size:x}")

    with open(dst, "wb") as f:
        f.write(data)

print(f"\nAll .co files patched and copied to {DST_DIR}")
