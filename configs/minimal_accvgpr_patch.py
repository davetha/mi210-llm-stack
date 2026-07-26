#!/usr/bin/env python3
"""
Minimal patch: ONLY opcode swap + e_flags + vgpr_count fix.
KEEP all AccVGPR operands/data flow (gfx90a supports AccVGPR natively).
gfx90a has 512 total registers (256 VGPR + 256 AccVGPR) same as gfx942.
The vgpr_count=512 on gfx942 means "total register file".
On gfx90a, vgpr_count means "VGPR only" (max 256).
The runtime derives AccVGPR count = 512 - vgpr_count.
"""
import struct, os

CO_SRC = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
CO_DST = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"

with open(CO_SRC, "rb") as f:
    data = bytearray(f.read())

# Layer 1: e_flags
orig = struct.unpack_from("<I", data, 48)[0]
struct.pack_into("<I", data, 48, (orig & ~0xFF) | 0x3f)
print(f"e_flags: 0x{orig:08x} → 0x{(orig & ~0xFF) | 0x3f:08x}")

# Layer 2: MFMA opcode ONLY (D3E1→D3CD), KEEP AccVGPR operands
text_off, text_size = 0x1000, 0xc16c
count = 0
for off in range(text_off, text_off + text_size - 7, 4):
    w0 = struct.unpack_from("<I", data, off)[0]
    if ((w0 >> 16) & 0xFFFF) == 0xD3E1:
        new_w0 = (w0 & 0x0000FFFF) | (0xD3CD << 16)
        struct.pack_into("<I", data, off, new_w0)
        count += 1
print(f"Patched {count} MFMA opcodes (AccVGPR operands PRESERVED)")

# Layer 3: vgpr_count 512→256 (correct msgpack uint16)
key = b".vgpr_count"
mk = bytes([0xA0 | len(key)]) + key
pos = data.find(mk)
if pos >= 0:
    val_start = pos + len(mk)
    fmt = data[val_start]
    if fmt == 0xCD:  # uint16
        old = (data[val_start+1] << 8) | data[val_start+2]
        data[val_start + 1] = 0  # 256 >> 8 = 1? No: 256 = 0x0100
        data[val_start + 2] = 0  # Wait, 256 in big-endian uint16 = 0x0100
        # 256 = 0x0100: high byte=0x01, low byte=0x00
        data[val_start + 1] = 0x01
        data[val_start + 2] = 0x00
        print(f"vgpr_count: {old} → 256 (uint16)")
    else:
        print(f"WARNING: vgpr_count format byte 0x{fmt:02x}, expected 0xCD")

os.makedirs(os.path.dirname(CO_DST), exist_ok=True)
import os
with open(CO_DST, "wb") as f:
    f.write(data)
print(f"Wrote: {CO_DST}")
print("Strategy: Keep AccVGPR split (gfx90a supports it), just swap MFMA opcode")
