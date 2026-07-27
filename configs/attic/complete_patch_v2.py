#!/usr/bin/env python3
"""
COMPLETE Binary Patcher: gfx942 MLA .co → gfx90a
ALL layers:
1. e_flags: mach 0x4c→0x3f
2. MFMA opcode: D3E1→D3CD
3. MFMA src reg type: clear bits 27,28 (AccVGPR→VGPR)
4. MFMA dst reg type: clear bit 15 (AccVGPR→VGPR)
5. ds_read_b128: DBFE→D9FE (AccVGPR→VGPR destination)
6. v_accvgpr_write/read: D3D840/D3D940→D14100, word1 18xx→00xx
7. VGPR count: 205→255
"""
import struct, os

CO_SRC = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
CO_DST = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"

with open(CO_SRC, "rb") as f:
    data = bytearray(f.read())

stats = {"mfma": 0, "ds_read_acc": 0, "accvgpr_write": 0, "accvgpr_read": 0, "vgpr_note": 0}

# Layer 1: e_flags
orig = struct.unpack_from("<I", data, 48)[0]
struct.pack_into("<I", data, 48, (orig & ~0xFF) | 0x3f)

text_off, text_size = 0x1000, 0xc16c

for off in range(text_off, text_off + text_size - 7, 4):
    w0 = struct.unpack_from("<I", data, off)[0]
    w1 = struct.unpack_from("<I", data, off + 4)[0]
    upper16 = (w0 >> 16) & 0xFFFF

    # Layer 2-4: MFMA instructions
    if upper16 == 0xD3E1:
        new_w0 = (w0 & 0x00007FFF) | (0xD3CD << 16)
        new_w1 = w1 & ~((1 << 27) | (1 << 28))
        struct.pack_into("<I", data, off, new_w0)
        struct.pack_into("<I", data, off + 4, new_w1)
        stats["mfma"] += 1
        continue

    # Layer 5: ds_read_b128 with AccVGPR destination (DBFE→D9FE)
    # Pattern: word0 has DBFE in upper 16 bits
    if upper16 == 0xDBFE:
        new_w0 = (w0 & 0x0000FFFF) | (0xD9FE << 16)
        struct.pack_into("<I", data, off, new_w0)
        stats["ds_read_acc"] += 1
        continue

    # Layer 6: v_accvgpr_write_b32 (D3D940xx) and v_accvgpr_read_b32 (D3D840xx)
    b3 = (w0 >> 24) & 0xFF
    b2 = (w0 >> 16) & 0xFF

    if b3 == 0xD3 and b2 in (0xD8, 0xD9):
        vdst = w0 & 0xFF
        new_w0 = 0xD1410000 | vdst
        new_w1 = w1 & 0x00FFFFFF
        struct.pack_into("<I", data, off, new_w0)
        struct.pack_into("<I", data, off + 4, new_w1)
        if b2 == 0xD9:
            stats["accvgpr_write"] += 1
        else:
            stats["accvgpr_read"] += 1

# Layer 7: VGPR count — handled by correct_vgpr.py (msgpack uint16 encoding)

os.makedirs(os.path.dirname(CO_DST), exist_ok=True)
with open(CO_DST, "wb") as f:
    f.write(data)

print(f"Patches applied:")
for k, v in stats.items():
    print(f"  {k}: {v}")
print(f"Wrote: {CO_DST}")
