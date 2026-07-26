#!/usr/bin/env python3
"""Analyze fmoe_b16.co to find all opcodes and identify unsupported ones."""
import struct, os, collections

HSA = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa"
SRC = os.path.join(HSA, "gfx942", "fmoe_b16.co")
DST = os.path.join(HSA, "gfx90a", "fmoe_b16.co")

# Known gfx90a MFMA opcodes (from AMD GCN ISA manual)
GFX90A_MFMA = {
    0xD340: "v_mfma_f32_4x4x4f32",
    0xD341: "v_mfma_f32_4x4x4bf16",  # actually this might not exist on 90a
    0xD342: "v_mfma_f32_4x4x4f16",
    0xD343: "v_mfma_f32_4x4x4bf16_90a",
    0xD3C0: "v_mfma_f32_16x16x4f16",
    0xD3C2: "v_mfma_f32_4x4x4f16",
    0xD3C3: "v_mfma_f32_4x4x4bf16_90a",
    0xD3CB: "v_mfma_f32_16x16x8xf16",
    0xD3CD: "v_mfma_f32_16x16x16f16",
    0xD3D0: "v_mfma_i32_32x32x4i8",
    0xD3D8: "v_mfma_i32_16x16x16i8",
    0xD3E5: "v_mfma_f32_16x16x8xf32",
    0xD3E7: "v_mfma_f32_32x32x1f32",
    0xD3E8: "v_mfma_f32_32x32x2f32",
    0xD3EC: "v_mfma_f32_32x32x4bf16",
    0xD344: "v_mfma_f32_4x4x2f16",
    0xD348: "v_mfma_f32_4x4x1f32",
    0xD360: "v_mfma_f32_16x16x8xbf16",  # not sure
    0xD368: "v_mfma_f32_16x16x16bf16_90a",  # might exist
}

# Known gfx942+ only (CDNA3)
GFX942_PLUS = {
    0xD3E1: "v_mfma_f32_16x16x16_bf16",      # gfx940+
    0xD3E3: "v_mfma_f32_16x16x8xbf16",        # unsure
    0xD3E4: "v_mfma_f32_16x16x4xbf16_942",    # unsure
    0xD3EB: "v_mfma_f32_16x16x32_bf16",       # gfx950+
    0xD3F0: "v_mfma_f32_16x16x32_fp8",        # gfx940+ FP8
    0xD3F2: "v_mfma_f32_32x32x4_fp8",         # gfx940+ FP8
    0xD3F8: "v_mfma_f32_16x16x32_fp8_950",    # gfx950
    0xD3FA: "v_mfma_f32_32x32x4_fp8_950",     # gfx950
}

# Other GCN opcodes (sample - VOP3P has many)
KNOWN_OTHER = {
    0xD140: "v_mov_b32",
    0xD3D8: "v_accvgpr_write",
    0xD3D9: "v_accvgpr_read",
    0xDBFE: "ds_read_b128",
    0xD9FE: "ds_read_b128_alt",
    0xDBFF: "ds_write_b128",
    0xD9FF: "ds_write_b128_alt",
}


def find_text(data):
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_flags = struct.unpack_from("<Q", data, sh + 8)[0]
        if sh_flags & 0x4:
            text_off = struct.unpack_from("<Q", data, sh + 24)[0]
            text_size = struct.unpack_from("<Q", data, sh + 32)[0]
            return text_off, text_size
    return 0x1000, 0


for label, path in [("gfx942 (ORIGINAL)", SRC), ("gfx90a (PATCHED)", DST)]:
    if not os.path.exists(path):
        print(f"\n{label}: MISSING")
        continue
    with open(path, "rb") as f:
        data = f.read()
    text_off, text_size = find_text(data)
    print(f"\n=== {label} ===")
    print(f"  file: {path}")
    print(f"  text: 0x{text_size:x} bytes ({text_size} bytes)")

    # Scan every 4-byte word's upper 16 bits
    opcode_counts = collections.Counter()
    suspicious_opcodes = collections.Counter()  # not in any known set

    for off in range(text_off, text_off + text_size - 3, 4):
        w0 = struct.unpack_from("<I", data, off)[0]
        upper = (w0 >> 16) & 0xFFFF
        opcode_counts[upper] += 1

    # Categorize
    gfx90a_native = 0
    gfx942_only = 0
    known_other = 0
    unknown = collections.Counter()

    for opcode, count in sorted(opcode_counts.items()):
        if opcode in GFX90A_MFMA:
            gfx90a_native += count
        elif opcode in GFX942_PLUS:
            gfx942_only += count
            print(f"  GFX942+ ONLY: 0x{opcode:04X} ({GFX942_PLUS[opcode]}) x{count}")
        elif opcode in KNOWN_OTHER:
            known_other += count
        elif opcode >= 0xD000 and opcode < 0xE000:
            # VOP3P format but unknown - could be MFMA variant
            unknown[opcode] += count
        # else: not VOP3P, ignore (could be SOP, VOP1, etc)

    print(f"\n  Summary:")
    print(f"    gfx90a native MFMA ops: {gfx90a_native}")
    print(f"    gfx942+ only ops:       {gfx942_only}  <-- THESE CAUSE ILLEGAL_INSTR")
    print(f"    known other ops:        {known_other}")
    print(f"    unknown VOP3P:          {sum(unknown.values())}")
    if unknown:
        print(f"    unknown VOP3P opcodes (top 20):")
        for op, cnt in unknown.most_common(20):
            print(f"      0x{op:04X}: {cnt}")
