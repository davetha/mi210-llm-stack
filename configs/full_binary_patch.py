#!/usr/bin/env python3
"""
Full Binary Patcher for gfx942 MLA .co → gfx90a
1. Patch e_flags: mach 0x4c → 0x3f
2. Patch MFMA opcode: D3E1 → D3CD (word0 upper 16 bits)
3. Patch register type: Clear bits 27,28 in word1 (AccVGPR → VGPR)
"""
import struct, os, shutil

CO_SRC = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
CO_DST = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
OBJDUMP = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump"

GFX90A_MACH = 0x3f

with open(CO_SRC, "rb") as f:
    data = bytearray(f.read())

print(f"Source: {len(data)} bytes")

# Step 1: Patch e_flags
orig_flags = struct.unpack_from("<I", data, 48)[0]
new_flags = (orig_flags & ~0xFF) | GFX90A_MACH
struct.pack_into("<I", data, 48, new_flags)
print(f"e_flags: 0x{orig_flags:08x} → 0x{new_flags:08x}")

# Step 2+3: Find .text section and patch MFMA instructions
e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]

text_offset = text_size = 0
for i in range(e_shnum):
    sh = e_shoff + i * e_shentsize
    sh_flags = struct.unpack_from("<Q", data, sh + 8)[0]
    if sh_flags & 0x4:  # SHF_EXECINSTR
        text_offset = struct.unpack_from("<Q", data, sh + 24)[0]
        text_size = struct.unpack_from("<Q", data, sh + 32)[0]
        break

print(f".text: offset=0x{text_offset:x}, size=0x{text_size:x}")

# Patch each MFMA instruction
count = 0
for offset in range(text_offset, text_offset + text_size - 7, 4):
    word0 = struct.unpack_from("<I", data, offset)[0]
    upper16 = (word0 >> 16) & 0xFFFF

    if upper16 == 0xD3E1:  # v_mfma_f32_16x16x16_bf16
        word1 = struct.unpack_from("<I", data, offset + 4)[0]

        # Patch word0: Change opcode D3E1 → D3CD (F16 variant)
        # ALSO clear bit 15 (destination AccVGPR → VGPR flag)
        new_word0 = (word0 & 0x00007FFF) | (0xD3CD << 16)

        # Patch word1: Clear bits 27 and 28 (AccVGPR → VGPR for src0 and src1)
        # These bits control whether src0/src1 are AccVGPR (1) or VGPR (0)
        new_word1 = word1 & ~( (1 << 27) | (1 << 28) )

        if count < 5:
            print(f"  MFMA #{count+1} @0x{offset:x}: "
                  f"w0 0x{word0:08x}→0x{new_word0:08x}, "
                  f"w1 0x{word1:08x}→0x{new_word1:08x} "
                  f"(bits27,28: {(word1>>27)&1},{(word1>>28)&1}→0,0)")

        struct.pack_into("<I", data, offset, new_word0)
        struct.pack_into("<I", data, offset + 4, new_word1)
        count += 1

print(f"\nPatched {count} MFMA instructions (opcode + register type)")

# Write patched file
os.makedirs(os.path.dirname(CO_DST), exist_ok=True)
with open(CO_DST, "wb") as f:
    f.write(data)
print(f"Wrote: {CO_DST}")

# Verify with disassembler
import subprocess
result = subprocess.run([OBJDUMP, "-d", "--mcpu=gfx90a", CO_DST],
                       capture_output=True, text=True)
mfma_lines = [l for l in result.stdout.split("\n") if "mfma" in l.lower()]
print(f"\nDisassembler found {len(mfma_lines)} MFMA instructions:")
for line in mfma_lines[:5]:
    print(f"  {line.strip()}")
if len(mfma_lines) > 5:
    print(f"  ... and {len(mfma_lines)-5} more")

# Verify register types changed (should show v[] not a[])
acc_lines = [l for l in mfma_lines if "a[" in l and "mfma" in l.lower()]
v_lines = [l for l in mfma_lines if "v[" in l and "mfma" in l.lower()]
print(f"\nRegister type check:")
print(f"  MFMA with a[] (AccVGPR): {len(acc_lines)}")
print(f"  MFMA with v[] (VGPR): {len(v_lines)}")
if len(acc_lines) == 0:
    print("  ✅ ALL AccVGPR references converted to VGPR!")
else:
    print(f"  ⚠️ {len(acc_lines)} still have AccVGPR references:")
    for l in acc_lines[:3]:
        print(f"    {l.strip()}")
