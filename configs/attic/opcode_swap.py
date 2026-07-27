#!/usr/bin/env python3
"""
Targeted opcode replacement: D3E1 (BF16 MFMA) → D3CD (F16 MFMA)
in the gfx942 MLA .co file. Keep ALL other bytes identical.
"""
import struct, os, shutil

CO_SRC = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
CO_DST = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
OBJDUMP = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/llvm-objdump"

GFX942_MACH = 0x4c
GFX90A_MACH = 0x3f

# Read the original .co
with open(CO_SRC, "rb") as f:
    data = bytearray(f.read())

print(f"Original .co: {len(data)} bytes")

# Step 1: Patch e_flags (mach field)
orig_flags = struct.unpack_from("<I", data, 48)[0]
new_flags = (orig_flags & ~0xFF) | GFX90A_MACH
struct.pack_into("<I", data, 48, new_flags)
print(f"e_flags: 0x{orig_flags:08x} → 0x{new_flags:08x}")

# Step 2: Find and replace MFMA opcodes
# The MFMA instruction encoding starts with a specific opcode word.
# From disassembly: v_mfma_f32_16x16x16_bf16 uses opcode bytes starting with D3E1
# v_mfma_f32_16x16x16f16 uses D3CD
#
# GCN MFMA instructions are 12 bytes (3 × 32-bit words):
# Word 0: [opcode bits | dest reg | modifiers]
# Word 1: [src0 | src1 | src2 | modifiers]  
# Word 2: [more modifiers]
#
# The opcode is encoded in the upper bits of word 0.
# For v_mfma_f32_16x16x16_bf16: word 0 starts with 0xD3E1xxxx
# For v_mfma_f32_16x16x16f16:  word 0 starts with 0xD3CDxxxx

# Scan the .text section for MFMA instructions
# The .text section offset and size can be found from the ELF section headers
e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]

text_offset = 0
text_size = 0
for i in range(e_shnum):
    sh_offset = e_shoff + i * e_shentsize
    sh_type = struct.unpack_from("<I", data, sh_offset + 4)[0]
    sh_off = struct.unpack_from("<Q", data, sh_offset + 24)[0]
    sh_size = struct.unpack_from("<Q", data, sh_offset + 32)[0]
    sh_flags = struct.unpack_from("<Q", data, sh_offset + 8)[0]
    if sh_flags & 0x4:  # SHF_EXECINSTR
        text_offset = sh_off
        text_size = sh_size
        print(f".text section: offset=0x{sh_off:x}, size=0x{sh_size:x}")
        break

# Scan for MFMA opcodes
# Each instruction is 12 bytes. We scan word-aligned.
# The opcode pattern for v_mfma_f32_16x16x16_bf16 is 0xD3E1 in the upper 16 bits
# of the first word (little-endian).

count_found = 0
count_replaced = 0
replacements = []

for offset in range(text_offset, text_offset + text_size - 11, 4):
    word0 = struct.unpack_from("<I", data, offset)[0]
    
    # Check if this is v_mfma_f32_16x16x16_bf16
    # The opcode is in the upper 16 bits when stored as little-endian
    # Actually, GCN uses 32-bit instruction words. The opcode field varies.
    # From disassembly: D3E10020 is the encoding
    # So word0 = 0xD3E10020 or similar (upper 16 bits = 0xD3E1)
    upper16 = (word0 >> 16) & 0xFFFF
    
    if upper16 == 0xD3E1:
        count_found += 1
        word1 = struct.unpack_from("<I", data, offset + 4)[0]
        
        # Replace opcode: D3E1 → D3CD
        new_word0 = (word0 & 0x0000FFFF) | (0xD3CD << 16)
        
        old_bytes = struct.pack("<I", word0) + struct.pack("<I", word1)
        new_word0_bytes = struct.pack("<I", new_word0)
        
        # Show first few replacements
        if count_found <= 5:
            print(f"  MFMA #{count_found} at offset 0x{offset:x}: "
                  f"word0=0x{word0:08x} → 0x{new_word0:08x}, word1=0x{word1:08x}")
        
        replacements.append((offset, word0, new_word0))
        
        # Apply the patch
        struct.pack_into("<I", data, offset, new_word0)
        count_replaced += 1

print(f"\nFound {count_found} MFMA instructions, replaced {count_replaced} opcodes")

# Write patched file
os.makedirs(os.path.dirname(CO_DST), exist_ok=True)
with open(CO_DST, "wb") as f:
    f.write(data)
print(f"Wrote patched .co to {CO_DST}")

# Verify by disassembling with gfx90a
print(f"\n=== Verify: disassemble patched .co as gfx90a ===")
import subprocess
result = subprocess.run(
    [OBJDUMP, "-d", "--mcpu=gfx90a", CO_DST],
    capture_output=True, text=True
)
# Show MFMA instructions
mfma_lines = [l for l in result.stdout.split("\n") if "mfma" in l.lower()]
print(f"MFMA instructions found in patched binary: {len(mfma_lines)}")
for line in mfma_lines[:5]:
    print(f"  {line.strip()}")
if len(mfma_lines) > 5:
    print(f"  ... and {len(mfma_lines) - 5} more")

# Show any errors
if result.stderr:
    errors = [l for l in result.stderr.split("\n") if "error" in l.lower() or "warning" in l.lower()]
    for e in errors[:5]:
        print(f"  {e.strip()}")
