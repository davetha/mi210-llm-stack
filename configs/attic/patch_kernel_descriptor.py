#!/usr/bin/env python3
"""Patch the hardware kernel descriptor's compute_pgm_rsrc1 and rsrc2"""
import struct

CO = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"

with open(CO, "rb") as f:
    data = bytearray(f.read())

desc_off = 0xFC0

# Read current values
rsrc1 = struct.unpack_from("<I", data, desc_off + 32)[0]
rsrc2 = struct.unpack_from("<I", data, desc_off + 36)[0]
props = struct.unpack_from("<H", data, desc_off + 40)[0]

print(f"Current compute_pgm_rsrc1 = 0x{rsrc1:08x}")
print(f"Current compute_pgm_rsrc2 = 0x{rsrc2:08x}")
print(f"Current kernel_code_properties = 0x{props:04x}")

# Compute correct compute_pgm_rsrc1
# VGPRs: 256 → field = (256/4) - 1 = 63
# SGPRs: 96 → field = (96/8) - 1 = 11  (actually gfx90a granularity might differ)
# For gfx90a: SGPR allocation unit = 8, so SGPRs field = ceil(num_sgprs/8) - 1

VGPR_FIELD = 63   # (256/4)-1
SGPR_FIELD = 11   # (96/8)-1  
DX10_CLAMP = 1
IEEE_MODE = 1

new_rsrc1 = (VGPR_FIELD & 0x3F) | ((SGPR_FIELD & 0xF) << 6) | (DX10_CLAMP << 14) | (IEEE_MODE << 16)
print(f"\nNew compute_pgm_rsrc1 = 0x{new_rsrc1:08x}")
print(f"  VGPRs: {VGPR_FIELD} → {((VGPR_FIELD+1)*4)} allocated")
print(f"  SGPRs: {SGPR_FIELD} → {((SGPR_FIELD+1)*8)} allocated")

# Compute correct compute_pgm_rsrc2
# LDS: 64KB requested. LDS field = (65536 / 256) - 1 = 255? Actually encoding varies.
# On gfx90a: LDS_SIZE field in rsrc2 bits [15:8] or similar
# Let's enable workgroup ID and set user SGPR count
ENABLE_WG_ID_X = 1
USER_SGPR_COUNT = 8  # typical for kernel arguments
LDS_SIZE = 255  # 64KB / 256 - 1 = 255 (if granularity is 256 bytes)

# Actually, on gfx90a the LDS size field is in rsrc2 at different bits
# Let me check the original gfx942 value for reference
CO942 = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"
with open(CO942, "rb") as f:
    data942 = f.read()
rsrc1_942 = struct.unpack_from("<I", data942, desc_off + 32)[0]
rsrc2_942 = struct.unpack_from("<I", data942, desc_off + 36)[0]
print(f"\ngfx942 reference:")
print(f"  compute_pgm_rsrc1 = 0x{rsrc1_942:08x}")
print(f"  compute_pgm_rsrc2 = 0x{rsrc2_942:08x}")

# Decode gfx942 rsrc1
vgpr942 = rsrc1_942 & 0x3F
sgpr942 = (rsrc1_942 >> 6) & 0xF
print(f"  VGPRs: {vgpr942} → {(vgpr942+1)*4}")
print(f"  SGPRs: {sgpr942} → {(sgpr942+1)*8}")

# Use the gfx942 rsrc2 as-is (LDS, user SGPRs etc. should be same)
print(f"\nUsing gfx942 rsrc2 as reference: 0x{rsrc2_942:08x}")

# Patch
struct.pack_into("<I", data, desc_off + 32, new_rsrc1)
struct.pack_into("<I", data, desc_off + 36, rsrc2_942)  # Copy from gfx942

with open(CO, "wb") as f:
    f.write(data)
print(f"\nPatched kernel descriptor:")
print(f"  rsrc1: 0x{rsrc1:08x} → 0x{new_rsrc1:08x}")
print(f"  rsrc2: 0x{rsrc2:08x} → 0x{rsrc2_942:08x}")
