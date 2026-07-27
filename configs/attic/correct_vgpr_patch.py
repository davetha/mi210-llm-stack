#!/usr/bin/env python3
"""Correctly read and patch vgpr_count (msgpack uint16 encoded)"""
import struct

CO = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx90a/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"

with open(CO, "rb") as f:
    data = bytearray(f.read())

key = b".vgpr_count"
mk = bytes([0xA0 | len(key)]) + key
pos = data.find(mk)
if pos < 0:
    print("Key not found!")
    exit(1)

val_start = pos + len(mk)
fmt_byte = data[val_start]
print(f"Format byte at 0x{val_start:x}: 0x{fmt_byte:02x}")

if fmt_byte == 0xCD:
    # uint16: next 2 bytes are big-endian value
    actual_val = (data[val_start + 1] << 8) | data[val_start + 2]
    print(f"msgpack uint16: vgpr_count = {actual_val}")
    print(f"  Raw bytes: {data[val_start:val_start+3].hex()}")

    # Patch to 256 (enough for all VGPR references including v255)
    new_val = 256
    data[val_start + 1] = (new_val >> 8) & 0xFF
    data[val_start + 2] = new_val & 0xFF
    print(f"  Patched to {new_val}")

elif fmt_byte <= 0x7F:
    # positive fixint
    print(f"positive fixint: vgpr_count = {fmt_byte}")
    # Need to change to uint16 format (0xCD + 2 bytes)
    # But this adds 2 bytes, shifting everything...
    # Better: use uint8 (0xCC + 1 byte) if value <= 255
    new_val = 255  # max VGPR
    data[val_start] = 0xCC  # uint8 marker
    # But we need to INSERT a byte, not replace...
    # Since original is 1 byte and uint8 is 2 bytes, we'd shift by 1
    # This would corrupt the .note section
    print("ERROR: Cannot expand fixint to uint8 without corrupting section")
    print("Using original value instead")
else:
    print(f"Unknown format: 0x{fmt_byte:02x}")

# Verify
fmt_byte = data[val_start]
if fmt_byte == 0xCD:
    actual_val = (data[val_start + 1] << 8) | data[val_start + 2]
    print(f"Verified: vgpr_count = {actual_val}")

with open(CO, "wb") as f:
    f.write(data)
print("Done!")
