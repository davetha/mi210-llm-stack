#!/usr/bin/env python3
"""
Analyze register collisions in the patched MLA kernel.
On gfx942, VGPR v[N] and AccVGPR a[N] are SEPARATE registers.
After patching both to VGPR, if original v[100] and a[100] exist,
they collide as v[100].
"""
import struct

CO = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa/gfx942/mla/mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"

with open(CO, "rb") as f:
    data = f.read()

text_off, text_size = 0x1000, 0xc16c

# Track VGPR and AccVGPR usage
vgpr_used = set()      # VGPR registers used by non-AccVGPR instructions
accvgpr_used = set()   # AccVGPR registers used by MFMA/accvgpr instructions

for off in range(text_off, text_off + text_size - 7, 4):
    w0 = struct.unpack_from("<I", data, off)[0]
    w1 = struct.unpack_from("<I", data, off + 4)[0]
    upper16 = (w0 >> 16) & 0xFFFF

    # MFMA instructions: extract AccVGPR source operands
    if upper16 == 0xD3E1:
        vdst = w0 & 0xFF
        src0_acc = (w1 >> 27) & 1
        src1_acc = (w1 >> 28) & 1
        src0 = w1 & 0xFF
        src1 = (w1 >> 9) & 0xFF
        dst_acc = (w0 >> 15) & 1

        if src0_acc:
            accvgpr_used.update(range(src0, src0+2))
        else:
            vgpr_used.update(range(src0, src0+2))

        if src1_acc:
            accvgpr_used.update(range(src1, src1+2))
        else:
            vgpr_used.update(range(src1, src1+2))

        if dst_acc:
            accvgpr_used.update(range(vdst, vdst+4))
        else:
            vgpr_used.update(range(vdst, vdst+4))

    # v_accvgpr_write_b32: D3D940xx
    elif upper16 & 0xFFFF00 == 0xD3D900 or (w0 >> 24 == 0xD3 and (w0 >> 16) & 0xFF == 0xD9):
        vdst = w0 & 0xFF
        accvgpr_used.add(vdst)

    # v_accvgpr_read_b32: D3D840xx
    elif w0 >> 24 == 0xD3 and (w0 >> 16) & 0xFF == 0xD8:
        src = w1 & 0xFF
        accvgpr_used.add(src)

    # ds_read_b128 a[...]: DBFE
    elif upper16 == 0xDBFE:
        dst = (w1 >> 8) & 0xFF
        accvgpr_used.update(range(dst, dst+4))

    # ds_read_b128 v[...]: D9FE
    elif upper16 == 0xD9FE:
        dst = (w1 >> 8) & 0xFF
        vgpr_used.update(range(dst, dst+4))

    # v_mov_b32 and other VOP1/VOP3 to VGPR
    # (simplified: just track from known patterns)
    elif (w0 >> 24) == 0x7E:  # VOP1 e32 format
        vdst = (w0 >> 17) & 0x7F
        vgpr_used.add(vdst)

print(f"VGPR registers used (non-AccVGPR): {sorted(vgpr_used)[:20]}... max={max(vgpr_used) if vgpr_used else 0}")
print(f"AccVGPR registers used: {sorted(accvgpr_used)[:20]}... max={max(accvgpr_used) if accvgpr_used else 0}")

# Check for collisions
collisions = vgpr_used & accvgpr_used
print(f"\nRegister COLLISIONS: {len(collisions)} registers")
if collisions:
    print(f"  Colliding registers: {sorted(collisions)[:30]}...")
    print(f"\n  These registers are used by BOTH VGPR and AccVGPR instructions.")
    print(f"  After patching to all-VGPR, the AccVGPR data OVERWRITES VGPR data!")
    print(f"  This is the likely cause of the memory fault.")
else:
    print("  No collisions found!")
    print(f"  VGPR range: {min(vgpr_used)}-{max(vgpr_used)}")
    print(f"  AccVGPR range: {min(accvgpr_used)}-{max(accvgpr_used)}")
    if max(vgpr_used) < min(accvgpr_used):
        print("  Ranges don't overlap - safe to merge!")
