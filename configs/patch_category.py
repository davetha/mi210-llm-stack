#!/usr/bin/env python3
"""
Generalized AITER .co patcher: patches any category of gfx942 ASM kernels to gfx90a.

Usage:
    python patch_category.py <category>           # patch all .co in category
    python patch_category.py <category> --dry-run # inspect without writing
    python patch_category.py <category> --inspect # only analyze, no patching

Categories: mla, fmoe, fmoe_2stages, topksoftmax, pa, fmha_v3_fwd, bf16gemm,
            topk_per_row_decode, topk_per_row_prefill

Applies the proven 3-layer patch:
1. ELF e_flags: mach 0x4c (gfx942) -> 0x3f (gfx90a)
2. MFMA opcode: D3E1 (v_mfma_f32_16x16x16_bf16) -> D3CD (v_mfma_f32_16x16x16f16)
3. vgpr_count: 512 -> 256 (msgpack uint16)

Also reports additional opcode statistics so we can detect categories that use
instructions beyond the standard MFMA swap (e.g., FP8 ops on i8gemm/fp8gemm).
"""
import struct, os, shutil, sys, collections

HSA_ROOT = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa"
SRC_TEMPLATE = os.path.join(HSA_ROOT, "gfx942", "{category}")
DST_TEMPLATE = os.path.join(HSA_ROOT, "gfx90a", "{category}")

# Opcode upper-16-bit patterns we care about (VOP3P word0 high bits)
OPCODES = {
    0xD3E1: "v_mfma_f32_16x16x16_bf16  (gfx940+)",     # -> D3CD
    0xD3CD: "v_mfma_f32_16x16x16f16    (gfx90a native)",
    0xD3EC: "v_mfma_f32_32x32x4bf16    (gfx90a native)",
    0xD3E2: "v_mfma_f32_16x16x16f16    (gfx90a, alt)",
    0xD3E0: "v_mfma_f32_16x16x16f16    (gfx90a, alt2)",
    0xD3D8: "v_mfma_i32_16x16x16i8     (gfx90a)",
    0xD3E5: "v_mfma_f32_16x16x8xf32    (gfx90a)",
    0xD3E7: "v_mfma_f32_32x32x1f32     (gfx90a)",
    # FP8 / MXFP opcodes - gfx942+ only, CANNOT be patched
    0xD3EB: "v_mfma_f32_16x16x32_bf16  (gfx950 only)",
    0xD3F0: "v_mfma_f32_16x16x32_fp8   (gfx940+ FP8)",
    0xD3F2: "v_mfma_f32_32x32x4_fp8    (gfx940+ FP8)",
    0xD3F8: "v_mfma_f32_16x16x32_fp8   (gfx950 FP8)",
    0xD3FA: "v_mfma_f32_32x32x4_fp8    (gfx950 FP8)",
    0xD340: "v_mfma_f32_16x16x4f32     (gfx90a)",
    0xD342: "v_mfma_f32_4x4x4f32       (gfx90a)",
    0xD343: "v_mfma_f32_4x4x4bf16      (gfx90a)",
    0xD3C0: "v_mfma_f32_16x16x4f16     (gfx90a)",
    0xD3C2: "v_mfma_f32_4x4x4f16       (gfx90a)",
    0xD3CB: "v_mfma_f32_16x16x8xf16    (gfx90a)",
    0xD3C3: "v_mfma_f32_4x4x4bf16      (gfx90a)",
    # Other interesting VOP3P opcodes (sample)
    0xD140: "v_mov_b32                  (VOP3P)",
    0xD3D8: "v_accvgpr_write            (AccVGPR write)",
    0xD3D9: "v_accvgpr_read             (AccVGPR read)",
    0xDBFE: "ds_read_b128               (LDS load)",
    0xD9FE: "ds_read_b128_alt           (gfx90a LDS variant)",
}

# Opcodes that are UNSUPPORTED on gfx90a and CANNOT be patched
UNSUPPORTED = {
    0xD3EB, 0xD3F0, 0xD3F2, 0xD3F8, 0xD3FA,  # FP8/MXFP family
}

def find_text_section(data):
    """Find .text section offset and size from ELF section headers."""
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shnum = struct.unpack_from("<H", data, 0x3C)[0]
    e_shentsize = struct.unpack_from("<H", data, 0x3A)[0]
    for i in range(e_shnum):
        sh = e_shoff + i * e_shentsize
        sh_flags = struct.unpack_from("<Q", data, sh + 8)[0]
        if sh_flags & 0x4:  # SHF_EXECINSTRUCTIONS
            text_off = struct.unpack_from("<Q", data, sh + 24)[0]
            text_size = struct.unpack_from("<Q", data, sh + 32)[0]
            return text_off, text_size
    return 0x1000, 0  # fallback


def patch_co_file(data):
    """Apply 3-layer patch to a .co file's bytes. Returns (mfma_count, unsupported_hits, opcode_hist)."""
    # Layer 1: e_flags
    orig = struct.unpack_from("<I", data, 48)[0]
    struct.pack_into("<I", data, 48, (orig & ~0xFF) | 0x3f)

    # Layer 2: MFMA opcode swap
    text_off, text_size = find_text_section(data)
    mfma_count = 0
    unsupported_hits = collections.Counter()
    opcode_hist = collections.Counter()

    for off in range(text_off, text_off + text_size - 3, 4):
        w0 = struct.unpack_from("<I", data, off)[0]
        opcode = (w0 >> 16) & 0xFFFF
        if opcode in (0xD3E1, 0xD3F0):  # BF16 16x16x16 -> F16 16x16x16
            new_w0 = (w0 & 0x0000FFFF) | (0xD3CD << 16)
            struct.pack_into("<I", data, off, new_w0)
            mfma_count += 1
        if opcode in UNSUPPORTED:
            unsupported_hits[opcode] += 1
        if opcode in OPCODES:
            opcode_hist[opcode] += 1

    # Layer 3: vgpr_count (msgpack uint16)
    key = b".vgpr_count"
    mk = bytes([0xA0 | len(key)]) + key
    pos = data.find(mk)
    vgpr_old = None
    if pos >= 0:
        val_start = pos + len(mk)
        fmt = data[val_start]
        if fmt == 0xCD:
            vgpr_old = (data[val_start + 1] << 8) | data[val_start + 2]
            if vgpr_old > 256:
                data[val_start + 1] = 0x01  # 256 >> 8
                data[val_start + 2] = 0x00  # 256 & 0xFF
        elif fmt == 0xCE:  # uint32
            vgpr_old = struct.unpack_from(">I", data, val_start + 1)[0]

    return mfma_count, unsupported_hits, opcode_hist, vgpr_old, (text_off, text_size)


def gather_files(src_dir):
    """Walk src_dir recursively, returning list of (relative_path, full_path) for .co files.
    Also returns sets of csv and other files."""
    co_files = []
    csv_files = []
    other_files = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, src_dir)
            if f.endswith(".co"):
                co_files.append((rel, full))
            elif f.endswith(".csv"):
                csv_files.append((rel, full))
            else:
                other_files.append((rel, full))
    return sorted(co_files), sorted(csv_files), sorted(other_files)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable categories: {sorted(os.listdir(os.path.join(HSA_ROOT, 'gfx942')))}")
        sys.exit(1)

    category = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    inspect_only = "--inspect" in sys.argv

    src_dir = SRC_TEMPLATE.format(category=category)
    dst_dir = DST_TEMPLATE.format(category=category)

    if not os.path.isdir(src_dir):
        print(f"ERROR: source category not found: {src_dir}")
        print(f"Available: {sorted(os.listdir(os.path.join(HSA_ROOT, 'gfx942')))}")
        sys.exit(1)

    if not inspect_only:
        os.makedirs(dst_dir, exist_ok=True)

    print(f"=== Category: {category} ===")
    print(f"Source: {src_dir}")
    print(f"Dest:   {dst_dir}{'(DRY RUN)' if dry_run else ''}{'(INSPECT ONLY)' if inspect_only else ''}")
    print()

    co_files, csv_files, other = gather_files(src_dir)

    print(f"Files: {len(co_files)} .co, {len(csv_files)} .csv, {len(other)} other (recursive)")
    if other:
        subdirs = sorted(set(os.path.dirname(r) for r, _ in other))
        print(f"  Other locations: {subdirs}")
    print()

    # Aggregate stats across all kernels
    total_mfma = 0
    total_unsupported = collections.Counter()
    total_opcodes = collections.Counter()
    kernel_results = []

    for rel_path, src in co_files:
        with open(src, "rb") as f:
            data = bytearray(f.read())

        # Work on a copy
        work = bytearray(data) if not inspect_only else data
        mfma, unsupported, opcodes, vgpr_old, (text_off, text_size) = patch_co_file(work)

        total_mfma += mfma
        total_unsupported.update(unsupported)
        total_opcodes.update(opcodes)
        kernel_results.append((rel_path, mfma, vgpr_old, text_size, dict(unsupported)))

        flag = ""
        if unsupported:
            flag = " [UNPATCHABLE: " + ", ".join(f"{OPCODES.get(o, hex(o))}={c}" for o, c in sorted(unsupported.items())) + "]"

        print(f"  {rel_path}: mfma={mfma}, vgpr={vgpr_old}->{256 if vgpr_old and vgpr_old>256 else vgpr_old}, text=0x{text_size:x}{flag}")

        if not inspect_only and not dry_run and not unsupported:
            dst = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(work)

    # Copy CSVs unmodified (kernel selection tables)
    if not inspect_only and not dry_run:
        for rel_path, src in csv_files:
            dst = os.path.join(dst_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print()
    print(f"=== Totals ===")
    print(f"  Total MFMA opcodes swapped: {total_mfma}")
    print(f"  Total unsupported opcodes:  {sum(total_unsupported.values())}")
    if total_unsupported:
        print(f"  UNSUPPORTED detail:")
        for op, count in sorted(total_unsupported.items()):
            print(f"    {OPCODES.get(op, hex(op))}: {count}")
    print()
    print(f"  Opcode histogram (top 10):")
    for op, count in total_opcodes.most_common(10):
        print(f"    {hex(op):>8} {OPCODES.get(op, '?'):50s} : {count}")

    patchable = sum(1 for r in kernel_results if not r[4])
    unpatchable = len(kernel_results) - patchable
    print()
    print(f"  Patchable:   {patchable}/{len(kernel_results)}")
    print(f"  Unpatchable: {unpatchable}/{len(kernel_results)}")

    if unpatchable and not inspect_only:
        print(f"\nWARNING: {unpatchable} kernels contain unsupported opcodes and were NOT patched.")
        print("These likely use FP8/MXFP instructions that don't exist on gfx90a.")
        print("The category may still be usable if ATOM dispatches to CK/Triton fallback for these shapes.")

    return 0 if patchable > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
