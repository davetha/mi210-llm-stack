#!/usr/bin/env python3
"""Patch root-level .co files in gfx942/ (not in subdirectories).

These are dispatcher/wrapper kernels like fmoe_b16.co, pa_a16w16_b16.co, etc.
that AiterAsmKernel loads directly by name.
"""
import os, sys, shutil
sys.path.insert(0, "/tmp")
from patch_category import patch_co_file, OPCODES

HSA_ROOT = "/opt/python/lib/python3.14/site-packages/aiter_meta/hsa"
SRC = os.path.join(HSA_ROOT, "gfx942")
DST = os.path.join(HSA_ROOT, "gfx90a")

os.makedirs(DST, exist_ok=True)

# Get root-level .co files only (not in subdirs)
root_cos = sorted([f for f in os.listdir(SRC) if f.endswith(".co") and os.path.isfile(os.path.join(SRC, f))])
print(f"Root-level .co files in gfx942: {len(root_cos)}")
for f in root_cos:
    print(f"  {f}")
print()

import collections
total_mfma = 0
total_unsup = collections.Counter()
patched = 0
skipped = 0

for fname in root_cos:
    src = os.path.join(SRC, fname)
    dst = os.path.join(DST, fname)

    with open(src, "rb") as f:
        data = bytearray(f.read())

    work = bytearray(data)
    mfma, unsup, opcodes, vgpr_old, (text_off, text_size) = patch_co_file(work)
    total_mfma += mfma
    total_unsup.update(unsup)

    flag = ""
    if unsup:
        flag = f" [SKIP: {dict(unsup)}]"
        skipped += 1
    else:
        with open(dst, "wb") as f:
            f.write(work)
        patched += 1

    print(f"  {fname}: mfma={mfma}, vgpr={vgpr_old}, text=0x{text_size:x}{flag}")

print()
print(f"Patched: {patched}/{len(root_cos)}, skipped: {skipped}")
print(f"Total MFMA swaps: {total_mfma}")
if total_unsup:
    print(f"Unsupported opcodes: {dict(total_unsup)}")
