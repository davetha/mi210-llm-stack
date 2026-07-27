#!/usr/bin/env python3
"""Report which kernels in a HIP shared library actually use MFMA on gfx90a.

A CMake flag reporting ON is not evidence that matrix instructions are being
emitted. This answers the question at the instruction level: it pulls the
`.hip_fatbin` section out of a .so, splits it into the per-TU AMDGPU code
objects, and disassembles the ones whose kernel names match a filter.

Used to establish that llama.cpp's `fattn-mma-f16` kernels already issue
`v_mfma_f32_16x16x16f16` on gfx90a with GGML_HIP_ROCWMMA_FATTN=OFF.
See docs/22-rocwmma-flash-attention-gfx90a.md.

    python3 scan_fatbin_mfma.py /src/build/bin/libggml-hip.so.0.17.0 flash_attn
"""
import os
import re
import struct
import subprocess
import sys
import tempfile

LLVM_BIN = os.environ.get("LLVM_BIN", "/opt/rocm/llvm/bin")
EM_AMDGPU = 224


def extract_code_objects(so_path, outdir):
    """Split the .hip_fatbin section into individual AMDGPU ELF code objects."""
    fatbin = os.path.join(outdir, "fatbin.bin")
    subprocess.run([f"{LLVM_BIN}/llvm-objcopy",
                    f"--dump-section=.hip_fatbin={fatbin}", so_path],
                   check=True, capture_output=True)
    data = open(fatbin, "rb").read()

    objects, off = [], 0
    while True:
        i = data.find(b"\x7fELF", off)
        if i < 0:
            break
        off = i + 4
        # 64-bit ELF for the AMDGPU machine type; skip the host object
        if i + 64 > len(data) or data[i + 4] != 2:
            continue
        if struct.unpack_from("<H", data, i + 18)[0] != EM_AMDGPU:
            continue
        e_shoff = struct.unpack_from("<Q", data, i + 40)[0]
        e_shentsize = struct.unpack_from("<H", data, i + 58)[0]
        e_shnum = struct.unpack_from("<H", data, i + 60)[0]
        path = os.path.join(outdir, f"co{len(objects):03d}.elf")
        open(path, "wb").write(data[i:i + e_shoff + e_shentsize * e_shnum])
        objects.append(path)
    return objects


def kernel_names(co):
    out = subprocess.run([f"{LLVM_BIN}/llvm-nm", co],
                         capture_output=True, text=True).stdout
    syms = [l.split(" T ", 1)[1] for l in out.splitlines() if " T " in l]
    if not syms:
        return []
    demangled = subprocess.run(["c++filt"], input="\n".join(syms),
                               capture_output=True, text=True).stdout
    return demangled.splitlines()


def mfma_histogram(co, mcpu):
    out = subprocess.run([f"{LLVM_BIN}/llvm-objdump", "-d", f"--mcpu={mcpu}", co],
                         capture_output=True, text=True).stdout
    hist = {}
    for m in re.finditer(r"v_mfma_[a-z0-9_]+", out):
        hist[m.group(0)] = hist.get(m.group(0), 0) + 1
    return hist


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    so_path = sys.argv[1]
    name_filter = sys.argv[2] if len(sys.argv) > 2 else ""
    mcpu = os.environ.get("MCPU", "gfx90a")

    with tempfile.TemporaryDirectory() as tmp:
        objects = extract_code_objects(so_path, tmp)
        print(f"{len(objects)} {mcpu} code objects in {so_path}\n")

        total = 0
        for co in objects:
            names = [n for n in kernel_names(co) if name_filter in n]
            if not names:
                continue
            hist = mfma_histogram(co, mcpu)
            if not hist:
                continue
            count = sum(hist.values())
            total += count
            print(f"{os.path.basename(co)}: {len(names)} matching kernels, "
                  f"{count} MFMA")
            for op, n in sorted(hist.items(), key=lambda kv: -kv[1]):
                print(f"    {n:6d}  {op}")
            print(f"    e.g. {names[0][:110]}")

        print(f"\ntotal MFMA in kernels matching {name_filter!r}: {total}")
        if total == 0:
            print("WARNING: no MFMA found - these kernels are not using the "
                  "matrix cores.")


if __name__ == "__main__":
    main()
