#!/usr/bin/env python3
"""Create comprehensive CUDA compat stubs by scanning ATen headers, then compile."""
import os, re, subprocess, glob

stub_dir = "/tmp/cuda_compat"
os.makedirs(stub_dir, exist_ok=True)

torch_inc = "/opt/python/lib/python3.14/site-packages/torch/include"
rocm_inc = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/include"
rocm_core_inc = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/include"

# 1. Scan all ATen headers for CUDA includes
cuda_headers = set()
for path in [f"{torch_inc}/ATen", f"{torch_inc}/c10/cuda"]:
    for root, dirs, files in os.walk(path):
        for fname in files:
            if fname.endswith(('.h', '.hpp')):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        for line in f:
                            m = re.match(r'\s*#include\s+[<"]([^>"]+)[>"]', line)
                            if m:
                                inc = m.group(1)
                                # Match CUDA-style headers
                                if inc.startswith('cu') and inc.endswith('.h') and 'hip/' not in inc:
                                    cuda_headers.add(inc)
                                elif inc in ['nvrtc.h', 'nvToolsExt.h', 'cuda.h', 'cuda_profiler_api.h']:
                                    cuda_headers.add(inc)
                except:
                    pass

print(f"Found {len(cuda_headers)} CUDA headers referenced:")
for h in sorted(cuda_headers):
    print(f"  {h}")

# 2. Create stubs for all of them
for h in cuda_headers:
    stub_path = os.path.join(stub_dir, h)
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, 'w') as f:
        f.write(f"// Auto-generated CUDA compat stub for {h}\n")
        f.write("#pragma once\n")
        f.write("#define __HIP_PLATFORM_AMD__ 1\n")
        # Some headers map to HIP equivalents
        if 'runtime' in h:
            f.write(f'#include <hip/hip_runtime.h>\n')
        elif 'blas' in h:
            f.write(f'// rocBLAS provides compat at link time\n')
        elif 'sparse' in h:
            f.write(f'// rocSPARSE provides compat at link time\n')
        elif 'solver' in h:
            f.write(f'// rocSOLVER provides compat at link time\n')
        f.write('\n')

print(f"\nCreated {len(cuda_headers)} stubs")

# 3. Compile
torch_lib = "/opt/python/lib/python3.14/site-packages/torch/lib"
rocm_lib = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib"
fa_lib = "/opt/python/lib/python3.14/site-packages"

cmd = [
    "g++", "-shared", "-fPIC", "-std=c++17", "-O2",
    "-D__HIP_PLATFORM_AMD__=1",
    "-D__HIP_PLATFORM_HCC__=1",
    "/models/fa_wrapper.cpp",
    f"-I{stub_dir}",
    f"-I{rocm_inc}",
    f"-I{rocm_core_inc}",
    f"-I{torch_inc}",
    f"-I{torch_inc}/torch/csrc/api/include",
    f"-L{fa_lib}", "-lflash_attn_2_cuda",
    f"-L{torch_lib}", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-lc10", "-lc10_hip",
    f"-L{rocm_lib}", "-lamdhip64",
    f"-Wl,-rpath,{torch_lib}:{rocm_lib}:{fa_lib}",
    "-o", "/models/fa_wrapper.so"
]

print(f"\nCompiling with g++...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    errors = [l for l in result.stderr.split('\n') if 'error:' in l or 'fatal' in l]
    print(f"Errors ({len(errors)}):")
    for e in errors[:10]:
        print(f"  {e.strip()}")
else:
    print("EXIT: 0")

if os.path.exists("/models/fa_wrapper.so"):
    print(f"\nSUCCESS: fa_wrapper.so ({os.path.getsize('/models/fa_wrapper.so')} bytes)")
else:
    print("\nFAILED")
