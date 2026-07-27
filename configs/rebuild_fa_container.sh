#!/bin/bash
# rebuild_fa_container.sh — Rebuild the fa-build container from scratch
# Based on docs/16-complete-technical-reference.md
set -ex

IMAGE="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"

# Remove old container if exists
docker rm -f fa-build 2>/dev/null || true

# Start container
docker run -d --name fa-build \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host --shm-size 64G --cap-add SYS_PTRACE \
  -v /mnt/llm-storage:/models \
  -v /tmp/atom:/build/ATOM \
  -v /tmp/opencode/repos/mi210-llm-stack:/build/mi210-llm-stack \
  "$IMAGE" sleep infinity

echo "Container started. Installing packages..."

# 1. Install flydsl
docker exec fa-build pip install flydsl==0.2.2

# 2. Clone and install AITER v0.1.17
docker exec fa-build bash -c '
  git clone --depth 1 --branch v0.1.17 https://github.com/ROCm/aiter.git /tmp/aiter_v017
  cd /tmp/aiter_v017 && pip install --no-build-isolation .
'

# 3. Extract CK headers from official wheel
docker exec fa-build bash -c '
  cd /tmp
  wget -q https://github.com/ROCm/aiter/releases/download/v0.1.17/amd_aiter-0.1.17+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
  python -m zipfile -e amd_aiter-0.1.17+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ck_wheel
  
  SITE=/opt/python/lib/python3.14/site-packages
  
  # Copy CK include headers
  cp -r ck_wheel/aiter_meta/csrc/include/* $SITE/aiter_meta/csp/include/ 2>/dev/null || true
  mkdir -p $SITE/aiter_meta/csp/include
  cp -r ck_wheel/aiter_meta/csrc/include/* $SITE/aiter_meta/csp/include/
  
  # Copy CK (composable_kernel) headers  
  mkdir -p $SITE/aiter_meta/3rdparty/composable_kernel
  cp -r ck_wheel/aiter_meta/3rdparty/composable_kernel/* $SITE/aiter_meta/3rdparty/composable_kernel/
  
  rm -rf ck_wheel amd_aiter-*.whl
'

# 4. Install ATOM v0.1.5
docker exec fa-build bash -c '
  cd /build/ATOM && pip install --no-deps .
'

# 5. Install sitecustomize.py (Triton pre-import — CRITICAL)
docker exec fa-build bash -c '
  cat > /opt/python/lib/python3.14/site-packages/sitecustomize.py << "SITEEOF"
import triton
SITEEOF
'

# 6. Install ROCm 7.14 (co-install with 7.2)
docker exec fa-build bash -c '
  cd /tmp
  wget -q https://repo.radeon.com/rocm/installer/rocm-runfile-installer/rocm-rel-7.14/rocm-installer-7.14.0-6.run
  apt-get update && apt-get install -y rsync
  bash rocm-installer-7.14.0-6.run deps=install gfx=gfx90a --nodiskspace rocm
  ln -sf /opt/rocm-7.2.0/core-7.14 /opt/rocm
'

# 7. Apply binary patches (all 1,251 .co files)
docker exec fa-build bash -c '
  cd /build/mi210-llm-stack
  python configs/patch_category.py
  python configs/patch_root_cos.py
'

echo "Container rebuild complete!"
echo "Testing basic import..."
docker exec fa-build python -c "import aiter; import atom; print('OK: imports work')"
