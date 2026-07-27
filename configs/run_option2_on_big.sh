#!/bin/bash
# run_option2_on_big.sh — Run on big (192.168.1.252) to test Option 2
# Pulls latest repo, rebuilds container, applies patch, runs tests.
set -ex

REPO_DIR=/tmp/mi210-llm-stack
IMAGE="rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0"

# 1. Pull latest repo
if [ -d "$REPO_DIR" ]; then
    cd $REPO_DIR && git pull
else
    git clone https://github.com/davetha/mi210-llm-stack.git $REPO_DIR
    cd $REPO_DIR
fi

# 2. Pull docker image (if not present)
docker images | grep -q "$IMAGE" || docker pull "$IMAGE"

# 3. Start container
docker rm -f fa-build 2>/dev/null || true
docker run -d --name fa-build \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host --shm-size 64G --cap-add SYS_PTRACE \
  -v /mnt/llm-storage:/models \
  -v $REPO_DIR:/build/mi210-llm-stack \
  "$IMAGE" sleep infinity

# 4. Install packages
docker exec fa-build pip install flydsl==0.2.2
docker exec fa-build bash -c '
  pip uninstall -y amd-aiter 2>/dev/null || true
  git clone --depth 1 --branch v0.1.17 https://github.com/ROCm/aiter.git /tmp/aiter_v017
  cd /tmp/aiter_v017 && pip install --no-build-isolation .
'

# 5. Extract CK headers
docker exec fa-build bash -c '
  cd /tmp
  wget -q https://github.com/ROCm/aiter/releases/download/v0.1.17/amd_aiter-0.1.17+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
  python -m zipfile -e amd_aiter-0.1.17+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl ck_wheel
  SITE=/opt/python/lib/python3.14/site-packages
  mkdir -p $SITE/aiter_meta/csp/include
  cp -r ck_wheel/aiter_meta/csrc/include/* $SITE/aiter_meta/csp/include/
  mkdir -p $SITE/aiter_meta/3rdparty/composable_kernel
  cp -r ck_wheel/aiter_meta/3rdparty/composable_kernel/* $SITE/aiter_meta/3rdparty/composable_kernel/
  rm -rf ck_wheel amd_aiter-*.whl
'

# 6. Install ATOM
docker exec fa-build bash -c '
  cd /tmp
  git clone --depth 1 --branch v0.1.5 https://github.com/ROCm/ATOM.git /build/ATOM
  cd /build/ATOM && pip install --no-deps .
'

# 7. sitecustomize.py (CRITICAL — Triton pre-import)
docker exec fa-build bash -c 'echo "import triton" > /opt/python/lib/python3.14/site-packages/sitecustomize.py'

# 8. Apply binary patches
docker exec fa-build bash -c '
  cd /build/mi210-llm-stack
  python configs/patch_category.py
  python configs/patch_root_cos.py
'

# 9. Verify imports
docker exec fa-build python -c "import aiter; import atom; print('Imports OK')"

# 10. Apply Option 2 patch
docker exec fa-build python /build/mi210-llm-stack/configs/patch_option2_reshape_and_cache.py

echo ""
echo "========================================"
echo "  Container ready. Running tests..."
echo "========================================"

# 11. Run comprehensive model_ops tests
docker exec fa-build python /build/mi210-llm-stack/tests/test_atom_model_ops.py 2>&1 | tee /tmp/test_model_ops.log

echo ""
echo "========================================"
echo "  Running end-to-end pa_fwd_asm test..."
echo "========================================"

# 12. Run end-to-end test (pa_fwd_asm decode — NO ATOM_USE_UNIFIED_ATTN)
docker exec -e ATOM_LOADER_USE_THREADPOOL=0 fa-build python -m atom.examples.simple_inference \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 256 \
  --max-tokens 10 \
  --block-size 64 \
  --enforce-eager \
  --level 0 2>&1 | tee /tmp/test_pa_fwd_asm.log

echo ""
echo "========================================"
echo "  Done. Check logs:"
echo "  /tmp/test_model_ops.log"
echo "  /tmp/test_pa_fwd_asm.log"
echo "========================================"
