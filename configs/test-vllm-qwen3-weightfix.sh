#!/bin/bash
# test-vllm-qwen3-weightfix.sh — Test vLLM weight loading with workaround for ROCm hang
# The vLLM optimization task found that real safetensors loading hangs on MI210.
# This script tries 3 workarounds identified from code analysis:
#   1. --load-format safetensors (avoid fastsafetensors ParallelLoader deadlock)
#   2. VLLM_WORKER_MULTIPROC_METHOD=spawn (avoid fork-based deadlock)
#   3. VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 (disable CUDA pin_memory during load)
#
# PREREQUISITES:
#   1. Production stopped: docker stop llama-swap && docker stop llama-main rpc0 rpc1
#   2. Qwen3 model at /mnt/llm-storage/Qwen3-235B-A22B-GPTQ-Int4/
#
PORT="${1:-8000}"
MODEL="/models/Qwen3-235B-A22B-GPTQ-Int4"
IMAGE="llama-vllm025:gfx90a-fixed"

echo "=== vLLM Weight Loading Workaround Test ==="
echo "Model: $MODEL"
echo "Port: $PORT"
echo ""

# Try each workaround
for ATTEMPT in 1 2 3; do
    echo ""
    echo "=== ATTEMPT $ATTEMPT ==="
    case $ATTEMPT in
        1)
            echo "Workaround: --load-format safetensors (standard path, no fastsafetensors)"
            EXTRA_ARGS="--load-format safetensors"
            EXTRA_ENV=""
            ;;
        2)
            echo "Workaround: VLLM_WORKER_MULTIPROC_METHOD=spawn + --load-format safetensors"
            EXTRA_ARGS="--load-format safetensors"
            EXTRA_ENV="-e VLLM_WORKER_MULTIPROC_METHOD=spawn"
            ;;
        3)
            echo "Workaround: All three combined"
            EXTRA_ARGS="--load-format safetensors"
            EXTRA_ENV="-e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1"
            ;;
    esac

    docker stop vllm-weight-test 2>/dev/null; docker rm vllm-weight-test 2>/dev/null

    timeout 300 docker run --rm --name vllm-weight-test --network host \
        --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
        --ipc=host --shm-size=16g -v /mnt/llm-storage:/models \
        -e VLLM_USE_AITER=0 \
        -e VLLM_USE_TRITON_FLASH_ATTN=1 \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e PYTORCH_ROCM_ARCH=gfx90a \
        $EXTRA_ENV \
        --entrypoint python $IMAGE \
        /models/vllm-weight-test.py 2>&1 | tail -30 &

    PID=$!
    echo "Waiting up to 5 min for attempt $ATTEMPT (PID $PID)..."

    # Check every 10s for 5 min
    for i in $(seq 1 30); do
        sleep 10
        if docker exec vllm-weight-test curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q ok; then
            echo "SUCCESS on attempt $ATTEMPT after ${i}0s!"
            echo "Workaround that works:"
            echo "  EXTRA_ARGS: $EXTRA_ARGS"
            echo "  EXTRA_ENV: $EXTRA_ENV"
            docker stop vllm-weight-test 2>/dev/null
            exit 0
        fi
    done

    echo "Attempt $ATTEMPT timed out after 5 min"
    kill $PID 2>/dev/null
    docker stop vllm-weight-test 2>/dev/null; docker rm vllm-weight-test 2>/dev/null
done

echo ""
echo "=== ALL ATTEMPTS FAILED ==="
echo "The weight-loading hang persists with all workarounds."
echo "Next steps: check vLLM GitHub issues for ROCm + GPTQ weight loading bugs,"
echo "or try building vLLM from source with a newer ROCm version."
