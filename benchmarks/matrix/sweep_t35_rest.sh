#!/bin/sh
B=/mnt/llm-storage/bench-matrix
arm() {
  L=$1; Q=$2; E=$3; D=$4; T=$5; shift 5
  echo ""; echo "######## $(date -u +%H:%M:%S) $L (tp=$T) ########"
  LC=110000; [ "$E" = "llamacpp" ] && LC=262144
  TP=$T LONGCTX_TOKENS=$LC READY_TIMEOUT=16000 $B/bin/run_arm.sh "$L" 35B "$Q" "$E" "$D" "$@" || echo "!! $L nonzero (recorded)"
}
arm t35-w8a8   w8a8   vllm-aiter $B/t35-w8a8      1 --max-model-len 131072 --safetensors-load-strategy=prefetch
arm t35-fp8    fp8    vllm-aiter $B/t35-fp8       1 --max-model-len 131072 --safetensors-load-strategy=prefetch
arm t35-q8     q8_0   llamacpp   $B/t35-gguf-q8   1 --ctx-size 131072
arm t35-bf16   bf16   vllm-aiter $B/t35-bf16      2 --max-model-len 131072 --safetensors-load-strategy=prefetch
echo "SWEEP-DONE $(date -u)"
