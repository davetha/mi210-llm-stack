#!/bin/bash
# run-vllm-aiter-ab.sh [config...] -- the A/B behind vllm-aiter-asm-gfx90a.md.
#
# For each config: start vLLM, wait for readiness, run bench_vllm_serving.py,
# print which AITER kernels the run actually loaded, tear down. Defaults to all
# three configs. Run inside the container that has vLLM + aiter installed, with
# configs/enable_vllm_aiter_gfx90a.py already applied.
#
#     ./run-vllm-aiter-ab.sh                     # all three
#     ./run-vllm-aiter-ab.sh aiter-fa-asm stock  # a subset
set -u

MODEL=${MODEL:-Qwen/Qwen3-14B}
PORT=${PORT:-8000}
OUTDIR=${OUTDIR:-/root}
BENCH=${BENCH:-/root/bench_vllm_serving.py}
# CASES trims the sweep, as promptlen:concurrency pairs. Useful when a
# configuration is slow enough that the full grid would take hours -- the FP8
# runs are ~15x slower per token than bf16.
CASES=${CASES:-}
# Device 0 in HIP terms. Note this is the card rocm-smi calls GPU[1] -- the
# orderings are reversed, and picking the wrong one fails with a free-memory
# error that looks like a config problem.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(aiter-fa-asm aiter-fa stock)

# `vllm serve` forks a VLLM::EngineCore whose cmdline does not match the
# parent. Killing only the parent leaves it holding ~58 GB and the next config
# dies claiming insufficient free memory. Match it explicitly, then wait for
# the driver to actually hand the memory back -- a fixed sleep is not enough.
kill_vllm() {
  ps -eo pid,cmd | grep -iE '[v]llm serve|[V]LLM::' | awk '{print $1}' | xargs -r kill -9
  for _ in $(seq 1 30); do
    used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep Used | sed -n 2p | awk '{print $NF}')
    [ -n "${used:-}" ] && [ "$used" -lt 2000000000 ] && return 0
    sleep 5
  done
  echo "WARNING: VRAM still held after 150s (used=${used:-unknown})"
}

start_server() {
  local cfg="$1" log="$2"
  # Plugins off: ATOM registers a vLLM *platform* plugin that replaces
  # RocmPlatform, and with it the backend selection under test.
  local env=(VLLM_PLUGINS= AITER_LOG_LEVEL=info)
  local extra=()

  # The AITER master switch also enables AITER linear/MoE/RMSNorm/FP8-BMM, none
  # of which are attention. They stay off so the A/B isolates attention.
  local attn_only=(
    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=1
    VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0
    VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_FP8BMM=0
  )

  case "$cfg" in
    stock)          env+=(VLLM_ROCM_USE_AITER=0) ;;
    aiter-fa)       env+=("${attn_only[@]}" VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0)
                    extra=(--attention-config '{"backend":"ROCM_AITER_FA"}') ;;
    aiter-fa-asm)   env+=("${attn_only[@]}" VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1)
                    extra=(--attention-config '{"backend":"ROCM_AITER_FA"}') ;;
    *) echo "unknown config: $cfg" >&2; return 2 ;;
  esac

  env "${env[@]}" vllm serve "$MODEL" \
      --port "$PORT" --dtype bfloat16 --max-model-len 8192 \
      --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
      --max-num-seqs 64 "${extra[@]}" > "$log" 2>&1 &
}

for cfg in "${CONFIGS[@]}"; do
  log="$OUTDIR/serve-$cfg.log"
  echo "=== $cfg: starting $(date +%H:%M:%S) ==="
  kill_vllm
  start_server "$cfg" "$log" || continue

  ready=0
  for _ in $(seq 1 300); do
    grep -q "Application startup complete" "$log" 2>/dev/null && { ready=1; break; }
    if grep -qE "EngineCore failed to start|Engine core initialization failed" "$log" 2>/dev/null; then
      echo "=== $cfg: STARTUP FAILED ==="
      grep -oE "ValueError: .*|RuntimeError: .*" "$log" | head -3
      break
    fi
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "=== $cfg: NOT READY, skipping benchmark ==="
    kill_vllm
    continue
  fi
  echo "=== $cfg: ready $(date +%H:%M:%S) ==="

  bench_args=(--label "$cfg" --model "$MODEL" --out "$OUTDIR/results-$cfg.json")
  [ -n "$CASES" ] && bench_args+=(--cases $CASES)
  python "$BENCH" "${bench_args[@]}" 2>&1 | tee "$OUTDIR/bench-$cfg.log"

  # Kernel provenance, read AFTER traffic: aiter logs LoadKernel lazily, on
  # first use. An empty list here for an aiter-* config means the ASM path was
  # never reached and the numbers above are a fallback, not an ASM result.
  echo "=== $cfg: kernels loaded ==="
  grep -o "LoadKernel: .*" "$log" | sort -u

  kill_vllm
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
