#!/usr/bin/env bash
# Round 47: characterise llama.cpp NATIVE multi-GPU on the production models.
#
# WHY THIS ROUND EXISTS, AND WHY IT IS NOT THE A/B IT WAS MEANT TO BE.
#
# The plan was RPC vs native: production runs every model through
# `--rpc 127.0.0.1:5005x,5005y` with one ggml-rpc-server pinned per card, and
# nobody had measured what that localhost hop costs against llama.cpp's own
# multi-GPU split.
#
# That A/B cannot be run, because the RPC arm no longer exists. Every model in
# /mnt/llm-storage/llama-swap-config.yaml launches `llama-rocm714-rpc:latest`,
# and THAT IMAGE IS NOT ON THE BOX. Surviving llama images are
# llama-rocm714{,-bench,-forcemmq}, and none of them contains
# `ggml-rpc-server` -- only `llama-server`. configs/Dockerfile.llama-bench
# records why: the base was built with `cmake --build build -t llama-server`,
# the server target alone. No build recipe for the RPC image exists in the repo
# or on the host.
#
# So production is not idle, it is BROKEN: every llama-swap model would fail
# with "Unable to find image". Restoring it means either rebuilding an
# RPC-capable image (-DGGML_RPC=ON plus the rpc-server target) or moving to
# native multi-GPU, which needs no rebuild and removes the socket hop.
#
# This round measures the second option on the two production checkpoints, so
# the choice is made on numbers rather than on the fact that it starts.
#
# Verified before writing this: native split serves the 48.5 GB coder model
# across both cards -- GPU0 25.30 GB / GPU1 26.92 GB -- healthy in 40 s, and
# returns an exact one-word ACKNOWLEDGED.
#
# WHAT THIS ROUND CANNOT TELL YOU: what RPC was costing. There is no RPC arm to
# compare against. A native number here is a baseline for the restored
# configuration, not a delta against the old one, and must not be reported as
# one.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 47: llama.cpp native multi-GPU, production models ==="

# The image production's config asks for, and what actually exists.
if docker image inspect llama-rocm714-rpc:latest >/dev/null 2>&1; then
    echo "NOTE: llama-rocm714-rpc:latest now EXISTS -- the RPC arm is runnable"
    echo "again and this round should be extended into the intended A/B."
else
    echo "confirmed: llama-rocm714-rpc:latest is absent; RPC arm not runnable"
fi
export LLAMA_IMAGE=llama-rocm714:latest

# Production settings that are NOT reproduced here, deliberately:
#   -c 262144 -np 8   (coder) -- a 262k context across 8 slots is a throughput
#                     configuration; this measures single-stream latency, and
#                     serve_llamacpp.sh pins --parallel 1.
#   -ctk q8_0 -ctv q8_0 -- KV quantisation is a separate axis. serve_llamacpp.sh
#                     documents that -ctv q4_1 was 11.5x slower on prefill here,
#                     so mixing it in would confound the baseline. Both caches
#                     stay at default precision, as in every other arm.
# 32768 keeps the KV allocation sane and makes LONGCTX_TOKENS clamp to 27,852 --
# the same prompt size every vLLM arm in this repo used, so the numbers sit on
# the same axis.
export READY_TIMEOUT=1800

run_one() {  # label  model-dir
    echo ""
    echo "=== $(date -u +%T) arm: $1  ($2) ==="
    "$BIN/run_arm.sh" "$1" 80B gguf llamacpp "$2" --ctx-size 32768 2>&1 | tail -8
    echo "arm $1 rc=${PIPESTATUS[0]}"
}

run_one rd47-coder-q4   /mnt/llm-storage/coder-next-q4
run_one rd47-thinking-q5 /mnt/llm-storage/qwen3-next-thinking-abl

echo ""
echo "=== $(date -u +%T) round 47 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
arms = [("rd47-coder-q4", "Coder-Next Q4_K_M (48.5 GB)"),
        ("rd47-thinking-q5", "Qwen3-Next-80B Thinking Q5_K_M (56.7 GB)")]
rows = [("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
        ("cold16k", "ttft_s_median", "cold16k ttft"),
        ("longctx", "implied_prefill_tps_median", "longctx prefill"),
        ("longctx", "decode_tps_median", "longctx decode"),
        ("longctx", "ttft_s_median", "longctx ttft")]
print(f"{'metric':<17}" + "".join(f"{n.split('(')[0].strip():>32}" for _, n in arms))
print("-" * 82)
for wl, key, name in rows:
    cells = ""
    for a, _ in arms:
        f = os.path.join(R, f"{a}-{wl}.json")
        v = json.load(open(f)).get(key) if os.path.isfile(f) else None
        cells += f"{v:32.2f}" if isinstance(v, (int, float)) else f"{'-':>32}"
    print(f"{name:<17}{cells}")
print()
for a, n in arms:
    f = os.path.join(R, f"{a}-longctx.json")
    if os.path.isfile(f):
        d = json.load(open(f))
        print(f"  {n}: correctness probe = {d.get('correctness_probe_pass')}")
print()
print("BASELINE, NOT A DELTA. There is no RPC arm to compare against -- the")
print("image production's config requires is gone. These numbers describe the")
print("restored native configuration only.")
PY
