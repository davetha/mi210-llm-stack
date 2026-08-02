#!/usr/bin/env bash
# Round 48: what does the RPC hop cost? RPC vs native multi-GPU, one binary.
#
# Production routes every model through `--rpc 127.0.0.1:5005x,5005y` with one
# ggml-rpc-server pinned per card. Nobody had measured that against llama.cpp's
# own multi-GPU split, and docs/47 could not, because the RPC image had been
# lost. configs/Dockerfile.llama-rpc rebuilds it from the SAME source tree the
# baseline was compiled from (/src at 67b9b0e, inside llama-rocm714:latest).
#
# ONE BINARY, ONE VARIABLE. Both arms run llama-rocm714-rpc:latest. The native
# arm omits --rpc and lets llama.cpp split layers itself; the RPC arm starts the
# two rpc-server containers and passes --rpc, exactly as production did.
# Comparing the new image against llama-rocm714:latest instead would vary the
# GGML_RPC compile flag AND the transport together, and any difference would be
# unattributable -- the mistake configs/Dockerfile.llama-forcemmq documents.
#
# WHAT TO EXPECT, SO A NULL IS READABLE. Both rpc-servers and both cards are on
# this one host, so RPC here is a local socket carrying tensors that native
# would pass as pointers. The cost should appear as a
# per-layer latency tax, i.e. in DECODE (many small transfers per token) more
# than in prefill (few large ones). If RPC is free, the two arms match and
# production's pattern is costless. If it is not, decode moves first.
#
# NOISE. docs/46 measured this rig's decode bar at 1.036x for a single pair of
# arms; prefill and TTFT need only ~1.005x. Those bars were measured on vLLM
# arms, and llama.cpp may sit differently -- so a decode delta under ~3.6% here
# is reported as inconclusive rather than as an effect, and the prefill and TTFT
# rows carry more weight.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
LOGS=$BASE/logs
cd "$BASE"

IMG=llama-rocm714-rpc:latest
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "FATAL: missing $IMG -- build configs/Dockerfile.llama-rpc first"; exit 1; }
docker run --rm --entrypoint test "$IMG" -x /src/build/bin/ggml-rpc-server \
  || { echo "FATAL: $IMG has no ggml-rpc-server"; exit 1; }

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 48: RPC vs native multi-GPU, same binary ==="

MODEL=/mnt/llm-storage/coder-next-q4     # production's checkpoint, 48.5 GB
export LLAMA_IMAGE="$IMG"
export READY_TIMEOUT=1800

cleanup_rpc() { docker rm -f rd48-rpc0 rd48-rpc1 >/dev/null 2>&1 || true; }
trap cleanup_rpc EXIT

# THE NETWORKING TRAP. Production runs its main llama-server with
# `--network host`, so `--rpc 127.0.0.1:5005x` reaches the host-networked
# rpc-servers. bin/serve_llamacpp.sh instead publishes a port on the default
# BRIDGE network, where 127.0.0.1 is the container's OWN loopback -- pointing
# --rpc at it would give connection-refused, and the arm would fail for a
# reason that has nothing to do with RPC's cost. So the rpc-servers bind
# 0.0.0.0 and the client dials the bridge gateway instead.
GW="$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)"
[ -n "$GW" ] || { echo "FATAL: could not resolve the docker bridge gateway"; exit 1; }
echo "bridge gateway for RPC dial-out: $GW"

# ---- arm 1: native -------------------------------------------------------
echo ""
echo "=== $(date -u +%T) arm: rd48-native (no --rpc) ==="
"$BIN/run_arm.sh" rd48-native 80B gguf llamacpp "$MODEL" --ctx-size 32768 2>&1 | tail -8
echo "arm rd48-native rc=${PIPESTATUS[0]}"

# ---- arm 2: RPC ----------------------------------------------------------
# One rpc-server per card, pinned by HIP_VISIBLE_DEVICES, mirroring
# /mnt/llm-storage/launch-coder-q4-rpc.sh.
echo ""
echo "=== $(date -u +%T) starting per-card RPC backends ==="
cleanup_rpc
for i in 0 1; do
    docker run -d --name "rd48-rpc$i" --init --network host \
      --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -e HIP_VISIBLE_DEVICES=$i \
      --entrypoint /src/build/bin/ggml-rpc-server "$IMG" \
      -H 0.0.0.0 -p "5006$i" >/dev/null
done
sleep 8
for i in 0 1; do
    if docker ps --format '{{.Names}}' | grep -q "rd48-rpc$i"; then
        echo "  rd48-rpc$i up on 5006$i (card $i)"
    else
        echo "FATAL: rd48-rpc$i did not stay up:"; docker logs "rd48-rpc$i" 2>&1 | tail -5
        exit 1
    fi
done

echo ""
echo "=== $(date -u +%T) arm: rd48-rpc (--rpc, two backends) ==="
# LLAMA_NO_LOCAL_GPU=1 is REQUIRED here, and its absence invalidated the first
# attempt. Production's main llama-server container carries no --device flags --
# all compute goes through the rpc-servers. serve_llamacpp.sh passes them by
# default, so llama.cpp enumerated 2 local ROCm devices AND 2 RPC backends,
# four backends over two physical cards. It double-allocated them, measured
# prefill at 0.658x, and then died in the rocr VMFaultHandler mid-longctx. That
# was a harness misconfiguration, not a cost of RPC.
LLAMA_NO_LOCAL_GPU=1 "$BIN/run_arm.sh" rd48-rpc 80B gguf llamacpp "$MODEL" \
    --ctx-size 32768 --rpc "$GW:50060,$GW:50061" 2>&1 | tail -8
echo "arm rd48-rpc rc=${PIPESTATUS[0]}"

# Proof the RPC arm actually used the backends rather than silently falling
# back to local devices -- the whole comparison rests on this.
for i in 0 1; do
    n=$(docker logs "rd48-rpc$i" 2>&1 | grep -c . || true)
    echo "  rd48-rpc$i log lines: ${n:-0}"
done
grep -icE "rpc" "$LOGS/rd48-rpc.serverlog" 2>/dev/null \
    | sed 's/^/  rpc mentions in server log: /'

cleanup_rpc

echo ""
echo "=== $(date -u +%T) round 48 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
def get(a, wl, k):
    f = os.path.join(R, f"{a}-{wl}.json")
    return json.load(open(f)).get(k) if os.path.isfile(f) else None
rows = [("cold16k", "implied_prefill_tps_median", "cold16k prefill"),
        ("cold16k", "ttft_s_median", "cold16k ttft"),
        ("longctx", "implied_prefill_tps_median", "longctx prefill"),
        ("longctx", "decode_tps_median", "longctx decode"),
        ("longctx", "ttft_s_median", "longctx ttft")]
print(f"{'metric':<17}{'native':>11}{'RPC':>11}{'RPC/native':>12}")
print("-" * 51)
for wl, k, name in rows:
    a, b = get("rd48-native", wl, k), get("rd48-rpc", wl, k)
    if a is None and b is None:
        continue
    ca = f"{a:11.2f}" if isinstance(a, (int, float)) else f"{'-':>11}"
    cb = f"{b:11.2f}" if isinstance(b, (int, float)) else f"{'-':>11}"
    fac = f"{b/a:11.3f}x" if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else f"{'-':>12}"
    print(f"{name:<17}{ca}{cb}{fac}")
for a in ("rd48-native", "rd48-rpc"):
    d = get(a, "longctx", "correctness_probe_pass")
    print(f"  {a}: correctness = {d}")
print()
print("RPC/native BELOW 1.0 on a throughput row means RPC is SLOWER (it is the")
print("numerator). Decode should move first if the loopback hop costs anything;")
print("treat a decode delta under ~3.6% as inconclusive per docs/46.")
PY
