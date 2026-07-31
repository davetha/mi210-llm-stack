#!/usr/bin/env bash
# Does enabling PCIe peer-to-peer speed up TP=2? One variable, two arms.
#
# THE PREMISE THAT TURNED OUT TO BE WRONG. env/gfx90a-common.env has carried
# NCCL_P2P_DISABLE=1 since the beginning, with the justification "the two MI210s
# have NO xGMI bridge and cannot peer-to-peer". The first half is true. The
# second does not follow from it, and it is false. Found by outside review from
# https://github.com/Andrei-Dr (PR #35).
#
# The driver says so, in /sys/class/kfd/kfd/topology/nodes/{1,2}/p2p_links/0:
#
#   type 2 (HSA_IOLINKTYPE_PCIEXPRESS -- XGMI would be 11), bidirectional
#   max_bandwidth 32000 MB/s
#   flags 3 = Override + NonCoherent, so bit 4 (NoPeerToPeerDMA) is CLEAR
#
# and the runtime agrees, measured 2026-07-31 on this box:
#
#   can_device_access_peer(0,1) = True   (and the reciprocal)
#   cuda:0 -> cuda:1     26.98 GB/s      84% of PCIe 4.0 x16 theoretical
#   cuda:1 -> cuda:0     26.97 GB/s
#   via pinned host      14.16 GB/s      what an allreduce pays with P2P off
#
# So the peer path is real and is 1.9x the staged path. Every TP=2 allreduce
# ever measured in this project took the 14 GB/s route.
#
# WHAT THIS DOES NOT ASSUME. That 1.9x on a bulk copy becomes 1.9x on decode, or
# any speedup at all. Three reasons it might not:
#
#   1. Allreduce moves small buffers, where latency dominates bandwidth, and the
#      ACS ReqRedir+/CmpltRedir+ state on both upstream bridges routes peer
#      traffic through the root complex rather than bridge-to-bridge. The 26.98
#      GB/s already includes that redirect, so it is not fatal -- but a latency
#      penalty it hides at 512 MiB may dominate at allreduce sizes.
#   2. Decode here sits ~3.1x off its achievable-bandwidth bound (docs/25 1c)
#      with full CUDA-graph capture already on, so much of per-token cost is
#      in-kernel and untouched by interconnect.
#   3. TP=2 currently returns 1.28x (33.80 -> 43.40 t/s), and the gap to 2x may
#      be compute or memory rather than the link.
#
# A null result is therefore a real and expected outcome, and is worth recording
# either way -- the premise in the env file is wrong regardless of whether
# fixing it is worth anything.
#
# AND THE ORIGINAL STALL WAS REAL. Something made RCCL hang during collective
# setup, and neither the driver reading nor the bandwidth measurement explains
# it. Arm B may hang. READY_TIMEOUT is deliberately short so that a hang costs
# 10 minutes and is recorded as the result it is, rather than idling the FIFO.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 31: NCCL_P2P_DISABLE A/B at TP=2 ==="

# The 30B W8A8 checkpoint: 60 s load, so both arms are cheap, and it is the
# model docs/28 characterises most thoroughly at TP=2 -- 7,278 prefill / 43.4
# decode -- which gives the A arm a published value to reproduce. An arm that
# fails to reproduce it invalidates the comparison before the B arm is read.
MODEL=$BASE/t35-w8a8
IMG=rocm-vllm-aiter-gfx90a:pa256k

[ -d "$MODEL" ] || { echo "FATAL: $MODEL missing"; bench_release; exit 1; }

# Re-measure the interconnect inside this round rather than citing the numbers
# in the header. If the A/B comes out null, the first question will be whether
# P2P was actually available at the time, and a measurement taken hours earlier
# on a different container does not answer it.
echo ""
echo "=== 0. confirm the peer path is live right now ==="
docker run --rm --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
  --security-opt seccomp=unconfined --ipc=host --shm-size 8G \
  -v "$BIN/p2p_bw.py":/tmp/p2p_bw.py \
  --entrypoint python3 "$IMG" /tmp/p2p_bw.py 2>&1 | tee "$BASE/logs/rd31-p2p-probe.log"

if ! grep -q "P2P is functional" "$BASE/logs/rd31-p2p-probe.log"; then
    echo "FATAL: peer path not confirmed; the A/B below would be uninterpretable."
    bench_release
    exit 1
fi

# 10 minutes. A 30B W8A8 at TP=2 loads in ~60 s, so anything past this is the
# collective-setup stall, not slow loading.
export READY_TIMEOUT=600
export TP=2
export VLLM_IMAGE="$IMG"

for p2p in 1 0; do
    label="rd31-t35w8a8-tp2-p2pdisable${p2p}"
    echo ""
    echo "=== $(date -u +%T) arm: NCCL_P2P_DISABLE=$p2p ==="
    # Exported, not passed through VLLM_EXTRA_ENV: serve_vllm_aiter.sh reads it
    # as ${NCCL_P2P_DISABLE:-1} on its own -e line. Going through EXTRA_ENV
    # would emit a SECOND -e for the same key and rely on docker's last-one-wins
    # ordering, which is exactly the kind of implicit dependency that makes an
    # arm unattributable when the script is later reordered.
    export NCCL_P2P_DISABLE=$p2p

    if "$BIN/run_arm.sh" "$label" 35B w8a8 vllm "$MODEL" \
         --max-model-len 32768 2>&1 | tail -25; then
        echo "arm $label completed"
    else
        echo "arm $label FAILED (recorded)"
    fi
done

echo ""
echo "=== $(date -u +%T) round 31 done ==="
echo ""
echo "READING THIS. Compare decode_tps_median between the two results files."
echo "The A arm (p2pdisable1) should land near docs/28's 43.4 t/s; if it does"
echo "not, stop -- the comparison is invalid before the B arm is even read."
for p2p in 1 0; do
    f="$BASE/results/rd31-t35w8a8-tp2-p2pdisable${p2p}.json"
    if [ -f "$f" ]; then
        python3 -c "
import json; d=json.load(open('$f'))
print(f\"  P2P_DISABLE=$p2p  prefill {d.get('implied_prefill_tps_median')}  decode {d.get('decode_tps_median')}\")"
    else
        echo "  P2P_DISABLE=$p2p  NO RESULT (see results/*-FAILED.json)"
    fi
done

bench_release
