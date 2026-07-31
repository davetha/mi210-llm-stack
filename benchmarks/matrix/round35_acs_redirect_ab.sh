#!/usr/bin/env bash
# Does removing the PCIe ACS root-complex redirect help serving?
#
# BACKGROUND. Both MI210s sit behind upstream bridges (0000:86:00.0 and
# 0000:c2:00.0) whose ACS control register reads 0x001d -- SrcValid, ReqRedir,
# CmpltRedir, UpstreamFwd. The two redirect bits force peer traffic up to the
# root complex instead of routing bridge-to-bridge. docs/40 raised this as the
# next lever after P2P itself measured +11.2% prefill (round 31), on the stated
# condition that plain P2P show a gain first. It did.
#
# WHAT IS ALREADY SETTLED, and why this round is still worth running.
# Clearing the redirect (ECAP_ACS+6.w 0x001d -> 0x0011) changes bulk peer
# bandwidth NOT AT ALL:
#
#   redirect ON    cuda:0->1  26.98 GB/s     cuda:1->0  26.97 GB/s
#   redirect OFF   cuda:0->1  26.99 GB/s     cuda:1->0  26.98 GB/s
#
# That is within 0.04%, and it is the expected result for that probe:
# benchmarks/matrix/p2p_bw.py moves 512 MiB per copy, which is bandwidth-bound,
# and a redirect costs latency rather than bandwidth. It does not rule out an
# effect on the small buffers an allreduce actually moves -- which is why this
# round measures serving rather than another microbenchmark.
#
# Note the shape of the P2P result this is chasing: round 31 found P2P worth
# +11.2% on PREFILL and nothing on decode, because prefill allreduces move large
# activation tensors and decode allreduces move small ones. If the redirect
# matters anywhere it should be the opposite -- latency-bound decode -- so a
# decode-only improvement here would be mechanistically coherent, and a
# prefill-only one would be suspicious.
#
# amd_iommu=off on this host, so ACS is buying no isolation while costing the
# redirect. Both states are restored at exit regardless of outcome.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
BRIDGES="0000:86:00.0 0000:c2:00.0"
ACS_ON=001d      # SrcValid + ReqRedir + CmpltRedir + UpstreamFwd
ACS_OFF=0011     # SrcValid + UpstreamFwd
cd "$BASE"

acs_set() {  # acs_set <hexword>
    local v="$1" up
    for up in $BRIDGES; do sudo setpci -s "$up" ECAP_ACS+6.w="$v"; done
    for up in $BRIDGES; do
        printf "    %s -> 0x%s\n" "$up" "$(sudo setpci -s "$up" ECAP_ACS+6.w)"
    done
}

# Restore on ANY exit path. Leaving a machine's PCIe routing altered because a
# benchmark died is not acceptable, and this is the only thing in this round
# that outlives the process.
restore_acs() {
    echo ""
    echo "=== restoring ACS redirect (0x$ACS_ON) ==="
    acs_set "$ACS_ON"
}
trap restore_acs EXIT INT TERM

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 35: ACS redirect A/B ==="

# P2P must be ON or the ACS redirect is irrelevant -- with NCCL_P2P_DISABLE=1
# there is no peer traffic to redirect and both arms would be identical by
# construction. This is the one knob that must NOT be pinned to the old default.
export NCCL_P2P_DISABLE=0
export TP=2
export VLLM_IMAGE=vllm-mi210:latest
export VLLM_TUNED_CONFIG_FOLDER=
export READY_TIMEOUT=900

for state in redirect-off redirect-on; do
    echo ""
    echo "=== $(date -u +%T) arm: $state ==="
    case "$state" in
        redirect-off) acs_set "$ACS_OFF" ;;
        redirect-on)  acs_set "$ACS_ON"  ;;
    esac
    "$BIN/run_arm.sh" "rd35-w8a8-$state" 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
        --max-model-len 32768 2>&1 | tail -14
    echo "arm rd35-w8a8-$state rc=${PIPESTATUS[0]}"
done

echo ""
echo "=== $(date -u +%T) round 35 done ==="
python3 - <<'PY'
import json, os
R = "/mnt/llm-storage/bench-matrix/results"
print(f"{'workload':<9} {'metric':<8} {'redir ON':>10} {'redir OFF':>10} {'factor':>9}")
print("-" * 52)
for w in ("cold16k", "longctx"):
    for metric, key in (("prefill", "implied_prefill_tps_median"),
                        ("decode", "decode_tps_median"),
                        ("ttft", "ttft_s_median")):
        v = {}
        for st in ("redirect-on", "redirect-off"):
            f = os.path.join(R, f"rd35-w8a8-{st}-{w}.json")
            v[st] = json.load(open(f)).get(key) if os.path.isfile(f) else None
        on, off = v["redirect-on"], v["redirect-off"]
        if on is None and off is None:
            continue
        fac = f"{off / on:>8.3f}x" if isinstance(on, (int, float)) and isinstance(off, (int, float)) and on else "        -"
        fo = f"{on:10.2f}" if isinstance(on, (int, float)) else f"{str(on):>10}"
        ff = f"{off:10.2f}" if isinstance(off, (int, float)) else f"{str(off):>10}"
        print(f"{w:<9} {metric:<8} {fo} {ff} {fac}")
PY

echo ""
echo "READING THIS. Bulk peer bandwidth is IDENTICAL with and without the"
echo "redirect (26.98 vs 26.99 GB/s), so any effect here is latency on small"
echo "collectives. Expect it in DECODE if anywhere; a prefill-only change would"
echo "contradict the mechanism and should be treated as drift. TTFT is lower-"
echo "is-better, so its factor inverts."
