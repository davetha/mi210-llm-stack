#!/usr/bin/env bash
# Does PCIe peer-to-peer actually work between these two MI210s?
#
# WHY THIS EXISTS. env/gfx90a-common.env sets NCCL_P2P_DISABLE=1 with the
# comment "the two MI210s have NO xGMI bridge and cannot peer-to-peer". The
# first half is true. The second does not follow, and the driver disagrees:
#
#   /sys/class/kfd/kfd/topology/nodes/{1,2}/p2p_links/0/properties
#     type 2              HSA_IOLINKTYPE_PCIEXPRESS  (XGMI would be 11)
#     max_bandwidth 32000 32 GB/s, PCIe 4.0 x16
#     flags 3
#
# hsakmttypes.h:504-511 defines the flag bits as
#   Override:1  NonCoherent:1  NoAtomics32bit:1  NoAtomics64bit:1  NoPeerToPeerDMA:1
# so flags 3 = Override + NonCoherent, and bit 4 NoPeerToPeerDMA is CLEAR.
# Override being set means these flags are authoritative rather than inferred
# from the link type.
#
# xGMI absent and P2P absent are two different claims. This script settles the
# second one with a measurement instead of an inference, because the cost of
# being wrong is that every tensor-parallel allreduce is staged through host
# memory -- and TP=2 currently returns 1.28x (33.80 -> 43.40 tok/s decode,
# results/t35-w8a8-longctx.json vs docs/25 item 1c) rather than anything near 2x.
#
# HOW THIS COULD BE WRONG. The driver advertising a link is not proof that RCCL
# can use it. ACS redirect on the upstream bridges (0000:86:00.0, 0000:c2:00.0)
# can force peer traffic through the root complex; that still works but is not
# free. And the original stall that motivated NCCL_P2P_DISABLE=1 was real --
# this script does not explain it, it only tests whether the premise given for
# it holds.
set -uo pipefail

echo "=== 1. what the driver says ==="
for n in /sys/class/kfd/kfd/topology/nodes/*/; do
    id=$(basename "$n")
    name=$(cat "$n/name" 2>/dev/null)
    [ -d "$n/p2p_links" ] || continue
    for l in "$n"/p2p_links/*/; do
        [ -d "$l" ] || continue
        t=$(awk '/^type /{print $2}' "$l/properties" 2>/dev/null)
        bw=$(awk '/^max_bandwidth /{print $2}' "$l/properties" 2>/dev/null)
        fl=$(awk '/^flags /{print $2}' "$l/properties" 2>/dev/null)
        to=$(awk '/^node_to /{print $2}' "$l/properties" 2>/dev/null)
        case "$t" in
            2)  tn="PCIe" ;;
            11) tn="XGMI" ;;
            *)  tn="type$t" ;;
        esac
        # bit 4 of flags is NoPeerToPeerDMA
        if [ $(( fl & 16 )) -ne 0 ]; then p2p="BLOCKED (NoPeerToPeerDMA set)"; else p2p="permitted"; fi
        echo "  node $id ($name) -> node $to : $tn, ${bw} MB/s, flags=$fl, P2P DMA $p2p"
    done
done

echo
echo "=== 2. ACS control state on the upstream bridges ==="
# ACSCap is capability; ACSCtl is what is actually enabled. Only ACSCtl matters:
# with redirect bits set, peer traffic is forced up to the root complex.
#
# lspci NEEDS ROOT to decode extended capabilities. Run as an ordinary user it
# prints no ACSCtl line at all, which is indistinguishable here from a bridge
# that genuinely has no ACS -- and the original version of this script then went
# on to assert the ACS state in its trailer regardless. Say which case it is.
acs_read=0
for dev in $(lspci -D | grep -iE "Instinct|Processing accel" | cut -d' ' -f1); do
    up=$(basename "$(dirname "$(readlink -f /sys/bus/pci/devices/$dev)")")
    echo "  $dev upstream $up:"
    line=$(lspci -vvs "$up" 2>/dev/null | grep -E "ACSCtl")
    if [ -z "$line" ] && [ "$(id -u)" -ne 0 ]; then
        line=$(sudo -n lspci -vvs "$up" 2>/dev/null | grep -E "ACSCtl")
    fi
    if [ -n "$line" ]; then
        printf '%s\n' "$line" | sed 's/^/    /'
        acs_read=1
    elif [ "$(id -u)" -ne 0 ]; then
        echo "    UNREAD -- lspci needs root to decode ACS. Re-run with sudo."
    else
        echo "    (no ACS capability on this bridge)"
    fi
done

echo
echo "=== 3. measured peer bandwidth ==="
if command -v rocm-bandwidth-test >/dev/null 2>&1; then
    rocm-bandwidth-test -m 268435456 2>&1 | sed -n '/Unidirectional/,/^$/p' | head -30
else
    echo "  rocm-bandwidth-test not on PATH."
    echo "  Run inside a ROCm container, e.g.:"
    echo "    docker exec <container> rocm-bandwidth-test -m 268435456"
    echo "  A nonzero device-to-device row is the proof; absence of the tool is not evidence."
fi

echo
echo "Interpretation: a PCIe P2P link with NoPeerToPeerDMA clear AND a nonzero"
echo "device-to-device bandwidth row together justify testing TP=2 without"
echo "NCCL_P2P_DISABLE=1. Either one alone does not."
echo
if [ "$acs_read" = "1" ]; then
    echo "ACS state above was READ from this box."
else
    echo "ACS state above was NOT read -- treat any ACS claim below as history,"
    echo "not as a measurement of the machine you are on."
fi
echo
echo "SETTLED on this box, 2026-07-31. Both links PCIe/32000/flags=3 (P2P DMA"
echo "permitted). ACSCtl reads ReqRedir+ CmpltRedir+ on both upstream bridges,"
echo "so peer traffic is redirected through the root complex rather than routed"
echo "bridge-to-bridge -- functional, but not the short path."
echo
echo "That redirect does NOT erase the benefit, which was the open worry when"
echo "this script was written. Measured peer copy is 26.98 GB/s against 14.16"
echo "GB/s staged through pinned host memory, and the redirect is already in"
echo "that 26.98. The A/B followed: +11.2% prefill and -10.3% TTFT at TP=2,"
echo "decode unchanged (+1.3%, noise). NCCL_P2P_DISABLE now defaults to 0."
echo "See docs/40 and docs/37 section 3.4."
echo
echo "Still untested: pcie_acs_override=downstream, which would remove the"
echo "redirect. It is now worth evaluating, since plain P2P did show a gain."
