#!/usr/bin/env bash
# Set the sustained power cap on every MI210 (PCI device 0x740f) in this box.
#
# WHY THIS EXISTS. docs/53 measured the cards pinned at exactly 199-200 W under
# sustained load while clocking 1235-1375 MHz against a 1700 MHz boost -- a
# power cap, not a thermal limit, with power1_cap_max reading 300 W. Raising it
# gained 8.4% on prefill at 300 W and 2.9% at 250 W, with decode flat (it is
# memory-bound and mclk was already at its 1600 MHz maximum). 250 W was chosen
# to take the safe part of that gain while giving back 100 W across both cards.
#
# WHY IT LOOKS UP CARDS BY DEVICE ID. hwmon numbering is not stable across
# reboots -- /sys/class/drm/card2/device/hwmon/hwmon10 today may be hwmon9
# tomorrow, and this box has a THIRD AMD device (a 190 W part) that must never
# be written. Matching vendor 0x1002 + device 0x740f is the only stable way to
# name exactly the two MI210s.
#
# SAFETY. sysfs silently CLAMPS an out-of-range value rather than rejecting it,
# so every write is read back and verified. A cap that did not apply is
# reported as a failure, not passed over in silence -- the first version of the
# benchmark that produced this setting printed "caps restored" while the cards
# sat at 300 W, which is the failure mode this guards against.
set -uo pipefail

WATTS="${1:-${GPU_POWER_CAP_WATTS:-250}}"
TARGET_UW=$(( WATTS * 1000000 ))
MI210_DEVICE=0x740f

found=0
failed=0

for d in /sys/class/drm/card*/device; do
    [ -e "$d/vendor" ] || continue
    [ "$(cat "$d/vendor" 2>/dev/null)" = "0x1002" ] || continue
    [ "$(cat "$d/device" 2>/dev/null)" = "$MI210_DEVICE" ] || continue

    hw=$(ls -d "$d"/hwmon/hwmon* 2>/dev/null | head -1)
    [ -n "$hw" ] || { echo "no hwmon under $d" >&2; failed=1; continue; }

    max=$(cat "$hw/power1_cap_max" 2>/dev/null || echo 0)
    if [ "$TARGET_UW" -gt "$max" ]; then
        echo "refusing ${WATTS}W on $hw: exceeds power1_cap_max of $(( max / 1000000 ))W" >&2
        failed=1
        continue
    fi

    echo "$TARGET_UW" > "$hw/power1_cap" 2>/dev/null || {
        echo "write failed on $hw (need root?)" >&2; failed=1; continue; }

    # Read back. sysfs clamps rather than erroring, so a successful write is
    # not evidence the value took effect.
    got=$(cat "$hw/power1_cap" 2>/dev/null || echo 0)
    if [ "$got" = "$TARGET_UW" ]; then
        echo "$(basename "$(dirname "$d")"): cap = $(( got / 1000000 ))W (verified)"
        found=$(( found + 1 ))
    else
        echo "$(basename "$(dirname "$d")"): asked ${WATTS}W, reads $(( got / 1000000 ))W" >&2
        failed=1
    fi
done

if [ "$found" -eq 0 ]; then
    echo "no MI210 (device $MI210_DEVICE) found -- nothing set" >&2
    exit 1
fi
[ "$failed" -eq 0 ] || exit 1
echo "set ${WATTS}W on ${found} MI210(s)"
