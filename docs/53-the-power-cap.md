# The power cap: the largest lever, and it was not software

Every software lead in `docs/50`–`docs/52` fought for 1–3%. The cards were
running at two thirds of their rated power the whole time.

## The finding

`docs/51` sampled clocks every 20 s through round 57 and found, in **every**
sample:

```
sclk    1235-1375 MHz   against a 1700 MHz boost
power   199-200 W       flat, every sample
junction 67-71 C        comfortable
```

Flat power with thermal headroom is a **power cap**, not a thermal limit:

```
card2  power1_cap = 200W   power1_cap_max = 300W
card3  power1_cap = 200W   power1_cap_max = 300W
```

`card1` is a third AMD device with a 190 W cap and is **not** an MI210 — any
tooling here matches on PCI device `0x740f` so it cannot be touched by accident.

## The measurement

`round63_power_cap_300w.sh`, TP=2, three caps, prefill and decode driven
separately:

| cap | prefill tok/s | vs 200W | median TTFT | decode tok/s | vs 200W |
|---:|---:|---:|---:|---:|---:|
| 200 W | 5078.63 | 1.000× | 1704.74 ms | 318.69 | 1.000× |
| 250 W | 5226.44 | **1.029×** | 1614.70 ms | 316.89 | 0.994× |
| 300 W | 5503.41 | **1.084×** | 1612.33 ms | 316.33 | 0.993× |

**Prefill gains, decode does not** — which is the prediction written into the
round *before* it ran. Clock feeds compute-bound prefill directly; decode is
memory-bound and `mclk` was already at its 1600 MHz maximum, so there was
nothing for extra power to buy. A round where both moved equally would have
been measuring variance, and the round says so in its own output.

Thermals never approached the limits: peak junction 63 °C and HBM 69 °C, against
watchdog thresholds of 90/85 and hardware criticals of 100/94.

## What was deployed, and why not 300 W

**250 W**, persisted. It takes the safe part of the gain and hands back 100 W
across the pair. The 300 W step measured better (+8.4%) but the extra 5.5
points cost another 100 W, and this box runs continuously.

```
configs/set-gpu-power-cap.sh     finds MI210s by device ID, writes, VERIFIES
configs/gpu-power-cap.service    re-applies at boot (sysfs does not persist)
```

The script looks cards up by `vendor 0x1002 + device 0x740f` rather than by
hwmon path, because **hwmon numbering is not stable across reboots** — today's
`hwmon10` may be tomorrow's `hwmon9` — and a third AMD device is present that
must never be written.

## Caveats I am not going to smooth over

**The clock samples do not prove the mechanism.** Every prefill sample reads
800 MHz, because the sampler fires 25 s in and lands between bursts rather than
during compute. Decode samples show 1700 MHz at 138–142 W, *below* even the
200 W cap. So there is a consistent, monotonic prefill gain across three steps —
1.000 → 1.029 → 1.084 — and **no direct evidence that clocks rose because of the
cap**. The correlation is clean; the causal link is inferred. A rerun with
sampling that actually catches prefill under load would settle it.

**HBM is the real thermal constraint, not junction.** Verified on the hardware
rather than assumed:

```
temp2 junction  crit 100 C   emergency 105 C
temp3 mem       crit  94 C   emergency  99 C
```

Memory trips 6 °C earlier, and the cards are asymmetric — one idles with HBM at
55 °C while its junction reads 44 °C, i.e. **memory hotter than the die**. The
first draft of the watchdog read junction only and would have let HBM reach
within 4 °C of critical before reacting. Any future power work on this box must
watch `temp3`.

## The failure worth recording

The first run of round 63 printed

```
CAPS ARE RESTORED TO 200W BY THE EXIT TRAP.
```

while both cards sat at **300 W**. `restore_caps` never ran — there is no "caps
restored" line anywhere in the log — and that message was an unconditional
`echo` asserting a safety property nothing had checked. It was caught only by
reading live sysfs instead of believing the script.

This is the same defect `docs/50` and `docs/51` recorded twice already —
writing the conclusion into the instrument — but applied to a **hardware safety
claim** rather than a throughput number, which makes it materially worse: a
false safety message stops anyone from looking. The restore path now writes,
reads back, and prints the exact recovery commands if the values disagree.

Related, and cheaper: the round reported prefill as missing for all three steps
because the parser matched `Total Token throughput` while vLLM prints
`Total token`. The data was there the whole time.
