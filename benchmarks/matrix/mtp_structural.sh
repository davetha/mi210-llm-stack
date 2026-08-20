#!/usr/bin/env bash
# Startup-only probe: what does enabling speculation change structurally?
# No benchmark -- start each arm, capture the lines that describe KV capacity,
# attention/GDN backend selection and graph capture, then tear down.
set -uo pipefail
PORT=8033
OUT=~/cu-mask-probe/results
mkdir -p "$OUT"

for spec in base mtp1; do
  arm=base; n=""
  [ "$spec" = "mtp1" ] && { arm=mtp; n=1; }
  name="mtp-${arm}${n:+-$n}"
  echo "########## $spec ##########"
  ~/cu-mask-probe/mtp_serve.sh "$arm" ${n:+$n} >/dev/null 2>&1 &
  sleep 25
  for i in $(seq 1 240); do
    curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -qx "$name" || break
    sleep 5
  done
  docker logs "$name" > "$OUT/log-$spec.txt" 2>&1
  grep -iE "kv cache|concurrency|reorder|backend|graph capturing|Capturing CUDA graphs \(|memory profiling|available_kv|gdn|mamba" \
      "$OUT/log-$spec.txt" | sed 's/^.*\] //' | sort -u | head -40
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
done
echo STRUCT_DONE
