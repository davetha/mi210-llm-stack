#!/bin/sh
# GLM-4.6 IQ3_XS on llama.cpp, requeued.
#
# The original tier-4 sweep skipped this arm: it checks the download status
# before running, and the 148 GB GGUF had not finished when it looked. The
# guard did the right thing -- refusing to load a half-downloaded model, which
# would mmap fine and emit garbage -- but the arm still needs running now that
# fetch_model.py has verified every file against its API size.
#
# This is the arm the ~400B tier exists for: ~148 GB against 128 GB of VRAM, so
# 20-30 GB of experts must live in system RAM. -n-cpu-moe 20 is an untuned
# starting point; if it OOMs, that is recorded rather than quietly shrunk.
B=/mnt/llm-storage/bench-matrix
BIN=$B/bin
while ps -eo cmd | grep -qE "[s]weep_glm.sh|[s]weep_t235|[b]f16_last|[r]erun_t80_cold|[r]etry_t80_awq8|[s]weep_int8_formats"; do sleep 60; done
while docker ps --format "{{.Names}}" | grep -q "^bench-"; do sleep 30; done
if ! grep -q "OK .*glm-gguf-iq3xs" $B/dl-big.status 2>/dev/null; then
  echo "### SKIPPED: glm-gguf-iq3xs download still not verified"; exit 0
fi
echo "### GLM-4.6 IQ3_XS on llama.cpp $(date -u)"
TP=1 LONGCTX_TOKENS=28000 READY_TIMEOUT=20000 \
  $BIN/run_arm.sh glm46-iq3xs 400B iq3_xs llamacpp $B/glm-gguf-iq3xs \
  --ctx-size 32768 --n-cpu-moe 20 \
  || echo "!! glm46-iq3xs failed (recorded)"
echo "### GLM GGUF done $(date -u)"
