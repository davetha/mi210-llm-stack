#!/bin/sh
# launch-mimo-fa-off-test.sh — MAINTENANCE-WINDOW ONLY FA-off test for MiMo 230B
#
# Purpose: test whether -fa off works on MiMo (vs crashing like DSV2-Lite's MLA
# did) and whether it's faster. See changes/08-mimo-fa-off-test.md.
#
# ⚠️  DO NOT run while production mimo is live. This is a SECOND full 174 GB
#     model instance — it will OOM if production 'mimo' is still loaded.
#     Run only during a scheduled maintenance window AFTER:
#       docker stop llama-main      # (stops the production mimo container)
#
# Differences from production launch-mimo.sh (the variables under test):
#   -fa off            (production: -fa on)
#   -ctk f16 -ctv f16  (production: q8_0/q4_1 — quantized V REQUIRES -fa on,
#                       so it cannot load with -fa off; f16 is the only type
#                       that loads without FA on this build)
#   -c 16384           (production: 65536 — reduced so f16 KV cache fits VRAM)
#   port 8099, container llama-mimo-faoff, alias mimo-faoff
#   no session restore (test instance, not the warm-session path)
#
# Usage:
#   sh /mnt/llm-storage/launch-mimo-fa-off-test.sh
# Then benchmark http://127.0.0.1:8099 with the same prompts used for the
# FA-on baseline (see changes/08-mimo-fa-off-test.md §3).

PORT="${1:-8099}"

docker run --rm --name llama-mimo-faoff --network host --init \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
  --ipc=host --shm-size=16g -v /mnt/llm-storage:/models \
  --entrypoint /src/build/bin/llama-server llama-rocm714:latest \
  -m "/models/mimo-v25/Q4_K/Huihui-MiMo-V2.5-abliterated-Q4_K-00001-of-00021.gguf" \
  -ngl 999 \
  -ot "blk\.([0-9]|1[0-9]|2[0-4])\.ffn.*exps=CPU,blk\.(2[5-9]|3[0-6])\.ffn.*exps=ROCm0,blk\.(3[7-9]|4[0-8])\.ffn.*exps=ROCm1" \
  --host 127.0.0.1 --port "$PORT" -c 16384 -b 2048 -ub 2048 -np 1 \
  -fa off \
  -ctk f16 -ctv f16 \
  --jinja -a mimo-faoff --no-warmup --no-webui
# NOTE: foreground (no '&' / no 'wait') — the operator runs this in a tmux pane
# during the maintenance window and watches the load log for:
#   - "model loaded"           -> FA-off LOADS on MiMo (unlike quantized-V)
#   - GGML_ASSERT ... out of bounds  -> FA-off CRASHES on the split (like MLA)
#   - CUDA/HIP OOM              -> f16 KV at this -c does not fit; retry -c 8192
