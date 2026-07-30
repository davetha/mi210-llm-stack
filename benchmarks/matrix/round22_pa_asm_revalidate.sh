#!/usr/bin/env bash
# Re-validate pa_fwd_asm, and record why it has never run in a single benchmark.
#
# THE FINDING THAT PROMPTED THIS. Across every serverlog in this project, an ASM
# paged-attention .co has been loaded exactly ZERO times. Only fmha_v3_fwd
# (prefill) ASM ever loads -- 44 times each for the two hd128 kernels. Meanwhile
# `gqa_ratio: 8` appears 92 times in JIT compilations, and gqa8 is precisely what
# pa_bf16_noquant_gqa8_1tg_4w.co serves. Every eligible decode fell back to a HIP
# template while the matching ASM binary sat unused on disk.
#
# So EVERY decode number in docs/28 -- tier 1 through tier 4 -- is non-ASM.
#
# WHY, AND IT IS NOT A GATE. `pa_fwd_asm` occurs exactly six times in the entire
# installed vLLM 0.23.1 tree:
#
#   rocm_aiter_fa.py:1152   a comment
#   _aiter_ops.py:141       a docstring
#   _aiter_ops.py:2536      the wrapper definition
#   _aiter_ops.py:2554      its docstring
#   _aiter_ops.py:2556      `from aiter import pa_fwd_asm` inside the wrapper
#   _aiter_ops.py:2558      the wrapper calling the real function
#
# There is NO call site. Nothing in vLLM ever invokes
# rocm_aiter_ops.pa_fwd_asm(). The AITER FA backend's decode path calls
# torch.ops.aiter.paged_attention_v1 (line 1350) and nothing else. aiter's own
# pa_fwd_asm has no architecture gate either -- it is a plain wrapper.
#
# So this is unreachable through vLLM 0.23 on ANY architecture. Not gfx90a
# specific, not a flag we failed to set, not a misread gate: dead code. That also
# means no environment variable can turn it on, and the earlier plan to "find the
# dispatch condition" was chasing a branch that does not exist.
#
# WHAT THIS ROUND DOES, AND DOES NOT DO. It re-runs the two standalone tests that
# produced the 48/48 result in docs/18, against the CURRENT image, to establish
# whether the kernels still validate before anyone invests in wiring them in.
# docs/16 is explicit that the 48/48 was standalone and that ATOM integration
# faulted, so "it works" has never meant "it works through a server".
#
# It does NOT patch vLLM. Adding a call site means matching kernarg layout, KV
# cache layout and scale handling by hand -- and docs/18 records that the
# original pa_fwd_asm blocker was exactly a stale JIT module whose kernarg layout
# predated the .co files, which ran at full speed and silently discarded every
# store. A wrong patch here does not crash, it produces confident garbage. That
# work needs the standalone result first.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
cd "$BASE"

. "$BIN/wait_for_bench.sh"
bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) round 22: re-validating pa_fwd_asm standalone ==="

IMG=rocm-vllm-aiter-gfx90a:pa256k
REPO=/mnt/llm-storage/bench-matrix

for t in test_pa_fwd_asm_gfx90a.py test_pa_fwd_asm_e2e.py; do
    if [ ! -f "$REPO/tests/$t" ]; then
        echo "!! $REPO/tests/$t not found -- skipping"
        continue
    fi
    echo "--- $t ---"
    timeout 1800 docker run --rm \
        --device /dev/kfd --device /dev/dri --group-add 44 --group-add 991 \
        --security-opt seccomp=unconfined --ipc=host --shm-size 16G \
        -e HSA_NO_SCRATCH_RECLAIM=1 -e AITER_LOG_LEVEL=info \
        -v "$REPO":/repo -v /mnt/llm-storage:/models \
        --entrypoint bash "$IMG" -c "cd /repo && python3 tests/$t" \
        2>&1 | tail -25
    echo "--- $t exit: ${PIPESTATUS[0]} ---"
done

echo "=== $(date -u +%T) round 22 done ==="
cat <<'NOTE'

WHAT A PASS DOES AND DOES NOT MEAN
  PASS  -> the kernels are numerically correct on this image, and wiring a call
           site into vLLM becomes a bounded piece of work with a known-good
           reference to check against.
  FAIL  -> the repatched binaries have regressed against the current ROCm/aiter,
           and docs/18's result no longer holds for this image.

Either way the vLLM-side conclusion is unchanged: there is no call site, so no
configuration of flags will make ASM paged attention run under vLLM 0.23.
NOTE
