#!/usr/bin/env bash
# Does AITER's MLA ASM actually run on gfx90a? 30 GB to answer it, not 400.
#
# THE QUESTION. docs/19 records that mla.py gates its ASM paths on gfx942/gfx950,
# which is why the README's MLA "breakthrough" was retracted -- the numbers were
# real but measured the Triton/CK fallback. Every MLA model has been treated as
# ASM-ineligible since.
#
# READING THE INSTALLED SOURCE DOES NOT SUPPORT THAT:
#
#   - The gfx942 / gfx950 / gfx1250 checks in aiter/mla.py are PERF-TUNING
#     branches -- wg_per_split, num_warps, block sizes -- not availability gates.
#   - aiter/jit/core.py:882 reads `if (get_gfx() not in ("gfx942", "gfx90a")`,
#     i.e. gfx90a is explicitly inside an allowed set.
#   - 11 mla .co objects EXIST in the gfx90a tree, including
#     mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co, which matches the
#     aiter.mla_decode_stage1_asm_fwd call site in mla.py.
#
# That is the same shape as three claims this project already overturned:
# fmha_v3_fwd ("unreachable" -> 80/80 exact), pa_fwd_asm ("gfx942 binaries cannot
# run on gfx90a" -> 48/48 exact, the blocker was a stale JIT module), and
# VLLM_ROCM_USE_AITER_MOE ("fails gate" -> never gated, we disabled it by hand).
# This would be the fourth. It is equally possible the objects are present and
# simply never selected -- which is why this measures rather than argues.
#
# WHY DeepSeek-V2-Lite. 16B total / 2.4B active, MLA, ~31 GB in bf16. docs/15
# already lists "validate DeepSeek-V2 Lite 16B (MoE + MLA)" as a next step that
# was never done. And the head count matters: it has 16 attention heads, which
# matches the QH16 objects on disk --
#   mla_a16w16_qh16_m16x4_n16x1_coex0_mask1.co
#   MLA_A16W16_1TG_4W_32mx1_16nx1_Coex0_Msk1_QH16.co
#
# TP=1 ON PURPOSE. Tensor parallelism splits query heads across ranks; at TP=2
# each rank would see 8 heads and no longer match a QH16 kernel. 31 GB fits one
# 64 GB card with room, so TP=1 costs nothing and removes the confound. A TP=2
# arm runs afterwards precisely to see whether splitting heads loses the ASM.
#
# WHAT THIS DECIDES. If MLA ASM loads, every MLA model is re-opened -- including
# GLM-5.2, whose Int4-Int8Mix build is ~400 GB and would otherwise not be worth
# the download. If it does not load, that is a constraint for the whole MLA
# family and the 400 GB is saved.
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
DEST=$BASE/dsv2-lite
cd "$BASE"

. "$BIN/wait_for_bench.sh"

# Fetch before claiming -- rounds 12 and 14 both held the bench lock through a
# 100+ GB download and idled the GPUs for 35 and 40 minutes respectively.
if [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST" 2>/dev/null)" ]; then
    echo "=== $(date -u +%T) fetching DeepSeek-V2-Lite-Chat (~31 GB), lock NOT held ==="
    python3 "$BIN/fetch_model.py" deepseek-ai/DeepSeek-V2-Lite-Chat "$DEST" \
        --connections 1 --concurrent 4 \
        || { echo "!! fetch failed"; exit 1; }
fi

bench_claim
echo "=== $(date -u +%T) waiting for other bench work ==="
bench_wait_for_others
echo "=== $(date -u +%T) starting round 21 (MLA ASM probe) ==="

# VLLM_ROCM_USE_AITER_MLA is NOT set by serve_vllm_aiter.sh, so it has always
# taken vLLM's default. Set it explicitly in both directions rather than relying
# on that default, so the A/B is a real comparison and not an assumption about
# what the default is.
run() {  # run <label> <tp> <extra serve args...>
    local label="$1" tp="$2"; shift 2
    echo "--- $label  tp=$tp  [env: ${ARM_ENV:-none}] ---"
    VLLM_EXTRA_ENV="${ARM_ENV:-}" \
    LONGCTX_TOKENS=28000 ARM_TIMEOUT=3600 READY_TIMEOUT=1800 \
        "$BIN/run_arm.sh" "$label" 16B bf16 vllm-aiter "$DEST" \
        --tensor-parallel-size "$tp" \
        --max-model-len 32768 \
        --max-num-batched-tokens 8192 \
        --no-enable-prefix-caching \
        --trust-remote-code \
        "$@" \
        || echo "!! $label failed (recorded)"
}

ARM_ENV="-e VLLM_ROCM_USE_AITER_MLA=1"
run dsv2lite-mla-asm-tp1 1

ARM_ENV="-e VLLM_ROCM_USE_AITER_MLA=0"
run dsv2lite-mla-off-tp1 1

# TP=2 splits 16 query heads to 8 per rank. If the ASM loads at TP=1 and not
# here, head count is the selection criterion and that is worth knowing before
# planning any large MLA model, which would need TP=2 to fit at all.
ARM_ENV="-e VLLM_ROCM_USE_AITER_MLA=1"
run dsv2lite-mla-asm-tp2 2

echo "=== $(date -u +%T) round 21 done ==="
echo
echo "======================= THE ACTUAL VERDICT ======================="
# The .co load line is the proof, not the backend-selection line. docs/28 makes
# this rule explicit because an earlier result in this repo credited AITER for a
# run that never loaded a single kernel.
for l in dsv2lite-mla-asm-tp1 dsv2lite-mla-off-tp1 dsv2lite-mla-asm-tp2; do
    log="logs/$l.serverlog"
    if [ ! -f "$log" ]; then echo "  $l: no serverlog"; continue; fi
    n=$(grep -c "LoadKernel.*mla" "$log" 2>/dev/null || echo 0)
    if [ "$n" -gt 0 ]; then
        echo "  $l: *** MLA ASM LOADED ($n kernels) ***"
        grep -o "LoadKernel.*mla[^ ]*\.co" "$log" | sort -u | head -4 | sed "s/^/      /"
    else
        echo "  $l: no MLA .co loaded (Triton/CK fallback)"
        grep -oiE "mla[a-z_]*backend[^,]*|Using .* MLA" "$log" 2>/dev/null | sort -u | head -2 | sed "s/^/      /"
    fi
done
echo
echo "--- throughput ---"
for l in dsv2lite-mla-asm-tp1 dsv2lite-mla-off-tp1 dsv2lite-mla-asm-tp2; do
    for w in cold16k longctx; do
        f="results/$l-$w.json"
        [ -f "$f" ] && python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('  %-24s %-8s prefill=%9.1f decode=%7.2f' % (sys.argv[2], sys.argv[3],
    d.get('implied_prefill_tps_median') or 0, d.get('decode_tps_median') or 0))" "$f" "$l" "$w"
    done
done
