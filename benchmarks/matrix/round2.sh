#!/usr/bin/env bash
# Round 2 of the quantization matrix: the experiments docs/26 and docs/27
# predict but do not measure.
#
# Run on the MI210 host:
#   cd /mnt/llm-storage/bench-matrix && ./bin/round2.sh 2>&1 | tee logs/round2.log
#
# SERIAL BY CONSTRUCTION. Round 1 lost hours to two circular deadlocks built out
# of eight scripts that waited on each other. Every arm here runs in sequence in
# one process; a failing arm records status=FAILED and the next one starts. There
# is nothing to coordinate and nothing that can wait on anything else.
#
# Each experiment names the claim it tests, so a result that contradicts the doc
# is recognisable as such rather than being read as noise.
set -uo pipefail

BASE=/mnt/llm-storage/bench-matrix
BIN=$BASE/bin
MODELS=$BASE            # models sit directly under bench-matrix/, matching round 1
mkdir -p "$BASE/results" "$BASE/logs"

fetch() {  # fetch <dest-dir> <hf-repo>
    local dest="$1" repo="$2"
    [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ] && { echo "have $repo"; return 0; }
    echo "### fetching $repo"
    # --connections 1: the Xet CDN signs per-byte-range URLs, so aria2 --split
    # produces holey files that look complete. Round 1 lost a 45 GB download to
    # this before the cause was found.
    python3 "$BIN/fetch_model.py" "$repo" "$dest" --connections 1 --concurrent 8
}

banner() { echo; echo "###############################################################"; echo "# $*"; echo "###############################################################"; }

# ---------------------------------------------------------------------------
# E0  Isolate TP=2 from the model. Cheapest and most overdue arm in the set.
#
# Every tier-2 headline is confounded: the 80B awq-int8 row (6,679 tok/s) is the
# ONLY arm in the whole matrix that ran on two cards, so it was never comparable
# to the TP=1 rows beside it. docs/24 says so, but saying so is not the same as
# measuring it, and the matrix still has no arm that varies TP alone.
#
# The 80B cannot supply the control -- 40.95 GiB per rank means ~82 GiB total,
# which will not fit on one 64 GB card. So run the control from the other side:
# the 30B w8a8 at TP=2 against its own measured TP=1 row (3.20 s / 4,739 tok/s).
# Same weights, same flags, same engine, TP the only variable.
#
# 30 GB and a 123 s load, so this costs minutes. It should have been in round 1.
# ---------------------------------------------------------------------------
banner "E0  30B w8a8 at TP=2 -- the TP control the matrix never had"
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t35-w8a8-tp2 35B w8a8 vllm-aiter "$BASE/t35-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E1  Does W8A8 beat W8A16 on the SAME 80B architecture?
#
# docs/26 predicts yes. The tier-2 measurement used
# cyankiwi/...-AWQ-8bit, which despite the name is compressed-tensors
# pack-quantized with input_activations=null -- weight-only, so it never reached
# v_mfma_i32_16x16x16i8. This repo is int-quantized with 8-bit dynamic per-token
# activations, and is the same model family at the same TP.
#
# Confounder to respect: the W8A16 arm was Thinking, this is Instruct. Different
# post-training, same architecture. Report it as such.
# ---------------------------------------------------------------------------
banner "E1  80B W8A8 (tests docs/26 prediction: W8A8 > W8A16 on Qwen3-Next)"
fetch "$MODELS/t80-w8a8" RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t80-w8a8 80B w8a8 vllm-aiter "$MODELS/t80-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E2  A model that hits BOTH fast paths at once.
#
# Nothing in round 1 does. The 30B W8A8 got INT8 MFMA + hd128 ASM but is a MoE
# with a small active set; Qwen3-Next got INT8 but head_dim=256 so no ASM at all
# (docs/25 item 7). Llama-3.3-70B W8A8 is dense, head_dim=128, 131k native:
# INT8 GEMM *and* AITER ASM attention, simultaneously, verified from config.json.
#
# Verify from the serverlog, not from the timing:
#   grep -c 'fwd_hd128_bf16.*\.co' logs/t70-w8a8.serverlog   -> must be > 0
# ---------------------------------------------------------------------------
banner "E2  Llama-3.3-70B W8A8, dense hd128 (tests docs/26 'sweet spot')"
fetch "$MODELS/t70-w8a8" RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t70-w8a8 70B w8a8 vllm-aiter "$MODELS/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E3  Is the ASM win additive with W8A8, or does it overlap?
#
# The +12.8% in docs/24 was measured on AWQ-Int4. The claim in docs/27 -- that
# the ASM kernels are bf16 attention and therefore quantization-independent --
# predicts the same delta on a W8A8 model. Same model, same flags, AITER off then
# on. This is the decisive A/B and it has never been run on a W8A8 arm.
# ---------------------------------------------------------------------------
banner "E3  AITER A/B on W8A8 (tests docs/27: ASM is quantization-independent)"
LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t70-w8a8-noaiter 70B w8a8 vllm "$MODELS/t70-w8a8" \
    --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching

# ---------------------------------------------------------------------------
# E4  W4A8-int8: which kernel does vLLM actually pick on gfx90a?
#
# docs/27 audits the mixed-precision registry statically and concludes every
# ROCm-reachable kernel rejects int8 activations, so this must either fail to
# find a kernel or silently fall back to a w4a16 path. A 1.5B model makes that a
# two-minute question instead of an hour.
#
# The result is the LOG, not the throughput. Read it:
#   grep -iE 'kernel|w4a8|marlin|triton|cannot implement' logs/probe-w4a8.serverlog
# A clean start is NOT confirmation the int8 path ran -- that is the exact
# mistake docs/24 documents for ROCM_AITER_FA. Confirm which kernel was chosen.
# ---------------------------------------------------------------------------
banner "E4  W4A8-int8 kernel probe (tests docs/27: no ROCm kernel takes int8 acts)"
fetch "$MODELS/probe-w4a8" alishafique/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a8int8-llmcompressor
READY_TIMEOUT=600 "$BIN/run_arm.sh" probe-w4a8 1.5B w4a8-int8 vllm-aiter "$MODELS/probe-w4a8" \
    --max-model-len 32768 --no-enable-prefix-caching
echo "--- kernel selection for the W4A8 probe ---"
grep -iE "kernel|w4a8|marlin|triton|machete|cannot implement|not supported" \
    "$BASE/logs/probe-w4a8.serverlog" 2>/dev/null | head -40

# ---------------------------------------------------------------------------
# E5  MoE tuning, with the variable vLLM actually reads.
#
# Round 1 set MOE_CONFIG_DIR, which vLLM ignores; the correct name is
# VLLM_TUNED_CONFIG_FOLDER. That mistake would have produced a confident
# "tuning does not help" from a run where the config was never loaded.
#
# Bounded to batch sizes 1 and 64. The full sweep is unbounded -- it grew
# 704 -> 2,660 -> 4,990 -> 7,810 candidates and eats a day per tier.
# ---------------------------------------------------------------------------
banner "E5  MoE tuning with VLLM_TUNED_CONFIG_FOLDER (round 1 used the wrong var)"
"$BIN/tune_moe_targeted.sh" || echo "E5 tuning did not complete -- continuing"

# ---------------------------------------------------------------------------
# E6  bf16 baseline. Still the only missing cell in tier 1.
#
# TP=2 is mandatory: 61 GB will not share a 64 GB card with a KV cache. Expect
# ~697 s/shard in _load_w13 (docs/25 item 1) -- roughly 3 hours before it serves.
# That is why it is here and not first.
# ---------------------------------------------------------------------------
banner "E6  30B bf16 baseline at TP=2 (slow: ~3h load, run last of the vLLM arms)"
# Guarded: an out-of-band bf16 arm was already running when round 2 was written,
# and re-running it would burn three hours to reproduce a cell we have. Only the
# FAILED marker from round 1 is treated as absent.
if ls "$BASE"/results/t35-bf16-cold16k.json >/dev/null 2>&1; then
    echo "E6 skipped: t35-bf16-cold16k.json already exists"
else
    fetch "$MODELS/t35-bf16" Qwen/Qwen3-30B-A3B-Thinking-2507
    READY_TIMEOUT=14400 LONGCTX_TOKENS=110000 "$BIN/run_arm.sh" t35-bf16 35B bf16 vllm-aiter "$MODELS/t35-bf16" \
        --tensor-parallel-size 2 --max-model-len 131072 --no-enable-prefix-caching
fi

# ---------------------------------------------------------------------------
# E7  GLM-4.6 GGUF, with the -ngl / --n-cpu-moe interaction handled correctly.
#
# Two round-1 attempts failed on this and the mechanism is now understood:
# common_fit_params() is all-or-nothing, so setting EITHER -ngl or --n-cpu-moe
# disables auto-fit for the whole model. They must be set together.
#
# The third attempt SUCCEEDED out of band, with no placement flags at all:
# 8.51 t/s decode at 25.8k, 12.83 t/s at short context, correctness PASS. But
# read what it actually did before reading it as an offload result -- llama.cpp's
# auto-fit put 135.57 GB of a 139 GB model on the two cards, so only ~3 GiB was
# ever CPU-resident. That is a *capability* result (a 357B model runs almost
# entirely in 128 GB of VRAM at IQ3_XS) and NOT a measurement of CPU offload.
#
# Which makes the sweep below the real test, with the auto-fit run as its N≈0
# anchor. Forcing 30/45/60 expert layers to CPU is strictly more offload than
# auto-fit chose, so decode should degrade -- and docs/26 predicts roughly
# LINEARLY in the pinned-layer count. If the curve is not linear, the bandwidth
# model is wrong and the doc must say so. This is a real test, not a formality.
# ---------------------------------------------------------------------------
# FIXED: this pointed at $MODELS/glm46-q4km, which does not exist -- all three
# arms died instantly with "gguf_init_from_file: failed to open ... (No such
# file or directory)". The fetch would not have saved it either: bartowski's
# repo carries every quant, and without --include it pulls terabytes.
#
# Using the IQ3_XS already on disk is also the BETTER experiment. The auto-fit
# run that serves as the N~0 anchor was IQ3_XS, so this varies only
# --n-cpu-moe against it. Comparing a Q4_K_M sweep to an IQ3_XS anchor would
# have confounded quant with placement -- the same mistake as the TP=2 rows.
banner "E7  GLM-4.6 IQ3_XS + --n-cpu-moe sweep (tests docs/26 bandwidth model)"
for N in 30 45 60; do
    echo "--- n-cpu-moe=$N ---"
    ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
        "$BIN/run_arm.sh" "glm46-iq3xs-ncmoe$N" 400B iq3_xs llamacpp "$MODELS/glm-gguf-iq3xs" \
        --ctx-size 32768 -ub 2048 --flash-attn on -ngl 999 --n-cpu-moe "$N"
done

# ---------------------------------------------------------------------------
# E8  The tier-4 hypothesis: the problem was never the bit width, it was 3 GiB.
#
# IQ3_XS is 139 GB and usable VRAM is 135.57 GB, so auto-fit left ~3 GiB on the
# CPU -- and prefill collapsed to 181 t/s while decode stayed a healthy 8.5.
# That asymmetry is the tell: a bandwidth limit would slow both. A small
# CPU-resident fraction that every token must pass through slows prefill
# specifically, because prefill pushes ALL prompt tokens through it.
#
# So the prediction is that dropping under the VRAM line matters far more than
# the two bits of precision it costs. unsloth's UD-IQ2_M is 121.9 GB -- fits with
# ~14 GB left for KV -- and its dynamic scheme keeps attention and shared experts
# at higher precision while pushing only routed experts down, which is the right
# place to lose bits on a MoE.
#
# Falsifiable: if prefill stays near 181 t/s, the CPU-residency explanation is
# wrong and the cost is somewhere else entirely. Either way this is the decisive
# experiment for the tier, and it is worth more than another point on the sweep.
# ---------------------------------------------------------------------------
banner "E8  GLM-4.6 UD-IQ2_M, fully GPU-resident (tests: prefill collapse is the 3 GiB)"
# --include is mandatory here: the repo carries every quant from TQ1_0 to BF16,
# which is several terabytes. Without it this pulls the lot.
if [ ! -d "$MODELS/glm-gguf-ud-iq2m" ] || [ -z "$(ls -A "$MODELS/glm-gguf-ud-iq2m" 2>/dev/null)" ]; then
    python3 "$BIN/fetch_model.py" unsloth/GLM-4.6-GGUF "$MODELS/glm-gguf-ud-iq2m" \
        --include UD-IQ2_M --connections 1 --concurrent 8
fi
ARM_TIMEOUT=7200 READY_TIMEOUT=3600 \
    "$BIN/run_arm.sh" glm46-udiq2m 400B ud_iq2_m llamacpp "$MODELS/glm-gguf-ud-iq2m" \
    --ctx-size 32768 -ub 2048 --flash-attn on
echo "--- did it fit? compare against IQ3_XS's 135.57 GB / 181 t/s prefill ---"
grep -aE "offloaded|buffer size|CPU_Mapped" "$BASE/logs/glm46-udiq2m.serverlog" 2>/dev/null | head -10

# ---------------------------------------------------------------------------
# E9  The payoff for the 256k patch: does the decode cliff actually go away?
#
# docs/23 measured the cliff at 10x, and the matrix has the concrete number:
# 30B AWQ at --max-model-len 262144 decodes 0.7485 t/s at 241k, reproducibly
# (two reps, 0.7468 / 0.7502). Configured at 131072 the same engine does 33.8
# t/s at 101k. The gap is the Triton fallback baked in at graph capture.
#
# configs/extend_rocm_pa_256k_gfx9.py extends the reduction dispatch to
# npar_loops=16, and tests/test_rocm_pa_256k.py has VERIFIED the numerics --
# custom vs Triton, 4-7e-3 at 139k/196k/262k, with the control identical on both
# builds and the 266k ceiling still declining. So a throughput number from this
# build is now meaningful rather than a faster wrong answer.
#
# Same model, same flags, same 262144 config as the 0.7485 t/s row. The only
# variable is the image. Anything near 0.75 means the patch did not reach the
# decode path; anything near 30 means the cliff is gone.
# ---------------------------------------------------------------------------
banner "E9  256k decode cliff, patched build (baseline to beat: 0.7485 t/s @ 241k)"
if docker image inspect rocm-vllm-aiter-gfx90a:pa256k >/dev/null 2>&1; then
    # run_arm reads the image from serve_vllm_aiter.sh; VLLM_IMAGE overrides it.
    VLLM_IMAGE=rocm-vllm-aiter-gfx90a:pa256k \
    ARM_TIMEOUT=7200 LONGCTX_TOKENS=262144 \
        "$BIN/run_arm.sh" t35-awq-256k-patched 35B awq vllm-aiter "$BASE/t35-awq" \
        --max-model-len 262144 --no-enable-prefix-caching
    echo "--- compare: stock at this config decoded 0.7485 t/s at 241,583 tokens ---"
else
    echo "E9 skipped: rocm-vllm-aiter-gfx90a:pa256k not built"
fi

# ---------------------------------------------------------------------------
# E10  Re-run the loader probes on an idle box.
#
# All three were run while the bf16 arm held both cards, and round 3 came back
# incoherent: `empty + copy_` interleaved measured 12,000x FASTER than the copy
# alone, and the slow figure landed on 1001.29 / 1000.98 ms -- essentially
# exactly 1.000 s across independent runs. Costs that scale with data do not do
# that. Whatever was measured was contention, not the loader.
#
# This runs last precisely because that is when nothing else holds a GPU. What
# it needs to settle: line 771 sustains 21.7 GB/s unobstructed, yet the real
# TP=2 load moves ~2.2 MB/s per rank. That gap is still unexplained, and three
# hypotheses (storage, strided copies, allocation) are now eliminated.
# ---------------------------------------------------------------------------
banner "E10  loader probes on an idle box (round 3 was confounded by bf16)"
for p in probe_loader.py probe_loader2.py probe_loader3.py probe_loader4.py; do
    echo "--- $p ---"
    docker run --rm --device /dev/kfd --device /dev/dri \
        --group-add 44 --group-add 991 --ipc host --shm-size 8g \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -v /mnt/llm-storage:/models:ro -v "$BIN":/bin2:ro \
        --entrypoint python3 rocm-vllm-aiter-gfx90a:latest "/bin2/$p" 2>&1 \
        | grep -vE "^INFO|importing.py"
done

# ---------------------------------------------------------------------------
# E11  Is the 3.6-hour load fixed by one flag?
#
# py-spy --native shows the loader blocked in hsakmt_ioctl -- a driver ioctl,
# 8/8 samples. The implied mechanism is per-transfer registration of each fresh
# mmap'd expert tensor. vLLM's default path is mmap-backed (safe_open +
# get_tensor); its `eager` strategy is not:
#
#   with open(st_file, "rb") as f:
#       state_dict = load(f.read())    # whole shard, one heap allocation
#
# If the mechanism is right, eager registers once per shard instead of once per
# tensor, and this is a FLAG rather than a patch.
#
# This does not need a full arm. Shard times are ~815 s each at present, so two
# shards is a decisive read: if shard 1 lands anywhere near 700 s the idea is
# dead, and if it lands in tens of seconds it is a ~100x deployment win on every
# bf16 and GPTQ model. Killed after two shards either way.
# ---------------------------------------------------------------------------
banner "E11  bf16 load with --safetensors-load-strategy=eager (baseline 815 s/shard)"
docker rm -f bench-eager-probe >/dev/null 2>&1 || true
VLLM_IMAGE=rocm-vllm-aiter-gfx90a:latest \
    "$BIN/serve_vllm_aiter.sh" "$BASE/t35-bf16" bench-eager-probe 8123 \
    --tensor-parallel-size 2 --max-model-len 131072 \
    --safetensors-load-strategy=eager --no-enable-prefix-caching \
    >/dev/null 2>&1 || echo "eager probe would not start"
echo "watching shard timings (kill after 2)..."
for _ in $(seq 1 90); do
    line=$(docker logs bench-eager-probe 2>&1 | grep -aoE "Completed \| [0-9]+/16 \[[0-9:]+<[0-9:]+, [0-9.]+s/it" | tail -1)
    [ -n "$line" ] && echo "  $line"
    echo "$line" | grep -q "2/16" && break
    docker ps --filter "name=^bench-eager-probe$" --format '{{.Names}}' | grep -q . || {
        echo "  container exited early"; break; }
    sleep 40
done
echo "--- eager result above; mmap baseline was 697 / 774 s for shards 1 and 2 ---"
docker logs bench-eager-probe > "$BASE/logs/eager-probe.serverlog" 2>&1 || true
docker rm -f bench-eager-probe >/dev/null 2>&1 || true

banner "round 2 complete"
python3 "$BIN/summarize_results.py" "$BASE/results"
