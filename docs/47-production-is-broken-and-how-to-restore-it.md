# Production is broken, native multi-GPU restores it, and MoE decode barely notices the quant level

**Date**: 2026-08-02 · Script: `benchmarks/matrix/round47_llamacpp_native.sh` ·
Results: `benchmarks/matrix/results/rd47-*.json`

`docs/46` established that production serves through llama.cpp, not vLLM. This
went looking for optimizations on that side and found a fault first.

## Production cannot start

`/mnt/llm-storage/llama-swap-config.yaml` launches every model from one image:

```
llama-rocm714-rpc:latest
```

**That image does not exist on the host.** The full image list holds
`llama-rocm714`, `llama-rocm714-bench` and `llama-rocm714-forcemmq` — no `-rpc`
build. Every llama-swap model would fail with *"Unable to find image"*, which is
why nothing listens on `:8091`/`:8092` and the llama-swap log ends at
`shutdown complete`.

It cannot be fixed by retagging. Both surviving images were checked and neither
contains `ggml-rpc-server`, only `llama-server`.
`configs/Dockerfile.llama-bench` records the cause: the base was built with
`cmake --build build -t llama-server` — the server target alone. The RPC image
was a separate build, and **no recipe for it exists** in this repo or on the
host.

The repo's committed copy of the config carries the same reference, so this is
not drift between box and git; the dependency is simply gone.

## Native multi-GPU works, and needs no rebuild

llama.cpp splits layers across visible devices by itself. Verified on the
production `coder-next-q4` checkpoint through the repo's existing
`bin/serve_llamacpp.sh` (which has always been the native path):

| evidence | value |
|---|---|
| health | ok after **40 s** |
| GPU0 VRAM used | **25.30 GB** |
| GPU1 VRAM used | **26.92 GB** |
| correctness probe | exact one-word `ACKNOWLEDGED` |

52.2 GB across the pair for a 48.5 GB model plus context — the split is real,
not one card carrying everything.

## Baseline for the restored configuration

Round 47, single stream, `--ctx-size 32768` (so the long-context prompt clamps
to 27,852 tokens — the same prompt size every vLLM arm in this repo used).

| metric | Coder-Next Q4_K_M (48.5 GB) | Qwen3-Next-80B Thinking Q5_K_M (56.7 GB) |
|---|---:|---:|
| cold16k prefill (t/s) | 2558.13 | 2510.47 |
| cold16k ttft (s) | 5.93 | 6.04 |
| longctx prefill (t/s) | 2690.14 | 2647.00 |
| **longctx decode (t/s)** | **73.35** | **72.34** |
| longctx ttft (s) | 9.60 | 9.74 |
| correctness probe | **pass** | **pass** |

**This is a baseline, not a delta.** There is no RPC arm to compare against, so
these numbers describe the restored native configuration only. What RPC was
costing remains unmeasured and cannot be measured without rebuilding that
image.

Two production settings are deliberately not reproduced, and the numbers should
not be read as production-identical: `-c 262144 -np 8` (a throughput
configuration; this measures single-stream latency at `--parallel 1`) and
`-ctk q8_0 -ctv q8_0` (KV quantisation is a separate axis, and
`serve_llamacpp.sh` documents `-ctv q4_1` measuring 11.5× slower on prefill
here).

## The incidental result: MoE decode barely notices the quant level

These two checkpoints are the same architecture class — Qwen3-Next A3B, ~3B
active parameters — at **different quantisations**, 8.2 GB apart in total size.
If decode were bandwidth-bound on active weights, Q4_K_M should stream roughly
20% fewer bytes per token than Q5_K_M and decode correspondingly faster.

Measured difference: **73.35 vs 72.34 t/s — 1.4%**, with prefill 1.6% apart.

So on this hardware, **dropping an A3B MoE from Q5_K_M to Q4_K_M should be
expected to buy ~1–2% of decode, not ~20%.** Decode here is dominated by
something other than active-expert weight streaming — attention, KV traffic,
routing, and cross-card synchronisation are all fixed with respect to the
weight quantisation.

That closes a lead this project raised: *"the 80B is Q5_K_M while the others are
Q4_K_M; on a bandwidth-bound decode that is ~20% fewer bytes"*. The premise is
arithmetically right and the conclusion does not follow, because decode is not
bandwidth-bound on those bytes. **Do not re-download an 80B checkpoint at a
lower quantisation expecting a decode win** — the ceiling is a couple of
percent, and it costs output quality.

Caveat worth stating: these are different fine-tunes (Coder vs Thinking), not
one model at two quantisations, so this is a strong indication rather than a
controlled experiment. A controlled version would quantise one checkpoint two
ways, and given the size of the effect it is probably not worth the download.

## Restoring production

Two options.

**Rebuild the RPC image** — llama.cpp source with `-DGGML_RPC=ON` and the
`rpc-server` target, then retag `llama-rocm714-rpc:latest`. Restores exactly
what was running. No recipe exists, so this is a from-scratch build.

**Move to native multi-GPU** — no rebuild. In `llama-swap-config.yaml`, for
each model: change the image to `llama-rocm714:latest` and delete the
`--rpc 127.0.0.1:500xx,127.0.0.1:500xx` argument. Everything else — `-ngl 999`,
`-c`, `-np`, `-fa on`, `-ctk/-ctv q8_0`, `--jinja`, `-a` — is unchanged. The
per-card `rpc0`/`rpc1` containers are then unnecessary.

The second is recommended: it is a text edit against an image that already
exists, it is verified above to serve both production checkpoints correctly,
and it removes a localhost socket hop from every token. Its one cost is that it
forecloses ever measuring what the RPC hop was worth.
