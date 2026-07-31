# Tuning `fused_moe` for MI210 — and the filename that cost 7.5 GPU-hours

**Date**: 2026-07-31 · **Hardware**: 2× MI210 (gfx90a/CDNA2)
**Model**: `Qwen3-30B-A3B-Thinking-2507` W8A8 (`compressed-tensors`), TP=2
**Config produced**: `benchmarks/matrix/moe-configs-mi210/`
**Rounds**: 33 (tuning), 34 (A/B)

vLLM ships **317** tuned `fused_moe` configs — MI300X, MI308X, MI325X, MI350X,
MI355X, and a long tail of NVIDIA parts. **None is for an MI210**, so gfx90a
tile selection falls out of heuristics. `docs/33` measured the consequence:
decode shapes select `16x16x16bf16_1k` at 191–196 MACs per issued instruction
while larger shapes select `32x32x8bf16_1k` at 485–497, with ~450–500
instructions of fixed overhead either way.

This is the first time the supported tuner has run on this hardware at all.

---

## 1. The filename, and a retraction

**The runtime opens this file:**

```
E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
```

**With** the `dtype` tag, for a W8A8 checkpoint. N=384 because
`moe_intermediate_size` is 768 and the serving arm is TP=2; E=128 from
`num_experts`.

An earlier revision of round 33 argued at length for the *untagged* name and was
wrong. The reasoning is worth recording because it looked airtight:

> `fused_moe.py:1595` calls `_get_config_dtype_str` with `use_fp8_w8a8`,
> `use_int8_w8a16` and `use_int4_w4a16` — there is **no `use_int8_w8a8`**
> parameter, and `config.py:36-66` has no branch for one. So a W8A8 MoE falls
> through to `return None`, and `get_config_file_name` renders an empty dtype
> selector.

Every one of those observations is true. The conclusion is still wrong, because
**the `compressed-tensors` W8A8 serving path sets `use_int8_w8a16=True`** — so
the absence of a `use_int8_w8a8` parameter means the opposite of how it reads.

`docs/25` item 3b, the header of `tune_moe_w8a8_v4.sh`, and the docstring of
`configs/fix_benchmark_moe_int8_w8a16.py` all said `int8_w8a16` and were right.
Three files agreeing with each other is not the same as checking what the
process does.

**How it was caught.** Round 34 asserts on the config load *before* reading any
number, and vLLM says exactly which paths it tried:

```
Using default MoE config. Performance might be sub-optimal! Config file not
found at .../E=128,N=384,device_name=AMD_Instinct_MI210,dtype=int8_w8a16.json
```

Without that assertion the round would have reported a clean-looking *"tuning
doesn't help on MI210"*, backed by two identical runs of the same code path.
`install_moe_config.sh` names the hazard precisely: *"the tuned arm would have
reported 'no improvement' from a config that was never loaded — a false negative
indistinguishable from a real one."*

The wrongly-named config is archived on the box under
`moe-configs-mi210/wrong-dtype-auto-DO-NOT-USE/` and deliberately **not
renamed**: `--dtype auto` benchmarked a different kernel, and its schema differs
(it carries `matrix_instr_nonkdim` and `kpack`, which the `int8_w8a16` config
does not have at all). A config generated under the wrong scheme is worse than
none.

---

## 2. What the tuner produced

Six batch sizes, covering decode and modest concurrency:

| M | BLOCK M/N/K | GROUP_M | warps | stages | waves_per_eu |
|---:|---|---:|---:|---:|---:|
| 1 | 16/32/256 | 1 | 2 | 2 | 4 |
| 2 | 16/64/256 | 1 | 4 | 2 | 0 |
| 4 | 16/64/256 | 1 | 4 | 2 | 0 |
| 8 | 16/64/256 | 1 | 4 | 2 | 2 |
| 16 | 16/64/256 | 4 | 4 | 2 | 2 |
| 32 | 16/32/256 | 4 | 2 | 2 | 2 |

`BLOCK_SIZE_M = 16` and `BLOCK_SIZE_K = 256` throughout; the search moved
`BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps` and `waves_per_eu`.

**M=64 and M=128 are absent** — see §3 — and that turns out to be fatal. See §5.

---

## 3. Cost, and what it costs *you* to repeat this

**`configs/fix_benchmark_moe_int8_w8a16.py` is mandatory.** Without it the
`int8_w8a16` path dies in `const_data_ptr` before measuring anything — the
failure `docs/25` 3b recorded as "not fixable from the invocation", correctly,
and which blocked this work for weeks. Found and fixed by
[Andrei-Dr](https://github.com/Andrei-Dr).

**Tuning time roughly doubles per batch size**, measured:

| M | wall clock | outcome |
|---:|---:|---|
| 1–4 | ~12 min each | OK |
| 8 | ~28 min | OK |
| 16 | ~60 min | OK |
| 32 | **~89 min** | OK |
| 64 | >90 min | ✗ timeout |
| 128 | ~5 h projected | ✗ abandoned |

Budget accordingly. A 45-minute cap failed on **every** size from 8 upward — six
consecutive timeouts, ~4.5 GPU-hours for nothing.

**Tune one batch size per invocation and merge between them.**
`benchmark_moe.py:996` builds `{batch_size: config}` across the whole list and
`save_configs` writes **one file at the end**, so a crash anywhere discards
everything. That is not hypothetical: a Ray `BenchmarkWorker` died mid-run —

```
Worker exit type: SYSTEM_ERROR ... connection error code 2. End of file.
```

— taking three completed batch sizes with it. Ray names the OOM killer as the
first likely cause and **that was not it**: no OOM in `dmesg`, 461 GB free, no
amdgpu fault, no HIP error. Cause still unknown.

Note the symmetric trap, which is why the naive fix fails: because the tuner
rewrites the *whole* file each run containing only the size it just tuned,
invoking it per size **without merging** makes each run overwrite the last.
`tune_moe_merge.sh` exists for exactly this and predates all of it.

Total spend: ~7.5 GPU-hours on the wrongly-named config, ~5.5 on the right one.

---

## 4. What round 34 can and cannot show

The A/B is built so it cannot flatter itself:

- **It asserts the load first.** A tuned arm that logs `Using default MoE config`
  fails the round rather than reporting a number.
- **Prefill is a built-in control.** Only M=1–32 are tuned; prefill runs chunked
  at `-ub 2048`, so it takes the same heuristic in *both* arms and must not move.
  If prefill moves too, that is drift and the run should be rejected, not
  reported as a win.
- `VLLM_TUNED_CONFIG_FOLDER` rather than baking into the image, so it is
  reversible and needs no rebuild. `get_moe_configs` checks that folder *before*
  the shipped configs.

---

## 5. The result: the tuned config makes everything slower

Round 34, W8A8 TP=2, load verified in both arms:

| workload | metric | stock | tuned | factor |
|---|---|---:|---:|---:|
| cold 16k | prefill | 5,945.95 | 4,672.43 | **0.786×** |
| longctx | prefill | 4,667.07 | 3,824.84 | **0.820×** |
| longctx | decode | 50.88 | 47.05 | **0.925×** |

**Do not install this config.** It is committed as a record of what the tuner
produces on this hardware, not as a recommendation.

### Why prefill moved, and why §4's "control" was not one

`fused_moe.py:1328`:

```python
config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
```

**The nearest tuned entry wins, and there is no fallback.** Once a tuned file
exists for a given `(E, N, dtype)`, it captures *every* M. With only M=1–32
present, a chunked prefill at M≈2048 is served by the **M=32** config — a tile
tuned for a shape 64× smaller. That is the 20% prefill loss, and it is a
mechanism, not noise.

So this document's own §2 previously claimed "uncovered shapes fall back to
exactly the heuristics they use today", and round 33's header and PRs #48/#49
claimed "a partial config is not a broken one". **All of that is wrong.** There
is no such fallback.

> **A partial `fused_moe` config is worse than none.** Either cover the full M
> range the workload touches, or ship nothing. This is the single most useful
> thing on this page and it is not documented upstream.

It also means round 34's design was flawed even though its assertion held.
Prefill was presented as an untouched control; it was in fact the most heavily
affected variable. The load assertion caught the *previous* error and this one
slipped past it, because the check verified *that* a config loaded, not *which
shapes it would capture*.

### Decode, at a size that was tuned

M=1 **is** in the config, and decode still lost 7.5%. So this is not only the
clamping artefact — at least one genuinely tuned shape underperformed the
shipped heuristic. The likely reason is that `benchmark_moe.py` times the MoE
kernel in isolation, while the serving path runs it interleaved with attention,
KV traffic and RCCL collectives on the same device; a tile that wins standalone
need not win in that context.

### What this says about `docs/33`

`docs/33` measured decode selecting `16x16x16bf16_1k` at 191–196 MACs/instruction
against 485–497 for the `32x32x8` tile larger shapes get, and read that as decode
sitting on the wrong side of a 2.5× spread. **This round found no way to collect
it.** The supported tuner, given the correct dtype and a real search, produced
nothing that beats the heuristic at decode shapes.

That is consistent with the spread describing a *ceiling reachable only at large
M* rather than headroom available at M=1 — at M=1 a 32-row tile wastes 31/32 of
the M dimension regardless of how it is configured. `docs/33`'s measurement
stands; the inference that it represented recoverable decode headroom does not.

### Worth trying, if someone returns to this

- **Tune the full M range**, including 512–4096. Budget ~5 h/size above M=64 on
  this hardware (§3), so this is an overnight job, and it is the only way to
  find out whether the losses above are entirely the clamping artefact.
- **Tune in situ.** A config selected by standalone kernel timing is not
  obviously the one that wins under a live serving mix.
