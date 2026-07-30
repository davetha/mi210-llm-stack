# The 3.4-hour load has a shipped fix: `--load-format sharded_state`

**Date**: 2026-07-30 · **Found by** [Andrei-Dr](https://github.com/Andrei-Dr) ·
Verified end-to-end against the installed vLLM (`0.23.1.dev1+g9ddef7117`).

`docs/25` item 1 is the longest investigation in this repo: bf16 MoE checkpoints
load at **0.0023 GiB/s**, 12,366 s for a 30B model that takes 60 s as W8A8 — a
207× gap, traced by `py-spy --native` to `hsakmt_ioctl` under `_load_w13`, with
three attempted patches all refuted. The whole effort went into *fixing* the
loader. **The question never asked was whether a different loader backend skips
it entirely.** One does, and it ships.

## Why it cannot hit the slow path

`model_executor/model_loader/sharded_state_loader.py`, read from the installed
tree — the entire weight-load body:

```python
state_dict = self._filter_subtensors(model.state_dict())
for key, tensor in self.iterate_over_files(filepaths):
    param_data = state_dict[key].data
    ...
    param_data.copy_(tensor)
    state_dict.pop(key)
```

**No `weight_loader` call. No `_load_w13`. No `moe_wna16_weight_loader`.** It
walks the rank's pre-sharded files and does one flat `copy_` into an
already-allocated runtime parameter. The per-expert iteration that produced the
ioctl storm has no analogue here — there is no per-expert fresh mmap to register.

That also explains why this dodges the *other* pathology in `docs/25`: the
super-linear +18%-across-six-shards trend, which scales with loaded-tensor count.
A flat copy per pre-existing parameter has no such term.

## Verified present in this install

| piece | location |
|---|---|
| load path | `sharded_state_loader.py`, flat `param_data.copy_(tensor)` |
| `--load-format sharded_state` | documented in `config/load.py:52` |
| save API | `ShardedStateLoader.save_model` (line 179) |
| engine hook | `EngineCore.save_sharded_state` (`v1/engine/core.py:800`) |
| **example script, in the image** | `/app/vllm/examples/features/sharded_state/save_sharded_state_offline.py` |

It also emits its own verification line, `"Loading weights took %.2f seconds"`,
so success is measurable without inference.

## Workflow

```bash
# once — pays the full 3.4 h through the normal loader
python /app/vllm/examples/features/sharded_state/save_sharded_state_offline.py \
    --model /models/bench-matrix/t35-bf16 \
    --tensor-parallel-size 2 \
    --output /models/bench-matrix/t35-bf16-sharded

# thereafter
vllm serve /models/bench-matrix/t35-bf16-sharded \
    --load-format sharded_state --tensor-parallel-size 2
```

## Constraints, from the source rather than assumed

- **Pre-sharded only.** The snapshot must be generated first; there is no
  any-to-any conversion at load time.
- **Bound to the TP layout.** The file glob embeds `rank=`, so reload must use the
  identical `--tensor-parallel-size`. No resharding.
- **Engine-version-specific.** It snapshots `model.state_dict()` *after*
  `process_weights_after_loading`, i.e. runtime tensors, not checkpoint tensors.
  Regenerate after a vLLM bump.
- **Costs disk.** A second full copy of the weights per TP layout.

## Why this matters more than it looks

`docs/28` currently advises against bf16 for anything that restarts, on the
strength of "12,366 s against 60 s". That rule exists *because* of the loader, not
because of bf16 — bf16 is the **fastest inference** configuration measured at
TP=2 (7,578 t/s prefill, 62.6 t/s decode at tier 1). If the load becomes a
one-time cost, the rule inverts for any long-lived server.

It also unblocks the `fmoe` ASM test (`docs/33`): the only configuration that can
reach those kernels is a **bf16 MoE model**, because all 8 gfx90a objects are
`noquant{Fp16,Bf16}`. Round 19 currently queues two bf16 arms at 3.4 h of loading
each; with a snapshot that becomes one 3.4 h conversion plus fast reloads, which
makes iterating on that path practical rather than an overnight commitment.

## Ruled out as loader fixes

Recorded so they are not re-litigated. All four funnel through the same
per-expert `weight_loader`, or attack the I/O path this repo already proved
irrelevant (storage delivers the model in 13.19 s at 4.63 GB/s):

| option | why not |
|---|---|
| **fastsafetensors** | ROCm-supported and genuinely faster file→CPU, but that axis is not the bottleneck. Upstream #48644: on a Qwen3-35B-A3B MoE it drives the layerwise path to allocate >40 GB of temp buffers. Still uses the same `weight_loader`. |
| **runai_streamer** (non-sharded) | ROCm-capable, same per-expert loader. |
| **tensorizer** | CUDA/S3-oriented, no verified ROCm path, still per-tensor deserialize. |
| **ServerlessLLM** (2401.14351) / **Tangram** (2512.01357) | 3.6–8.2× gains are I/O-path, and the loader is coupled to their serving system rather than exposed as a vLLM `--load-format`. |

Upstream context: **PR #46766** independently root-causes the fused-MoE copy and
names `expert_data.copy_(loaded_weight)` with a non-contiguous **CPU** source at
"3–4 seconds per weight". Note that is *not* what `configs/fast_moe_expert_load.py`
attempted — that made the already-contiguous narrowed **destination** contiguous,
which `docs/25` refuted twice. Their claim is unverified on ROCm and sits awkwardly
against the `hsakmt_ioctl` evidence, so it should not be trusted here, but it
confirms the path is upstream-known and quantization-agnostic.
