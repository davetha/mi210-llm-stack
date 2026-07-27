# Attic — superseded patch scripts

> ⚠️ **Nothing in this directory should be run.** Every script here applies a
> patch that is now known to be **wrong**. They are kept because they are the
> evidence trail for how the wrong conclusion was reached — the same reason
> [`docs/17`](../../docs/17-pa-fwd-investigation-transcript.md) is kept.

**Use [`../repatch_gfx942_to_gfx90a.py`](../repatch_gfx942_to_gfx90a.py) instead.**
The correct pipeline is documented in
[`docs/19-aiter-operator-port-matrix.md`](../../docs/19-aiter-operator-port-matrix.md).

## What these got wrong

Two errors run through all of them.

**1. `D3E1 → D3CD` — substituting f16 MFMA for bf16 MFMA.** This was premised on
"gfx90a has no BF16 MFMA". It does: `v_mfma_f32_16x16x16bf16_1k`, opcode
**D3E7**. Reinterpreting bf16 bit patterns as f16 silently corrupts results —
it does not crash, which is what made it so costly. Measured effect on bf16
gqa16 paged attention: rel_rms 272 and 0.2% element match, versus 5.8e-4 and
100% with the correct D3E7 substitution.

**2. The `vgpr_count` rewrite (512 → 256).** Premised on gfx90a having a
separate AGPR file that the kernel descriptor must account for. gfx90a has a
**unified** VGPR/AGPR register file, so the rewrite was never necessary. The
related "0 `v_accvgpr` instructions implies incompatible" reasoning is wrong for
the same reason.

## Contents

| Script | What it does wrong |
|---|---|
| `patch_category.py` | `:94` writes `0xD3CD`; `:102-110` rewrites `vgpr_count`. Also scans `.text` at a blind 4-byte stride with **no instruction-length decoding**, so the second word of a 64-bit instruction, or a literal that happens to match, is silently corrupted. |
| `patch_root_cos.py` | Inherits both errors via `from patch_category import`. Also could never run as documented — `:8` inserts `/tmp` on `sys.path` before the import. |
| `patch_all_mla.py` | `:53` `0xD3CD`; `:57-58` `vgpr_count`. |
| `opcode_swap.py` | `:85` `0xD3CD`. |
| `full_binary_patch.py` | `:54` `0xD3CD`, plus clears bit 15 to rewrite AccVGPR destinations to VGPR. |
| `complete_patch_v2.py` | `:36` `0xD3CD`, plus a 6-layer AccVGPR rewrite. |
| `minimal_accvgpr_patch.py` | `:29` `0xD3CD`; `:34-35` `vgpr_count`. Header states the disproven "AccVGPR = 512 − vgpr_count" theory. |
| `correct_vgpr_patch.py` | `vgpr_count` only. Despite the name, the change is unnecessary. |
| `reg_collision_analysis.py` | Analysis premised on the separate-register-file theory. |
| `patch_kernel_descriptor.py` | Rewrites the kernel descriptor, which needs no change. Referenced by no document. |

`docs/16` section 8 previously instructed readers to run the first two. That
recipe has been replaced with the correct pipeline.
