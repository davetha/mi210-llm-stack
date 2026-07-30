# Three gates close the tensor-parallel collective path, and two rest on the same misread

**Date**: 2026-07-31 · **Hardware**: 2× MI210 (gfx90a/CDNA2), PCIe 4.0 x16, no xGMI
**Status**: driver claim **verified**; performance effect **unmeasured**

TP=2 returns **1.28×**, not ~2×, from doubling both bandwidth and compute:

| arm | decode @101k |
|---|---:|
| `t35-w8a8` TP=1 (`results/t35-w8a8-longctx.json`) | 33.80 tok/s |
| `t35-w8a8` TP=2 (`docs/25` item 1c) | 43.40 tok/s |

Three independent switches disable the collective fast path. Two of them are
justified by "there is no xGMI", which is true but is not the same claim as
"there is no peer-to-peer".

## Gate 1 — `NCCL_P2P_DISABLE=1`, and the driver disagrees with its reason

`env/gfx90a-common.env` said the cards "cannot peer-to-peer". From
`/sys/class/kfd/kfd/topology/nodes/{1,2}/p2p_links/0/properties`:

```
type 2                 HSA_IOLINKTYPE_PCIEXPRESS   (XGMI would be 11)
node_from 1 node_to 2  and the reciprocal on node 2
max_bandwidth 32000    32 GB/s, PCIe 4.0 x16
flags 3
```

`hsakmttypes.h:504-511` defines the flag bits as `Override, NonCoherent,
NoAtomics32bit, NoAtomics64bit, NoPeerToPeerDMA`. **`flags 3` = Override +
NonCoherent, so bit 4 `NoPeerToPeerDMA` is clear**, and `Override` being set
makes those flags authoritative rather than inferred from the link type.

So the driver advertises an **enabled PCIe P2P link with device-to-device DMA
permitted**. There is no xGMI link (none of type 11 exists) — that part was
always right. The two claims were conflated.

Supporting: both BARs are fully mapped at **64 G** (Region 0, prefetchable), and
`amd_iommu=off` removes translation from the peer path.

## Gate 2 — `disable_custom_all_reduce=True`, same reasoning

Set because there is no xGMI, so vLLM's custom allreduce "may have nothing to
bind to". That rests on the same premise as gate 1 and inherits the same
correction. Whether the custom kernels can actually use a PCIe peer path is a
separate question and is **not** answered here.

## Gate 3 — the fusion pass cannot run, and fails loudly rather than degrading

`compilation/passes/pass_manager.py`:

```python
if current_platform.is_cuda():                              # line 44
    from .fusion.allreduce_rms_fusion import AllReduceFusionPass

...
if self.pass_config.fuse_allreduce_rms:                     # line 154
    if rocm_aiter_ops.is_enabled():
        self.passes += [RocmAiterAllReduceFusionPass(config)]
    else:
        self.passes += [AllReduceFusionPass(config)]        # line 158 -- NameError on ROCm
```

On ROCm `is_cuda()` is false, so the symbol is never imported and line 158 raises
`NameError`.

**Severity, stated precisely, because the obvious framing overstates it.** The
auto-default cannot reach line 158. `config/vllm.py:140-145`:

```python
if current_platform.is_rocm():
    return (rocm_aiter_ops.is_enabled() and cfg.parallel_config.tensor_parallel_size > 1)
```

Auto-enabling `fuse_allreduce_rms` on ROCm *requires* `rocm_aiter_ops.is_enabled()`
— the same condition that selects the Rocm pass. So the `else` branch is
unreachable under auto-config and the CUDA pass is dead code on ROCm by
construction. It is reachable only through an explicit `-O` /
`--compilation-config` override, where it then dies on an undefined name instead
of degrading or reporting that the platform lacks the pass.

**Latent, not a live crash.** Worth fixing upstream on its own merits —
architecture-neutral and small — but it should not be pitched as "ROCm crashes by
default", because it does not.

## What this does and does not establish

**Established:** the driver advertises PCIe P2P with DMA permitted; the reason
recorded for gates 1 and 2 is wrong; gate 3 is a real upstream defect.

**Not established:** that enabling P2P makes anything faster. The stall that
originally motivated `NCCL_P2P_DISABLE=1` was real, and this does not explain it
— it only removes the stated justification. RCCL may still fail for an unrelated
reason (version, container privileges, ACS).

**A measured qualification.** `ACSCtl` on both upstream bridges
(`0000:86:00.0`, `0000:c2:00.0`) reads `ReqRedir+ CmpltRedir+ UpstreamFwd+`, so
peer traffic is **redirected through the root complex** rather than routed
bridge-to-bridge. Functional, but not the short path. Since `amd_iommu=off`, ACS
is buying little isolation here while still costing the redirect —
`pcie_acs_override=downstream` is a further lever, to be evaluated only if plain
P2P first shows a gain.

## How this could be wrong

- **A driver-advertised link is not a working RCCL path.** KFD describing the
  topology says nothing about whether RCCL's transport selection succeeds.
- **ACS redirect may erase the benefit.** If root-complex round-tripping costs as
  much as host staging, enabling P2P changes little.
- **The 1.28× may not be the interconnect at all.** Decode sits ~3.1× off its
  achievable-bandwidth bound (`docs/25` addendum) and full CUDA-graph capture is
  already on, so a large share of per-token cost is in-kernel. P2P could be real
  and still not move decode.
- **Gate 2 may be correct for its own reasons.** vLLM's custom allreduce may
  require capabilities beyond peer DMA that gfx90a genuinely lacks.

## The experiment

`benchmarks/matrix/probe_p2p_gfx90a.sh` — driver claim, ACS control state, and
measured peer bandwidth via `rocm-bandwidth-test`. Then one A/B: a TP=2 arm with
`NCCL_P2P_DISABLE=0` against the same arm at `=1`. The flag is now
`${NCCL_P2P_DISABLE:-1}` so the override runs without editing the env file.

Order matters: **do not enable P2P and re-tune at the same time.** A nonzero
device-to-device bandwidth row is the precondition; the A/B is the result.
