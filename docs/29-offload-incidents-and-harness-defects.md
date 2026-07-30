# Weights-in-RAM: what failed, why, and what the harness let through

**Date**: 2026-07-29/30 · **Hardware**: 2× MI210 (gfx90a), 499 GB DDR4
**Kernel**: 7.0.0-28-generic · **ROCm**: 7.14

Two attempts at "keep weights in system RAM, compute on the GPU" both failed on
the same night, for unrelated reasons. Neither result is the one that was
expected, and the process failures around them cost more time than the
experiments did. All of it is recorded here because the failures are the
findings.

Related: `docs/28` (model matrix), `docs/24` (quantization), `docs/25` (backlog).

---

## 1. XNACK unified memory — a write-permission fault, not a missing page

`HSA_XNACK=1` + `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` + `-ngl 999` on GLM-4.6
IQ3_XS (139 GB against 128 GB of VRAM) aborted during weight load:

```
llama-server: hsa-runtime/core/runtime/runtime.cpp:2026:
  Runtime::VMFaultHandler: Assertion `false && "GPU memory access fault."'
```

### What the kernel actually reported

```
amdgpu 0000:c3:00.0: [gfxhub0] retry page fault    (src_id:0 ring:0   vmid:3 pasid:32616)
amdgpu 0000:c3:00.0: [gfxhub0] no-retry page fault (src_id:0 ring:208 vmid:3 pasid:32616)
  Process llama-server pid 1214268
  in page starting at address 0x000071d331ff2000 from IH client 0x1b (UTCL2)
  VM_L2_PROTECTION_FAULT_STATUS:0x00341051
    Faulty UTCL2 client ID: TCP (0x8)
    MORE_FAULTS:       0x1
    WALKER_ERROR:      0x0
    PERMISSION_FAULTS: 0x5
    MAPPING_ERROR:     0x0
    RW:                0x1
```

**This refutes the obvious explanation.** The first thing suspected was
`amdgpu.noretry` — the module parameter reads `-1` (auto), and without
recoverable retry faults, demand paging cannot work at all. But the log opens
with a **`retry page fault`**, so retry faults are enabled and the mechanism is
engaging. Setting `amdgpu.noretry=0` would change nothing.

What actually failed is narrower and stranger:

| field | value | meaning |
|---|---|---|
| `MAPPING_ERROR` | `0x0` | the page **is** mapped |
| `WALKER_ERROR` | `0x0` | the page-table walk succeeded |
| `PERMISSION_FAULTS` | `0x5` | the access was denied on **permissions** |
| `RW` | `0x1` | it was a **write** |
| client | `TCP` | ordinary vector store from a compute kernel |

So this is not "the data has not been paged in yet", which is the case demand
paging exists to service. It is **a device write to a resident managed page that
is mapped without write permission**, and the driver failed to promote it: the
recoverable `retry` fault degraded into a fatal `no-retry` fault at the same
address, repeating with `MORE_FAULTS: 0x1`.

### The blast radius, which is the real reason not to retry this casually

The userspace abort left an rwsem with no live owner, and amdgpu's SVM eviction
workers stacked up behind it:

```
INFO: task kworker/44:10 blocked for more than 122 seconds.
Workqueue: events svm_range_evict_svm_bo_worker [amdgpu]
... blocked on an rw-semaphore, but the owner is not found.
```

Load average reached **70 and was still climbing** on a box that also serves
production. Killing the arm drained it (70 → 34 → 20 → 2) and both GPUs
recovered without a reset, enumerating normally at 42–47 °C idle.

That a userspace process aborting can wedge kernel workers this way is an
amdgpu robustness bug independent of whatever provoked the fault.

### Remaining suspects, in order

1. **Multi-GPU managed sharing.** `-ngl 999` on GLM-4.6 spans both cards, and a
   managed page touched by two devices can end up mapped read-only on one while
   the other writes. Cross-device permission promotion is the fragile path, and
   every UVM attempt so far has been two-GPU — this has never been isolated.
2. **Code objects are not built for XNACK.** `libggml-hip.so` carries
   `amdgcn-amd-amdhsa--gfx90a`, i.e. "xnack any", from `AMDGPU_TARGETS=gfx90a`.
   Not `gfx90a:xnack+`.
3. **THP interaction.** Huge-page splitting during HMM migration is a known
   source of exactly this shape of permission mismatch.

The kernel is not missing the machinery: `CONFIG_HMM_MIRROR=y`,
`CONFIG_DEVICE_PRIVATE=y`, `CONFIG_ZONE_DEVICE=y`.

### Status

`benchmarks/matrix/round15_unified_mem.sh` refuses to run without
`ALLOW_UVM_HANG=1`. The prefill-amortisation argument for RAM-resident weights
is **untested, not refuted** — the fault happens at load, before any of it is
exercised. An isolation probe (1 GPU that fits / 2 GPUs that fit / 1 GPU
oversubscribed) is the next step and would separate "UVM is broken here" from
"multi-GPU UVM is broken here". The second would leave it useful.

---

## 2. vLLM prefetch offload — died in NCCL, and the harness did not notice

`--offload-backend prefetch --offload-params experts --offload-group-size 4
--offload-num-in-group 3` on GLM-4.6 AWQ (TP=2) crashed at 22:47:17:

```
Worker_TP1: RuntimeError: NCCL error: unhandled cuda error
  in ncclAllReduce
```

Two workers went zombie. **The container stayed up**, so `run_arm.sh` — which
polls `/v1/models` and separately checks whether the container exited — saw
neither condition and kept waiting. It sat for over an hour printing
`shm_broadcast: no available block` once a minute, and would have burned the
full 5400 s `READY_TIMEOUT`. Four more arms behind it would each have done the
same: roughly six hours to learn nothing.

### It is the offloader, not leftover damage

The crash came 18 minutes after the XNACK incident on the same GPUs, so
attribution was genuinely ambiguous. A control settled it: **`t35-w8a8` at TP=2
loaded and served normally — 130 s, 41.88 GiB KV cache, no errors.** NCCL and
both cards are healthy, so the prefetch offloader failed on its own merits.

Untested mitigations, in order of suspicion: `--enforce-eager` (the offloader
patches module forwards and joins a private copy stream into CUDA-graph
captures, which is the most likely thing to break RCCL collectives), and
`VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1`.

---

## 3. Harness defects found along the way

Four, of which three are fixed. Recorded because each one silently produced
wrong or expensive results rather than failing loudly.

### Fetching inside the bench lock (rounds 12 and 14) — FIXED

Both claimed the FIFO lock and then downloaded 100+ GB, holding the queue head
while nothing computed: 35 minutes for round 12 (whose CDN connection stalled
outright, two sockets in `SYN-SENT`) and 40 for round 14. A download needs no
GPU. Both now fetch before claiming.

Round 12 was fixed first, and round 14 was not checked at the time — the same
shape had been copied into it. Fixing the instance that fails visibly is not
fixing the bug.

### `scp` over a running script — FIXED (by killing and relaunching)

bash reads a script incrementally from an open fd by **byte offset**, and `scp`
truncates and rewrites in place, same inode. Deploying the round-14 fix while
round 14 sat inside a 40-minute download meant that when the download returned,
bash would have resumed reading at a stale offset into different text.

Confirmed by comparing `stat -L /proc/<pid>/fd/255` against `stat` on the path —
identical inode. Kill and relaunch, or deploy under a new filename.

### A crashed-but-alive container burns the full READY_TIMEOUT — NOT YET FIXED

See §2. `run_arm.sh` needs to scan the container log for worker-fatal patterns
(`NCCL error`, `RuntimeError`, a dead `Worker_TP`) during the readiness poll and
fail fast. Note that a naive "any ERROR line" check would false-positive: the
benign `Failed to import Triton kernels` line appears on every healthy vLLM
start.

### A diagnostic container invisible to the FIFO — NOT YET FIXED

The `t35-w8a8` control in §2 was deliberately named `nccl-probe`, **without** the
`bench-` prefix, so that it would not confuse `bench_wait_for_others`. That is
exactly what made it dangerous: the FIFO's container check is
`docker ps | grep '^bench-'`, so an unprefixed container is invisible, and
round 17 launched arms straight on top of it while it held ~42 GiB of KV cache
per rank.

```
23:56      nccl-probe launched (TP=2, 41.88 GiB KV per rank)
23:59:48   t35-q4km-mmqbase  -> cudaMalloc failed: out of memory
23:59:59   t35-q4km-forcemmq -> out of memory
00:00      glm46-iq3xs-mmqbase starts, overlapping probe teardown
```

All three round-17 arms are invalid and must be re-run: the Qwen3-30B pair OOMed
against the probe's KV cache, and the GLM baseline reported 132.4 t/s against a
~196 t/s expectation because `NGL=auto` fitted itself to VRAM the probe was
still holding.

The lesson is that **anything holding VRAM must be visible to the lock**. A
manual diagnostic is bench work; it should either take a claim or use the
`bench-` prefix, not opt out of both.

---

## What still needs doing

| item | state |
|---|---|
| Fail-fast on crashed-but-alive containers | **done** — `run_arm.sh` scans for worker-fatal patterns |
| Make diagnostics visible to the FIFO | **done** — the check now matches `^(bench\|probe)-` |
| Re-run round 17 (MMQ A/B) on clean GPUs | **running** (01:39) |
| UVM isolation probe (1-GPU / 2-GPU / oversubscribed) | designed, not run |
| vLLM prefetch retry with `--enforce-eager` | not started |

### Why the contaminated round 17 had to be thrown away rather than salvaged

The invalid A/B looked like a *spectacular* result, which is precisely the
danger:

| arm | prefill @15k | prefill @25.8k | decode |
|---|---:|---:|---:|
| `mmqbase` (ran during the probe) | 132.4 | 130.8 | **1.48** |
| `forcemmq` (ran after it) | 218.8 | 189.2 | **8.57** |

Read naively that is +65% prefill and a **5.8× decode win** for
`GGML_CUDA_FORCE_MMQ`. It is not. A GEMM dispatch change cannot plausibly move
decode 5.8×; VRAM starvation can, because `NGL=auto` fitted fewer layers onto
the GPU while the probe held ~42 GiB per rank. The baseline ran degraded and the
treatment ran clean.

For scale, `forcemmq`'s 218.8 against the *historical* clean auto-fit baseline
of 195.9 is ~+12% prefill with decode essentially unchanged (8.57 vs 8.51) —
a modest, mechanically plausible result, and the one the re-run should confirm
or refute. The lesson is that a contaminated control can manufacture a headline
number, and headline numbers are exactly what nobody re-checks.
