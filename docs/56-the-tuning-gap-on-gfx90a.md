# The tuning gap on gfx90a — the W4A16 fast-path audit

Round 77 and the static audit behind it. The question that started it: **which fast paths does W4A16 actually hit on gfx90a, and what is missing?**

## What "untuned" does and does not mean here

An earlier draft of this page was titled *"nothing on this box is tuned"*, which is wrong and worth correcting up front. What the audit below measures is that **upstream ships no tuned kernel configs for this architecture** — a statement about what arrives in the container, not about the work done on this machine.

This project has tuned one config itself: `E=128,N=384,…,dtype=int8_w8a16.json` for the 30B, at a cost of ~5.5 GPU-hours ([`docs/41`](41-moe-tuning-mi210.md)). It measured **0.786× prefill** and is therefore not deployed — `serve_vllm_aiter.sh:38` leaves `VLLM_TUNED_CONFIG_FOLDER` empty by default, so nothing loads it. Round 77 is tuning a second, for `int4_w4a16`.

Separately, a large amount of *patching* work is in `configs/` — the CK int8 GEMM, the AITER attention and master-gate carve-outs, the ASM paths, the 256k paged-attention extension. Those open code paths; they are not kernel tuning, and none of them populates the tables below.

So the accurate framing: **the shipped tuning tables are empty for gfx90a, one config has been produced locally and measured harmful, and a second is in progress.**

## Every upstream tuned-config table is empty for this card

vLLM ships **317** tuned `fused_moe` configs — MI300X, MI308X, MI325X, MI350X, MI355X and a long tail of NVIDIA parts. The count for MI210:

```
(total MI210 configs: 0)
```

AITER's tuning tables are the same story. All 21 CSVs, `gfx90a` rows against total rows:

| file | gfx90a / total | | file | gfx90a / total |
|---|---:|---|---|---:|
| `a4w4_blockscale_tuned_gemm.csv` | **0** / 1470 | | `a8w8_tuned_gemm.csv` | **0** / 579 |
| `a8w8_blockscale_tuned_gemm.csv` | **0** / 6630 | | `bf16_tuned_gemm.csv` | **0** / 112 |
| `a8w8_bpreshuffle_tuned_gemm.csv` | **0** / 598 | | `tuned_fmoe.csv` | **0** / 617 |
| `asm_a8w8_gemm.csv` | **0** / 33 | | `tuned_grouped_fmoe.csv` | **0** / 84 |

…and the remaining thirteen, all zero.

`docs/43` noted this for `a8w8_tuned_gemm.csv` specifically while explaining why the CK GEMM ran untuned instances. It is not one file — it is every file, on both stacks, for every dtype.

With no shipped entry and no locally-produced one in use, kernel selection for these paths falls out of heuristics. That is worth holding in mind when reading a long run of small or negative results: `docs/50`–`docs/53` fought for 1–3% across eleven knobs, and `docs/55` round 74 found the CK GEMM at 0.982× on the 80B. Each of those A/Bs was valid — both arms ran the same untuned kernels — but they were measuring *around* a kernel nobody had tuned for this architecture, not measuring a tuned baseline.

## What W4A16 hits, read from a running server

Not inferred from config tables — `docs/50`'s method note is that a gap in a config table is a hypothesis, and `docs/55`'s is that a flag being set proves nothing about which kernel ran.

| path | status | evidence |
|---|---|---|
| **AITER FA** | ✅ hit | `Overriding with ROCM_AITER_FA out of potential backends: ['ROCM_AITER_FA', 'ROCM_ATTN', 'TRITON_ATTN']` |
| **AITER paged attention** | ✅ hit | `pa_v1` JIT-built: `gqa_ratio=8, head_size=256, block_size=32, partition_size=256` |
| Tuned MoE config | ❌ miss | `Config file not found at E=512,N=256,device_name=AMD_Instinct_MI210,dtype=int4_w4a16.json` |
| CK int8 GEMM | ❌ N/A | needs int8 activations; W4A16 has 16-bit |
| ASM paged attention | ❌ not loaded | no `.co` objects; needs `num_seqs × heads > 2·cu_num` = 208 |
| Dense linear | `TritonW4A16LinearKernel` | untuned |
| GDN layers | Triton/FLA | `Using Triton/FLA GDN prefill kernel (head_k_dim=128)` |
| MoE method | `CompressedTensorsWNA16MoEMethod` | dequantizes 4-bit to 16-bit |

**AITER FA being hit is why W4A16's prefill is only 9.5% behind W8A8** (7,059 vs 7,806 t/s, round 75) despite dequantizing to bf16. The attention fast path is orthogonal to the weight format.

## Two leads checked and closed

### MoE stride padding — already measured, not a win

`VLLM_ROCM_MOE_PADDING` reaches only the unquantized path (three files: `envs.py`, `fused_moe/oracle/unquantized.py`, `unquantized_fused_moe_method.py`), so no quantized checkpoint receives it. `configs/enable_moe_padding_int8_rocm.py` extends it to INT8, and W4A16's shapes would qualify on the same terms:

| tensor | packed shape | row stride | eligible |
|---|---|---:|---|
| `gate_proj` | [512, 256] I32 | 1024 B | ✅ |
| `up_proj` | [512, 256] I32 | 1024 B | ✅ |
| `down_proj` | [2048, 64] I32 | 256 B | ❌ |

It looks like a free ~10% (upstream's Mixtral figure). **It is not.** `docs/45` round 44 already measured the INT8 version on this box at **0.961× decode**, prefill flat — with the patch verifiably working (w13 stride moved 2048 → 2304 B, w2 correctly untouched). Same mechanism, so the prior for an int4 port is negative and it was not built.

Recorded because the idea is attractive enough to be re-proposed, and the counter-evidence is one round away.

### Similar-footprint quants converge on the same kernel

`_POSSIBLE_KERNELS[PlatformEnum.ROCM]` for weight-only quantization:

```
RDNA3W4A16LinearKernel, RDNAHybridW4A16LinearKernel,
TritonW4A16LinearKernel, ConchLinearKernel, ExllamaLinearKernel
```

Marlin, Machete, AllSpark and Cutlass W4A8 are **CUDA-only** — not in the ROCm list at all. The two RDNA kernels gate on `on_gfx1100()`. So on CDNA2 every 4-bit weight-only format — GPTQ, AWQ, compressed-tensors — falls through to the **same** `TritonW4A16LinearKernel`.

**Switching 4-bit checkpoint format cannot change which kernel runs.** The lever is tuning that kernel, not sourcing a different quantization.

### Why the RDNA3 kernel cannot be reached by a binary patch

Worth recording because this project's history is full of "the gate says gfx942 but the kernel runs fine on gfx90a" — `docs/43`'s CK GEMM being exactly that. This is not one of those. Three independent blockers:

**1. It is not compiled.** `CMakeLists.txt:1376`:

```cmake
if(VLLM_GPU_ARCHES MATCHES "gfx1100")
    "csrc/rocm/q_gemm_rdna3.cu"
    "csrc/rocm/q_gemm_rdna3_wmma.cu"
    "csrc/rocm/moe_q_gemm_rdna3.cu")
```

`torch.ops._rocm_C.gptq_gemm_rdna3` does not exist in the built extension. There is no gate to carve out.

**2. The WMMA variant uses an instruction CDNA2 does not have.** `v_wmma_*` is RDNA3-only; CDNA2 has MFMA, a different unit with different fragment layouts. Unlike `docs/49`'s `v_mfma_f32_16x16x16bf16_1k` vs `_bf16`, this is not two names for one opcode.

**3. Wave32 is baked into the data layout.** From the kernel's own comments:

> *"Wave32 input fragment storage is **doubled** — lanes 16..31 hold a copy of lanes 0..15"*
> *"lane t holds COLUMN n=lane_lo of the 16x16 output… lanes 0..15 = even rows, lanes 16..31 = odd rows"*
> *"CAS-loop on a 32-bit word covering 2 packed lanes… pairs adjacent lanes via shfl_xor"*

Hardcoded 32-lane mappings. gfx90a is wave64, so these would produce **silently wrong numbers** rather than failing — the same hazard `docs/03` records for TurboQuant's wave64 path, and the reason KIVI2 was deliberately written as pure scalar C.

Porting it means rewriting the fragment layouts for wave64 and replacing WMMA with MFMA. That is writing a new kernel, not patching a gate. **And the CDNA equivalent already exists** — Triton emits MFMA on gfx90a, so `TritonW4A16LinearKernel` *is* the CDNA path. It is not missing; it is untuned.

## Round 77: tuning the one thing that is actually missing

`benchmark_moe.py --dtype int4_w4a16` works on this shape unpatched (the local `fix_benchmark_moe_int8_w8a16.py` patch touches only the int8 path). E=512 and N=256 come from `num_experts` and `moe_intermediate_size / TP`, matching the filename the server asks for exactly.

M=1 completed in **~14 minutes** over 8,000 candidate configurations:

```json
"1": { "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128,
       "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
       "waves_per_eu": 4, "SPLIT_K": 1 }
```

### The trap this round is built to avoid

`docs/41` tuned this same kernel for `int8_w8a16` and **the tuned config made everything slower**:

| workload | metric | stock | tuned | factor |
|---|---|---:|---:|---:|
| cold 16k | prefill | 5,945.95 | 4,672.43 | **0.786×** |
| longctx | prefill | 4,667.07 | 3,824.84 | 0.820× |
| longctx | decode | 50.88 | 47.05 | 0.925× |

Not because tuning does not work, but because only M=1–32 were tuned and **vLLM's nearest-M matching has no fallback** — prefill shapes at M > 32 matched a config tuned for a decode shape. A partial config is worse than none.

So round 77 sweeps M = 1…2048, tunes one size per invocation (the tuner writes one file at the end, so a crash discards everything — `docs/41` lost three completed sizes to a Ray worker death), merges into an accumulator after each, and **refuses to bless a config with gaps in the prefill range**.

If the large sizes exceed the 3-hour per-size cap, the honest outcome is "do not deploy this," not a partial config.
