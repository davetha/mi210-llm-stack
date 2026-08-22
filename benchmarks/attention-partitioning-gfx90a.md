# Attention partitioning on gfx90a — measured

2026-08-22. Every number here is measured on 2× MI210 against Qwen3.8-27B-ablit
W8A8, TP=2, block_size 784, head_size 256, hybrid interleaved KV.
Narrative and root causes: [`docs/62`](../docs/62-attention-partitioning-and-the-kernel-nobody-was-running.md).

Script: [`bench_attention_partition.py`](bench_attention_partition.py).

## Read this first: which kernel a config actually runs

Two Triton attention paths matter, and **a speculative-decode deployment only
ever reaches the second**:

| path | when | patched by |
|---|---|---|
| paged decode (`kernel_paged_attention_2d`) | `query_len == 1` | `triton-pa-seq-partition` |
| spec-verify (`context_attention_fwd` cached scan) | `query_len == num_spec + 1` | `triton-verify-ctx-partition` |

With DFlash2 at n=8 every decode step has `query_len = 9`, so `max_query_len > 1`
routes to `context_attention_fwd` and the paged-decode kernels are never called.
Benchmarking only the first path and deploying it produces **exactly zero**
end-to-end change; that is measured below, not hypothesised.

## End-to-end, production config (spec decode ON)

`tools/ctx_probe.py`, decode rate separated from TTFT, `completion_tokens/wall`
on a non-streaming request.

| context | baseline | + both patches | speedup |
|--------:|---------:|---------------:|--------:|
| 2K | 168 | 178.8 | 1.06× |
| 41K | 30.6 | **92.0** | **3.0×** |
| 101K | 13.6 | **49.0** | **3.6×** |

Both text profiles at 41K, same server: list 30.6 → 73.4, prose 6.9 → 17.5.
A separate run at the same length gave 92.0 on the list profile, so **73–92 is
the honest band** for 41K, not a point.

Attribution: the decode-path patch alone had already been deployed and measured
at 167.5 / 31.1 / 12.4 — no change. The gain is the verify-path patch.

## End-to-end, spec decode OFF

The configuration that does exercise the paged-decode path. Two boots, same
procedure, control measured on the same box in the same session.

| context | shipped | partitioned | speedup |
|--------:|--------:|------------:|--------:|
| 2K | 33.7 | 26.5 | 0.79× |
| 41K | 5.5 | **28.4** | **5.2×** |
| 100K | 2.6 | **30.1** | **11.6×** |

Decode becomes essentially flat in context length. The list/prose gap at 41K
closes to 28.3 / 29.1 — that spread was attention serialisation, not text.

Short context regresses 21%. That is the cost of sizing the partition grid from
the cudagraph bound rather than the live sequence; see `docs/62` §5.

## Kernel level

`bench_attention_partition.py`, median of 20, block table sized from
`--ctx-bound 262144` so the partition count is the one a captured graph gets.

**Paged decode (`query_len` 1)**

| seq_len | batch | unpartitioned | partitioned | speedup |
|--------:|------:|--------------:|------------:|--------:|
| 2048 | 1 | 0.557 ms | 0.410 ms | 1.36× |
| 2048 | 4 | 0.524 ms | 0.413 ms | 1.27× |
| 40960 | 1 | 8.645 ms | 0.420 ms | **20.58×** |
| 40960 | 4 | 7.906 ms | 1.081 ms | 7.31× |
| 102400 | 1 | 20.719 ms | 0.687 ms | **30.15×** |
| 102400 | 4 | 21.208 ms | 2.517 ms | 8.43× |

**Spec-decode verify (`query_len` 9)**

| ctx | batch | unpartitioned | partitioned | speedup |
|----:|------:|--------------:|------------:|--------:|
| 2048 | 1 | 0.469 ms | 0.340 ms | 1.38× |
| 2048 | 4 | 0.677 ms | 0.733 ms | **0.92×** |
| 40960 | 1 | 7.249 ms | 1.037 ms | **6.99×** |
| 40960 | 4 | 11.927 ms | 5.322 ms | 2.24× |
| 102400 | 1 | 17.961 ms | 2.499 ms | **7.19×** |
| 102400 | 4 | 29.573 ms | 12.010 ms | 2.46× |

The 0.92× at short context and batch 4 is a real small regression, kept in the
table rather than dropped: with the batch already supplying 4× the programs and
the sequence too short to fill any partition, the reduce pass is pure overhead.

Kernel speedups are much larger than the end-to-end ones because attention is
one term of a decode step; the model still has to run 40 layers of GEMMs.

## Free paged-attention kernel correctness by block size

Custom HIP kernel vs a float32 reference, forcing the custom path on. head_size
128 and 256, seq_len 1024 and 5000; worst max-relative-error shown.

| block_size | before fix | after fix |
|---:|---:|---:|
| 16 / 32 / 64 | 3.0e-3 … 4.9e-3 | 3.0e-3 … 4.9e-3 |
| 128 | **1.13** | 3.7e-3 |
| 256 | **1.56** | 4.2e-3 |
| 512 | **2.82** | 4.0e-3 |
| 784 | **3.45** | 4.3e-3 |
| 1024 | **5.02** | 3.8e-3 |

256 is the diagnostic row: it divides `T_PAR_SIZE` (256) but not 64, so it fails
on the V bug alone while 64 passes. Keep it in any regression suite.

As a pytest regression test this is 20 failures / 12 passes on the unfixed image
and 32/32 on the fixed one.

## Caveats

- Production runs the two Triton patches as **bind-mounted Python files**, not a
  rebuilt image. Dropping a mount silently costs ~3× long-context decode.
- The image built for the free-kernel fix passed tier 0 only (`--no-gpu`); the
  kernel is verified by the test above, not by the repo's hardware tiers, and it
  is not what production runs.
- TTFT is unchanged. 101K still costs ~93 s of prefill.
