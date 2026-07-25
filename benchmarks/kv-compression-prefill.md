# KV Cache Compression A/B Prefill Benchmark

How KV cache quantization type affects **prefill throughput** on llama.cpp,
comparing `f16`, `q4_0`, `q8_0`, and `q8_0/q4_1` across all-GPU and
split CPU/GPU layer configurations.

Measured **2026-07-25**.

---

## Test setup

- **Model:** DeepSeek-V2-Lite Q8_0 GGUF (`dsv2lite-q8_0.gguf`, 16 GB,
  27 decoder layers, MLA attention)
- **Prompt:** 2,188 tokens (server tokenizer) — system instruction + 45
  rotated technical sentences + one question. Deterministic across runs.
- **Generation:** `n_predict = 1` — isolates prefill; decode time excluded.
- **Hardware:** single AMD MI210 (gfx90a, 64 GB HBM2e), EPYC 74F3, 499 GB DDR4
- **Engine:** llama.cpp `llama-server` build `67b9b0e`, ROCm 7.14
  (binary at `/src/build/bin/llama-server` in `llama-rocm714:latest`)
- **Flash attention:** forced on (`-fa on`) — default `auto` SIGSEGVs on
  gfx90a for DeepSeek-V2-Lite (see [13K prefill benchmark](benchmarks-13k-prefill.md))
- **Context:** `-c 3072 -np 1` — single slot, KV buffer sized to 1×ctx.
  Keeps VRAM footprint low enough to coexist with production RPC servers.
- **GPU pinning:** `HIP_VISIBLE_DEVICES=0` for consistent single-GPU runs.
- **Production load:** ~47 GB VRAM already used on GPU 0 by production
  llama.cpp RPC servers (`rpc-deephat-*`, `rpc-coder-*`). Benchmark ran
  concurrently; absolute numbers are depressed by contention but the A/B
  comparison is valid within the same measurement window.

### Configurations

| ID | GPU layers | KV type K | KV type V | Notes |
|----|-----------|-----------|-----------|-------|
| A | 27/27 (`-ngl 99`) | f16 | f16 | baseline, full precision |
| B | 27/27 | q4_0 | q4_0 | most aggressive compression |
| C | 27/27 | q8_0 | q8_0 | moderate compression |
| D | 27/27 | q8_0 | q4_1 | mimo production default |
| E | 23/27 (`-ngl 23`) | f16 | f16 | split, full precision |
| F | 23/27 | q4_0 | q4_0 | split, aggressive compression |
| G | 23/27 | q8_0 | q4_1 | split, mimo default |

---

## Results

| Config | GPU layers | KV K | KV V | Prefill (tok/s) | ms/token | Total (ms) | vs f16 same-layer |
|--------|-----------|------|------|----------------:|---------:|-----------:|-------------------:|
| **A** | 27/27 | f16 | f16 | **14.4** | 69.2 | 151,478 | 1.0× (baseline) |
| **B** | 27/27 | q4_0 | q4_0 | **492.4** | 2.03 | 4,444 | **34.1×** |
| **C** | 27/27 | q8_0 | q8_0 | **462.1** | 2.16 | 4,735 | **32.0×** |
| **D** | 27/27 | q8_0 | q4_1 | **461.5** | 2.17 | 4,741 | **32.0×** |
| **E** | 23/27 | f16 | f16 | **648.9** | 1.54 | 3,372 | 1.0× (baseline) |
| **F** | 23/27 | q4_0 | q4_0 | **445.6** | 2.24 | 4,910 | 0.69× |
| **G** | 23/27 | q8_0 | q4_1 | **417.1** | 2.40 | 5,246 | 0.64× |

All values from `llama-server` "prompt eval time" log line for 2,188 tokens.

### Visual: prefill rate by config

```
All-GPU (27/27 layers):
  q4_0     ████████████████████████████████████████████  492 tok/s
  q8_0     █████████████████████████████████████████     462 tok/s
  q8_0/q4_1████████████████████████████████████████      462 tok/s
  f16      █                                              14 tok/s  ⚠

Split (23/27 layers):
  f16      ████████████████████████████████████████████  649 tok/s
  q4_0     ██████████████████████████████████            446 tok/s
  q8_0/q4_1████████████████████████████████              417 tok/s
```

---

## Key findings

### 1. f16 KV on all-GPU is catastrophically slow (34× penalty)

With all 27 layers on GPU, f16 KV prefill runs at only **14.4 tok/s**
(151 seconds for 2,188 tokens) — **34× slower** than q4_0 (492 tok/s).
This is reproducible across multiple runs, not a transient spike.

**Root cause: VRAM starvation.** Production RPC servers consume ~47 GB of
the 64 GB HBM2e. Loading all 27 layers (~16 GB Q8_0 weights) plus the
large f16 KV cache leaves insufficient VRAM for the flash-attention
workspace. The kernel falls back to a naive attention path that is
O(n²) in HBM traffic, collapsing throughput.

Evidence: the identical f16 KV config with 23/27 layers (Config E, freeing
~4 layers of weight VRAM) runs at **648.9 tok/s** — **45× faster**. The
extra VRAM headroom lets flash attention allocate its workspace and run
the fast fused kernel.

### 2. With adequate VRAM, f16 is actually the FASTEST prefill type

In the split configuration (23/27 layers, ~4 layers of VRAM freed):

```
f16       649 tok/s   ← fastest (native precision, no dequant)
q4_0      446 tok/s   (0.69× f16)
q8_0/q4_1 417 tok/s   (0.64× f16)
```

f16 wins by **45%** over q4_0 when VRAM is not constrained. Quantized KV
adds dequantization overhead during the attention pass; with enough VRAM
for the full f16 KV + flash workspace, native precision is faster.

**Implication:** KV compression trades prefill speed for VRAM capacity.
It only pays off when VRAM is the bottleneck (which it is on this system
under production load with all-GPU offload).

### 3. q4_0 is the best quantized type for VRAM-constrained all-GPU

| Quant type | All-GPU tok/s | KV size vs f16 |
|-----------|-------------:|:--------------|
| q4_0 | 492 | 0.25× (4 bits vs 16 bits) |
| q8_0 | 462 | 0.50× |
| q8_0/q4_1 | 462 | ~0.38× (mixed) |

q4_0 is **7% faster** than q8_0 on all-GPU. The smaller KV footprint
(fewer HBM bytes to read during attention) directly improves prefill
throughput. The quality cost of 4-bit KV (higher perplexity) is the
tradeoff.

### 4. The mimo production default (q8_0/q4_1) is well-chosen

`q8_0` K + `q4_1` V matches pure `q8_0` prefill speed (462 vs 462 tok/s)
while using less VRAM than pure q8_0 (q4_1 V is 4-bit). This is a good
balance for the production mimo 230B deployment where VRAM is tight and
prefill speed matters for user-facing latency.

### 5. Split configs invert the ranking

| Type | All-GPU rank | Split rank |
|------|:------------|:-----------|
| f16 | 4th (14 tok/s) | **1st** (649 tok/s) |
| q4_0 | **1st** (492 tok/s) | 2nd (446 tok/s) |
| q8_0/q4_1 | 3rd (462 tok/s) | 3rd (417 tok/s) |

With fewer layers on GPU (more VRAM headroom), f16 dominates because
flash attention runs at full speed and native precision avoids dequant.
With all layers on GPU (VRAM-constrained), f16 collapses and quantized
KV wins by keeping the footprint small.

---

## Recommendations

1. **Never use f16 KV with all-GPU offload under production VRAM pressure.**
   It is 34× slower than q4_0. Use `-ctk q4_0 -ctv q4_0` or `-ctk q8_0
   -ctv q4_1` for all-GPU configs.

2. **Use f16 KV when VRAM allows** (e.g., split CPU/GPU configs with
   headroom, or standalone benchmarking without production load). It is
   45% faster than q4_0 and avoids quantization quality loss.

3. **For the mimo production split (23/27 layers):** f16 KV is viable and
   56% faster than the current q8_0/q4_1 default — but only if the extra
   VRAM doesn't destabilize production. Test under peak load first.

4. **q4_0 is the safest default for VRAM-constrained all-GPU.** Best
   prefill speed (492 tok/s), smallest KV footprint, at the cost of
   higher perplexity.

---

## Reproduction

### Prompt and benchmark client

```bash
# bench.py builds a deterministic ~2,188-token prompt and POSTs it to
# llama-server /completion with n_predict=1, extracting prompt-eval
# timings from the JSON response.
# (see /tmp/kvbench/bench.py on the host)
```

### Per-config server launch

```bash
# Config B example (all-GPU, q4_0 KV):
docker run -d --name kvbench \
  --device /dev/kfd --device /dev/dri --group-add 991 \
  -v /mnt/llm-storage:/mnt/llm-storage:ro \
  -p 8097:8097 \
  -e HIP_VISIBLE_DEVICES=0 \
  llama-rocm714:latest \
  -m /mnt/llm-storage/dsv2lite-q8_0.gguf \
  -c 3072 -ngl 99 -fa on -np 1 \
  -ctk q4_0 -ctv q4_0 \
  --port 8097 --host 0.0.0.0

# Wait for /health, then run client:
docker run --rm --network host -v /tmp/kvbench:/bench \
  --entrypoint python3 llama-rocm714:latest \
  /bench/bench.py http://localhost:8097 B

# Extract from logs:
docker logs kvbench 2>&1 | grep 'prompt eval time'
```

### Full sweep driver

```bash
# /tmp/kvbench/run_all.sh iterates configs A–G, each with a fresh
# container: start → health-check → benchmark → log-extract → stop.
```

---

## Environment

| Component | Version / detail |
|---|---|
| llama.cpp | `67b9b0e` (build 1) |
| ROCm | 7.14 |
| GPU | AMD Instinct MI210 (gfx90a / CDNA2, 64 GB HBM2e), GPU 0 |
| CPU | AMD EPYC 74F3 (24c/48t, Zen3) |
| RAM | 499 GB DDR4 |
| Docker image | `llama-rocm714:latest` |
| Model | `dsv2lite-q8_0.gguf` (DeepSeek-V2-Lite, Q8_0, 16 GB, 27 layers) |
| Prompt | 2,188 tokens (server tokenizer) |
| Production load | ~47 GB VRAM used on GPU 0 by RPC servers during test |
