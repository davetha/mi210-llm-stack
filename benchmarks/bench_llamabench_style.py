#!/usr/bin/env python3
"""llama-bench-style depth sweep for vLLM.

Replicates `llama-bench -p 2048 -n 256 -d <depths> -o md` semantics:

  pp2048 @ dN : N tokens already in context (prefix-cached), measure the rate
                at which the NEXT 2048 prompt tokens are processed.
  tg256  @ dN : N tokens in context, measure generation rate over 256 tokens.

Requires the server to run with --enable-prefix-caching so the depth prefix is
warm and only the 2048-token delta is billed to pp (matching llama-bench).

Uses integer token IDs for exact prompt lengths, and streaming to separate
TTFT (prefill) from inter-token latency (decode).
"""
import json, statistics, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8033"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "q36-awq"
DEPTHS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
          else ["0", "8192", "16384", "32768", "65536", "98304", "131072", "262144"])]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
PP, TG = 2048, 256
LABEL = sys.argv[5] if len(sys.argv) > 5 else "qwen3.6 27B AWQ"
# Unique per invocation so pp deltas are never served from a warm prefix cache.
RUN_SALT = int(time.time()) % 1000000 * 31

# Deterministic token IDs; avoid specials (keep well inside vocab, skip low ids).
def toks(n, seed=1234):
    x = seed
    out = []
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(1000 + (x % 60000))
    return out

def stream(prompt_ids, max_tokens):
    """Return (ttft_s, total_s, n_out)."""
    payload = {"model": MODEL, "prompt": prompt_ids, "max_tokens": max_tokens,
               "temperature": 0, "stream": True}
    if max_tokens > 1:
        payload["ignore_eos"] = True
    req = urllib.request.Request(f"{BASE}/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; n = 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            s = line.decode().strip()
            if not s.startswith("data: "):
                continue
            if s.endswith("[DONE]"):
                break
            if ttft is None:
                ttft = time.time() - t0
            n += 1
    return ttft, time.time() - t0, n

def mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

rows = []
for d in DEPTHS:
    base_ids = toks(d) if d else []
    # Warm the depth prefix into the prefix cache (not timed).
    if d:
        try:
            stream(base_ids, 1)
        except Exception as e:
            rows.append((f"pp{PP} @ d{d}", None, None, f"warm failed: {type(e).__name__}"))
            continue

    # ---- pp: 2048 NEW tokens on top of the cached depth prefix ----
    # Each rep uses a DIFFERENT 2048-token delta, otherwise prefix caching
    # serves reps 2..N from cache and the measured rate is meaningless.
    pp_rates = []
    for rep in range(REPS):
        # Salt the delta seed per process run, otherwise a re-run of this script
        # against a server with --enable-prefix-caching replays cached deltas
        # and reports absurd rates (tens of thousands of tok/s).
        pp_ids = base_ids + toks(PP, seed=90001 + rep * 7919 + RUN_SALT)
        try:
            ttft, _, _ = stream(pp_ids, 1)
            if ttft and ttft > 0:
                pp_rates.append(PP / ttft)
        except Exception as e:
            print(f"  [pp d{d}] {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
    m, s = mean_sd(pp_rates)
    rows.append((f"pp{PP}" + (f" @ d{d}" if d else ""), m, s, None))
    print(f"{rows[-1][0]:>22} | {m:10.2f} ± {s:7.2f}", flush=True)

    # ---- tg: 256 tokens generated with depth tokens in context ----
    tg_rates = []
    for _ in range(REPS):
        try:
            ttft, tot, n = stream(base_ids if d else toks(8), TG)
            dec = tot - (ttft or 0)
            if n > 1 and dec > 0:
                tg_rates.append((n - 1) / dec)
        except Exception as e:
            print(f"  [tg d{d}] {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
    m, s = mean_sd(tg_rates)
    rows.append((f"tg{TG}" + (f" @ d{d}" if d else ""), m, s, None))
    print(f"{rows[-1][0]:>22} | {m:10.2f} ± {s:7.2f}", flush=True)

print()
print("| model | backend | test | t/s |")
print("| --- | --- | ---: | ---: |")
for name, m, s, err in rows:
    if err:
        print(f"| {LABEL} | ROCm/vLLM | {name} | {err} |")
    else:
        print(f"| {LABEL} | ROCm/vLLM | {name} | {m:.2f} ± {s:.2f} |")
