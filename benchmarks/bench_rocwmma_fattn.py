"""Measure llama-server prefill and generation at long context, cache-cold.

Written to settle whether `-DGGML_HIP_ROCWMMA_FATTN=ON` helps on gfx90a. It does
not -- see docs/22-rocwmma-flash-attention-gfx90a.md -- but the harness is
general: it is the right way to compare any two llama-server builds on prefill,
which is where long-context serving actually spends its time.

Two things here are not optional, and both were learned the hard way:

**Every prompt is unique.** llama-server keeps a prefix cache across requests,
and a repeated prompt returns a `prompt_ms` near zero. That reads as a
spectacular speedup and is entirely an artifact. Each prompt is seeded from a
fresh UUID so no two runs -- and no two arms of an A/B -- can share a prefix.

**Timings come from the server, not the wall clock.** llama-server reports
`timings.prompt_ms` and `timings.predicted_ms`, which separate prefill from
decode. Wall-clock timing conflates the two, and at 16k the prefill dominates
so badly that a decode regression would be invisible.

The prompt ends with an instruction to reply `ACKNOWLEDGED`, so a build that is
fast because it is computing garbage is visible as a correctness failure rather
than reported as a win. A backend that is fast and wrong is not a result.

    python3 bench_rocwmma_fattn.py http://127.0.0.1:8092 baseline
    python3 bench_rocwmma_fattn.py http://127.0.0.1:8094 rocwmma --tokens 16384,24576

Run each arm against an *otherwise idle* GPU. On a two-card host pin the two
builds to different cards (`HIP_VISIBLE_DEVICES`) rather than running them
back to back on one, and never benchmark while another model is resident.
"""
import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
import uuid

# Deliberately mundane words: the point is to defeat the prefix cache, not to
# make the model work hard. Content does not affect prefill cost, only length.
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu north south east west upper lower inner outer first second"
).split()

# Measured against timings.prompt_n on Qwen3's BPE: these space-separated words
# average ~1.5 tokens each, not 1. `--tokens` is therefore a *target*, and the
# table prints the real prompt_n the server reports -- always quote that column,
# never the target. Retune this only if a different tokenizer drifts far enough
# to matter; the comparison itself is unaffected, since both arms of an A/B are
# fed the same target and land on the same length.
TOKENS_PER_WORD = 1.5


def build_prompt(target_tokens):
    """A unique filler prompt of roughly `target_tokens`, ending in a question."""
    tail = (
        "\n\nIgnore all of the text above; it is padding. "
        "Reply with exactly one word: ACKNOWLEDGED"
    )
    seed = uuid.uuid4().hex
    nwords = int(target_tokens / TOKENS_PER_WORD)
    # Seed-derived indices make the body unique per call while keeping it
    # deterministic given the seed, so a run can be reproduced from the log.
    state = int(seed, 16)
    body = []
    for _ in range(nwords):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        body.append(WORDS[(state >> 33) % len(WORDS)])
    return f"[run {seed}]\n" + " ".join(body) + tail


def request(base, prompt, max_tokens, timeout):
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/v1/chat/completions",
        data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read())


def measure(base, target_tokens, max_tokens, timeout):
    """One cache-cold request. Returns a dict, or raises."""
    body = request(base, build_prompt(target_tokens), max_tokens, timeout)
    t = body.get("timings")
    if not t:
        raise RuntimeError(
            "server returned no `timings` object -- this harness needs "
            "llama-server's OpenAI endpoint, which reports prompt_ms and "
            "predicted_ms separately. A generic OpenAI proxy (litellm) strips "
            "them; point at the llama-server port directly.")
    text = body["choices"][0]["message"]["content"].strip()
    prompt_ms, prompt_n = t["prompt_ms"], t["prompt_n"]
    pred_ms, pred_n = t["predicted_ms"], t["predicted_n"]
    if prompt_ms <= 0 or prompt_n <= 0:
        raise RuntimeError(
            f"prompt_ms={prompt_ms} prompt_n={prompt_n} -- the prefix cache was "
            "hit despite cache_prompt:false. Restart the server between arms.")
    return {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ms,
        "prompt_tps": prompt_n / (prompt_ms / 1000.0),
        "pred_n": pred_n,
        "pred_tps": pred_n / (pred_ms / 1000.0) if pred_ms > 0 else float("nan"),
        # Substring, not equality: some models wrap the word in punctuation or
        # a short preamble. The check is "did it read the instruction at the
        # very end of 16k of noise", which a broken attention kernel fails.
        "ok": "ACKNOWLEDGED" in text.upper(),
        "reply": text[:60],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="llama-server base URL, e.g. http://127.0.0.1:8092")
    ap.add_argument("label", help="name for this arm, e.g. baseline / rocwmma")
    ap.add_argument("--tokens", default="16384,24576",
                    help="comma-separated prompt sizes (default 16384,24576)")
    ap.add_argument("--reps", type=int, default=2,
                    help="cache-cold requests per size (default 2)")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="tokens to generate, for the decode rate (default 256)")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    sizes = [int(s) for s in args.tokens.split(",")]
    print(f"arm: {args.label}   server: {args.url}")
    print(f"\n{'target':>8} {'prompt_n':>9} {'prefill ms':>11} "
          f"{'prefill t/s':>12} {'gen t/s':>9}  verdict")

    failures = 0
    summary = {}
    for size in sizes:
        rates = []
        for _ in range(args.reps):
            try:
                r = measure(args.url, size, args.max_tokens, args.timeout)
            except (urllib.error.URLError, RuntimeError, KeyError) as exc:
                print(f"{size:>8} {'-':>9} {'-':>11} {'-':>12} {'-':>9}  "
                      f"FAIL {str(exc).strip().splitlines()[0][:70]}")
                failures += 1
                continue
            verdict = "PASS" if r["ok"] else f"WRONG (got {r['reply']!r})"
            if not r["ok"]:
                failures += 1
            print(f"{size:>8} {r['prompt_n']:>9} {r['prompt_ms']:>11.0f} "
                  f"{r['prompt_tps']:>12.1f} {r['pred_tps']:>9.1f}  {verdict}")
            if r["ok"]:
                rates.append(r["prompt_tps"])
        if rates:
            summary[size] = statistics.median(rates)

    if summary:
        print(f"\nmedian prefill rate ({args.label}):")
        for size, rate in summary.items():
            print(f"  {size:>6} tok: {rate:8.1f} tok/s")
    print("\nWRONG means the build ran but could not follow an instruction "
          "buried at the end of the context -- do not quote its throughput.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
