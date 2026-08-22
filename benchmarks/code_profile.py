#!/usr/bin/env python3
"""Third text profile: actual code generation, the thing this box is used for.

The list/prose split brackets the extremes of draft acceptance, but neither is
what a coding agent emits. This measures the real regime. No no-spec arm is
needed: without a drafter the decode rate is text-INDEPENDENT (measured
37.1/37.1, 36.0/36.0, 32.4/32.4 for list/prose at 2K/41K/101K), so the existing
n=0 numbers are the reference for any profile.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8063/v1/chat/completions"
METRICS = "http://127.0.0.1:8063/metrics"
SENT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "

TASK = ("Ignore the filler above. Write a complete Python module implementing "
        "an LRU cache with a doubly linked list and a dict: class LRUCache with "
        "__init__(capacity), get(key), put(key, value), plus a _Node class and "
        "docstrings. Output only code.")


def metrics():
    try:
        txt = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for k in ("num_drafts_total", "num_draft_tokens_total",
              "num_accepted_tokens_total"):
        for line in txt.splitlines():
            if line.startswith("vllm:spec_decode_" + k):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out


def one(prompt, max_tokens=400):
    body = {"model": "qwen38-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    first = end = None
    usage = None
    for raw in urllib.request.urlopen(req, timeout=3600):
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data: ") or "[DONE]" in line:
            continue
        d = json.loads(line[6:])
        if d.get("usage") and d["usage"].get("completion_tokens"):
            usage = d["usage"]
        ch = d.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"):
            now = time.time()
            if first is None:
                first = now
            end = now
    if not (usage and first and end and end > first):
        return None
    return (usage["completion_tokens"] - 1) / (end - first), usage.get("prompt_tokens", 0)


NOSPEC = {2000: 37.1, 41000: 36.0, 101000: 32.4}

print("### code-generation profile, n=8 (n=0 reference is text-independent)")
print("%-7s %8s | %-24s | %-8s | %s"
      % ("ctx", "prompt", "decode tok/s med [min,max]", "vs n=0", "accept | tok/draft"))
for target in (2000, 41000, 101000):
    n = max(1, (target - 400) // 16)
    prompt = SENT * n + "\n\n" + TASK
    before = metrics()
    rates = []
    ptok = 0
    for _ in range(3):
        r = one(prompt)
        if r:
            rates.append(r[0]); ptok = r[1]
    after = metrics()
    if not rates:
        print("%-7s ERROR" % target); continue
    rates.sort()
    med = rates[len(rates) // 2]
    dd = after.get("num_drafts_total", 0) - before.get("num_drafts_total", 0)
    dt = after.get("num_draft_tokens_total", 0) - before.get("num_draft_tokens_total", 0)
    da = after.get("num_accepted_tokens_total", 0) - before.get("num_accepted_tokens_total", 0)
    acc = ("%.3f | %.2f" % (da / dt, da / dd)) if dt and dd else "n/a"
    print("%-7s %8d | %8.1f [%6.1f,%6.1f]   | %6.2fx  | %s"
          % ("%dK" % (target // 1000), ptok, med, rates[0], rates[-1],
             med / NOSPEC[target], acc))
    sys.stdout.flush()
