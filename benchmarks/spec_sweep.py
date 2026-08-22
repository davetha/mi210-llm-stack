#!/usr/bin/env python3
"""Sweep num_speculative_tokens against context length AND text profile.

Answers: is DFlash2 still worth anything at long context now that attention is
partitioned, and does the answer flip between predictable and unpredictable
text?

Method notes, all of them learned the hard way on this box:
  * Decode rate is measured from the FIRST CONTENT TOKEN to the last, so TTFT
    is excluded. The token count comes from usage.completion_tokens, never from
    counting SSE chunks -- vLLM coalesces ~2.7 tokens per chunk under spec
    decode and chunk-counting undercounts ~3x.
  * Three replicates per cell, reported as median [min, max]. A single number
    here is not a result.
  * Acceptance is read from the server's own spec-decode counters, deltaed
    around each cell, so it describes that cell rather than the process
    lifetime. accepted/draft is the acceptance rate; accepted/drafts is the
    amortisation factor, which is what actually buys throughput.
  * The two profiles differ only in what the model is asked to emit against an
    identical haystack, so acceptance is the only thing that moves.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8063/v1/chat/completions"
METRICS = "http://127.0.0.1:8063/metrics"
SENT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. "

PROFILES = {
    "list": "Write a numbered list from 1 to 200, one number per line. "
            "Output only the list.",
    "prose": "Ignore the repeated filler above. Write a detailed, flowing "
             "explanation of how a write-ahead log keeps a database crash-safe, "
             "covering checkpoints, redo and undo, and group commit. Do not use "
             "lists or headings.",
}


def metrics():
    try:
        txt = urllib.request.urlopen(METRICS, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for key in ("num_drafts_total", "num_draft_tokens_total",
                "num_accepted_tokens_total"):
        for line in txt.splitlines():
            if line.startswith("vllm:spec_decode_" + key):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def one(prompt, max_tokens):
    body = {"model": "qwen38-27b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
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
    return {"ttft": first - t0,
            "decode_tps": (usage["completion_tokens"] - 1) / (end - first),
            "prompt": usage.get("prompt_tokens", 0),
            "completion": usage["completion_tokens"]}


def cell(target_tok, profile, reps, max_tokens):
    n = max(1, (target_tok - 400) // 16)
    haystack = SENT * n
    prompt = haystack + "\n\n" + PROFILES[profile]
    before = metrics()
    runs = []
    for _ in range(reps):
        r = one(prompt, max_tokens)
        if r:
            runs.append(r)
    after = metrics()
    if not runs:
        return None
    rates = sorted(r["decode_tps"] for r in runs)
    d_drafts = after.get("num_drafts_total", 0) - before.get("num_drafts_total", 0)
    d_draft_t = after.get("num_draft_tokens_total", 0) - before.get("num_draft_tokens_total", 0)
    d_acc = after.get("num_accepted_tokens_total", 0) - before.get("num_accepted_tokens_total", 0)
    return {
        "prompt": runs[0]["prompt"],
        "median": rates[len(rates) // 2], "lo": rates[0], "hi": rates[-1],
        "ttft": runs[0]["ttft"],
        "acc_rate": (d_acc / d_draft_t) if d_draft_t else None,
        "acc_per_draft": (d_acc / d_drafts) if d_drafts else None,
    }


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "arm"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print("### ARM %s (reps=%d)" % (label, reps))
    print("%-7s %-6s %8s | %-26s | %8s | %s"
          % ("ctx", "prof", "prompt", "decode tok/s med [min,max]", "TTFT",
             "accept (rate | tok/draft)"))
    for target, mt in ((2000, 300), (41000, 300), (101000, 300)):
        for prof in ("list", "prose"):
            c = cell(target, prof, reps, mt)
            if not c:
                print("%-7s %-6s %8s | ERROR" % (target // 1000, prof, "-"))
                continue
            acc = ("%.3f | %.2f" % (c["acc_rate"], c["acc_per_draft"])
                   if c["acc_rate"] is not None else "n/a (no spec)")
            print("%-7s %-6s %8d | %8.1f [%6.1f,%6.1f]      | %7.1fs | %s"
                  % ("%dK" % (target // 1000), prof, c["prompt"],
                     c["median"], c["lo"], c["hi"], c["ttft"], acc))
            sys.stdout.flush()
