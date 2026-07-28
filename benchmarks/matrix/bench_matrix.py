"""Cross-engine benchmark harness: TTFT at cold 16k, sustained decode at long context.

This is the instrument for the MI210 quantization matrix. It has to produce
numbers that are comparable across **vLLM and llama.cpp**, which is the whole
difficulty: the two engines report timings in incompatible ways, and neither
one's native fields can be compared to the other's.

  llama.cpp  returns a `timings` object with `prompt_ms` / `predicted_ms`.
             `prompt_ms` is prefill compute only -- it excludes queueing,
             tokenization, and detokenization.
  vLLM       returns no such thing. It has no per-request prefill field on the
             OpenAI endpoint at all.

So the primary measurement here is **streaming time-to-first-token**: send the
request with `stream: true`, start a clock, stop it when the first token with
actual content arrives. That is engine-agnostic, it is what a user of a coding
assistant actually feels, and it is the only number that means the same thing on
both engines. Where a server *also* reports native timings, they are recorded
alongside as `native_*` for cross-validation -- never mixed into the primary
metric, because they measure a strictly smaller interval.

Two workloads:

  cold16k   The coding-assistant scenario: a first, large context block.
            Reports TTFT and an implied prefill rate.
  longctx   Sustained decode with a very large context resident. Reports decode
            tokens/sec measured *after* first token, so prefill cost is
            excluded from the rate.

**Prefix caching is the trap that invalidates this entire benchmark.** vLLM V1
enables automatic prefix caching by default and llama.cpp keeps a prefix cache
per slot; a repeated prompt returns a TTFT near zero and reads as a spectacular
result. Every prompt built here is seeded from a fresh UUID **in its first
tokens**, so no two requests can share a prefix. `--verify-cold` additionally
fails the run if a repeat is suspiciously faster than the first, which catches
the case where caching is defeating you silently.

    python3 bench_matrix.py --url http://127.0.0.1:8000 --model my-model \
        --label tier35-awq --workload cold16k --reps 3 --out results/x.json

Every run writes a JSON record. The matrix table is built from those records,
never from a human transcribing terminal output.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu north south east west upper lower inner outer first second "
    "module handler buffer pointer struct kernel thread vector matrix tensor cache"
).split()

# Empirical for Qwen/Llama-family BPE on space-separated lowercase words. This
# is only used to *aim* at a token count; the true count is read back from the
# server when it reports one, and the aimed-at target is always recorded so a
# reader can tell the difference. Do not treat it as exact.
TOKENS_PER_WORD = 1.5

# Below this many generated tokens, a decode rate is scheduler jitter rather
# than throughput, so it is reported as n/a instead of as a number someone
# might quote.
MIN_TOKENS_FOR_DECODE_RATE = 32


def build_prompt(target_tokens, marker):
    """Unique filler of roughly `target_tokens`, with the uniqueness FIRST.

    The UUID must lead. A unique *suffix* would still share a cacheable prefix
    with every other prompt of the same length, which is exactly the failure
    this is meant to prevent.
    """
    nwords = max(1, int(target_tokens / TOKENS_PER_WORD))
    state = int(marker, 16)
    body = []
    for _ in range(nwords):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        body.append(WORDS[(state >> 33) % len(WORDS)])
    return (
        f"session {marker}\n" + " ".join(body) +
        "\n\nThe text above is padding; ignore it. "
        "Reply with exactly one word: ACKNOWLEDGED"
    )


def stream_request(url, model, prompt, max_tokens, timeout, extra=None):
    """Streamed completion. Returns timing dict measured client-side.

    ttft_s is measured to the first chunk carrying non-empty content. Chunks
    with empty deltas (role announcements, keepalives) are deliberately not
    counted -- treating those as "first token" would understate TTFT by however
    long the server takes to actually produce text.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1234,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    first_tok_time = None
    ntok = 0
    text = []
    usage = None

    # `timeout` here is urllib's PER-SOCKET-OPERATION timeout, not a deadline
    # for the whole request. Passing the full budget (3600 s) means a single
    # read that never returns blocks for an hour.
    #
    # Observed on the 357B GLM arm: the server logged the request as
    # "200 OK" and went idle at 0% GPU with 0 running requests, while the
    # client sat in poll_schedule_timeout accumulating zero CPU for 17
    # minutes -- the stream ended without the parser seeing `[DONE]`, and
    # HTTP keep-alive left the connection open with nothing more to send.
    #
    # A per-read cap bounds that. It has to be generous enough for a genuine
    # long prefill, where the gap before the first token IS the measurement:
    # 230k tokens took 346 s on llama.cpp, so the floor is minutes, not
    # seconds.
    read_timeout = min(timeout, max(600, timeout // 4))
    try:
        resp_cm = urllib.request.urlopen(req, timeout=read_timeout)
    except urllib.error.HTTPError as exc:
        # Surface the server's own explanation. Without this the failure is a
        # bare "HTTP Error 400: Bad Request" with no hint, and the actual cause
        # is in the body -- e.g. llama.cpp rejecting a prompt longer than the
        # server's --ctx-size, which is a harness misconfiguration rather than
        # anything wrong with the model.
        try:
            body = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001
            body = "<no body>"
        raise RuntimeError(f"HTTP {exc.code} from server: {body}") from exc

    with resp_cm as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                # Reasoning models split their output across two fields, and
                # which one is used depends on the ENGINE, not the model.
                # llama.cpp (with --jinja) emits the <think> block as
                # `reasoning_content` and only the post-reasoning answer as
                # `content`; vLLM leaves it all in `content` unless a reasoning
                # parser is configured. Reading only `content` therefore sees
                # NOTHING at all from llama.cpp on a short generation that has
                # not escaped the think block yet -- the stream looks empty
                # even though the server reports it generated tokens fine.
                # TTFT means time to the first token the model produced, on
                # whichever channel it produced it.
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if not piece:
                    continue
                if ttft is None:
                    ttft = time.perf_counter() - t0
                    first_tok_time = time.perf_counter()
                ntok += 1
                text.append(piece)
    total = time.perf_counter() - t0

    if ttft is None or first_tok_time is None:
        raise RuntimeError("stream produced no content chunks")

    # Decode rate excludes the first token and all prefill: it is measured from
    # first-token arrival to last-token arrival. With ntok<=1 there is no
    # interval to measure and the rate is undefined rather than zero.
    decode_s = time.perf_counter() - first_tok_time
    decode_tps = (ntok - 1) / decode_s if ntok > 1 and decode_s > 0 else None

    return {
        "ttft_s": ttft,
        "total_s": total,
        "stream_tokens": ntok,
        "decode_tps": decode_tps,
        # Head AND tail. A reasoning model puts its actual answer at the very
        # END, after a </think> block that can run for hundreds of tokens, so
        # a head-only excerpt would make every correctness check fail on
        # exactly the models most worth testing.
        "text": ("".join(text)[:200] + " ...[snip]... " + "".join(text)[-300:]
                 if sum(len(t) for t in text) > 500 else "".join(text)),
        "usage": usage,
    }


def native_timings(url, model, prompt, max_tokens, timeout):
    """Non-streamed request, to harvest llama.cpp's `timings` if present.

    Returns None on any engine that does not provide them (vLLM). This is
    recorded for cross-validation only and is never the primary metric.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "seed": 1234,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            body = json.loads(fh.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    t = body.get("timings")
    if not t:
        return None
    out = {"native_prompt_n": t.get("prompt_n"), "native_prompt_ms": t.get("prompt_ms"),
           "native_predicted_n": t.get("predicted_n"),
           "native_predicted_ms": t.get("predicted_ms")}
    if t.get("prompt_ms"):
        out["native_prefill_tps"] = t["prompt_n"] / (t["prompt_ms"] / 1000.0)
    return out


def read_vram(vram_cmd):
    """Sample VRAM via a user-supplied shell command returning bytes-used lines.

    Kept as an injected command rather than hardcoded rocm-smi because the
    harness usually runs on a different host from the GPUs.
    """
    if not vram_cmd:
        return None
    try:
        out = subprocess.run(vram_cmd, shell=True, capture_output=True,
                             text=True, timeout=60).stdout
    except subprocess.SubprocessError:
        return None
    vals = [int(tok) for tok in out.split() if tok.isdigit() and len(tok) > 6]
    if not vals:
        return None
    return {"per_gpu_bytes": vals, "total_gb": round(sum(vals) / 1e9, 2)}


def probe_correctness(url, model, timeout):
    """One cheap request that proves the model and its kernels actually work.

    This is deliberately SEPARATE from the timed runs. The obvious design --
    append "reply ACKNOWLEDGED" to the 16k benchmark prompt and check the
    output -- fails on reasoning models: Qwen3-*-Thinking emits a <think>
    block first and needs ~250 tokens before it reaches the answer, so a
    TTFT run with max_tokens=8 always reports WRONG even though the model is
    perfectly healthy. Raising max_tokens on every timed rep instead would
    add a minute of pointless generation to each one.

    So: correctness gets a short prompt and a generous budget, timing gets a
    long prompt and a tiny budget. Neither compromises the other.

    Returns (ok, detail).
    """
    prompt = "Reply with exactly one word: ACKNOWLEDGED"
    try:
        r = stream_request(url, model, prompt, 2048, timeout)
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        return False, f"probe request failed: {exc}"
    # Search the whole stream, including past a </think> block.
    ok = "ACKNOWLEDGED" in r["text"].upper()
    return ok, r["text"][-120:]


def warmup(url, model, target_tokens, timeout):
    """One discarded request at the real shape, to absorb first-use costs.

    The first request against a fresh server pays for Triton JIT compilation of
    the MoE and attention kernels, and any deferred graph capture. Measured on
    an 80B MoE: first request 124.8 s, subsequent requests 3.4 s at the same
    shape -- a 37x difference that is warmup, not throughput.

    Without this, that first rep either poisons the mean or, worse, trips the
    cold-cache assertion below: "first is slow because of JIT" and "later ones
    are fast because of prefix caching" look identical from the outside. The
    harness previously reported the former as the latter and voided a run that
    was fine.

    Warmup uses the same prompt length as the timed reps, because kernel
    selection is shape-dependent -- warming at 16 tokens does not compile the
    kernels a 16k prompt will use. It gets its own UUID like every other
    prompt, so it cannot seed a prefix cache for the timed reps.
    """
    # Warmup gets its own, tighter budget. It is not a measurement, so there is
    # no reason to let it consume the full per-arm timeout -- and a warmup that
    # runs for tens of minutes means something is wrong, not that the model is
    # large. Capped at 15 minutes or the arm's own timeout, whichever is less.
    try:
        stream_request(url, model, build_prompt(target_tokens, uuid.uuid4().hex),
                       8, min(timeout, 900))
    except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
        # A warmup failure is not fatal on its own; the timed reps will fail
        # too and report properly. Say so rather than swallowing it.
        print(f"  warmup request failed ({exc}) -- continuing to timed reps")


def run_workload(args):
    reps = args.reps
    target = args.prompt_tokens
    records = []

    for i in range(reps):
        marker = uuid.uuid4().hex
        prompt = build_prompt(target, marker)
        t_start = time.time()
        r = stream_request(args.url, args.model, prompt, args.max_tokens,
                           args.timeout)
        r["rep"] = i
        r["marker"] = marker
        r["target_prompt_tokens"] = target
        r["wall_start"] = t_start
        if r["usage"]:
            r["actual_prompt_tokens"] = r["usage"].get("prompt_tokens")
        # implied prefill rate from TTFT; this INCLUDES queueing/tokenization,
        # which is the honest end-to-end figure but is not comparable to
        # llama.cpp's prefill-only prompt_ms.
        n = r.get("actual_prompt_tokens") or target
        r["implied_prefill_tps"] = n / r["ttft_s"] if r["ttft_s"] > 0 else None
        # A decode rate computed over a handful of intervals is noise, not a
        # measurement: cold16k generates 8 tokens, so the "rate" is dominated
        # by scheduler jitter on the first few steps. Only the longctx
        # workload generates enough to characterise sustained decode.
        if r["stream_tokens"] < MIN_TOKENS_FOR_DECODE_RATE:
            r["decode_tps"] = None
            r["decode_tps_note"] = (
                f"suppressed: only {r['stream_tokens']} tokens generated, "
                f"need >={MIN_TOKENS_FOR_DECODE_RATE} for a meaningful rate")
        records.append(r)
        dec = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "n/a"
        print(f"  rep {i}: ttft={r['ttft_s']:.3f}s  "
              f"prompt_tok={n}  gen={r['stream_tokens']}  decode_tps={dec}")

    ttfts = [r["ttft_s"] for r in records]
    if args.verify_cold and len(ttfts) > 1:
        # A genuine cold prompt should not get dramatically faster on repeat.
        # If it does, prefix caching is active and every number here is void.
        if min(ttfts[1:]) < ttfts[0] * 0.5:
            raise SystemExit(
                f"COLD-CACHE CHECK FAILED: first TTFT {ttfts[0]:.3f}s but a later "
                f"rep hit {min(ttfts[1:]):.3f}s. Prefix caching is defeating the "
                f"unique-prompt strategy -- disable APC "
                f"(vLLM: --no-enable-prefix-caching) and re-run. Numbers voided.")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="model name the server expects")
    ap.add_argument("--label", required=True, help="e.g. tier80-awq-vllm")
    ap.add_argument("--workload", choices=["cold16k", "longctx"], required=True)
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="override; defaults to 16384 (cold16k) or 262144 (longctx)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override; defaults to 8 (cold16k) or 256 (longctx)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--verify-cold", action="store_true", default=True)
    ap.add_argument("--no-verify-cold", dest="verify_cold", action="store_false")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the discarded warmup request. Only for measuring "
                         "first-request cost deliberately -- otherwise JIT "
                         "compilation lands in rep 0 and trips --verify-cold.")
    ap.add_argument("--vram-cmd", default=None,
                    help="shell command printing VRAM bytes-used per GPU")
    ap.add_argument("--engine", default="unknown", help="vllm | llamacpp")
    ap.add_argument("--quant", default="unknown")
    ap.add_argument("--tier", default="unknown")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.prompt_tokens is None:
        args.prompt_tokens = 16384 if args.workload == "cold16k" else 262144
    if args.max_tokens is None:
        # cold16k measures TTFT, so generate almost nothing. longctx needs a
        # real generation run to get a stable decode rate.
        args.max_tokens = 8 if args.workload == "cold16k" else 256

    print(f"{args.label} / {args.workload} / {args.engine} / {args.quant}")
    print(f"  target prompt tokens: {args.prompt_tokens}, max_tokens {args.max_tokens}")

    vram_before = read_vram(args.vram_cmd)

    # Prove the model works before spending time timing it. A backend that is
    # fast because it is computing garbage must be reported as broken, not as
    # a result -- this repo has published fallback-path numbers as real ones
    # before (docs 16 and 17) precisely because nothing checked.
    probe_ok, probe_detail = probe_correctness(args.url, args.model, args.timeout)
    print(f"  correctness probe: {'PASS' if probe_ok else 'FAIL'}  ({probe_detail!r})")

    # Absorb first-use JIT/graph-capture cost at the real prompt shape before
    # timing anything. See warmup() -- without it, an 80B MoE's 124.8 s first
    # request against 3.4 s steady state trips the cold-cache assertion.
    if not args.no_warmup:
        print(f"  warmup ({args.prompt_tokens} tok, discarded)...", flush=True)
        t_warm = time.perf_counter()
        warmup(args.url, args.model, args.prompt_tokens, args.timeout)
        print(f"  warmup took {time.perf_counter() - t_warm:.1f}s")

    records = run_workload(args)
    vram_after = read_vram(args.vram_cmd)

    # Report the median, not the mean: a single scheduler hiccup or a background
    # process skews a 3-rep mean badly, and these runs are too expensive to
    # repeat enough times for the mean to settle.
    ttfts = sorted(r["ttft_s"] for r in records)
    decs = [r["decode_tps"] for r in records if r["decode_tps"]]
    summary = {
        "label": args.label,
        "tier": args.tier,
        "quant": args.quant,
        "engine": args.engine,
        "workload": args.workload,
        "model": args.model,
        "target_prompt_tokens": args.prompt_tokens,
        "actual_prompt_tokens": records[0].get("actual_prompt_tokens"),
        "ttft_s_median": statistics.median(ttfts),
        "ttft_s_min": ttfts[0],
        "ttft_s_max": ttfts[-1],
        "implied_prefill_tps_median": statistics.median(
            [r["implied_prefill_tps"] for r in records if r["implied_prefill_tps"]]),
        "decode_tps_median": statistics.median(decs) if decs else None,
        "correctness_probe_pass": probe_ok,
        "correctness_probe_detail": probe_detail,
        "reps": len(records),
        "vram_before": vram_before,
        "vram_after": vram_after,
        "notes": args.notes,
        "records": records,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  TTFT median   : {summary['ttft_s_median']:.3f} s")
    print(f"  prefill t/s   : {summary['implied_prefill_tps_median']:.1f} (end-to-end)")
    if summary["decode_tps_median"]:
        print(f"  decode t/s    : {summary['decode_tps_median']:.1f}")
    if vram_after:
        print(f"  VRAM total    : {vram_after['total_gb']} GB")
    if not probe_ok:
        print("  !! correctness probe FAILED -- this backend is producing "
              "wrong output; do not quote its throughput")
    print(f"  wrote {args.out}")
    return 0 if probe_ok else 1


if __name__ == "__main__":
    sys.exit(main())
