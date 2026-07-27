"""Measure vLLM serving throughput and latency against an OpenAI-compatible endpoint.

Written for the gfx90a AITER ASM A/B (see vllm-aiter-asm-gfx90a.md), where the
question is whether AITER's hand-written ASM attention kernels change real
serving numbers, not microbenchmark numbers. That demands the two runs differ
in exactly one thing, so this harness pins everything a server would otherwise
be free to vary:

  * Prompt length is exact, verified by re-encoding, not guessed from word count.
  * Every prompt is unique, because vLLM enables prefix caching by default and
    identical prompts would turn prefill into a cache hit -- measuring nothing.
    Servers under test should ALSO be started with --no-enable-prefix-caching;
    the unique prompts are belt and braces.
  * Output length is fixed with ignore_eos, so decode work is identical across
    runs and TPOT is comparable.
  * Concurrency is a true in-flight limit: each worker issues its next request
    the moment the previous returns.

TTFT comes from the first streamed chunk that carries text. TPOT is the mean
inter-token time after that first token, which is the number that actually
tracks decode-kernel speed.

    python bench_vllm_serving.py --label asm-on --out results-asm-on.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp

# (prompt_tokens, concurrency) -- short and long prompts at three in-flight depths.
DEFAULT_CASES = [
    (128, 1), (128, 8), (128, 32),
    (4096, 1), (4096, 8), (4096, 32),
]


WORDS = (
    "system memory kernel buffer thread cache latency vector matrix tensor "
    "compile register pointer stream socket packet cluster gradient weight "
    "attention decode prefill throughput bandwidth pipeline scheduler entropy "
    "sequence context window token embedding channel residual normalize scale"
).split()


def build_prompts(tokenizer, n: int, prompt_len: int, seed: int,
                  style: str = "tokens") -> list[str]:
    """n distinct prompts, each exactly prompt_len tokens when re-encoded.

    style="tokens" draws random token ids, which maximises entropy and defeats
    any prefix cache. Some servers reject the resulting text -- llama.cpp drops
    the connection on a third of them -- so style="words" builds from a small
    word list instead, still unique per prompt but always valid UTF-8 text.

    Either way the result is re-encoded and trimmed/padded until the length is
    exact, because decode/encode is not a bijection.
    """
    rng = random.Random(seed)
    vocab = tokenizer.vocab_size
    special = set(tokenizer.all_special_ids)
    prompts = []
    for _ in range(n):
        if style == "words":
            text = " ".join(rng.choice(WORDS) for _ in range(prompt_len * 2))
        else:
            ids = []
            while len(ids) < prompt_len:
                tid = rng.randrange(vocab)
                if tid not in special:
                    ids.append(tid)
            text = tokenizer.decode(ids)
        # Re-encode and correct the length, which drifts because decode/encode
        # is not a bijection.
        for _ in range(8):
            enc = tokenizer.encode(text, add_special_tokens=False)
            if len(enc) == prompt_len:
                break
            if len(enc) > prompt_len:
                text = tokenizer.decode(enc[:prompt_len])
            elif style == "words":
                text = text + " " + " ".join(
                    rng.choice(WORDS) for _ in range(prompt_len - len(enc)))
            else:
                pad = [t for t in (rng.randrange(vocab) for _ in range(prompt_len - len(enc)))
                       if t not in special]
                text = text + tokenizer.decode(pad)
        prompts.append(text)
    return prompts


async def one_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    tok_times = []
    ntok = 0
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            choices = chunk.get("choices") or []
            if not choices or not choices[0].get("text"):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            tok_times.append(now)
            ntok += 1
    end = time.perf_counter()
    if ttft is None:
        raise RuntimeError("stream produced no text")
    # Decode time excludes the prefill that produced the first token.
    tpot = ((tok_times[-1] - tok_times[0]) / (len(tok_times) - 1)
            if len(tok_times) > 1 else float("nan"))
    return {"ttft": ttft, "tpot": tpot, "ntok": ntok, "latency": end - t0}


async def run_case(base_url, model, prompts, concurrency, max_tokens, n_requests,
                   force_close=False):
    url = f"{base_url}/v1/completions"
    queue = asyncio.Queue()
    for i in range(n_requests):
        queue.put_nowait(prompts[i % len(prompts)])

    results, errors = [], []
    timeout = aiohttp.ClientTimeout(total=1800)
    # llama.cpp closes the socket after each streamed response, so a pooled
    # keep-alive connection is dead by the next request and every other one
    # fails with ServerDisconnectedError. force_close dials a fresh connection
    # each time. vLLM honours keep-alive, so it does not need this.
    conn = aiohttp.TCPConnector(limit=concurrency + 4, force_close=force_close)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        async def worker():
            while True:
                try:
                    prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    results.append(await one_request(session, url, model, prompt, max_tokens))
                except Exception as exc:  # surfaced, never silently dropped
                    errors.append(repr(exc))

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0

    if errors:
        print(f"    {len(errors)} request(s) FAILED, first: {errors[0]}", file=sys.stderr)
    if not results:
        return None

    out_tok = sum(r["ntok"] for r in results)
    return {
        "concurrency": concurrency,
        "n_requests": len(results),
        "n_errors": len(errors),
        "wall_s": wall,
        "output_toks": out_tok,
        "output_tok_per_s": out_tok / wall,
        "ttft_ms_mean": statistics.mean(r["ttft"] for r in results) * 1e3,
        "ttft_ms_p50": statistics.median(r["ttft"] for r in results) * 1e3,
        "tpot_ms_mean": statistics.mean(r["tpot"] for r in results) * 1e3,
        "latency_s_mean": statistics.mean(r["latency"] for r in results),
    }


async def main_async(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    async with aiohttp.ClientSession() as s:
        async with s.get(f"{args.base_url}/v1/models") as r:
            served = (await r.json())["data"][0]["id"]
    print(f"server reports model: {served}")

    cases = DEFAULT_CASES if not args.cases else [
        tuple(int(x) for x in c.split(":")) for c in args.cases
    ]

    out = {"label": args.label, "model": served, "max_tokens": args.max_tokens, "cases": []}
    for prompt_len, conc in cases:
        n_requests = max(args.min_requests, conc * args.requests_per_worker)
        prompts = build_prompts(tokenizer, n_requests, prompt_len,
                                seed=args.seed, style=args.prompt_style)

        # Warm up so the first measured request never pays JIT or allocator cost.
        await run_case(args.base_url, served, prompts[:2], min(2, conc), 16,
                       min(2, conc), force_close=args.force_close)

        print(f"  prompt={prompt_len:>5} conc={conc:>3} n={n_requests} ...", flush=True)
        res = await run_case(args.base_url, served, prompts, conc,
                             args.max_tokens, n_requests,
                             force_close=args.force_close)
        if res is None:
            print("    all requests failed; recording failure", file=sys.stderr)
            res = {"concurrency": conc, "failed": True}
        res["prompt_tokens"] = prompt_len
        out["cases"].append(res)
        if not res.get("failed"):
            print(f"    ttft {res['ttft_ms_mean']:8.1f} ms | tpot {res['tpot_ms_mean']:6.2f} ms"
                  f" | {res['output_tok_per_s']:8.1f} out tok/s")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests-per-worker", type=int, default=3)
    ap.add_argument("--min-requests", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-style", choices=["tokens", "words"], default="tokens",
                    help="'words' for servers that reject random-token text")
    ap.add_argument("--force-close", action="store_true",
                    help="new TCP connection per request; required for llama.cpp")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="override cases as promptlen:concurrency, e.g. 128:8 4096:32")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
