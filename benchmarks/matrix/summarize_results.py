"""Build the matrix table from the result JSONs. Never transcribe by hand.

Every row comes from a file in results/. Arms that FAILED appear in the table
with their reason rather than vanishing from it -- "which quantization wins on
MI210" is not answerable without also knowing which ones will not run at all,
and a format silently missing from a results table reads as untested rather
than as broken.

    python3 summarize_results.py [--results DIR] [--markdown]
"""
import argparse
import glob
import json
import os


def load(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! unreadable {os.path.basename(path)}: {exc}")
            continue
        rows.append(d)
    return rows


def fmt(v, spec="", dash="-"):
    if v is None:
        return dash
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = load(args.results)
    if not rows:
        print(f"no results in {args.results}")
        return 1

    failed = [r for r in rows if r.get("status") == "FAILED"]
    ok = [r for r in rows if r.get("status") != "FAILED"]

    # One line per (label, workload). A label appears twice, once per workload,
    # which is what makes the cold-vs-long comparison readable side by side.
    ok.sort(key=lambda r: (str(r.get("tier")), str(r.get("quant")),
                           str(r.get("engine")), str(r.get("workload"))))

    cols = ("tier", "quant", "engine", "workload", "ctx", "TTFT s",
            "prefill t/s", "decode t/s", "weights GiB", "KV GiB", "ok")

    def cells(r):
        fp = r.get("engine_footprint") or {}
        return (
            str(r.get("tier", "?")),
            str(r.get("quant", "?")),
            str(r.get("engine", "?")),
            str(r.get("workload", "?")),
            fmt(r.get("actual_prompt_tokens") or r.get("target_prompt_tokens"), ",d"),
            fmt(r.get("ttft_s_median"), ".2f"),
            fmt(r.get("implied_prefill_tps_median"), ",.0f"),
            fmt(r.get("decode_tps_median"), ".1f"),
            fmt(fp.get("weights_gib"), ".2f"),
            fmt(fp.get("kv_cache_gib"), ".1f"),
            "PASS" if r.get("correctness_probe_pass", r.get("all_correct")) else "FAIL",
        )

    table = [cells(r) for r in ok]

    if args.markdown:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for row in table:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(cols[i]), *(len(r[i]) for r in table)) if table
                  else len(cols[i]) for i in range(len(cols))]
        print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
        print("  ".join("-" * w for w in widths))
        for row in table:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))

    if failed:
        print("\nFAILED ARMS (kept deliberately -- a format that cannot run is "
              "a result, not an omission):")
        for r in failed:
            print(f"  {r.get('tier','?'):>4} {r.get('quant','?'):<8} "
                  f"{r.get('engine','?'):<12} {r.get('reason','?')}")
            tail = (r.get("server_log_tail") or "")
            for line in tail.splitlines():
                if any(k in line for k in ("Error", "error:", "NotImplementedError",
                                           "ValueError", "RuntimeError")):
                    print(f"        {line.strip()[:150]}")
                    break

    print(f"\n{len(ok)} result(s), {len(failed)} failed arm(s).")
    bad = [r for r in ok
           if not r.get("correctness_probe_pass", r.get("all_correct"))]
    if bad:
        print(f"!! {len(bad)} arm(s) failed their correctness probe -- their "
              f"throughput must not be quoted:")
        for r in bad:
            print(f"     {r.get('label')} / {r.get('workload')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
