"""Download a Hugging Face model with aria2c, selectively.

`big` has no pip and no huggingface_hub, but it does have aria2c, which pulls at
~108 MiB/s with -x16 against the HF CDN versus ~15 MiB/s for a single-stream
wget. So this resolves the file list from the HF API and hands aria2c a download
manifest, rather than installing a Python toolchain on the host.

**Selectivity is the point, not a nicety.** A naive "download the whole repo"
across this benchmark matrix would pull several TB of material that is never
loaded: `original/` directories holding a second copy of the weights in another
format, consolidated single-file variants that duplicate the sharded ones, and
in GGUF repos *every* quantization when only one is wanted. `--include` and the
default exclusions exist because the volume has 1.8 TB free and the full matrix
does not fit without staging.

    # a safetensors model, everything needed to load it
    python3 fetch_model.py Qwen/Qwen3-30B-A3B-Thinking-2507 /mnt/llm-storage/t35-bf16

    # one quant out of a GGUF repo that holds a dozen
    python3 fetch_model.py unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF \
        /mnt/llm-storage/t35-gguf-q4km --include 'Q4_K_M'

    # see what it would do, and how big, without downloading
    python3 fetch_model.py <repo> <dir> --include Q8_0 --dry-run

Always `--dry-run` first when the repo is unfamiliar. The printed total is the
number that decides whether it fits.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

API = "https://huggingface.co/api/models"
CDN = "https://huggingface.co"

# Never useful for loading a model, and occasionally enormous. `original/` in
# particular is a full second copy of the weights in the upstream format.
DEFAULT_EXCLUDE = (
    ".gitattributes", "README.md", "LICENSE", ".msgpack", ".h5",
    "original/", "onnx/", "coreml/", "/.cache",
)


def api_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mi210-bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as fh:
        return json.loads(fh.read())


def list_files(repo, revision="main"):
    tree = api_json(f"{API}/{repo}/tree/{revision}?recursive=1")
    out = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        out.append((entry["path"], size))
    return out


def select(files, include, exclude_extra):
    excl = list(DEFAULT_EXCLUDE) + list(exclude_extra or [])
    keep = []
    for path, size in files:
        if any(x in path for x in excl):
            continue
        if include:
            # A config/tokenizer file is always needed even when --include names
            # a weight pattern, otherwise the download is unloadable. Match the
            # pattern OR be a small non-weight file.
            is_weight = path.endswith((".safetensors", ".gguf", ".bin", ".pt"))
            if is_weight and not any(inc.lower() in path.lower() for inc in include):
                continue
        keep.append((path, size))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HF repo id, e.g. Qwen/Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("dest", help="destination directory")
    ap.add_argument("--include", action="append", default=None,
                    help="substring a WEIGHT file must contain (repeatable). "
                         "Non-weight files are always kept.")
    ap.add_argument("--exclude", action="append", default=None)
    ap.add_argument("--revision", default="main")
    # Parallelism goes ACROSS files, never within one. Hugging Face serves LFS
    # weights through its Xet CDN, which signs each redirect for a SPECIFIC byte
    # range -- the policy embeds `ByteRange.ExpectedHeader`. aria2 resolves the
    # redirect once and reuses it for every connection in a split, so with
    # --split=16 exactly one connection is inside the signed range and the other
    # fifteen get HTTP 403. Symptom is a download that appears to progress (some
    # shards land) while others abort with errorCode=22 status=403.
    # Downloading whole files concurrently gets the same aggregate bandwidth
    # without ever issuing an unsigned range request.
    ap.add_argument("--connections", type=int, default=1,
                    help="connections per FILE. Leave at 1: >1 breaks on "
                         "Xet-backed HF repos (per-range signed URLs -> 403).")
    ap.add_argument("--concurrent", type=int, default=8,
                    help="files downloaded at once. This is the real throughput knob.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        files = list_files(args.repo, args.revision)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot list {args.repo}: {exc}", file=sys.stderr)
        return 2
    if not files:
        print(f"ERROR: {args.repo} listed zero files", file=sys.stderr)
        return 2

    keep = select(files, args.include, args.exclude)
    if not keep:
        print(f"ERROR: filter matched nothing. Repo has:", file=sys.stderr)
        for p, s in sorted(files, key=lambda x: -x[1])[:25]:
            print(f"   {s/1e9:8.2f} GB  {p}", file=sys.stderr)
        return 2

    total = sum(s for _, s in keep)
    weights = [(p, s) for p, s in keep if s > 10_000_000]
    print(f"repo   : {args.repo}@{args.revision}")
    print(f"dest   : {args.dest}")
    print(f"files  : {len(keep)} selected of {len(files)} ({len(weights)} large)")
    print(f"total  : {total/1e9:.1f} GB")
    for p, s in sorted(weights, key=lambda x: -x[1])[:12]:
        print(f"   {s/1e9:8.2f} GB  {p}")
    if len(weights) > 12:
        print(f"   ... and {len(weights)-12} more")

    if args.dry_run:
        print("\n(dry run, nothing downloaded)")
        return 0

    # Fail before downloading rather than halfway through: a partial model that
    # still mmaps will load with a zeroed tail and degenerate into repeated
    # tokens, which is far harder to diagnose than a refusal up front.
    st = os.statvfs(os.path.dirname(os.path.abspath(args.dest)) or "/")
    free = st.f_bavail * st.f_frsize
    if free < total * 1.05:
        print(f"\nERROR: need ~{total/1e9:.0f} GB, only {free/1e9:.0f} GB free "
              f"on that filesystem. Free space or stage elsewhere.", file=sys.stderr)
        return 2

    os.makedirs(args.dest, exist_ok=True)

    # Purge unresumable partials before starting.
    #
    # aria2 writes a `.aria2` control file recording which byte ranges of a
    # download have actually landed. With that file present, --continue resumes
    # correctly. WITHOUT it, aria2 falls back to "resume from current EOF" --
    # and a file left over from a multi-connection attempt can have HOLES in the
    # middle, because separate connections write at separate offsets. Resuming
    # such a file appends to the end and yields a full-SIZED file with zeroed
    # gaps: it passes the size check below, mmaps fine, and produces garbage
    # tokens at inference. That is the worst possible failure -- silent.
    # So: any incomplete file with no control file is deleted, not resumed.
    purged = 0
    for path, size in keep:
        full = os.path.join(args.dest, path)
        if not os.path.exists(full) or not size:
            continue
        have = os.path.getsize(full)
        if have == size:
            continue
        if not os.path.exists(full + ".aria2"):
            os.remove(full)
            purged += 1
    if purged:
        print(f"purged {purged} unresumable partial file(s) "
              f"(incomplete, no aria2 control file -- may contain holes)")

    manifest = os.path.join(args.dest, ".aria2-input.txt")
    with open(manifest, "w") as fh:
        for path, _ in keep:
            url = f"{CDN}/{args.repo}/resolve/{args.revision}/{path}"
            fh.write(f"{url}\n")
            fh.write(f"  dir={os.path.join(args.dest, os.path.dirname(path))}\n")
            fh.write(f"  out={os.path.basename(path)}\n")

    cmd = [
        "aria2c", "--input-file", manifest,
        f"--max-connection-per-server={args.connections}",
        f"--split={args.connections}",
        f"--max-concurrent-downloads={args.concurrent}",
        "--continue=true", "--auto-file-renaming=false",
        "--allow-overwrite=false", "--file-allocation=none",
        "--summary-interval=30", "--console-log-level=warn",
        "--retry-wait=5", "--max-tries=5",
    ]
    print("\n" + " ".join(cmd) + "\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"\naria2c exited {rc} -- re-run the same command to resume; "
              f"partial files are kept and --continue picks them up.",
              file=sys.stderr)
        return rc

    # A model that is short a shard loads anyway under mmap and produces
    # garbage, so verify byte-for-byte against the API sizes before declaring
    # success. This check is the whole reason to prefer this over a bare wget.
    bad = []
    for path, size in keep:
        full = os.path.join(args.dest, path)
        if not os.path.exists(full):
            bad.append((path, "missing", size, 0))
        elif size and os.path.getsize(full) != size:
            bad.append((path, "size", size, os.path.getsize(full)))
    if bad:
        print(f"\nERROR: {len(bad)} file(s) incomplete:", file=sys.stderr)
        for p, why, want, got in bad[:10]:
            print(f"   {why:8s} {p}  want {want} got {got}", file=sys.stderr)
        return 2

    os.remove(manifest)
    print(f"\nOK: {len(keep)} files, {total/1e9:.1f} GB verified in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
