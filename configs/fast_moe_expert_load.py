"""Make vLLM's per-expert MoE weight load fast at TP>1.

Loading a MoE checkpoint with tensor parallelism is pathologically slow, and it
is not I/O. Measured on 2x MI210:

    Qwen3-30B-A3B  bf16        TP=2   697 s per shard   (~3 h for 61 GB)
    Qwen3-235B     GPTQ-Int4   TP=2   810 s per shard   (~7 h for 32 shards)

while the same files read at **3.0 GB/s** with `dd`. `py-spy` on the pinned
worker puts every sample in the same place:

    _load_w13 (fused_moe/layer.py:762)
    weight_loader (fused_moe/layer.py:1138)
    load_weights (models/qwen3_moe.py:627)

The cause
---------
In `_load_w13` / `_load_w2`, the checkpoint tensor is sliced per TP rank:

    loaded_weight = loaded_weight.narrow(shard_dim, start_offset, narrow_size)
    ...
    expert_data.copy_(loaded_weight)

`loaded_weight` is an **mmap'd CPU tensor**, and `narrow()` on the shard
dimension returns a **non-contiguous view** of it. `copy_()` from a
non-contiguous host tensor to device cannot issue a single DMA -- it degrades
into a strided gather. That runs once per expert per layer: 128 experts x 48
layers = 6,144 strided host-to-device transfers on the 30B, and 128 x 94 on the
235B.

The narrowing only happens when `tp_size > 1`, which is why TP=1 loads in
seconds (18 s per shard measured on the 80B) and TP=2 takes hours.

The fix
-------
Materialise the slice contiguously on the host before the transfer, so the copy
becomes one linear DMA instead of a gather:

    if not loaded_weight.is_contiguous():
        loaded_weight = loaded_weight.contiguous()
    expert_data.copy_(loaded_weight)

Memory cost is bounded and small: `.contiguous()` copies **one expert's slice**,
not the tensor -- single-digit MB -- and it is freed immediately. This is not
the same as calling `.contiguous()` on a whole checkpoint tensor, which would
be reckless.

Why this might not be the whole story
-------------------------------------
`expert_data` is itself a narrowed **device** tensor and may also be
non-contiguous, making the destination a scatter. That cost is device-side and
far cheaper than a host-side gather over PCIe, so it is left alone -- but if
this patch does not deliver the expected speedup, the destination side is the
next place to look, not the source.

Verify by load time, which is printed by vLLM itself:

    INFO [gpu_model_runner.py] Model loading took X GiB memory and Y seconds

    python fast_moe_expert_load.py [--revert] [--check]
"""
import argparse
import glob
import os
import sys

REL = "vllm/model_executor/layers/fused_moe/layer.py"

# The same statement appears at several call sites with different indentation.
# Each is patched independently so a partial match reports honestly rather than
# silently leaving hot paths untouched.
#
# Every pattern is anchored on a LEADING NEWLINE. Without it the 8-space form is
# a substring of the 12- and 16-space forms, so counts and replacements collide
# across call sites -- an 8-space "match" would fire inside every deeper one and
# corrupt the indentation it inserted.
_SITES = [
    (
        "\n        expert_data.copy_(loaded_weight)\n",
        "\n        # Contiguous source: narrow() on the shard dim leaves an mmap'd\n"
        "        # CPU view strided, and copy_ to device then degrades from one\n"
        "        # DMA into a per-element gather, once per expert per layer.\n"
        "        # Slice-sized allocation (single-digit MB), freed immediately.\n"
        "        if not loaded_weight.is_contiguous():\n"
        "            loaded_weight = loaded_weight.contiguous()\n"
        "        expert_data.copy_(loaded_weight)\n",
    ),
    (
        "\n            expert_data.copy_(loaded_weight)\n",
        "\n            # Contiguous source -- see the note at the 8-space site.\n"
        "            if not loaded_weight.is_contiguous():\n"
        "                loaded_weight = loaded_weight.contiguous()\n"
        "            expert_data.copy_(loaded_weight)\n",
    ),
    (
        "\n                expert_data.copy_(loaded_weight)\n",
        "\n                # Contiguous source -- see the note at the 8-space site.\n"
        "                if not loaded_weight.is_contiguous():\n"
        "                    loaded_weight = loaded_weight.contiguous()\n"
        "                expert_data.copy_(loaded_weight)\n",
    ),
]


def find_layer_py() -> str:
    override = os.environ.get("VLLM_MOE_LAYER_PATH")
    if override:
        return override
    roots = []
    try:
        import vllm

        vllm_file = getattr(vllm, "__file__", None)
        if vllm_file:
            roots.append(os.path.dirname(os.path.dirname(vllm_file)))
    except ImportError:
        pass
    for pat in ("/opt/python/lib/python*/site-packages",
                "/usr/lib/python*/site-packages",
                "/usr/local/lib/python*/*-packages"):
        roots.extend(glob.glob(pat))
    for base in roots:
        cand = os.path.join(base, REL)
        if os.path.exists(cand):
            return cand
    raise SystemExit(f"cannot find {REL}. Set VLLM_MOE_LAYER_PATH.")


def apply(path: str, revert: bool = False, check: bool = False) -> int:
    try:
        text = open(path).read()
    except OSError as exc:
        print(f" ERROR    {path}: {exc}")
        return 1

    original = text
    patched_n = 0
    plain_n = 0

    for plain, wrapped in _SITES:
        if revert:
            n = text.count(wrapped)
            if n:
                text = text.replace(wrapped, plain)
                patched_n += n
            continue
        n_wrapped = text.count(wrapped)
        # Count plain occurrences that are NOT already inside a wrapped block.
        n_plain = text.count(plain) - n_wrapped
        patched_n += n_wrapped
        plain_n += n_plain
        if not check and n_plain:
            text = text.replace(wrapped, "\0")           # protect done sites
            text = text.replace(plain, wrapped)
            text = text.replace("\0", wrapped)

    name = os.path.basename(path)
    if check:
        print(f" {'patched' if patched_n and not plain_n else 'unpatched':>9}  "
              f"{name}  ({plain_n} unpatched / {patched_n} patched call sites)")
        print(f"           {path}")
        return 0 if (plain_n + patched_n) else 1

    if text == original:
        print(f" already    {name}  ({patched_n} call sites already patched)")
        return 0

    open(path, "w").write(text)
    action = "reverted" if revert else "patched"
    print(f" {action:<10} {name}")
    print(f"            {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    path = args.path or find_layer_py()
    rc = apply(path, revert=args.revert, check=args.check)
    if rc == 0 and not args.check and not args.revert:
        print("\nOK. Compare against the pre-patch figure in vLLM's own log:")
        print("    Model loading took X GiB memory and Y seconds")
        print("Baseline on this hardware: 697 s/shard (30B bf16 TP=2),")
        print("810 s/shard (235B GPTQ-Int4 TP=2).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
