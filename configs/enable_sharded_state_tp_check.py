"""Reject sharded_state checkpoints saved at a different TP size than we run.

vLLM's `ShardedStateLoader.load_weights` (in `sharded_state_loader.py`) loads a
TP=2 checkpoint into a TP=1 instance without complaint. We reproduced this: a
Qwen3-30B-A3B checkpoint saved at tensor_parallel_size=2, loaded at TP=1,
logged 194 warnings -- including `model.embed_tokens.weight receiving 75968 of
151936 rows` (exactly half the embedding table, one rank's worth) -- then
printed "Application startup complete" and answered every prompt with
"!!!!!!!!". No exception, no failed health check.

Root cause is in `load_weights`'s per-tensor copy loop:

    for dim, size in enumerate(tensor.shape):
        if size < param_shape[dim]:
            param_data = param_data.narrow(dim, 0, size)

Every rank-N-of-2 shard is smaller than its TP=1 parameter in at least one
dim, so this narrows the destination and copies the shard into the front of
it, leaving the rest of the parameter at its uninitialized allocation value.
The comment above the loop says this narrowing exists for LoRA padding:

    # If loading with LoRA enabled, additional padding may
    # be added to certain parameters. We only load into a
    # narrowed view of the parameter data.

But the guard is only `size < param_shape[dim]` -- a bare inequality, so it
cannot distinguish "this is LoRA padding" from "this is a shard of the wrong
shape entirely". The string `lora` does not appear anywhere else in this file;
nothing computes an expected LoRA pad amount to check against. And the
function already demonstrates it knows how to fail on a size mismatch: a few
lines later, if `state_dict` still has keys left after the loop, it raises
`ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")`. There is
no symmetric check for "shard smaller than expected for a non-LoRA reason".

Fix: infer the checkpoint's saved TP size from its shard filenames (the
pattern's `{rank}` field), and raise before any tensor is touched if it
disagrees with `get_tensor_model_parallel_world_size()`. Checkpoints whose
configured pattern has no `{rank}` field are left alone -- the existing
"could not find checkpoint files" error still covers those.

This is a BACKPORT of a fix submitted upstream:
    branch fix/sharded-state-tp-mismatch on github.com/davetha/vllm
    (see vllm/model_executor/model_loader/sharded_state_loader.py there)
Confirmed still present and unfixed on upstream vllm-project/vllm main as of
this writing. DELETE this script once that PR lands and the installed vLLM
version includes the check.

    python enable_sharded_state_tp_check.py [--revert] [--check] [--assert-patched]

SITE (the site-packages root) defaults to the container's path below, and can
be overridden with --site or the MI210_PATCH_SITE env var -- useful for
testing this script against a scratch copy of the file.
"""
import argparse
import os
import sys

DEFAULT_SITE = "/opt/python/lib/python3.14/site-packages"
SITE = os.environ.get("MI210_PATCH_SITE", DEFAULT_SITE)

_RELPATH = "vllm/model_executor/model_loader/sharded_state_loader.py"


def _target(site: str) -> str:
    return f"{site}/{_RELPATH}"


_IMPORT_OLD = """\
import collections
import glob
import os
import time
"""

_IMPORT_NEW = """\
import collections
import glob
import os
import re
import time
"""

_METHOD_OLD = """\
    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank
"""

_METHOD_NEW = '''\
    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def _saved_tp_size(self, local_model_path: str) -> int | None:
        """Infer the tensor parallel size a sharded checkpoint was saved with.

        Args:
            local_model_path: Directory or S3 prefix holding the checkpoint.

        Returns:
            The number of distinct ranks present, or `None` if the configured
            pattern has no rank field or no matching files were found.
        """
        if "{rank}" not in self.pattern:
            return None
        name_re = re.compile(
            re.escape(self.pattern)
            .replace(re.escape("{rank}"), r"(\\d+)")
            .replace(re.escape("{part}"), r"\\d+")
        )
        any_rank = self.pattern.format(rank="*", part="*")
        if is_s3(local_model_path):
            paths = s3_glob(path=local_model_path, allow_pattern=[f"*{any_rank}"])
        else:
            paths = glob.glob(os.path.join(local_model_path, any_rank))
        ranks = {
            int(m.group(1))
            for p in paths
            if (m := name_re.fullmatch(os.path.basename(p)))
        }
        return len(ranks) or None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
'''

_CHECK_OLD = """\
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = model_weights

        rank = get_tensor_model_parallel_rank()
"""

_CHECK_NEW = """\
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        local_model_path = model_weights

        # Without this, a rank whose shard is smaller than its parameter is
        # narrowed and partially filled below, leaving the model silently
        # half-initialized rather than failing.
        tp_size = get_tensor_model_parallel_world_size()
        saved_tp_size = self._saved_tp_size(local_model_path)
        if saved_tp_size is not None and saved_tp_size != tp_size:
            raise ValueError(
                f"Sharded checkpoint at '{local_model_path}' was saved with "
                f"tensor_parallel_size={saved_tp_size}, but this instance has "
                f"tensor_parallel_size={tp_size}. Sharded checkpoints cannot be "
                f"resharded; re-save it with tensor_parallel_size={tp_size}."
            )

        rank = get_tensor_model_parallel_rank()
"""


def _patches(site: str):
    target = _target(site)
    # (path, old, new, expected occurrence count)
    return [
        (target, _IMPORT_OLD, _IMPORT_NEW, 1),
        (target, _METHOD_OLD, _METHOD_NEW, 1),
        (target, _CHECK_OLD, _CHECK_NEW, 1),
    ]


def apply(
    revert: bool = False,
    check: bool = False,
    assert_patched: bool = False,
    site: str = SITE,
) -> int:
    failures = 0
    for path, old, new, want in _patches(site):
        try:
            text = open(path).read()
        except OSError as exc:
            print(f" ERROR    {path}: {exc}")
            failures += 1
            continue
        name = path.rsplit("/", 1)[-1]
        n_old, n_new = text.count(old), text.count(new)

        if check or assert_patched:
            state = "patched" if n_new == want and n_old == 0 else (
                "unpatched" if n_old == want and n_new == 0 else "UNKNOWN"
            )
            print(f" {state:>9}  {name} "
                  f"(unpatched {n_old} / patched {n_new}, want {want})")
            if assert_patched:
                failures += 0 if state == "patched" else 1
            elif n_old + n_new != want and not (n_old == want and n_new == want):
                # --check just wants to recognize the tree's state; only flag
                # counts that make no sense at all (upstream moved).
                failures += 1
            continue

        if revert:
            old, new = new, old
            n_old, n_new = n_new, n_old

        if n_old == 0 and n_new == want:
            print(f" already    {name}")
            continue
        if n_old != want:
            print(f" ERROR      {name}: expected {want} occurrence(s) of the "
                  f"target text, found {n_old}. Upstream source moved -- "
                  f"re-derive this patch, do not force it.")
            failures += 1
            continue
        open(path, "w").write(text.replace(old, new, want))
        print(f" {'reverted' if revert else 'patched'}   {name}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--check", action="store_true",
                     help="report state, write nothing. Passes on a stock "
                          "tree -- use --assert-patched to gate a build.")
    ap.add_argument("--assert-patched", action="store_true",
                     help="like --check, but exit nonzero unless every site "
                          "is PATCHED.")
    ap.add_argument("--site", default=None,
                     help="override the site-packages root (default: "
                          f"{SITE!r}, or $MI210_PATCH_SITE)")
    args = ap.parse_args()
    site = args.site if args.site is not None else SITE

    failures = apply(
        revert=args.revert,
        check=args.check,
        assert_patched=args.assert_patched,
        site=site,
    )
    if failures:
        print(f"\n{failures} site(s) did not match as expected.")
        return 1
    if not (args.check or args.assert_patched):
        print("\nOK. This is a backport of "
              "github.com/davetha/vllm fix/sharded-state-tp-mismatch -- "
              "delete this script once that PR lands upstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
