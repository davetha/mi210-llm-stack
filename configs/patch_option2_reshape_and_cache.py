#!/usr/bin/env python3
"""
Option 2 Patch: Force reshape_and_cache(asm_layout=True) in ATOM's attention_mha.py.

Bypasses fused_qk_norm_rope_cache_quant_shuffle (branch 1) and
fused_qk_rope_reshape_and_cache (branch 2), forcing the else branch (branch 3)
which uses separate RoPE + Q/K norm + aiter.reshape_and_cache(asm_layout=True).

This is the proven-correct write path from standalone testing.

Usage:
  python patch_option2_reshape_and_cache.py [--revert]
"""

import sys
import re
from pathlib import Path

ATTN_PATH = Path("/opt/python/lib/python3.14/site-packages/atom/model_ops/attention_mha.py")

def read_file():
    return ATTN_PATH.read_text()

def write_file(content):
    ATTN_PATH.write_text(content)

def is_patched(content):
    return "ATOM_FORCE_ASM_CACHE_WRITE" in content

def apply_patch():
    content = read_file()
    if is_patched(content):
        print("Already patched.")
        return

    # Patch 1: Add env var check to skip fused branches on gfx90a
    # Branch 1: fused_qk_norm_rope_cache_quant_shuffle
    old_cond1 = """        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
        ):"""

    new_cond1 = """        from aiter.jit.utils.chip_info import get_gfx as _get_gfx
        _force_asm_cache = _get_gfx() == "gfx90a"
        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
            and not _force_asm_cache
        ):"""

    if old_cond1 not in content:
        print("ERROR: Could not find branch 1 conditional. File may have changed.")
        sys.exit(1)
    content = content.replace(old_cond1, new_cond1)

    # Branch 2: fused_qk_rope_reshape_and_cache
    old_cond2 = "        elif use_triton_attn and self.rotary_emb is not None:"
    new_cond2 = "        elif use_triton_attn and self.rotary_emb is not None and not _force_asm_cache:"

    if old_cond2 not in content:
        print("ERROR: Could not find branch 2 conditional.")
        sys.exit(1)
    content = content.replace(old_cond2, new_cond2)

    write_file(content)
    print("Patch applied: forced reshape_and_cache(asm_layout=True) for gfx90a.")
    print("Both fused branches (fused_qk_norm_rope_cache_quant_shuffle and")
    print("fused_qk_rope_reshape_and_cache) are bypassed on gfx90a.")

def revert_patch():
    content = read_file()
    if not is_patched(content):
        print("Not patched, nothing to revert.")
        return

    # Revert branch 1
    content = content.replace(
        """        from aiter.jit.utils.chip_info import get_gfx as _get_gfx
        _force_asm_cache = _get_gfx() == "gfx90a"
        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
            and not _force_asm_cache
        ):""",
        """        if (
            self.rotary_emb is not None
            and self.q_norm is not None
            and self.k_norm is not None
        ):"""
    )

    # Revert branch 2
    content = content.replace(
        "        elif use_triton_attn and self.rotary_emb is not None and not _force_asm_cache:",
        "        elif use_triton_attn and self.rotary_emb is not None:"
    )

    write_file(content)
    print("Patch reverted.")

if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert_patch()
    else:
        apply_patch()
