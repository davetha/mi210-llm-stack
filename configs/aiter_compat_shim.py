"""sitecustomize.py — AITER 0.1.13 compatibility shim for ATOM v0.1.5.

ATOM v0.1.5 expects a newer AITER than our pinned 0.1.13. This shim adds
stubs for missing symbols so that ATOM's imports succeed. Stubs raise
NotImplementedError when called, so dense model paths (Llama, Qwen) work
while MoE/sparse/MLA-specific paths fail loudly if invoked.

Loaded automatically by Python at startup (sitecustomize.py convention).
"""
import sys
import types
import importlib
import importlib.machinery


def _make_stub(name, kind="function"):
    """Create a stub that raises NotImplementedError when called."""
    if kind == "function":
        def _stub(*args, **kwargs):
            raise NotImplementedError(
                f"aiter.{name} is a stub (not in AITER 0.1.13). "
                f"Required by ATOM v0.1.5 but only needed for advanced model paths."
            )
        return _stub
    elif kind == "class":
        class _StubClass:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    f"aiter.{name} is a stub class (not in AITER 0.1.13)."
                )
        _StubClass.__name__ = name
        _StubClass.__qualname__ = f"aiter.{name}"
        return _StubClass
    elif kind == "contextmanager":
        from contextlib import contextmanager
        @contextmanager
        def _cm(*args, **kwargs):
            yield
        _cm.__name__ = name
        return _cm


def _make_stub_module(fullname, attrs=None):
    """Create a fake Python module with stub attributes."""
    mod = types.ModuleType(fullname)
    mod.__package__ = fullname
    mod.__loader__ = importlib.machinery.BuiltinImporter()
    if attrs:
        for attr_name, attr_value in attrs.items():
            setattr(mod, attr_name, attr_value)
    sys.modules[fullname] = mod
    return mod


def _patch_aiter():
    """Add missing symbols to the aiter namespace."""
    try:
        import aiter
    except ImportError:
        return  # AITER not installed, nothing to patch

    # Context managers (no-op stubs — safe for all paths)
    _cms = [
        "graph_marker", "fast_mode", "intra_batch_mode",
        "maybe_dual_stream_forward",
    ]
    for name in _cms:
        if not hasattr(aiter, name):
            setattr(aiter, name, _make_stub(name, "contextmanager"))

    # Boolean/config flags
    if not hasattr(aiter, "_use_mla_ps_kernel"):
        aiter._use_mla_ps_kernel = lambda *a, **kw: False

    # MoE functions (dense models never call these)
    _moe_funcs = [
        "topk_gating", "indexer_score_topk", "moe_forward",
        "rocm_aiter_fused_moe", "rocm_aiter_biased_grouped_topk_impl",
        "rocm_aiter_grouped_topk_impl",
    ]
    for name in _moe_funcs:
        if not hasattr(aiter, name):
            setattr(aiter, name, _make_stub(name, "function"))

    # GEMM (may have alternative in our AITER)
    if not hasattr(aiter, "tuned_gemm"):
        # Fall back to hipb_mm if available
        if hasattr(aiter, "hipb_mm"):
            aiter.tuned_gemm = aiter.hipb_mm
        else:
            aiter.tuned_gemm = _make_stub("tuned_gemm", "function")

    # Attention classes (dense models use flash_attn_func directly)
    _attn_classes = [
        "unified_attention_with_output_base",
        "unified_attention_with_output",
        "v4_attention_with_output",
        "linear_attention_with_output_base",
        "mla_attention",
        "atom_vllm_mha_attention",
        "atom_vllm_mla_attention",
        "sparse_attn_indexer",
        "sparse_attn_indexer_plugin_mode",
        "sparse_attn_indexer_sglang_plugin_mode",
        "mhc_pre",
    ]
    for name in _attn_classes:
        if not hasattr(aiter, name):
            setattr(aiter, name, _make_stub(name, "class"))

    # Stub missing submodules
    _stub_submodules = {
        "aiter.ops.flydsl.kernels.fused_compress_attn": {
            "flydsl_fused_compress_attn": _make_stub("flydsl_fused_compress_attn"),
        },
        "aiter.ops.flydsl.kernels.fused_compress_attn_hca": {
            "flydsl_fused_compress_attn_hca": _make_stub("flydsl_fused_compress_attn_hca"),
        },
        "aiter.ops.flydsl.linear_attention_prefill_kernels": {
            "flydsl_gdr_prefill": _make_stub("flydsl_gdr_prefill"),
        },
        "aiter.ops.flydsl.moe_common": {
            "GateMode": type("GateMode", (), {"__init__": lambda s, *a, **kw: (_ for _ in ()).throw(NotImplementedError("GateMode stub"))}),
        },
        "aiter.ops.pa_sparse_prefill_opus": {
            "pa_sparse_prefill_opus": _make_stub("pa_sparse_prefill_opus"),
        },
        "aiter.ops.triton.attention.mla": {
            "mla_decode_fwd": _make_stub("mla_decode_fwd"),
            "mla_prefill_fwd": _make_stub("mla_prefill_fwd"),
        },
        "aiter.ops.triton.kv_cache": {
            "cat_and_cache_mla": _make_stub("cat_and_cache_mla"),
        },
    }

    for modname, attrs in _stub_submodules.items():
        if modname not in sys.modules:
            try:
                importlib.import_module(modname)
            except ImportError:
                _make_stub_module(modname, attrs)


# Apply the patch at import time
_patch_aiter()
