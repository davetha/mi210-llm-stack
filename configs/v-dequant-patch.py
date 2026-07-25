#!/usr/bin/env python3
"""V dequant patch for non-FA attention path.
Allows quantized V cache without FlashAttention by dequantizing V before transpose.
"""
import sys
import re

CONTEXT_FILE = "/mnt/llm-storage/turbo-build/src/src/llama-context.cpp"
GRAPH_FILE = "/mnt/llm-storage/turbo-build/src/src/llama-graph.cpp"

def patch_context_runtime():
    """Relax the runtime check in llama_context init that throws on quantized V without FA."""
    with open(CONTEXT_FILE, "r") as f:
        content = f.read()

    old = '''        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }'''

    new = '''        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                LLAMA_LOG_WARN("%s: quantized V cache without flash_attn - will dequantize V in non-FA path\\n", __func__);
            }
        }'''

    if old not in content:
        print("ERROR: Runtime check block not found in llama-context.cpp")
        return False

    if 'will dequantize V in non-FA path' in content and new in content:
        print("PATCH 1B ALREADY APPLIED: llama-context.cpp runtime check")
        return True

    content = content.replace(old, new)
    with open(CONTEXT_FILE, "w") as f:
        f.write(content)
    print("PATCH 1B APPLIED: llama-context.cpp runtime check relaxed to warning")
    return True

def patch_context():
    """Relax the constraint: log warning instead of returning nullptr."""
    with open(CONTEXT_FILE, "r") as f:
        content = f.read()

    old = '''    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\\n", __func__);
        return nullptr;
    }'''

    new = '''    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_WARN("%s: V cache quantization without flash_attn - will dequantize V in non-FA path\\n", __func__);
    }'''

    if old not in content:
        print("ERROR: Constraint block not found in llama-context.cpp")
        print("Looking for similar patterns...")
        for line_no, line in enumerate(content.split('\n'), 1):
            if 'V cache quantization requires' in line:
                print(f"  Line {line_no}: {line}")
        return False

    if new in content:
        print("PATCH 1 ALREADY APPLIED: llama-context.cpp")
        return True

    content = content.replace(old, new)
    with open(CONTEXT_FILE, "w") as f:
        f.write(content)
    print("PATCH 1 APPLIED: llama-context.cpp - constraint relaxed to warning")
    return True

def patch_graph():
    """Add dequantize step before transpose in non-FA V path."""
    with open(GRAPH_FILE, "r") as f:
        content = f.read()

    # The non-FA V path at ~line 2180
    old = '''        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }'''

    new = '''        if (!v_trans) {
            // V-dequant patch: dequantize V to F16 before transpose.
            // Block quantization formats (q4_0, q4_1, q8_0, kivi2, turbo3) cannot be transposed
            // in quantized form - the block layout assumes contiguous rows.
            // Without this dequant, quantized V cache requires FlashAttention.
            if (ggml_is_quantized(v->type)) {
                struct ggml_tensor * v_f16 = ggml_new_tensor(ctx0, GGML_TYPE_F16, ggml_n_dims(v), v->ne);
                v = ggml_cpy(ctx0, v, v_f16);
                cb(v, "v_dequant", il);
            }
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }'''

    if old not in content:
        print("ERROR: V transpose block not found in llama-graph.cpp")
        print("Searching for similar patterns...")
        for line_no, line in enumerate(content.split('\n'), 1):
            if 'v_cont' in line or ('v_trans' in line and 'if' in line):
                print(f"  Line {line_no}: {line.rstrip()}")
        return False

    if 'v_dequant' in content:
        print("PATCH 2 ALREADY APPLIED: llama-graph.cpp")
        return True

    content = content.replace(old, new)
    with open(GRAPH_FILE, "w") as f:
        f.write(content)
    print("PATCH 2 APPLIED: llama-graph.cpp - V dequant added before transpose")
    return True

if __name__ == "__main__":
    ok1a = patch_context()
    ok1b = patch_context_runtime()
    ok2 = patch_graph()
    if ok1a and ok1b and ok2:
        print("\n=== V DEQUANT PATCH SUCCESSFUL ===")
        print("All 3 patches applied. Ready to build.")
        sys.exit(0)
    else:
        print("\n=== PATCH FAILED ===")
        sys.exit(1)
