# Change Log: FlashAttention Patching Opportunity

## Date: 2026-07-25

## Summary
Identified that the "V cache quantization requires flash_attn" constraint is
a code completeness gap, not a hardware limitation. The non-FA attention path
has K dequantization but is missing V dequantization. Adding it (~20-50 lines)
would unlock compressed V cache with fast FA-off attention.

## Evidence
- Benchmark: FA-off standard attention = 1,984 tok/s (fast)
- Benchmark: FA-on fallback = 770 tok/s (2.6× slower)
- ROCm standalone FA (CK backend): works on gfx90a, verified correct
- gfx90a HAS Matrix Core (contrary to initial assumption)
- The constraint is in llama.cpp source code, not hardware

## What's Needed
1. Find the non-FA attention kernel in llama.cpp source
2. Locate the K dequantization code (template for V)
3. Add V dequantization using the same pattern
4. Remove or relax the "requires flash_attn" check
5. Test: -ctk q4_0 -ctv q4_0 -fa off

## Expected Impact
| Config | Current | With Patch |
|---|---|---|
| q4_0 V + FA-off | ❌ Won't load | ✅ 1984 tok/s prefill, 162 decode |
| KV VRAM savings | 0% (forced to f16 V) | 75% (q4_0 V works with FA-off) |
| Max context | Limited by f16 V size | 4× larger context with q4_0 V |

## Status
Analysis complete. Source code investigation in progress. Patch not yet written.
