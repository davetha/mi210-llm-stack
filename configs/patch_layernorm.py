import re

path = "/opt/python/lib/python3.14/site-packages/sglang/srt/layers/layernorm.py"
with open(path) as f:
    src = f.read()

# Find and replace the 6-arg call with 4-arg in-place call
# The pattern in forward_hip around line 265
old = '''            fused_add_rms_norm(
                out, x, residual_out, residual, self.weight.data, self.variance_epsilon
            )
            return out, residual_out'''

new = '''            fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual'''

if old in src:
    src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)
    print("PATCHED: fused_add_rms_norm 6-arg -> 4-arg")
else:
    print("ERROR: pattern not found")
    # Show what's actually there for debugging
    for i, line in enumerate(src.split('\n')):
        if 'fused_add_rms_norm' in line and 'forward_hip' not in line:
            print(f"  line {i+1}: {line.strip()}")
