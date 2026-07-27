#!/usr/bin/env python3
"""
End-to-end test: pa_fwd_asm through ATOM's default dispatch (no ATOM_USE_UNIFIED_ATTN).
Tests Option 2 patch: forced reshape_and_cache(asm_layout=True).

Run WITHOUT ATOM_USE_UNIFIED_ATTN to get pa_fwd_asm decode path:
  python -m atom.examples.simple_inference \
    --model Qwen/Qwen3-0.6B --tensor-parallel-size 1 \
    --max-model-len 256 --max-tokens 10 --block-size 64 --enforce-eager --level 0
"""

import os
import sys
import time
import subprocess
import signal

def test_no_env():
    """Test WITHOUT ATOM_USE_UNIFIED_ATTN — should use pa_fwd_asm for decode."""
    print("\n" + "="*60)
    print("  Test: pa_fwd_asm decode (no ATOM_USE_UNIFIED_ATTN)")
    print("="*60)
    
    env = os.environ.copy()
    # Do NOT set ATOM_USE_UNIFIED_ATTN — we want default dispatch
    env.pop("ATOM_USE_UNIFIED_ATTN", None)
    env["ATOM_LOADER_USE_THREADPOOL"] = "0"
    
    cmd = [
        sys.executable, "-m", "atom.examples.simple_inference",
        "--model", "Qwen/Qwen3-0.6B",
        "--tensor-parallel-size", "1",
        "--max-model-len", "256",
        "--max-tokens", "10",
        "--block-size", "64",
        "--enforce-eager",
        "--level", "0",
    ]
    
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  ENV: ATOM_USE_UNIFIED_ATTN unset")
    print()
    
    start = time.time()
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid
    )
    
    try:
        stdout, _ = proc.communicate(timeout=300)
        elapsed = time.time() - start
        rc = proc.returncode
        
        # Check for memory fault
        if "memory access fault" in stdout.lower() or "SIGSEGV" in stdout:
            print("  ❌ FAIL: Memory access fault detected!")
            # Print last 30 lines for debugging
            lines = stdout.strip().split('\n')
            for line in lines[-30:]:
                print(f"  {line}")
            return False
        
        if rc == 0:
            print(f"  ✅ PASS: EXIT=0, {elapsed:.1f}s")
            # Print generated text
            for line in stdout.split('\n'):
                if any(kw in line.lower() for kw in ['output', 'generated', 'text', 'token']):
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"  ❌ FAIL: EXIT={rc}, {elapsed:.1f}s")
            lines = stdout.strip().split('\n')
            for line in lines[-20:]:
                print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("  ❌ FAIL: Timeout (300s)")
        return False

def test_with_unified():
    """Test WITH ATOM_USE_UNIFIED_ATTN=1 — should use Triton decode (known working)."""
    print("\n" + "="*60)
    print("  Test: Triton decode (ATOM_USE_UNIFIED_ATTN=1) — control")
    print("="*60)
    
    env = os.environ.copy()
    env["ATOM_USE_UNIFIED_ATTN"] = "1"
    env["ATOM_LOADER_USE_THREADPOOL"] = "0"
    
    cmd = [
        sys.executable, "-m", "atom.examples.simple_inference",
        "--model", "Qwen/Qwen3-0.6B",
        "--tensor-parallel-size", "1",
        "--max-model-len", "256",
        "--max-tokens", "10",
        "--block-size", "64",
        "--enforce-eager",
        "--level", "0",
    ]
    
    print(f"  CMD: {' '.join(cmd)}")
    print(f"  ENV: ATOM_USE_UNIFIED_ATTN=1")
    print()
    
    start = time.time()
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, preexec_fn=os.setsid
    )
    
    try:
        stdout, _ = proc.communicate(timeout=300)
        elapsed = time.time() - start
        rc = proc.returncode
        
        if rc == 0:
            print(f"  ✅ PASS: EXIT=0, {elapsed:.1f}s (control)")
            return True
        else:
            print(f"  ❌ FAIL: EXIT={rc}")
            return False
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        print("  ❌ FAIL: Timeout (300s)")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", action="store_true", help="Also run control test")
    args = parser.parse_args()
    
    # First apply the Option 2 patch
    print("Applying Option 2 patch (force reshape_and_cache)...")
    rc = os.system(f"python /build/mi210-llm-stack/configs/patch_option2_reshape_and_cache.py")
    if rc != 0:
        print("Failed to apply patch!")
        sys.exit(1)
    print("Patch applied.\n")
    
    # Test pa_fwd_asm
    result_asm = test_no_env()
    
    # Optionally test control
    if args.control:
        result_control = test_with_unified()
    
    # Summary
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(f"  pa_fwd_asm decode: {'PASS' if result_asm else 'FAIL'}")
    if args.control:
        print(f"  Triton decode:     {'PASS' if result_control else 'FAIL'}")
    
    sys.exit(0 if result_asm else 1)
