#!/usr/bin/env python3
"""
Comprehensive KV x FlashAttention x Layer-Split benchmark harness.
Runs INSIDE the tqbench container against llama-server on localhost:PORT.
Controls server lifecycle per test, measures prefill + decode + correctness.
"""
import subprocess, time, json, urllib.request, urllib.error, os, signal, sys, re

BIN     = "/build/src/build/bin/llama-server"
MODEL   = "/models/dsv2lite-q8_0.gguf"
GPU     = "1"          # GPU1 has ~20.5GB free; GPU0 is full
PORT    = 8098
CTX     = 4096
LOGDIR  = "/build/bench/logs"
RESULTS = "/build/bench/results.jsonl"

os.makedirs(LOGDIR, exist_ok=True)

# Deterministic ~2000+ token prompt (technical passage repeated).
# Server reports actual prompt_n in timings; we just need it long & non-cached.
_PARA = (
    "The memory hierarchy of a modern accelerator determines the achievable "
    "roofline for transformer inference. On CDNA2 devices such as the MI210, "
    "the absence of xGMI means that tensor-parallel workloads must traverse "
    "the PCIe fabric for every all-reduce, which caps the effective bandwidth "
    "at roughly thirty-two gigabytes per second per direction. When the KV "
    "cache is quantized below sixteen bits, the attention kernel must perform "
    "on-the-fly dequantization inside the register file, trading arithmetic "
    "intensity for a smaller memory footprint. FlashAttention fuses the "
    "softmax and matmul stages so that the HBM traffic for a causal mask is "
    "reduced from quadratic to linear in the sequence length, which is why "
    "long-context prefill benefits disproportionately. Multi-head latent "
    "attention further compresses the KV projection into a low-rank buffer, "
    "so the prefill cost is dominated by the query-key dot products rather "
    "than the projection matmuls. "
)
PROMPT_2000 = (_PARA * 11)  # ~2065 tokens (187 tok/para measured)

def log(msg):
    print(f"[harness] {msg}", flush=True)

def http(path, data=None, timeout=180):
    url = f"http://127.0.0.1:{PORT}{path}"
    if data is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def server_cmd(ngl, ctk, ctv, fa, ctkcpu, ctvcpu):
    cmd = (f"{BIN} -m {MODEL} -ngl {ngl} -c {CTX} -np 1 -fa {fa} "
           f"--host 127.0.0.1 --port {PORT} --no-webui --temp 0.0")
    if ctk:  cmd += f" -ctk {ctk}"
    if ctv:  cmd += f" -ctv {ctv}"
    if ctkcpu: cmd += f" -ctk-cpu {ctkcpu}"
    if ctvcpu: cmd += f" -ctv-cpu {ctvcpu}"
    return cmd

def kill_server():
    subprocess.run("pkill -9 -f llama-server", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def start_server(tid, ngl, ctk, ctv, fa, ctkcpu, ctvcpu):
    kill_server()
    logf = os.path.join(LOGDIR, f"test_{tid}.log")
    cmd = server_cmd(ngl, ctk, ctv, fa, ctkcpu, ctvcpu)
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = GPU
    with open(logf, "w") as lf:
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=lf,
                             stderr=subprocess.STDOUT, start_new_session=True)
    ready = False
    err = ""
    for _ in range(90):                       # up to 90s for cold first load
        time.sleep(1)
        try:
            with open(logf) as lf:
                txt = lf.read()
        except Exception:
            txt = ""
        if "model loaded" in txt:
            ready = True
            break
        if "error loading" in txt or "exiting due to" in txt or "out of memory" in txt:
            err = "LOAD_FAIL: " + " ".join(
                l for l in txt.splitlines()
                if any(k in l.lower() for k in ("error", "fail", "oom", "out of memory", "abort"))
            )[:400]
            break
        # also accept health endpoint
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            ready = True
            break
        except Exception:
            pass
    if ready:
        time.sleep(2)   # let the HTTP thread settle
    return p, logf, ready, err

def measure(prefill=True):
    if prefill:
        data = {"prompt": PROMPT_2000, "n_predict": 8,
                "temperature": 0.0, "stream": False}
    else:
        data = {"prompt": "Tell me a long story about a brave knight who "
                          "ventured into the mountains. ",
                "n_predict": 200, "temperature": 0.0, "stream": False}
    resp = http("/completion", data, timeout=240)
    t = resp.get("timings", {}) or {}
    return resp, t

def correctness():
    data = {"prompt": "Question: What is 2+2?\nAnswer with only the number.\nAnswer:",
            "n_predict": 8, "temperature": 0.0, "stream": False}
    try:
        resp = http("/completion", data, timeout=30)
        out = (resp.get("content") or "").strip()
        return out, ("4" in out[:30])
    except Exception as e:
        return f"<err {e}>", False

def grep_perf(logf, prefill_n=None, decode_n=None):
    """Pull the prompt-eval + eval (decode) log lines matching measured token counts."""
    pe = ev = ""
    try:
        with open(logf) as f:
            txt = f.read()
        pel = re.findall(r"prompt eval time =.*?tokens per second\)", txt)
        evl = [l for l in re.findall(r"eval time =.*?tokens per second\)", txt)
               if "prompt eval" not in l]
        # match by token count when known
        def pick(lines, n, key):
            if n is not None:
                for l in lines:
                    if f"/  {n} {key}" in l or f"/ {n} {key}" in l:
                        return l.strip()
            return lines[-1].strip() if lines else ""
        pe = pick(pel, prefill_n, "tokens")
        ev = pick(evl, decode_n, "runs")
    except Exception:
        pass
    return pe, ev

# ---- test matrix ----------------------------------------------------------
TESTS = [
    # id, group, name, ngl, ctk, ctv, fa, ctkcpu, ctvcpu
    (1,  "allgpu",   "f16/f16 FA-off",            99, "f16",   "f16",   "off", None,    None),
    (2,  "allgpu",   "f16/f16 FA-on",             99, "f16",   "f16",   "on",  None,    None),
    (3,  "allgpu",   "q8_0/q8_0 FA-off",          99, "q8_0",  "q8_0",  "off", None,    None),
    (4,  "allgpu",   "q8_0/q8_0 FA-on",           99, "q8_0",  "q8_0",  "on",  None,    None),
    (5,  "allgpu",   "q4_0/q4_0 FA-off",          99, "q4_0",  "q4_0",  "off", None,    None),
    (6,  "allgpu",   "q4_0/q4_0 FA-on",           99, "q4_0",  "q4_0",  "on",  None,    None),
    (7,  "allgpu",   "q8_0/q4_1 FA-off (prod)",   99, "q8_0",  "q4_1",  "off", None,    None),
    (8,  "allgpu",   "q8_0/q4_1 FA-on",           99, "q8_0",  "q4_1",  "on",  None,    None),
    (9,  "allgpu",   "turbo3/turbo3 FA-off",      99, "turbo3","turbo3","off", None,    None),
    (10, "allgpu",   "turbo3/turbo3 FA-on",       99, "turbo3","turbo3","on",  None,    None),
    (11, "split",    "f16/f16 FA-off",            23, "f16",   "f16",   "off", None,    None),
    (12, "split",    "f16/f16 FA-on",             23, "f16",   "f16",   "on",  None,    None),
    (13, "split",    "q8_0/q4_1 FA-off (prod)",   23, "q8_0",  "q4_1",  "off", None,    None),
    (14, "split",    "q8_0/q4_1 FA-on",           23, "q8_0",  "q4_1",  "on",  None,    None),
    (15, "split",    "q4_0/q4_0 FA-off",          23, "q4_0",  "q4_0",  "off", None,    None),
    (16, "split",    "q4_0/q4_0 FA-on",           23, "q4_0",  "q4_0",  "on",  None,    None),
    (17, "perlayer", "f16-GPU + turbo3-CPU FA-off", 23, "f16", "f16",   "off", "turbo3","turbo3"),
    (18, "perlayer", "f16-GPU + turbo3-CPU FA-on",  23, "f16", "f16",   "on",  "turbo3","turbo3"),
    (19, "perlayer", "q4_0-GPU + turbo3-CPU FA-off",23, "q4_0","q4_0",  "off", "turbo3","turbo3"),
    (20, "perlayer", "q4_0-GPU + turbo3-CPU FA-on", 23, "q4_0","q4_0",  "on",  "turbo3","turbo3"),
]

def run_one(t):
    tid, grp, name, ngl, ctk, ctv, fa, ctkcpu, ctvcpu = t
    log(f"=== TEST {tid} [{grp}] {name} (ngl={ngl} ctk={ctk} ctv={ctv} fa={fa} "
        f"ctkcpu={ctkcpu}) ===")
    rec = {"id": tid, "group": grp, "name": name, "ngl": ngl,
           "ctk": ctk, "ctv": ctv, "fa": fa,
           "ctk_cpu": ctkcpu or "", "ctv_cpu": ctvcpu or "",
           "status": "OK"}
    p, logf, ready, err = start_server(tid, ngl, ctk, ctv, fa, ctkcpu, ctvcpu)
    if not ready:
        rec["status"] = err or "LOAD_FAIL"
        log(f"  LOAD FAILED: {rec['status']}")
        kill_server()
        return rec
    # device/free memory from log
    try:
        with open(logf) as f:
            devtxt = f.read()
        m = re.search(r"ROCm0\s*:\s*(.*?)\(([0-9]+) MiB,\s*([0-9]+) MiB free\)", devtxt)
        if m:
            rec["gpu_name"] = m.group(1).strip()
            rec["gpu_total_mib"] = int(m.group(2))
            rec["gpu_free_mib"] = int(m.group(3))
    except Exception:
        pass
    try:
        # warmup (small, distinct prompt) so kernels are JIT'd
        http("/completion", {"prompt": "hello world", "n_predict": 1,
                             "temperature": 0.0, "stream": False}, timeout=60)
        # ---- prefill measurement (cold, ~2000 tok) ----
        _, tp = measure(prefill=True)
        rec["prefill_tokens"] = tp.get("prompt_n")
        rec["prefill_ms"]     = round(tp.get("prompt_ms", 0), 2)
        rec["prefill_tps"]    = round(tp.get("prompt_per_second", 0), 1)
        rec["prefill_log"]    = ""   # filled below
        # ---- decode measurement (short prompt, 200 tok) ----
        _, td = measure(prefill=False)
        rec["decode_tokens"]  = td.get("predicted_n")
        rec["decode_tps"]     = round(td.get("predicted_per_second", 0), 1)
        # ---- correctness ----
        out, ok = correctness()
        rec["correct"] = bool(ok)
        rec["output"]  = out[:80]
        pe, ev = grep_perf(logf, rec.get("prefill_tokens"), rec.get("decode_tokens"))
        rec["prefill_log"] = pe
        rec["decode_log"]  = ev
        log(f"  prefill={rec['prefill_tps']} tok/s (n={rec['prefill_tokens']}) "
            f"decode={rec['decode_tps']} tok/s correct={rec['correct']} "
            f"out={rec['output']!r}")
    except Exception as e:
        rec["status"] = f"RUN_ERR: {type(e).__name__}: {str(e)[:200]}"
        log(f"  RUN ERROR: {rec['status']}")
    finally:
        kill_server()
    return rec

def main():
    only = sys.argv[1:]            # optional list of test ids to run
    with open(RESULTS, "a") as rf:
        rf.write(f"# bench run started {time.strftime('%Y-%m-%dT%H:%M:%S')} GPU={GPU}\n")
    log(f"running {len(only) or len(TESTS)} tests; GPU={GPU} PORT={PORT} CTX={CTX}")
    for t in TESTS:
        if only and str(t[0]) not in only:
            continue
        rec = run_one(t)
        with open(RESULTS, "a") as rf:
            rf.write(json.dumps(rec) + "\n")
        kill_server()
    log("ALL DONE")

if __name__ == "__main__":
    main()
