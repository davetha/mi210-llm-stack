# Change 04 — Session Persistence (KV Save/Restore + TTL)

Eliminate the TTL-eviction cold-start pain: every 30 min idle = full re-prefill.

## Problem

mimo's TTL was 1800s (30 min). After 30 min idle, llama-swap evicts the model.
The next request triggers a full model reload + full prompt prefill (~500s for
an 18K-token system prompt). The in-session KV cache works (2nd request with
same prefix is 8× faster), but it's lost on eviction.

## What was changed (3 files)

### 1. `launch-mimo.sh` — wrapper rewrite → [`configs/launch-mimo.sh`](../configs/launch-mimo.sh)

- **Comma-separated `-ot`** — forward-compat fix for llama.cpp PR #26049
  (repeated `-ot` flags now silently drop all but the last on master). The old
  space-separated form would break.
- **`--slot-save-path`** added — enables the `/slots/0?action=save|restore` API.
- **Background container + auto-restore loop** — waits for server health check,
  then POSTs `/slots/0?action=restore` to load a warm session file if one exists.
- Non-`exec` pattern: backgrounds `docker run`, does the restore, then `wait`s
  on the PID.

### 2. `warm-mimo-session.sh` — new file → [`configs/warm-mimo-session.sh`](../configs/warm-mimo-session.sh)

One-time warmup script:
1. Sends a representative system prompt to mimo.
2. Saves the KV session via `/slots/0?action=save`.

Run **once** after deploying the wrapper. Future mimo restarts auto-restore this
session, so only *new* tokens need prefilling.

```bash
sh /mnt/llm-storage/warm-mimo-session.sh 5803
```

### 3. `llama-swap-config.yaml` — TTL change → [`configs/llama-swap-config.yaml`](../configs/llama-swap-config.yaml)

```yaml
mimo:
  ttl: 86400   # was 1800 (30 min → 24 h)
```

Keeps mimo resident much longer, so the in-session KV cache persists across idle
periods.

## What works

| Feature | Status |
|---------|--------|
| Save/restore API | ✅ works (10 MB file, ~10ms save, ~5ms restore, `n_restored` correct) |
| In-session KV cache | ✅ 2nd request with same prefix = **8× faster** (0.6s vs 4.9s for 106 tokens) |
| Auto-restore on startup | ✅ fires correctly after health check |

## What does NOT work (this llama.cpp build, commit 67b9b0e)

**Restored KV does NOT enable prefill skip.** Despite `sim_best=1.000` and
`f_keep=0.922` (perfect LCP match), all 106 tokens are still reprocessed (4.4s).
The restore loads KV data into the slot but the prefill path doesn't treat it as
valid cached state. `--cache-reuse 64` (KV shifting) does not change this.

This is a limitation of the restore implementation in this version, not a
configuration issue.

## Practical impact

The TTL raise (1800→86400) is the effective fix: it keeps mimo resident for 24h,
so the in-session KV cache persists across normal idle periods. The first request
after a *true* restart (manual or crash) still pays the full prefill cost, but
day-to-day operation no longer cycles through cold starts every 30 min.

The session save/restore infrastructure is in place for when llama.cpp upstream
fixes the prefill-skip-on-restore path.
