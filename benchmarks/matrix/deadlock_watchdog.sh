#!/usr/bin/env bash
# Detects stalled bench work and says so loudly.
#
# Four deadlocks in one session, all from scripts waiting on `pgrep -f`
# patterns that matched themselves or their own parent shell. wait_for_bench.sh
# removes that specific class, but the general failure -- a script alive,
# nothing running, no progress -- can happen for other reasons: a container that
# died without the harness noticing, a fetch that stalled, a wait condition that
# is simply never satisfiable.
#
# So this checks the OBSERVABLE symptom rather than any particular cause:
#
#   a round script is alive
#   AND no bench-* container is running
#   AND no tuner or fetch is running
#   AND the newest result file has not changed
#
# ...for longer than STALL_MIN. That combination means work is queued behind
# something that will never finish.
#
# It only reports by default. --reap also kills the oldest stalled waiter, on
# the theory that a self-deadlocked script is better dead than blocking the
# queue -- but that is opt-in, because killing the wrong thing mid-arm costs
# hours.
#
#   ./deadlock_watchdog.sh              # one check, report only
#   ./deadlock_watchdog.sh --reap       # one check, kill stalled waiters
#   ./deadlock_watchdog.sh --daemon     # check every 10 min, report only
set -uo pipefail
BASE=/mnt/llm-storage/bench-matrix
STALL_MIN="${STALL_MIN:-30}"
STAMP="$BASE/.watchdog_stamp"
LOG="$BASE/logs/watchdog.log"
REAP=0; DAEMON=0
for a in "$@"; do
    [ "$a" = "--reap" ] && REAP=1
    [ "$a" = "--daemon" ] && DAEMON=1
done
mkdir -p "$BASE/logs"

check_once() {
    local now scripts containers busy newest last_newest age_min
    now=$(date -u +%s)

    # Round scripts alive, by argv -- safe here because we only COUNT them and
    # never wait on the result, so a self-match cannot deadlock anything.
    scripts=$(ps -eo pid,cmd | grep -E "bash \./bin/round[0-9]" | grep -v grep || true)
    [ -z "$scripts" ] && { echo "$(date -u +%T) idle: no round scripts running" >> "$LOG"; return 0; }

    containers=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c '^bench-' || true)
    busy=$(ps -eo cmd | grep -cE "[m]oe-tune|[f]etch_model.py|[b]enchmark_moe.py" || true)
    newest=$(ls -t "$BASE"/results/*.json 2>/dev/null | head -1)
    newest=$(stat -c %Y "$newest" 2>/dev/null || echo 0)

    if [ "$containers" -gt 0 ] || [ "$busy" -gt 0 ]; then
        echo "$now $newest" > "$STAMP"
        echo "$(date -u +%T) ok: containers=$containers busy=$busy" >> "$LOG"
        return 0
    fi

    # Nothing running. Has anything changed since we last looked?
    if [ -f "$STAMP" ]; then
        read -r last_ts last_newest < "$STAMP"
        if [ "$newest" != "$last_newest" ]; then
            echo "$now $newest" > "$STAMP"; return 0
        fi
        age_min=$(( (now - last_ts) / 60 ))
        if [ "$age_min" -ge "$STALL_MIN" ]; then
            {
                echo "=============================================="
                echo "$(date -u +%FT%TZ) STALL: ${age_min}m with scripts alive,"
                echo "  no bench container, no tuner/fetch, no new results."
                echo "$scripts"
                echo "=============================================="
            } | tee -a "$LOG"
            if [ "$REAP" = "1" ]; then
                local pid
                pid=$(echo "$scripts" | awk '{print $1}' | head -1)
                echo "$(date -u +%T) REAPING stalled pid $pid" | tee -a "$LOG"
                kill "$pid" 2>/dev/null || true
                echo "$now $newest" > "$STAMP"
            fi
            return 1
        fi
    else
        echo "$now $newest" > "$STAMP"
    fi
    return 0
}

if [ "$DAEMON" = "1" ]; then
    while true; do check_once; sleep 600; done
else
    check_once && echo "no stall detected" || echo "STALL DETECTED -- see $LOG"
fi
