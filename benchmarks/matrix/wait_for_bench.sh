#!/usr/bin/env bash
# Shared "wait until no other bench work is running" helper.
#
# WHY THIS EXISTS. Four separate deadlocks in one session, all the same bug:
# a script waiting on `pgrep -f <pattern>` where something OTHER than the work
# matched the pattern.
#
#   1. round2_chain waited on "fetch_round2.sh" and matched the bash -c wrapper
#      whose heredoc CONTAINED the script text. Hung 8 hours.
#   2. `pkill -f "round2_followup"` killed its own ssh shell.
#   3. A diagnostic `pgrep -f fetch_round2` matched itself.
#   4. round6 waited on "[b]in/round[0-9]_" -- which matches
#      bin/round6_spec_dense.sh, i.e. itself. Hung until noticed.
#
# The lesson is not "write better regexes". It is that `pgrep -f` matches
# COMMAND LINES, and your own command line routinely contains the pattern you
# are searching for -- as does any shell that spawned you, any editor with the
# file open, and any `grep` you run to debug it.
#
# So: no pattern matching. Track work by PID file, and check liveness with
# kill -0, skipping our own PID and our parent's.
#
#   . "$BIN/wait_for_bench.sh"
#   bench_claim              # register this script as running work
#   bench_wait_for_others    # block until every OTHER claim is gone
#
# Claims are removed on exit via trap, and stale claims (dead PIDs) are
# reaped by the waiter, so a killed script cannot deadlock the next one.
BENCH_RUN_DIR="${BENCH_RUN_DIR:-/mnt/llm-storage/bench-matrix/.running}"

bench_claim() {
    mkdir -p "$BENCH_RUN_DIR"
    echo $$ > "$BENCH_RUN_DIR/$$.pid"
    trap 'rm -f "$BENCH_RUN_DIR/$$.pid"' EXIT INT TERM
}

bench_wait_for_others() {
    local me=$$ parent=$PPID
    mkdir -p "$BENCH_RUN_DIR"
    while true; do
        local blocking=0 f pid
        for f in "$BENCH_RUN_DIR"/*.pid; do
            [ -e "$f" ] || continue
            pid=$(cat "$f" 2>/dev/null)
            [ -n "$pid" ] || { rm -f "$f"; continue; }
            # Reap claims whose process is gone -- a killed script must not
            # deadlock everything queued behind it.
            if ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; continue; fi
            [ "$pid" = "$me" ] && continue
            [ "$pid" = "$parent" ] && continue
            blocking=1
        done
        # Containers are unambiguous -- a name either starts with bench- or it
        # does not, and no shell command line can impersonate one.
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^bench-'; then
            blocking=1
        fi
        [ "$blocking" = "0" ] && return 0
        sleep 60
    done
}
