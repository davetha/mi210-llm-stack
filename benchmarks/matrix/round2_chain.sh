#!/usr/bin/env bash
# Waits for work already in flight, then runs round 2.
#
# DEADLOCK POSTMORTEM -- the first version of this script hung for 8 hours.
# It waited with:
#
#     while pgrep -f "fetch_round2.sh" >/dev/null 2>&1; do sleep 120; done
#
# The fetch had long since finished. What still matched was the leftover
# `bash -c` wrapper that CREATED this script via a heredoc -- its command line
# contained the whole script text, including the literal string
# "fetch_round2.sh". So the chain waited on its own parent, forever.
#
# The comment in that version claimed "waiting on a process you did not start,
# and that has no opinion about you, has no cycle to form." That was right about
# processes and wrong about *patterns*: `pgrep -f` matches command lines, and a
# command line can contain the pattern for reasons that have nothing to do with
# the process doing the work.
#
# Fixed by waiting on a PID recorded by the writer, not on a text match.
set -uo pipefail
cd /mnt/llm-storage/bench-matrix

wait_for_pidfile() {  # wait_for_pidfile <file> <label>
    local f="$1" label="$2" pid
    [ -f "$f" ] || { echo "no $f -- assuming $label already done"; return 0; }
    pid=$(cat "$f" 2>/dev/null) || return 0
    [ -n "$pid" ] || return 0
    echo "=== $(date -u +%T) waiting for $label (pid $pid) ==="
    while kill -0 "$pid" 2>/dev/null; do sleep 120; done
    echo "=== $(date -u +%T) $label done ==="
}

echo "=== $(date -u +%T) waiting for any bench container to exit ==="
while docker ps --format '{{.Names}}' | grep -q '^bench-'; do sleep 120; done
echo "=== $(date -u +%T) no bench containers running ==="

wait_for_pidfile fetch_round2.pid "round-2 fetches"

du -sh t80-w8a8 t70-w8a8 probe-w4a8 2>/dev/null

# A failed fetch is a reason to skip that arm, not to abort: E0, E5, E7, E10 and
# E11 need no new weights and are worth having on their own.
echo "=== $(date -u +%T) starting round 2 ==="
exec ./bin/round2.sh
