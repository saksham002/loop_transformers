#!/bin/bash
# Poll the 48-job grid. Emit a line per milestone (stdout -> Monitor events) and
# a bulleted Slack summary every MILESTONE completions. Exits when all are done.
# State lives in the repo, never /tmp (which is rotated and would break resumes).

STATE=/home/saksham3/logs/loop/watch_state.txt
MILESTONE=12
TOTAL=48

mkdir -p /home/saksham3/logs/loop
last_reported=$(cat "$STATE" 2>/dev/null || echo 0)

while true; do
    # Keep the NEWEST entry per job name: a resubmitted job has several sacct
    # rows, and taking the first would report a stale FAILED forever.
    stats=$(sacct -X --starttime=now-2days --format=JobName%32,State -n -P 2>/dev/null \
            | grep "^g_" | awk -F'|' '{last[$1]=$2} END {for (n in last) print n"|"last[n]}')

    done_n=$(echo "$stats" | grep -c "COMPLETED")
    run_n=$(echo "$stats"  | grep -cE "RUNNING")
    pend_n=$(echo "$stats" | grep -cE "PENDING|REQUEUED")
    fail_n=$(echo "$stats" | grep -cE "FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY")
    bl_n=$(wc -w < /home/saksham3/projects/AIRe/loop_transformers/blacklist.txt 2>/dev/null || echo 0)

    # Slack + event on each MILESTONE crossing.
    if [ "$done_n" -ge $((last_reported + MILESTONE)) ]; then
        last_reported=$(( (done_n / MILESTONE) * MILESTONE ))
        echo "$last_reported" > "$STATE"
        echo "MILESTONE completed=${done_n}/${TOTAL} running=${run_n} pending=${pend_n} failed=${fail_n}"
        python ~/utils/slack.py "loop_transformers grid progress
• completed: ${done_n}/${TOTAL}
• running: ${run_n}   pending: ${pend_n}
• failed (auto-relaunched): ${fail_n}
• blacklisted nodes: ${bl_n}" >/dev/null 2>&1
    fi

    if [ "$done_n" -ge "$TOTAL" ]; then
        echo "ALL_COMPLETE completed=${done_n}/${TOTAL}"
        python ~/utils/slack.py "loop_transformers grid FINISHED
• completed: ${done_n}/${TOTAL}
• blacklisted nodes: ${bl_n}
• next: collect_results.py regenerates experiments.md" >/dev/null 2>&1
        exit 0
    fi
    sleep 600
done
