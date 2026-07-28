#!/bin/bash
# 2x2x3 grid: {bptt,stopgrad} x {fixed,curr} x {parity,s4,a5} x 4 seeds = 48 jobs.
# Optional first arg: comma-separated nodes to exclude (bad wandb connectivity).
set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers
mkdir -p /home/saksham3/logs/loop

EXCLUDE="${1:-}"
EXCL_ARG=""
if [ -n "${EXCLUDE}" ]; then
    EXCL_ARG="--exclude=${EXCLUDE}"
    echo "excluding nodes: ${EXCLUDE}"
fi

# Scale per task, matching the paper: s for the parity pilot, m for group tasks.
for cfg in "parity s" "s4 m" "a5 m"; do
    set -- $cfg
    TASK=$1
    SCALE=$2
    for ARM in bptt stopgrad; do
        for SCHED in fixed curr; do
            for SEED in 86 87 88 89; do
                NAME="g_${TASK}_${ARM}_${SCHED}_s${SEED}"
                sbatch ${EXCL_ARG} --job-name="${NAME}" grid.sbatch \
                    "${TASK}" "${SCALE}" "${ARM}" "${SCHED}" "${SEED}" >/dev/null
                # Stagger so simultaneous wandb.init calls don't collide.
                sleep 3
            done
        done
    done
done

squeue -u saksham3 -h -o "%j" | grep -c "^g_" | xargs echo "grid jobs in queue:"
