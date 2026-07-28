#!/bin/bash
# Launch the 3 tasks x 2 arms grid. Scale s for parity, m for the group tasks,
# matching the paper's per-task choices.
set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers
mkdir -p /home/saksham3/logs/loop

for cfg in "parity s" "s4 m" "a5 m"; do
    set -- $cfg
    TASK=$1
    SCALE=$2
    for ARM in bptt stopgrad; do
        sbatch --job-name="loop_${TASK}_${ARM}" run.sbatch "${TASK}" "${ARM}" "${SCALE}"
    done
done
