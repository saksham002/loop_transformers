#!/bin/bash
# stopgrad x currmax: our stage ladder (n = 1,2,4,8,16,32 x 10k) but the stage
# value is read as n_max and the length is sampled uniformly below it, as the
# paper's curriculum does. Isolates that one factor from `curr`.
# 3 tasks x 4 seeds = 12 runs, on preempt.
set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers
mkdir -p /home/saksham3/logs/loop

BL=$(cat blacklist.txt 2>/dev/null | tr '\n' ',' | sed 's/,$//')
EXCL=""
[ -n "$BL" ] && EXCL="--exclude=${BL}" && echo "excluding: ${BL}"

for cfg in "z60 s" "s4 m" "a5 m"; do
    set -- $cfg
    TASK=$1
    SCALE=$2
    for SEED in 86 87 88 89; do
        # stopgrad peaks under 0.6GB even at scale m, so one GPU is ample here.
        sbatch ${EXCL} --gres=gpu:1 \
            --job-name="p_${TASK}_stopgrad_currmax_s${SEED}" \
            scripts/train.sbatch "${TASK}" "${SCALE}" stopgrad currmax "${SEED}" >/dev/null
        sleep 3
    done
done

squeue -u saksham3 -h -o "%j" | grep -c "currmax" | xargs echo "currmax jobs queued:"
