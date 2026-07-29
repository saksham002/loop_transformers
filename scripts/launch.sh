#!/bin/bash
# Submit the grid: 28 runs.
#
#   Z60  (scale s): 2x2 ablation {bptt,stopgrad} x {fixed,curr}  + paper replication
#   S4   (scale m): paper replication
#   A5   (scale m): paper replication, with the horizon curriculum per the paper
# 4 seeds each (86-89).
set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers
mkdir -p /home/saksham3/logs/loop

BL=$(cat blacklist.txt 2>/dev/null | tr '\n' ',' | sed 's/,$//')
EXCL=""
[ -n "$BL" ] && EXCL="--exclude=${BL}" && echo "excluding: ${BL}"

submit() {  # task scale arm sched seed [horizon]
    # Scale-m runs are data-parallel over 2 GPUs: the global batch of 256 splits
    # 128/128, halving per-device memory from 21.4GB to 10.7GB. That is what
    # keeps BPTT at T=32 off the OOM cliff on the 40GB A100 nodes.
    # preempt has no L40S node with 2 free GPUs (31 alloc / 15 mix), so the
    # scale-m runs queue on general instead, where 19 L40S nodes have capacity.
    local GRES="--gres=gpu:1"
    local PART=""
    # general requires QOS=normal; preempt_qos is rejected outright there.
    [ "$2" = "m" ] && GRES="--gres=gpu:L40S:2" && PART="--partition=general --qos=normal"
    sbatch ${EXCL} ${GRES} ${PART} --job-name="p_$1_$3_$4_s$5" scripts/train.sbatch "$@" >/dev/null
    sleep 3
}

for SEED in 86 87 88 89; do
    # Z60 ablation: 2 arms x 2 schedules
    for ARM in bptt stopgrad; do
        for SCHED in fixed curr; do
            submit z60 s "${ARM}" "${SCHED}" "${SEED}"
        done
    done
    # Paper replication (BPTT, paper schedule). A5 additionally gets the
    # horizon curriculum, which the paper applies to chained A5/S5 only.
    submit z60 s bptt paper "${SEED}"
    submit s4  m bptt paper "${SEED}"
    submit a5  m bptt paper "${SEED}" --horizon
done

squeue -u saksham3 -h -o "%j" | grep -c "^p_" | xargs echo "grid2 jobs queued:"
