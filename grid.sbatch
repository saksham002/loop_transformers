#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Measured MaxRSS is ~1.2G; 8G leaves ample headroom while keeping the billing
# weight (and so the scheduling priority) low enough to backfill readily.
#SBATCH --mem=8G
# 6h, not 24h: backfill can only slot a job into a gap >= its time limit, so an
# oversized request leaves jobs stuck behind a contended queue. Runs need ~2-3h.
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/home/saksham3/logs/loop/%x_%j.out
#SBATCH --error=/home/saksham3/logs/loop/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers

TASK=$1
SCALE=$2
ARM=$3
SCHED=$4     # "curr" or "fixed"
SEED=$5

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"

CKPT=/data/user_data/saksham3/checkpoints/loop/${TASK}_${ARM}_${SCHED}_s${SEED}

EXTRA=""
if [ "${SCHED}" = "curr" ]; then
    EXTRA="--curriculum"
fi

# Run python directly: srun inside sbatch hits a CPU-bind conflict on babel.
/data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
    --task "${TASK}" \
    --arm "${ARM}" \
    --scale "${SCALE}" \
    --n 32 \
    --steps 60000 \
    --seed "${SEED}" \
    --ckpt-dir "${CKPT}" \
    ${EXTRA}
