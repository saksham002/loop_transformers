#!/bin/bash
#SBATCH --job-name=loopcur
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/saksham3/logs/loop/%x_%j.out
#SBATCH --error=/home/saksham3/logs/loop/%x_%j.out
#SBATCH --open-mode=append

set -euo pipefail

cd /home/saksham3/projects/AIRe/loop_transformers

TASK=$1
SCALE=$2

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"

# Distinct checkpoint dir: must not resume the completed fixed-n stopgrad runs.
CKPT=/data/user_data/saksham3/checkpoints/loop/${TASK}_stopgrad_curriculum_${SCALE}_seed86

/data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
    --task "${TASK}" \
    --arm stopgrad \
    --scale "${SCALE}" \
    --n 32 \
    --curriculum \
    --seed 86 \
    --run-tag curriculum \
    --ckpt-dir "${CKPT}"
