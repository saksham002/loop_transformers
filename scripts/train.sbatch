#!/bin/bash
#SBATCH --partition=preempt
# Default for the scale-s Z60 cells. The scale-m paper runs (S4/A5) are
# submitted with --gres=gpu:L40S:2 on the sbatch command line, which overrides
# this: BPTT at T=32, batch 256, scale m OOMs on the 40GB A100 nodes.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
SCHED=$4     # fixed | curr | paper
SEED=$5
HORIZON=${6:-}   # "--horizon" for A5 replication, per the paper

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"

CKPT=/data/user_data/saksham3/checkpoints/loop/${TASK}_${ARM}_${SCHED}_s${SEED}

# Run python directly: srun inside sbatch hits a CPU-bind conflict on babel.
/data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
    --task "${TASK}" \
    --arm "${ARM}" \
    --scale "${SCALE}" \
    --schedule "${SCHED}" \
    --n 32 \
    --steps 60000 \
    --seed "${SEED}" \
    --ckpt-dir "${CKPT}" \
    ${HORIZON}
