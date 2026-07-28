#!/bin/bash
#SBATCH --job-name=loop
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
ARM=$2
SCALE=$3

# JAX preallocates ~75% of the GPU by default, which makes peak-memory
# reporting meaningless; the runs are tiny so on-demand allocation is fine.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"

CKPT=/data/user_data/saksham3/checkpoints/loop/${TASK}_${ARM}_n32_${SCALE}_seed86

# Run python directly: srun inside the batch script requests a CPU binding that
# conflicts with the job's cgroup allocation ("Unable to satisfy cpu bind request").
/data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
    --task "${TASK}" \
    --arm "${ARM}" \
    --scale "${SCALE}" \
    --n 32 \
    --steps 40000 \
    --seed 86 \
    --ckpt-dir "${CKPT}"
