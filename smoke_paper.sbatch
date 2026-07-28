#!/bin/bash
#SBATCH --job-name=smoke_paper
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclude=babel-x5-24,babel-w9-16,babel-w9-28,babel-z5-28
#SBATCH --output=/home/saksham3/logs/loop/smoke_paper_%j.out
#SBATCH --error=/home/saksham3/logs/loop/smoke_paper_%j.out

set -euo pipefail
cd /home/saksham3/projects/AIRe/loop_transformers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"

# A5 exercises the horizon curriculum; z60 exercises the new task.
for T in a5:m:--horizon z60:s:; do
    TASK=${T%%:*}; REST=${T#*:}; SCALE=${REST%%:*}; EXTRA=${REST#*:}
    echo "===== ${TASK} scale=${SCALE} ${EXTRA} ====="
    /data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
        --task "${TASK}" --arm bptt --scale "${SCALE}" \
        --schedule paper ${EXTRA} \
        --steps 1200 --eval-every 1200 --ckpt-every 100000 --seed 0 \
        --run-tag smoke \
        --ckpt-dir "/data/user_data/saksham3/checkpoints/loop/_smokepaper_${TASK}"
done
echo "SMOKE_PAPER_OK"
