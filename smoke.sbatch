#!/bin/bash
#SBATCH --job-name=loop_smoke
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/saksham3/logs/loop/smoke_%j.out
#SBATCH --error=/home/saksham3/logs/loop/smoke_%j.out

set -euo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb
mkdir -p "${WANDB_DIR}"
cd /home/saksham3/projects/AIRe/loop_transformers

nvidia-smi --query-gpu=name,memory.total --format=csv

# Confirm the compute node can reach wandb before trusting online mode.
/data/user_data/saksham3/uv/loop/bin/python -c "
import wandb, os
r = wandb.init(project='loop_transformers', group='_connectivity', name='_smoke_probe')
wandb.log({'probe': 1})
print('WANDB_MODE_OK', r.settings.mode, r.url, flush=True)
wandb.finish()
"

# Both arms, few steps, to compare peak memory at T=32.
for ARM in stopgrad bptt; do
    echo "===== ARM=${ARM} ====="
    /data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
        --task a5 --arm "${ARM}" --scale m --n 32 \
        --steps 200 --eval-every 200 --ckpt-every 100000 \
        --ckpt-dir "/data/user_data/saksham3/checkpoints/loop/_smoke_${ARM}"
done
echo "SMOKE_OK"
