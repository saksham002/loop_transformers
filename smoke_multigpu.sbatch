#!/bin/bash
#SBATCH --job-name=smoke_mgpu
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=babel-x5-24,babel-w9-16,babel-w9-28,babel-z5-28
#SBATCH --output=/home/saksham3/logs/loop/smoke_mgpu_%j.out
#SBATCH --error=/home/saksham3/logs/loop/smoke_mgpu_%j.out

set -euo pipefail
cd /home/saksham3/projects/AIRe/loop_transformers
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR=/data/user_data/saksham3/wandb

# Equivalence at a batch that fits comfortably on ONE device, so the control
# cannot OOM. Sharding correctness does not depend on batch size; the previous
# attempt compared at batch 256 and the single-device control ran out of memory
# on top of the multi-device buffers already held in the same process.
echo "===== sharding correctness (batch 64) ====="
/data/user_data/saksham3/uv/loop/bin/python - <<'PY'
import jax, numpy as np, optax
import loop_transformers.train as T
from loop_transformers.model import LoopedTransformer
from loop_transformers.tasks import build_task

print("devices:", jax.devices())
mesh = T.make_mesh()
print("mesh:", dict(mesh.shape))

tab, ident, order = build_task("a5")
m = LoopedTransformer(vocab=order, n_classes=order, scale="m")
p0 = m.init(jax.random.PRNGKey(86), 32)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=0.01))

def step(msh, arm, batch):
    p = T.replicate(msh, p0); st = T.replicate(msh, tx.init(p0))
    _, _, l = T.train_step(p, st, jax.random.PRNGKey(0), m, tx, tab, ident,
                           32, arm, batch, 32, 32, msh)
    return float(l)

single = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
for arm in ("bptt", "stopgrad"):
    lm, ls = step(mesh, arm, 64), step(single, arm, 64)
    d = abs(lm - ls)
    print(f"  {arm:9s} multi={lm:.6f} single={ls:.6f} |diff|={d:.3e}")
    assert d < 1e-4, f"data parallelism changed the {arm} loss"
print("SHARDING_MATCHES_SINGLE_DEVICE")
PY

echo "===== per-device memory at the real batch (256) ====="
/data/user_data/saksham3/uv/loop/bin/python - <<'PY'
import jax, optax
import loop_transformers.train as T
from loop_transformers.model import LoopedTransformer
from loop_transformers.tasks import build_task
mesh = T.make_mesh()
tab, ident, order = build_task("a5")
m = LoopedTransformer(vocab=order, n_classes=order, scale="m")
p0 = m.init(jax.random.PRNGKey(86), 32)
tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, weight_decay=0.01))
p = T.replicate(mesh, p0); st = T.replicate(mesh, tx.init(p0))
for _ in range(3):
    p, st, l = T.train_step(p, st, jax.random.PRNGKey(0), m, tx, tab, ident,
                            32, "bptt", 256, 32, 32, mesh)
s = jax.local_devices()[0].memory_stats()
print(f"  batch 256 on {jax.device_count()} devices: peak/device={s['peak_bytes_in_use']/1e9:.2f} GB")
print("  (single-device reference measured earlier: 21.41 GB)")
PY

echo "===== short real run on 2 GPUs (a5 paper arm, horizon) ====="
/data/user_data/saksham3/uv/loop/bin/python -m loop_transformers.train \
    --task a5 --arm bptt --scale m --schedule paper --horizon \
    --steps 600 --eval-every 600 --ckpt-every 100000 --grid-every 600 --seed 0 \
    --run-tag mgpu \
    --ckpt-dir /data/user_data/saksham3/checkpoints/loop/_smoke_mgpu
echo "SMOKE_MGPU_OK"
