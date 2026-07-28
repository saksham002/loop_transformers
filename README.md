# Looped transformers: stopgrad-truncated training

Does a weight-tied looped transformer still learn if the loss is applied with a
`stop_gradient` at every forward call except the final one — supervising only the
terminal output — so that the loop count `T` can scale without the O(T) memory of
backpropagation through time?

```python
h = h0
for t in range(T - 1):
    h = F(h, e)                 # no gradient path
h = F(stop_grad(h), e)          # only this application is differentiated
loss = CE(head(LN(h)), y)       # terminal only, as in the paper
```

Built against *"When Does Recurrence Become an Algorithm? Convergence Selection
in Weight-Tied Looped Transformers"* ([arXiv:2607.20594](https://arxiv.org/abs/2607.20594)).
The paper always backpropagates through all `T` applications; the `stopgrad` arm
here is not something it tests.

## Tasks

Per-position prefix products: given `g_1 ... g_n`, predict `p_i = g_1 o ... o g_i`
at every position `i`. Implemented for Z2 (parity), Z60, S4 and A5.

## Design

| Factor | Levels |
|---|---|
| Gradient | `bptt` (through all T) vs `stopgrad` (detach at `h^{T-1}`) |
| Schedule | `fixed` (n = T = 32), `curr` (n = 1,2,4,8,16,32), `paper` (Appendix D) |

Z60 carries the 2x2 ablation plus the paper replication; S4 and A5 carry the
replication only. Four seeds per cell, 60k steps each. See
[`project_status/experiments.md`](project_status/experiments.md) for the design
and results, and [`project_status/ideas.md`](project_status/ideas.md) for the
mechanistic argument.

## Layout

| Path | Purpose |
|---|---|
| `loop_transformers/tasks.py` | Cayley tables and prefix-product sampling |
| `loop_transformers/model.py` | Weight-tied looped transformer, both gradient arms |
| `loop_transformers/train.py` | Training loop, curricula, eval, checkpoints, wandb |
| `grid2.sbatch`, `launch_grid2.sh` | Current grid: one cell per job |
| `relaunch_failed.py` | Resubmits failures, blacklists bad nodes |
| `collect_results.py` | Regenerates the results table in `experiments.md` |
| `plot_results.py` | Summary figures from wandb |
| `tests/` | Group axioms, prefix products, gradient-equivalence proof |

## Running

```bash
uv venv /path/to/env && uv pip install -e ".[dev]"
python -m pytest tests/ -q
./launch_grid2.sh                 # submit the grid (SLURM)
python collect_results.py         # refresh the results table
python plot_results.py            # write figures
```

Single-run example:

```bash
python -m loop_transformers.train \
    --task a5 --arm stopgrad --scale m --schedule paper --horizon \
    --steps 60000 --seed 86 --ckpt-dir /path/to/ckpt
```

Multi-GPU is automatic: the global batch is split across every visible device
(mesh of size 1 on a single GPU), so the batch size stays as the paper specifies.

## Notes

Implementation and infrastructure gotchas are recorded in
[`project_status/codebase_notes.md`](project_status/codebase_notes.md).
