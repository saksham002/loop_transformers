# Codebase notes

## Layout

| Path | Purpose |
|---|---|
| `loop_transformers/tasks.py` | Cayley tables for Z2/S4/A5; prefix products via `lax.scan` |
| `loop_transformers/model.py` | Weight-tied looped transformer; both gradient arms |
| `loop_transformers/train.py` | Training loop, curriculum, eval, orbax checkpoints, wandb |
| `scripts/train.sbatch` | One grid cell: `TASK SCALE ARM SCHED SEED [--horizon]` |
| `scripts/launch.sh` | Submits the grid; scale-m cells go to `general` with 2 GPUs |
| `scripts/watch.sh` | Milestone progress + Slack summaries |
| `scripts/relaunch.py` | Polls sacct, resubmits failures, blacklists bad nodes |
| `analysis/collect_results.py` | Pulls wandb summaries, writes the table into `experiments.md` |
| `analysis/plot_results.py` | Summary figures from wandb |
| `scratch/` | Untracked: superseded launchers, smoke tests, one-off utilities |
| `tests/test_setup.py` | 9 tests incl. the gradient-equivalence proof for stopgrad |

Env: `/data/user_data/saksham3/uv/loop/` (jax 0.10.2 + CUDA12, flax 0.12.8,
optax, orbax, wandb). Checkpoints: `/data/user_data/saksham3/checkpoints/loop/`.
Logs: `~/logs/loop/`. wandb project `loop_transformers`, grouped by task.

## How the two arms differ

Only in gradient flow; the forward computation is identical, which
`test_arms_agree_in_forward` asserts.

`bptt` uses `lax.scan` over T applications — reverse-differentiable, so it
stores per-iteration residuals and costs O(T) memory.

`stopgrad` runs the first T-1 applications inside `lax.fori_loop` with
`stop_gradient` applied to **both** the block params and the embedding `e`. That
makes the whole prefix a constant w.r.t. the differentiated params, so
reverse-mode never enters the loop and memory is O(1) in T. Detaching only `h`
would leave `e`'s gradient path alive and force AD through the loop.

Verified by `test_stopgrad_equals_single_application_gradient`: the gradient
equals that of one block application on a frozen `h^{T-1}`, exactly.

## Non-obvious things

**Causal attention is required.** With NoPE and bidirectional attention the model
is permutation-equivariant and cannot represent a prefix task at all. Causal
masking is what lets a NoPE model distinguish positions.

**`h^0 = 0`.** The paper does not specify it; input injection re-supplies `e`
every loop, so the first application sees `[0; e]`.

**Curriculum stage is derived from the step counter** (`curriculum_n(step)`), so a
preempted job resumes into the correct stage with no extra bookkeeping. Each
stage change triggers a JAX recompile (n is a static argname) — 6 per run, cheap.

**Param counts.** Scale s = 430,210; scale m = 1,723,672 (S4) / 1,742,140 (A5).
The paper states scale s is "~1.6M params", but 2 pre-norm layers at d=128 with a
4d FFN arithmetically gives ~430K; ~1.6M is what d=256 yields. Implemented as
specified rather than back-fitted to the stated count.

## Infrastructure gotchas (each cost a launch cycle)

**Never `srun` inside an sbatch script on babel.** It requests a CPU binding that
conflicts with the job's own cgroup: `Unable to satisfy cpu bind request`, exit
64, ~3s. Call the interpreter directly.

**SLURM snapshots the batch script at submission.** Editing a launch script cannot
corrupt in-flight jobs, but already-queued jobs keep the old version — after
fixing a launcher bug, pending jobs must be resubmitted.

**wandb can hang at init and kill the job.** Two jobs starting `wandb.init`
simultaneously on one node exceeded the 90s default and raised `CommError`.
Worse, the orphaned `wandb-core` child kept the SLURM cgroup alive, so the job
showed RUNNING with frozen logs, holding a GPU while training nothing. Now:
3 retries at 120s, then a loud `WANDB_INIT_FAILED node=<host>` and a hard exit so
`scripts/relaunch.py` can blacklist the node and resubmit. Submissions are
staggered 3s apart.

**Offline fallback is deliberately removed.** Every run must land in wandb; a
failure should surface as a job failure the relauncher acts on, not a silently
degraded run.

**preempt requeues automatically.** Checkpoints every 2000 steps; an interrupted
write leaves a `*.orbax-checkpoint-tmp` dir which `latest_step()` correctly
ignores, so resume falls back to the last complete checkpoint.
