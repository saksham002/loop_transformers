"""Train a weight-tied looped transformer with T = n on a prefix-product task.

Two arms, both supervising only the terminal output LN(h^T) as in the paper:
  bptt      backprop through all T loop applications (paper baseline)
  stopgrad  detach at h^{T-1}, so only the final application is differentiated
"""

import argparse
import functools
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from jax.sharding import NamedSharding, PartitionSpec as P

from loop_transformers.model import LoopedTransformer
from loop_transformers.tasks import build_task, sample_batch

# Lengths used for the length-generalization eval; T = n at eval time too.
EVAL_LENGTHS = (32, 64, 128)
EVAL_BATCHES = 8
WANDB_PROJECT = "loop_transformers"

# (n, T) sweep saved as an image every GRID_EVERY steps. n and T are decoupled
# here — the model can run any loop count on any length — which is what exposes
# the paper's budget contract (positions solved per loop).
GRID_N = (2, 4, 8, 16, 32, 64, 128)
GRID_T = (1, 2, 4, 8, 16, 32, 64, 128)
GRID_BATCHES = 2  # 56 cells per grid, so keep each cell cheap
GRID_EVERY = 10000

# Length curriculum: (n, steps at that n). T = n throughout, so the first stage
# is a single differentiated application and each later stage inherits a state
# produced by the already-trained block.
CURRICULUM = ((1, 10000), (2, 10000), (4, 10000), (8, 10000), (16, 10000), (32, 10000))
CURRICULUM_STEPS = sum(s for _, s in CURRICULUM)

# The paper's own length curriculum, Appendix D, verbatim: "stages
# n_max in {4,8,16,32} (6k-12k steps each), promotion at per-token accuracy
# >= 0.98; lengths sampled uniformly within a stage." The 6k-12k range is read
# as the observed spread of an accuracy-gated stage, so a stage ends on
# promotion or at the 12k cap, whichever comes first.
PAPER_STAGES = (4, 8, 16, 32)
PAPER_STAGE_CAP = 12000
PAPER_PROMOTE_ACC = 0.98
PAPER_PROMOTE_EVERY = 500
# Horizon curriculum, "used for chained A5/S5": the per-position loss is
# truncated to the first h positions, h ramping 4 -> n_max within each stage.
PAPER_HORIZON_START = 4


def curriculum_n(step):
    """Sequence length (and loop count) for a given step of the curriculum."""
    upto = 0
    for n, dur in CURRICULUM:
        upto += dur
        if step < upto:
            return n
    return CURRICULUM[-1][0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", required = True, choices = ["parity", "z60", "s4", "a5"]
    )
    p.add_argument("--arm", required = True, choices = ["bptt", "stopgrad"])
    p.add_argument("--scale", required = True, choices = ["s", "m"])
    p.add_argument("--n", type = int, default = 32)
    p.add_argument("--steps", type = int, default = 60000)
    p.add_argument("--batch", type = int, default = 256)
    p.add_argument("--lr", type = float, default = 3e-4)
    p.add_argument("--weight-decay", type = float, default = 0.01)
    p.add_argument("--clip", type = float, default = 1.0)
    p.add_argument("--seed", type = int, default = 86)
    p.add_argument("--eval-every", type = int, default = 1000)
    p.add_argument("--ckpt-every", type = int, default = 2000)
    p.add_argument("--ckpt-dir", required = True)
    # Appended to the run name/id, so throwaway runs cannot claim the wandb id
    # of a real one and get resumed by it later.
    p.add_argument("--run-tag", default = "")
    p.add_argument("--curriculum", action = "store_true")
    # "paper" reproduces Appendix D exactly: stages {4,8,16,32}, lengths sampled
    # uniformly within a stage, promotion at per-token acc >= 0.98, T = n.
    p.add_argument("--schedule", choices = ["fixed", "curr", "paper"], default = None)
    # Horizon curriculum; the paper applies it to chained A5/S5 only.
    p.add_argument("--horizon", action = "store_true")
    p.add_argument("--grid-every", type = int, default = GRID_EVERY)
    return p.parse_args()


def make_mesh():
    """1-D mesh over every local device, for batch-parallel training.

    Works unchanged on a single GPU (mesh of size 1), so the same code path
    serves both the scale-s single-GPU runs and the scale-m multi-GPU ones.

    Built with the Mesh constructor rather than jax.make_mesh: the latter marks
    axes Explicit, which with_sharding_constraint rejects.
    """
    return jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))


def shard_batch(mesh, x):
    """Split an array along its batch axis across the mesh."""
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data")))


def replicate(mesh, tree):
    """Put a pytree on every device (params, optimiser state)."""
    return jax.device_put(tree, NamedSharding(mesh, P()))


def loss_fn(params, model, x, y, T, arm, horizon):
    """Cross-entropy on the terminal output, truncated to the first `horizon`
    positions (the paper's horizon curriculum).

    Masked rather than sliced so `horizon` stays a traced value: slicing would
    make it a static argument and force a recompile at every ramp increment.
    Passing horizon >= n recovers the plain mean over all positions.
    """
    logits = model.apply(params, x, T, arm)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mask = (jnp.arange(ce.shape[-1]) < horizon).astype(ce.dtype)
    return (ce * mask).sum() / (mask.sum() * ce.shape[0])


@functools.partial(
    jax.jit, static_argnames = ("model", "tx", "T", "arm", "batch", "n", "mesh")
)
def train_step(
    params, opt_state, key, model, tx, table, identity, T, arm, batch, n, horizon, mesh
):
    x, y = sample_batch(key, table, identity, batch, n)
    # Data parallelism: the global batch is split across devices, gradients are
    # all-reduced by GSPMD. Global batch stays 256 as the paper specifies.
    x, y = shard_batch(mesh, x), shard_batch(mesh, y)
    loss, grads = jax.value_and_grad(loss_fn)(params, model, x, y, T, arm, horizon)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@functools.partial(jax.jit, static_argnames = ("model", "batch", "n", "T", "mesh"))
def eval_step(params, key, model, table, identity, batch, n, T = None, mesh = None):
    x, y = sample_batch(key, table, identity, batch, n)
    if mesh is not None:
        x, y = shard_batch(mesh, x), shard_batch(mesh, y)
    # Evaluation always runs the full loop forward; the arms differ only in
    # where gradients flow, so both are evaluated with the plain rollout.
    # T defaults to n, but can be set independently for the (n, T) sweep.
    pred = model.apply(params, x, n if T is None else T, "bptt").argmax(axis = -1)
    correct = pred == y
    return correct.mean(), correct.all(axis = -1).mean()


def init_wandb(run, group, config):
    """Start the wandb run, retrying a few times before giving up.

    Every run must land in wandb, so there is no offline fallback: exhausting the
    retries raises, which fails the job and lets the relauncher blacklist the
    node and resubmit elsewhere.
    """
    base = dict(
        project = WANDB_PROJECT,
        group = group,
        name = run,
        id = run,
        resume = "allow",
        config = config,
    )
    for attempt in range(3):
        try:
            return wandb.init(**base, settings = wandb.Settings(init_timeout = 120))
        except Exception as e:
            print(f"[wandb] init attempt {attempt + 1}/3 failed: {e}", flush = True)
            # Drop the half-open service, or the next attempt inherits its state.
            wandb.teardown()

    # No offline fallback: every run must land in wandb. Failing loudly with a
    # greppable marker lets the relauncher blacklist this node and resubmit.
    print(f"[wandb] WANDB_INIT_FAILED node={socket.gethostname()}", flush = True)
    raise RuntimeError(f"wandb init failed after 3 attempts on {socket.gethostname()}")


def _eval_metrics(acc):
    """Flatten {n: (token_acc, seq_acc)} into wandb keys."""
    out = {}
    for n, (tok, seq) in acc.items():
        out[f"eval/n{n}/token_acc"] = tok
        out[f"eval/n{n}/seq_acc"] = seq
    return out


def evaluate(params, model, table, identity, batch, seed, lengths = EVAL_LENGTHS,
             mesh = None):
    """Per-token and per-sequence accuracy at each eval length, T = n."""
    out = {}
    for n in lengths:
        tok, seq = [], []
        for i in range(EVAL_BATCHES):
            key = jax.random.PRNGKey(seed + 1000 * n + i)
            t, s = eval_step(params, key, model, table, identity, batch, n,
                             mesh = mesh)
            tok.append(float(t))
            seq.append(float(s))
        out[n] = (float(np.mean(tok)), float(np.mean(seq)))
    return out


def eval_grid(params, model, table, identity, batch, seed, mesh = None):
    """Token and sequence accuracy over the full (n, T) sweep."""
    tok = np.zeros((len(GRID_N), len(GRID_T)))
    seq = np.zeros((len(GRID_N), len(GRID_T)))
    for i, n in enumerate(GRID_N):
        for j, T in enumerate(GRID_T):
            ts, ss = [], []
            for b in range(GRID_BATCHES):
                key = jax.random.PRNGKey(seed + 7919 * n + 104729 * T + b)
                t, s = eval_step(params, key, model, table, identity, batch, n, T,
                                 mesh = mesh)
                ts.append(float(t))
                ss.append(float(s))
            tok[i, j] = np.mean(ts)
            seq[i, j] = np.mean(ss)
    return tok, seq


def grid_figure(tok, seq, run, step, chance):
    """Annotated heatmaps of the (n, T) sweep; returns a matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize = (13, 5.2))
    for ax, data, title in (
        (axes[0], tok, f"token accuracy (chance {chance:.4f})"),
        (axes[1], seq, "sequence accuracy"),
    ):
        im = ax.imshow(data, vmin = 0, vmax = 1, cmap = "viridis", aspect = "auto")
        ax.set_xticks(range(len(GRID_T)), [str(t) for t in GRID_T])
        ax.set_yticks(range(len(GRID_N)), [str(n) for n in GRID_N])
        ax.set_xlabel("T (loops)")
        ax.set_ylabel("n (sequence length)")
        ax.set_title(title)
        for i in range(len(GRID_N)):
            for j in range(len(GRID_T)):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}", ha = "center", va = "center",
                        fontsize = 7, color = "white" if v < 0.6 else "black")
        fig.colorbar(im, ax = ax, fraction = 0.046)
    fig.suptitle(f"{run} — step {step}", fontsize = 11)
    fig.tight_layout()
    return fig


def log_grid(params, model, table, identity, batch, seed, run, step, order, out_dir,
             mesh = None):
    """Compute the (n, T) sweep, save a PNG, and log the image to wandb."""
    tok, seq = eval_grid(params, model, table, identity, batch, seed, mesh)
    fig = grid_figure(tok, seq, run, step, 1.0 / order)
    out_dir.mkdir(parents = True, exist_ok = True)
    path = out_dir / f"grid_step{step:06d}.png"
    fig.savefig(path, dpi = 120)
    wandb.log({"grid/nT": wandb.Image(str(path))}, step = step)
    # Diagonal T = n is the contract the paper trains under; worth its own trace.
    for i, n in enumerate(GRID_N):
        if n in GRID_T:
            j = GRID_T.index(n)
            wandb.log({f"grid/diag_n{n}_seq": float(seq[i, j])}, step = step)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[grid] step {step} saved {path}", flush = True)


def main():
    args = parse_args()
    # Name encodes the design directly: arm x schedule, per task and seed.
    sched = args.schedule or ("curr" if args.curriculum else "fixed")
    run = f"{args.task}_{args.arm}_{sched}_s{args.seed}"
    if args.run_tag:
        run = f"{run}_{args.run_tag}"
    print(f"[run] {run}", flush = True)
    print(f"[node] {socket.gethostname()}", flush = True)
    print(f"[devices] {jax.devices()}", flush = True)

    mesh = make_mesh()
    n_dev = jax.device_count()
    if args.batch % n_dev:
        raise ValueError(f"batch {args.batch} not divisible by {n_dev} devices")
    print(
        f"[mesh] {n_dev} device(s), global batch {args.batch}, "
        f"{args.batch // n_dev} per device",
        flush = True,
    )

    table, identity, order = build_task(args.task)
    model = LoopedTransformer(vocab = order, n_classes = order, scale = args.scale)

    params = model.init(jax.random.PRNGKey(args.seed), args.n)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"[task] {args.task} order={order}  params={n_params:,}", flush = True)

    if sched == "paper":
        # 60k to match the other four arms per task (the single intentional
        # deviation from Appendix D). Stages promote per the paper's rules; once
        # the final stage is reached it simply holds for the remaining budget.
        total_steps = args.steps
        print(
            f"[curriculum] paper stages n_max={PAPER_STAGES} cap={PAPER_STAGE_CAP} "
            f"promote>={PAPER_PROMOTE_ACC} horizon={args.horizon} T=n",
            flush = True,
        )
    elif sched == "curr":
        total_steps = CURRICULUM_STEPS
        stages = " ".join(f"n{n}:{s}" for n, s in CURRICULUM)
        print(f"[curriculum] {stages}  total={total_steps}", flush = True)
    else:
        total_steps = args.steps

    # id=run so a requeued job after preemption resumes the same wandb run.
    init_wandb(
        run,
        args.task,
        dict(
            task = args.task,
            arm = args.arm,
            scale = args.scale,
            n = args.n,
            T = args.n,
            steps = args.steps,
            batch = args.batch,
            lr = args.lr,
            weight_decay = args.weight_decay,
            clip = args.clip,
            seed = args.seed,
            order = order,
            n_params = n_params,
            curriculum = args.curriculum,
            schedule = sched,
            horizon = args.horizon,
            total_steps = total_steps,
        ),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(args.clip),
        optax.adamw(learning_rate = args.lr, weight_decay = args.weight_decay),
    )
    opt_state = tx.init(params)
    # Weights and optimiser state live on every device; only the batch is split.
    params = replicate(mesh, params)
    opt_state = replicate(mesh, opt_state)

    mgr = ocp.CheckpointManager(
        args.ckpt_dir,
        options = ocp.CheckpointManagerOptions(max_to_keep = 2, create = True),
    )
    grid_dir = Path(args.ckpt_dir) / "grids"

    # Paper stages are promotion-gated, so stage position cannot be derived from
    # the step count and has to be checkpointed alongside the weights.
    stage = np.array(0, dtype = np.int32)
    stage_start = np.array(0, dtype = np.int32)

    def ckpt_state():
        return {
            "params": params,
            "opt_state": opt_state,
            "stage": stage,
            "stage_start": stage_start,
        }

    start_step = 0
    latest = mgr.latest_step()
    if latest is not None:
        restored = mgr.restore(latest, args = ocp.args.StandardRestore(ckpt_state()))
        params, opt_state = restored["params"], restored["opt_state"]
        stage = np.asarray(restored["stage"])
        stage_start = np.asarray(restored["stage_start"])
        start_step = latest
        print(
            f"[resume] restored from step {latest} "
            f"stage={int(stage)} stage_start={int(stage_start)}",
            flush = True,
        )

    t0 = time.time()
    running = []
    last_loss = float("nan")
    for step in range(start_step, total_steps):
        if sched == "paper":
            n_max = PAPER_STAGES[int(stage)]
            # "lengths sampled uniformly within a stage"; drawn from the step so
            # a resumed job reproduces the same sequence.
            n = int(np.random.default_rng(args.seed * 1000003 + step).integers(1, n_max + 1))
            if args.horizon:
                frac = min(1.0, (step - int(stage_start)) / PAPER_STAGE_CAP)
                h = PAPER_HORIZON_START + frac * (n_max - PAPER_HORIZON_START)
                horizon = min(n, max(1, int(round(h))))
            else:
                horizon = n
        elif sched == "curr":
            # n is a function of the step, so a resumed job lands back in the
            # right stage without extra bookkeeping.
            n = curriculum_n(step)
            horizon = n
        else:
            n = args.n
            horizon = n

        key = jax.random.fold_in(jax.random.PRNGKey(args.seed), step)
        params, opt_state, loss = train_step(
            params, opt_state, key, model, tx, table, identity,
            n, args.arm, args.batch, n, horizon, mesh,
        )
        running.append(float(loss))

        # Promotion: advance a stage on per-token accuracy >= 0.98 at n_max, or
        # when the stage hits its cap. The final stage simply holds.
        if sched == "paper" and (step + 1) % PAPER_PROMOTE_EVERY == 0:
            n_max = PAPER_STAGES[int(stage)]
            if int(stage) < len(PAPER_STAGES) - 1:
                acc = evaluate(
                    params, model, table, identity, args.batch, args.seed, (n_max,),
                    mesh = mesh,
                )
                tok = acc[n_max][0]
                capped = (step + 1 - int(stage_start)) >= PAPER_STAGE_CAP
                if tok >= PAPER_PROMOTE_ACC or capped:
                    stage = np.array(int(stage) + 1, dtype = np.int32)
                    stage_start = np.array(step + 1, dtype = np.int32)
                    why = "acc" if tok >= PAPER_PROMOTE_ACC else "cap"
                    print(
                        f"[promote] step {step + 1} -> n_max="
                        f"{PAPER_STAGES[int(stage)]} ({why}, tok={tok:.4f})",
                        flush = True,
                    )
                    wandb.log({"train/stage": int(stage)}, step = step + 1)

        if (step + 1) % 100 == 0:
            rate = (step + 1 - start_step) / (time.time() - t0)
            last_loss = float(np.mean(running))
            print(
                f"[step {step + 1}] n={n} loss={last_loss:.4f} {rate:.2f} it/s",
                flush = True,
            )
            wandb.log(
                {
                    "train/loss": float(np.mean(running)),
                    "train/it_per_s": rate,
                    "train/n": n,
                    "train/horizon": horizon,
                },
                step = step + 1,
            )
            running = []

        # Peak device memory is the quantity the stopgrad arm is meant to make
        # O(1) in T, so record it once the steady state is reached.
        if step + 1 == 100:
            stats = jax.local_devices()[0].memory_stats()
            if stats is not None:
                peak = stats["peak_bytes_in_use"] / 1e9
                print(f"[mem] peak_bytes_in_use={peak:.3f} GB", flush = True)
                wandb.log({"mem/peak_gb": peak}, step = step + 1)

        if (step + 1) % args.eval_every == 0:
            # Also evaluate at the current stage length, so progress is visible
            # before the model can handle the full n=32 comparison points.
            lengths = tuple(sorted(set(EVAL_LENGTHS + (n,))))
            acc = evaluate(
                params, model, table, identity, args.batch, args.seed, lengths,
                mesh = mesh,
            )
            parts = " ".join(
                f"n{ln}:tok={t:.4f},seq={s:.4f}" for ln, (t, s) in acc.items()
            )
            print(f"[eval {step + 1}] {parts}", flush = True)
            wandb.log(_eval_metrics(acc), step = step + 1)

        if (step + 1) % args.grid_every == 0:
            log_grid(
                params, model, table, identity, args.batch, args.seed,
                run, step + 1, order, grid_dir, mesh = mesh,
            )

        if (step + 1) % args.ckpt_every == 0:
            mgr.save(
                step + 1,
                args = ocp.args.StandardSave(ckpt_state()),
            )

    mgr.save(
        total_steps,
        args = ocp.args.StandardSave(ckpt_state()),
    )
    mgr.wait_until_finished()
    acc = evaluate(params, model, table, identity, args.batch, args.seed, mesh = mesh)
    parts = " ".join(f"n{ln}:tok={t:.4f},seq={s:.4f}" for ln, (t, s) in acc.items())
    print(f"[final] {parts}", flush = True)
    print(f"[final_loss] {last_loss:.6f}", flush = True)
    wandb.log(_eval_metrics(acc), step = total_steps)
    summary = {f"final/{k[5:]}": v for k, v in _eval_metrics(acc).items()}
    summary["final/train_loss"] = last_loss
    wandb.summary.update(summary)
    wandb.finish()
    print("[done]", flush = True)


if __name__ == "__main__":
    main()
