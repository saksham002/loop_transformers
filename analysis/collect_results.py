"""Read finished runs from wandb and regenerate the results table in
project_status/experiments.md, averaging over seeds."""

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import wandb

ENTITY = "sakshamsingh2002-carnegie-mellon-university"
PROJECT = "loop_transformers"
OUT = Path("/home/saksham3/projects/AIRe/loop_transformers/project_status/experiments.md")
TASKS = ["z60", "s4", "a5"]
ARMS = ["bptt", "stopgrad"]
SCHEDS = ["fixed", "curr", "paper"]

# Which (arm, schedule) cells each task actually runs. Z60 carries the 2x2
# ablation plus the replication; S4 and A5 are replication-only.
DESIGN = {
    "z60": [("bptt", "fixed"), ("bptt", "curr"),
            ("stopgrad", "fixed"), ("stopgrad", "curr"), ("bptt", "paper")],
    "s4": [("bptt", "paper")],
    "a5": [("bptt", "paper")],
}
KEYS = [
    ("final/train_loss", "loss"),
    ("final/n32/token_acc", "n32_tok"),
    ("final/n32/seq_acc", "n32_seq"),
    ("final/n64/token_acc", "n64_tok"),
    ("final/n64/seq_acc", "n64_seq"),
    ("final/n128/token_acc", "n128_tok"),
    ("final/n128/seq_acc", "n128_seq"),
]


def fmt(vals):
    """mean +/- sd over seeds, or '-' when nothing finished."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f}±{stdev(vals):.4f}"


def main():
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))

    # cell[(task, arm, sched)][metric] = [value per seed]
    cell = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(set)
    for r in runs:
        parts = r.name.split("_")
        if len(parts) < 4:
            continue
        task, arm, sched, seed = parts[0], parts[1], parts[2], parts[3]
        if task not in TASKS or arm not in ARMS or sched not in SCHEDS:
            continue
        if r.state != "finished":
            continue
        seeds[(task, arm, sched)].add(seed)
        for key, short in KEYS:
            v = r.summary.get(key)
            # A job preempted after its last step resumes with nothing left to
            # run, so final/train_loss is never written. wandb's summary still
            # holds the last logged train/loss, so fall back to that.
            if v is None and key == "final/train_loss":
                v = r.summary.get("train/loss")
            cell[(task, arm, sched)][short].append(v)

    lines = []
    lines.append("# Results\n")
    lines.append("Z60 carries the 2x2 ablation {bptt, stopgrad} x {fixed-n, "
                 "curriculum} plus the paper replication; S4 and A5 carry the "
                 "replication only. 4 seeds (86-89), 60k steps each.\n")
    lines.append("`paper` = Appendix D exactly: stages n_max in {4,8,16,32}, "
                 "lengths sampled uniformly within a stage, promotion at "
                 "per-token acc >= 0.98, T = n; horizon curriculum on A5 only.\n")
    lines.append("Values are mean±sd over completed seeds. Token chance: Z60 "
                 "0.0167, S4 0.0417, A5 0.0167; seq chance is ~0 at n=32.\n")

    for task in TASKS:
        lines.append(f"\n## {task}\n")
        lines.append("| arm | schedule | seeds | train loss | n=32 tok | n=32 seq "
                     "| n=64 tok | n=64 seq | n=128 tok | n=128 seq |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for arm, sched in DESIGN[task]:
            c = cell[(task, arm, sched)]
            ns = len(seeds[(task, arm, sched)])
            label = "**paper**" if sched == "paper" else sched
            lines.append(
                f"| {arm} | {label} | {ns} | {fmt(c['loss'])} | "
                f"{fmt(c['n32_tok'])} | {fmt(c['n32_seq'])} | "
                f"{fmt(c['n64_tok'])} | {fmt(c['n64_seq'])} | "
                f"{fmt(c['n128_tok'])} | {fmt(c['n128_seq'])} |"
            )

    OUT.parent.mkdir(parents = True, exist_ok = True)
    header = OUT.read_text().split("<!-- RESULTS -->")[0] if OUT.exists() else ""
    OUT.write_text(header + "<!-- RESULTS -->\n" + "\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
