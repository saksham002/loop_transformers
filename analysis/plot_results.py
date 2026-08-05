"""Fetch every run from wandb and render readable summary plots.

Writes into project_status/figures/:
  fig1_n32_seq.png        headline: n=32 sequence accuracy per cell, seeds shown
  fig2_length_gen.png     accuracy vs eval length (32/64/128), per task
  fig3_train_loss.png     final train loss vs the marginal-predictor floor
  fig4_curves.png         training-loss curves, seeds overlaid

Usage:  python analysis/plot_results.py [--out DIR]
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "sakshamsingh2002-carnegie-mellon-university"
PROJECT = "loop_transformers"
TASKS = ["z60", "s4", "a5"]
ORDER = {"z60": 60, "s4": 24, "a5": 60, "parity": 2}
# (arm, schedule) -> label, colour. Kept stable across every figure.
CELLS = [
    ("bptt", "fixed", "BPTT · fixed n", "#4C72B0"),
    ("bptt", "curr", "BPTT · curriculum", "#55A868"),
    ("stopgrad", "fixed", "stopgrad · fixed n", "#C44E52"),
    ("stopgrad", "curr", "stopgrad · curriculum", "#8172B2"),
    ("bptt", "paper", "paper replication", "#CCB974"),
]
DESIGN = {
    task: [("bptt", "fixed"), ("bptt", "curr"),
           ("stopgrad", "fixed"), ("stopgrad", "curr"),
           ("stopgrad", "currmax"), ("bptt", "paper")]
    for task in TASKS
}
EXPECTED_RUNS = {
    f"{task}_{arm}_{sched}_s{seed}"
    for task, cells in DESIGN.items()
    for arm, sched in cells
    for seed in range(86, 90)
}


def fetch():
    """{(task, arm, sched): [run, ...]} for finished runs only."""
    api = wandb.Api()
    out = defaultdict(list)
    for r in api.runs(f"{ENTITY}/{PROJECT}"):
        if r.name not in EXPECTED_RUNS or r.state != "finished":
            continue
        parts = r.name.split("_")
        task, arm, sched = parts[0], parts[1], parts[2]
        out[(task, arm, sched)].append(r)
    return out


def _seed_values(runs, key, fallback = None):
    vals = []
    for r in runs:
        v = r.summary.get(key)
        if v is None and fallback:
            v = r.summary.get(fallback)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return vals


def fig_headline(data, out):
    """n=32 sequence accuracy: the metric that separates the cells."""
    fig, axes = plt.subplots(1, len(TASKS), figsize = (14, 4.4), sharey = True)
    for ax, task in zip(axes, TASKS):
        xs, means, errs, cols, labels = [], [], [], [], []
        for i, (arm, sched, label, colour) in enumerate(CELLS):
            v = _seed_values(data.get((task, arm, sched), []), "final/n32/seq_acc")
            if not v:
                continue
            xs.append(len(xs))
            means.append(np.mean(v))
            errs.append(np.std(v) if len(v) > 1 else 0.0)
            cols.append(colour)
            labels.append(f"{label}\n(n={len(v)})")
            ax.scatter([xs[-1]] * len(v), v, color = "k", s = 14, zorder = 3, alpha = .65)
        ax.bar(xs, means, yerr = errs, color = cols, capsize = 4, zorder = 2)
        ax.set_xticks(xs, labels, rotation = 30, ha = "right", fontsize = 7)
        ax.set_title(f"{task}  (order {ORDER[task]})")
        ax.set_ylim(0, 1.02)
        ax.grid(axis = "y", alpha = .3)
    axes[0].set_ylabel("n=32 sequence accuracy")
    fig.suptitle("Exact-match accuracy at the training length (dots = seeds)")
    fig.tight_layout()
    fig.savefig(out / "fig1_n32_seq.png", dpi = 140)
    plt.close(fig)


def fig_length_gen(data, out):
    """Accuracy vs eval length — none of these extrapolate far."""
    lengths = [32, 64, 128]
    fig, axes = plt.subplots(2, len(TASKS), figsize = (14, 7), sharex = True)
    for col, task in enumerate(TASKS):
        for row, metric in enumerate(["token_acc", "seq_acc"]):
            ax = axes[row][col]
            for arm, sched, label, colour in CELLS:
                runs = data.get((task, arm, sched), [])
                if not runs:
                    continue
                ys = [np.mean(_seed_values(runs, f"final/n{n}/{metric}") or [np.nan])
                      for n in lengths]
                ax.plot(lengths, ys, "o-", color = colour, label = label, lw = 1.8)
            if metric == "token_acc":
                ax.axhline(1.0 / ORDER[task], ls = ":", c = "gray", label = "chance")
                ax.set_title(f"{task}")
            ax.set_xscale("log", base = 2)
            ax.set_xticks(lengths, [str(x) for x in lengths])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha = .3)
            if col == 0:
                ax.set_ylabel(metric.replace("_", " "))
            if row == 1:
                ax.set_xlabel("eval length n (T = n)")
    axes[0][-1].legend(fontsize = 7, loc = "upper right")
    fig.suptitle("Length generalisation (trained at n=32)")
    fig.tight_layout()
    fig.savefig(out / "fig2_length_gen.png", dpi = 140)
    plt.close(fig)


def fig_train_loss(data, out):
    """Final train loss against ln(order): the marginal-predictor floor."""
    fig, axes = plt.subplots(1, len(TASKS), figsize = (14, 4.2))
    for ax, task in zip(axes, TASKS):
        floor = math.log(ORDER[task])
        xs, means, errs, cols, labels = [], [], [], [], []
        for arm, sched, label, colour in CELLS:
            v = _seed_values(data.get((task, arm, sched), []),
                             "final/train_loss", "train/loss")
            if not v:
                continue
            xs.append(len(xs))
            means.append(np.mean(v))
            errs.append(np.std(v) if len(v) > 1 else 0.0)
            cols.append(colour)
            labels.append(label)
        ax.bar(xs, means, yerr = errs, color = cols, capsize = 4)
        ax.axhline(floor, ls = "--", c = "crimson",
                   label = f"ln({ORDER[task]}) = {floor:.3f}")
        ax.set_xticks(xs, labels, rotation = 30, ha = "right", fontsize = 7)
        ax.set_title(task)
        ax.legend(fontsize = 7)
        ax.grid(axis = "y", alpha = .3)
    axes[0].set_ylabel("final training loss")
    fig.suptitle("Final loss vs the marginal-predictor floor (at the floor = learned nothing)")
    fig.tight_layout()
    fig.savefig(out / "fig3_train_loss.png", dpi = 140)
    plt.close(fig)


def fig_curves(data, out):
    """Training-loss curves, one panel per task, seeds overlaid."""
    fig, axes = plt.subplots(1, len(TASKS), figsize = (14, 4.4))
    for ax, task in zip(axes, TASKS):
        for arm, sched, label, colour in CELLS:
            runs = data.get((task, arm, sched), [])
            for k, r in enumerate(runs):
                try:
                    h = r.history(keys = ["train/loss"], samples = 300, pandas = False)
                except Exception:
                    continue
                pts = [(d["_step"], d["train/loss"]) for d in h if "train/loss" in d]
                if not pts:
                    continue
                s, l = zip(*sorted(pts))
                ax.plot(s, l, color = colour, alpha = .55, lw = 1.1,
                        label = label if k == 0 else None)
        ax.axhline(math.log(ORDER[task]), ls = "--", c = "crimson", lw = 1)
        ax.set_title(task)
        ax.set_xlabel("step")
        ax.grid(alpha = .3)
    axes[0].set_ylabel("train loss")
    axes[-1].legend(fontsize = 7)
    fig.suptitle("Training curves (dashed = marginal-predictor floor)")
    fig.tight_layout()
    fig.savefig(out / "fig4_curves.png", dpi = 140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default =
                    "/home/saksham3/projects/AIRe/loop_transformers/project_status/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents = True, exist_ok = True)

    data = fetch()
    n = sum(len(v) for v in data.values())
    print(f"fetched {n} finished run(s) across {len(data)} cell(s)")
    for (task, arm, sched), runs in sorted(data.items()):
        print(f"  {task:6s} {arm:9s} {sched:6s} seeds={len(runs)}")

    fig_headline(data, out)
    fig_length_gen(data, out)
    fig_train_loss(data, out)
    fig_curves(data, out)
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
