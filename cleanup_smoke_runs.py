"""Delete wandb runs that hold an id but contain no usable data.

Used for runs that crashed during wandb.init (before any training step) and
therefore lock their run id server-side with "run ID is in use". Only the names
listed in DELETE are touched; everything else in the project is left alone.
"""

import wandb

ENTITY = "sakshamsingh2002-carnegie-mellon-university"
PROJECT = "loop_transformers"
DELETE = {"a5_bptt_n32_m_seed86", "a5_stopgrad_n32_m_seed86"}

api = wandb.Api()
runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print(f"project has {len(runs)} run(s)")

for r in runs:
    tag = "DELETE" if r.name in DELETE else "keep  "
    steps = r.summary.get("_step", "?") if r.summary else "?"
    print(f"  [{tag}] {r.name}  state={r.state}  last_step={steps}")

for r in runs:
    if r.name in DELETE:
        r.delete()
        print(f"deleted {r.name}")

print("done")
