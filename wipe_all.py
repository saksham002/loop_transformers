"""Delete every wandb run in the project. Clean slate before the 2x2x3 grid."""

import wandb

ENTITY = "sakshamsingh2002-carnegie-mellon-university"
PROJECT = "loop_transformers"

api = wandb.Api()
runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
print(f"deleting {len(runs)} run(s)")
for r in runs:
    print(f"  deleting {r.name} (id={r.id})")
    r.delete()
print("done")
