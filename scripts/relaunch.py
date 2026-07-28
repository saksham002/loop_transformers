"""Watch the grid; resubmit failed jobs, blacklisting nodes that break wandb.

A job whose log contains WANDB_INIT_FAILED could not reach wandb from its node.
Those nodes go into blacklist.txt and are passed to sbatch --exclude on every
subsequent submission, so the grid stops landing on them.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("/home/saksham3/logs/loop")
REPO = Path("/home/saksham3/projects/AIRe/loop_transformers")
BLACKLIST = REPO / "blacklist.txt"
SCALES = {"z60": "s", "s4": "m", "a5": "m"}
POLL_SECONDS = 300


def sh(cmd):
    return subprocess.run(cmd, shell = True, capture_output = True, text = True).stdout.strip()


def load_blacklist():
    if BLACKLIST.exists():
        return {n for n in BLACKLIST.read_text().split() if n}
    return set()


def save_blacklist(nodes):
    BLACKLIST.write_text("\n".join(sorted(nodes)) + "\n")


def grid_jobs():
    """{job_name: (jobid, state)} for the newest job per name."""
    out = sh("sacct -X --starttime=now-3days --format=JobID,JobName%40,State -n -P")
    jobs = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        jobid, name, state = parts[0], parts[1], parts[2].split()[0]
        if name.startswith("p_"):
            jobs[name] = (jobid, state)
    return jobs


# Failures that are the node's fault, not the job's: a node that cannot reach
# wandb or cannot init CUDA will break every job sent to it, so blacklist it.
# CUDA_ERROR_UNKNOWN is deliberately NOT here: it is emitted during teardown by
# jobs that finished perfectly well, so treating it as a node fault blacklists
# healthy nodes and starves the queue. Only these two mean the node is unusable.
NODE_FAULTS = (
    r"WANDB_INIT_FAILED node=(\S+)",
    r"no supported devices found for platform CUDA",
)


def bad_node(jobid, name):
    """Node name if this job died of a node-level fault, else None."""
    for log in LOG_DIR.glob(f"{name}_{jobid}.out"):
        text = log.read_text(errors = "ignore")
        for pat in NODE_FAULTS:
            if re.search(pat, text):
                m = re.search(r"\[node\] (\S+)", text)
                return m.group(1) if m else None
    return None


def scan_all_logs_for_faults():
    """Every node that has ever emitted a node-level fault.

    Job state is not a reliable trigger: a CUDA fault under --requeue puts the
    job back to PENDING rather than FAILED, so a state-driven check misses it and
    the requeued job can land on the same dead node again.
    """
    nodes = set()
    for log in LOG_DIR.glob("p_*.out"):
        text = log.read_text(errors = "ignore")
        # A run that reached [done] proves the node works; any error after that
        # is teardown noise and must not condemn the node.
        if "[done]" in text:
            continue
        if any(re.search(p, text) for p in NODE_FAULTS):
            m = re.search(r"\[node\] (\S+)", text)
            if m:
                nodes.add(m.group(1))
    return nodes


def resubmit(name, blacklist):
    _, task, arm, sched, seed = name.split("_")
    seed = seed.lstrip("s")
    excl = f"--exclude={','.join(sorted(blacklist))}" if blacklist else ""
    script = "scripts/train.sbatch"
    # Scale-m runs need a 48GB card; the 40GB A100 nodes OOM at T=32 batch 256.
    # Scale-m needs 2 GPUs; preempt has no L40S node with 2 free, so use general.
    gres = "--gres=gpu:L40S:2 --partition=general" if SCALES[task] == "m" else ""
    # The paper applies the horizon curriculum to chained A5 only.
    horizon = "--horizon" if (sched == "paper" and task == "a5") else ""
    cmd = (
        f"cd {REPO} && sbatch {excl} {gres} --job-name={name} {script} "
        f"{task} {SCALES[task]} {arm} {sched} {seed} {horizon}"
    )
    print(f"  resubmitting {name}: {sh(cmd)}", flush = True)


def main():
    while True:
        # Re-read every tick: the file may be edited out-of-band, and holding a
        # stale in-memory copy means resubmitting straight back to a bad node.
        blacklist = load_blacklist()

        # Catch faults on requeued jobs too, which never surface as FAILED.
        found = scan_all_logs_for_faults()
        if found - blacklist:
            print(f"  blacklisting {sorted(found - blacklist)} (log scan)", flush = True)
            blacklist |= found
            save_blacklist(blacklist)

        jobs = grid_jobs()
        done = sum(1 for _, s in jobs.values() if s == "COMPLETED")
        running = sum(1 for _, s in jobs.values() if s in ("RUNNING", "PENDING", "REQUEUED"))
        bad = {n: v for n, v in jobs.items()
               if v[1] in ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY")}

        print(f"[{time.strftime('%H:%M:%S')}] completed={done} active={running} "
              f"failed={len(bad)} blacklist={len(blacklist)}", flush = True)

        for name, (jobid, state) in bad.items():
            node = bad_node(jobid, name)
            if node and node not in blacklist:
                blacklist.add(node)
                save_blacklist(blacklist)
                print(f"  blacklisted {node} (node-level fault)", flush = True)
            resubmit(name, blacklist)

        if jobs and running == 0 and not bad:
            print(f"ALL_GRID_JOBS_DONE completed={done}/{len(jobs)}", flush = True)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
