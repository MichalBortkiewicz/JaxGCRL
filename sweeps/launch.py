"""Write (and optionally submit) one packed job array for a set.

Groups the selected runs into array tasks of at most --pack runs each (run sequentially in the task).
Skips runs that already have DONE or FAILED_NAN. Submits through scripts/safe_sbatch.sh (queue guard).
usage:
  python sweeps/launch.py <set> [--filter REGEX] [--pack K] [--time HH:MM:SS] [--cap N] [--wave NAME] [--dry-run]
NOTE: this file only writes text files and calls sbatch; it never imports JAX, so it may run on the login node.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=std_{set}_{wave}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-{last}%{cap}
#SBATCH --output={repo}/logs/{set}/%x_%A_%a.log
source {env_sh}
export CODE_DIR={code_dir}
echo "=== code $CODE_DIR @ $(git -C $CODE_DIR rev-parse --short HEAD)"
echo "=== start $(date) job ${{SLURM_JOB_ID}} task ${{SLURM_ARRAY_TASK_ID}} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GROUPS_FILE={repo}/scripts/{set}/{wave}_groups.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $GROUPS_FILE)
for NAME in $LINE; do
  bash {repo}/sweeps/run_one.sh {set} $NAME || echo "!!! $NAME failed (exit $?)"
done
echo "=== end $(date)"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--filter", default=".*")
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--auto-pack", action="store_true", help="pack by the runtime model to <= 80%% of --time")
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--mem", default="64G")
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--partition", default=os.environ.get("SSAC_PARTITION"))
    ap.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT_NAME"))
    ap.add_argument("--extra-sbatch", default=os.environ.get("SSAC_EXTRA_SBATCH", ""),
                    help="extra #SBATCH lines, ';'-separated, e.g. '--requeue;--qos=preempt' (Jubail)")
    ap.add_argument("--wave", default="w1")
    ap.add_argument("--max-tasks", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-started", action="store_true")
    ap.add_argument("--limit-tasks", type=int, default=None, help="submit only the first N packed tasks")
    ap.add_argument("--after", default=None, help="start only after this job id ends (afterany)")
    ap.add_argument("--no-freeze", action="store_true", help="run from the live repo instead of a frozen worktree")
    a = ap.parse_args()
    if not a.account or not a.partition:
        sys.exit("set SLURM_ACCOUNT_NAME and SSAC_PARTITION (source env.sh) or pass --account / --partition")
    if a.cap > 12:
        sys.exit("cap > 12 needs the user OK")
    h, m, s = map(int, a.time.split(":"))
    if h + m / 60 > 48:
        sys.exit("walltime > 48 h needs the user's OK")

    runs_tsv = os.path.join(REPO, "scripts", a.set, "runs.tsv")
    names = [l.split("\t")[0] for l in open(runs_tsv) if l.strip()]
    pat = re.compile(a.filter)
    # runs already assigned to a wave whose job is still queued or running
    queued = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                            capture_output=True, text=True).stdout.split()
    assigned = set()
    for gf in os.listdir(os.path.join(REPO, "scripts", a.set)):
        if gf.endswith("_groups.txt") and f"std_{a.set}_{gf[:-len('_groups.txt')]}" in queued:
            assigned.update(open(os.path.join(REPO, "scripts", a.set, gf)).read().split())
    todo = []
    for n in names:
        if not pat.search(n):
            continue
        if n in assigned and not a.include_started:
            continue
        d = os.path.join(REPO, "checkpoints", a.set, n)
        if any(os.path.exists(os.path.join(d, mk)) for mk in ("DONE", "FAILED_NAN")):
            continue
        if os.path.exists(os.path.join(d, "starts.jsonl")) and not a.include_started:
            continue  # started earlier: running now or interrupted; resubmit explicitly with --include-started
        todo.append(n)
    if a.auto_pack:
        import estimate  # noqa: E402  (sweeps/estimate.py)

        packed = sorted(estimate.pack(todo, 0.8 * (h + m / 60)), key=lambda x: x[0])  # shortest task first
        too_long = [g for hh, g in packed if hh > 0.8 * (h + m / 60)]
        if too_long:
            sys.exit(f"runs estimated longer than 80% of walltime: {too_long}")
        groups = [g for hh, g in packed]
        for hh, g in packed:
            print(f"  est {hh:5.1f} h: {' '.join(g)}")
    else:
        groups = [todo[i:i + a.pack] for i in range(0, len(todo), a.pack)]
    if a.limit_tasks:
        groups = groups[: a.limit_tasks]
    if len(groups) > a.max_tasks:
        sys.exit(f"{len(groups)} array tasks > {a.max_tasks}; use --filter or a larger --pack and submit in waves")
    if not groups:
        print("nothing to run")
        return
    os.makedirs(os.path.join(REPO, "logs", a.set), exist_ok=True)
    gfile = os.path.join(REPO, "scripts", a.set, f"{a.wave}_groups.txt")
    sfile = os.path.join(REPO, "scripts", a.set, f"{a.wave}.sbatch")
    print(f"{len(todo)} runs in {len(groups)} array tasks (pack {a.pack}, cap {a.cap}, time {a.time})")
    for i, g in enumerate(groups):
        print(f"  task {i}: {' '.join(g)}")
    if a.dry_run:
        return
    code_dir = REPO
    if not a.no_freeze:
        # Freeze the code for this wave (PLAN.md §4.B.7): a git worktree of the current HEAD inside the project.
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            sys.exit("uncommitted changes in tracked files; commit before launching a frozen wave")
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        code_dir = os.path.join(os.path.dirname(REPO), "frozen", f"{a.set}_{a.wave}_{head}")
        if not os.path.exists(code_dir):
            subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", code_dir, head], check=True)
        print("frozen code:", code_dir)
    with open(gfile, "w") as f:
        f.write("\n".join(" ".join(g) for g in groups) + "\n")
    with open(sfile, "w") as f:
        f.write(TEMPLATE.format(set=a.set, wave=a.wave, mem=a.mem, time=a.time, last=len(groups) - 1,
                                cap=min(a.cap, len(groups)), repo=REPO, code_dir=code_dir,
                                partition=a.partition, account=a.account,
                                env_sh=os.path.join(os.path.dirname(REPO), "env.sh")))
    if a.extra_sbatch:
        txt = open(sfile).read()
        extra = "".join(f"#SBATCH {x.strip()}\n" for x in a.extra_sbatch.split(";") if x.strip())
        txt = txt.replace("#SBATCH --output", extra + "#SBATCH --output", 1)
        open(sfile, "w").write(txt)
    test = subprocess.run(["sbatch", "--test-only", sfile], capture_output=True, text=True)
    print((test.stdout + test.stderr).strip())
    extra = [f"--dependency=afterany:{a.after}"] if a.after else []
    subprocess.run(["bash", os.path.join(REPO, "sweeps", "safe_sbatch.sh"), str(len(groups)), *extra, sfile], check=True)


if __name__ == "__main__":
    main()
