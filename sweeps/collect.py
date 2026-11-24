"""Collect every run of one or more sets into results/all_runs.csv (one row per planned or launched run).

Score = mean of eval/episode_success ("time at goal": steps within the goal threshold per 1001-step episode,
averaged over 256 eval episodes) over the last 5 evaluations. metrics.csv is de-duplicated by step (latest row
kept), because a resumed run may log a step twice.
Status: DONE, FAILED_NAN, FAILED (traceback/OOM in log), TIMEOUT, INCOMPLETE (has metrics, no marker),
MISSING (planned in runs.tsv, nothing on disk).
usage: python sweeps/collect.py smoke gate1 gate2 ...   (CPU only; run via sbatch on cpu_short or standalone)
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME_RE = re.compile(r"^(?P<method>sac|td3|crl)_(?P<variant>.+?)_d(?P<depth>\d+)_(?P<env>.+)_s(?P<seed>\d+)$")


def parse_name(n):
    m = NAME_RE.match(n)
    if not m:
        return {}
    d = m.groupdict()
    d["depth"], d["seed"] = int(d["depth"]), int(d["seed"])
    return d


def load_metrics(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    df = pd.read_csv(path)
    return df.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)


def log_status(set_name, name):
    """Look at the Slurm logs that mention this run for OOM / timeout / tracebacks."""
    status = None
    for lp in glob.glob(os.path.join(REPO, "logs", set_name, "*.log")):
        try:
            txt = open(lp, errors="ignore").read()
        except OSError:
            continue
        if f"=== {name} ::" not in txt:
            continue
        seg = txt.split(f"=== {name} ::")[-1].split("\n=== ")[0:3]
        seg = "\n".join(seg)
        if "RESOURCE_EXHAUSTED" in seg or "Out of memory" in seg or "out of memory" in seg:
            status = "FAILED_OOM"
        elif "DUE TO TIME LIMIT" in txt.split(f"=== {name} ::")[-1]:
            status = "TIMEOUT"
        elif "Traceback" in seg:
            status = "FAILED"
    return status


def collect(set_name):
    rows = []
    tsv = os.path.join(REPO, "scripts", set_name, "runs.tsv")
    planned = [l.split("\t")[0] for l in open(tsv) if l.strip()] if os.path.exists(tsv) else []
    on_disk = [os.path.basename(p) for p in glob.glob(os.path.join(REPO, "checkpoints", set_name, "*"))]
    for name in sorted(set(planned) | set(on_disk)):
        d = os.path.join(REPO, "checkpoints", set_name, name)
        row = {"set": set_name, "run": name, **parse_name(name)}
        cfg_path = os.path.join(d, "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))
            row["commit"] = str(cfg.get("commit", ""))[:8]
            row["job_id_first"] = cfg.get("slurm_job_id")
            c = cfg.get("config", {})
            for k in ("total_env_steps", "num_envs", "batch_size", "discounting", "learning_rate", "unroll_length",
                      "network_type", "use_ln", "use_residual", "activation", "her_strategy", "utd_ratio"):
                row[k] = c.get(k)
        starts = os.path.join(d, "starts.jsonl")
        row["n_starts"] = sum(1 for _ in open(starts)) if os.path.exists(starts) else 0
        df = load_metrics(os.path.join(d, "metrics.csv"))
        if df is not None and "eval/episode_success" in df:
            ev = df.dropna(subset=["eval/episode_success"])
            row["last_step"] = int(ev["step"].iloc[-1])
            row["n_evals"] = len(ev)
            row["score"] = float(ev["eval/episode_success"].iloc[-5:].mean())
            row["score_any"] = float(ev["eval/episode_success_any"].iloc[-5:].mean()) if "eval/episode_success_any" in ev else np.nan
            row["best_eval"] = float(ev["eval/episode_success"].max())
            for k in ("training/walltime", "training/sps", "training/q_mean", "training/critic_grad_norm"):
                if k in ev:
                    row[k.split("/")[1]] = float(ev[k].iloc[-1])
        if os.path.exists(os.path.join(d, "DONE")):
            row["status"] = "DONE"
        elif os.path.exists(os.path.join(d, "FAILED_NAN")):
            row["status"] = "FAILED_NAN"
        elif not os.path.exists(d):
            row["status"] = "MISSING"
        else:
            row["status"] = log_status(set_name, name) or "INCOMPLETE"
        rows.append(row)
    return rows


if __name__ == "__main__":
    sets = sys.argv[1:]
    rows = [r for s in sets for r in collect(s)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    out = os.path.join(REPO, "results", "all_runs.csv")
    if os.path.exists(out):  # keep rows of other sets
        old = pd.read_csv(out)
        old = old[~old["set"].isin(sets)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out, index=False)
    cols = [c for c in ("set", "run", "status", "n_evals", "last_step", "score", "score_any", "walltime", "sps") if c in df]
    with pd.option_context("display.max_rows", 500, "display.width", 250):
        print(df[df["set"].isin(sets)][cols].to_string(index=False))
