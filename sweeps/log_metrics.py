"""Recover per-eval metrics from the training logs ("step: X, key: value +/- s" lines printed by
MetricsRecorder.print_progress). Used for runs whose metrics.csv lost the training/* columns (header bug
fixed in 6259b88: smoke, gate1, gate3, gate2 SAC wave). Writes checkpoints/<set>/<run>/metrics_from_log.csv.
usage: python sweeps/log_metrics.py <set>     (CPU only)
"""

import glob
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINE = re.compile(r"step: (\d+), ([\w/]+): ([-+0-9.eEnaif]+) \+/-")


def parse_run(set_name, run):
    rows = {}
    for lp in sorted(glob.glob(os.path.join(REPO, "logs", set_name, "*"))):
        txt = open(lp, errors="ignore").read()
        if f"=== {run} ::" not in txt and f"/{run}/" not in txt:
            continue
        # take the segments that belong to this run
        segs = txt.split(f"=== {run} ::")[1:] if f"=== {run} ::" in txt else [txt]
        for seg in segs:
            seg = seg.split("\n=== ")[0] if "\n=== " in seg else seg
            for m in LINE.finditer(seg):
                step, key, val = int(m.group(1)), m.group(2), m.group(3)
                try:
                    rows.setdefault(step, {})[key] = float(val)
                except ValueError:
                    pass
    if not rows:
        return None
    return pd.DataFrame([{"step": s, **v} for s, v in sorted(rows.items())])


if __name__ == "__main__":
    set_name = sys.argv[1]
    for d in sorted(glob.glob(os.path.join(REPO, "checkpoints", set_name, "*"))):
        run = os.path.basename(d)
        df = parse_run(set_name, run)
        if df is None:
            print(run, "no log lines found")
            continue
        df.to_csv(os.path.join(d, "metrics_from_log.csv"), index=False)
        print(run, len(df), "rows,", len(df.columns), "columns")
