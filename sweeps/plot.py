"""Figures for a set: depth-scaling (score vs depth, one panel per env) and learning curves.
usage: python sweeps/plot.py <set> [--envs ...]    reads results/all_runs.csv and checkpoints/<set>/*/metrics.csv
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLORS = {"sac_plain": "#7f7f7f", "sac_full": "#1f77b4", "sac_full_no_ln": "#9467bd", "sac_full_no_res": "#8c564b",
          "sac_full_no_swish": "#e377c2", "td3_plain": "#bcbd22", "td3_full": "#ff7f0e", "td3_full_no_ln": "#d62728",
          "td3_full_no_res": "#17becf", "td3_full_no_swish": "#2ca02c", "crl_scaled": "#000000"}
_BASE = set(COLORS)


_EXTRA = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]


def color(arm):
    if arm not in COLORS:
        COLORS[arm] = _EXTRA[len([k for k in COLORS if k not in _BASE]) % len(_EXTRA)]
    return COLORS[arm]


def depth_figure(df, set_name, envs, out):
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.2), squeeze=False)
    for ax, env in zip(axes[0], envs):
        sub = df[(df["env"] == env) & df["score"].notna()]
        for arm, g in sub.groupby(sub["method"] + "_" + sub["variant"]):
            stats = g.groupby("depth")["score"].agg(["mean", "std", "count"]).reset_index()
            se = stats["std"].fillna(0) / np.sqrt(stats["count"])
            c = color(arm)
            ax.plot(stats["depth"], stats["mean"], "-o", color=c, label=f"{arm}")
            ax.fill_between(stats["depth"], stats["mean"] - se, stats["mean"] + se, color=c, alpha=0.2)
            ax.scatter(g["depth"], g["score"], color=c, alpha=0.3, s=10)
            for _, r in stats.iterrows():
                ax.annotate(f"n={int(r['count'])}", (r["depth"], r["mean"]), fontsize=6, alpha=0.6)
        failed = df[(df["env"] == env) & df["status"].isin(["FAILED_NAN", "FAILED_OOM", "FAILED", "TIMEOUT"])]
        for _, r in failed.iterrows():
            ax.scatter([r["depth"]], [0], marker="x", color="red", s=30)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_xticklabels([4, 8, 16, 32, 64])
        ax.set_xlabel("depth (Dense layers in residual blocks)")
        ax.set_ylabel("time at goal (steps of 1001), mean of last 5 evals")
        ax.set_title(env)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{set_name}: mean over seeds ± 1 s.e.; dots = seeds; red x = failed run")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


def curves_figure(set_name, envs, depth, out, metric="eval/episode_success"):
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.0), squeeze=False)
    for ax, env in zip(axes[0], envs):
        for p in sorted(glob.glob(os.path.join(REPO, "checkpoints", set_name, f"*_d{depth}_{env}_s*", "metrics.csv"))):
            name = os.path.basename(os.path.dirname(p))
            arm = name.split(f"_d{depth}_")[0]
            m = pd.read_csv(p).drop_duplicates("step", keep="last")
            if metric not in m:
                continue
            m = m.dropna(subset=[metric])
            ax.plot(m["step"] / 1e6, m[metric], color=color(arm), alpha=0.8, label=arm)
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), fontsize=7)
        ax.set_title(f"{env}, depth {depth}")
        ax.set_xlabel("env steps (M)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--depths", nargs="+", type=int, default=[4, 64])
    ap.add_argument("--metrics", nargs="+", default=["eval/episode_success"])
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(REPO, "results", "all_runs.csv"))
    df = df[df["set"] == a.set]
    envs = a.envs or sorted(df["env"].dropna().unique())
    figdir = os.path.join(REPO, "reports", "figures")
    os.makedirs(figdir, exist_ok=True)
    if df["depth"].nunique() > 1:
        depth_figure(df, a.set, envs, os.path.join(figdir, f"{a.set}_depth_scaling.png"))
    for d in a.depths:
        for metric in a.metrics:
            tag = metric.split("/")[-1]
            curves_figure(a.set, envs, d, os.path.join(figdir, f"{a.set}_curves_d{d}_{tag}.png"), metric)
