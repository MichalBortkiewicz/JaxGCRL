# Depth scaling of SAC+HER and TD3+HER (fork of JaxGCRL `a2740ce`)

This fork tests whether SAC and TD3 with hindsight relabelling improve with network depth when given the
network of "1000 Layer Networks for Self-Supervised RL" (Wang et al. 2025: residual blocks of
4 × (Dense → LayerNorm → Swish)) and the same hyperparameters as scaled CRL. Every change is behind a flag
whose default reproduces the original code (bug fixes excepted).

## Main flags (SAC and TD3)

| Flag | Meaning |
|---|---|
| `--network-type scaled --n-hidden D` | `ScaledMLP` of depth D (multiple of 4): stem + D/4 residual blocks + linear head |
| `--use-residual / --use-ln / --activation swish` | `full` network (all on); `plain` = `--no-use-residual --no-use-ln --activation relu` |
| `--actor-n-hidden A` | actor depth (default = critic depth) |
| `--her-strategy future --her-ratio 0.8` | standard "future" HER (default `final` = original JaxGCRL HER) |
| `--n-step N` | N-step TD targets inside the sampled window |
| `--critic-loss categorical` | categorical critic (51 atoms on [0, 1/(1-γ)], cross-entropy; LAC, Kang et al. 2026) |
| `--cat-projection scalar\|distributional`, `--num-atoms`, `--v-min`, `--v-max`, `--critic-head-ln` | categorical critic options |
| `--rep-metrics` (default on) | weight norms, feature rank (srank/erank), dormant fraction, feature norm, TD3 tanh saturation, once per epoch |
| `--mask-truncation --fix-target-policy` (TD3) | Brax-SAC truncation mask; standard target-policy update |
| `--auto-resume --run-dir DIR` | exact checkpoint/resume (replay buffer included), `DONE` / `FAILED_NAN` markers |

LAC-like setting: `--critic-loss categorical --n-step 4 --actor-n-hidden 4`.

## Running sweeps on Slurm

1. Copy `env.sh.example` to `<project>/env.sh` (one folder above this repo), edit the paths, account and
   partition, and create the Python environment at `$SSAC/env` (`pip install -e .` plus `tyro pytest pandas`).
2. `python sweeps/matrix.py <set>` writes `scripts/<set>/runs.tsv` (sets: `smoke`, `stage1`, `lac1`, `extra`, ...).
3. `python sweeps/launch.py <set> --auto-pack --time 46:00:00 --cap N` packs runs into one-GPU array tasks
   (sequential runs per GPU) and submits through `sweeps/safe_sbatch.sh` (queue limits, job ledger).
4. `sweeps/collect.py` and `sweeps/plot.py` gather `metrics.csv` files into tables and figures.

Tests: `pytest tests/` (CPU is enough).
