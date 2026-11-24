import hashlib
import json
import logging
import os
import pickle
import pprint
import socket
import subprocess
import sys
import time

import tyro
from brax.io import model

import wandb
from jaxgcrl.agents.common.train_utils import NonFiniteError, write_marker
from jaxgcrl.utils.config import Config
from jaxgcrl.utils.env import MetricsRecorder, create_env


def main(config: Config):
    """Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Creates the following directory structure:
        ./runs/
            run_{name}_s_{seed}/  # Run-specific directory
                args.pkl          # Saved command-line arguments
                ckpt/            # Model checkpoints
    Initializes wandb logging if enabled. Runs training with profiling and
    saves profiling results.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_dir = config.run.run_dir or f"./runs/run_{config.run.exp_name}_s_{config.run.seed}"
    if config.run.auto_resume:
        for marker in ("DONE", "FAILED_NAN"):
            if os.path.exists(os.path.join(run_dir, marker)):
                logging.info("SKIP, %s exists in %s", marker, run_dir)
                print(f"SKIP {run_dir}: {marker} exists", flush=True)
                return
    resuming = config.run.auto_resume and os.path.exists(os.path.join(run_dir, "latest.pkl"))

    info = {**vars(config.run), **vars(config.agent)}

    utd_ratio = (
        config.run.num_envs
        * config.run.episode_length
        * config.agent.train_step_multiplier
        / config.agent.batch_size
    ) / (config.run.num_envs * config.agent.unroll_length)
    info["utd_ratio"] = utd_ratio
    info["agent"] = type(config.agent).__name__

    logging.info("Arguments:\n%s", pprint.pformat(info))

    os.makedirs(run_dir, exist_ok=True)
    repo = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "unknown", None
    run_meta = {
        "commit": commit,
        "dirty": dirty,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "start": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resuming": resuming,
        "argv": sys.argv,
    }
    if not os.path.exists(os.path.join(run_dir, "config.json")):
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"config": info, **run_meta}, f, indent=2, default=str)
    with open(os.path.join(run_dir, "starts.jsonl"), "a") as f:
        f.write(json.dumps(run_meta, default=str) + "\n")

    wandb_id = config.run.wandb_id
    if wandb_id is None and config.run.auto_resume:
        name = "".join(c if c.isalnum() or c in "-_" else "-" for c in config.run.exp_name)[:100]
        wandb_id = f"{name}-{hashlib.md5(os.path.abspath(run_dir).encode()).hexdigest()[:8]}"
    wandb.init(
        project=config.run.wandb_project_name,
        group=config.run.wandb_group,
        name=config.run.exp_name,
        config={**info, **run_meta},
        mode=config.run.wandb_mode if config.run.log_wandb else "disabled",
        id=wandb_id,
        resume="allow" if wandb_id else None,
    )
    print(f"W&B run: {getattr(wandb.run, 'url', None)}", flush=True)

    env = create_env(env_name=config.run.env, backend=config.run.backend)
    if config.run.eval_env:
        eval_env = create_env(env_name=config.run.eval_env, backend=config.run.backend)
    else:
        eval_env = env

    ckpt_dir = run_dir + "/ckpt"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + "/args.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    metrics_to_collect = [
        "eval/episode_dist",
        "eval/episode_reward",
        "eval/episode_reward_ctrl",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_survive",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_easy",
        "eval/episode_success_hard",
        "training/actor_loss",
        "training/log_alpha",
        "training/alpha_loss",
        "training/critic_loss",
        "training/entropy",
        "training/sps",
        # scaled-depth study diagnostics (also written, with every other scalar metric, to metrics.csv)
        "training/alpha",
        "training/walltime",
        "training/q_mean",
        "training/q_max",
        "training/target_q_mean",
        "training/critic_grad_norm",
        "training/actor_grad_norm",
        "training/reward_mean",
        "training/her_relabel_frac",
        "training/truncation_frac",
        "training/buffer_current_size",
        "eval/avg_episode_length",
    ]

    metrics_recorder = MetricsRecorder(
        config.run.total_env_steps,
        metrics_to_collect,
        run_dir,
        config.run.exp_name,
        mode=config.run.wandb_mode,
        csv_path=os.path.join(run_dir, "metrics.csv"),
        render=config.run.render,
    )

    try:
        _, params, _ = config.agent.train_fn(
            train_env=env,
            eval_env=eval_env,
            config=config.run,
            progress_fn=metrics_recorder.progress,
        )
    except NonFiniteError as e:
        logging.error("FAILED_NAN: %s", e)
        write_marker(run_dir, "FAILED_NAN", {"error": str(e), "last_metrics": metrics_recorder.last_metrics, **run_meta})
        if wandb.run is not None:
            wandb.run.summary["status"] = "FAILED_NAN"
        wandb.finish()
        print(f"FAILED_NAN {run_dir}: {e}", flush=True)
        return
    model.save_params(ckpt_dir + "/final", params)
    write_marker(
        run_dir,
        "DONE",
        {"final_metrics": metrics_recorder.last_metrics, "end": time.strftime("%Y-%m-%d %H:%M:%S"), **run_meta},
    )
    if wandb.run is not None:
        wandb.run.summary["status"] = "DONE"
    wandb.finish()
    print(f"DONE {run_dir}", flush=True)


def cli():
    tyro.cli(
        main,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
        ),
    )


if __name__ == "__main__":
    cli()
