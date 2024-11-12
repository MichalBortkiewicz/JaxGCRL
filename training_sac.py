import functools
import json
import os
import pickle

import wandb
import math
from brax.io import model
from pyinstrument import Profiler

from src.baselines.sac import train
from utils import MetricsRecorder, get_env_config, create_env, create_eval_env, create_parser, render


def main(args):

    env = create_env(args)
    eval_env = create_eval_env(args)
    config = get_env_config(args)

    os.makedirs('./runs', exist_ok=True)
    run_dir = './runs/run_{name}_s_{seed}'.format(name=args.exp_name, seed=args.seed)
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    train_fn = functools.partial(
        train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        discounting=args.discounting,
        learning_rate=args.critic_lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        multiplier_num_sgd_steps=args.multiplier_num_sgd_steps,
        config=config,
        max_devices_per_host=1,
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        seed=args.seed,
        eval_env=eval_env
    )

    metrics_recorder = MetricsRecorder(args.num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is Nan")

    metrics_to_collect = [
        "eval/episode_reward",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_ctrl",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/actor_loss",
        "training/critic_loss",
        "training/sps",
        "training/entropy",
        "training/alpha",
        "training/alpha_loss",
        "training/entropy",
    ]

    def progress(num_steps, metrics):
        for key in metrics_to_collect:
            ensure_metric(metrics, key)
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        metrics_recorder.log_wandb()
        metrics_recorder.print_progress()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    model.save_params(ckpt_dir + '/final', params)
    render(make_inference_fn, params, env, run_dir, args.exp_name)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )
    sgd_to_env = (
        args.num_envs
        * args.episode_length
        * args.multiplier_num_sgd_steps
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    print(f"SGD steps per env steps: {sgd_to_env}")
    args.sgd_to_env = sgd_to_env

    wandb.init(
        project=args.project_name,
        group=args.group_name,
        name=args.exp_name,
        config=vars(args),
        mode="online" if args.log_wandb else "disabled",
    )

    with Profiler(interval=0.1) as profiler:
        main(args)
    profiler.print()
    profiler.open_in_browser()
