import functools
import json
import os
import pickle

import wandb
from brax.io import model
from pyinstrument import Profiler

from src.baselines.td3.td3_train import train
from utils import MetricsRecorder, get_env_config, create_env, create_eval_env, create_parser


def main(args):
    """
    Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments specifying configuration parameters for the
        training and evaluation processes.

    """
    env = create_env(**vars(args))
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
        normalize_observations=False,
        action_repeat=args.action_repeat,
        discounting=args.discounting,
        learning_rate=args.critic_lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        max_devices_per_host=1,
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        seed=args.seed,
        eval_env=eval_env,
        config=config,
    )

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

    metrics_recorder = MetricsRecorder(args.num_timesteps, metrics_to_collect, run_dir, args.exp_name)

    make_policy, params, _ = train_fn(environment=env, progress_fn=metrics_recorder.progress)
    model.save_params(ckpt_dir + '/final', params)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )
    utd_ratio = (
        args.num_envs
        * args.episode_length
        * args.train_step_multiplier
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    print(f"Updates per environment step: {utd_ratio}")
    args.utd_ratio = utd_ratio

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
