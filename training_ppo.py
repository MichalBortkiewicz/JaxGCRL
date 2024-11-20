import functools
import json
import os
import pickle

import wandb
from brax.io import model
from pyinstrument import Profiler

from src.baselines.ppo import train
from utils import MetricsRecorder, create_env, create_eval_env, create_parser, render


def main(args):

    env = create_env(args)
    eval_env = create_eval_env(args)


    os.makedirs('./runs', exist_ok=True)
    run_dir = './runs/run_{name}_s_{seed}'.format(name=args.exp_name, seed=args.seed)
    ckpt_dir = run_dir + '/ckpt'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)


    # We want ratio of sgd steps to env steps to be roughly equal to 1:16
    num_minibatches = 16 #(16 * args.num_envs) // (args.batch_size * args.unroll_length * args.action_repeat)
    print(f"Num_minibatches {num_minibatches}")
    sgd_to_env_step_ratio = args.num_envs / (args.batch_size * args.unroll_length * num_minibatches * args.action_repeat)
    print(f"SGD to ENV step ratio: {sgd_to_env_step_ratio}")

    train_fn = functools.partial(
        train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        unroll_length=args.unroll_length,
        discounting=args.discounting,
        learning_rate=args.critic_lr,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=1,
        clipping_epsilon=0.3,
        gae_lambda=0.95,
        max_devices_per_host=1,
        seed=args.seed,
        eval_env=eval_env
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
    metrics_recorder = MetricsRecorder(args.num_timesteps, metrics_to_collect)

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=metrics_recorder.progress)

    os.makedirs("./params", exist_ok=True)
    model.save_params(f'./params/param_{args.exp_name}_s_{args.seed}', params)
    render(make_inference_fn, params, env, "./renders", args.exp_name)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )
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