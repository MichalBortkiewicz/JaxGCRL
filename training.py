import functools
import json
import os
import pickle

import wandb
from brax.io import model
from pyinstrument import Profiler


from src.train import train
from utils import MetricsRecorder, get_env_config, create_env, create_eval_env, create_parser, render


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
        max_replay_size=args.max_replay_size,
        min_replay_size=args.min_replay_size,
        num_evals=args.num_evals,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        contrastive_loss_fn=args.contrastive_loss_fn,
        energy_fn=args.energy_fn,
        logsumexp_penalty=args.logsumexp_penalty,
        l2_penalty=args.l2_penalty,
        resubs=not args.no_resubs,
        num_envs=args.num_envs,
        num_eval_envs=args.num_eval_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        unroll_length=args.unroll_length,
        multiplier_num_sgd_steps=args.multiplier_num_sgd_steps,
        config=config,
        checkpoint_logdir=ckpt_dir,
        eval_env=eval_env,
        use_ln=args.use_ln,
        h_dim=args.h_dim,
        n_hidden=args.n_hidden,
        repr_dim=args.repr_dim,
        visualization_interval=args.visualization_interval,
    )

    metrics_to_collect = [
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_hard",
        "eval/episode_success_easy",
        "eval/episode_dist",
        "eval/episode_reward_survive",
        "training/crl_critic_loss",
        "training/actor_loss",
        "training/binary_accuracy",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
        "training/sps",
        "training/entropy",
        "training/alpha",
        "training/alpha_loss",
        "training/entropy",
        "training/sa_repr_mean",
        "training/g_repr_mean",
        "training/sa_repr_std",
        "training/g_repr_std",
        "training/l_align",
        "training/l_unif",
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
