import argparse
import functools
import os

import jax
import wandb
from brax.io import model
from brax.io import html


from crl.train import train
from envs.reacher import Reacher
from utils import MetricsRecorder

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training script arguments')
    parser.add_argument('--exp_name', type=str, default="test", help='Name of the experiment')
    parser.add_argument('--num_timesteps', type=int, default=1000000, help='Number of training timesteps')
    parser.add_argument('--max_replay_size', type=int, default=100000, help='Maximum size of replay buffer')
    parser.add_argument('--num_evals', type=int, default=50, help='Number of evaluations')
    parser.add_argument('--reward_scaling', type=float, default=0.1, help='Scaling factor for rewards')
    parser.add_argument('--episode_length', type=int, default=50, help='Maximum length of each episode')
    parser.add_argument('--normalize_observations', type=bool, default=True, help='Whether to normalize observations')
    parser.add_argument('--action_repeat', type=int, default=1, help='Number of times to repeat each action')
    parser.add_argument('--grad_updates_per_step', type=int, default=2, help='Number of gradient updates per step')
    parser.add_argument('--discounting', type=float, default=0.99, help='Discounting factor for rewards')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_envs', type=int, default=2048, help='Number of environments')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--unroll_length', type=int, default=50, help='Length of the env unroll')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    wandb.init(project="crl", name=args.exp_name, config=vars(args))

    env = Reacher()

    train_fn = functools.partial(
        train,
        num_timesteps=args.num_timesteps,
        max_replay_size=args.max_replay_size,
        num_evals=args.num_evals,
        reward_scaling=args.reward_scaling,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        grad_updates_per_step=args.grad_updates_per_step,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        unroll_length=args.unroll_length
    )

    metrics_recorder = MetricsRecorder(args.num_timesteps)

    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0

    metrics_to_collect = [
        "eval/episode_reward",
        "training/crl_critic_loss",
        "training/critic_loss",
        "training/crl_actor_loss",
        "training/actor_loss",
        "training/binary_accuracy",
        "training/categorical_accuracy",
        "training/logits_pos",
        "training/logits_neg",
        "training/logsumexp",
    ]

    def progress(num_steps, metrics):
        for key in metrics_to_collect:
            ensure_metric(metrics, key)
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        metrics_recorder.log_wandb()
        metrics_recorder.plot_progress()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    os.makedirs("./params", exist_ok=True)
    model.save_params(f'./params/param_{args.exp_name}_s_{args.seed}', params)
