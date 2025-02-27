import functools
import json
import os
import pickle

import tyro
import wandb
from brax.io import model
from pyinstrument import Profiler


from src.train import train
from utils import Args


def main(args: Args):
    """Main function for training and evaluation.

    Sets up directories, initializes logging, runs training, and saves results.

    Args:
        args: Command-line arguments for configuring the training run.
            Contains parameters for model architecture, training hyperparameters,
            logging settings, etc.

    Creates the following directory structure:
        ./runs/
            run_{name}_s_{seed}/  # Run-specific directory
                args.pkl          # Saved command-line arguments
                ckpt/            # Model checkpoints
    Initializes wandb logging if enabled. Runs training with profiling and
    saves profiling results.
    """

    os.makedirs('./runs', exist_ok=True)
    run_dir = f'./runs/run_{args.exp_name}_s_{args.seed}'
    ckpt_dir = run_dir + '/ckpt'
    args.run_dir = run_dir
    args.ckpt_dir = ckpt_dir
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    train(args)


if __name__ == "__main__":
    args = tyro.cli(Args)

    print("Arguments:")
    print(
        json.dumps(
            vars(args), sort_keys=True, indent=4
        )
    )

    # NOTE: if you want to modify the utd_ratio, you can modify training_step method in src/train.py
    # For default args utd = 1024*999/256/1024/62 =~ 0.063
    utd_ratio = (
        args.num_envs
        * (args.episode_length-1)
        / args.batch_size
    ) / (args.num_envs * args.unroll_length)
    args.utd_ratio = utd_ratio
    print(f"Updates per environment step: {utd_ratio}")
    assert args.num_envs* (args.episode_length-1)%args.batch_size == 0, "Can't divide envs*episode_length-1 by batch_size - modify these parameters"

    main(args)
