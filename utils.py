import argparse
from collections import namedtuple
from datetime import datetime

from matplotlib import pyplot as plt
import wandb

from envs.ant import Ant
from envs.debug_env import Debug
from envs.half_cheetah import Halfcheetah
from envs.reacher import Reacher
from envs.pusher import Pusher, PusherReacher
from envs.hard_ant import HardAnt

Config = namedtuple(
    "Config",
    "debug discount obs_dim goal_start_idx goal_end_idx unroll_length episode_length repr_dim random_goals use_old_trans_actor use_old_trans_alpha disable_entropy_actor use_traj_idx_wrapper",
)


def create_parser():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument("--exp_name", type=str, default="test", help="Name of the experiment")
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--max_replay_size", type=int, default=50000, help="Maximum size of replay buffer")
    parser.add_argument("--min_replay_size", type=int, default=8192, help="Minimum size of replay buffer")
    parser.add_argument("--num_evals", type=int, default=50, help="Number of evaluations")
    parser.add_argument("--episode_length", type=int, default=50, help="Maximum length of each episode")
    parser.add_argument("--action_repeat", type=int, default=2, help="Number of times to repeat each action")
    parser.add_argument("--discounting", type=float, default=0.997, help="Discounting factor for rewards")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--unroll_length", type=int, default=50, help="Length of the env unroll")
    parser.add_argument(
        "--multiplier_num_sgd_steps",
        type=int,
        default=1,
        help="Multiplier of total number of gradient steps resulting from other args.",
    )
    parser.add_argument("--env_name", type=str, default="reacher", help="Name of the environment to train on")
    parser.add_argument(
        "--normalize_observations", default=False, action="store_true", help="Whether to normalize observations"
    )
    parser.add_argument("--use_tested_args", default=False, action="store_true", help="Whether to use tested arguments")
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to wandb")
    parser.add_argument('--policy_lr', type=float, default=6e-4)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--contrastive_loss_fn', type=str, default='binary')
    parser.add_argument('--energy_fun', type=str, default='l2')
    parser.add_argument('--backend', type=str, default=None)
    parser.add_argument('--logsumexp_penalty', type=float, default=0.0)
    parser.add_argument('--random_goals', type=float, default=0.0, help="Propotion of random goals to use in the actor loss")
    parser.add_argument('--use_old_trans_actor', default=False, action="store_true", help="Whether to train actor with old style transitions (unflattened)")
    parser.add_argument('--use_old_trans_alpha', default=False, action="store_true", help="Whether to train alpha with old style transitions (unflattened)")
    parser.add_argument('--disable_entropy_actor', default=False, action="store_true", help="Whether to disable entropy in actor")
    parser.add_argument('--use_traj_idx_wrapper', default=False, action="store_true", help="Whether to use debug wrapper with info about envs, seeds and trajectories")
    return parser


def create_env(args: argparse.Namespace) -> object:
    env_name = args.env_name
    if env_name == "reacher":
        env = Reacher(backend=args.backend or "generalized")
    elif env_name == "ant":
        env = Ant(
            backend=args.backend or "spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
    elif env_name == "hard_ant":
        env = HardAnt(
            backend=args.backend or "spring",
            exclude_current_positions_from_observation=False,
            terminate_when_unhealthy=True,
        )
    elif env_name == "cheetah":
        env = Halfcheetah(
            backend="mjx",
            exclude_current_positions_from_observation=False,
        )
    elif env_name == "debug":
        env = Debug(backend=args.backend or "spring")
    elif env_name == "pusher_easy":
        env=Pusher(backend=args.backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env=Pusher(backend=args.backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env=PusherReacher(backend=args.backend or "generalized")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def get_env_config(args: argparse.Namespace):
    if args.env_name == "debug":
        config = Config(
            debug=True,
            discount=args.discounting,
            obs_dim=3,
            goal_start_idx=1,
            goal_end_idx=3,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "reacher":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=10,
            goal_start_idx=4,
            goal_end_idx=7,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "cheetah":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=18,
            goal_start_idx=0,
            goal_end_idx=1,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "pusher_easy" or args.env_name == "pusher_hard":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=20,
            goal_start_idx=10,
            goal_end_idx=13,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "pusher_reacher":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=17,
            goal_start_idx=14,
            goal_end_idx=17,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "ant":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=29,
            goal_start_idx=0,
            goal_end_idx=2,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    elif args.env_name == "hard_ant":
        config = Config(
            debug=False,
            discount=args.discounting,
            obs_dim=31,
            goal_start_idx=-4,
            goal_end_idx=-2,
            unroll_length=args.unroll_length,
            episode_length=args.episode_length,
            repr_dim=64,
            random_goals=args.random_goals,
            use_old_trans_actor=args.use_old_trans_actor,
            use_old_trans_alpha=args.use_old_trans_alpha,
            disable_entropy_actor=args.disable_entropy_actor,
            use_traj_idx_wrapper=args.use_traj_idx_wrapper
        )
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")
    return config


def get_tested_args(args):  # Parse arguments
    if args.env_name == "reacher":
        # NOTE: it was tested on old RB, which was flat and used grad_updates_per_step
        parameters = {
            "num_evals": 25,
            "seed": 1,
            "num_timesteps": 5000000,
            "batch_size": 256,
            "num_envs": 1024,
            "exp_name": "crl_proper_500",
            "episode_length": 1000,
            "unroll_length": 50,
            "action_repeat": 1,
            "min_replay_size": 1000,
            "normalize_observations": True,
        }
    elif args.env_name == "ant":
        # Best works with bigger networks - 4x1024
        parameters = {
            "num_evals": 50,
            "seed": 1,
            "num_timesteps": 50000000,
            "batch_size": 256,
            "num_envs": 1024,
            "exp_name": "ant_repro",
            "episode_length": 1000,
            "unroll_length": 50,
            "action_repeat": 1,
            "min_replay_size": 1000,
            "normalize_observations": True,
        }
    else:
        raise ValueError(f"Unknown environment: {args.env_name}")
    # Update only changed args
    for key, value in vars(args).items():
        if key in parameters:
            setattr(args, key, parameters[key])
    return args


class MetricsRecorder:
    def __init__(self, num_timesteps):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]

        self.max_x, self.min_x = num_timesteps * 1.1, 0

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

    def plot_progress(self):
        num_plots = len(self.y_data)
        num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")

    def print_times(self):
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")
