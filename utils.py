import argparse
import os
from collections import namedtuple
from datetime import datetime

import jax
from brax.io import html

from matplotlib import pyplot as plt
import wandb

from envs.ant import Ant
from envs.half_cheetah import Halfcheetah
from envs.reacher import Reacher
from envs.pusher import Pusher, PusherReacher
from envs.pusher2 import Pusher2
from envs.ant_ball import AntBall
from envs.ant_maze import AntMaze
from envs.humanoid import Humanoid
from envs.humanoid_maze import HumanoidMaze
from envs.ant_push import AntPush
from envs.manipulation.arm_reach import ArmReach
from envs.manipulation.arm_grasp import ArmGrasp
from envs.manipulation.arm_push_easy import ArmPushEasy
from envs.manipulation.arm_push_hard import ArmPushHard
from envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from envs.manipulation.arm_binpick_hard import ArmBinpickHard
from envs.simple_maze import SimpleMaze


def create_parser():
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument("--exp_name", type=str, default="test", help="Name of the wandb experiment")
    parser.add_argument("--group_name", type=str, default="test", help="Name of the wandb group of experiment")
    parser.add_argument("--project_name", type=str, default="crl", help="Name of the wandb project of experiment")
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--max_replay_size", type=int, default=10000, help="Maximum size of replay buffer")
    parser.add_argument("--min_replay_size", type=int, default=8192, help="Minimum size of replay buffer")
    parser.add_argument("--num_evals", type=int, default=50, help="Number of evaluations")
    parser.add_argument("--episode_length", type=int, default=50, help="Maximum length of each episode")
    parser.add_argument("--action_repeat", type=int, default=2, help="Number of times to repeat each action")
    parser.add_argument("--discounting", type=float, default=0.997, help="Discounting factor for rewards")
    parser.add_argument("--num_envs", type=int, default=256, help="Number of environments")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--unroll_length", type=int, default=50, help="Length of the env unroll")
    parser.add_argument("--multiplier_num_sgd_steps", type=int, default=1, help="Multiplier of total number of gradient steps resulting from other args.",)
    parser.add_argument("--env_name", type=str, default="reacher", help="Name of the environment to train on")
    parser.add_argument("--normalize_observations", default=False, action="store_true", help="Whether to normalize observations")
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to wandb")
    parser.add_argument('--policy_lr', type=float, default=3e-4)
    parser.add_argument('--alpha_lr', type=float, default=3e-4)
    parser.add_argument('--critic_lr', type=float, default=3e-4)
    parser.add_argument('--contrastive_loss_fn', type=str, default='symmetric_infonce')
    parser.add_argument('--energy_fn', type=str, default='l2')
    parser.add_argument('--backend', type=str, default=None)
    parser.add_argument('--no_resubs', default=False, action='store_true', help="Not use resubstitution (diagonal) for logsumexp in contrastive cross entropy")
    parser.add_argument('--use_ln', default=False, action='store_true', help="Whether to use layer normalization for preactivations in hidden layers")
    parser.add_argument('--use_c_target', default=False, action='store_true', help="Use learnable c_target param in contrastive loss")
    parser.add_argument('--logsumexp_penalty', type=float, default=0.0)
    parser.add_argument('--l2_penalty', type=float, default=0.0)
    parser.add_argument('--exploration_coef', type=float, default=0.0)
    parser.add_argument('--random_goals', type=float, default=0.0, help="Propotion of random goals to use in the actor loss")
    parser.add_argument('--disable_entropy_actor', default=False, action="store_true", help="Whether to disable entropy in actor")
    parser.add_argument('--eval_env', type=str, default=None, help="Whether to use separate environment for evaluation")
    parser.add_argument("--h_dim", type=int, default=256, help="Width of hidden layers")
    parser.add_argument("--n_hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument('--repr_dim', type=int, default=64, help="Dimension of the representation")
    parser.add_argument('--use_dense_reward', default=False, action="store_true", help="Whether to use sparse reward in env")
    parser.add_argument('--use_her', default=False, action="store_true", help="Whether to use HER for SAC")
    return parser


def create_env(args: argparse.Namespace) -> object:
    env_name = args.env_name
    if env_name == "reacher":
        env = Reacher(backend=args.backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=args.backend or "spring")
    elif env_name == "ant_ball":
        env = AntBall(backend=args.backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert args.backend == "mjx"
        env = AntPush(backend=args.backend)
    elif "maze" in env_name:
        if "ant" in env_name: 
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
            env = AntMaze(backend=args.backend or "spring", maze_layout_name=env_name[4:])
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend=args.backend or "spring", maze_layout_name=env_name[9:])
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=args.backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=args.backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=args.backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=args.backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=args.backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=args.backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=args.backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=args.backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=args.backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=args.backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=args.backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=args.backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def create_eval_env(args: argparse.Namespace) -> object:
    if not args.eval_env:
        return None
    
    eval_arg = argparse.Namespace(**vars(args))
    eval_arg.env_name = args.eval_env
    return create_env(eval_arg)

def get_env_config(args: argparse.Namespace):
    legal_envs = ["reacher", "cheetah", "pusher_easy", "pusher_hard", "pusher_reacher", "pusher2",
                  "ant", "ant_push", "ant_ball", "humanoid", "arm_reach", "arm_grasp", 
                  "arm_push_easy", "arm_push_hard", "arm_binpick_easy", "arm_binpick_hard"]
    if args.env_name not in legal_envs and "maze" not in args.env_name:
        raise ValueError(f"Unknown environment: {args.env_name}")

    args_dict = vars(args)
    Config = namedtuple("Config", [*args_dict.keys()])
    config = Config(*args_dict.values())
    
    return config


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


def render(inf_fun_factory, params, env, exp_dir, exp_name):
    inference_fn = inf_fun_factory(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)
    for i in range(5000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        if i % 1000 == 0:
            state = jit_env_reset(rng=rng)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})
