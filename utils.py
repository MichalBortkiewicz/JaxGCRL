import argparse
from dataclasses import dataclass
import os
from collections import namedtuple
from datetime import datetime
from typing import List, Optional

import jax
import math
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
from envs.ant_ball_maze import AntBallMaze
from envs.simple_maze import SimpleMaze


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    cuda: bool = True
    log_wandb: bool = True
    wandb_project_name: str = "exploration"
    wandb_mode: str = 'online'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    checkpoint: bool = False
    visualization_interval: int = 5

    # Environment specific arguments
    env_name: str = "ant"
    episode_length: int = 1000
    backend: Optional[str] = None
    eval_env: Optional[str] = None
    action_repeat: int = 1
    num_eval_envs: int = 128
    use_dense_reward: bool = False

    # Algorithm specific arguments
    total_env_steps: int = 50000000
    num_evals: int = 50
    num_envs: int = 1024
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    use_her: bool = False
    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int  = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    repr_dim: int = 64
    use_ln: bool = False

    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""


def create_env(env_name: str, backend: str = None, **kwargs) -> object:
    """
    This function creates and returns an appropriate environment object based on the specified environment name and
    backend.

    Args:
        env_name (str): Name of the environment.
        backend (str): Backend to be used for the environment.

    Returns:
        object: The instantiated environment object.

    Raises:
        ValueError: If the specified environment name is unknown.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized")
    elif env_name == "ant":
        env = Ant(backend=backend or "spring")
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True)
    elif env_name == "ant_ball":
        env = AntBall(backend=backend or "spring")
    elif env_name == "ant_push":
        # This is stable only in mjx backend
        assert backend == "mjx" or backend is None
        env = AntPush(backend=backend or "mjx")
    elif "maze" in env_name:
        if "ant_ball" in env_name:
            env = AntBallMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        elif "ant" in env_name:
            # Possible env_name = {'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze'}
            env = AntMaze(backend=backend or "spring", maze_layout_name=env_name[4:])
        elif "humanoid" in env_name:
            # Possible env_name = {'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze'}
            env = HumanoidMaze(backend=backend or "spring", maze_layout_name=env_name[9:])
        else:
            # Possible env_name = {'simple_u_maze', 'simple_big_maze', 'simple_hardest_maze'}
            env = SimpleMaze(backend=backend or "spring", maze_layout_name=env_name[7:])
    elif env_name == "cheetah":
        env = Halfcheetah()
    elif env_name == "pusher_easy":
        env = Pusher(backend=backend or "generalized", kind="easy")
    elif env_name == "pusher_hard":
        env = Pusher(backend=backend or "generalized", kind="hard")
    elif env_name == "pusher_reacher":
        env = PusherReacher(backend=backend or "generalized")
    elif env_name == "pusher2":
        env = Pusher2(backend=backend or "generalized")
    elif env_name == "humanoid":
        env = Humanoid(backend=backend or "spring")
    elif env_name == "arm_reach":
        env = ArmReach(backend=backend or "mjx")
    elif env_name == "arm_grasp":
        env = ArmGrasp(backend=backend or "mjx")
    elif env_name == "arm_push_easy":
        env = ArmPushEasy(backend=backend or "mjx")
    elif env_name == "arm_push_hard":
        env = ArmPushHard(backend=backend or "mjx")
    elif env_name == "arm_binpick_easy":
        env = ArmBinpickEasy(backend=backend or "mjx")
    elif env_name == "arm_binpick_hard":
        env = ArmBinpickHard(backend=backend or "mjx")
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env


def create_eval_env(args: argparse.Namespace) -> object:
    """
    Creates an evaluation environment based on the provided arguments.

    This function generates a new environment specifically for evaluation based on
    args.eval_env. If args.eval_env is not specified, the function returns None.

    Args:
        args (argparse.Namespace): The arguments containing configuration for the
                                   environment, including eval_env.

    Returns:
        object: The created evaluation environment, or None if args.eval_env is not
                specified.
    """
    if not args.eval_env:
        return None
    
    eval_arg = argparse.Namespace(**vars(args))
    eval_arg.env_name = args.eval_env
    return create_env(**vars(eval_arg))

def get_env_config(args: argparse.Namespace):
    """
    Generate and validate environment configuration based on input arguments.

    This function takes an argparse.Namespace object, validates the specified environment name
    against a list of legal environments, and returns a configuration named tuple constructed
    from the input arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The input arguments containing the environment name and other configuration settings.

    Returns
    -------
    Config
        A named tuple containing the configuration derived from the input arguments.

    Raises
    ------
    ValueError
        If the specified environment name is not in the list of legal environments or does not
        contain the word 'maze'.
    """
    legal_envs = ["reacher", "cheetah", "pusher_easy", "pusher_hard", "pusher_reacher", "pusher2",
                  "ant", "ant_push", "ant_ball", "humanoid", "arm_reach", "arm_grasp",
                  "arm_push_easy", "arm_push_hard", "arm_binpick_easy", "arm_binpick_hard"]
    if args.env_name not in legal_envs and "maze" not in args.env_name:
        raise ValueError(f"Unknown environment: {args.env_name}")

    # TODO: round num_envs to nearest valid value instead of throwing error
    if ((args.episode_length - 1) * args.num_envs) % args.batch_size != 0:
        raise ValueError("(episode_length - 1) * num_envs must be divisible by batch_size")

    args_dict = vars(args)
    Config = namedtuple("Config", [*args_dict.keys()])
    config = Config(*args_dict.values())
    
    return config


class MetricsRecorder:
    """
    Initialize the MetricsRecorder with the specified number of timesteps
    and the metrics to be collected.

    Parameters:
    total_env_steps (int): The maximum number of timesteps for recording metrics.
    metrics_to_collect (List[str]): List of metric names that are to be collected.
    exp_dir (str): Directory to save renders to.
    exp_name (str): Experiment name for naming rendered trajectory visualizations.
    """
    def __init__(self, total_env_steps: int, metrics_to_collect: List[str], exp_dir, exp_name):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]
        self.metrics_to_collect = metrics_to_collect
        self.exp_dir = exp_dir
        self.exp_name = exp_name

        self.max_x, self.min_x = total_env_steps * 1.1, 0

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
        
    def progress(self, num_steps, metrics, make_policy, params, env, do_render=True):
        for key in self.metrics_to_collect:
            self.ensure_metric(metrics, key)
        
        if do_render:
            render(make_policy, params, env, self.exp_dir, self.exp_name, num_steps)
        
        self.record(num_steps, {key: value for key, value in metrics.items() if key in self.metrics_to_collect})
        self.log_wandb()
        self.print_progress()
    
    @staticmethod
    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is Nan")

def render(make_policy, params, env, exp_dir, exp_name, num_steps):
    """
    Renders a given environment over a series of steps and stores the resulting
    HTML file to a specified directory. Logs the rendered HTML using wandb.

    This function initializes the environment and the inference function, then
    runs the environment for a fixed number of steps, periodically resetting.
    It collects the state of the environment at each step, renders the HTML,
    stores the result, and logs it.

    Parameters:
    inf_fun_factory : Callable
        A factory function that returns an inference function when provided with
        'params'.
    params : any
        Parameters for the 'inf_fun_factory'.
    env : object
        The environment object with 'reset', 'step', and 'sys.tree_replace' methods.
    exp_dir : str
        The directory where the rendered HTML file will be saved.
    exp_name : str
        The file name to be used for the saved HTML (without extension).
    num_steps : int
        The number of environment steps taken so far (used for naming the file).

    Returns:
    None
    """
    policy = make_policy(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    rollout = []
    key = jax.random.PRNGKey(seed=1)
    key, subkey = jax.random.split(key)
    state = jit_env_reset(rng=subkey)
    for i in range(5000):
        rollout.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = jit_policy(state.obs[None], subkey) # Policy requires batched dimension
        action = action[0] # Remove batch dimension
        state = jit_env_step(state, action)
        if i % 1000 == 0:
            key, subkey = jax.random.split(key)
            state = jit_env_reset(rng=subkey)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})
