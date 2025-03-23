import argparse
import logging
import math
import os
from collections import namedtuple
from datetime import datetime
from typing import List

import flax.linen as nn
import jax
import wandb_osh
from brax.io import html
from matplotlib import pyplot as plt
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from jaxgcrl.envs.ant import Ant
from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_ball_maze import AntBallMaze
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.ant_push import AntPush
from jaxgcrl.envs.half_cheetah import Halfcheetah
from jaxgcrl.envs.humanoid import Humanoid
from jaxgcrl.envs.humanoid_maze import HumanoidMaze
from jaxgcrl.envs.manipulation.arm_binpick_easy import ArmBinpickEasy
from jaxgcrl.envs.manipulation.arm_binpick_hard import ArmBinpickHard
from jaxgcrl.envs.manipulation.arm_grasp import ArmGrasp
from jaxgcrl.envs.manipulation.arm_push_easy import ArmPushEasy
from jaxgcrl.envs.manipulation.arm_push_hard import ArmPushHard
from jaxgcrl.envs.manipulation.arm_reach import ArmReach
from jaxgcrl.envs.pusher import Pusher, PusherReacher
from jaxgcrl.envs.pusher2 import Pusher2
from jaxgcrl.envs.reacher import Reacher
from jaxgcrl.envs.simple_maze import SimpleMaze

legal_envs = (
    "ant",
    "ant_random_start",
    "ant_ball",
    "ant_push",
    "humanoid",
    "reacher",
    "cheetah",
    "pusher_easy",
    "pusher_hard",
    "pusher_reacher",
    "pusher2",
    "arm_reach",
    "arm_grasp",
    "arm_push_easy",
    "arm_push_hard",
    "arm_binpick_easy",
    "arm_binpick_hard",
    "ant_ball_maze",
    "ant_u_maze",
    "ant_big_maze",
    "ant_hardest_maze",
    "humanoid_u_maze",
    "humanoid_big_maze",
    "humanoid_hardest_maze",
    "simple_u_maze",
    "simple_big_maze",
    "simple_hardest_maze",
)


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


def get_env_config(args: argparse.Namespace):
    """
    Generate and validate environment configuration based on input arguments.

    This function takes an argparse.Namespace object, validates the specified environment name
    against a list of legal environments, and returns a configuration named tuple constructed
    from the input arguments.

    Parameters
    ----------
    args : dataclass
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
    if args.env_name not in legal_envs:
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

    def __init__(
        self,
        total_env_steps: int,
        metrics_to_collect: List[str],
        exp_dir,
        exp_name,
        mode,
    ):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]
        self.metrics_to_collect = metrics_to_collect
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.mode = mode

        self.max_x, self.min_x = total_env_steps * 1.1, 0

        if mode == "offline":
            wandb_osh.set_log_level("ERROR")
        self.trigger_sync = TriggerWandbSyncHook()

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(int(num_steps))

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

        if self.mode == "offline":
            self.trigger_sync()

    def plot_progress(self):
        num_plots = len(self.y_data)
        # Calculate number of rows needed for 2 columns
        num_rows = (num_plots + 1) // 2

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
            logging.info(
                f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}"
            )

    def print_times(self):
        logging.info(f"time to jit: {self.times[1] - self.times[0]}")
        logging.info(f"time to train: {self.times[-1] - self.times[1]}")

    def progress(self, num_steps, metrics, make_policy, params, env, do_render=True):
        for key in self.metrics_to_collect:
            self.ensure_metric(metrics, key)

        if do_render:
            render(make_policy, params, env, self.exp_dir, self.exp_name, num_steps)

        self.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in self.metrics_to_collect},
        )
        self.log_wandb()
        self.print_progress()

    @staticmethod
    def ensure_metric(metrics, key):
        if key not in metrics:
            metrics[key] = 0
        else:
            if math.isnan(metrics[key]):
                raise Exception(f"Metric: {key} is NaN in metrics: {metrics}")


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
        action, _ = jit_policy(state.obs[None], subkey)  # Policy requires batched dimension
        action = action[0]  # Remove batch dimension
        state = jit_env_step(state, action)
        if i % 1000 == 0:
            key, subkey = jax.random.split(key)
            state = jit_env_reset(rng=subkey)

    url = html.render(env.sys.tree_replace({"opt.timestep": env.dt}), rollout, height=1024)
    with open(os.path.join(exp_dir, f"{exp_name}_{num_steps}.html"), "w") as file:
        file.write(url)
    wandb.log({"render": wandb.Html(url)})


def render_policy(params, save_path, env, actor, eval_env, vis_length):
    """Renders the policy and saves it as an HTML file."""

    # JIT compile the rollout function
    @jax.jit
    def policy_step(env_state, actor_params):
        means, _ = actor.apply(actor_params, env_state.obs)
        actions = nn.tanh(means)
        next_state = env.step(env_state, actions)
        return next_state, env_state  # Return current state for visualization

    rollout_states = []
    for i in range(10):
        env = create_env(eval_env) if type(eval_env) == str else eval_env

        # Initialize environment
        rng = jax.random.PRNGKey(seed=i + 1)
        env_state = jax.jit(env.reset)(rng)

        # Collect rollout using jitted function
        for _ in range(vis_length):
            env_state, current_state = policy_step(env_state, params)
            rollout_states.append(current_state.pipeline_state)

    # Render and save
    html_string = html.render(env.sys, rollout_states)
    render_path = f"{save_path}/vis.html"
    with open(render_path, "w") as f:
        f.write(html_string)
    wandb.log({"vis": wandb.Html(html_string)})
