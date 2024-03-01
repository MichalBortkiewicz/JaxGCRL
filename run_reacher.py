from brax.envs.reacher import Reacher
import functools
from datetime import datetime

import jax
import mujoco
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from crl_new.train import train
from etils import epath
from jax import numpy as jp
from matplotlib import pyplot as plt
from mujoco import mjx


if __name__ == "__main__":
    env_name = 'reacher'
    env = envs.get_environment(env_name)

    #%%
    train_fn = functools.partial(
        train,
        num_timesteps=1000000,
        num_evals=500,
        reward_scaling=0.1,
        episode_length=50,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        # num_envs=2048,
        # batch_size=1024,
        # For debug purposes
        num_envs=16,
        batch_size=64,
        seed=0,
    )


    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 0, -50


    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])
        print(f"step: {num_steps}, reward: {y_data[-1]:.3f} +/- {ydataerr[-1]:.3f}")
        plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.1])
        plt.ylim([min_y, max_y])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()


    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")