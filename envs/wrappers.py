from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from brax.envs.base import Wrapper, Env

# This factory will return wrapped env and config with updated obs_dim, goal_start_idx, goal_end_idx
def traj_index_wrapper_factory(env: PipelineEnv, config):
    wrapped_env = TrajIndexWrapper(env, config)
    new_config = config._replace(obs_dim = config.obs_dim + 3, goal_end_idx = config.goal_end_idx + 3)

    return (wrapped_env, new_config)


# Reverse what wrapper is doing
def extract_info_from_obs(obs, config):
    jax.debug.print("OBS_SHAPE {sh}\n\n", sh=obs.shape)
    old_obs = jp.concatenate([obs[:,:config.goal_end_idx-3],obs[:,config.goal_end_idx:-3]], axis=1)
    info_1 = obs[:,config.goal_end_idx-3:config.goal_end_idx]
    info_2 = obs[:,-3:]

    return (old_obs, info_1, info_2)

# This is a debug env that keeps track of each transitions:
# env_id, seed, step_number, by appending them to observation
# It is usefull to check, whether transitions don't get scrambled in replay buffer.
class TrajIndexWrapper(Wrapper):
    def __init__(self, env: PipelineEnv, config):
        super().__init__(env)
        self.obs_dim = config.obs_dim
        self.goal_start_idx = config.goal_start_idx
        self.goal_end_idx = config.goal_end_idx
        self.goal_len = self.goal_end_idx - self.goal_start_idx

    def replace_obs(self, state: State):
        env_id = state.info["env"]
        seed = state.info["seed"]
        steps = state.info["steps"] if "steps" in state.info else 0
        info_obs = jp.array([env_id, seed, steps])

        new_obs = jp.zeros(self.obs_dim + self.goal_len + 6)

        new_obs = jp.concatenate([
            state.obs[:self.goal_end_idx],
            info_obs,
            state.obs[self.goal_end_idx:],
            info_obs
        ])

        return state.replace(obs=new_obs)

    def reset(self, rng: jax.Array) -> State:
        rng1, rng2 = jax.random.split(rng, 2)
        state = self.env.reset(rng1)
        env = rng2[0] % 10000
        info = state.info
        info["env"] = env
        state.info.update(info)

        return self.replace_obs(state)
    
    def step(self, state: State, action: jax.Array) -> State:
        new_state = self.env.step(state, action)
        return self.replace_obs(new_state)
        
    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.reset(rng)
        return reset_state.obs.shape[-1]