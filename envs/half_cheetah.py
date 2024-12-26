import os
from typing import Tuple
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp

# This is based on original Half Cheetah environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/half_cheetah.py

class Halfcheetah(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        backend="mjx",
        dense_reward: bool = False,
        **kwargs
    ):
        assert backend in ["mjx"]
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "half_cheetah.xml")
        sys = mjcf.load(path)

        n_frames = 5

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self.dense_reward = dense_reward
        self.state_dim = 18
        self.goal_indices = jnp.array([0])
        self.goal_dist = 0.5

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng, 2)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # Since this is mostly test/debug env the target is fixed
        qpos = qpos.at[-1:].set(5)
        qvel = qvel.at[-1:].set(0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "x_position": zero,
            "x_velocity": zero,
            "reward_ctrl": zero,
            "reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        x_velocity = (
            pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
        ) / self.dt
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state)

        dist = jnp.linalg.norm(obs[:1] - obs[-1:])
        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 2., dtype=float)

        if self.dense_reward:
            reward = ctrl_cost - dist
        else:
            reward = success

        state.metrics.update(
            x_position=pipeline_state.x.pos[0, 0],
            x_velocity=x_velocity,
            reward=reward,
            reward_ctrl=-ctrl_cost,
            dist=dist,
            success=success,
            success_easy=success_easy
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q[:-1]
        velocity = pipeline_state.qd[:-1]
        target_pos = pipeline_state.x.pos[-1][:1]

        return jnp.concatenate((position, velocity, target_pos))