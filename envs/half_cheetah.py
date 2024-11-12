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
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="mjx",
        dense_reward: bool = False,
        **kwargs
    ):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "half_cheetah.xml")
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.003125})
            n_frames = 16
            gear = jnp.array([120, 90, 60, 120, 100, 100])
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self.dense_reward = dense_reward
        self.state_dim = 18
        self.goal_indices = jnp.array([0])
        self.goal_dist = 0.5

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        _, target = self._random_target(rng)
        qpos = qpos.at[-1:].set(target)
        qvel = qvel.at[-1:].set(0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
            "x_position": zero,
            "x_velocity": zero,
            "reward_ctrl": zero,
            "reward_run": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed}

        x_velocity = (
            pipeline_state.x.pos[0, 0] - pipeline_state0.x.pos[0, 0]
        ) / self.dt
        forward_reward = self._forward_reward_weight * x_velocity
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
            reward_run=forward_reward,
            reward_ctrl=-ctrl_cost,
            dist=dist,
            success=success,
            success_easy=success_easy
        )

        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q[:-1]
        velocity = pipeline_state.qd[:-1]

        target_pos = pipeline_state.x.pos[-1][:1]

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jnp.concatenate((position, velocity, target_pos))

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1 = jax.random.split(rng, 2)
        dist = 5
        target_x = dist
        return rng, jnp.array([target_x])