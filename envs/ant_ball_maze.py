import os
from typing import Tuple
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
from envs.common import make_maze, sample_choice

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py

RESET = R = 'r'
GOAL = G = 'g'
BALL = B = 'b'


U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, B, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]



BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, 0, 1, 1, G, G, 1],
            [1, 0, 0, 1, G, G, G, 1],
            [1, 1, B, G, B, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]


class AntBallMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        reset_noise_scale=0.1,
        backend="spring",
        maze_layout_name="big_maze",
        maze_size_scaling=4.0,
        dense_reward: bool = False,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "ant_ball.xml")

        if maze_layout_name == "u_maze":
            maze_layout = U_MAZE
        elif maze_layout_name == "big_maze":
            maze_layout = BIG_MAZE
        else:
            raise ValueError(f"Unknown maze layout: {maze_layout_name}")
        
        xml_string, possible_starts, possible_goals, possible_balls = make_maze(xml_path, maze_layout, maze_size_scaling)

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.possible_balls = possible_balls

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )


        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._object_idx = self.sys.link_names.index('object')
        self.dense_reward = dense_reward

        self.state_dim = 31
        self.goal_indices = jnp.array([28, 29])
        self.goal_dist = 0.5
        

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        start = sample_choice(rng2, self.possible_starts)
        target = sample_choice(rng3, self.possible_goals)
        obj = sample_choice(rng4, self.possible_balls)

        q = q.at[:2].set(start)
        q = q.at[-4:].set(jnp.concatenate([obj, target]))
        qd = qd.at[-4:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_survive": zero,
            "reward_ctrl": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        old_obs = self._get_obs(pipeline_state0)
        # Distance between goal and object
        old_dist = jnp.linalg.norm(old_obs[-2:] - old_obs[-4:-2])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[-2:] - obs[-4:-2])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 2., dtype=float)

        if self.dense_reward:
            reward = 10*vel_to_target + healthy_reward - ctrl_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            dist=dist,
            success=success,
            success_easy=success_easy
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target and object q, qd
        qpos = pipeline_state.q[:-4]
        qvel = pipeline_state.qd[:-4]

        target_pos = pipeline_state.x.pos[-1][:2]
        object_position = pipeline_state.x.pos[self._object_idx][:2]

        return jnp.concatenate([qpos] + [qvel] + [object_position] + [target_pos])