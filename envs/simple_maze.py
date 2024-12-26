import os

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
from envs.common import sample_choice, make_maze

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = 'r'
GOAL = G = 'g'


U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

U_MAZE_EVAL = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, G, G, 1],
               [1, 1, 1, 1, 1]]



BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                 [1, R, 0, 1, 1, G, G, 1],
                 [1, 0, 0, 1, 0, G, G, 1],
                 [1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 1, G, 0, 1, G, 1],
                 [1, 0, G, G, 1, G, G, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class SimpleMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        reset_noise_scale=0.1,
        backend="spring",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "simple_maze.xml")

        if maze_layout_name == "u_maze":
            maze_layout = U_MAZE
        elif maze_layout_name == "u_maze_eval":
            maze_layout = U_MAZE_EVAL
        elif maze_layout_name == "big_maze":
            maze_layout = BIG_MAZE
        elif maze_layout_name == "big_maze_eval":
            maze_layout = BIG_MAZE_EVAL
        elif maze_layout_name == "hardest_maze":
            maze_layout = HARDEST_MAZE
        else:
            raise ValueError(f"Unknown maze layout: {maze_layout_name}")
        
        xml_string, possible_starts, possible_goals, _ = make_maze(xml_path, maze_layout, maze_size_scaling)

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

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
        self._reset_noise_scale = reset_noise_scale
        
        self.state_dim = 4
        self.goal_indices = jnp.array([0, 1])
        self.goal_dist = 0.5


    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # set the start and target q, qd
        start = sample_choice(rng2, self.possible_starts)
        target = sample_choice(rng3, self.possible_goals)

        q = q.at[:2].set(start)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state)

        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        success = jnp.array(dist < self.goal_dist, dtype=float)
        success_easy = jnp.array(dist < 2., dtype=float)
        reward = -dist - ctrl_cost
        state.metrics.update(
            reward_survive=0.0,
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
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=0.0
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])
