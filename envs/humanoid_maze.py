from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
import os
import xml.etree.ElementTree as ET
from envs.common import sample_choice, make_maze

# This is based on original Humanoid environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/humanoid.py

# This is chosen to be very close to the z coordinate of the humanoid torso, when it is standing straight
TARGET_Z_COORD = 1.25

# Maze creation adapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py
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

MAZE_HEIGHT = 0.5


class HumanoidMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        backend='spring',
        maze_layout_name="u_maze",
        maze_size_scaling=2.0, # Was 4.0 for antmaze -- just trying to make it tractable
        **kwargs,
    ):
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "humanoid_maze.xml")
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
        
        # NOTE: maze height of 0.5 means that humanoid CAN jump over the walls, to prevent that increase to 1.5
        xml_string, possible_starts, possible_goals, _ = make_maze(xml_path, maze_layout, maze_size_scaling, maze_height=0.5)

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
            gear = jnp.array([
              350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
              350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._target_ind = self.sys.link_names.index('target')

        self.state_dim = 268
        self.goal_indices = jnp.array([0, 1, 2])

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(rng, [self.sys.q_size()], minval=low, maxval=hi)
        qvel = jax.random.uniform(rng1, [self.sys.qd_size()], minval=low, maxval=hi)

        # Set the start and target qpos and qvel
        start = sample_choice(rng2, self.possible_starts)
        target = sample_choice(rng3, self.possible_goals)
        
        qpos = qpos.at[:2].set(start)
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-2:].set(0)       

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        
        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'dist': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            "success": zero,
            "success_easy": zero,
        }
    
        state = State(pipeline_state, obs, reward, done, metrics)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state, action)
        distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        reward = -distance_to_target + healthy_reward - ctrl_cost
        success = jnp.array(distance_to_target < 0.5, dtype=float)
        success_easy = jnp.array(distance_to_target < 2., dtype=float)
        state.metrics.update(
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jnp.linalg.norm(com_after),
            dist=distance_to_target,
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            success=success,
            success_easy=success_easy,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd


        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jnp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jnp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd)

        target_pos = pipeline_state.x.pos[-1][:2]

        return jnp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
            target_pos,
            jnp.array([TARGET_Z_COORD]), # Height of the target is fixed
        ])

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(
                    jax.vmap(jnp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray