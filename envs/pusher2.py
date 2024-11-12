import os
from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jnp

# This is based on original Pusher environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/pusher.py

def safe_norm(x: jax.Array, axis=None):
    """
    Adapted from brax, fixed for axis.
    Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for jnp.linal.norm(0), see
    https://github.com/google/jax/issues/3058 for details.
    """

    is_zero = jnp.allclose(x, 0.0)
    # temporarily swap x with ones if is_zero, then swap back
    x = x + is_zero * 1.0
    n = jnp.linalg.norm(x, axis=axis) * (1.0 - is_zero)

    return n

class Pusher2(PipelineEnv):
    def __init__(self, backend='generalized', **kwargs):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', "pusher2.xml")
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=0.001)
            sys = sys.replace(
                actuator=sys.actuator.replace(gear=jnp.array([20.0] * sys.act_size()))
            )
            n_frames = 50

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        # The tips_arm body gets fused with r_wrist_roll_link, so we use the parent
        # r_wrist_flex_link for tips_arm_idx.
        self._tips_arm_idx = self.sys.link_names.index('r_wrist_flex_link')
        self._object_idxs = jnp.array([self.sys.link_names.index('object1'), self.sys.link_names.index('object2')])
        self._goal_idxs = jnp.array([self.sys.link_names.index('goal1'), self.sys.link_names.index('goal2')])
        
        self.state_dim = 23
        self.goal_indices = jnp.array([10, 11, 12, 13, 14, 15])

    def reset(self, rng: jax.Array) -> State:
        qpos = self.sys.init_q

        rng, rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 9)

        # randomly orient the object
        cylinder_pos1 = jnp.concatenate([
            jax.random.uniform(rng, (1,), minval=-0.35, maxval=-0.05),
            jax.random.uniform(rng1, (1,), minval=0.25, maxval=0.45 - 1e-6),
        ])

        cylinder_pos2 = jnp.concatenate([
            jax.random.uniform(rng2, (1,), minval=-0.35, maxval=-0.05),
            jax.random.uniform(rng3, (1,), minval=0.45 + 1e-6, maxval=0.65),
        ])

        goal_pos1 = jnp.concatenate([
            jax.random.uniform(rng4, (1,), minval=-0.70, maxval=0.30),
            jax.random.uniform(rng5, (1,), minval=-0.15, maxval=0.375 - 1e-6),
        ])

        goal_pos2 = jnp.concatenate([
            jax.random.uniform(rng4, (1,), minval=-0.70, maxval=0.30),
            jax.random.uniform(rng7, (1,), minval=0.375 + 1e-6, maxval=0.9),
        ])

        # constrain minimum distance of object to goal
        norm1 = safe_norm(cylinder_pos1 - goal_pos1)
        scale1 = jnp.where(norm1 < 0.17, 0.17 / norm1, 1.0)
        cylinder_pos1 *= scale1

        norm2 = safe_norm(cylinder_pos2 - goal_pos2)
        scale2 = jnp.where(norm2 < 0.17, 0.17 / norm2, 1.0)
        cylinder_pos2 *= scale2

        qpos = qpos.at[-8:].set(jnp.concatenate([cylinder_pos1, goal_pos1, cylinder_pos2, goal_pos2]))
        qvel = jax.random.uniform(
            rng6, (self.sys.qd_size(),), minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-8:].set(0.0)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jnp.zeros(3)
        metrics = {
          'reward_dist': zero, 
          'reward_ctrl': zero, 
          'reward_near': zero,
          'success': zero,
          'success_easy': zero,
        }

        info = {"seed": 0}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        assert state.pipeline_state is not None
        x_i = state.pipeline_state.x.vmap().do(
            base.Transform.create(pos=self.sys.link.inertia.transform.pos)
        )
        vec_1 = x_i.pos[self._object_idxs] - x_i.pos[self._tips_arm_idx]
        vec_2 = x_i.pos[self._object_idxs] - x_i.pos[self._goal_idxs]

        obj_to_goal_dist = safe_norm(vec_2, axis=-1)
        reward_near = -jnp.mean(safe_norm(vec_1, axis=-1))

        reward_dist = -obj_to_goal_dist.sum()
        reward_ctrl = -jnp.square(action).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        pipeline_state = self.pipeline_step(state.pipeline_state, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]

        info = {"seed": seed}

        obs = self._get_obs(pipeline_state)
        state.metrics.update(
            reward_near=reward_near,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            success=jnp.all(obj_to_goal_dist < 0.1).astype(float),
            success_easy=jnp.sum(obj_to_goal_dist < 0.1, dtype=float),
        )
        state.info.update(info)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observes pusher body position and velocities."""
        x_i = pipeline_state.x.vmap().do(
            base.Transform.create(pos=self.sys.link.inertia.transform.pos)
        )

        return jnp.concatenate([
            # state
            pipeline_state.q[:7], # Rotations of arm joints [7, ]
            x_i.pos[self._tips_arm_idx], # Arm tip position [3, ]
            x_i.pos[self._object_idxs].reshape(-1), # Movable object position [3 * num_objects, ]
            pipeline_state.qd[:7], # Rotational velocities of arm joints [7, ]
            # goal
            x_i.pos[self._goal_idxs].reshape(-1), # This is the position we want the object to end up in [3 * num_objects, ]
        ])