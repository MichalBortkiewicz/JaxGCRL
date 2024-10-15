from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Reach: Move end of arm to random goal.
- Observation space: 28-dim obs + 3-dim goal.
- Action space:      7-dim, each element in [-1, 1], corresponding to joint target angles.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmReach(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_reach_mjx.xml"
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_reach"
        self.episode_length = 100 

        self.goal_indices = jnp.array([15, 16, 17]) # End-effector position
        self.completion_goal_indices = jnp.array([15, 16, 17]) # Identical
        self.state_dim = 28

        self.arm_noise_scale = 1
        self.goal_noise_scale = 0.3
        
    def _get_initial_state(self, rng):
        q = self.sys.init_q + self.arm_noise_scale * jax.random.uniform(rng, [self.sys.q_size()])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, rng):
        goal = jnp.array([0, 0.3, 0.3]) + self.goal_noise_scale * jax.random.uniform(rng, [3])
        return goal
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the end of the arm close enough to the goal?
        eef_pos = obs[self.completion_goal_indices]
        goal_eef_pos = goal[:3]
        dist = jnp.linalg.norm(eef_pos - goal_eef_pos)

        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.25, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        
        return success, success_easy, success_hard
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (28-dim)
         - q (7-dim): joint angles
         - qd (7-dim): joint angular velocities
         - t (1-dim): normalized timestep
         - End of arm (13-dim): position/angle/velocity/angular velocity
         
        Goal space (3-dim): position of end of arm
        """
        q = pipeline_state.q
        qd = pipeline_state.qd
        t = jnp.array([timestep])
        
        eef_index = 6 # Links 1-7 are indices 0-6. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_x_rot = pipeline_state.x.rot[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        eef_xd_angvel = pipeline_state.xd.ang[eef_index]
        
        return jnp.concatenate([q] + [qd] + [t] + [eef_x_pos] + [eef_x_rot] + [eef_xd_vel] + [eef_xd_angvel] + [goal])