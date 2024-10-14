from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Pick-Place Hard: Move a cube from a random location on the blue region to a random goal on the red region. The regions are moderately-sized.
- Observation space: 46-dim obs + 7-dim goal.
- Action space:      9-dim, each element in [-1, 1], corresponding to joint target angles and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmPickplaceHard(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_pickplace_hard_mjx.xml"
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_pickplace_hard"
        self.episode_length = 200

        self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 45]) # Cube position, EEF position, and gripper finger distance
        self.completion_goal_indices = jnp.array([0, 1, 2]) # Cube position
        self.state_dim = 46

        self.arm_noise_scale = 1
        self.cube_noise_scale = 0.3
        self.goal_noise_scale = 0.3
        
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2])
        cube_q_remaining = self.sys.init_q[2:7]
        arm_q = self.sys.init_q[7:] + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 7])
        
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
    
    def _get_initial_goal(self, rng):
        rng, subkey = jax.random.split(rng)
        cube_goal_pos = jnp.array([0.3, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(subkey, [3])
        eef_goal_pos = cube_goal_pos + jnp.array([0, 0, 0.15])
        gripper_openness_goal = jnp.array([0.1])

        goal = jnp.concatenate([cube_goal_pos] + [eef_goal_pos] + [gripper_openness_goal])
        return goal
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the cube close enough to the goal? 
        current_cube_pos = obs[self.completion_goal_indices]
        goal_pos = goal[:3]
        dist = jnp.linalg.norm(current_cube_pos - goal_pos)

        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.15, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (46-dim)
         - q (16-dim): 7-dim cube position/angle, 7-dim joint angles, 2-dim finger offset
         - qd (15-dim): 6-dim cube velocity/angular velocity, 7-dim joint angular velocities, 2-dim finger offset rate of change
         - t (1-dim): normalized timestep
         - End-effector (13-dim): position/angle/velocity/angular velocity
         - Fingers (1-dim): finger distance
         
        Goal space (7-dim): position of cube, position of end-effector, distance between fingers
        """
        q = pipeline_state.q
        qd = pipeline_state.qd
        t = jnp.array([timestep])
        
        eef_index = 7 # Cube is 0, then links 1-7 are indices 1-7. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
        eef_x_pos = pipeline_state.x.pos[eef_index]
        eef_x_rot = pipeline_state.x.rot[eef_index]
        eef_xd_vel = pipeline_state.xd.vel[eef_index]
        eef_xd_angvel = pipeline_state.xd.ang[eef_index]

        left_finger_index = 8
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 9
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None] # [None] expands dims from 0 to 1
        
        return jnp.concatenate([q] + [qd] + [t] + [eef_x_pos] + [eef_x_rot] + [eef_xd_vel] + [eef_xd_angvel] + [finger_distance] + [goal])