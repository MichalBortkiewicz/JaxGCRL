from brax import base
from brax.envs.base import State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp
from envs.manipulation.arm_envs import ArmEnvs

"""
Grasp: Close fingers on opposite sides of a cube.
- Observation space: 39-dim obs + 10-dim goal.
- Action space:      9-dim, each element in [-1, 1], corresponding to joint target angles and finger closedness.

See _get_obs() and ArmEnvs._convert_action() for details.
"""
class ArmGrasp(ArmEnvs):
    def _get_xml_path(self):
        return "envs/assets/panda_grasp_mjx.xml"
    
    # See ArmEnvs._set_environment_attributes for descriptions of attributes
    def _set_environment_attributes(self):
        self.env_name = "arm_grasp"
        self.episode_length = 100

        self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # Cube position, left and right finger positions, and gripper finger distance
        self.completion_goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # Identical
        self.state_dim = 39

        self.arm_noise_scale = 1
        self.cube_noise_scale = 0.3
        
    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2])
        cube_q_remaining = self.sys.init_q[2:7]
        arm_q = self.sys.init_q[7:] + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 7])
        
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd
        
    def _get_initial_goal(self, rng):
        # Note that we have no better way than to specify the finger goal positions and the cube goal positions, but the actual
        # requirement for goal completion is looser (finger midpoint near cube center + gripper is closed enough).
        cube_goal_pos = q[:3]
        left_finger_goal_pos = cube_goal_pos + jnp.array([0, 0.03, 0])
        right_finger_goal_pos = cube_goal_pos + jnp.array([0, -0.03, 0])
        gripper_openness_goal = jnp.array([0.06]) # The cube itself is 0.06 wide
        
        goal = jnp.concatenate([cube_goal_pos] + [left_finger_goal_pos] + [right_finger_goal_pos] + [gripper_openness_goal])
        return goal
        
    def _compute_goal_completion(self, obs, goal):
        # Goal occupancy: is the midpoint of the fingers close enough to the cube, and is the gripper closed enough?
        # Technically, only success_hard is properly gripping the cube, but success/success_easy are for signs of life.
        cube_pos = obs[:3]
        left_finger_pos = obs[32:35]
        right_finger_pos = obs[35:38]
        finger_midpoint = (left_finger_pos + right_finger_pos) / 2
        cube_to_finger_midpoint_dist = jnp.linalg.norm(cube_pos - finger_midpoint)

        gripper_openness = obs[38]
        goal_gripper_openness = goal[9]
        gripper_openness_difference = jnp.linalg.norm(gripper_openness - goal_gripper_openness)

        success = jnp.array(
            jnp.all(jnp.array([
                cube_to_finger_midpoint_dist < 0.05,
                gripper_openness_difference < 0.02
            ])), 
            dtype=float
        )
        success_easy = jnp.array(
            jnp.all(jnp.array([
                cube_to_finger_midpoint_dist < 0.15,
                gripper_openness_difference < 0.05
            ])), 
            dtype=float
        )
        success_hard = jnp.array(
            jnp.all(jnp.array([
                cube_to_finger_midpoint_dist < 0.01,
                gripper_openness_difference < 0.005
            ])), 
            dtype=float
        )
        return success, success_easy, success_hard
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (39-dim)
         - q (16-dim): 7-dim cube position/angle, 7-dim joint angles, 2-dim finger offset
         - qd (15-dim): 6-dim cube velocity/angular velocity, 7-dim joint angular velocities, 2-dim finger offset rate of change
         - t (1-dim): normalized timestep
         - Fingers (7-dim): 6-dim positions of fingers, 1-dim distance between fingers
         
        Goal space (10-dim): position of cube, position of fingers, distance between fingers
        """
        q = pipeline_state.q
        qd = pipeline_state.qd
        t = jnp.array([timestep])
        
        # Cube is 0, then links 1-7 are indices 1-7. Fingers are 8 and 9.
        left_finger_index = 8
        left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
        right_finger_index = 9
        right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
        finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None] # [None] expands dims from 0 to 1
        
        return jnp.concatenate([q] + [qd] + [t] + [left_finger_x_pos] + [right_finger_x_pos] + [finger_distance] + [goal])