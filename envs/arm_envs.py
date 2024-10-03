from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp

"""
Observation and action spaces listed below. See _get_obs() and convert_action() for details.

Reach 
- Observation space: 28-dim obs + 3-dim goal.
- Action space:      7-dim, each element in [-1, 1], corresponding to joint target angles.

Grasp 
- Observation space: 39-dim obs + 10-dim goal.
- Action space:      9-dim, each element in [-1, 1], corresponding to joint target angles and finger closedness.

Pick-Place Easy/Hard, Binpick:
- Observation space: 46-dim obs + 7-dim goal.
- Action space:      9-dim, each element in [-1, 1], corresponding to joint target angles and finger closedness.
"""
class ArmEnvs(PipelineEnv):
    def __init__(self, backend="mjx", env_name="arm_pickplace_easy", **kwargs):
        # Load XML for environment, configure environment information
        # 
        # Remarks:
        # - Episode lengths are merely a recommendation + used for scaling timestep
        # - Goal indices are for data collection/goal-conditioning; completion goal indices are for reward/goal completion checking
        # - Obs_dim is for the observation dimension WITHOUT the goal appended; brax's env.observation_size gives the dimension WITH goal appended
        # - Noise scale is for the range of uniform-distribution noise
        #   - Arm: perturb all joints (in radians) and gripper position (in meters)
        #   - Cube: perturb x, y of starting cube location
        #   - Goal: perturb x, y of cube destination or x, y, z of reach location
        self.env_name = env_name
        if env_name == "arm_reach":
            sys = mjcf.load("envs/assets/panda_reach_mjx.xml")
            
            self.episode_length = 100 
            
            self.goal_indices = jnp.array([15, 16, 17]) # End-effector position
            self.completion_goal_indices = jnp.array([15, 16, 17]) # Identical
            self.obs_dim = 28

            self.arm_noise_scale = 1
            self.goal_noise_scale = 0.3
        
        elif env_name == "arm_grasp":
            sys = mjcf.load("envs/assets/panda_grasp_mjx.xml")
            
            self.episode_length = 100
            
            self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # Cube position, left and right finger positions, and gripper finger distance
            self.completion_goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # Identical
            self.obs_dim = 39

            self.cube_noise_scale = 0.3
            self.arm_noise_scale = 1
            
        elif env_name == "arm_pickplace_easy":
            sys = mjcf.load("envs/assets/panda_pickplace_easy_mjx.xml")
            
            self.episode_length = 200
            
            self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 45]) # Cube position, EEF position, and gripper finger distance
            self.completion_goal_indices = jnp.array([0, 1, 2]) # Cube position
            self.obs_dim = 46

            self.arm_noise_scale = 1
            self.cube_noise_scale = 0.1
            self.goal_noise_scale = 0.1
        
        elif env_name == "arm_pickplace_hard":
            sys = mjcf.load("envs/assets/panda_pickplace_hard_mjx.xml")
            
            self.episode_length = 200
            
            self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 45]) # Cube position, EEF position, and gripper finger distance
            self.completion_goal_indices = jnp.array([0, 1, 2]) # Cube position
            self.obs_dim = 46

            self.arm_noise_scale = 1
            self.cube_noise_scale = 0.3
            self.goal_noise_scale = 0.3
        
        elif env_name == "arm_binpick":
            sys = mjcf.load("envs/assets/panda_binpick_mjx.xml")
            
            self.episode_length = 200
            
            self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 45]) # Cube position, EEF position, and gripper finger distance
            self.completion_goal_indices = jnp.array([0, 1, 2]) # Cube position
            self.obs_dim = 46

            self.arm_noise_scale = 1
            self.cube_noise_scale = 0.15
            self.goal_noise_scale = 0.15
        
        else:
            raise Exception(f"Unknown environment name: {env_name}.") 
        
        # Select backend and manage simulation parameters
        if backend == "mjx":
            sys = sys.tree_replace({
                "opt.timestep": 0.01,
                "opt.iterations": 4,
                "opt.ls_iterations": 8,
            })
            
            self.n_frames = 4
            kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)
            super().__init__(sys=sys, backend=backend, **kwargs)
        else:
            raise Exception("Use the mjx backend for stability/reasonable speed.")
        
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        
        # Variance in starting arm/cube state and goal state to avoid overfitting/closed loop control
        if self.env_name == "arm_reach":
            rng, subkey = jax.random.split(rng)
            q = self.sys.init_q + self.arm_noise_scale * jax.random.uniform(subkey, [self.sys.q_size()])
            qd = jnp.zeros([self.sys.qd_size()])
            
        elif self.env_name in {"arm_grasp", "arm_pickplace_easy", "arm_pickplace_hard", "arm_binpick"}:
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2])
            cube_q_remaining = self.sys.init_q[2:7]
            arm_q = self.sys.init_q[7:] + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 7])
            q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [arm_q])
            qd = jnp.zeros([self.sys.qd_size()])
            
        else:
            raise Exception(f"Starting state setup not implemented for environment: {self.env_name}.")
        
        # Initialize state
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0
        
        # Define goal distribution
        if self.env_name == "arm_reach":
            rng, subkey = jax.random.split(rng)
            goal = jnp.array([0, 0.3, 0.3]) + self.goal_noise_scale * jax.random.uniform(subkey, [3])
            
        elif self.env_name == "arm_grasp":
            # Note that we have no better way than to specify the finger goal positions and the cube goal positions, but the actual
            # requirement for goal completion is looser (finger midpoint near cube center + gripper is closed enough).
            cube_goal_pos = q[:3]
            left_finger_goal_pos = cube_goal_pos + jnp.array([0, 0.03, 0])
            right_finger_goal_pos = cube_goal_pos + jnp.array([0, -0.03, 0])
            gripper_openness_goal = jnp.array([0.06]) # The cube itself is 0.06 wide
            goal = jnp.concatenate([cube_goal_pos] + [left_finger_goal_pos] + [right_finger_goal_pos] + [gripper_openness_goal])
        
        elif self.env_name in {"arm_pickplace_easy", "arm_pickplace_hard", "arm_binpick"}:
            rng, subkey = jax.random.split(rng)
            cube_goal_pos = jnp.array([0.1, 0.6, 0.03]) + jnp.array([self.goal_noise_scale, self.goal_noise_scale, 0]) * jax.random.uniform(subkey, [3])
            eef_goal_pos = cube_goal_pos + jnp.array([0, 0, 0.15])
            gripper_openness_goal = jnp.array([0.1])
            goal = jnp.concatenate([cube_goal_pos] + [eef_goal_pos] + [gripper_openness_goal])
        
        else:
            raise Exception(f"Goal distribution not implemented for environment: {self.env_name}.")
            
        # Get other components for state (obs, reward, etc.)
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        
        # Fill info variable
        rng, subkey = jax.random.split(rng)
        info = {
            "seed": 0, # Seed is required, but fill it with a dummy value
            "goal": goal, 
            "timestep": 0.0, 
            "postexplore_timestep": jax.random.uniform(subkey) # Assumes timestep is normalized between 0 and 1
        } 
        
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        
        # Run mujoco step, compute non-goal-completion information
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, self.convert_action(action))
        
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
        done = 0.0
        
        # Compute goal completion and reward
        if self.env_name == "arm_reach":
            # Goal occupancy: is the end of the arm close enough to the goal?
            eef_pos = obs[self.completion_goal_indices]
            goal_info = state.info["goal"]
            goal_eef_pos = goal_info[:3]
            dist = jnp.linalg.norm(eef_pos - goal_eef_pos)
            
            reward = jnp.array(dist < 0.1, dtype=float)
            success = reward
            success_easy = jnp.array(dist < 0.25, dtype=float)
            success_hard = jnp.array(dist < 0.03, dtype=float)
        elif self.env_name == "arm_grasp":
            # Goal occupancy: is the midpoint of the fingers close enough to the cube, and is the gripper closed enough?
            # Technically, only success_hard is properly gripping the cube, but success/success_easy are for signs of life.
            goal_info = state.info["goal"]
            cube_pos = obs[:3]
            left_finger_pos = obs[32:35]
            right_finger_pos = obs[35:38]
            finger_midpoint = (left_finger_pos + right_finger_pos) / 2
            cube_to_finger_midpoint_dist = jnp.linalg.norm(cube_pos - finger_midpoint)

            gripper_openness = obs[38]
            goal_gripper_openness = goal_info[9]
            gripper_openness_difference = jnp.linalg.norm(gripper_openness - goal_gripper_openness)

            reward = jnp.array(
                jnp.all(jnp.array([
                    cube_to_finger_midpoint_dist < 0.05,
                    gripper_openness_difference < 0.02
                ])), 
                dtype=float
            )
            success = reward
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
        elif self.env_name in {"arm_pickplace_easy", "arm_pickplace_hard", "arm_binpick"}:
            # Goal occupancy: is the cube close enough to the goal? 
            current_cube_pos = obs[self.completion_goal_indices]
            goal_info = state.info["goal"]
            goal_pos = goal_info[:3]
            dist = jnp.linalg.norm(current_cube_pos - goal_pos)
            
            reward = jnp.array(dist < 0.1, dtype=float)
            success = reward
            success_easy = jnp.array(dist < 0.15, dtype=float)
            success_hard = jnp.array(dist < 0.03, dtype=float)
        else:
            raise Exception(f"Goal completion metric not implemented for environment: {self.env_name}.")
        
        # Fill in state variables
        info = {**state.info, "timestep": timestep}
        state.metrics.update(success=success, success_easy=success_easy, success_hard=success_hard)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
    
    """
    We deviate from the existing interface by allowing the goal to be modified directly,
    since this is useful for exploration during trajectory collection.
    
    Additionally, the goal is now virtual, and is no longer a physical object in simulation. While 
    a physical representation better for visualization of positional goals, many goals are not only
    positions. It could include orientation, velocity, timestep, gripper open-ness, etc. which cannot be 
    represented naturally with an object.
    """
    def update_goal(self, state: State, goal: jax.Array) -> State:
        updated_info = {**state.info, "goal": goal} # Dictionary unpacking to return updated dict
        return state.replace(info=updated_info)
    
    """
    Converts the [-1, 1] actions to the corresponding target angle or gripper strength.
    We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.
    """
    def convert_action(self, action: jax.Array) -> jax.Array:
        if self.env_name == "arm_reach":
            min_value = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            max_value = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        else:
            # Flip the gripper action values to make -1 open, 1 closed.
            min_value = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 255, 255])
            max_value = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0, 0])
        
        # This offset and multiplier yields f(-1) = min_value, f(1) = max_value
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2
        converted_action = offset + action * multiplier 
        return converted_action

    """
    Observation spaces and goal spaces are defined as follows, per environment.
    
    Reach
    - Observation (28-dim)
      - q (7-dim): joint angles
      - qd (7-dim): joint angular velocities
      - t (1-dim): normalized timestep
      - End of arm (13-dim): position/angle/velocity/angular velocity
    - Goal (3-dim): position of end of arm
    
    Grasp
    - Observation (39-dim)
      - q (16-dim): 7-dim cube position/angle, 7-dim joint angles, 2-dim finger offset
      - qd (15-dim): 6-dim cube velocity/angular velocity, 7-dim joint angular velocities, 2-dim finger offset rate of change
      - t (1-dim): normalized timestep
      - Fingers (7-dim): 6-dim positions of fingers, 1-dim distance between fingers
    - Goal (10-dim): position of cube, position of fingers, distance between fingers
    
    Pick-Place Easy/Hard, Binpick:
    - Observation (46-dim)
      - q (16-dim): 7-dim cube position/angle, 7-dim joint angles, 2-dim finger offset
      - qd (15-dim): 6-dim cube velocity/angular velocity, 7-dim joint angular velocities, 2-dim finger offset rate of change
      - t (1-dim): normalized timestep
      - End-effector (13-dim): position/angle/velocity/angular velocity
      - Fingers (1-dim): finger distance
    - Goal (7-dim): position of cube, position of end-effector, distance between fingers
    """
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        q = pipeline_state.q
        qd = pipeline_state.qd
        t = jnp.array([timestep])
        
        if self.env_name == "arm_reach":
            eef_index = 6 # Links 1-7 are indices 0-6. The end-effector (eef) base is merged with link 7, so we say link 7 index = eef index.
            eef_x_pos = pipeline_state.x.pos[eef_index]
            eef_x_rot = pipeline_state.x.rot[eef_index]
            eef_xd_vel = pipeline_state.xd.vel[eef_index]
            eef_xd_angvel = pipeline_state.xd.ang[eef_index]
            return jnp.concatenate([q] + [qd] + [t] + [eef_x_pos] + [eef_x_rot] + [eef_xd_vel] + [eef_xd_angvel] + [goal])
        
        elif self.env_name == "arm_grasp":
            # Cube is 0, then links 1-7 are indices 1-7. Fingers are 8 and 9.
            left_finger_index = 8
            left_finger_x_pos = pipeline_state.x.pos[left_finger_index]
            right_finger_index = 9
            right_finger_x_pos = pipeline_state.x.pos[right_finger_index]
            finger_distance = jnp.linalg.norm(right_finger_x_pos - left_finger_x_pos)[None] # [None] expands dims from 0 to 1
            return jnp.concatenate([q] + [qd] + [t] + [left_finger_x_pos] + [right_finger_x_pos] + [finger_distance] + [goal])
        
        elif self.env_name in {"arm_pickplace_easy", "arm_pickplace_hard", "arm_binpick"}: 
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
        
        else:
            raise Exception(f"Observation retrieval not implemented for environment: {self.env_name}.")