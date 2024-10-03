from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp


# Observation space: 39 dims for the actual observation, 10 dims for goal position. See _get_obs() for details.
# Action space:      9 dimensional, [-1, 1] each. The first 7 correspond to joint target angles (each joint has a different range, see
#                    convert_action). The last 2 correspond to gripper finger positions.
class ArmGrasp(PipelineEnv):
    def __init__(self, backend="mjx", **kwargs):
        # Load XML and manage simulation parameters
        if backend == "mjx":
            sys = mjcf.load("envs/assets/panda_grasp_mjx.xml")
            sys = sys.tree_replace({
                "opt.timestep": 0.01,
                "opt.iterations": 4,
                "opt.ls_iterations": 8,
            })
            self.n_frames = 4
            self.episode_length = 100 # Merely a recommendation + used for postexplore timestep
        elif backend == "positional":
            sys = mjcf.load("envs/assets/panda_grasp_positional.xml")
            sys = sys.tree_replace({
                "opt.timestep": 0.0003,
                "opt.iterations": 50,
                "opt.ls_iterations": 100,
            })
            self.n_frames = 10
            self.episode_length = 1000 # Merely a recommendation + used for postexplore timestep
        else:
            raise Exception("Please use mjx or positional backends for better speed/stability tradeoffs.")
            
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)
        
        # Set additional configuration information
        self.goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # For data collection/goal-conditioning: cube position, left and right finger positions, 
                                                                             # and gripper finger distance
        self.completion_goal_indices = jnp.array([0, 1, 2, 32, 33, 34, 35, 36, 37, 38]) # For reward/checking completion of goal: identical to goal indices
        self.obs_dim = 39
        
        self.cube_noise_scale = 0.3
        self.arm_noise_scale = 1
        
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        # Variance in starting arm state and cube/goal state to avoid overfitting/closed loop control
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        cube_q_xy = self.sys.init_q[:2] + self.cube_noise_scale * jax.random.uniform(subkey1, [2])
        cube_q_remaining = self.sys.init_q[2:7]
        arm_q = self.sys.init_q[7:] + self.arm_noise_scale * jax.random.uniform(subkey2, [self.sys.q_size() - 7])
        q = jnp.concatenate([cube_q_xy] + [cube_q_remaining] + [arm_q])
        qd = jnp.zeros([self.sys.qd_size()])

        # Initialize state
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0

        # Define goal distribution. Note that we have no better way than to specify the finger goal positions and
        # the cube goal positions, but the actual requirement for goal completion is looser (finger midpoint near cube center + gripper is closed enough)
        cube_goal_pos = q[:3]
        left_finger_goal_pos = cube_goal_pos + jnp.array([0, 0.03, 0])
        right_finger_goal_pos = cube_goal_pos + jnp.array([0, -0.03, 0])
        gripper_openness_goal = jnp.array([0.06]) # The cube itself is 0.06 wide
        goal = jnp.concatenate([cube_goal_pos] + [left_finger_goal_pos] + [right_finger_goal_pos] + [gripper_openness_goal])
                              
        # Get other components for state (obs, reward, etc.)
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        
        # Fill info
        rng, subkey = jax.random.split(rng)
        info = {
            "seed": 0, # Seed is required, but fill it with a dummy value
            "goal": goal, 
            "timestep": 0.0, 
            "postexplore_timestep": self.episode_length * jax.random.uniform(subkey)
        } 
        
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, self.convert_action(action))
        timestep = state.info["timestep"] + 1
        
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
        done = 0.0
        
        # We measure goal occupancy by checking whether the midpoint of the fingers is close to the cube, and whether the gripper
        # is closed enough. We don't directly use the completion goal indices for the goal (other than goal_gripper_openness), but 
        # just use it to indicate to users what parts of the current observation are being used to assess goal completion.
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
                cube_to_finger_midpoint_dist < 0.01,
                gripper_openness_difference < 0.005
            ])), 
            dtype=float
        )
        success_easy = jnp.array(
            jnp.all(jnp.array([
                cube_to_finger_midpoint_dist < 0.03,
                gripper_openness_difference < 0.015
            ])), 
            dtype=float
        )
        info = {**state.info, "timestep": timestep}
        
        state.metrics.update(success=reward, success_easy=success_easy)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
    
    # We deviate from the existing interface by allowing the goal to be modified directly,
    # since this is useful for exploration during trajectory collection.
    #
    # Additionally, the goal is now virtual, and is no longer a physical object in simulation. While 
    # a physical representation better for visualization of positional goals, many goals are not only
    # positions. It could include orientation, velocity, timestep, gripper open-ness, etc. which cannot be 
    # represented naturally with an object.
    def update_goal(self, state: State, goal: jax.Array) -> State:
        updated_info = {**info, "goal": goal} # Dictionary unpacking to return updated dict
        return state.replace(info=updated_info)
    
    # Converts the [-1, 1] actions to the corresponding target angle or strength.
    # We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.
    def convert_action(self, action: jax.Array) -> jax.Array:
        # Flip the gripper action values to make -1 open, 1 closed.
        min_value = jnp.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 255, 255])
        max_value = jnp.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0, 0])
        
        # This offset and multiplier yields f(-1) = min_value, f(1) = max_value
        offset = (min_value + max_value) / 2
        multiplier = (max_value - min_value) / 2
        converted_action = offset + action * multiplier 
        return converted_action


    # NOTE: This deviates from the usual convention by also taking the goal as input, since we're storing
    # this outside of the pipeline state.
    #
    # 39 dimensional observation space (NOT including goal -- hence, obs_dim = 39, but the returned observation has dim 39 + goal dims):
    # - 16 for q
    #   - 7 for cube: 3 pos + 4 quaternion
    #   - 9 joint angles/lengths, with joints being links 1 through 7 + 2 fingers
    # - 15 for qd
    #   - 6 for cube: 3 velocity + 3 angular velocity
    #   - 9 joint angular/linear velocities
    # - 1 for timestep
    # - 6 for positions of fingers
    # - 1 for distance between left and right fingers (measures gripper open/closedness)
    #
    # Goal space: position of cube, position of fingers, finger distance.
    # - This is just for guiding the goal-conditioned policy and Q-critic; we evaluate task completion using only cube position.
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
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