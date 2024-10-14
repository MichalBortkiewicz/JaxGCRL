from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import mujoco
import jax
from jax import numpy as jnp

class ArmEnvs(PipelineEnv):
    def __init__(self, backend="mjx", **kwargs):
        # Configure environment information (e.g. env name, noise scale, observation dimension, goal indices) and load XML
        self._set_environment_attributes()
        xml_path = self._get_xml_path()
        sys = mjcf.load(xml_path)
        
        # Configure backend
        sys = sys.tree_replace({
            "opt.timestep": 0.01,
            "opt.iterations": 4,
            "opt.ls_iterations": 8,
        })
        self.n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", self.n_frames)
        
        # Initialize brax PipelineEnv
        if backend != "mjx":
            raise Exception("Use the mjx backend for stability/reasonable speed.")
        super().__init__(sys=sys, backend=backend, **kwargs)
            
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        
        # Initialize simulator state
        rng, subkey = jax.random.split(rng)
        q, qd = self._get_initial_state(subkey) # Injects noise to avoid overfitting/open loop control
        pipeline_state = self.pipeline_init(q, qd)
        timestep = 0.0
        
        # Sample a goal and fill info variable
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        goal = self._get_initial_goal(subkey1)
        info = {
            "seed": 0, # Seed is required, but fill it with a dummy value
            "goal": goal, 
            "timestep": 0.0, 
            "postexplore_timestep": jax.random.uniform(subkey2) # Assumes timestep is normalized between 0 and 1
        }
            
        # Get other components for state (observation, reward, metrics)
        obs = self._get_obs(pipeline_state, goal, timestep)
        reward, done, zero = jnp.zeros(3)
        metrics = {"success": zero, "success_easy": zero, "success_hard": zero}
        
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        
        # Run mujoco step
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, self._convert_action_to_actuator_input(action))
        
        # Compute variables for state update, including observation and goal/reward
        timestep = state.info["timestep"] + 1 / self.episode_length
        obs = self._get_obs(pipeline_state, state.info["goal"], timestep)
        
        success, success_easy, success_hard = self._compute_goal_completion(obs, state.info["goal"])
        state.metrics.update(success=success, success_easy=success_easy, success_hard=success_hard)
        
        reward = success
        done = 0.0
        info = {**state.info, "timestep": timestep}
        
        new_state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=info)
        return new_state
    
    def update_goal(self, state: State, goal: jax.Array) -> State:
        """
        We deviate from the existing interface by allowing the goal to be modified directly,
        since this is useful for exploration during trajectory collection.

        Additionally, the goal is now virtual, and is no longer a physical object in simulation. While 
        a physical representation is better for visualization of positional goals, many goals are not only
        positions. It could include orientation, velocity, timestep, gripper open-ness, etc. which cannot be 
        represented naturally with an object.
        """
        
        updated_info = {**state.info, "goal": goal} # Dictionary unpacking to return updated dict
        return state.replace(info=updated_info)
    
    def _convert_action_to_actuator_input(self, action: jax.Array) -> jax.Array:
        """
        Converts the [-1, 1] actions to the corresponding target angle or gripper strength.
        We use the exact numbers for radians specified in the XML, even if they might be cleaner in terms of pi.
        """
            
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
    
    # Methods to be overridden by specific environments
    def _get_xml_path(self):
        raise NotImplementedError

    def _set_environment_attributes(self):
        """
        Attribute descriptions:
        - Episode lengths are merely a recommendation + used for scaling timestep
        - Goal indices are for data collection/goal-conditioning; completion goal indices are for reward/goal completion checking
        - State_dim is for the observation dimension WITHOUT the goal appended; brax's env.observation_size gives the dimension WITH goal appended
        - Noise scale is for the range of uniform-distribution noise
            - Arm: perturb all joints (in radians) and gripper position (in meters)
            - Cube: perturb x, y of starting cube location
            - Goal: perturb x, y of cube destination or x, y, z of reach location
        """
        raise NotImplementedError
        
    def _get_initial_state(self, rng):
        raise NotImplementedError
        
    def _get_initial_goal(self, rng):
        raise NotImplementedError
        
    def _compute_goal_completion(self, obs, goal):
        raise NotImplementedError
        
    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        raise NotImplementedError