from typing import Tuple
from brax import base
from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mpe import simple

# Changes made:
# All obs and actions are typed as arrays instead of dicts when sent/received
# This invariant exists to maintain training script convention
# We convert to dicts in this env class

# TODO: metrics update in state info if want to

NUM_AGENTS = 2

class SimpleMPEMARL(Env):
    def __init__(self):
        self.env = simple.SimpleMPE(num_agents = NUM_AGENTS, action_type="Continuous")

    def get_obs(self, state, obs, targets):
        def create_obs(ob, target):
            abs_pos = state.p_pos[self.env.num_agents] - ob[2:] #subtract rel pos from landmark pos
            return jp.concatenate((ob[:2], abs_pos, target), axis=0)
          
        return jp.stack(list(jax.tree.map(create_obs, obs, targets).values())) #convert dict to jp array

    @property
    def observation_size(self) -> int:
        return 6 #vel, pos, target_pos

    @property
    def action_size(self) -> int:
        return 5 #from simple mpe documentation

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")
        return "There is no backend!!!"

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        key1, key2, key3 = jax.random.split(rng, 3)
        
        # Set the targets
        targets = self._random_target(key1)

        # Reset the internal MPE environment
        obs, state = self.env.reset(key2)

        reward, done, zero = jp.zeros(3)
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
        info = {"seed": 0, "mpe_key": key3, "mpe_target": targets}

        state = State(state, self.get_obs(state, obs, targets), reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # Generate action dict
        actions = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        
        # Step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # Process obs into our format
        obs = self.get_obs(mpe_state, obs, state.info["mpe_target"])
        
        # Set trajectory id to differentiate between episodes
        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "mpe_key": key}
        state.info.update(info)
        
        reward, _ = jp.zeros(2)
        return state.replace(
            pipeline_state=mpe_state, obs=obs, reward=reward, done=list(dones.values())[0].astype(jp.float32)
        )
    
    # Generate uniformly random target for each agent
    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        keys = jax.random.split(rng, self.env.num_agents)
        dist = 1
        ang = jp.pi * 2.0 * jax.vmap(jax.random.uniform)(keys)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        target_arr = jp.concatenate((target_x[:,None], target_y[:,None]), axis=1)
        return {a: target_arr[i] for i, a in enumerate(self.env.agents)}