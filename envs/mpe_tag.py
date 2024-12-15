from typing import Tuple
from brax import base
from brax.envs.base import State, Env
import jax
from jax import numpy as jp
from jaxmarl.environments.mpe import simple_tag

NUM_GOOD = 1
NUM_ADV = 3
NUM_OBS = 10

class MPETagCoop(Env):
    def __init__(self):
        self.env = simple_tag.SimpleTagMPE(
            num_good_agents=NUM_GOOD,
            num_adversaries=NUM_ADV,
            num_obs=NUM_OBS,
            action_type="Continuous")

    def get_obs(self, state, obs):
        adv_obs = jp.array([obs[a] for a in self.env.adversaries])
        good_obs = jp.array([obs[a] for a in self.env.good_agents])
        adv_pos, good_pos = adv_obs[:,2:4], good_obs[:,2:4]
        dist_mat = (adv_pos[:,None,:] - good_pos[None,:,:])**2
        dist_mat = jp.sum(dist_mat, axis=2)
        min_dist = jp.min(dist_mat)
        
        add_obs = jp.zeros((adv_obs.shape[0], 2))
        add_obs = add_obs.at[:,0].set(min_dist)
        return jp.concatenate((adv_obs, add_obs), axis=1)
                
    @property
    def observation_size(self) -> int:
        return 4 + 2*NUM_OBS + 2*(self.env.num_agents-1) + 2*(NUM_GOOD) + 2 # pos, vel, landmarks, other_pos, other_vel, dist, goal

    @property
    def action_size(self) -> int:
        return 5 # From simple mpe documentation

    @property
    def backend(self) -> str:
        raise Exception("This environment does not use a brax backend")
        return "There is no backend!!!"

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        key1, key2, key3 = jax.random.split(rng, 3)
        
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
        info = {"seed": 0, "mpe_key": key3, "mpe_target": 0}

        state = State(state, self.get_obs(state, obs), reward, done, metrics)
        state.info.update(info)
        return state
        
    def good_agents_policy(self, state):
        adv_pos = jp.array([state.p_pos[self.env.a_to_i[a]] for a in self.env.adversaries])
        good_pos = jp.array([state.p_pos[self.env.a_to_i[a]] for a in self.env.good_agents])
        vecs = good_pos[:,None,:] - adv_pos[None,:,:]
        vecs = vecs / (jp.sum(vecs**2, axis=2)**(3/2))[:,:,None]
        move_dir = jp.sum(vecs, axis=1)
        
        actions = jp.zeros((move_dir.shape[0], 5))
        actions = actions.at[:,[2,4]].set(move_dir)
        actions = actions.at[:,[1,3]].set(-1 * move_dir)
        actions = jp.where(actions < 0, 0, actions)
        good_actions = {a: actions[i] for i, a in enumerate(self.env.good_agents)}
        return good_actions

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        key, key_s = jax.random.split(state.info["mpe_key"], 2)
        
        # Generate action dict for adversaries and good agents
        actions = {agent: action[i] for i, agent in enumerate(self.env.adversaries)}
        actions.update(self.good_agents_policy(state.pipeline_state))
        
        # Step internal environment
        obs, mpe_state, rewards, dones, infos = self.env.step(key_s, state.pipeline_state, actions)

        # Process obs into our format
        obs = self.get_obs(mpe_state, obs)
        
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