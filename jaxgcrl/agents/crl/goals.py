from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from flax.struct import dataclass

@dataclass
class GoalProposer(ABC):
    @abstractmethod
    def propose_goals(self, replay_buffer, buffer_state, train_env, key):
        '''Goal proposal algorithm. This should return a (batch_size, goal_size) array of proposed goals. In the current setup, this should not modify env_state or training_state'''
        pass

@dataclass
class ReplayBufferGoalProposal(GoalProposer):
    def propose_goals(self, replay_buffer, buffer_state, train_env, key):
        buffer_state, sampled_transitions = replay_buffer.sample(buffer_state)
        
        traj_ids = sampled_transitions.extras["state_extras"]["traj_id"]  # (num_envs, episode_length)
        observations = sampled_transitions.observation  # (num_envs, episode_length, obs_size)
        
        def get_last_state(obs_seq, traj_id_seq):
            """Get the last state for each trajectory"""
            # Find the last index where this trajectory appears
            seq_len = obs_seq.shape[0]
            # Assuming the trajectory runs through the sequence, find its last occurrence
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        # Extract last states for each batch element
        last_states = jax.vmap(get_last_state)(observations, traj_ids)  # (batch_size, state_size)
        
        # Extract goal positions from these last states
        proposed_goals = last_states[:, train_env.goal_indices]  # (batch_size, goal_size)
        
        return proposed_goals, buffer_state