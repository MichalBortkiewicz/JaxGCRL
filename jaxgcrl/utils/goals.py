"""Shared goal proposal utilities for goal-conditioned RL agents.

This module provides base classes and simple goal proposers that can be used
across different agents (CRL, TD3, etc.).
"""
from abc import ABC, abstractmethod
from typing import Literal

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class GoalProposer(ABC):
    """Abstract base class for goal proposal algorithms."""
    
    @abstractmethod
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, **kwargs):
        """Goal proposal algorithm. Returns a (batch_size, goal_size) array of proposed goals.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            env: Training environment (must have goal_indices and state_dim attributes)
            env_state: Current environment state (contains current observations)
            key: JAX random key
            **kwargs: Additional arguments for specific proposers (e.g., actor, critic params)
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state
        """
        pass


@dataclass
class ReplayBufferGoalProposal(GoalProposer):
    """Proposes goals by sampling final states from replay buffer trajectories."""
    
    def propose_goals(self, replay_buffer, buffer_state, env, env_state, key, **kwargs):
        buffer_state, sampled_transitions = replay_buffer.sample(buffer_state)
        traj_ids = sampled_transitions.extras["state_extras"]["traj_id"]  # (num_envs, episode_length)
        observations = sampled_transitions.observation  # (num_envs, episode_length, obs_size)
        
        def get_last_state(obs_seq, traj_id_seq):
            """Get the last state for each trajectory"""
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        # Extract last states for each batch element
        last_states = jax.vmap(get_last_state)(observations, traj_ids)  # (batch_size, state_size)
        # Extract goal positions from these last states
        proposed_goals = last_states[:, env.goal_indices]  # (batch_size, goal_size)
        
        return proposed_goals, buffer_state


def create_goal_proposer(
    proposer_name: Literal["replay_buffer"],
) -> GoalProposer:
    """Factory function to create a goal proposer by name.
    
    Args:
        proposer_name: Name of the goal proposer to create.
            - "replay_buffer": Sample final states from trajectories
            
    Returns:
        GoalProposer instance
    """
    if proposer_name == "replay_buffer":
        return ReplayBufferGoalProposal()
    else:
        raise ValueError(f"Unknown goal proposer: {proposer_name}. "
                        f"Available: replay_buffer")


def mix_goals(
    original_goals: jnp.ndarray,
    proposed_goals: jnp.ndarray,
    goal_proposal_prob: float,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Mix original goals with proposed goals based on probability.
    
    Args:
        original_goals: (batch_size, goal_dim) array of original environment goals
        proposed_goals: (batch_size, goal_dim) array of proposed goals
        goal_proposal_prob: Probability of using proposed goal for each sample
        key: JAX random key
        
    Returns:
        mixed_goals: (batch_size, goal_dim) array of mixed goals
        use_proposed_mask: (batch_size, 1) boolean array indicating which goals were proposed
    """
    use_proposed_mask = jax.random.bernoulli(key, goal_proposal_prob, shape=(proposed_goals.shape[0], 1))
    mixed_goals = jnp.where(use_proposed_mask, proposed_goals, original_goals)
    return mixed_goals, use_proposed_mask
