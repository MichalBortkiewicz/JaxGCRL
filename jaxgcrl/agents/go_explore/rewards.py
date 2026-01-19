"""Reward function interfaces for Go-Explore framework."""
from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class RewardFunction(ABC):
    """Abstract base class for reward functions.
    
    Reward functions compute rewards for transitions, which may be different
    from the environment reward (e.g., intrinsic rewards for exploration).
    """
    
    @abstractmethod
    def compute_reward(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        next_obs: jnp.ndarray,
        env_reward: jnp.ndarray,
        done: jnp.ndarray,
        info: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Compute reward for a transition.
        
        Args:
            obs: Current observations (batch_size, obs_dim)
            action: Actions taken (batch_size, action_dim)
            next_obs: Next observations (batch_size, obs_dim)
            env_reward: Environment reward (batch_size,)
            done: Done flags (batch_size,)
            info: Optional info dict from environment
            
        Returns:
            Rewards (batch_size,)
        """
        pass
