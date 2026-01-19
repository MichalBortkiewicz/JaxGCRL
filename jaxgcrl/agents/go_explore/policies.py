"""Policy interfaces for Go-Explore framework."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class GoalConditionedPolicy(ABC):
    """Abstract base class for goal-conditioned policies.
    
    A goal-conditioned policy takes observations (which include goals) and outputs actions.
    """
    
    @abstractmethod
    def apply(self, params, obs: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the policy to observations.
        
        Args:
            params: Policy parameters
            obs: Observations with shape (batch_size, obs_dim) where obs includes goal
            rng: Optional random key for stochastic policies
            
        Returns:
            Tuple of (means, log_stds) or (actions, log_probs) depending on policy type
        """
        pass
    
    @abstractmethod
    def sample_action(self, params, obs: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False) -> jnp.ndarray:
        """Sample an action from the policy.
        
        Args:
            params: Policy parameters
            obs: Observations with shape (batch_size, obs_dim)
            rng: Random key
            deterministic: If True, return mean action (no noise)
            
        Returns:
            Actions with shape (batch_size, action_dim)
        """
        pass


@dataclass
class ExploratoryPolicy(ABC):
    """Abstract base class for exploratory policies.
    
    An exploratory policy may or may not be goal-conditioned. It's used for exploration
    after the deterministic goal-conditioned rollout phase.
    """
    
    @abstractmethod
    def apply(self, params, obs: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply the policy to observations.
        
        Args:
            params: Policy parameters
            obs: Observations with shape (batch_size, obs_dim)
            rng: Optional random key for stochastic policies
            
        Returns:
            Tuple of (means, log_stds) or (actions, log_probs) depending on policy type
        """
        pass
    
    @abstractmethod
    def sample_action(self, params, obs: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False) -> jnp.ndarray:
        """Sample an action from the policy.
        
        Args:
            params: Policy parameters
            obs: Observations with shape (batch_size, obs_dim)
            rng: Random key
            deterministic: If True, return mean action (no noise)
            
        Returns:
            Actions with shape (batch_size, action_dim)
        """
        pass
    
    @abstractmethod
    def is_goal_conditioned(self) -> bool:
        """Whether this exploratory policy is goal-conditioned.
        
        Returns:
            True if the policy uses goals from observations, False otherwise
        """
        pass


@dataclass
class SameAsGoalConditionedPolicy(ExploratoryPolicy):
    """Exploratory policy that is the same as the goal-conditioned policy.
    
    This is a convenience class for when you want to use the same policy for both
    goal-conditioned and exploratory phases, but with different noise levels.
    """
    goal_conditioned_policy: GoalConditionedPolicy
    
    def apply(self, params, obs: jnp.ndarray, rng: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.goal_conditioned_policy.apply(params, obs, rng)
    
    def sample_action(self, params, obs: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False) -> jnp.ndarray:
        return self.goal_conditioned_policy.sample_action(params, obs, rng, deterministic)
    
    def is_goal_conditioned(self) -> bool:
        return True
