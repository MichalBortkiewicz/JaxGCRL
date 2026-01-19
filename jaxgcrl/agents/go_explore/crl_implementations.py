"""Concrete implementations of Go-Explore components for CRL."""
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.agents.go_explore.policies import GoalConditionedPolicy, ExploratoryPolicy, SameAsGoalConditionedPolicy
from jaxgcrl.agents.go_explore.rewards import RewardFunction
from jaxgcrl.agents.go_explore.rollout import GoExploreRollout
from jaxgcrl.agents.crl.networks import Actor


@dataclass
class CRLGoalConditionedPolicy(GoalConditionedPolicy):
    """Goal-conditioned policy using CRL's Actor network."""
    actor: Actor
    
    def apply(self, params, obs: jnp.ndarray, rng=None):
        """Apply the policy to observations.
        
        Returns:
            Tuple of (means, log_stds)
        """
        return self.actor.apply(params, obs)
    
    def sample_action(self, params, obs: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False) -> jnp.ndarray:
        """Sample an action from the policy."""
        means, log_stds = self.actor.apply(params, obs)
        
        if deterministic:
            actions = jnp.tanh(means)
        else:
            stds = jnp.exp(log_stds)
            actions = jnp.tanh(means + stds * jax.random.normal(rng, shape=means.shape, dtype=means.dtype))
        
        return actions


@dataclass
class CRLExploratoryPolicy(ExploratoryPolicy):
    """Exploratory policy using CRL's Actor network.
    
    This can be the same as the goal-conditioned policy or a separate policy.
    """
    actor: Actor
    is_goal_conditioned_flag: bool = True  # Whether this policy uses goals
    
    def apply(self, params, obs: jnp.ndarray, rng=None):
        """Apply the policy to observations.
        
        Returns:
            Tuple of (means, log_stds)
        """
        return self.actor.apply(params, obs)
    
    def sample_action(self, params, obs: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False) -> jnp.ndarray:
        """Sample an action from the policy."""
        means, log_stds = self.actor.apply(params, obs)
        
        if deterministic:
            actions = jnp.tanh(means)
        else:
            stds = jnp.exp(log_stds)
            actions = jnp.tanh(means + stds * jax.random.normal(rng, shape=means.shape, dtype=means.dtype))
        
        return actions
    
    def is_goal_conditioned(self) -> bool:
        return self.is_goal_conditioned_flag
