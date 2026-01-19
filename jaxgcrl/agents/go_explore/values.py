"""Value function interfaces for Go-Explore framework."""
from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class ValueFunction(ABC):
    """Abstract base class for value functions V(s) or V(s, g).
    
    Value functions estimate the expected return from a state (and optionally goal).
    """
    
    @abstractmethod
    def apply(self, params, obs: jnp.ndarray) -> jnp.ndarray:
        """Compute value for observations.
        
        Args:
            params: Value function parameters
            obs: Observations (batch_size, obs_dim)
            
        Returns:
            Values (batch_size,)
        """
        pass


@dataclass
class QFunction(ABC):
    """Abstract base class for Q-functions Q(s, a) or Q(s, a, g).
    
    Q-functions estimate the expected return from a state-action pair (and optionally goal).
    """
    
    @abstractmethod
    def apply(self, params, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Compute Q-value for state-action pairs.
        
        Args:
            params: Q-function parameters
            obs: Observations (batch_size, obs_dim)
            action: Actions (batch_size, action_dim)
            
        Returns:
            Q-values (batch_size,)
        """
        pass
