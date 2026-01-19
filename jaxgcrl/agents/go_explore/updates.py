"""Training update interfaces for Go-Explore framework."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class PolicyUpdate(ABC):
    """Abstract base class for policy update functions."""
    
    @abstractmethod
    def update(
        self,
        policy_params,
        transitions,
        value_params: Any = None,
        key: Any = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """Update policy parameters.
        
        Args:
            policy_params: Current policy parameters
            transitions: Batch of transitions
            value_params: Optional value function parameters (for actor-critic)
            key: Optional random key
            **kwargs: Additional arguments (networks, context, etc.)
            
        Returns:
            Tuple of (new_policy_params, metrics_dict)
        """
        pass


@dataclass
class ValueUpdate(ABC):
    """Abstract base class for value function update functions."""
    
    @abstractmethod
    def update(
        self,
        value_params,
        transitions,
        policy_params: Any = None,
        key: Any = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, jnp.ndarray]]:
        """Update value function parameters.
        
        Args:
            value_params: Current value function parameters
            transitions: Batch of transitions
            policy_params: Optional policy parameters (for actor-critic)
            key: Optional random key
            **kwargs: Additional arguments (networks, context, etc.)
            
        Returns:
            Tuple of (new_value_params, metrics_dict)
        """
        pass
