"""Algorithm interface for CRL training loop.

This module defines the abstract base class that algorithms must implement
to be used with CRL's unified training loop.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.agents.crl.crl import TrainingState, Transition


@dataclass
class Algorithm(ABC):
    """Abstract base class for algorithms that can be plugged into CRL.
    
    Algorithms implement this interface to define:
    1. How to roll out goal-conditioned policy deterministically
    2. How to roll out exploratory policy with noise
    3. How to update goal-conditioned policy parameters
    4. How to update exploratory policy parameters
    
    CRL will call these methods in sequence during training.
    """
    
    @abstractmethod
    def rollout_goal_conditioned_deterministic(
        self,
        training_state: TrainingState,
        env_state: Any,
        main_buffer_state: Any,
        goal_conditioned_buffer_state: Any,
        key: jax.random.PRNGKey,
        env: Any,
        main_replay_buffer: Any,
        goal_conditioned_replay_buffer: Any,
        propose_goals_fn: Any,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """Roll out goal-conditioned policy with no noise (deterministic).
        
        Args:
            training_state: Current training state
            env_state: Current environment state
            main_buffer_state: Main replay buffer state (for combined trajectories)
            goal_conditioned_buffer_state: Goal-conditioned policy replay buffer state
            key: Random key
            env: Environment
            main_replay_buffer: Main replay buffer
            goal_conditioned_replay_buffer: Goal-conditioned policy replay buffer
            propose_goals_fn: Function to propose goals for new episodes
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (new_env_state, new_main_buffer_state, new_goal_conditioned_buffer_state, goal_conditioned_data)
            where goal_conditioned_data is the transitions collected (for combining with exploratory trajectory)
        """
        pass
    
    @abstractmethod
    def rollout_exploratory(
        self,
        training_state: TrainingState,
        env_state: Any,
        main_buffer_state: Any,
        exploratory_buffer_state: Any,
        key: jax.random.PRNGKey,
        env: Any,
        main_replay_buffer: Any,
        exploratory_replay_buffer: Any,
        propose_goals_fn: Any,
        **kwargs
    ) -> Tuple[Any, Any, Any]:
        """Roll out exploratory policy with noise.
        
        Args:
            training_state: Current training state
            env_state: Current environment state
            main_buffer_state: Main replay buffer state (for combined trajectories)
            exploratory_buffer_state: Exploratory policy replay buffer state
            key: Random key
            env: Environment
            main_replay_buffer: Main replay buffer
            exploratory_replay_buffer: Exploratory policy replay buffer
            propose_goals_fn: Function to propose goals for new episodes
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (new_env_state, new_main_buffer_state, new_exploratory_buffer_state)
        """
        pass
    
    @abstractmethod
    def update_goal_conditioned_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: Dict[str, Any],
        context: Dict[str, Any],
        key: jax.random.PRNGKey,
        **kwargs
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        """Update goal-conditioned policy parameters (and any additional networks).
        
        Args:
            training_state: Current training state
            transitions: Batch of transitions
            networks: Dictionary of networks (actor, sa_encoder, g_encoder, etc.)
            context: Dictionary of context variables (config, sizes, etc.)
            key: Random key
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (new_training_state, metrics_dict)
        """
        pass
    
    @abstractmethod
    def update_exploratory_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: Dict[str, Any],
        context: Dict[str, Any],
        key: jax.random.PRNGKey,
        **kwargs
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
        """Update exploratory policy parameters (and any additional networks).
        
        Args:
            training_state: Current training state
            transitions: Batch of transitions
            networks: Dictionary of networks (actor, sa_encoder, g_encoder, etc.)
            context: Dictionary of context variables (config, sizes, etc.)
            key: Random key
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (new_training_state, metrics_dict)
        """
        pass
    
    def initialize_additional_states(
        self,
        key: jax.random.PRNGKey,
        **kwargs
    ) -> Dict[str, TrainState]:
        """Initialize any additional network states needed by this algorithm.
        
        Args:
            key: Random key
            **kwargs: Additional arguments (networks, config, etc.)
            
        Returns:
            Dictionary of additional TrainState objects
        """
        return {}
    
    def get_transitions_for_goal_conditioned_update(
        self,
        transitions: Transition,
        **kwargs
    ) -> Transition:
        """Filter/process transitions for goal-conditioned policy update.
        
        By default, returns all transitions. Algorithms can override to filter
        or process transitions differently.
        
        Args:
            transitions: All transitions
            **kwargs: Additional arguments
            
        Returns:
            Filtered/processed transitions for goal-conditioned update
        """
        return transitions
    
    def get_transitions_for_exploratory_update(
        self,
        transitions: Transition,
        **kwargs
    ) -> Transition:
        """Filter/process transitions for exploratory policy update.
        
        By default, returns all transitions. Algorithms can override to filter
        or process transitions differently.
        
        Args:
            transitions: All transitions
            **kwargs: Additional arguments
            
        Returns:
            Filtered/processed transitions for exploratory update
        """
        return transitions
