"""Algorithm interface for CRL training loop.

This module defines the abstract base class that algorithms must implement
to be used with CRL's unified training loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState

if TYPE_CHECKING:
    # Import types only for type-checking to avoid circular runtime imports.
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
        networks: Dict[str, Any],
        context: Dict[str, Any],
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
        num_exploratory_steps: int,
        networks: Dict[str, Any],
        context: Dict[str, Any],
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
            num_exploratory_steps: Number of exploratory steps to take
            networks: Dictionary of networks (actor, sa_encoder, g_encoder, etc.)
            context: Dictionary of context variables (config, sizes, etc.)
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
    
    def sample_exploratory_transitions(
        self,
        exploratory_replay_buffer: Any,
        exploratory_buffer_state: Any,
        sampling_key: Any,
        batch_size: int,
        discounting: float,
        state_size: int,
        goal_indices: Any,
        flatten_batch_fn: Any,
        **kwargs
    ) -> Tuple[Optional[Transition], Any]:
        """Sample transitions from exploratory buffer for training.
        
        By default, returns (None, buffer_state), which means use main buffer transitions.
        Algorithms can override to sample from exploratory buffer instead.
        
        Args:
            exploratory_replay_buffer: Exploratory replay buffer
            exploratory_buffer_state: Exploratory buffer state
            sampling_key: Random key for sampling
            batch_size: Batch size
            discounting: Discount factor
            state_size: State dimension
            goal_indices: Goal indices
            flatten_batch_fn: Function to flatten batch
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (transitions, new_buffer_state)
            - transitions: Transitions from exploratory buffer, or None to use main buffer
            - new_buffer_state: Updated buffer state
        """
        return None, exploratory_buffer_state
    
    @abstractmethod
    def propose_goals(
        self,
        env_state: Any,
        training_state: TrainingState,
        main_buffer_state: Any,
        key: jax.random.PRNGKey,
        env: Any,
        main_replay_buffer: Any,
        networks: Dict[str, Any],
        context: Dict[str, Any],
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any]:
        """Propose goals for new episodes.
        
        Args:
            env_state: Current environment state
            training_state: Current training state
            main_buffer_state: Main replay buffer state
            key: Random key
            env: Environment
            main_replay_buffer: Main replay buffer
            networks: Dictionary of networks (actor, sa_encoder, g_encoder, etc.)
            context: Dictionary of context variables (config, sizes, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (proposed_goals, was_proposed_goal_mask, new_main_buffer_state)
            - proposed_goals: (batch_size, goal_dim) array of proposed goals
            - was_proposed_goal_mask: (batch_size,) boolean mask indicating which envs have proposed goals
            - new_main_buffer_state: Updated buffer state (may be same if no sampling occurred)
        """
        pass