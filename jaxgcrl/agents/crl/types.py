"""Shared types for CRL agent.

This module contains types that are shared across multiple CRL modules to avoid
circular import dependencies.
"""
from typing import NamedTuple, Any
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState


@dataclass
class TrainingState:
    """Contains training state for the learner"""
    optimal_goal_proposal_prob: jnp.ndarray
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState
    # Additional networks for experimental methods (e.g., exploratory policy, value functions)
    # Use a frozen dict or empty dict for JAX compatibility
    additional_states: Any = None  # Dict[str, TrainState] for extensibility


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()
