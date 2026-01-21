"""Goal proposer interfaces and implementations for CRL algorithms."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from jaxgcrl.agents.crl.types import TrainingState
from jaxgcrl.agents.crl.goals import mix_goals
from jaxgcrl.agents.crl.goals_utils import get_final_states_from_batch


@dataclass
class GoalProposer(ABC):
    """Abstract base class for goal proposers.
    
    Goal proposers generate goals for new episodes based on various strategies
    (e.g., sampling from replay buffers, using learned models, etc.).
    """
    
    goal_proposal_prob: float = 1.0  # Probability of using proposed goals
    
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
        source_replay_buffer: Any,
        source_buffer_state: Any,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any, Any]:
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
            source_replay_buffer: The replay buffer to sample goals from
            source_buffer_state: The state of the source replay buffer
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (proposed_goals, was_proposed_goal_mask, main_buffer_state, source_buffer_state)
            - proposed_goals: (batch_size, goal_dim) array of proposed goals
            - was_proposed_goal_mask: (batch_size,) boolean mask indicating which envs have proposed goals
            - main_buffer_state: Updated main buffer state (may be same if no sampling occurred)
            - source_buffer_state: Updated source buffer state
        """
        pass


@dataclass
class FinalReplayBufferProposer(GoalProposer):
    """Goal proposer that samples final states from trajectories in a replay buffer.
    
    This proposer samples trajectories from the source replay buffer and extracts
    the final states (goals) from each trajectory, then randomly selects one
    goal per environment.
    """
    
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
        source_replay_buffer: Any,
        source_buffer_state: Any,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Any, Any]:
        """Propose goals from final states of trajectories in the source replay buffer."""
        proposal_key, mix_key = jax.random.split(key)
        
        # Compare current traj_id with stored traj_id to detect resets
        current_traj_id = env_state.info["traj_id"]
        stored_traj_id = env_state.info.get("last_traj_id", current_traj_id - 1)
        is_new_episode = current_traj_id != stored_traj_id  # shape (num_envs,)
        
        # Sample trajectories from the source replay buffer
        source_buffer_state, candidate_transitions = source_replay_buffer.sample(
            source_buffer_state
        )
        
        # Extract final states from sampled trajectories
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation
        goal_indices = context["goal_indices"]
        if isinstance(goal_indices, (list, tuple)):
            goal_indices = jnp.array(goal_indices)
        
        # Get final states (goals) from each trajectory
        final_goals = get_final_states_from_batch(candidate_obs, traj_ids, goal_indices)
        
        # Sample one goal per environment from the candidate goals
        batch_size = env_state.obs.shape[0]
        num_candidates = final_goals.shape[0]
        indices = jax.random.randint(proposal_key, (batch_size,), 0, num_candidates)
        new_goals = final_goals[indices]  # (batch_size, goal_dim)
        
        # Get original goals from environment
        goal_size = goal_indices.shape[0] if hasattr(goal_indices, 'shape') else len(goal_indices)
        original_goals = env_state.obs[:, -goal_size:]
        
        # Mix goals based on goal_proposal_prob
        mixed_goals, use_proposed_mask = mix_goals(original_goals, new_goals, self.goal_proposal_prob, mix_key)
        
        # Only update goals for environments that are starting a new episode
        proposed_goals = jnp.where(
            is_new_episode[:, None],
            mixed_goals,
            env_state.info.get("proposed_goals", original_goals)
        )
        was_proposed_goal_mask = jnp.where(
            is_new_episode,
            use_proposed_mask.squeeze(-1),
            env_state.info.get("was_proposed_goal_mask", jnp.zeros_like(use_proposed_mask.squeeze(-1)))
        )
        
        return proposed_goals, was_proposed_goal_mask, main_buffer_state, source_buffer_state
