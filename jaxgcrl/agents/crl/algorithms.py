"""Default algorithm implementations for CRL."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from jaxgcrl.agents.crl.algorithm import Algorithm
from jaxgcrl.agents.crl.losses import update_actor_and_alpha, update_critic
from jaxgcrl.agents.crl.goals import ReplayBufferGoalProposal, mix_goals
from jaxgcrl.agents.crl.goals_utils import get_final_states_from_batch

if TYPE_CHECKING:
    # Import types only for type-checking to avoid circular runtime imports.
    from jaxgcrl.agents.crl.crl import TrainingState, Transition
else:
    # Runtime imports - use string annotations to avoid circular import
    TrainingState = Any
    Transition = Any


@dataclass
class DefaultCRLAlgorithm(Algorithm):
    """Default CRL algorithm implementation.
    
    This implements the standard CRL training loop:
    - Standard rollout with noise
    - Standard CRL updates for both goal-conditioned and exploratory policies
    - Simple replay buffer goal proposal
    """
    
    num_deterministic_steps: int = 0
    goal_proposal_prob: float = 0.0
    goal_proposal_warmup_steps: int = 0
    use_adaptive_mixing: bool = False
    adaptive_mixing_warmup_steps: int = 0
    interpolate_to_env_goals: bool = False
    
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
    ):
        """Propose goals for new episodes using replay buffer goal proposal."""
        proposal_key, mix_key = jax.random.split(key)
        
        # Compare current traj_id with stored traj_id to detect resets
        current_traj_id = env_state.info["traj_id"]
        stored_traj_id = env_state.info.get("last_traj_id", current_traj_id - 1)
        is_new_episode = current_traj_id != stored_traj_id  # shape (num_envs,)
        
        # Use simple replay buffer goal proposal
        goal_proposer = ReplayBufferGoalProposal()
        new_goals, main_buffer_state = goal_proposer.propose_goals(
            main_replay_buffer, main_buffer_state,
            training_state, env, env_state,
            proposal_key,
            networks["actor"], training_state.actor_state.params, training_state.critic_state.params,
            networks["sa_encoder"], networks["g_encoder"]
        )
        goal_indices = context["goal_indices"]
        if isinstance(goal_indices, (list, tuple)):
            goal_size = len(goal_indices)
        else:
            goal_size = goal_indices.shape[0] if hasattr(goal_indices, 'shape') else len(goal_indices)
        original_goals = env_state.obs[:, -goal_size:]
        
        # Compute mixing probability
        if self.use_adaptive_mixing:
            curr_goal_proposal_prob = jax.lax.cond(
                training_state.env_steps >= self.adaptive_mixing_warmup_steps,
                lambda: training_state.optimal_goal_proposal_prob,
                lambda: 0.5,
            )
        elif self.interpolate_to_env_goals:
            progress_frac = training_state.env_steps / context.get("total_env_steps", 1e6)
            curr_goal_proposal_prob = self.goal_proposal_prob * (1 - progress_frac)
        else:
            curr_goal_proposal_prob = self.goal_proposal_prob

        # Mix goals for new episodes
        mixed_goals, use_proposed_mask = mix_goals(original_goals, new_goals, curr_goal_proposal_prob, mix_key)

        # Apply warmup: only use proposed goals after warmup period
        should_use_proposed = training_state.env_steps >= self.goal_proposal_warmup_steps
        new_proposed_goals = jax.lax.cond(
            should_use_proposed,
            lambda: mixed_goals,
            lambda: original_goals,
        )
        new_was_proposed_mask = jax.lax.cond(
            should_use_proposed,
            lambda: use_proposed_mask.squeeze(-1),
            lambda: jnp.zeros_like(use_proposed_mask.squeeze(-1)),
        )

        # Only update goals for environments that are starting a new episode
        # Keep existing goals for environments mid-episode
        proposed_goals = jnp.where(
            is_new_episode[:, None],
            new_proposed_goals,
            env_state.info.get("proposed_goals", original_goals)
        )
        was_proposed_goal_mask = jnp.where(
            is_new_episode,
            new_was_proposed_mask,
            env_state.info.get("was_proposed_goal_mask", jnp.zeros_like(new_was_proposed_mask))
        )

        return proposed_goals, was_proposed_goal_mask, main_buffer_state
    
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
        actor_step_fn: Any,
        **kwargs
    ):
        """Roll out goal-conditioned policy deterministically.
        
        Inserts transitions into goal_conditioned_buffer only.
        Main buffer will be updated after combining with exploratory trajectory.
        """
        if self.num_deterministic_steps == 0:
            return env_state, main_buffer_state, goal_conditioned_buffer_state, None
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals for new episodes using algorithm's goal proposal method
            proposed_goals, was_proposed_goal_mask, main_buffer_state = self.propose_goals(
                env_state=env_state,
                training_state=training_state,
                main_buffer_state=main_buffer_state,
                key=proposal_key,
                env=env,
                main_replay_buffer=main_replay_buffer,
                networks=networks,
                context=context,
            )
            
            # Store traj_id before step
            pre_step_traj_id = env_state.info["traj_id"]
            
            # Update env_state.info with goals
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            
            # Deterministic actor step
            env_state, transition = actor_step_fn(
                training_state.actor_state,
                env,
                env_state,
                proposed_goals,
                was_proposed_goal_mask,
                current_key,
                extra_fields=("truncation", "traj_id"),
                deterministic=True,
            )
            
            # Preserve info
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            env_state.info["last_traj_id"] = pre_step_traj_id
            
            return (env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, next_key), transition
        
        (env_state, _, main_buffer_state, goal_conditioned_buffer_state, _), data = jax.lax.scan(
            f, (env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, key), (), length=self.num_deterministic_steps
        )
        # Insert into goal-conditioned buffer only
        goal_conditioned_buffer_state = goal_conditioned_replay_buffer.insert(goal_conditioned_buffer_state, data)
        return env_state, main_buffer_state, goal_conditioned_buffer_state, data
    
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
        networks: Dict[str, Any],
        context: Dict[str, Any],
        actor_step_fn: Any,
        num_exploratory_steps: int,
        goal_conditioned_data: Any = None,
        **kwargs
    ):
        """Roll out exploratory policy with noise.
        
        Inserts transitions into exploratory_buffer only.
        After this, combines deterministic + exploratory trajectories and inserts into main buffer.
        """
        pass
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, main_buffer_state, exploratory_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals for new episodes using algorithm's goal proposal method
            proposed_goals, was_proposed_goal_mask, main_buffer_state = self.propose_goals(
                env_state=env_state,
                training_state=training_state,
                main_buffer_state=main_buffer_state,
                key=proposal_key,
                env=env,
                main_replay_buffer=main_replay_buffer,
                networks=networks,
                context=context,
            )
            
            # Store traj_id before step
            pre_step_traj_id = env_state.info["traj_id"]
            
            # Update env_state.info with goals
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            
            # Exploratory actor step (with noise)
            env_state, transition = actor_step_fn(
                training_state.actor_state,
                env,
                env_state,
                proposed_goals,
                was_proposed_goal_mask,
                current_key,
                extra_fields=("truncation", "traj_id"),
                deterministic=False,
            )
            
            # Preserve info
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            env_state.info["last_traj_id"] = pre_step_traj_id
            
            return (env_state, training_state, main_buffer_state, exploratory_buffer_state, next_key), transition
        
        (env_state, _, main_buffer_state, exploratory_buffer_state, _), exploratory_data = jax.lax.scan(
            f, (env_state, training_state, main_buffer_state, exploratory_buffer_state, key), (), length=num_exploratory_steps
        )
        # Insert into exploratory buffer only
        exploratory_buffer_state = exploratory_replay_buffer.insert(exploratory_buffer_state, exploratory_data)
        
        # Combine deterministic + exploratory trajectories and insert into main buffer as single trajectory
        if goal_conditioned_data is not None and self.num_deterministic_steps > 0:
            # Concatenate along time dimension (axis 0)
            combined_data = jax.tree_util.tree_map(
                lambda det, exp: jnp.concatenate([det, exp], axis=0),
                goal_conditioned_data,
                exploratory_data
            )
        elif goal_conditioned_data is None and self.num_deterministic_steps == 0:
            # Only exploratory part
            combined_data = exploratory_data
        else:
            # Only deterministic part (shouldn't happen if num_exploratory > 0, but handle it)
            combined_data = goal_conditioned_data
        
        # Insert combined trajectory into main buffer
        main_buffer_state = main_replay_buffer.insert(main_buffer_state, combined_data)
        
        return env_state, main_buffer_state, exploratory_buffer_state
    
    def update_goal_conditioned_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: dict,
        context: dict,
        key: jax.random.PRNGKey,
        **kwargs
    ):
        """Update goal-conditioned policy using standard CRL update."""
        key, critic_key, actor_key = jax.random.split(key, 3)
        
        # Standard CRL update: update actor (with alpha) and critic
        training_state, actor_metrics = update_actor_and_alpha(
            context, networks, transitions, training_state, actor_key
        )
        training_state, critic_metrics = update_critic(
            context, networks, transitions, training_state, critic_key
        )
        
        # Combine metrics (no gradient_steps increment here - handled in CRL wrapper)
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        # Prefix metrics
        prefixed_metrics = {f"goal_conditioned/{k}": v for k, v in metrics.items()}
        
        return training_state, prefixed_metrics
    
    def update_exploratory_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: dict,
        context: dict,
        key: jax.random.PRNGKey,
        **kwargs
    ):
        """Update exploratory policy using standard CRL update."""
        key, critic_key, actor_key = jax.random.split(key, 3)
        
        training_state, actor_metrics = update_actor_and_alpha(
            context, networks, transitions, training_state, actor_key
        )
        training_state, critic_metrics = update_critic(
            context, networks, transitions, training_state, critic_key
        )
        
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        # Prefix metrics
        prefixed_metrics = {f"exploratory/{k}": v for k, v in metrics.items()}
        
        return training_state, prefixed_metrics


@dataclass
class DualCRLAlgorithm(Algorithm):
    """Dual CRL algorithm implementation.
    
    This algorithm:
    - Proposes goals from final states of goal-conditioned policy's replay buffer
    - Both policies are goal-conditioned
    - Goal-conditioned policy trains on main (combined) buffer
    - Exploratory policy trains on its own buffer only
    """
    
    num_deterministic_steps: int = 0
    goal_proposal_prob: float = 1.0  # Probability of using proposed goals (typically 1.0 for dual_crl)
    
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
        goal_conditioned_replay_buffer: Any = None,
        goal_conditioned_buffer_state: Any = None,
        **kwargs
    ):
        """Propose goals from final states of goal-conditioned policy's replay buffer."""
        proposal_key, mix_key = jax.random.split(key)
        
        # Compare current traj_id with stored traj_id to detect resets
        current_traj_id = env_state.info["traj_id"]
        stored_traj_id = env_state.info.get("last_traj_id", current_traj_id - 1)
        is_new_episode = current_traj_id != stored_traj_id  # shape (num_envs,)
        
        # Sample trajectories from goal-conditioned replay buffer
        goal_conditioned_buffer_state, candidate_transitions = goal_conditioned_replay_buffer.sample(
            goal_conditioned_buffer_state
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
        
        # Mix goals (but with goal_proposal_prob = 1.0, so always use proposed)
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
        
        return proposed_goals, was_proposed_goal_mask, main_buffer_state
    
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
        actor_step_fn: Any,
        **kwargs
    ):
        """Roll out goal-conditioned policy deterministically."""
        if self.num_deterministic_steps == 0:
            return env_state, main_buffer_state, goal_conditioned_buffer_state, None
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals from goal-conditioned buffer
            proposed_goals, was_proposed_goal_mask, main_buffer_state = self.propose_goals(
                env_state=env_state,
                training_state=training_state,
                main_buffer_state=main_buffer_state,
                key=proposal_key,
                env=env,
                main_replay_buffer=main_replay_buffer,
                networks=networks,
                context=context,
                goal_conditioned_replay_buffer=goal_conditioned_replay_buffer,
                goal_conditioned_buffer_state=goal_conditioned_buffer_state,
            )
            
            # Store traj_id before step
            pre_step_traj_id = env_state.info["traj_id"]
            
            # Update env_state.info with goals
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            
            # Deterministic actor step (goal-conditioned)
            env_state, transition = actor_step_fn(
                training_state.actor_state,
                env,
                env_state,
                proposed_goals,
                was_proposed_goal_mask,
                current_key,
                extra_fields=("truncation", "traj_id"),
                deterministic=True,
            )
            
            # Preserve info
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            env_state.info["last_traj_id"] = pre_step_traj_id
            
            return (env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, next_key), transition
        
        (env_state, _, main_buffer_state, goal_conditioned_buffer_state, _), data = jax.lax.scan(
            f, (env_state, training_state, main_buffer_state, goal_conditioned_buffer_state, key), (), length=self.num_deterministic_steps
        )
        # Insert into goal-conditioned buffer only
        goal_conditioned_buffer_state = goal_conditioned_replay_buffer.insert(goal_conditioned_buffer_state, data)
        return env_state, main_buffer_state, goal_conditioned_buffer_state, data
    
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
        networks: Dict[str, Any],
        context: Dict[str, Any],
        actor_step_fn: Any,
        num_exploratory_steps: int,
        goal_conditioned_data: Any = None,
        goal_conditioned_replay_buffer: Any = None,
        goal_conditioned_buffer_state: Any = None,
        **kwargs
    ):
        """Roll out exploratory policy with noise (also goal-conditioned)."""
        
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, main_buffer_state, exploratory_buffer_state, goal_conditioned_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals from goal-conditioned buffer (same as deterministic rollout)
            proposed_goals, was_proposed_goal_mask, main_buffer_state = self.propose_goals(
                env_state=env_state,
                training_state=training_state,
                main_buffer_state=main_buffer_state,
                key=proposal_key,
                env=env,
                main_replay_buffer=main_replay_buffer,
                networks=networks,
                context=context,
                goal_conditioned_replay_buffer=goal_conditioned_replay_buffer,
                goal_conditioned_buffer_state=goal_conditioned_buffer_state,
            )
            
            # Store traj_id before step
            pre_step_traj_id = env_state.info["traj_id"]
            
            # Update env_state.info with goals
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            
            # Exploratory actor step (with noise, but still goal-conditioned)
            env_state, transition = actor_step_fn(
                training_state.actor_state,
                env,
                env_state,
                proposed_goals,
                was_proposed_goal_mask,
                current_key,
                extra_fields=("truncation", "traj_id"),
                deterministic=False,
            )
            
            # Preserve info
            env_state.info["proposed_goals"] = proposed_goals
            env_state.info["was_proposed_goal_mask"] = was_proposed_goal_mask
            env_state.info["last_traj_id"] = pre_step_traj_id
            
            return (env_state, training_state, main_buffer_state, exploratory_buffer_state, goal_conditioned_buffer_state, next_key), transition
        
        (env_state, _, main_buffer_state, exploratory_buffer_state, goal_conditioned_buffer_state, _), exploratory_data = jax.lax.scan(
            f, (env_state, training_state, main_buffer_state, exploratory_buffer_state, goal_conditioned_buffer_state, key), (), length=num_exploratory_steps
        )
        # Insert into exploratory buffer only
        exploratory_buffer_state = exploratory_replay_buffer.insert(exploratory_buffer_state, exploratory_data)
        
        # Combine deterministic + exploratory trajectories and insert into main buffer
        if goal_conditioned_data is not None and self.num_deterministic_steps > 0:
            combined_data = jax.tree_util.tree_map(
                lambda det, exp: jnp.concatenate([det, exp], axis=0),
                goal_conditioned_data,
                exploratory_data
            )
        elif goal_conditioned_data is None and self.num_deterministic_steps == 0:
            combined_data = exploratory_data
        else:
            combined_data = goal_conditioned_data
        
        # Insert combined trajectory into main buffer
        main_buffer_state = main_replay_buffer.insert(main_buffer_state, combined_data)
        
        return env_state, main_buffer_state, exploratory_buffer_state
    
    def update_goal_conditioned_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: dict,
        context: dict,
        key: jax.random.PRNGKey,
        **kwargs
    ):
        """Update goal-conditioned policy using standard CRL update on main buffer."""
        key, critic_key, actor_key = jax.random.split(key, 3)
        
        # Standard CRL update: update actor (with alpha) and critic
        training_state, actor_metrics = update_actor_and_alpha(
            context, networks, transitions, training_state, actor_key
        )
        training_state, critic_metrics = update_critic(
            context, networks, transitions, training_state, critic_key
        )
        
        # Combine metrics
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        # Prefix metrics
        prefixed_metrics = {f"goal_conditioned/{k}": v for k, v in metrics.items()}
        
        return training_state, prefixed_metrics
    
    def update_exploratory_policy(
        self,
        training_state: TrainingState,
        transitions: Transition,
        networks: dict,
        context: dict,
        key: jax.random.PRNGKey,
        **kwargs
    ):
        """Update exploratory policy using standard CRL update on exploratory buffer."""
        key, critic_key, actor_key = jax.random.split(key, 3)
        
        # Standard CRL update: update actor (with alpha) and critic
        training_state, actor_metrics = update_actor_and_alpha(
            context, networks, transitions, training_state, actor_key
        )
        training_state, critic_metrics = update_critic(
            context, networks, transitions, training_state, critic_key
        )
        
        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        
        # Prefix metrics
        prefixed_metrics = {f"exploratory/{k}": v for k, v in metrics.items()}
        
        return training_state, prefixed_metrics
    
    def sample_exploratory_transitions(
        self,
        exploratory_replay_buffer: Any,
        exploratory_buffer_state: Any,
        sampling_key: jax.random.PRNGKey,
        batch_size: int,
        discounting: float,
        state_size: int,
        goal_indices: Any,
        flatten_batch_fn: Any,
        **kwargs
    ) -> Tuple[Transition, Any]:
        """Sample transitions from exploratory buffer for training."""
        # Sample from exploratory buffer
        exploratory_buffer_state, transitions = exploratory_replay_buffer.sample(exploratory_buffer_state)
        
        # Process transitions same way as main buffer
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_batch_fn, in_axes=(None, 0, 0))(
            (discounting, state_size, tuple(goal_indices)),
            transitions,
            batch_keys,
        )
        
        # Flatten and permute
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
        )
        permutation = jax.random.permutation(sampling_key, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )
        
        return transitions, exploratory_buffer_state
