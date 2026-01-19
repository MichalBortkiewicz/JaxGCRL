"""Default algorithm implementations for CRL."""
from typing import Any
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from jaxgcrl.agents.crl.algorithm import Algorithm
from jaxgcrl.agents.crl.crl import TrainingState, Transition
from jaxgcrl.agents.crl.losses import update_actor_and_alpha, update_critic


@dataclass
class DefaultCRLAlgorithm(Algorithm):
    """Default CRL algorithm implementation.
    
    This implements the standard CRL training loop:
    - Standard rollout with noise
    - Standard CRL updates for both goal-conditioned and exploratory policies
    """
    
    num_deterministic_steps: int = 0
    num_exploratory_steps: int = 0
    
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
        actor_step_fn: Any,
        **kwargs
    ):
        """Roll out goal-conditioned policy deterministically.
        
        Inserts transitions into goal_conditioned_buffer only.
        Main buffer will be updated after combining with exploratory trajectory.
        """
        if self.num_deterministic_steps == 0:
            return env_state, main_buffer_state, goal_conditioned_buffer_state
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, goal_conditioned_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals for new episodes (using main buffer for goal proposal)
            proposed_goals, was_proposed_goal_mask, _ = propose_goals_fn(
                env_state, training_state, main_buffer_state, proposal_key
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
            
            return (env_state, training_state, goal_conditioned_buffer_state, next_key), transition
        
        (env_state, _, goal_conditioned_buffer_state, _), data = jax.lax.scan(
            f, (env_state, training_state, goal_conditioned_buffer_state, key), (), length=self.num_deterministic_steps
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
        propose_goals_fn: Any,
        actor_step_fn: Any,
        unroll_length: int,
        goal_conditioned_data: Any = None,
        **kwargs
    ):
        """Roll out exploratory policy with noise.
        
        Inserts transitions into exploratory_buffer only.
        After this, combines deterministic + exploratory trajectories and inserts into main buffer.
        """
        num_exploratory = self.num_exploratory_steps if self.num_exploratory_steps > 0 else (unroll_length - self.num_deterministic_steps)
        
        if num_exploratory == 0:
            # Still need to insert combined trajectory into main buffer if we have deterministic data
            if goal_conditioned_data is not None and self.num_deterministic_steps > 0:
                # Combine trajectories and insert into main buffer
                combined_data = goal_conditioned_data  # Only deterministic part
                main_buffer_state = main_replay_buffer.insert(main_buffer_state, combined_data)
            return env_state, main_buffer_state, exploratory_buffer_state
        
        @jax.jit
        def f(carry, unused_t):
            env_state, training_state, exploratory_buffer_state, current_key = carry
            current_key, next_key, proposal_key = jax.random.split(current_key, 3)
            
            # Propose goals for new episodes (using main buffer for goal proposal)
            proposed_goals, was_proposed_goal_mask, _ = propose_goals_fn(
                env_state, training_state, main_buffer_state, proposal_key
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
            
            return (env_state, training_state, exploratory_buffer_state, next_key), transition
        
        (env_state, _, exploratory_buffer_state, _), exploratory_data = jax.lax.scan(
            f, (env_state, training_state, exploratory_buffer_state, key), (), length=num_exploratory
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
        
        training_state, actor_metrics = update_actor_and_alpha(
            context, networks, transitions, training_state, actor_key
        )
        training_state, critic_metrics = update_critic(
            context, networks, transitions, training_state, critic_key
        )
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
        
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
