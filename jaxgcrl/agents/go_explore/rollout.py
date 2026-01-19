"""Rollout strategies for Go-Explore framework."""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from jaxgcrl.agents.go_explore.policies import GoalConditionedPolicy, ExploratoryPolicy
from jaxgcrl.agents.go_explore.rewards import RewardFunction


class Transition(NamedTuple):
    """Container for a transition - matches CRL's Transition structure."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: dict


@dataclass
class RolloutStrategy(ABC):
    """Abstract base class for rollout strategies."""
    
    @abstractmethod
    def rollout_step(
        self,
        env_state,
        policy_params,
        key: jax.random.PRNGKey,
        step_idx: int,
        **kwargs
    ) -> Tuple:
        """Perform a single rollout step.
        
        Args:
            env_state: Current environment state
            policy_params: Policy parameters
            key: Random key
            step_idx: Current step index in the rollout
            **kwargs: Additional arguments (policy, env, etc.)
            
        Returns:
            Tuple of (new_env_state, transition)
        """
        pass


@dataclass
class GoExploreRollout(RolloutStrategy):
    """Go-Explore style rollout: deterministic goal-conditioned steps first, then exploratory.
    
    This rollout strategy:
    1. Uses goal-conditioned policy deterministically for first N steps
    2. Uses exploratory policy (with noise) for remaining steps
    3. Applies reward function to compute rewards
    """
    num_deterministic_steps: int  # Number of deterministic goal-conditioned steps
    goal_conditioned_policy: GoalConditionedPolicy
    exploratory_policy: ExploratoryPolicy
    reward_function: RewardFunction
    
    def rollout_step(
        self,
        env_state,
        goal_conditioned_policy_params,
        exploratory_policy_params,
        proposed_goals: jnp.ndarray,
        was_proposed_goal_mask: jnp.ndarray,
        key: jax.random.PRNGKey,
        step_idx: int,
        env,
        extra_fields: Tuple = (),
    ) -> Tuple:
        """Perform a single rollout step.
        
        Args:
            env_state: Current environment state
            goal_conditioned_policy_params: Goal-conditioned policy parameters
            exploratory_policy_params: Exploratory policy parameters
            proposed_goals: Proposed goals (batch_size, goal_dim)
            was_proposed_goal_mask: Mask indicating which envs have proposed goals (batch_size,)
            key: Random key
            step_idx: Current step index (0-indexed)
            env: Environment
            extra_fields: Extra fields to extract from env info
            
        Returns:
            Tuple of (new_env_state, transition)
        """
        batch_size = env_state.obs.shape[0]
        step_keys = jax.random.split(key, batch_size)
        
        # Determine which policy to use based on step index
        use_goal_conditioned = step_idx < self.num_deterministic_steps
        
        if use_goal_conditioned:
            # Use goal-conditioned policy deterministically
            # Update observations with proposed goals
            goal_indices = env.goal_indices
            goal_size = len(goal_indices)
            new_obs = env_state.obs.at[:, -goal_size:].set(proposed_goals)
            env_state = env_state.replace(obs=new_obs)
            
            # Sample actions deterministically (no noise)
            actions = self.goal_conditioned_policy.sample_action(
                goal_conditioned_policy_params,
                new_obs,
                step_keys[0],  # Key not used for deterministic
                deterministic=True
            )
        else:
            # Use exploratory policy with noise
            if self.exploratory_policy.is_goal_conditioned():
                # If exploratory policy is goal-conditioned, update obs with goals
                goal_indices = env.goal_indices
                goal_size = len(goal_indices)
                new_obs = env_state.obs.at[:, -goal_size:].set(proposed_goals)
                env_state = env_state.replace(obs=new_obs)
            else:
                # Exploratory policy doesn't use goals, keep original obs
                new_obs = env_state.obs
            
            # Sample actions with noise
            actions = self.exploratory_policy.sample_action(
                exploratory_policy_params,
                new_obs,
                step_keys[0],
                deterministic=False
            )
        
        # Step environment
        nstate = env.step(env_state, actions)
        
        # Extract extra fields
        state_extras = {x: nstate.info[x] for x in extra_fields}
        state_extras["was_proposed_goal_mask"] = was_proposed_goal_mask
        state_extras["use_goal_conditioned"] = jnp.full((batch_size,), use_goal_conditioned, dtype=jnp.bool_)
        
        # Compute rewards using reward function
        rewards = self.reward_function.compute_reward(
            obs=env_state.obs,
            action=actions,
            next_obs=nstate.obs,
            env_reward=nstate.reward,
            done=nstate.done,
            info=nstate.info
        )
        
        # Create transition
        transition = Transition(
            observation=env_state.obs,
            action=actions,
            reward=rewards,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )
        
        return nstate, transition
