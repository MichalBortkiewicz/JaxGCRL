"""CRL-specific goal proposal algorithms.

This module extends the base goal proposers from jaxgcrl.utils.goals with
CRL-specific proposers that use contrastive learning networks.
"""
import io

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.struct import dataclass
from PIL import Image

from jaxgcrl.agents.crl.losses import energy_fn
from jaxgcrl.agents.crl.goals_utils import (
    get_last_state_from_trajectory,
    get_final_states_from_batch,
    sample_random_states_from_batch,
    expand_goal_to_state,
    zero_out_non_goal_indices,
    compute_q_value_single_critic,
    stack_ensemble_params,
    compute_q_values_ensemble,
    compute_v_and_sigma_ensemble,
    compute_energy_for_state_goal_pairs,
    gaussian_kernel_density,
    estimate_log_density_knn,
    compute_kl_divergence_empirical,
    create_2x2_scatter_plot,
    create_energy_histogram_plot,
    should_log_at_interval,
    create_goal_selection_plot,
    create_env_goal_ranking_plot,
)
# Import base classes and utilities from shared module
from jaxgcrl.utils.goals import (
    GoalProposer,
    ReplayBufferGoalProposal as BaseReplayBufferGoalProposal,
    mix_goals,
)

# Re-export for convenience
__all__ = ['GoalProposer', 'ReplayBufferGoalProposal', 'FisherTraceGoalProposal', 
           'MediumEnergyGoalProposal', 'MetricPreservationGoalProposal', 'QEpistemicGoalProposal', 
           'DISCOVERGoalProposal', 'mix_goals']



"""
UCGR (Unsupervised Contrastive Goal-Reaching) Goal Proposer

Implementation following Algorithm 1 from the paper:
"Unsupervised Contrastive Goal-Reaching" by Turkman, Ghugare, and Eysenbach (2025)

The key innovation is the MinLSE (Minimum LogSumExp) goal selection strategy:
    S(g) = log Σ_{i=1}^K exp(f(s_i, a_i, g))
    g* = argmin_{g ∈ G_cand} S(g)

where f(s, a, g) is the critic function from contrastive RL.
"""

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from typing import Any


def flatten_trajectory_data(observations, actions, traj_ids, state_size):
    """Flatten 3D trajectory data to 2D for batch processing.
    
    Args:
        observations: (N, K, obs_dim) where N = num trajectory samples, K = episode_length
        actions: (N, K, action_dim)
        traj_ids: (N, K)
        state_size: size of state portion of observation
        
    Returns:
        states: (N * K, state_size) flattened states
        actions: (N * K, action_dim) flattened actions
    """
    N, K = observations.shape[:2]
    states = observations[:, :, :state_size].reshape(-1, state_size)
    actions_flat = actions.reshape(-1, actions.shape[-1])
    return states, actions_flat

@dataclass
class DISCOVERGoalProposal(GoalProposer):
    """DISCOVER goal proposal algorithm.
    
    For each starting state, selects the candidate goal g that maximizes:
    alpha_t(V(s0, g) + sigma(s0, g)) + (1 - alpha_t)(V(g, g*) + sigma(g, g*))
    
    Where:
    - V(s, g) is the mean value estimate across the ensemble
    - sigma(s, g) is the standard deviation of value estimates across the ensemble
    - s0 is the start state
    - g is a candidate goal
    - g* is an environment goal (averaged over all env goals)
    - alpha_t is updated based on goal achievement proportion p
    """
    energy_fn_name: str
    num_ensemble: int = 5  # Number of critics in the ensemble
    alpha_0: float = 0.5  # Initial alpha value
    target_prob: float = 0.5  # Target goal achievement probability p
    LOG_INTERVAL_STEPS: int = 1000000  # Log visualizations every N environment steps

    def propose_goals(
        self, 
        replay_buffer, 
        buffer_state, 
        training_state, 
        train_env, 
        env_state, 
        key,
        actor, 
        actor_params, 
        critic_params, 
        sa_encoder, 
        g_encoder
    ):
        """Propose goals using DISCOVER algorithm.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            training_state: Current training state
            train_env: Training environment
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (contains ensemble of sa_encoder and g_encoder params)
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state
        """
        assert hasattr(train_env, 'possible_goals'), \
            "Environment must store property `possible_goals` for DISCOVERGoalProposal."
        
        # Get current states from env_state
        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        batch_size = current_states.shape[0]
        goal_indices = train_env.goal_indices
        
        # Get environment goals
        env_goals = train_env.possible_goals  # (num_env_goals, goal_dim)
        num_env_goals = env_goals.shape[0]
        
        # Sample candidate goals from replay buffer - one random state per trajectory
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation
        
        # Sample one random state from each trajectory
        key, sample_key = jax.random.split(key)
        # Get full states (not just goal portions) for computing V(g, g*)
        def sample_random_full_state_from_trajectory(obs_seq, traj_id_seq, rng_key):
            """Sample a random full state from a trajectory."""
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            num_valid = jnp.sum(mask.astype(jnp.int32))
            random_idx = jax.random.randint(rng_key, (), 0, num_valid)
            sorted_indices = jnp.argsort(-mask.astype(jnp.int32))
            sampled_idx = sorted_indices[random_idx]
            return obs_seq[sampled_idx]
        
        num_trajs = candidate_obs.shape[0]
        sample_keys = jax.random.split(sample_key, num_trajs)
        candidate_states_full = jax.vmap(sample_random_full_state_from_trajectory)(
            candidate_obs, traj_ids, sample_keys
        )  # (num_candidates, obs_dim)
        
        # Extract goal portions for candidate selection
        candidate_goals = candidate_states_full[:, goal_indices]  # (num_candidates, goal_dim)
        # Extract full state portions (state_dim, not obs_dim which includes goal)
        candidate_states = candidate_states_full[:, :state_size]  # (num_candidates, state_dim)
        num_candidates = candidate_goals.shape[0]
        
        # Check if we have an ensemble
        is_ensemble = isinstance(critic_params["sa_encoder"], list)
        if not is_ensemble:
            raise ValueError("DISCOVERGoalProposal requires an ensemble of critics. Set use_critic_ensemble=True.")
        
        # Stack ensemble parameters into arrays for JAX-compatible indexing
        stacked_sa_params, stacked_g_params = stack_ensemble_params(critic_params)
        
        # Compute scores for all (state, candidate_goal) pairs
        # First compute the components separately, then combine with alpha_t
        def compute_score_components_for_state(state):
            """Compute score components for all candidate goals given a starting state."""
            def compute_components_for_candidate(candidate_state, candidate_goal):
                # Compute V(s0, g) and sigma(s0, g) using full state and goal
                v_s0g, sigma_s0g = compute_v_and_sigma_ensemble(
                    state, candidate_goal, actor, actor_params,
                    stacked_sa_params, stacked_g_params, sa_encoder, g_encoder,
                    self.energy_fn_name, is_goal_as_state=False
                )
                
                # For each environment goal g*, compute V(g, g*) and sigma(g, g*)
                # Use the full candidate_state (not expanded from goal)
                def compute_for_env_goal(env_goal):
                    v_ggstar, sigma_ggstar = compute_v_and_sigma_ensemble(
                        candidate_state, env_goal, actor, actor_params,
                        stacked_sa_params, stacked_g_params, sa_encoder, g_encoder,
                        self.energy_fn_name, is_goal_as_state=False
                    )
                    return v_ggstar, sigma_ggstar
                
                # Vectorize over environment goals
                v_ggstar_all, sigma_ggstar_all = jax.vmap(compute_for_env_goal)(env_goals)
                
                # Average over environment goals
                v_ggstar_mean = jnp.mean(v_ggstar_all)
                sigma_ggstar_mean = jnp.mean(sigma_ggstar_all)
                
                return v_s0g, sigma_s0g, v_ggstar_mean, sigma_ggstar_mean
            
            # Vectorize over candidate states and goals
            components = jax.vmap(compute_components_for_candidate)(candidate_states, candidate_goals)
            # Returns: (v_s0g, sigma_s0g, v_ggstar_mean, sigma_ggstar_mean) each of shape (num_candidates,)
            return components
        
        # Compute components for all states
        # Returns a tuple of 4 arrays, each of shape (batch_size, num_candidates)
        v_s0g_all, sigma_s0g_all, v_ggstar_mean_all, sigma_ggstar_mean_all = jax.vmap(compute_score_components_for_state)(current_states)
        
        # Compute p and update alpha_t
        # For now, we'll compute p from the replay buffer by checking if final states
        # achieved their goals. This is a placeholder and may need adjustment.
        # TODO: Get p from actual rollout statistics if available
        def compute_goal_achievement_proportion():
            """Compute proportion of trajectories that achieved their goal.
            
            This is a placeholder implementation that estimates p from the replay buffer.
            In practice, p should come from the actual rollout statistics.
            """
            # Get final states from trajectories for computing goal achievement
            final_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
            
            # Get goals from observations (they're concatenated at the end)
            goal_size = len(goal_indices)
            trajectory_goals = final_states[:, -goal_size:]  # (num_trajs, goal_dim)
            final_state_goals = final_states[:, goal_indices]  # (num_trajs, goal_dim)
            
            # Check if final states are close to their goals (within some threshold)
            # This threshold should match the environment's goal_reach_thresh
            # For now, use a reasonable defaultc
            goal_reach_thresh = getattr(train_env, 'goal_reach_thresh', 0.5)
            distances = jnp.linalg.norm(final_state_goals - trajectory_goals, axis=1)
            achieved = distances < goal_reach_thresh
            p = jnp.mean(achieved.astype(jnp.float32))
            
            return p
        
        # Compute p and update alpha_t
        p = compute_goal_achievement_proportion()
        
        # Get current alpha_t from module-level variable
        def get_current_alpha_callback(default_alpha):
            """Get current alpha from module-level variable."""
            if not hasattr(DISCOVERGoalProposal, '_alpha_t'):
                DISCOVERGoalProposal._alpha_t = float(default_alpha)
            # Return as numpy array (Python value), JAX will convert based on ShapedArray
            return np.array(DISCOVERGoalProposal._alpha_t, dtype=np.float32)
        
        current_alpha = jax.experimental.io_callback(
            get_current_alpha_callback,
            jax.core.ShapedArray((), jnp.float32),  # Return type: scalar float32
            jnp.float32(self.alpha_0)  # Default alpha
        )
        
        alpha_t_new = jnp.clip(current_alpha + 0.01 * (p - self.target_prob), 0.0, 1.0)
        
        # Update alpha_t (using a callback since we can't mutate in JAX)
        def update_alpha_callback(new_alpha):
            """Update alpha_t module-level variable."""
            DISCOVERGoalProposal._alpha_t = float(new_alpha)
        
        jax.experimental.io_callback(
            update_alpha_callback,
            None,
            alpha_t_new
        )
        
        # Combine components with alpha_t to get final scores
        
        # Compute final scores with alpha_t
        all_scores = (alpha_t_new * (v_s0g_all + sigma_s0g_all) + 
                     (1 - alpha_t_new) * (v_ggstar_mean_all + sigma_ggstar_mean_all))
        # (batch_size, num_candidates)
        
        # Select the candidate goal with maximum score for each state
        best_goal_indices = jnp.argmax(all_scores, axis=1)  # (batch_size,)
        proposed_goals = candidate_goals[best_goal_indices]  # (batch_size, goal_dim)
        
        # Log alpha_t, p, and visualization
        jax.experimental.io_callback(
            DISCOVERGoalProposal._log_discover_statistics,
            None,
            all_scores,
            candidate_goals,
            current_states,
            best_goal_indices,
            goal_indices,
            alpha_t_new,
            p,
            training_state.env_steps,
            self.LOG_INTERVAL_STEPS
        )

        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_discover_statistics(
        all_scores, candidate_goals, current_states, best_goal_indices,
        goal_indices, alpha_t, p, env_steps, log_interval_steps
    ):
        """Log DISCOVER statistics and create visualization."""
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'discover'):
            return
        
        metrics = {
            'discover/alpha_t': float(alpha_t),
            'discover/goal_achievement_proportion': float(p)
        }
        
        # Create visualization using shared utility
        def title_fn(state_idx, max_val, selected_val):
            return f'State {state_idx}: Max Score = {max_val:.4f}, Selected = {selected_val:.4f}'
        
        pil_image = create_2x2_scatter_plot(
            candidate_goals, current_states, goal_indices, all_scores,
            selected_indices=best_goal_indices, title_fn=title_fn,
            cmap='viridis', color_label='DISCOVER Score'
        )
        metrics['discover/goal_selection_visualization'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))

@dataclass
class UCGRGoalProposal:
    """
    Unsupervised Contrastive Goal-Reaching (UCGR) proposer.
    
    Attributes:
        energy_fn_name: Energy function to use ("dot" for inner product)
        num_samples: Number of (s, a) pairs to sample for MinLSE computation
    """
    energy_fn_name: str = "dot"  # Energy function: f(s,a,g) = φ(s,a)^T ψ(g)
    num_samples: int = 100  # Number of (s, a) pairs to sample
    
    def propose_goals(
        self, 
        replay_buffer, 
        buffer_state, 
        training_state, 
        train_env, 
        env_state, 
        key,
        actor, 
        actor_params, 
        critic_params, 
        sa_encoder, 
        g_encoder
    ):
        """
        Propose goals using the MinLSE strategy.
        
        This follows Algorithm 1, lines 9-11:
        1. Sample K (s, a) pairs from replay buffer
        2. For each (s, a) pair, find its trajectory's final state as candidate goal
        3. Compute S(g_j) = log Σ_i exp(f(s_i, a_i, g_j)) for each candidate goal
        4. Select g* = argmin_g S(g)
        
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """        
        batch_size = env_state.obs.shape[0]
        goal_indices = train_env.goal_indices
        state_size = train_env.state_dim
        K = self.num_samples  # Number of (s, a) pairs to sample
        
        # Sample trajectories from replay buffer
        buffer_state, sample_batch = replay_buffer.sample(buffer_state)
        
        # sample_batch.observation has shape (N, ep_len, obs_dim)
        observations = sample_batch.observation  # (N, ep_len, obs_dim)
        actions = sample_batch.action  # (N, ep_len, action_dim)
        traj_ids = sample_batch.extras["state_extras"]["traj_id"]  # (N, ep_len)
        
        N, ep_len = observations.shape[:2]
        
        # Randomly sample K indices from all (trajectory, timestep) pairs
        key, sample_key = jax.random.split(key)
        total_pairs = N * ep_len
        # Sample K random indices (with replacement if K > total_pairs)
        flat_indices = jax.random.randint(sample_key, (K,), 0, total_pairs)
        traj_indices = flat_indices // ep_len  # Which trajectory
        time_indices = flat_indices % ep_len   # Which timestep within trajectory
        
        # Extract the K sampled (s, a) pairs
        sampled_states = observations[traj_indices, time_indices, :state_size]  # (K, state_dim)
        sampled_actions = actions[traj_indices, time_indices]  # (K, action_dim)
        
        # For each sampled (s, a), find the final state of its trajectory
        # First get the traj_id for each sampled pair
        sampled_traj_ids = traj_ids[traj_indices, time_indices]  # (K,)
        
        def get_final_state_for_sample(traj_idx, time_idx, sampled_traj_id):
            """Get the final state of the trajectory containing this (s, a) pair."""
            obs_seq = observations[traj_idx]  # (ep_len, obs_dim)
            traj_id_seq = traj_ids[traj_idx]  # (ep_len,)
            
            # Find the last timestep with the same traj_id
            mask = traj_id_seq == sampled_traj_id
            last_idx = jnp.max(jnp.where(mask, jnp.arange(ep_len), 0))
            return obs_seq[last_idx, goal_indices]  # (goal_dim,)
        
        # Get candidate goals: final states for each sampled (s, a) pair's trajectory
        candidate_goals = jax.vmap(get_final_state_for_sample)(
            traj_indices, time_indices, sampled_traj_ids
        )  # (K, goal_dim)
        
        # Compute MinLSE scores using the K (s, a) pairs
        # For each goal g_j, compute S(g_j) = log Σ_i exp(f(s_i, a_i, g_j))
        
        # Compute state-action encodings φ(s_i, a_i)
        sa_pairs = jnp.concatenate([sampled_states, sampled_actions], axis=-1)  # (K, state_dim + action_dim)
        sa_encodings = sa_encoder.apply(critic_params["sa_encoder"], sa_pairs)  # (K, encoding_dim)
        
        # Compute goal encodings ψ(g_j)
        psi_g = g_encoder.apply(critic_params["g_encoder"], candidate_goals)  # (K, encoding_dim)
        
        # Compute all pairwise energies f(s_i, a_i, g_j) for i, j in [0, K)
        # Result: energies[i, j] = f(s_i, a_i, g_j)
        sa_rep = jnp.repeat(sa_encodings[:, None, :], K, axis=1)  # (K, K, encoding_dim)
        psi_rep = jnp.repeat(psi_g[None, :, :], K, axis=0)  # (K, K, encoding_dim)
        
        sa_flat = sa_rep.reshape(-1, sa_rep.shape[-1])  # (K*K, encoding_dim)
        psi_flat = psi_rep.reshape(-1, psi_rep.shape[-1])  # (K*K, encoding_dim)
        
        energies_flat = energy_fn(self.energy_fn_name, sa_flat, psi_flat)  # (K*K,)
        energies = energies_flat.reshape(K, K)  # (K, K) - energies[i, j] = f(s_i, a_i, g_j)
        
        # Compute scores: S(g_j) = log Σ_i exp(f(s_i, a_i, g_j))
        scores = jax.scipy.special.logsumexp(energies, axis=0)  # (K,)
        
        # Select goal with minimum score: g* = argmin_g S(g)
        min_idx = jnp.argmin(scores)
        proposed_goals = jnp.repeat(
            candidate_goals[min_idx][None, :],
            batch_size,
            axis=0,
        )

        return proposed_goals, buffer_state

@dataclass  
class MEGAGoalProposal:
    """Maximum Entropy Goal Achievement (MEGA) proposer.
    
    Selects goals from low-density regions of the achieved goal distribution
    to maximize exploration at the frontier of achievable goals.
    
    Based on Algorithm 2 from the MEGA paper.
    """
    num_candidates: int = 100  # Number of candidate goals to sample
    bandwidth: float = 0.1  # KDE bandwidth
    use_q_cutoff: bool = True  # Whether to eliminate unachievable goals using Q-values
    cutoff_percentile: float = 0.3  # Q-value percentile for cutoff (lower = more restrictive)
    energy_fn_name: str = "dot"  # Energy function to use for Q-value computation
    
    def sample_candidate_goals(self, replay_buffer, buffer_state, train_env, key):
        """Sample candidate goals from replay buffer.
        
        Args:
            replay_buffer: Replay buffer containing past transitions
            buffer_state: Current state of replay buffer
            train_env: Training environment (for goal_indices)
            key: JAX random key
            
        Returns:
            candidate_goals: (num_candidates, goal_dim) array of sampled goals
            buffer_state: Updated buffer state
        """
        goal_indices = train_env.goal_indices
        
        # Sample trajectories from replay buffer
        buffer_state, sample_batch = replay_buffer.sample(buffer_state)
        
        # sample_batch.observation has shape (N, ep_len, obs_dim)
        observations = sample_batch.observation
        N, ep_len = observations.shape[:2]
        
        # Sample num_candidates random states as candidate goals (any state, not just final)
        key, sample_key = jax.random.split(key)
        total_states = N * ep_len
        flat_indices = jax.random.randint(sample_key, (self.num_candidates,), 0, total_states)
        traj_indices = flat_indices // ep_len  # Which trajectory
        time_indices = flat_indices % ep_len   # Which timestep within trajectory
        
        # Extract candidate goals from sampled states
        candidate_goals = observations[traj_indices, time_indices][:, goal_indices]  # (num_candidates, goal_dim)
        
        return candidate_goals, buffer_state
    
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, 
                     actor, actor_params, critic_params, sa_encoder, g_encoder, candidate_goals=None):
        """Propose goals by selecting minimum density candidates from replay buffer.
        
        Args:
            replay_buffer: Replay buffer containing past transitions
            buffer_state: Current state of replay buffer
            training_state: Training state with networks
            train_env: Training environment
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters  
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            candidate_goals: Optional pre-sampled candidate goals (num_candidates, goal_dim).
                             If None, samples new candidates from replay buffer.
            
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        from jaxgcrl.agents.crl.losses import energy_fn
        
        batch_size = env_state.obs.shape[0]
        state_size = train_env.state_dim
        
        # Sample candidate goals if not provided
        if candidate_goals is None:
            key, sample_key = jax.random.split(key)
            candidate_goals, buffer_state = self.sample_candidate_goals(
                replay_buffer, buffer_state, train_env, sample_key
            )
        
        # For each environment state, select minimum density goal from candidates
        def select_goal_for_state(current_state):
            """Select minimum density goal for one environment state."""
            # Compute density for each candidate using KDE
            # Normalize for numerical stability
            mean = jnp.mean(candidate_goals, axis=0)
            std = jnp.std(candidate_goals, axis=0) + 1e-6
            
            candidates_normalized = (candidate_goals - mean) / std
            
            # Compute densities using Gaussian KDE
            densities = gaussian_kernel_density(candidates_normalized, candidates_normalized, self.bandwidth)
            
            # Optional: Filter unachievable goals using Q-values
            if self.use_q_cutoff:
                # Vectorized computation using utility function
                s_rep = jnp.repeat(current_state[None, :], len(candidate_goals), axis=0)
                q_values = compute_energy_for_state_goal_pairs(
                    s_rep, candidate_goals, actor, actor_params,
                    critic_params, sa_encoder, g_encoder, self.energy_fn_name
                )
                
                # Compute adaptive cutoff (percentile of Q-values)
                cutoff_value = jnp.percentile(q_values, self.cutoff_percentile * 100)
                
                # Set density of unachievable goals to infinity (so they won't be selected)
                densities = jnp.where(q_values >= cutoff_value, densities, jnp.inf)
            
            # Select minimum density candidate
            min_idx = jnp.argmin(densities)
            return candidate_goals[min_idx]
    
        
        # Process all states in batch
        current_states = env_state.obs[:, :state_size]
        proposed_goals = jax.vmap(select_goal_for_state)(current_states)
        
        return proposed_goals, buffer_state


@dataclass
class OMEGAGoalProposal:
    """OMEGA (annealing MEGA to desired goals) proposer.
    
    Anneals from MEGA exploration to desired goal distribution using α parameter
    that depends on KL divergence between desired and achieved distributions.
    
    α = 1 / max(b + D_KL(p_dg || p_ag), 1)
    
    With probability α: sample from desired goal distribution
    With probability 1-α: use MEGA to explore low-density regions
    
    Based on Algorithm 2 from the MEGA paper.
    """
    num_candidates: int = 100
    bandwidth: float = 0.1  
    use_q_cutoff: bool = True
    cutoff_percentile: float = 0.3
    energy_fn_name: str = "dot"
    bias_param: float = -3.0  # 'b' in paper, controls annealing speed (-3 recommended)
    alpha_update_freq: int = 1000  # Update α every N environment steps
    
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key,
                     actor, actor_params, critic_params, sa_encoder, g_encoder):
        """Propose goals by annealing between MEGA and desired goals.
        
        Returns:
            proposed_goals: (batch_size, goal_dim) array of proposed goals
            buffer_state: Updated buffer state
        """
        assert hasattr(train_env, 'possible_goals'), \
            "Environment must store property `possible_goals` for OMEGAGoalProposal."
        
        batch_size = env_state.obs.shape[0]
        
        # Get desired goals from environment
        desired_goals = train_env.possible_goals  # (num_env_goals, goal_dim)
        
        # Create MEGA proposer (used for both sampling and goal selection)
        mega_proposer = MEGAGoalProposal(
            num_candidates=self.num_candidates,
            bandwidth=self.bandwidth,
            use_q_cutoff=self.use_q_cutoff,
            cutoff_percentile=self.cutoff_percentile,
            energy_fn_name=self.energy_fn_name
        )
        
        # Sample candidate goals once - used for both KL divergence and MEGA
        key, sample_key = jax.random.split(key)
        achieved_goals, buffer_state = mega_proposer.sample_candidate_goals(
            replay_buffer, buffer_state, train_env, sample_key
        )  # (num_candidates, goal_dim)
        
        # Compute α based on KL divergence between desired and achieved goal distributions
        kl_div = compute_kl_divergence_empirical(desired_goals, achieved_goals, self.bandwidth)
        alpha = 1.0 / jnp.maximum(self.bias_param + kl_div, 1.0)
        
        # Log alpha value using wandb
        def log_alpha_callback(alpha_val, env_steps):
            """Log alpha to wandb."""
            metrics = {
                'omega/alpha': float(alpha_val),
            }
            wandb.log(metrics, step=int(env_steps))
        
        jax.experimental.io_callback(
            log_alpha_callback,
            None,
            alpha,
            training_state.env_steps
        )
        
        # Decide whether to use MEGA or environment goals
        key, choice_key, mega_key = jax.random.split(key, 3)
        use_env_goals = jax.random.uniform(choice_key, (batch_size,)) < alpha
        
        # Get MEGA goals using the same sampled candidates (no redundant sampling)
        mega_goals, buffer_state = mega_proposer.propose_goals(
            replay_buffer, buffer_state, training_state, train_env, env_state,
            mega_key, actor, actor_params, critic_params, sa_encoder, g_encoder,
            candidate_goals=achieved_goals  # Reuse the same samples
        )
        
        # Sample from desired goals for environments that should use env goals
        key, sample_key = jax.random.split(key)
        env_goal_indices = jax.random.randint(sample_key, (batch_size,), 0, len(desired_goals))
        sampled_env_goals = desired_goals[env_goal_indices]
        
        # Mix goals based on α
        proposed_goals = jnp.where(
            use_env_goals[:, None],
            sampled_env_goals,
            mega_goals
        )
        
        return proposed_goals, buffer_state


@dataclass 
class ReplayBufferGoalProposal(GoalProposer):
    """CRL-compatible wrapper for ReplayBufferGoalProposal.
    
    Accepts CRL-specific arguments but delegates to base implementation.
    """
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        # Delegate to base proposer, ignoring CRL-specific params
        base_proposer = BaseReplayBufferGoalProposal()
        return base_proposer.propose_goals(
            replay_buffer, buffer_state, train_env, env_state, key
        )


@dataclass
class FisherTraceGoalProposal(GoalProposer):
    energy_fn_name: str
    use_critic_gradients: bool = True  # Include critic (phi, psi encoder) gradients in Fisher trace
    use_actor_gradients: bool = False  # Include actor gradients in Fisher trace
    temperature: float = 0.0  # Temperature for softmax sampling (0 = argmax, higher = more uniform)
    propose_env_goals: bool = False  # If True, use environment goals instead of replay buffer samples
    LOG_INTERVAL_STEPS: int = 1000000  # Log visualizations every N environment steps
    _last_log_step: int = -500000  # Track last logged step (start negative to log first time)

    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        # Get current states from env_state
        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)

        if self.propose_env_goals:
            assert hasattr(train_env, 'possible_goals'), "Environment must have 'possible_goals' for propose_env_goals=True."
            candidate_goals = train_env.possible_goals  # (num_env_goals, goal_dim)
        else:
            # Sample one batch to get candidate final states
            buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
            traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
            candidate_obs = candidate_transitions.observation
            last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
            candidate_goals = last_states[:, train_env.goal_indices]  # (batch_size, goal_size)
        
        use_critic = self.use_critic_gradients
        use_actor = self.use_actor_gradients
        
        def compute_fisher_traces_for_state(state):
            def fisher_trace_for_goal(goal):
                obs = jnp.concatenate([state, goal])
                
                def get_action_and_q(actor_p):
                    means, log_stds = actor.apply(actor_p, obs[None, :])
                    action = jnp.tanh(means[0])
                    sa_pair = jnp.concatenate([state, action])
                    phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pair[None, :])[0]
                    psi_g = g_encoder.apply(critic_params['g_encoder'], goal[None, :])[0]
                    return energy_fn(self.energy_fn_name, phi_sa, psi_g)
                
                means, log_stds = actor.apply(actor_params, obs[None, :])
                action = jnp.tanh(means[0])
                sa_pair = jnp.concatenate([state, action])
                
                def log_q_value(phi_params, psi_params):
                    phi_sa = sa_encoder.apply(phi_params, sa_pair[None, :])[0]
                    psi_g = g_encoder.apply(psi_params, goal[None, :])[0]
                    return energy_fn(self.energy_fn_name, phi_sa, psi_g)
                
                total_fisher_trace = 0.0
                
                if use_critic:
                    grad_phi_params = jax.grad(lambda p: log_q_value(p, critic_params['g_encoder']))(
                        critic_params['sa_encoder']
                    )
                    grad_psi_params = jax.grad(lambda p: log_q_value(critic_params['sa_encoder'], p))(
                        critic_params['g_encoder']
                    )
                    flat_grad_phi = jax.flatten_util.ravel_pytree(grad_phi_params)[0]
                    flat_grad_psi = jax.flatten_util.ravel_pytree(grad_psi_params)[0]
                    fisher_trace_phi = jnp.sum(flat_grad_phi ** 2)
                    fisher_trace_psi = jnp.sum(flat_grad_psi ** 2)
                    total_fisher_trace += fisher_trace_phi + fisher_trace_psi
                
                if use_actor:
                    grad_actor_params = jax.grad(get_action_and_q)(actor_params)
                    flat_grad_actor = jax.flatten_util.ravel_pytree(grad_actor_params)[0]
                    fisher_trace_actor = jnp.sum(flat_grad_actor ** 2)
                    total_fisher_trace += fisher_trace_actor
                
                return total_fisher_trace
            
            # Vectorize over candidate goals
            fisher_traces = jax.vmap(fisher_trace_for_goal)(candidate_goals)
            return fisher_traces

        # Vectorize over all states
        all_fisher_traces = jax.vmap(compute_fisher_traces_for_state)(current_states)

        # For each state, select a candidate goal based on Fisher trace
        if self.temperature == 0.0:
            best_goal_indices = jnp.argmax(all_fisher_traces, axis=1)  # (batch_size,)
        else:
            logits = all_fisher_traces / self.temperature
            key, sample_key = jax.random.split(key)
            best_goal_indices = jax.random.categorical(sample_key, logits, axis=1)  # (batch_size,)

        proposed_goals = candidate_goals[best_goal_indices]  # (batch_size, goal_size)
                
        # Log Fisher trace statistics with visualization only at specified intervals
        jax.experimental.io_callback(
            FisherTraceGoalProposal._log_fisher_trace_statistics,
            None,
            all_fisher_traces,
            candidate_goals,
            current_states,
            train_env.goal_indices,
            training_state.env_steps,
            self.LOG_INTERVAL_STEPS
        )
        
        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_fisher_trace_statistics(all_fisher_traces, candidate_goals, current_states, goal_indices, env_steps, log_interval_steps):
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'fisher_trace'):
            return
            
        # all_fisher_traces: (batch_size, num_candidates)
        max_traces_per_state = jnp.max(all_fisher_traces, axis=1)  # (batch_size,)
        
        metrics = {
            'fisher_trace/max_trace_mean': float(jnp.mean(max_traces_per_state)),
            'fisher_trace/max_trace_std': float(jnp.std(max_traces_per_state)),
            'fisher_trace/max_trace_max': float(jnp.max(max_traces_per_state)),
            'fisher_trace/max_trace_min': float(jnp.min(max_traces_per_state)),
        }
        
        # Create visualization using shared utility
        def title_fn(state_idx, max_val, selected_val):
            max_trace_idx = int(np.argmax(all_fisher_traces[state_idx]))
            return f'State {state_idx}: Max Fisher Trace = {max_val:.4f} (Goal {max_trace_idx})'
        
        pil_image = create_2x2_scatter_plot(
            candidate_goals, current_states, goal_indices, all_fisher_traces,
            title_fn=title_fn, cmap='hot', color_label='Fisher Trace'
        )
        metrics['fisher_trace/trace_heatmaps'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class MediumEnergyGoalProposal(GoalProposer):
    '''Proposes goals by selecting final trajectory states with medium energy values.
    
    For each state in the batch:
    1. Sample one batch of candidate goals from replay buffer final states
    2. Compute energy values for all (state, candidate_goal) pairs
    3. For each state, select the candidate goal with median energy
    '''
    energy_fn_name: str
    selection_percentile: float
    
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        '''Propose goals with medium energy values.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            train_env: Training environment
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters
            sa_encoder: State-action encoder
            g_encoder: Goal encoder
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state
        '''
        # Get current states from env_state
        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        batch_size = current_states.shape[0]
        
        # Sample one batch to get candidate final states
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation
        last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, train_env.goal_indices]  # (batch_size, goal_size)
        
        # Compute energies for all (current_state, candidate_goal) pairs
        # This creates a batch_size x batch_size matrix of energies
        
        def compute_energies_for_state(state):
            '''For a single state, compute energies with all candidate goals.
            
            Args:
                state: (state_dim,) array
                
            Returns:
                energies: (batch_size,) array of energy values
            '''
            # Compute energies using utility function
            state_expanded = jnp.tile(state, (batch_size, 1))  # (batch_size, state_dim)
            energies = compute_energy_for_state_goal_pairs(
                state_expanded, candidate_goals, actor, actor_params,
                critic_params, sa_encoder, g_encoder, self.energy_fn_name
            )
            return energies
        
        # Compute energies for all states: (batch_size, batch_size)
        # Row i contains energies for current_states[i] with all candidate_goals
        all_energies = jax.vmap(compute_energies_for_state)(current_states)
        
        # For each state, find the candidate goal with median energy
        def select_median_energy_goal(energies):
            '''Select the goal with median energy.
            
            Args:
                energies: (batch_size,) array of energy values
                
            Returns:
                goal_idx: scalar index of the median goal
            '''
            sorted_indices = jnp.argsort(energies)
            percentile_idx = int(batch_size * self.selection_percentile)
            return sorted_indices[percentile_idx]

        # Get median goal index for each state
        median_indices = jax.vmap(select_median_energy_goal)(all_energies)  # (batch_size,)
        
        # Select the corresponding goals
        proposed_goals = candidate_goals[median_indices]  # (batch_size, goal_size)

        # Log statistics to wandb
        jax.experimental.io_callback(
            MediumEnergyGoalProposal._log_energy_statistics,
            None,  # No return value
            all_energies,
            all_energies[jnp.arange(batch_size), median_indices],
            training_state.env_steps
        )
        
        return proposed_goals, buffer_state

    @staticmethod    
    def _log_energy_statistics(all_energies, selected_energies, env_steps):
        # Create visualization using shared utility
        pil_image = create_energy_histogram_plot(all_energies, selected_energies)
        
        # Aggregate statistics for scalar tracking
        energy_stats = {
            'goal_proposal/energy_mean_across_states': float(jnp.mean(all_energies)),
            'goal_proposal/selected_energy_mean': float(jnp.mean(selected_energies)),
            'goal_proposal/selected_energy_std': float(jnp.std(selected_energies)),
            'goal_proposal/per_state_energy_std_avg': float(jnp.mean(jnp.std(all_energies, axis=1))),
            # Image with all histograms
            'goal_proposal/energy_distributions': wandb.Image(pil_image),
        }
        
        wandb.log(energy_stats, step=int(env_steps))

@dataclass
class QEpistemicGoalProposal(GoalProposer):
    """Proposes goals by selecting those with highest epistemic uncertainty.
    
    Uses an ensemble of critics to estimate uncertainty. For each state in the batch:
    1. Sample candidate goals from replay buffer final states or environment goals
    2. For each (state, candidate_goal) pair, sample an action from the policy
    3. Compute Q-values for the triplet (state, action, goal) across the ensemble
    4. Select the goal with highest standard deviation across the ensemble
    
    This encourages exploration by selecting goals where the agent is most uncertain.
    """
    energy_fn_name: str
    num_ensemble: int = 5  # Number of critics in the ensemble
    use_env_goals: bool = False  # If True, use environment goals; if False, use replay buffer final states
    zero_center: bool = False  # If True, center each critic's predictions before computing std
    LOG_INTERVAL_STEPS: int = 1000000  # Log visualizations every N environment steps

    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        """Propose goals with highest epistemic uncertainty.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            training_state: Current training state
            train_env: Training environment
            env_state: Current environment state
            key: JAX random key
            actor: Actor network
            actor_params: Actor parameters
            critic_params: Critic parameters (contains ensemble of sa_encoder and g_encoder params)
            sa_encoder: State-action encoder network
            g_encoder: Goal encoder network
            
        Returns:
            proposed_goals: (batch_size, goal_size) array of proposed goals
            buffer_state: Updated buffer state
        """
        # Get current states from env_state
        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        batch_size = current_states.shape[0]
        
        # Get candidate goals based on configuration
        if self.use_env_goals:
            assert hasattr(train_env, 'possible_goals'), \
                "Environment must store property `possible_goals` for QEpistemicGoalProposal with use_env_goals=True."
            candidate_goals = train_env.possible_goals  # (num_candidate_goals, goal_size)
        else:
            # Sample from replay buffer final states
            buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
            traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
            candidate_obs = candidate_transitions.observation
            last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
            candidate_goals = last_states[:, train_env.goal_indices]  # (num_candidates, goal_size)
        
        num_candidates = candidate_goals.shape[0]
        
        # Stack ensemble parameters into arrays for JAX-compatible indexing
        stacked_sa_params, stacked_g_params = stack_ensemble_params(critic_params)
        
        def compute_q_values_for_state(state):
            """For a single state, compute Q-values across ensemble for all candidate goals.
            
            Args:
                state: (state_dim,) array
                
            Returns:
                all_q_values: (num_ensemble, num_candidates) array of Q-values
            """
            # Expand state to match number of candidates
            state_expanded = jnp.tile(state, (num_candidates, 1))  # (num_candidates, state_dim)
            
            # Compute Q-values using utility function
            all_q_values = compute_q_values_ensemble(
                state_expanded, candidate_goals, actor, actor_params,
                stacked_sa_params, stacked_g_params, sa_encoder, g_encoder,
                self.energy_fn_name, expand_goals=False
            )  # (num_ensemble, num_candidates)
            
            return all_q_values
        
        # Compute Q-values for all states: (batch_size, num_ensemble, num_candidates)
        all_ensemble_q_values = jax.vmap(compute_q_values_for_state)(current_states)
        
        # Optionally center each critic's predictions by subtracting its mean
        if self.zero_center:
            # Compute mean for each critic across all states and candidates
            critic_means = jnp.mean(all_ensemble_q_values, axis=(0, 2), keepdims=True)  # (1, num_ensemble, 1)
            # Subtract the mean from each critic's predictions to remove translational offset
            q_values_for_std = all_ensemble_q_values - critic_means  # (batch_size, num_ensemble, num_candidates)
        else:
            q_values_for_std = all_ensemble_q_values
        
        # Compute standard deviation across ensemble for each (state, candidate) pair
        all_q_stds = jnp.std(q_values_for_std, axis=1)  # (batch_size, num_candidates)
        
        # For each state, select the candidate goal with highest std
        best_goal_indices = jnp.argmax(all_q_stds, axis=1)  # (batch_size,)
        proposed_goals = candidate_goals[best_goal_indices]  # (batch_size, goal_size)
        
        # Log Q-epistemic statistics
        jax.experimental.io_callback(
            QEpistemicGoalProposal._log_q_epistemic_statistics,
            None,
            all_q_stds,
            all_ensemble_q_values,  # Pass raw Q-values for deviation plotting
            candidate_goals,
            current_states,
            train_env.goal_indices,
            training_state.env_steps,
            self.LOG_INTERVAL_STEPS
        )
        
        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_q_epistemic_statistics(all_q_stds, all_ensemble_q_values, candidate_goals, current_states, goal_indices, env_steps, log_interval_steps):
        """Log Q-epistemic uncertainty statistics."""
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'q_epistemic'):
            return
        
        # all_q_stds: (batch_size, num_candidates)
        # all_ensemble_q_values: (batch_size, num_ensemble, num_candidates)
        max_stds_per_state = jnp.max(all_q_stds, axis=1)  # (batch_size,)
        
        metrics = {
            'q_epistemic/max_std_mean': float(jnp.mean(max_stds_per_state)),
            'q_epistemic/max_std_std': float(jnp.std(max_stds_per_state)),
            'q_epistemic/max_std_max': float(jnp.max(max_stds_per_state)),
            'q_epistemic/max_std_min': float(jnp.min(max_stds_per_state)),
            'q_epistemic/mean_std_across_candidates': float(jnp.mean(all_q_stds)),
        }
        
        # Compute overall mean across all critics, states, and candidates
        # all_ensemble_q_values: (batch_size, num_ensemble, num_candidates)
        overall_mean = jnp.mean(all_ensemble_q_values)  # scalar
        
        # For each critic, compute:
        # 1. Mean absolute deviation from overall mean (scalar)
        # 2. Mean of that critic's predictions (scalar)
        # 3. Ratio of deviation to critic mean
        num_ensemble = all_ensemble_q_values.shape[1]
        
        for critic_idx in range(num_ensemble):
            critic_values = all_ensemble_q_values[:, critic_idx, :]  # (batch_size, num_candidates)
            
            # Mean absolute deviation from overall mean
            deviation = jnp.mean(jnp.abs(critic_values - overall_mean))
            
            # Mean of this critic's predictions
            critic_mean = jnp.mean(critic_values)
            
            # Ratio of deviation to critic mean
            ratio = deviation / (jnp.abs(critic_mean) + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Log as scalar metrics
            metrics[f'q_epistemic/critic_{critic_idx}_deviation'] = float(deviation)
            metrics[f'q_epistemic/critic_{critic_idx}_mean'] = float(critic_mean)
            metrics[f'q_epistemic/critic_{critic_idx}_deviation_ratio'] = float(ratio)
        
        # Create visualization using shared utility
        def title_fn(state_idx, max_val, selected_val):
            max_std_idx = int(np.argmax(all_q_stds[state_idx]))
            return f'State {state_idx}: Max Q-Std = {max_val:.4f} (Goal {max_std_idx})'
        
        pil_image = create_2x2_scatter_plot(
            candidate_goals, current_states, goal_indices, all_q_stds,
            title_fn=title_fn, cmap='hot', color_label='Q-value Std (Epistemic Uncertainty)'
        )
        metrics['q_epistemic/uncertainty_heatmaps'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))


@dataclass
class MetricPreservationGoalProposal(GoalProposer):
    energy_fn_name: str
    use_one_env_goal: bool = False
    use_kde_correction: bool = False
    use_waypoint_difficulty: bool = True
    use_max: bool = False  # If True, simply take max over all (g, h) pairs instead of using logsumexp
    zero_out_cand_goals: bool = True
    zero_out_state: bool = False  # If True, zero out the current state when computing energy terms
    propose_env_goals: bool = False  # If True, propose environment goals instead of waypoint goals
    goal_sampling_temperature: float = 1.0  # Temperature for softmax sampling over M matrix (0 = greedy, >0 = softmax)
    LOG_INTERVAL_STEPS: int = 1000000  # Log visualizations every N environment steps

    def propose_goals(self, replay_buffer, buffer_state, training_state,
                      train_env, env_state, key, actor, actor_params, critic_params,
                      sa_encoder, g_encoder):

        assert hasattr(train_env, 'possible_goals'), \
            "Environment must store property `possible_goals` for MetricPreservationGoalProposal."

        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch, state_dim)

        # --- candidate goals from replay buffer ---
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation

        last_states = jax.vmap(get_last_state_from_trajectory)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, train_env.goal_indices]  # (num_candidate_goals, goal_dim)
        candidate_goals_full = last_states[:, :state_size] # Full vector for final states achieved

        if self.zero_out_cand_goals:
            candidate_goals_full = jax.vmap(
                lambda g: expand_goal_to_state(g, state_size, train_env.goal_indices)
            )(candidate_goals)

        env_goals = train_env.possible_goals  # (num_env_goals, goal_dim)

        def energy_triplet(state):
            """Compute M[g,h] for a single state and return individual terms."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                state = zero_out_non_goal_indices(state, train_env.goal_indices)
            
            # Use utility function for KDE estimation
            proposed_goal_densities = estimate_log_density_knn(candidate_goals)
            
            num_cand = candidate_goals.shape[0]
            num_env = env_goals.shape[0]

            # f(s, a1, g)
            s1 = jnp.repeat(state[None, :], num_cand, axis=0)
            obs_sg = jnp.concatenate([s1, candidate_goals], axis=1)
            means, _ = actor.apply(actor_params, obs_sg)
            a1 = jnp.tanh(means)
            phi_sg = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s1, a1], axis=1))
            psi_g = g_encoder.apply(critic_params['g_encoder'], candidate_goals)
            f_sag = energy_fn(self.energy_fn_name, phi_sg, psi_g)  # (num_cand,)

            # f(g, a2, h)
            g_exp = jnp.repeat(candidate_goals_full[:, None, :], num_env, axis=1)  # (num_cand, num_env, state_dim)
            h_exp = jnp.repeat(env_goals[None, :, :], num_cand, axis=0)
            obs_gh = jnp.concatenate([g_exp, h_exp], axis=-1).reshape(num_cand * num_env, -1)
            means2, _ = actor.apply(actor_params, obs_gh)
            a2 = jnp.tanh(means2)
            phi_gh = sa_encoder.apply(critic_params['sa_encoder'],
                                      jnp.concatenate([g_exp.reshape(-1, g_exp.shape[-1]), a2], axis=1))
            psi_h = g_encoder.apply(critic_params['g_encoder'], env_goals)
            psi_h_rep = jnp.repeat(psi_h[None, :, :], num_cand, axis=0).reshape(num_cand * num_env, -1)
            f_gah = energy_fn(self.energy_fn_name, phi_gh, psi_h_rep).reshape(num_cand, num_env)

            # f(s, a3, h)
            s3 = jnp.repeat(state[None, :], num_env, axis=0)
            obs_sh = jnp.concatenate([s3, env_goals], axis=1)
            means3, _ = actor.apply(actor_params, obs_sh)
            a3 = jnp.tanh(means3)
            phi_sh = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s3, a3], axis=1))
            f_sah = energy_fn(self.energy_fn_name, phi_sh, psi_h)  # (num_env,)
            
            # Compute M matrix for goal selection
            term1 = f_sag[:, None]  # f(s, a1, g) - shape (num_cand, 1)
            term2 = f_gah  # f(g, a2, h) - shape (num_cand, num_env)
            term3 = f_sah[None, :]  # -f(s, a3, h) - shape (1, num_env)
            kde_term = proposed_goal_densities[:, None]  # KDE correction - shape (num_cand, 1)
            
            M = term2 - term3
            if self.use_waypoint_difficulty:
                M += term1
            if self.use_kde_correction:
                M += kde_term
            return M
        
        def energy_triplet_with_terms(state):
            """Compute M and all individual terms for visualization (only called for one state)."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                state = zero_out_non_goal_indices(state, train_env.goal_indices)
            
            num_cand = candidate_goals.shape[0]
            num_env = env_goals.shape[0]

            s1 = jnp.repeat(state[None, :], num_cand, axis=0)
            obs_sg = jnp.concatenate([s1, candidate_goals], axis=1)
            means, _ = actor.apply(actor_params, obs_sg)
            a1 = jnp.tanh(means)
            phi_sg = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s1, a1], axis=1))
            psi_g = g_encoder.apply(critic_params['g_encoder'], candidate_goals)
            f_sag = energy_fn(self.energy_fn_name, phi_sg, psi_g)

            g_exp = jnp.repeat(candidate_goals_full[:, None, :], num_env, axis=1)
            h_exp = jnp.repeat(env_goals[None, :, :], num_cand, axis=0)
            obs_gh = jnp.concatenate([g_exp, h_exp], axis=-1).reshape(num_cand * num_env, -1)
            means2, _ = actor.apply(actor_params, obs_gh)
            a2 = jnp.tanh(means2)
            phi_gh = sa_encoder.apply(critic_params['sa_encoder'],
                                      jnp.concatenate([g_exp.reshape(-1, g_exp.shape[-1]), a2], axis=1))
            psi_h = g_encoder.apply(critic_params['g_encoder'], env_goals)
            psi_h_rep = jnp.repeat(psi_h[None, :, :], num_cand, axis=0).reshape(num_cand * num_env, -1)
            f_gah = energy_fn(self.energy_fn_name, phi_gh, psi_h_rep).reshape(num_cand, num_env)

            s3 = jnp.repeat(state[None, :], num_env, axis=0)
            obs_sh = jnp.concatenate([s3, env_goals], axis=1)
            means3, _ = actor.apply(actor_params, obs_sh)
            a3 = jnp.tanh(means3)
            phi_sh = sa_encoder.apply(critic_params['sa_encoder'], jnp.concatenate([s3, a3], axis=1))
            f_sah = energy_fn(self.energy_fn_name, phi_sh, psi_h)

            proposed_goal_densities = estimate_log_density_knn(candidate_goals)
            
            term1 = f_sag[:, None]
            term2 = f_gah
            term3 = f_sah[None, :]
            kde_term = proposed_goal_densities[:, None]
            
            M = term2 - term3
            if self.use_waypoint_difficulty:
                M += term1
            if self.use_kde_correction:
                M += kde_term
            return M, term1, term2, term3, kde_term

        # compute M for all states (only M, not term matrices)
        energy_mats = jax.vmap(energy_triplet)(current_states)  # (batch, num_cand, num_env)
        
        # compute term matrices for ONE state only (for visualization)
        viz_state_idx = 0
        _, term1_single, term2_single, term3_single, kde_single = energy_triplet_with_terms(current_states[viz_state_idx])

        def select_goal_max(M):
            """Select goal using softmax sampling over M matrix if temperature > 0, else greedy."""
            if self.goal_sampling_temperature > 0:
                # Softmax sampling: flatten M, compute softmax, sample
                M_flat = M.flatten()
                logits = M_flat / self.goal_sampling_temperature
                probs = jax.nn.softmax(logits)
                idx_flat = jax.random.choice(key, a=M_flat.size, p=probs)
                g_idx, h_idx = jnp.unravel_index(idx_flat, M.shape)
            else:
                # Greedy: take argmax
                idx_flat = jnp.argmax(M)
                g_idx, h_idx = jnp.unravel_index(idx_flat, M.shape)
            return g_idx, h_idx

        def select_goal_minimax(M):
            # Step 1: worst-case slack for each candidate goal over all env goals
            worst_case_slack = jnp.max(M, axis=1)  # shape: (num_candidate_goals,)
            # Step 2: pick the candidate goal with minimal worst-case slack
            g_idx = jnp.argmin(worst_case_slack)
            h_idx = jnp.argmax(M[g_idx, :])
            return g_idx, h_idx
        
        def select_goal_minlogsumexp(M):
            score = -jax.scipy.special.logsumexp(M, axis=1)
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmin
                g_idx = jnp.argmin(-score)  # score is negative, so -score is positive
                h_idx = jnp.argmin(M[g_idx])
                return g_idx, h_idx
            g_idx = jax.random.choice(key, a=M.shape[0], p=weights)

            h_idx = jnp.argmin(M[g_idx])
            return g_idx, h_idx
        
        def select_goal_minlogsumexp_one_env(M, rand_key):
            """Select one random environment goal and compute weights using only that column."""
            rand_key_h, rand_key_g = jax.random.split(rand_key)

            # Randomly select one environment goal
            num_env_goals = M.shape[1]
            h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
            
            energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
            score = -energies_for_h  # Negative because we want to minimize
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmin
                g_idx = jnp.argmin(-score)  # score is negative, so -score is positive
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx
        
        def select_goal_maxlogsumexp(M):
            score = jax.scipy.special.logsumexp(M, axis=1)
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmax
                g_idx = jnp.argmax(score)
                h_idx = jnp.argmax(M[g_idx])
                return g_idx, h_idx
            g_idx = jax.random.choice(key, a=M.shape[0], p=weights)

            h_idx = jnp.argmax(M[g_idx])
            return g_idx, h_idx
        
        def select_goal_maxlogsumexp_one_env(M, rand_key):
            """Select one random environment goal and compute weights using only that column."""
            rand_key_h, rand_key_g = jax.random.split(rand_key)

            # Randomly select one environment goal
            num_env_goals = M.shape[1]
            h_idx = jax.random.choice(rand_key_h, a=jnp.arange(num_env_goals))
            
            energies_for_h = M[:, h_idx]  # (num_candidate_goals,)
            score = energies_for_h  # Positive because we want to maximize
            if self.goal_sampling_temperature > 0:
                logits = score / self.goal_sampling_temperature
                weights = jax.nn.softmax(logits)
            else:
                # Greedy: take argmax
                g_idx = jnp.argmax(score)
                return g_idx, h_idx
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx

        if self.use_max:
            # Simple max selection over all (g, h) pairs
            best_g_indices, best_h_indices = jax.vmap(select_goal_max)(energy_mats)
        elif self.use_one_env_goal:
            # Split the key for each batch element
            batch_size = energy_mats.shape[0]
            batch_keys = jax.random.split(key, batch_size)
            if self.use_waypoint_difficulty:
                best_g_indices, best_h_indices = jax.vmap(select_goal_minlogsumexp_one_env)(energy_mats, batch_keys)
            else:
                best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp_one_env)(energy_mats, batch_keys)
        else:
            if self.use_waypoint_difficulty:
                best_g_indices, best_h_indices = jax.vmap(select_goal_minlogsumexp)(energy_mats) 
            else:
                best_g_indices, best_h_indices = jax.vmap(select_goal_maxlogsumexp)(energy_mats)


        # Select proposed goals: either candidate goals (waypoints) or environment goals
        if self.propose_env_goals:
            proposed_goals = env_goals[best_h_indices]  # (batch, goal_dim)
        else:
            proposed_goals = candidate_goals[best_g_indices]      # (batch, goal_dim)

        # Log visualizations only at specified intervals to reduce wandb storage
        jax.experimental.io_callback(
            MetricPreservationGoalProposal._log_goal_selection_viz,
            None,
            current_states,
            candidate_goals,
            env_goals,
            best_g_indices,
            best_h_indices,
            energy_mats,
            term1_single,
            term2_single,
            term3_single,
            kde_single,
            viz_state_idx,
            training_state.env_steps,
            train_env.goal_indices,
            train_env.x_bounds,
            train_env.y_bounds,
            self.LOG_INTERVAL_STEPS
        )

        return proposed_goals, buffer_state
    
    # Class variable to track last log step
    @staticmethod
    def _log_goal_selection_viz(current_states, candidate_goals, env_goals, 
                              best_g_indices, best_h_indices, energy_mats, 
                              term1_single, term2_single, term3_single, kde_single, viz_state_idx,
                              env_steps, goal_indices, x_bounds, y_bounds, log_interval_steps):
        """Visualize goal selection showing trajectory from current -> candidate -> env goals."""
        
        # Only log if enough steps have passed since last log
        if not should_log_at_interval(env_steps, log_interval_steps, 'metric_preservation'):
            return
        
        # Use viz_state_idx for env_goal_ranking plot, random for goal_selection plot
        num_states = current_states.shape[0]
        random_state_indices = np.random.choice(num_states, size=min(4, num_states), replace=False)
        # Make sure viz_state_idx is in random_state_indices for consistency
        random_state_indices[0] = int(viz_state_idx)
        
        # Generate both visualizations using shared utilities
        pil_image1 = create_goal_selection_plot(
            current_states, candidate_goals, env_goals, best_g_indices, best_h_indices, energy_mats, 
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        pil_image2 = create_env_goal_ranking_plot(
            current_states, candidate_goals, env_goals, energy_mats, 
            term1_single, term2_single, term3_single, kde_single, viz_state_idx,
            goal_indices, x_bounds, y_bounds
        )
        
        metrics = {
            'metric_preservation/goal_selection_viz': wandb.Image(pil_image1),
            'metric_preservation/env_goal_rankings': wandb.Image(pil_image2),
        }
        
        wandb.log(metrics, step=int(env_steps))
