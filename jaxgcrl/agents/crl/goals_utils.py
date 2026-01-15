"""Shared utilities for goal proposal algorithms.

This module contains reusable functions for common operations across different
goal proposers, including trajectory processing, Q-value computation, and visualization.
"""
import io
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb

from jaxgcrl.agents.crl.losses import energy_fn


# ============================================================================
# Trajectory Processing Utilities
# ============================================================================

def get_last_state_from_trajectory(obs_seq, traj_id_seq):
    """Get the last state for a single trajectory.
    
    Args:
        obs_seq: (seq_len, obs_dim) observation sequence
        traj_id_seq: (seq_len,) trajectory IDs
        
    Returns:
        last_state: (obs_dim,) last observation in the trajectory
    """
    seq_len = obs_seq.shape[0]
    mask = traj_id_seq == traj_id_seq[0]
    last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
    return obs_seq[last_idx]


def get_final_states_from_batch(observations, traj_ids, goal_indices):
    """Extract final states from each trajectory in a batch.
    
    Args:
        observations: (N, K, obs_dim) sampled observations where N = num trajectory samples, K = episode_length
        traj_ids: (N, K) trajectory IDs for each timestep
        goal_indices: indices to extract goal dimensions from observation
        
    Returns:
        final_goals: (N, goal_dim) final state goals from each sampled trajectory
    """
    last_states = jax.vmap(get_last_state_from_trajectory)(observations, traj_ids)  # (N, obs_dim)
    final_goals = last_states[:, goal_indices]  # (N, goal_dim)
    return final_goals


def sample_random_state_from_trajectory(obs_seq, traj_id_seq, rng_key):
    """Sample a random state from a trajectory.
    
    Args:
        obs_seq: (seq_len, obs_dim) observation sequence
        traj_id_seq: (seq_len,) trajectory IDs
        rng_key: JAX random key
        
    Returns:
        sampled_state: (obs_dim,) randomly sampled observation from the trajectory
    """
    seq_len = obs_seq.shape[0]
    # Find indices that belong to the same trajectory (same traj_id as first element)
    mask = traj_id_seq == traj_id_seq[0]
    # Count how many valid indices we have
    num_valid = jnp.sum(mask.astype(jnp.int32))
    # Sample a random index from valid ones
    random_idx = jax.random.randint(rng_key, (), 0, num_valid)
    # Get all indices, sorted by mask (True values first when descending)
    sorted_indices = jnp.argsort(-mask.astype(jnp.int32))
    # Take the random_idx-th valid index
    sampled_idx = sorted_indices[random_idx]
    return obs_seq[sampled_idx]


def sample_random_states_from_batch(observations, traj_ids, goal_indices, key):
    """Sample one random state from each trajectory in a batch.
    
    Args:
        observations: (N, K, obs_dim) sampled observations
        traj_ids: (N, K) trajectory IDs
        goal_indices: indices to extract goal dimensions
        key: JAX random key
        
    Returns:
        sampled_goals: (N, goal_dim) randomly sampled goals from each trajectory
    """
    num_trajs = observations.shape[0]
    sample_keys = jax.random.split(key, num_trajs)
    
    sampled_states = jax.vmap(sample_random_state_from_trajectory)(
        observations, traj_ids, sample_keys
    )
    sampled_goals = sampled_states[:, goal_indices]
    return sampled_goals


# ============================================================================
# Goal and State Manipulation Utilities
# ============================================================================

def expand_goal_to_state(goal, state_size, goal_indices):
    """Expand a goal to a full state vector with zeros elsewhere.
    
    Args:
        goal: (goal_dim,) goal vector
        state_size: size of full state vector
        goal_indices: indices where goal should be placed
        
    Returns:
        full_state: (state_size,) state vector with goal at goal_indices
    """
    full_state = jnp.zeros((state_size,), dtype=goal.dtype)
    return full_state.at[goal_indices].set(goal)


def zero_out_non_goal_indices(state, goal_indices):
    """Zero out everything except goal indices in a state.
    
    Args:
        state: (state_dim,) state vector
        goal_indices: indices to preserve
        
    Returns:
        zeroed_state: (state_dim,) state with only goal indices non-zero
    """
    zeroed_state = jnp.zeros_like(state)
    return zeroed_state.at[goal_indices].set(state[goal_indices])


# ============================================================================
# Q-Value Computation Utilities
# ============================================================================

def compute_q_value_single_critic(
    sa_params, g_params, sa_pairs, goals, sa_encoder, g_encoder, energy_fn_name
):
    """Compute Q-values for a single critic.
    
    Args:
        sa_params: State-action encoder parameters
        g_params: Goal encoder parameters
        sa_pairs: (num_pairs, state_dim + action_dim) state-action pairs
        goals: (num_pairs, goal_dim) goals
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function to use
        
    Returns:
        q_values: (num_pairs,) Q-values
    """
    phi_sa = sa_encoder.apply(sa_params, sa_pairs)
    psi_g = g_encoder.apply(g_params, goals)
    q_values = energy_fn(energy_fn_name, phi_sa, psi_g)
    return q_values


def stack_ensemble_params(critic_params):
    """Stack ensemble parameters into arrays for vectorized computation.
    
    Args:
        critic_params: Dictionary with 'sa_encoder' and 'g_encoder' lists
        
    Returns:
        stacked_sa_params: Stacked sa_encoder parameters
        stacked_g_params: Stacked g_encoder parameters
    """
    stacked_sa_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *critic_params['sa_encoder']
    )
    stacked_g_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *critic_params['g_encoder']
    )
    return stacked_sa_params, stacked_g_params


def compute_q_values_ensemble(
    states, goals, actor, actor_params, stacked_sa_params, stacked_g_params,
    sa_encoder, g_encoder, energy_fn_name, expand_goals=False, state_size=None, goal_indices=None
):
    """Compute Q-values across ensemble for state-goal pairs.
    
    Args:
        states: (num_pairs, state_dim) or (num_pairs, goal_dim) if expand_goals=True
        goals: (num_pairs, goal_dim) goal vectors
        actor: Actor network
        actor_params: Actor parameters
        stacked_sa_params: Stacked sa_encoder parameters (num_ensemble, ...)
        stacked_g_params: Stacked g_encoder parameters (num_ensemble, ...)
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function
        expand_goals: If True, states are actually goals that need expansion
        state_size: Required if expand_goals=True
        goal_indices: Required if expand_goals=True
        
    Returns:
        q_values: (num_ensemble, num_pairs) Q-values across ensemble
    """
    num_pairs = states.shape[0]
    
    # Expand goals to states if needed
    if expand_goals:
        full_states = jax.vmap(
            lambda g: expand_goal_to_state(g, state_size, goal_indices)
        )(states)
    else:
        full_states = states
    
    # Create observations
    obs = jnp.concatenate([full_states, goals], axis=1)  # (num_pairs, obs_dim)
    
    # Sample actions from policy
    means, log_stds = actor.apply(actor_params, obs)
    actions = jnp.tanh(means)  # (num_pairs, action_dim)
    
    # Compute state-action pairs
    sa_pairs = jnp.concatenate([full_states, actions], axis=1)
    
    # Compute Q-values for all ensemble members
    all_q_values = jax.vmap(
        lambda sa_p, g_p: compute_q_value_single_critic(
            sa_p, g_p, sa_pairs, goals, sa_encoder, g_encoder, energy_fn_name
        )
    )(stacked_sa_params, stacked_g_params)  # (num_ensemble, num_pairs)
    
    return all_q_values


def compute_v_and_sigma_ensemble(
    state, goal, actor, actor_params, stacked_sa_params, stacked_g_params,
    sa_encoder, g_encoder, energy_fn_name, is_goal_as_state=False,
    state_size=None, goal_indices=None
):
    """Compute mean and std of Q-values across ensemble for a state-goal pair.
    
    Args:
        state: (state_dim,) or (goal_dim,) if is_goal_as_state=True
        goal: (goal_dim,) goal vector
        actor: Actor network
        actor_params: Actor parameters
        stacked_sa_params: Stacked sa_encoder parameters
        stacked_g_params: Stacked g_encoder parameters
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function
        is_goal_as_state: If True, state is actually a goal that needs expansion
        state_size: Required if is_goal_as_state=True
        goal_indices: Required if is_goal_as_state=True
        
    Returns:
        v_mean: scalar, mean Q-value across ensemble
        sigma: scalar, std Q-value across ensemble
    """
    all_q_values = compute_q_values_ensemble(
        state[None, :], goal[None, :], actor, actor_params,
        stacked_sa_params, stacked_g_params, sa_encoder, g_encoder,
        energy_fn_name, expand_goals=is_goal_as_state,
        state_size=state_size, goal_indices=goal_indices
    )  # (num_ensemble, 1)
    
    v_mean = jnp.mean(all_q_values)
    sigma = jnp.std(all_q_values)
    return v_mean, sigma


def compute_energy_for_state_goal_pairs(
    states, goals, actor, actor_params, critic_params,
    sa_encoder, g_encoder, energy_fn_name
):
    """Compute energy/Q-values for state-goal pairs using single critic.
    
    Args:
        states: (num_pairs, state_dim) state vectors
        goals: (num_pairs, goal_dim) goal vectors
        actor: Actor network
        actor_params: Actor parameters
        critic_params: Critic parameters (single critic, not ensemble)
        sa_encoder: State-action encoder network
        g_encoder: Goal encoder network
        energy_fn_name: Name of energy function
        
    Returns:
        energies: (num_pairs,) energy/Q-values
    """
    num_pairs = states.shape[0]
    
    # Create observations
    obs = jnp.concatenate([states, goals], axis=1)  # (num_pairs, obs_dim)
    
    # Sample actions from policy
    means, log_stds = actor.apply(actor_params, obs)
    actions = jnp.tanh(means)  # (num_pairs, action_dim)
    
    # Compute state-action pairs
    sa_pairs = jnp.concatenate([states, actions], axis=1)
    
    # Compute Q-values
    phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pairs)
    psi_g = g_encoder.apply(critic_params['g_encoder'], goals)
    energies = energy_fn(energy_fn_name, phi_sa, psi_g)
    
    return energies


# ============================================================================
# Visualization Utilities
# ============================================================================

def create_goal_scatter_plot(
    candidate_goals, current_states, selected_indices, goal_indices,
    title_prefix="Goal Selection", x_bounds=None, y_bounds=None,
    color_by_values=None, color_label="Value"
):
    """Create a scatter plot visualization of goal selection.
    
    Args:
        candidate_goals: (num_candidates, goal_dim) candidate goals
        current_states: (batch_size, state_dim) current states
        selected_indices: (batch_size,) indices of selected goals
        goal_indices: indices to extract goal dimensions
        title_prefix: Prefix for plot title
        x_bounds: Optional x-axis bounds
        y_bounds: Optional y-axis bounds
        color_by_values: Optional (num_candidates,) values to color by
        color_label: Label for colorbar
        
    Returns:
        pil_image: PIL Image of the plot
    """
    batch_size = current_states.shape[0]
    num_plots = min(4, batch_size)
    random_state_indices = np.random.choice(batch_size, size=num_plots, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    current_goals = current_states[:, goal_indices]  # (batch_size, goal_dim)
    
    for plot_idx, state_idx in enumerate(random_state_indices):
        ax = axes[plot_idx]
        
        selected_goal_idx = selected_indices[state_idx]
        current_goal = current_goals[state_idx]
        selected_goal = candidate_goals[selected_goal_idx]
        
        # Color by values if provided
        if color_by_values is not None:
            scatter = ax.scatter(
                candidate_goals[:, 0], candidate_goals[:, 1],
                c=color_by_values, cmap='hot', s=150, alpha=0.8,
                edgecolors='black', linewidths=0.5, label='Candidate Goals'
            )
            plt.colorbar(scatter, ax=ax, label=color_label)
        else:
            ax.scatter(
                candidate_goals[:, 0], candidate_goals[:, 1],
                c='gray', alpha=0.3, s=50, label='Candidate Goals'
            )
        
        # Plot current state
        ax.scatter(
            current_goal[0], current_goal[1],
            c='cyan', s=400, marker='*',
            edgecolors='black', linewidths=2, zorder=5, label='Current State'
        )
        
        # Plot selected goal
        ax.scatter(
            selected_goal[0], selected_goal[1],
            c='red', s=200, marker='o',
            edgecolors='black', linewidths=2, zorder=4, label='Selected Goal'
        )
        
        ax.set_title(f'{title_prefix} - State {state_idx}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        if candidate_goals.shape[1] >= 2:
            ax.set_aspect('equal', adjustable='box')
        if x_bounds is not None:
            ax.set_xlim(x_bounds)
        if y_bounds is not None:
            ax.set_ylim(y_bounds)
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save to buffer and convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def log_visualization_with_interval(
    log_fn, *args, log_interval_steps, env_steps, last_logged_at_dict, key
):
    """Log visualization only at specified intervals.
    
    Args:
        log_fn: Function to call for logging
        *args: Arguments to pass to log_fn
        log_interval_steps: Steps between logs
        env_steps: Current environment steps
        last_logged_at_dict: Dictionary to track last log time (keyed by 'key')
        key: Key for tracking in dictionary
    """
    current_step = int(env_steps)
    last_logged = last_logged_at_dict.get(key, -500000)
    
    if current_step - last_logged >= log_interval_steps:
        log_fn(*args)
        last_logged_at_dict[key] = current_step


# ============================================================================
# Density Estimation Utilities
# ============================================================================

def gaussian_kernel_density(x, data, bandwidth):
    """Compute Gaussian kernel density estimate.
    
    Args:
        x: (n, d) points to evaluate density at
        data: (m, d) data points
        bandwidth: kernel bandwidth
        
    Returns:
        densities: (n,) density estimates
    """
    # Compute pairwise squared distances: ||x_i - data_j||^2
    diffs = x[:, None, :] - data[None, :, :]  # (n, m, d)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)  # (n, m)
    
    # Gaussian kernel: exp(-||x - data||^2 / (2 * bandwidth^2))
    kernel_vals = jnp.exp(-sq_dists / (2 * bandwidth ** 2))
    
    # Normalize by number of data points and bandwidth
    d = x.shape[-1]
    norm_const = (2 * jnp.pi * bandwidth ** 2) ** (d / 2)
    densities = jnp.mean(kernel_vals, axis=1) / norm_const
    
    return densities


def estimate_log_density_knn(goals_batch):
    """Estimate log p(s,g) using k-NN density estimation.
    
    Args:
        goals_batch: (n, d) goal samples
        
    Returns:
        log_densities: (n,) log density estimates
    """
    distances = jnp.sqrt(jnp.sum((goals_batch[:, None, :] - goals_batch[None, :, :]) ** 2, axis=2))
    
    # Get k-th nearest neighbor distance for each point
    k = int(np.sqrt(goals_batch.shape[0]))
    sorted_distances = jnp.sort(distances, axis=1)
    knn_distances = sorted_distances[:, k]  # k-th nearest neighbor distance
    
    # Density is inversely proportional to k-NN distance
    d = goals_batch.shape[1]
    log_densities = jnp.log(k / goals_batch.shape[0]) - d * jnp.log(knn_distances + 1e-10)
    
    return log_densities


def compute_kl_divergence_empirical(desired_goals, achieved_goals, bandwidth=0.1):
    """Compute empirical KL divergence D_KL(p_dg || p_ag) using KDE.
    
    Returns a large value if supports don't overlap (achieved doesn't cover desired).
    """
    # Normalize goals
    all_goals = jnp.concatenate([desired_goals, achieved_goals], axis=0)
    mean = jnp.mean(all_goals, axis=0)
    std = jnp.std(all_goals, axis=0) + 1e-6
    
    desired_normalized = (desired_goals - mean) / std
    achieved_normalized = (achieved_goals - mean) / std
    
    # Compute densities at desired goal samples
    p_desired = gaussian_kernel_density(desired_normalized, desired_normalized, bandwidth)
    p_achieved = gaussian_kernel_density(desired_normalized, achieved_normalized, bandwidth)
    
    # Add small epsilon to avoid log(0)
    p_desired = jnp.maximum(p_desired, 1e-10)
    p_achieved = jnp.maximum(p_achieved, 1e-10)
    
    # KL divergence: E[log(p_desired / p_achieved)]
    kl_div = jnp.mean(jnp.log(p_desired) - jnp.log(p_achieved))
    
    # Return large value if supports don't overlap
    kl_div = jnp.where(jnp.any(p_achieved < 1e-8), 1000.0, kl_div)
    
    return kl_div
