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

# Module-level dictionary to track last logged steps for interval-based logging
_last_logged_steps = {}


def create_2x2_scatter_plot(
    candidate_goals, current_states, goal_indices, values_per_state,
    selected_indices=None, title_fn=None, cmap='hot', color_label="Value",
    x_bounds=None, y_bounds=None, figsize=(14, 10)
):
    """Create a 2x2 grid scatter plot visualization.
    
    Args:
        candidate_goals: (num_candidates, goal_dim) candidate goals
        current_states: (batch_size, state_dim) current states
        goal_indices: indices to extract goal dimensions
        values_per_state: (batch_size, num_candidates) values to color by for each state
        selected_indices: Optional (batch_size,) indices of selected goals
        title_fn: Optional function(state_idx, max_val, selected_val) -> title string
        cmap: Colormap name
        color_label: Label for colorbar
        x_bounds: Optional x-axis bounds
        y_bounds: Optional y-axis bounds
        figsize: Figure size
        
    Returns:
        pil_image: PIL Image of the plot
    """
    batch_size = current_states.shape[0]
    num_plots = min(4, batch_size)
    random_state_indices = np.random.choice(batch_size, size=num_plots, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Extract goal portion from current states
    current_goals = current_states[:, goal_indices]  # (batch_size, goal_dim)
    
    for plot_idx, state_idx in enumerate(random_state_indices):
        ax = axes[plot_idx]
        
        values = values_per_state[state_idx]  # (num_candidates,)
        current_goal = current_goals[state_idx]  # (goal_dim,)
        
        # Color candidate goals by values
        scatter = ax.scatter(
            candidate_goals[:, 0], candidate_goals[:, 1],
            c=values, cmap=cmap, s=150, alpha=0.8,
            edgecolors='black', linewidths=0.5, label='Candidate Goals'
        )
        plt.colorbar(scatter, ax=ax, label=color_label)
        
        # Plot current state as a star
        ax.scatter(
            current_goal[0], current_goal[1],
            c='cyan', s=400, marker='*',
            edgecolors='black', linewidths=2, zorder=5, label='Current State'
        )
        
        # Plot selected goal if provided
        if selected_indices is not None:
            selected_goal_idx = selected_indices[state_idx]
            selected_goal = candidate_goals[selected_goal_idx]
            selected_val = float(values[selected_goal_idx])
            ax.scatter(
                selected_goal[0], selected_goal[1],
                c='red', s=200, marker='o',
                edgecolors='black', linewidths=2, zorder=4, label='Selected Goal'
            )
        else:
            selected_val = None
        
        max_val = float(jnp.max(values))
        
        # Set title
        if title_fn is not None:
            title = title_fn(state_idx, max_val, selected_val)
        else:
            title = f'State {state_idx}: Max = {max_val:.4f}'
            if selected_val is not None:
                title += f', Selected = {selected_val:.4f}'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
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


def create_energy_histogram_plot(all_energies, selected_energies, num_plots=4, figsize=(12, 8)):
    """Create histogram plots showing energy distributions.
    
    Args:
        all_energies: (batch_size, num_candidates) energy values
        selected_energies: (batch_size,) selected energy values
        num_plots: Number of states to plot
        figsize: Figure size
        
    Returns:
        pil_image: PIL Image of the plot
    """
    num_plots = min(num_plots, all_energies.shape[0])
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    batch_size = all_energies.shape[1]
    num_bins = max(10, int(np.sqrt(batch_size)))
    
    for i in range(num_plots):
        ax = axes[i]
        energies_for_state = all_energies[i]
        selected_energy = float(selected_energies[i])
        
        # Plot histogram
        ax.hist(energies_for_state, bins=num_bins, alpha=0.7, edgecolor='black')
        
        # Mark the selected energy with a vertical line
        ax.axvline(selected_energy, color='red', linestyle='--', linewidth=2, label='Selected')
        
        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title(f'State {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots if fewer than 4 states
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def should_log_at_interval(env_steps, log_interval_steps, key):
    """Check if we should log at this interval.
    
    Args:
        env_steps: Current environment steps
        log_interval_steps: Steps between logs
        key: Unique key for this logger
        
    Returns:
        should_log: Boolean indicating if we should log
    """
    current_step = int(env_steps)
    last_logged = _last_logged_steps.get(key, -500000)
    
    if current_step - last_logged >= log_interval_steps:
        _last_logged_steps[key] = current_step
        return True
    return False


def create_goal_selection_plot(
    current_states, candidate_goals, env_goals, best_g_indices, best_h_indices,
    energy_mats, goal_indices, random_state_indices, x_bounds, y_bounds, figsize=(16, 12)
):
    """Create goal selection visualization showing trajectory from current -> candidate -> env goals.
    
    Args:
        current_states: (batch_size, state_dim) current states
        candidate_goals: (num_candidates, goal_dim) candidate goals
        env_goals: (num_env_goals, goal_dim) environment goals
        best_g_indices: (batch_size,) indices of selected candidate goals
        best_h_indices: (batch_size,) indices of selected env goals
        energy_mats: (batch_size, num_candidates, num_env_goals) energy matrices
        goal_indices: indices to extract goal dimensions
        random_state_indices: (num_plots,) state indices to plot
        x_bounds: x-axis bounds
        y_bounds: y-axis bounds
        figsize: Figure size
        
    Returns:
        pil_image: PIL Image of the plot
    """
    num_states_to_plot = len(random_state_indices)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for plot_idx in range(num_states_to_plot):
        ax = axes[plot_idx]
        
        state_idx = random_state_indices[plot_idx]
        
        current_state = current_states[state_idx][goal_indices]
        
        selected_candidate_idx = int(best_g_indices[state_idx])
        selected_candidate = candidate_goals[selected_candidate_idx]
        
        M = energy_mats[state_idx]
        selected_env_idx = int(best_h_indices[state_idx])
        selected_env_goal = env_goals[selected_env_idx]
        
        ax.scatter(candidate_goals[:, 0], candidate_goals[:, 1], 
                c='gray', alpha=0.3, s=50, label='Candidate Goals (Buffer)', marker='o')
        
        ax.scatter(env_goals[:, 0], env_goals[:, 1], 
                c='blue', alpha=0.5, s=100, label='Environment Goals', marker='s')
        
        ax.scatter(current_state[0], current_state[1], 
                c='green', s=300, label='Current State', marker='*', 
                edgecolors='black', linewidths=2, zorder=5)
        
        ax.scatter(selected_candidate[0], selected_candidate[1], 
                c='red', s=200, label='Selected Candidate', marker='o',
                edgecolors='black', linewidths=2, zorder=4)
        
        ax.scatter(selected_env_goal[0], selected_env_goal[1], 
                c='purple', s=250, label='Paired Env Goal', marker='s',
                edgecolors='black', linewidths=2, zorder=4)
        
        ax.annotate('', xy=(selected_candidate[0], selected_candidate[1]),
                xytext=(current_state[0], current_state[1]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='orange', alpha=0.7))
        
        ax.annotate('', xy=(selected_env_goal[0], selected_env_goal[1]),
                xytext=(selected_candidate[0], selected_candidate[1]),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='purple', alpha=0.7))
        
        max_energy = float(energy_mats[state_idx][selected_candidate_idx, selected_env_idx])
        ax.text(0.02, 0.98, f'Max Energy: {max_energy:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'State {state_idx}: Goal Selection (Current → Candidate → Target)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_bounds)
        ax.set_ylim(y_bounds)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def create_env_goal_ranking_plot(
    current_states, candidate_goals, env_goals, energy_mats,
    term1_single, term2_single, term3_single, kde_single, viz_state_idx,
    goal_indices, x_bounds, y_bounds, figsize=(16, 16)
):
    """Create env goal ranking visualization showing M matrix and its component terms.
    
    Args:
        current_states: (batch_size, state_dim) current states
        candidate_goals: (num_candidates, goal_dim) candidate goals
        env_goals: (num_env_goals, goal_dim) environment goals
        energy_mats: (batch_size, num_candidates, num_env_goals) energy matrices
        term1_single: (num_candidates, 1) term 1 values
        term2_single: (num_candidates, num_env_goals) term 2 values
        term3_single: (1, num_env_goals) term 3 values
        kde_single: (num_candidates, 1) KDE values
        viz_state_idx: State index to visualize
        goal_indices: indices to extract goal dimensions
        x_bounds: x-axis bounds
        y_bounds: y-axis bounds
        figsize: Figure size
        
    Returns:
        pil_image: PIL Image of the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    
    num_env_goals = env_goals.shape[0]
    
    # Use the viz_state_idx that we computed terms for
    state_idx = int(viz_state_idx)
    env_idx = np.random.choice(num_env_goals)
    
    current_state = current_states[state_idx][goal_indices]
    env_goal = env_goals[env_idx]
    
    # Get M matrix for this state, use single-state term matrices
    M = energy_mats[state_idx]  # (num_candidates, num_env_goals)
    term1 = term1_single  # (num_candidates, 1)
    term2 = term2_single  # (num_candidates, num_env_goals)
    term3 = term3_single  # (1, num_env_goals)
    kde = kde_single  # (num_candidates, 1)
    
    # Extract values for this env_goal
    energies_full = M[:, env_idx]
    energies_term1 = term1[:, 0]  # Remove the singleton dimension
    energies_term2 = term2[:, env_idx]
    energies_term3 = jnp.repeat(term3[0, env_idx], M.shape[0], axis=0)  # Duplicate for all candidates
    energies_kde = kde[:, 0]  # Remove the singleton dimension
    
    # Plot 1: Full M matrix
    scatter1 = axes[0].scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                        c=energies_full, cmap='viridis', s=80, alpha=0.7,
                        edgecolors='black', linewidths=0.5)
    axes[0].scatter(env_goal[0], env_goal[1], c='red', s=400, marker='s', 
            edgecolors='black', linewidths=3, zorder=10, label=f'Env Goal {env_idx}')
    axes[0].scatter(current_state[0], current_state[1], c='green', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=9, label='Current State')
    plt.colorbar(scatter1, ax=axes[0], label='M[g, h]')
    axes[0].set_title(f'M Matrix: Full Combined Energy', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_bounds)
    axes[0].set_ylim(y_bounds)
    
    # Plot 2: Term 1 - f(s,a1,g)
    scatter2 = axes[1].scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                        c=energies_term1, cmap='plasma', s=80, alpha=0.7,
                        edgecolors='black', linewidths=0.5)
    axes[1].scatter(current_state[0], current_state[1], c='green', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=9, label='Current State')
    plt.colorbar(scatter2, ax=axes[1], label='f(s, w)')
    axes[1].set_title(f'Term 1: f(s, w)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_bounds)
    axes[1].set_ylim(y_bounds)
    
    # Plot 3: Term 2 - f(g,a2,h)
    scatter3 = axes[2].scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                        c=energies_term2, cmap='cool', s=80, alpha=0.7,
                        edgecolors='black', linewidths=0.5)
    axes[2].scatter(env_goal[0], env_goal[1], c='red', s=400, marker='s', 
            edgecolors='black', linewidths=3, zorder=10, label=f'Env Goal {env_idx}')
    plt.colorbar(scatter3, ax=axes[2], label='f(w, g)')
    axes[2].set_title(f'Term 2: f(w, g)', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_xlim(x_bounds)
    axes[2].set_ylim(y_bounds)
    
    # Plot 4: Term 3 - -f(s,a3,h)
    scatter4 = axes[3].scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                    c=energies_term3, cmap='RdBu', s=80, alpha=0.7,
                    edgecolors='black', linewidths=0.5)
    axes[3].scatter(env_goal[0], env_goal[1], c='red', s=400, marker='s', 
            edgecolors='black', linewidths=3, zorder=10, label=f'Env Goal {env_idx}')
    axes[3].scatter(current_state[0], current_state[1], c='green', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=9, label='Current State')
    plt.colorbar(scatter4, ax=axes[3], label='f(s, g)')
    axes[3].set_title(f'Term 3: f(s, g)', fontsize=12, fontweight='bold')
    axes[3].legend(loc='upper right', fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_aspect('equal', adjustable='box')
    axes[3].set_xlim(x_bounds)
    axes[3].set_ylim(y_bounds)
    
    # Plot 5: KDE correction
    scatter5 = axes[4].scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                    c=energies_kde, cmap='Spectral', s=80, alpha=0.7,
                    edgecolors='black', linewidths=0.5)
    axes[4].scatter(current_state[0], current_state[1], c='green', s=300, marker='*',
            edgecolors='black', linewidths=2, zorder=9, label='Current State')
    plt.colorbar(scatter5, ax=axes[4], label='log_density(g)')
    axes[4].set_title(f'Term 4: KDE Correction - log_density(g)', fontsize=12, fontweight='bold')
    axes[4].legend(loc='upper right', fontsize=9)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_aspect('equal', adjustable='box')
    axes[4].set_xlim(x_bounds)
    axes[4].set_ylim(y_bounds)
    
    # Plot 6: Environment goals ranked by max M value, waypoints colored by f(w, g)
    # For each env goal, find the waypoint that maximizes M
    max_m_per_env = jnp.max(M, axis=0)  # (num_env_goals,)
    best_waypoint_per_env = jnp.argmax(M, axis=0)  # (num_env_goals,)
    
    # Get f(w, g) values for the best waypoint of each env goal
    best_waypoint_energies = term2[best_waypoint_per_env, jnp.arange(num_env_goals)]
    
    scatter6 = axes[5].scatter(env_goals[:, 0], env_goals[:, 1],
                        c=max_m_per_env, cmap='plasma', s=200, alpha=0.8,
                        edgecolors='black', linewidths=1.5, label='Env Goals', marker='s')
    
    # For each env goal, draw a line to its best waypoint colored by f(w, g)
    for h_idx in range(num_env_goals):
        g_idx = int(best_waypoint_per_env[h_idx])
        waypoint = candidate_goals[g_idx]
        env_g = env_goals[h_idx]
        # Line color represents f(w, g) value
        f_wg_val = float(best_waypoint_energies[h_idx])
        min_f = float(jnp.min(best_waypoint_energies))
        max_f = float(jnp.max(best_waypoint_energies))
        normalized_f = (f_wg_val - min_f) / (max_f - min_f + 1e-6)
        axes[5].plot([waypoint[0], env_g[0]], [waypoint[1], env_g[1]], 
                    color=plt.cm.cool(normalized_f),
                    linewidth=1.5, alpha=0.6, zorder=2)
    
    # Also scatter the best waypoints for each env goal
    best_waypoints = candidate_goals[best_waypoint_per_env]
    scatter6b = axes[5].scatter(best_waypoints[:, 0], best_waypoints[:, 1],
                        c=best_waypoint_energies, cmap='cool', s=100, alpha=0.8,
                        edgecolors='red', linewidths=2, marker='o', label='Best Waypoints', zorder=4)
    
    plt.colorbar(scatter6, ax=axes[5], label='Max M[g, h]')
    axes[5].set_title(f'Env Goal Rankings: Max M value (size), Waypoint connections colored by f(w, g)', 
                     fontsize=12, fontweight='bold')
    axes[5].legend(loc='upper right', fontsize=9)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_aspect('equal', adjustable='box')
    axes[5].set_xlim(x_bounds)
    axes[5].set_ylim(y_bounds)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


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
