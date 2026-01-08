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
# Import base classes and utilities from shared module
from jaxgcrl.utils.goals import (
    GoalProposer,
    ReplayBufferGoalProposal as BaseReplayBufferGoalProposal,
    mix_goals,
)

# Re-export for convenience
__all__ = ['GoalProposer', 'ReplayBufferGoalProposal', 'FisherTraceGoalProposal', 
           'MediumEnergyGoalProposal', 'MetricPreservationGoalProposal', 'QEpistemicGoalProposal', 'mix_goals']


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
    LOG_INTERVAL_STEPS: int = 500000  # Log visualizations every N environment steps
    _last_log_step: int = -500000  # Track last logged step (start negative to log first time)

    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        # Get current states from env_state
        state_size = train_env.state_dim
        current_states = env_state.obs[:, :state_size]  # (batch_size, state_dim)
        
        # Sample one batch to get candidate final states
        buffer_state, candidate_transitions = replay_buffer.sample(buffer_state)
        traj_ids = candidate_transitions.extras["state_extras"]["traj_id"]
        candidate_obs = candidate_transitions.observation
        
        def get_last_state(obs_seq, traj_id_seq):
            """Get the last state for each trajectory"""
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        last_states = jax.vmap(get_last_state)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, train_env.goal_indices]  # (batch_size, goal_size)
        
        use_critic = self.use_critic_gradients
        use_actor = self.use_actor_gradients
        
        def compute_fisher_traces_for_state(state):
            def fisher_trace_for_goal(carry, goal):
                obs = jnp.concatenate([state, goal])
                
                def get_action_and_q(actor_p):
                    """Compute action from actor and Q-value."""
                    means, log_stds = actor.apply(actor_p, obs[None, :])
                    action = jnp.tanh(means[0])
                    sa_pair = jnp.concatenate([state, action])
                    phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pair[None, :])[0]
                    psi_g = g_encoder.apply(critic_params['g_encoder'], goal[None, :])[0]
                    return energy_fn(self.energy_fn_name, phi_sa, psi_g)
                
                # Get action for critic gradients (use stop_gradient on actor params)
                means, log_stds = actor.apply(actor_params, obs[None, :])
                action = jnp.tanh(means[0])
                sa_pair = jnp.concatenate([state, action])
                
                def log_q_value(phi_params, psi_params):
                    """Energy output is already log Q-function"""
                    phi_sa = sa_encoder.apply(phi_params, sa_pair[None, :])[0]
                    psi_g = g_encoder.apply(psi_params, goal[None, :])[0]
                    return energy_fn(self.energy_fn_name, phi_sa, psi_g)
                
                total_fisher_trace = 0.0
                
                # Critic gradients (phi and psi encoders)
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
                
                # Actor gradients
                if use_actor:
                    grad_actor_params = jax.grad(get_action_and_q)(actor_params)
                    flat_grad_actor = jax.flatten_util.ravel_pytree(grad_actor_params)[0]
                    fisher_trace_actor = jnp.sum(flat_grad_actor ** 2)
                    total_fisher_trace += fisher_trace_actor
                
                return carry, total_fisher_trace
            
            # Compute Fisher trace sequentially for each candidate goal to avoid memory explosion
            _, fisher_traces = jax.lax.scan(fisher_trace_for_goal, None, candidate_goals)
            
            return fisher_traces
        
        # Compute Fisher traces for all states sequentially
        def compute_traces_for_all_states(carry, state):
            fisher_traces = compute_fisher_traces_for_state(state)
            return carry, fisher_traces
        
        _, all_fisher_traces = jax.lax.scan(compute_traces_for_all_states, None, current_states)
        
        # For each state, select the candidate goal with the largest Fisher trace
        best_goal_indices = jnp.argmax(all_fisher_traces, axis=1)  # (batch_size,)
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
    
    # Class variable to track last log step
    _last_logged_at = -500000
    
    @staticmethod
    def _log_fisher_trace_statistics(all_fisher_traces, candidate_goals, current_states, goal_indices, env_steps, log_interval_steps):
        # Only log if enough steps have passed since last log
        current_step = int(env_steps)
        if current_step - FisherTraceGoalProposal._last_logged_at < log_interval_steps:
            return
        
        FisherTraceGoalProposal._last_logged_at = current_step
            
        # all_fisher_traces: (batch_size, num_candidates)
        max_traces_per_state = jnp.max(all_fisher_traces, axis=1)  # (batch_size,)
        
        metrics = {
            'fisher_trace/max_trace_mean': float(jnp.mean(max_traces_per_state)),
            'fisher_trace/max_trace_std': float(jnp.std(max_traces_per_state)),
            'fisher_trace/max_trace_max': float(jnp.max(max_traces_per_state)),
            'fisher_trace/max_trace_min': float(jnp.min(max_traces_per_state)),
        }
        
        # Create visualization of Fisher trace maps
        pil_image = FisherTraceGoalProposal._create_fisher_trace_heatmaps(all_fisher_traces, candidate_goals, current_states, goal_indices)
        metrics['fisher_trace/trace_heatmaps'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))
    
    @staticmethod
    def _create_fisher_trace_heatmaps(all_fisher_traces, candidate_goals, current_states, goal_indices):
        batch_size = all_fisher_traces.shape[0]
        num_candidates = all_fisher_traces.shape[1]
        
        # Extract goal portion from current states
        current_goals = current_states[:, goal_indices]  # (batch_size, goal_dim)
        
        # Select 4 random states
        num_plots = min(4, batch_size)
        random_state_indices = np.random.choice(batch_size, size=num_plots, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for plot_idx, state_idx in enumerate(random_state_indices):
            ax = axes[plot_idx]
            
            fisher_traces = all_fisher_traces[state_idx]  # (num_candidates,)
            current_goal = current_goals[state_idx]  # (goal_dim,)
            
            # Color by Fisher trace value
            scatter = ax.scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                                c=fisher_traces, cmap='hot', s=150, alpha=0.8,
                                edgecolors='black', linewidths=0.5, label='Candidate Goals')
            # Plot the current state as a star
            ax.scatter(current_goal[0], current_goal[1], c='cyan', s=400, marker='*',
                        edgecolors='black', linewidths=2, zorder=5, label='Current State')
        
            plt.colorbar(scatter, ax=ax, label='Fisher Trace')
            
            max_trace_idx = int(np.argmax(fisher_traces))
            max_trace_val = float(np.max(fisher_traces))
            
            ax.set_title(f'State {state_idx}: Max Fisher Trace = {max_trace_val:.4f} (Goal {max_trace_idx})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            if candidate_goals.shape[1] >= 2:
                ax.set_aspect('equal', adjustable='box')
        
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
        
        def get_last_state(obs_seq, traj_id_seq):
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        last_states = jax.vmap(get_last_state)(candidate_obs, traj_ids)
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
            # Create observations by concatenating state with each candidate goal
            state_expanded = jnp.tile(state, (batch_size, 1))  # (batch_size, state_dim)
            obs_batch = jnp.concatenate([state_expanded, candidate_goals], axis=1)
            
            # Sample actions from policy
            means, _ = actor.apply(actor_params, obs_batch)
            actions = jnp.tanh(means)  # (batch_size, action_dim)
            
            # Compute state-action representations
            sa_pairs = jnp.concatenate([state_expanded, actions], axis=1)
            phi_sa = sa_encoder.apply(critic_params['sa_encoder'], sa_pairs)
            
            # Compute goal representations
            psi_g = g_encoder.apply(critic_params['g_encoder'], candidate_goals)
            
            # Compute energy values
            energies = energy_fn(self.energy_fn_name, phi_sa, psi_g)  # (batch_size,)
            
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
        num_plots = min(4, all_energies.shape[0])
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
        axes = axes.flatten()
        
        batch_size = all_energies.shape[1]
        num_bins = max(10, int(jnp.sqrt(batch_size)))
        
        for i in range(num_plots):
            ax = axes[i]
            energies_for_state = all_energies[i]
            selected_energy = selected_energies[i].item()

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
        
        pil_image = Image.open(buf)
        
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
    LOG_INTERVAL_STEPS: int = 500000  # Log visualizations every N environment steps

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
            
            def get_last_state(obs_seq, traj_id_seq):
                """Get the last state for each trajectory"""
                seq_len = obs_seq.shape[0]
                mask = traj_id_seq == traj_id_seq[0]
                last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
                return obs_seq[last_idx]
            
            last_states = jax.vmap(get_last_state)(candidate_obs, traj_ids)
            candidate_goals = last_states[:, train_env.goal_indices]  # (num_candidates, goal_size)
        
        num_candidates = candidate_goals.shape[0]
        num_ensemble = self.num_ensemble
        
        # Stack ensemble parameters into arrays for JAX-compatible indexing
        # This converts list of pytrees into a pytree of stacked arrays
        stacked_sa_params = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), 
            *critic_params['sa_encoder']
        )
        stacked_g_params = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), 
            *critic_params['g_encoder']
        )
        
        def compute_q_for_single_critic(sa_params, g_params, sa_pairs, goals):
            """Compute Q-values for a single critic."""
            phi_sa = sa_encoder.apply(sa_params, sa_pairs)  # (num_candidates, repr_dim)
            psi_g = g_encoder.apply(g_params, goals)  # (num_candidates, repr_dim)
            q_values = energy_fn(self.energy_fn_name, phi_sa, psi_g)  # (num_candidates,)
            return q_values
        
        def compute_q_values_for_state(state):
            """For a single state, compute Q-values across ensemble for all candidate goals.
            
            Args:
                state: (state_dim,) array
                
            Returns:
                all_q_values: (num_ensemble, num_candidates) array of Q-values
            """
            # Create observations by concatenating state with each candidate goal
            state_expanded = jnp.tile(state, (num_candidates, 1))  # (num_candidates, state_dim)
            obs_batch = jnp.concatenate([state_expanded, candidate_goals], axis=1)
            
            # Sample actions from policy
            means, log_stds = actor.apply(actor_params, obs_batch)
            actions = jnp.tanh(means)  # (num_candidates, action_dim)
            
            # Compute state-action pairs
            sa_pairs = jnp.concatenate([state_expanded, actions], axis=1)
            
            # Compute Q-values for all ensemble members using vmap over the stacked params
            # vmap over the first axis (ensemble dimension) of the stacked params
            all_q_values = jax.vmap(
                lambda sa_p, g_p: compute_q_for_single_critic(sa_p, g_p, sa_pairs, candidate_goals)
            )(stacked_sa_params, stacked_g_params)  # (num_ensemble, num_candidates)
            
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
            candidate_goals,
            current_states,
            train_env.goal_indices,
            training_state.env_steps,
            self.LOG_INTERVAL_STEPS
        )
        
        return proposed_goals, buffer_state
    
    # Class variable to track last log step
    _last_logged_at = -500000
    
    @staticmethod
    def _log_q_epistemic_statistics(all_q_stds, candidate_goals, current_states, goal_indices, env_steps, log_interval_steps):
        """Log Q-epistemic uncertainty statistics."""
        # Only log if enough steps have passed since last log
        current_step = int(env_steps)
        if current_step - QEpistemicGoalProposal._last_logged_at < log_interval_steps:
            return
        
        QEpistemicGoalProposal._last_logged_at = current_step
        
        # all_q_stds: (batch_size, num_candidates)
        max_stds_per_state = jnp.max(all_q_stds, axis=1)  # (batch_size,)
        
        metrics = {
            'q_epistemic/max_std_mean': float(jnp.mean(max_stds_per_state)),
            'q_epistemic/max_std_std': float(jnp.std(max_stds_per_state)),
            'q_epistemic/max_std_max': float(jnp.max(max_stds_per_state)),
            'q_epistemic/max_std_min': float(jnp.min(max_stds_per_state)),
            'q_epistemic/mean_std_across_candidates': float(jnp.mean(all_q_stds)),
        }
        
        # Create visualization of Q-epistemic uncertainty maps
        pil_image = QEpistemicGoalProposal._create_q_epistemic_heatmaps(
            all_q_stds, candidate_goals, current_states, goal_indices
        )
        metrics['q_epistemic/uncertainty_heatmaps'] = wandb.Image(pil_image)
        
        wandb.log(metrics, step=int(env_steps))
    
    @staticmethod
    def _create_q_epistemic_heatmaps(all_q_stds, candidate_goals, current_states, goal_indices):
        """Create heatmap visualizations of Q-epistemic uncertainty."""
        batch_size = all_q_stds.shape[0]
        
        # Extract goal portion from current states
        current_goals = current_states[:, goal_indices]  # (batch_size, goal_dim)
        
        # Select 4 random states
        num_plots = min(4, batch_size)
        random_state_indices = np.random.choice(batch_size, size=num_plots, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for plot_idx, state_idx in enumerate(random_state_indices):
            ax = axes[plot_idx]
            
            q_stds = all_q_stds[state_idx]  # (num_candidates,)
            current_goal = current_goals[state_idx]  # (goal_dim,)
            
            # Color by Q-value standard deviation
            scatter = ax.scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                                c=q_stds, cmap='hot', s=150, alpha=0.8,
                                edgecolors='black', linewidths=0.5, label='Candidate Goals')
            # Plot the current state as a star
            ax.scatter(current_goal[0], current_goal[1], c='cyan', s=400, marker='*',
                        edgecolors='black', linewidths=2, zorder=5, label='Current State')
        
            plt.colorbar(scatter, ax=ax, label='Q-value Std (Epistemic Uncertainty)')
            
            max_std_idx = int(np.argmax(q_stds))
            max_std_val = float(np.max(q_stds))
            
            ax.set_title(f'State {state_idx}: Max Q-Std = {max_std_val:.4f} (Goal {max_std_idx})',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            if candidate_goals.shape[1] >= 2:
                ax.set_aspect('equal', adjustable='box')
        
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
    LOG_INTERVAL_STEPS: int = 500000  # Log visualizations every N environment steps

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

        def get_last_state(obs_seq, traj_id_seq):
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        # expand goals to full state_dim with zero elsewhere
        def expand_goal(goal):
            # goal: (goal_dim,)
            full = jnp.zeros((state_size,), dtype=goal.dtype)
            return full.at[train_env.goal_indices].set(goal)

        last_states = jax.vmap(get_last_state)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, train_env.goal_indices]  # (num_candidate_goals, goal_dim)
        candidate_goals_full = last_states[:, :state_size] # Full vector for final states achieved

        if self.zero_out_cand_goals:
            candidate_goals_full = jax.vmap(expand_goal)(candidate_goals)

        env_goals = train_env.possible_goals  # (num_env_goals, goal_dim)

        def energy_triplet(state):
            """Compute M[g,h] for a single state and return individual terms."""
            # Optionally zero out everything except goal indices
            if self.zero_out_state:
                zeroed_state = jnp.zeros_like(state)
                state = zeroed_state.at[train_env.goal_indices].set(state[train_env.goal_indices])
            
            def estimate_log_density_knn(goals_batch):
                """Estimate log p(s,g) using k-NN density estimation."""
                # Use all candidate observations as reference samples
                distances = jnp.sqrt(jnp.sum((goals_batch[:, None, :] - goals_batch[None, :, :]) ** 2, axis=2))
                
                # Get k-th nearest neighbor distance for each point
                k = int(np.sqrt(goals_batch.shape[0]))
                sorted_distances = jnp.sort(distances, axis=1)
                knn_distances = sorted_distances[:, k]  # k-th nearest neighbor distance
                
                # Density is inversely proportional to k-NN distance
                # Using volume of hypersphere: log(k/n) - d*log(r_k); this is up to a constant, but we don't care about the factor for goal selection here
                d = goals_batch.shape[1]
                log_densities = jnp.log(k / goals_batch.shape[0]) - d * jnp.log(knn_distances + 1e-10)
                
                return log_densities
            
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

            proposed_goal_densites = estimate_log_density_knn(candidate_goals)
            
            # Compute individual terms
            term1 = f_sag[:, None]  # f(s, a1, g) - shape (num_cand, 1)
            term2 = f_gah  # f(g, a2, h) - shape (num_cand, num_env)
            term3 = f_sah[None, :]  # -f(s, a3, h) - shape (1, num_env)
            kde_term = proposed_goal_densites[:, None]  # KDE correction - shape (num_cand, 1)
            
            M = term2 - term3
            if self.use_waypoint_difficulty:
                M += term1
            if self.use_kde_correction:
                M += kde_term
            return M, term1, term2, term3, kde_term

        # compute for all states
        energy_results = jax.vmap(energy_triplet)(current_states)
        energy_mats = energy_results[0]  # (batch, num_cand, num_env)
        term1_mats = energy_results[1]  # (batch, num_cand, 1)
        term2_mats = energy_results[2]  # (batch, num_cand, num_env)
        term3_mats = energy_results[3]  # (batch, 1, num_env)
        kde_mats = energy_results[4]  # (batch, num_cand, 1)

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
            weights = jax.nn.softmax(score)
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
            weights = jax.nn.softmax(-energies_for_h)
            g_idx = jax.random.choice(rand_key_g, a=jnp.arange(M.shape[0]), p=weights)
            
            return g_idx, h_idx
        
        def select_goal_maxlogsumexp(M):
            score = jax.scipy.special.logsumexp(M, axis=1)
            weights = jax.nn.softmax(score)
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
            weights = jax.nn.softmax(energies_for_h)
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
            term1_mats,
            term2_mats,
            term3_mats,
            kde_mats,
            training_state.env_steps,
            train_env.goal_indices,
            train_env.x_bounds,
            train_env.y_bounds,
            self.LOG_INTERVAL_STEPS
        )

        return proposed_goals, buffer_state
    
    # Class variable to track last log step
    _last_logged_at = -500000
    
    @staticmethod
    def _log_goal_selection_viz(current_states, candidate_goals, env_goals, 
                              best_g_indices, best_h_indices, energy_mats, term1_mats, term2_mats, term3_mats, kde_mats,
                              env_steps, goal_indices, x_bounds, y_bounds, log_interval_steps):
        """Visualize goal selection showing trajectory from current -> candidate -> env goals."""
        
        # Only log if enough steps have passed since last log
        current_step = int(env_steps)
        if current_step - MetricPreservationGoalProposal._last_logged_at < log_interval_steps:
            return
        
        MetricPreservationGoalProposal._last_logged_at = current_step
        
        # Randomly select 4 states to use in both visualizations
        num_states = current_states.shape[0]
        random_state_indices = np.random.choice(num_states, size=min(4, num_states), replace=False)
        
        # Generate both visualizations with the same states
        pil_image1 = MetricPreservationGoalProposal._create_goal_selection_plot(
            current_states, candidate_goals, env_goals, best_g_indices, best_h_indices, energy_mats, 
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        pil_image2 = MetricPreservationGoalProposal._create_env_goal_ranking_plot(
            current_states, candidate_goals, env_goals, energy_mats, term1_mats, term2_mats, term3_mats, kde_mats,
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        
        metrics = {
            'metric_preservation/goal_selection_viz': wandb.Image(pil_image1),
            'metric_preservation/env_goal_rankings': wandb.Image(pil_image2),
        }
        
        wandb.log(metrics, step=int(env_steps))

    @staticmethod
    def _create_goal_selection_plot(current_states, candidate_goals, env_goals, 
                                    best_g_indices, best_h_indices, energy_mats, goal_indices, random_state_indices, x_bounds, y_bounds):
        """Create the main goal selection visualization (2x2 grid)."""
        num_states_to_plot = len(random_state_indices)
    
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for plot_idx in range(num_states_to_plot):
            ax = axes[plot_idx]
            
            state_idx = random_state_indices[plot_idx]
            
            current_state = current_states[state_idx][goal_indices]

            selected_candidate_idx = best_g_indices[state_idx].item()
            selected_candidate = candidate_goals[selected_candidate_idx]
            
            M = energy_mats[state_idx]
            selected_env_idx = best_h_indices[state_idx].item()
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
            
            max_energy = M[selected_candidate_idx, selected_env_idx].item()
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

    @staticmethod
    def _create_env_goal_ranking_plot(current_states, candidate_goals, env_goals,
                                        energy_mats, term1_mats, term2_mats, term3_mats, kde_mats,
                                        goal_indices, random_state_indices, x_bounds, y_bounds):
        """Create env goal ranking visualization showing M matrix and its 4 component terms (3x2 grid with 6 plots).
        
        The M matrix is composed of:
        M[g,h] = f(s,a1,g) + f(g,a2,h) - f(s,a3,h) + KDE_correction
        
        Plot 1: Full M matrix
        Plot 2: Term 1 - f(s,a1,g)
        Plot 3: Term 2 - f(g,a2,h)  
        Plot 4: Term 3 - f(s,a3,h)
        Plot 5: KDE correction - log_density(g)
        Plot 6: Environment goals ranked by max M value, with waypoints colored by f(w, g)
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        num_env_goals = env_goals.shape[0]
        
        # Select just one state and one env goal
        state_idx = random_state_indices[0]
        env_idx = np.random.choice(num_env_goals)
        
        current_state = current_states[state_idx][goal_indices]
        env_goal = env_goals[env_idx]
        
        # Get all matrices for this state
        M = energy_mats[state_idx]  # (num_candidates, num_env_goals)
        term1 = term1_mats[state_idx]  # (num_candidates, 1)
        term2 = term2_mats[state_idx]  # (num_candidates, num_env_goals)
        term3 = term3_mats[state_idx]  # (1, num_env_goals)
        kde = kde_mats[state_idx]  # (num_candidates, 1)
        
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

        # Plot 5: KDE correction - log_density(g)
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
            g_idx = best_waypoint_per_env[h_idx]
            waypoint = candidate_goals[g_idx]
            env_g = env_goals[h_idx]
            # Line color represents f(w, g) value
            f_wg_val = best_waypoint_energies[h_idx]
            axes[5].plot([waypoint[0], env_g[0]], [waypoint[1], env_g[1]], 
                        color=plt.cm.cool(float((f_wg_val - jnp.min(best_waypoint_energies)) / 
                                              (jnp.max(best_waypoint_energies) - jnp.min(best_waypoint_energies) + 1e-6))),
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