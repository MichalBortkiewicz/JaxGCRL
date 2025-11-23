from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxgcrl.agents.crl.losses import energy_fn
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io
import numpy as np

@dataclass
class GoalProposer(ABC):
    @abstractmethod
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        '''Goal proposal algorithm. This should return a (batch_size, goal_size) array of proposed goals.
        
        Args:
            replay_buffer: Replay buffer to sample from
            buffer_state: Current buffer state
            train_env: Training environment
            env_state: Current environment state (contains current observations)
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
        pass


@dataclass
class ReplayBufferGoalProposal(GoalProposer):
    def propose_goals(self, replay_buffer, buffer_state, training_state, train_env, env_state, key, actor, 
                     actor_params, critic_params, sa_encoder, g_encoder):
        buffer_state, sampled_transitions = replay_buffer.sample(buffer_state)
        traj_ids = sampled_transitions.extras["state_extras"]["traj_id"]  # (num_envs, episode_length)
        observations = sampled_transitions.observation  # (num_envs, episode_length, obs_size)
        
        def get_last_state(obs_seq, traj_id_seq):
            """Get the last state for each trajectory"""
            seq_len = obs_seq.shape[0]
            mask = traj_id_seq == traj_id_seq[0]
            last_idx = jnp.max(jnp.where(mask, jnp.arange(seq_len), 0))
            return obs_seq[last_idx]
        
        # Extract last states for each batch element
        last_states = jax.vmap(get_last_state)(observations, traj_ids)  # (batch_size, state_size)
        # Extract goal positions from these last states
        proposed_goals = last_states[:, train_env.goal_indices]  # (batch_size, goal_size)
        
        return proposed_goals, buffer_state


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
class MetricPreservationGoalProposal(GoalProposer):
    energy_fn_name: str

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

        last_states = jax.vmap(get_last_state)(candidate_obs, traj_ids)
        candidate_goals = last_states[:, train_env.goal_indices]  # (num_candidate_goals, goal_dim)

        env_goals = train_env.possible_goals  # (num_env_goals, goal_dim)

        def energy_triplet(state):
            """Compute M[g,h] for a single state."""
            # expand goals to full state_dim with zero elsewhere
            def expand_goal(goal):
                # goal: (goal_dim,)
                full = jnp.zeros((state_size,), dtype=goal.dtype)
                return full.at[train_env.goal_indices].set(goal)
            
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
            candidate_goals_full = jax.vmap(expand_goal)(candidate_goals)
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

            # combine: f(s,a1,g) + f(g,a2,h) - f(s,a3,h); except we translate to Q function
            proposed_goal_densites = estimate_log_density_knn(candidate_goals)
            M = f_sag[:, None] + f_gah - f_sah[None, :] + proposed_goal_densites[:, None]
            return M

        # compute for all states
        energy_mats = jax.vmap(energy_triplet)(current_states)  # (batch, num_cand, num_env)

        def select_goal(M):
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
            weights = jnp.mean(jnp.exp(M), axis=1)
            weights = 1 / weights
            weights = weights / jnp.sum(weights)
            g_idx = jax.random.choice(key, a=jnp.arange(M.shape[0]), p=weights)
            h_idx = jnp.argmin(M[g_idx, :])
            return g_idx, h_idx

        best_g_indices, best_h_indices = jax.vmap(select_goal_minlogsumexp)(energy_mats)  # (batch,)
        proposed_goals = candidate_goals[best_g_indices]      # (batch, goal_dim)

        jax.experimental.io_callback(
            MetricPreservationGoalProposal._log_goal_selection_viz,
            None,
            current_states,
            candidate_goals,
            env_goals,
            best_g_indices,
            best_h_indices,
            energy_mats,
            training_state.env_steps,
            train_env.goal_indices,
            train_env.x_bounds,
            train_env.y_bounds
        )

        return proposed_goals, buffer_state
    
    @staticmethod
    def _log_goal_selection_viz(current_states, candidate_goals, env_goals, 
                              best_g_indices, best_h_indices, energy_mats, env_steps, goal_indices, x_bounds, y_bounds):
        """Visualize goal selection showing trajectory from current -> candidate -> env goals."""
        
        # Randomly select 4 states to use in both visualizations
        num_states = current_states.shape[0]
        random_state_indices = np.random.choice(num_states, size=min(4, num_states), replace=False)
        
        # Generate both visualizations with the same states
        pil_image1 = MetricPreservationGoalProposal._create_goal_selection_plot(
            current_states, candidate_goals, env_goals, best_g_indices, best_h_indices, energy_mats, 
            goal_indices, random_state_indices, x_bounds, y_bounds
        )
        pil_image2 = MetricPreservationGoalProposal._create_env_goal_ranking_plot(
            current_states, candidate_goals, env_goals, energy_mats, goal_indices, 
            random_state_indices, x_bounds, y_bounds
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
                                        energy_mats, goal_indices, random_state_indices, x_bounds, y_bounds):
        """Create env goal ranking visualization showing candidate energies (2x2 grid)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        num_env_goals = env_goals.shape[0]
        
        random_env_indices = np.random.choice(num_env_goals, size=min(4, num_env_goals), replace=False)
        
        for plot_idx in range(min(4, num_env_goals)):
            ax = axes[plot_idx]
            
            state_idx = random_state_indices[plot_idx]
            env_idx = random_env_indices[plot_idx]
            
            current_state = current_states[state_idx][goal_indices]
            env_goal = env_goals[env_idx]
            M = energy_mats[state_idx]
            
            energies_for_env = M[:, env_idx]
            
            scatter = ax.scatter(candidate_goals[:, 0], candidate_goals[:, 1],
                            c=energies_for_env, cmap='viridis', s=80, alpha=0.7,
                            edgecolors='black', linewidths=0.5)
            
            ax.scatter(env_goal[0], env_goal[1], c='red', s=400, marker='s', 
                    edgecolors='black', linewidths=3, zorder=10, label=f'Env Goal {env_idx}')
            
            ax.scatter(current_state[0], current_state[1], c='green', s=300, marker='*',
                    edgecolors='black', linewidths=2, zorder=9, label='Current State')
            
            plt.colorbar(scatter, ax=ax, label='Energy M[candidate, env_goal]')
            ax.set_title(f'State {state_idx} → Env Goal {env_idx}', fontsize=12, fontweight='bold')
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