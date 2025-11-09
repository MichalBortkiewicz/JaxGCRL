from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxgcrl.agents.crl.losses import energy_fn
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import io

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