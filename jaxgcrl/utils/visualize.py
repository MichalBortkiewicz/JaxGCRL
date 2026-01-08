import wandb
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import io
from PIL import Image
from jaxgcrl.agents.crl.losses import energy_fn

def visualize_goals_2d(start_xy, contrastive_goals_xy, proposed_goals_xy, 
                       last_traj_states_xy, intermediate_traj_states_xy, wandb_key,
                       x_bounds=None, y_bounds=None):
    '''Visualize 2D goals and trajectories with interactive Plotly.
    - start_xy: (num_samples, 2) array of start states
    - contrastive_goals_xy: (num_samples, 2) array of contrastive goals
    - proposed_goals_xy: (num_samples, 2) array of proposed goals
    - last_traj_states_xy: (num_samples, 2) array of last trajectory states
    - intermediate_traj_states_xy: (num_samples, num_intermediate_states, 2) array of intermediate trajectory states
    - wandb_key: str, key to log the plot in WandB
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert contrastive_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert last_traj_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert intermediate_traj_states_xy.shape[2] == 2, "Goal visualization only supported for 2D goals"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories and arrows first (so points appear on top)
    for i in range(num_samples):
        # Arrow from start state to contrastive goal
        fig.add_trace(go.Scatter(
            x=[start_xy[i, 0], contrastive_goals_xy[i, 0]],
            y=[start_xy[i, 1], contrastive_goals_xy[i, 1]],
            mode='lines',
            line=dict(color='red', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Intermediate trajectory states
        fig.add_trace(go.Scatter(
            x=intermediate_traj_states_xy[i, :, 0],
            y=intermediate_traj_states_xy[i, :, 1],
            mode='markers',
            marker=dict(color='purple', size=2, opacity=0.4),
            showlegend=(i == 0),
            name='Trajectory Points' if i == 0 else '',
            hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
        ))
        
        # Full trajectory line
        full_traj_xy = np.vstack([
            start_xy[i:i+1],
            intermediate_traj_states_xy[i],
            last_traj_states_xy[i:i+1]
        ])
        
        fig.add_trace(go.Scatter(
            x=full_traj_xy[:, 0],
            y=full_traj_xy[:, 1],
            mode='lines',
            line=dict(color='purple', width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Dashed line from proposed goal to last trajectory state
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], last_traj_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], last_traj_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1.5, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot main point clouds
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0],
        y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=4, opacity=0.6),
        name='Start States',
        hovertemplate='Start State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=contrastive_goals_xy[:, 0],
        y=contrastive_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='red', size=4, opacity=0.6),
        name='Contrastive Goals',
        hovertemplate='Contrastive Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0],
        y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='orange', size=4, opacity=0.6),
        name='Proposed Goals',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=last_traj_states_xy[:, 0],
        y=last_traj_states_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=4, opacity=0.6),
        name='Reached Goal',
        hovertemplate='Reached Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Configure axis settings based on whether bounds are provided
    xaxis_config = dict(scaleanchor="y", scaleratio=1, constrain='domain')
    yaxis_config = dict(constrain='domain')
    
    if x_bounds is not None:
        xaxis_config['range'] = list(x_bounds)
    
    if y_bounds is not None:
        yaxis_config['range'] = list(y_bounds)
    
    # Update layout
    fig.update_layout(
        title="Agent Trajectories and Goal Proposals",
        xaxis_title="x",
        yaxis_title="y",
        width=2100,
        height=2100,
        hovermode='closest',
        showlegend=True,
        xaxis=xaxis_config,
        yaxis=yaxis_config
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})


def visualize_kde_heatmap(data_xy, plot_title, wandb_key, x_bounds=None, y_bounds=None):
    '''Visualize heatmap of xy data in 2D using seaborn KDE.
    - data_xy: (num_points, 2) array of xy data
    - plot_title: str, title for the plot
    - wandb_key: str, key to log the plot in WandB
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert data_xy.shape[1] == 2, "Heatmap visualization only supported for 2D goals"
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create KDE plot
    sns.kdeplot(
        x=data_xy[:, 0],
        y=data_xy[:, 1],
        fill=True,
        cmap='viridis',
        ax=ax,
        cbar=True
    )
    
    # Set bounds if provided
    if x_bounds is not None:
        ax.set_xlim(x_bounds)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{plot_title} Distribution (KDE) for {data_xy.shape[0]} points')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})

def visualize_q_function_2d(actor, sa_encoder, g_encoder, actor_params, critic_params, 
                            state, goal_indices, x_bounds, y_bounds, wandb_key, 
                            energy_fn_name, grid_resolution=50):
    '''Visualize Q-function as a heatmap over 2D goal space with policy-generated actions.
    - actor: actor network
    - sa_encoder: state-action encoder network
    - g_encoder: goal encoder network
    - actor_params: actor network parameters
    - critic_params: critic network parameters
    - state: (state_dim,) array - the state to condition on
    - goal_indices: indices for goal dimensions
    - x_bounds: tuple (min, max) for x-axis range
    - y_bounds: tuple (min, max) for y-axis range
    - wandb_key: str, key to log the plot in WandB
    - energy_fn_name: str, type of energy function ('norm', 'l2', 'dot', 'cosine')
    - grid_resolution: int, number of points per axis
    - key: JAX random key for sampling actions
    '''
    # Create grid of goal positions
    x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    goals_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_resolution^2, 2)
    num_goals = goals_grid.shape[0]
    
    # Create observations by concatenating state with each goal
    state_expanded = np.tile(state, (num_goals, 1))  # (grid_resolution^2, state_dim)
    obs_batch = np.concatenate([state_expanded, goals_grid], axis=1)  # (grid_resolution^2, obs_dim)
    
    # Sample actions from policy for each goal
    means, _ = actor.apply(actor_params, obs_batch)
    actions = jax.nn.tanh(means)  # (grid_resolution^2, action_dim)
    
    # Encode state-action pairs
    sa_pairs = np.concatenate([state_expanded, actions], axis=1)  # (grid_resolution^2, state_dim + action_dim)
    
    # Handle both single critic and ensemble cases
    sa_encoder_params = critic_params['sa_encoder']
    g_encoder_params = critic_params['g_encoder']
    if isinstance(sa_encoder_params, list):
        # Ensemble case: use first critic for visualization
        sa_encoder_params = sa_encoder_params[0]
        g_encoder_params = g_encoder_params[0]
    
    phi_sa = sa_encoder.apply(sa_encoder_params, sa_pairs)  # (grid_resolution^2, repr_dim)
    
    # Encode all goals in batch
    psi_g = g_encoder.apply(g_encoder_params, goals_grid)  # (grid_resolution^2, repr_dim)

    q_values = energy_fn(energy_fn_name, phi_sa, psi_g)
    
    # Reshape back to grid
    q_grid = q_values.reshape(grid_resolution, grid_resolution)
    
    # Use percentile-based clipping to handle outliers better
    # This makes the colormap more informative for the main range
    q_flat = q_grid.flatten()
    vmin = np.percentile(q_flat, 2)  # 2nd percentile
    vmax = np.percentile(q_flat, 98)  # 98th percentile
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(q_grid, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                   origin='lower', cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    
    # Mark the current state position
    state_goal_pos = state[goal_indices]
    ax.plot(state_goal_pos[0], state_goal_pos[1], 'r*', markersize=20, 
            label=f'Current State: ({state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f})')
    
    ax.set_xlabel('Goal x')
    ax.set_ylabel('Goal y')
    ax.set_title(f'Q-Function Landscape\nState: [{state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-value', rotation=270, labelpad=20)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})


def visualize_td3_goals_2d(start_xy, proposed_goals_xy, final_states_xy, wandb_key,
                            intermediate_xy=None, x_bounds=None, y_bounds=None):
    '''Visualize 2D goals for TD3-style goal-conditioned RL.
    Shows trajectories from start state to final achieved state, with proposed goals.
    - start_xy: (num_samples, 2) array of start states
    - proposed_goals_xy: (num_samples, 2) array of proposed goals (from replay buffer)
    - final_states_xy: (num_samples, 2) array of final achieved states
    - wandb_key: str, key to log the plot in WandB
    - intermediate_xy: optional (num_samples, num_intermediate, 2) array of intermediate trajectory states
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert final_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    if intermediate_xy is not None:
        assert intermediate_xy.shape[2] == 2, "Intermediate trajectory visualization only supported for 2D"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories first (so points appear on top)
    for i in range(num_samples):
        # If we have intermediate states, plot full trajectory line through them
        if intermediate_xy is not None:
            # Build full trajectory: start -> intermediate -> final
            full_traj_xy = np.vstack([
                start_xy[i:i+1],
                intermediate_xy[i],
                final_states_xy[i:i+1]
            ])
            
            # Full trajectory line
            fig.add_trace(go.Scatter(
                x=full_traj_xy[:, 0],
                y=full_traj_xy[:, 1],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
            
            # Intermediate trajectory points
            fig.add_trace(go.Scatter(
                x=intermediate_xy[i, :, 0],
                y=intermediate_xy[i, :, 1],
                mode='markers',
                marker=dict(color='purple', size=3, opacity=0.5),
                showlegend=(i == 0),
                name='Trajectory Points' if i == 0 else '',
                hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
        else:
            # Simple line from start state to final state
            fig.add_trace(go.Scatter(
                x=[start_xy[i, 0], final_states_xy[i, 0]],
                y=[start_xy[i, 1], final_states_xy[i, 1]],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
        
        # Dashed line from proposed goal to final state (goal-achievement gap)
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], final_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], final_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot main point clouds
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0],
        y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=6, opacity=0.7),
        name='Start States',
        hovertemplate='Start State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0],
        y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='orange', size=6, opacity=0.7),
        name='Proposed Goals',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=final_states_xy[:, 0],
        y=final_states_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=6, opacity=0.7),
        name='Achieved States',
        hovertemplate='Achieved State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Configure axis settings based on whether bounds are provided
    xaxis_config = dict(scaleanchor="y", scaleratio=1, constrain='domain')
    yaxis_config = dict(constrain='domain')
    
    if x_bounds is not None:
        xaxis_config['range'] = list(x_bounds)
    
    if y_bounds is not None:
        yaxis_config['range'] = list(y_bounds)
    
    # Update layout
    fig.update_layout(
        title="TD3 Goal Proposals and Achievements",
        xaxis_title="x",
        yaxis_title="y",
        width=2100,
        height=2100,
        hovermode='closest',
        showlegend=True,
        xaxis=xaxis_config,
        yaxis=yaxis_config
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})


def visualize_td3_q_function_2d(policy_network, q_network, normalizer_params, policy_params, q_params,
                                state, goal_indices, x_bounds, y_bounds, wandb_key, 
                                grid_resolution=100):
    '''Visualize TD3 Q-function as a heatmap over 2D goal space with policy-generated actions.
    - policy_network: TD3 policy network
    - q_network: TD3 Q network
    - normalizer_params: normalizer parameters
    - policy_params: policy network parameters
    - q_params: Q network parameters
    - state: (state_dim,) array - the state to condition on
    - goal_indices: indices for goal dimensions
    - x_bounds: tuple (min, max) for x-axis range
    - y_bounds: tuple (min, max) for y-axis range
    - wandb_key: str, key to log the plot in WandB
    - grid_resolution: int, number of points per axis
    '''
    # Create grid of goal positions
    x = np.linspace(x_bounds[0], x_bounds[1], grid_resolution)
    y = np.linspace(y_bounds[0], y_bounds[1], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    goals_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_resolution^2, 2)
    num_goals = goals_grid.shape[0]
    
    # Create observations by concatenating state with each goal
    state_expanded = np.tile(state, (num_goals, 1))  # (grid_resolution^2, state_dim)
    obs_batch = jnp.concatenate([state_expanded, goals_grid], axis=1)  # (grid_resolution^2, obs_dim)
    
    # Sample actions from policy for each goal
    actions = policy_network.apply(normalizer_params, policy_params, obs_batch)  # (grid_resolution^2, action_dim)
    
    # Compute Q-values for state-action-goal tuples
    q_values_pair = q_network.apply(normalizer_params, q_params, obs_batch, actions)  # (grid_resolution^2, 2)
    # Take minimum of twin Q-networks
    q_values = jnp.min(q_values_pair, axis=-1)  # (grid_resolution^2,)
    
    # Reshape back to grid
    q_grid = np.array(q_values.reshape(grid_resolution, grid_resolution))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(q_grid, extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                   origin='lower', cmap='viridis', aspect='equal')
    
    # Mark the current state position
    state_goal_pos = state[goal_indices]
    ax.plot(state_goal_pos[0], state_goal_pos[1], 'r*', markersize=20, 
            label=f'Current State: ({state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f})')
    
    ax.set_xlabel('Goal x')
    ax.set_ylabel('Goal y')
    ax.set_title(f'TD3 Q-Function Landscape\nState: [{state_goal_pos[0]:.2f}, {state_goal_pos[1]:.2f}]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-value', rotation=270, labelpad=20)
    
    # Save to buffer and log to WandB
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    pil_image = Image.open(buf)
    wandb.log({wandb_key: wandb.Image(pil_image)})


def visualize_sac_goals_2d(start_xy, proposed_goals_xy, final_states_xy, wandb_key,
                            intermediate_xy=None, x_bounds=None, y_bounds=None):
    '''Visualize 2D goals for SAC-style goal-conditioned RL.
    Shows trajectories from start state to final achieved state, with proposed goals.
    - start_xy: (num_samples, 2) array of start states
    - proposed_goals_xy: (num_samples, 2) array of proposed goals (from replay buffer)
    - final_states_xy: (num_samples, 2) array of final achieved states
    - wandb_key: str, key to log the plot in WandB
    - intermediate_xy: optional (num_samples, num_intermediate, 2) array of intermediate trajectory states
    - x_bounds: tuple (min, max) for x-axis range, or None for auto
    - y_bounds: tuple (min, max) for y-axis range, or None for auto
    '''
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert final_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    if intermediate_xy is not None:
        assert intermediate_xy.shape[2] == 2, "Intermediate trajectory visualization only supported for 2D"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories first (so points appear on top)
    for i in range(num_samples):
        # If we have intermediate states, plot full trajectory line through them
        if intermediate_xy is not None:
            # Build full trajectory: start -> intermediate -> final
            full_traj_xy = np.vstack([
                start_xy[i:i+1],
                intermediate_xy[i],
                final_states_xy[i:i+1]
            ])
            
            # Full trajectory line
            fig.add_trace(go.Scatter(
                x=full_traj_xy[:, 0],
                y=full_traj_xy[:, 1],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
            
            # Intermediate trajectory points
            fig.add_trace(go.Scatter(
                x=intermediate_xy[i, :, 0],
                y=intermediate_xy[i, :, 1],
                mode='markers',
                marker=dict(color='purple', size=3, opacity=0.5),
                showlegend=(i == 0),
                name='Trajectory Points' if i == 0 else '',
                hovertemplate='Intermediate<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
            ))
        else:
            # Simple line from start state to final state
            fig.add_trace(go.Scatter(
                x=[start_xy[i, 0], final_states_xy[i, 0]],
                y=[start_xy[i, 1], final_states_xy[i, 1]],
                mode='lines',
                line=dict(color='purple', width=1.5),
                opacity=0.4,
                showlegend=(i == 0),
                name='Trajectory' if i == 0 else '',
                hoverinfo='skip'
            ))
        
        # Dashed line from proposed goal to final state (goal-achievement gap)
        fig.add_trace(go.Scatter(
            x=[proposed_goals_xy[i, 0], final_states_xy[i, 0]],
            y=[proposed_goals_xy[i, 1], final_states_xy[i, 1]],
            mode='lines',
            line=dict(color='orange', width=1, dash='dash'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot start states (green circles)
    fig.add_trace(go.Scatter(
        x=start_xy[:, 0], y=start_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=10, symbol='circle'),
        name='Start State',
        hovertemplate='Start<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Plot proposed goals (blue diamonds)
    fig.add_trace(go.Scatter(
        x=proposed_goals_xy[:, 0], y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='diamond'),
        name='Proposed Goal',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    # Plot final achieved states (red stars)
    fig.add_trace(go.Scatter(
        x=final_states_xy[:, 0], y=final_states_xy[:, 1],
        mode='markers',
        marker=dict(color='red', size=14, symbol='star'),
        name='Final State',
        hovertemplate='Final State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='SAC Goal Proposals and Achieved Trajectories',
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(range=x_bounds) if x_bounds else {},
        yaxis=dict(range=y_bounds) if y_bounds else {},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='closest',
        width=800,
        height=800,
    )
    
    # Log to WandB as interactive plot
    wandb.log({wandb_key: fig})