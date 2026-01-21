import wandb
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import io
import logging
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


def visualize_dual_crl_trajectories_2d(
    start_xy, 
    proposed_goals_xy, 
    last_traj_states_xy, 
    intermediate_traj_states_xy,
    goal_conditioned_final_xy,
    exploratory_final_xy,
    wandb_key,
    x_bounds=None, 
    y_bounds=None
):
    """Visualize 2D trajectories for dual CRL algorithm.
    
    Plots:
    - Start states
    - Proposed goals
    - Last trajectory states (combined trajectory end)
    - Intermediate trajectory states
    - Final state of goal-conditioned rollout
    - Final state of exploratory rollout
    
    Args:
        start_xy: (num_samples, 2) array of start states
        proposed_goals_xy: (num_samples, 2) array of proposed goals
        last_traj_states_xy: (num_samples, 2) array of last trajectory states (combined)
        intermediate_traj_states_xy: (num_samples, num_intermediate_states, 2) array of intermediate trajectory states
        goal_conditioned_final_xy: (num_samples, 2) array of final states from goal-conditioned rollout
        exploratory_final_xy: (num_samples, 2) array of final states from exploratory rollout
        wandb_key: str, key to log the plot in WandB
        x_bounds: tuple (min, max) for x-axis range, or None for auto
        y_bounds: tuple (min, max) for y-axis range, or None for auto
    """
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert last_traj_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert intermediate_traj_states_xy.shape[2] == 2, "Goal visualization only supported for 2D goals"
    assert goal_conditioned_final_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert exploratory_final_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    
    fig = go.Figure()
    
    num_samples = start_xy.shape[0]
    
    # Plot trajectories first (so points appear on top)
    for i in range(num_samples):
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
        
        # Line from start to goal-conditioned final state
        fig.add_trace(go.Scatter(
            x=[start_xy[i, 0], goal_conditioned_final_xy[i, 0]],
            y=[start_xy[i, 1], goal_conditioned_final_xy[i, 1]],
            mode='lines',
            line=dict(color='cyan', width=1.5, dash='dot'),
            opacity=0.4,
            showlegend=(i == 0),
            name='Goal-Conditioned Rollout' if i == 0 else '',
            hoverinfo='skip'
        ))
        
        # Line from goal-conditioned final to exploratory final
        fig.add_trace(go.Scatter(
            x=[goal_conditioned_final_xy[i, 0], exploratory_final_xy[i, 0]],
            y=[goal_conditioned_final_xy[i, 1], exploratory_final_xy[i, 1]],
            mode='lines',
            line=dict(color='magenta', width=1.5, dash='dot'),
            opacity=0.4,
            showlegend=(i == 0),
            name='Exploratory Rollout' if i == 0 else '',
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
        name='Final State (Combined)',
        hovertemplate='Final State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=goal_conditioned_final_xy[:, 0],
        y=goal_conditioned_final_xy[:, 1],
        mode='markers',
        marker=dict(color='cyan', size=5, opacity=0.7, symbol='square'),
        name='Goal-Conditioned Final',
        hovertemplate='Goal-Conditioned Final<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=exploratory_final_xy[:, 0],
        y=exploratory_final_xy[:, 1],
        mode='markers',
        marker=dict(color='magenta', size=5, opacity=0.7, symbol='diamond'),
        name='Exploratory Final',
        hovertemplate='Exploratory Final<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
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
        title="Dual CRL Trajectories",
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
    """Create a KDE heatmap visualization of 2D data.
    
    Args:
        data_xy: (N, 2) array of 2D points
        plot_title: str, title for the plot
        wandb_key: str, key to log the plot in WandB
        x_bounds: tuple (min, max) for x-axis range, or None for auto
        y_bounds: tuple (min, max) for y-axis range, or None for auto
    """
    if len(data_xy) == 0:
        return
    
    # Convert to numpy if needed
    if isinstance(data_xy, jnp.ndarray):
        data_xy = np.array(data_xy)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set bounds if provided
    if x_bounds is not None:
        ax.set_xlim(x_bounds)
    if y_bounds is not None:
        ax.set_ylim(y_bounds)
    
    # Create KDE heatmap
    try:
        sns.kdeplot(
            x=data_xy[:, 0],
            y=data_xy[:, 1],
            fill=True,
            cmap='viridis',
            ax=ax,
            levels=20
        )
    except Exception as e:
        # Fallback to scatter if KDE fails
        ax.scatter(data_xy[:, 0], data_xy[:, 1], alpha=0.3, s=1)
        logging.warning(f"KDE plot failed, using scatter instead: {e}")
    
    ax.set_title(plot_title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    # Convert to image and log to wandb
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image)
    wandb.log({wandb_key: wandb_image})
    plt.close(fig)


def visualize_sac_goals_2d(start_xy, proposed_goals_xy, final_states_xy, wandb_key,
                            intermediate_xy=None, x_bounds=None, y_bounds=None):
    """Visualize 2D goals and trajectories for SAC algorithm.
    
    Args:
        start_xy: (num_samples, 2) array of start states
        proposed_goals_xy: (num_samples, 2) array of proposed goals
        final_states_xy: (num_samples, 2) array of final states
        wandb_key: str, key to log the plot in WandB
        intermediate_xy: (num_samples, num_intermediate_states, 2) optional array of intermediate trajectory states
        x_bounds: tuple (min, max) for x-axis range, or None for auto
        y_bounds: tuple (min, max) for y-axis range, or None for auto
    """
    assert start_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert proposed_goals_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    assert final_states_xy.shape[1] == 2, "Goal visualization only supported for 2D goals"
    
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
        x=proposed_goals_xy[:, 0],
        y=proposed_goals_xy[:, 1],
        mode='markers',
        marker=dict(color='orange', size=4, opacity=0.6),
        name='Proposed Goals',
        hovertemplate='Proposed Goal<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=final_states_xy[:, 0],
        y=final_states_xy[:, 1],
        mode='markers',
        marker=dict(color='green', size=4, opacity=0.6),
        name='Final States',
        hovertemplate='Final State<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>'
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
        title="SAC Goal Proposals and Trajectories",
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


def visualize_q_function_2d(actor, sa_encoder, g_encoder, actor_params, critic_params,
                            state, goal_indices, x_bounds, y_bounds, wandb_key, energy_fn):
    """Visualize Q-function for a given state over a 2D goal space."""
    # Create a grid of goals
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    
    grid_size = 50
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Flatten grid
    goals_flat = np.stack([X.flatten(), Y.flatten()], axis=1)  # (grid_size^2, 2)
    
    # Get state components
    state_size = state.shape[0]
    state_vec = state[:state_size]
    
    # Create observations with goals
    obs_with_goals = np.tile(state_vec, (len(goals_flat), 1))
    obs_with_goals[:, goal_indices] = goals_flat
    
    # Compute Q-values (this is a simplified version - adjust based on your actual Q-function)
    # You'll need to adapt this to your actual Q-function computation
    # For now, this is a placeholder
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    # Q_vals = ... # Compute Q-values here
    # Q_grid = Q_vals.reshape(grid_size, grid_size)
    # im = ax.imshow(Q_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='viridis')
    # plt.colorbar(im, ax=ax)
    ax.set_title("Q-Function Visualization")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Convert to image and log to wandb
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image)
    wandb.log({wandb_key: wandb_image})
    plt.close(fig)
