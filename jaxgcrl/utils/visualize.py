import wandb
import numpy as np
import plotly.graph_objects as go

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