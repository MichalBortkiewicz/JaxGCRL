# CRL Algorithm API

This document describes how to implement custom algorithms for CRL by implementing the `Algorithm` interface.

## Overview

CRL now uses a unified training loop that follows this structure:
1. Roll out goal-conditioned policy with no noise (deterministic)
2. Roll out exploratory policy with noise
3. Update goal-conditioned policy parameters (and any additional networks)
4. Update exploratory policy parameters (and any additional networks)

To test different algorithms, simply implement the `Algorithm` interface and pass it to CRL.

## Algorithm Interface

All algorithms must inherit from `Algorithm` and implement these methods:

### Required Methods

#### `rollout_goal_conditioned_deterministic(...)`
Roll out the goal-conditioned policy deterministically (no noise).

**Parameters:**
- `training_state`: Current training state
- `env_state`: Current environment state
- `buffer_state`: Current replay buffer state
- `key`: Random key
- `env`: Environment
- `replay_buffer`: Replay buffer
- `propose_goals_fn`: Function to propose goals for new episodes
- `actor_step_fn`: Function to perform actor step
- `**kwargs`: Additional arguments

**Returns:**
- Tuple of `(new_env_state, new_buffer_state)`

#### `rollout_exploratory(...)`
Roll out the exploratory policy with noise.

**Parameters:** Same as `rollout_goal_conditioned_deterministic`

**Returns:**
- Tuple of `(new_env_state, new_buffer_state)`

#### `update_goal_conditioned_policy(...)`
Update goal-conditioned policy parameters (and any additional networks).

**Parameters:**
- `training_state`: Current training state
- `transitions`: Batch of transitions
- `networks`: Dictionary of networks (actor, sa_encoder, g_encoder, etc.)
- `context`: Dictionary of context variables (config, sizes, etc.)
- `key`: Random key
- `**kwargs`: Additional arguments

**Returns:**
- Tuple of `(new_training_state, metrics_dict)`

#### `update_exploratory_policy(...)`
Update exploratory policy parameters (and any additional networks).

**Parameters:** Same as `update_goal_conditioned_policy`

**Returns:**
- Tuple of `(new_training_state, metrics_dict)`

### Optional Methods

#### `initialize_additional_states(...)`
Initialize any additional network states needed by the algorithm.

**Parameters:**
- `key`: Random key
- `**kwargs`: Additional arguments (networks, config, etc.)

**Returns:**
- Dictionary of additional `TrainState` objects (default: empty dict)

#### `get_transitions_for_goal_conditioned_update(...)`
Filter/process transitions for goal-conditioned policy update.

**Parameters:**
- `transitions`: All transitions
- `**kwargs`: Additional arguments

**Returns:**
- Filtered/processed transitions (default: returns all transitions)

#### `get_transitions_for_exploratory_update(...)`
Filter/process transitions for exploratory policy update.

**Parameters:**
- `transitions`: All transitions
- `**kwargs`: Additional arguments

**Returns:**
- Filtered/processed transitions (default: returns all transitions)

## Example: Implementing a Custom Algorithm

```python
from jaxgcrl.agents.crl.algorithm import Algorithm
from jaxgcrl.agents.crl.crl import TrainingState, Transition
from flax.struct import dataclass
import jax
import jax.numpy as jnp

@dataclass
class MyCustomAlgorithm(Algorithm):
    """My custom algorithm implementation."""
    
    num_deterministic_steps: int = 10
    num_exploratory_steps: int = 50
    
    def rollout_goal_conditioned_deterministic(self, ...):
        # Your implementation
        return env_state, buffer_state
    
    def rollout_exploratory(self, ...):
        # Your implementation
        return env_state, buffer_state
    
    def update_goal_conditioned_policy(self, ...):
        # Your update logic
        return training_state, metrics
    
    def update_exploratory_policy(self, ...):
        # Your update logic
        return training_state, metrics
```

## Usage

```python
from jaxgcrl.agents.crl.crl import CRL
from jaxgcrl.agents.crl.algorithms import DefaultCRLAlgorithm
from my_custom_algorithm import MyCustomAlgorithm

# Option 1: Use default algorithm
crl = CRL(algorithm=DefaultCRLAlgorithm(num_deterministic_steps=0, num_exploratory_steps=0))

# Option 2: Use custom algorithm
crl = CRL(algorithm=MyCustomAlgorithm(num_deterministic_steps=10, num_exploratory_steps=50))

# Option 3: No algorithm specified (uses DefaultCRLAlgorithm automatically)
crl = CRL()
```

## Default Implementation

`DefaultCRLAlgorithm` provides a standard CRL implementation that:
- Performs standard rollouts with noise
- Uses standard CRL updates for both policies
- Can be configured with `num_deterministic_steps` and `num_exploratory_steps`

## Additional Networks

To add additional networks (e.g., separate exploratory policy, value functions), override `initialize_additional_states()` and store them in `training_state.additional_states`. Access them in your update methods as needed.
