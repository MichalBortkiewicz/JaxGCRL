# Go-Explore Framework

A modular framework for implementing Go-Explore style exploration algorithms in JaxGCRL.

## Overview

The Go-Explore framework provides a flexible, modular architecture for implementing exploration strategies that:
1. Use deterministic goal-conditioned policy steps first
2. Switch to exploratory policy for remaining steps
3. Support different reward functions and value functions
4. Enable easy experimentation with different components

## Architecture

### Core Components

1. **Policies** (`policies.py`):
   - `GoalConditionedPolicy`: Abstract base for goal-conditioned policies
   - `ExploratoryPolicy`: Abstract base for exploratory policies
   - `SameAsGoalConditionedPolicy`: Convenience wrapper when both policies are the same

2. **Rewards** (`rewards.py`):
   - `RewardFunction`: Abstract base for reward computation

3. **Values** (`values.py`):
   - `ValueFunction`: Abstract base for V(s) or V(s, g)
   - `QFunction`: Abstract base for Q(s, a) or Q(s, a, g)

4. **Rollout** (`rollout.py`):
   - `RolloutStrategy`: Abstract base for rollout strategies
   - `GoExploreRollout`: Implements Go-Explore style rollout with deterministic steps first

5. **Updates** (`updates.py`):
   - `PolicyUpdate`: Abstract base for policy updates (both goal-conditioned and exploratory)
   - `ValueUpdate`: Abstract base for value function updates

### CRL Integration

The framework is integrated into CRL through configuration options:

```python
use_go_explore: bool = False
go_explore_deterministic_steps: int = 0  # Number of deterministic steps
go_explore_exploratory_policy_same: bool = True  # Same policy for exploration?
go_explore_exploratory_policy_goal_conditioned: bool = True  # Does exploratory policy use goals?
```

When `use_go_explore=True`, the rollout will:
- Use deterministic (no noise) goal-conditioned policy for first `go_explore_deterministic_steps` steps
- Use exploratory policy (with noise) for remaining steps

## Usage Example

### Basic Usage

Enable Go-Explore in your CRL config:

```python
config = CRL(
    use_go_explore=True,
    go_explore_deterministic_steps=10,  # First 10 steps deterministic
    go_explore_exploratory_policy_same=True,  # Use same policy for exploration
    go_explore_exploratory_policy_goal_conditioned=True,  # Exploratory policy uses goals
    # ... other config options
)
```

### Creating Custom Components

#### Custom Exploratory Policy

```python
from jaxgcrl.agents.go_explore.policies import ExploratoryPolicy

@dataclass
class MyExploratoryPolicy(ExploratoryPolicy):
    def apply(self, params, obs, rng=None):
        # Your policy implementation
        pass
    
    def sample_action(self, params, obs, rng, deterministic=False):
        # Your action sampling
        pass
    
    def is_goal_conditioned(self):
        return False  # Or True if goal-conditioned
```

#### Custom Reward Function

```python
from jaxgcrl.agents.go_explore.rewards import RewardFunction

@dataclass
class MyRewardFunction(RewardFunction):
    def compute_reward(self, obs, action, next_obs, env_reward, done, info=None):
        # Compute reward (can use env_reward, add intrinsic rewards, etc.)
        intrinsic = self.compute_intrinsic(obs, action, next_obs, info)
        return env_reward + intrinsic
```

#### Custom Policy Update

```python
from jaxgcrl.agents.go_explore.updates import PolicyUpdate

@dataclass
class MyPolicyUpdate(PolicyUpdate):
    def update(self, policy_params, transitions, value_params=None, key=None, **kwargs):
        # Your update logic for policy
        return new_params, metrics
```

#### Custom Value Update

```python
from jaxgcrl.agents.go_explore.updates import ValueUpdate

@dataclass
class MyValueUpdate(ValueUpdate):
    def update(self, value_params, transitions, policy_params=None, key=None, **kwargs):
        # Your update logic for value function
        return new_params, metrics
```

## Implementation Details

### Rollout Logic

The rollout is implemented in `get_experience` in `crl.py`:

1. For each step in the rollout:
   - If `step_idx < go_explore_deterministic_steps`: Use deterministic goal-conditioned policy
   - Otherwise: Use exploratory policy with noise

2. The `actor_step` function now accepts:
   - `step_idx`: Current step index
   - `use_deterministic`: Whether to use deterministic actions

### Current Limitations

- Currently, the exploratory policy uses the same actor network as the goal-conditioned policy
- Training updates for both policies use the same loss functions
- Future work: Support separate networks and update functions for exploratory policy

## Extending the Framework

To add new exploration strategies:

1. **New Rollout Strategy**: Inherit from `RolloutStrategy` and implement `rollout_step`
2. **New Reward Function**: Inherit from `RewardFunction`
3. **New Policy Type**: Inherit from `GoalConditionedPolicy` or `ExploratoryPolicy`
4. **New Update Function**: Inherit from `PolicyUpdate`

The modular design allows you to mix and match components without modifying core CRL code.
