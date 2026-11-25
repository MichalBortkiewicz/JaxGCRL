# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.training import gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.agents.sac import losses as sac_losses
from brax.training.types import Params, Policy, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from . import networks

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


class Transition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    next_observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={"policy_extras": policy_extras, "state_extras": state_extras},
    )


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


@functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_batch(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
    del sample_key
    if config.use_her:
        # Find truncation indexes if present
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )

        # final_step_mask.shape == (seq_len, seq_len)
        final_step_mask = (
            is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        )
        final_step_mask = jnp.logical_and(
            final_step_mask,
            transition.extras["state_extras"]["truncation"][None, :],
        )
        non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

        # If final state is not present use original goal (i.e. don't change anything)
        new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
        binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

        new_goals = (
            binary_mask[:, None] * transition.observation[new_goals_idx][:, env.goal_indices]
            + jnp.logical_not(binary_mask)[:, None]
            * transition.observation[new_goals_idx][:, env.state_dim :]
        )

        # Transform observation
        state = transition.observation[:, : env.state_dim]
        new_obs = jnp.concatenate([state, new_goals], axis=1)

        # Recalculate reward
        dist = jnp.linalg.norm(new_obs[:, env.state_dim :] - new_obs[:, env.goal_indices], axis=1)
        new_reward = jnp.array(dist < env.goal_reach_thresh, dtype=float)

        # Transform next observation
        next_state = transition.next_observation[:, : env.state_dim]
        new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

        return transition._replace(
            observation=jnp.squeeze(new_obs),
            next_observation=jnp.squeeze(new_next_obs),
            reward=jnp.squeeze(new_reward),
        )

    return transition


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class SAC:
    """Soft Actor-Critic (SAC) agent."""

    learning_rate: float = 1e-4
    discounting: float = 0.9
    batch_size: int = 256
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    # target update rate
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 2
    # layer norm
    use_ln: bool = False
    # hindsight experience replay
    use_her: bool = False

    strategy = "off_policy"
    uses_replay_buffer = True
    checkpoint_name = "sac"

    def create(self, seed, obs_size, action_size, config, env, *, key=None, devices=None):
        del env
        from jaxgcrl.training.contracts import Learner, selected_devices

        devices = selected_devices(config) if devices is None else tuple(devices)
        key = jax.random.PRNGKey(seed) if key is None else key

        def normalize_fn(x, y):
            del y
            return x

        if self.normalize_observations:
            normalize_fn = running_statistics.normalize
        sac_network = networks.make_sac_networks(
            observation_size=obs_size,
            action_size=action_size,
            preprocess_observations_fn=normalize_fn,
            layer_norm=self.use_ln,
            hidden_layer_sizes=[self.h_dim] * self.n_hidden,
        )
        make_policy = networks.make_inference_fn(sac_network)

        alpha_optimizer = optax.adam(learning_rate=3e-4)
        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
            sac_network=sac_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
            action_size=action_size,
        )
        alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
        actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )

        def update_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

            alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
                training_state.alpha_params,
                training_state.policy_params,
                training_state.normalizer_params,
                transitions,
                key_alpha,
                optimizer_state=training_state.alpha_optimizer_state,
            )
            alpha = jnp.exp(training_state.alpha_params)
            critic_loss_val, q_params, q_optimizer_state = critic_update(
                training_state.q_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_q_params,
                alpha,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )
            actor_loss_val, policy_params, policy_optimizer_state = actor_update(
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.q_params,
                alpha,
                transitions,
                key_actor,
                optimizer_state=training_state.policy_optimizer_state,
            )

            new_target_q_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                training_state.target_q_params,
                q_params,
            )

            metrics = {
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
                "alpha_loss": alpha_loss_val,
                "alpha": jnp.exp(alpha_params),
            }

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                alpha_optimizer_state=alpha_optimizer_state,
                alpha_params=alpha_params,
                normalizer_params=training_state.normalizer_params,
            )
            return (new_training_state, key), metrics

        training_state = _init_training_state(
            key,
            obs_size,
            len(devices),
            sac_network,
            alpha_optimizer,
            policy_optimizer,
            q_optimizer,
        )
        return Learner(
            self,
            training_state,
            update_step,
            make_policy,
            lambda state: (state.normalizer_params, state.policy_params),
            devices,
            sac_network,
            flatten_batch,
        )

    def update(self, learner, batch, key):
        from jaxgcrl.training.contracts import update_minibatch

        return update_minibatch(learner, batch, key)

    def sample_actions(self, learner, obs, key, deterministic=False):
        policy = learner.make_policy(learner.inference_params(), deterministic=deterministic)
        return policy(obs, key)[0]

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[..., None] = lambda *args, **kwargs: None,
    ):
        from jaxgcrl.train import train

        return train(self, train_env, eval_env, config, progress_fn, randomization_fn)

    @staticmethod
    def reset_training_env(env, keys, devices):
        from jaxgcrl.training.contracts import collective_devices

        return jax.pmap(env.reset, devices=collective_devices(devices))(keys)

    def collection_policy(self, make_policy, params):
        return make_policy(params)

    def evaluation_policy(self, make_policy):
        return functools.partial(make_policy, deterministic=self.deterministic_eval)
