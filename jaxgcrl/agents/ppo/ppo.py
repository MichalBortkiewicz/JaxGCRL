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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.training import gradients, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


@dataclass
class PPO:
    """Proximal policy optimization (PPO) agent.
    Args:
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each train_env. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new train_env rollout
      num_resets_per_eval: the number of train_env resets to run between each
        eval. The train_env resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      deterministic_eval: whether to run the eval with a deterministic policy
      network_factory: function that generates networks for policy and value
        functions
      progress_fn: a user-defined callback function for reporting/plotting metrics
      normalize_advantage: whether to normalize advantage estimate
      restore_checkpoint_path: the path used to restore previous model params
    """

    learning_rate: float = 1e-4
    entropy_cost: float = 1e-4
    discounting: float = 0.9
    unroll_length: int = 10
    batch_size: int = 32
    num_minibatches: int = 16
    num_updates_per_batch: int = 2
    num_resets_per_eval: int = 0
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    clipping_epsilon: float = 0.3
    gae_lambda: float = 0.95
    deterministic_eval: bool = False
    normalize_advantage: bool = True
    restore_checkpoint_path: Optional[str] = None

    strategy = "on_policy"
    uses_replay_buffer = False
    train_step_multiplier = 1

    def create(self, seed, obs_size, action_size, config, env, *, key=None, devices=None):
        del env
        from jaxgcrl.training.contracts import Learner, selected_devices

        devices = selected_devices(config) if devices is None else tuple(devices)
        local_devices_to_use = len(devices)
        if key is None:
            global_key, _ = jax.random.split(jax.random.PRNGKey(seed))
            key_policy, key_value = jax.random.split(global_key)
        else:
            key_policy, key_value = key

        def normalize(x, y):
            del y
            return x

        if self.normalize_observations:
            normalize = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(
            obs_size,
            action_size,
            preprocess_observations_fn=normalize,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)

        optimizer = optax.adam(learning_rate=self.learning_rate)

        loss_fn = functools.partial(
            ppo_losses.compute_ppo_loss,
            ppo_network=ppo_network,
            entropy_cost=self.entropy_cost,
            discounting=self.discounting,
            reward_scaling=self.reward_scaling,
            gae_lambda=self.gae_lambda,
            clipping_epsilon=self.clipping_epsilon,
            normalize_advantage=self.normalize_advantage,
        )

        gradient_update_fn = gradients.gradient_update_fn(
            loss_fn,
            optimizer,
            pmap_axis_name=_PMAP_AXIS_NAME,
            has_aux=True,
        )

        def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (_, metrics), params, optimizer_state = gradient_update_fn(
                params,
                normalizer_params,
                data,
                key_loss,
                optimizer_state=optimizer_state,
            )

            return (optimizer_state, params, key), metrics

        def update_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
        ):
            del unused_t
            optimizer_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
                return x

            shuffled_data = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params),
                (optimizer_state, params, key_grad),
                shuffled_data,
                length=self.num_minibatches,
            )
            return (optimizer_state, params, key), metrics

        init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

        training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
            optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
            params=init_params,
            normalizer_params=running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32"))),
            env_steps=0,
        )
        if self.restore_checkpoint_path is not None and epath.Path(self.restore_checkpoint_path).exists():
            logging.info("restoring from checkpoint %s", self.restore_checkpoint_path)
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            target = training_state.normalizer_params, init_params
            (normalizer_params, init_params) = orbax_checkpointer.restore(
                self.restore_checkpoint_path, item=target
            )
            training_state = training_state.replace(normalizer_params=normalizer_params, params=init_params)

        training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use],
        )

        utd_ratio = self.batch_size * self.unroll_length * self.num_minibatches * config.action_repeat

        def update_rollout(carry, data):
            training_state, key = carry
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            (optimizer_state, params, key), metrics = jax.lax.scan(
                functools.partial(update_step, data=data, normalizer_params=normalizer_params),
                (training_state.optimizer_state, training_state.params, key),
                (),
                length=self.num_updates_per_batch,
            )
            state = training_state.replace(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + utd_ratio,
            )
            return (state, key), metrics

        return Learner(
            self,
            training_state,
            update_rollout,
            make_policy,
            lambda state: (state.normalizer_params, state.params.policy),
            devices,
            ppo_network,
        )

    def update(self, learner, rollout, key):
        from jaxgcrl.training.contracts import update_minibatch

        return update_minibatch(learner, rollout, key)

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
