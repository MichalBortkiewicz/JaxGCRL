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
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from brax import base, envs
from brax.training import acting, gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses, networks as ppo_networks
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

from envs.wrappers import TrajectoryIdWrapper
from utils.evaluator import Evaluator

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
    train_step_multiplier = 1

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        """PPO training.

        Args:
          train_env: the train_env to train
          eval_env: an optional train_env for eval only, defaults to `train_env`
          randomization_fn: a user-defined callback function that generates randomized environments
          progress_fn: a user-defined callback function for reporting/plotting metrics

        Returns:
          Tuple of (make_policy function, network params, metrics)
        """
        assert self.batch_size * self.num_minibatches % config.num_envs == 0
        xt = time.time()
        network_factory = ppo_networks.make_ppo_networks

        process_count = jax.process_count()
        process_id = jax.process_index()
        local_device_count = jax.local_device_count()
        local_devices_to_use = local_device_count
        if config.max_devices_per_host:
            local_devices_to_use = min(
                local_devices_to_use, config.max_devices_per_host
            )

        logging.info(
            "Device count: %d, process count: %d (id %d), local device count: %d, "
            "devices to be used count: %d",
            jax.device_count(),
            process_count,
            process_id,
            local_device_count,
            local_devices_to_use,
        )
        device_count = local_devices_to_use * process_count

        # The number of train_env steps executed for every training step.
        utd_ratio = (
            self.batch_size
            * self.unroll_length
            * self.num_minibatches
            * config.action_repeat
        )
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of training_step calls per training_epoch call.
        # equals to ceil(total_env_steps / (num_evals * utd_ratio *
        #                                 num_resets_per_eval))
        num_training_steps_per_epoch = np.ceil(
            config.total_env_steps
            / (num_evals_after_init * utd_ratio * max(self.num_resets_per_eval, 1))
        ).astype(int)

        key = jax.random.PRNGKey(config.seed)
        global_key, local_key = jax.random.split(key)
        del key
        local_key = jax.random.fold_in(local_key, process_id)
        local_key, key_env, eval_key = jax.random.split(local_key, 3)
        # key_networks should be global, so that networks are initialized the same
        # way for different processes.
        key_policy, key_value = jax.random.split(global_key)
        del global_key

        assert config.num_envs % device_count == 0

        v_randomization_fn = None
        if randomization_fn is not None:
            randomization_batch_size = config.num_envs // local_device_count
            # all devices gets the same randomization rng
            randomization_rng = jax.random.split(key_env, randomization_batch_size)
            v_randomization_fn = functools.partial(
                randomization_fn, rng=randomization_rng
            )

        if isinstance(train_env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        env = train_env
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        reset_fn = jax.jit(jax.vmap(env.reset))
        key_envs = jax.random.split(key_env, config.num_envs // process_count)
        key_envs = jnp.reshape(
            key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
        )
        env_state = reset_fn(key_envs)

        normalize = lambda x, y: x
        if self.normalize_observations:
            normalize = running_statistics.normalize
        ppo_network = network_factory(
            env_state.obs.shape[-1],
            env.action_size,
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

        def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
        ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
            training_state, state, key = carry
            update_key, key_generate_unroll, new_key = jax.random.split(key, 3)

            policy = make_policy(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                )
            )

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env,
                    current_state,
                    policy,
                    current_key,
                    self.unroll_length,
                    extra_fields=("truncation",),
                )
                return (next_state, next_key), data

            (state, _), data = jax.lax.scan(
                f,
                (state, key_generate_unroll),
                (),
                length=self.batch_size * self.num_minibatches // config.num_envs,
            )
            # Have leading dimensions (batch_size * num_minibatches, unroll_length)
            data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
            )
            assert data.discount.shape[1:] == (self.unroll_length,)

            # Update normalization params and normalize observations.
            normalizer_params = running_statistics.update(
                training_state.normalizer_params,
                data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(
                    update_step,
                    data=data,
                    normalizer_params=normalizer_params,
                ),
                (
                    training_state.optimizer_state,
                    training_state.params,
                    update_key,
                ),
                (),
                length=self.num_updates_per_batch,
            )

            new_training_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + utd_ratio,
            )
            return (new_training_state, state, new_key), metrics

        def training_epoch(
            training_state: TrainingState,
            state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            (training_state, state, _), loss_metrics = jax.lax.scan(
                training_step,
                (training_state, state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            return training_state, state, loss_metrics

        training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

        # Note that this is NOT a pure jittable method.
        def training_epoch_with_timing(
            training_state: TrainingState,
            env_state: envs.State,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, Metrics]:
            nonlocal training_walltime
            t = time.time()
            training_state, env_state = _strip_weak_type((training_state, env_state))
            result = training_epoch(training_state, env_state, key)
            training_state, env_state, metrics = _strip_weak_type(result)

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time
            sps = (
                num_training_steps_per_epoch
                * utd_ratio
                * max(self.num_resets_per_eval, 1)
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            return (
                training_state,
                env_state,
                metrics,
            )  # pytype: disable=bad-return-type  # py311-upgrade

        # Initialize model params and training state.
        init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

        training_state = (
            TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
                optimizer_state=optimizer.init(
                    init_params
                ),  # pytype: disable=wrong-arg-types  # numpy-scalars
                params=init_params,
                normalizer_params=running_statistics.init_state(
                    specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
                ),
                env_steps=0,
            )
        )

        if config.total_env_steps == 0:
            return (
                make_policy,
                (
                    training_state.normalizer_params,
                    training_state.params,
                ),
                {},
            )

        if (
            self.restore_checkpoint_path is not None
            and epath.Path(self.restore_checkpoint_path).exists()
        ):
            logging.info(
                "restoring from checkpoint %s",
                self.restore_checkpoint_path,
            )
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            target = training_state.normalizer_params, init_params
            (normalizer_params, init_params) = orbax_checkpointer.restore(
                self.restore_checkpoint_path, item=target
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params, params=init_params
            )

        training_state = jax.device_put_replicated(
            training_state,
            jax.local_devices()[:local_devices_to_use],
        )

        if not eval_env:
            eval_env = train_env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(eval_key, self.num_eval_envs),
            )

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = wrap_for_training(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )

        evaluator = Evaluator(
            eval_env,
            functools.partial(
                make_policy,
                deterministic=self.deterministic_eval,
            ),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        # Run initial eval
        metrics = {}
        if process_id == 0 and self.num_evals > 1:
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )
                ),
                training_metrics={},
            )
            logging.info(metrics)
            progress_fn(
                0,
                metrics,
                make_policy,
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )
                ),
                unwrapped_env,
            )

        training_metrics = {}
        training_walltime = 0
        current_step = 0
        for eval_epoch_num in range(num_evals_after_init):
            logging.info(
                "starting iteration %s %s",
                eval_epoch_num,
                time.time() - xt,
            )

            for _ in range(max(self.num_resets_per_eval, 1)):
                # optimization
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
                (
                    training_state,
                    env_state,
                    training_metrics,
                ) = training_epoch_with_timing(
                    training_state,
                    env_state,
                    epoch_keys,
                )
                current_step = int(_unpmap(training_state.env_steps))

                key_envs = jax.vmap(
                    lambda x, s: jax.random.split(x[0], s),
                    in_axes=(0, None),
                )(key_envs, key_envs.shape[1])
                # TODO: move extra reset logic to the AutoResetWrapper.
                env_state = (
                    reset_fn(key_envs) if self.num_resets_per_eval > 0 else env_state
                )

                if process_id == 0:
                    # Run evals.
                    metrics = evaluator.run_evaluation(
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                training_state.params.policy,
                            )
                        ),
                        training_metrics,
                    )
                    logging.info(metrics)
                    do_render = (eval_epoch_num % config.visualization_interval) == 0
                    progress_fn(
                        current_step,
                        metrics,
                        make_policy,
                        _unpmap(
                            (
                                training_state.normalizer_params,
                                training_state.params.policy,
                            )
                        ),
                        unwrapped_env,
                        do_render=do_render,
                    )

        total_steps = current_step
        assert total_steps >= config.total_env_steps

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
            )
        )
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return (make_policy, params, metrics)
