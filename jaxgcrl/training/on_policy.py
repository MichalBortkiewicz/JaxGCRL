"""On-policy rollout collection and training loop for PPO."""

import functools
import logging
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.training import acting, pmap
from brax.training.types import Metrics, PRNGKey
from brax.v1 import envs as envs_v1

from jaxgcrl.agents.ppo.ppo import TrainingState, _strip_weak_type, _unpmap
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator

from .contracts import collective_devices, selected_devices
from .services import HostServices

_PMAP_AXIS_NAME = "i"


def run(
    self,
    config,
    train_env,
    eval_env=None,
    randomization_fn=None,
    progress_fn=lambda *args, **kwargs: None,
):
    assert self.batch_size * self.num_minibatches % config.num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    devices = selected_devices(config)
    local_devices_to_use = len(devices)

    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = len(collective_devices(devices))

    # The number of train_env steps executed for every training step.
    utd_ratio = self.batch_size * self.unroll_length * self.num_minibatches * config.action_repeat
    num_evals_after_init = max(config.num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(total_env_steps / (num_evals * utd_ratio *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        config.total_env_steps / (num_evals_after_init * utd_ratio * max(self.num_resets_per_eval, 1))
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
        # all devices get the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    if isinstance(train_env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = TrajectoryIdWrapper(train_env)
    env = wrap_for_training(
        env,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        randomization_fn=v_randomization_fn,
    )
    unwrapped_env = train_env

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, config.num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    learner = self.create(
        config.seed,
        env.observation_size,
        env.action_size,
        config,
        env,
        key=(key_policy, key_value),
        devices=devices,
    )
    training_state = learner.state
    make_policy = learner.make_policy
    if config.total_env_steps == 0:
        return make_policy, learner.inference_params(), {}

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
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (self.unroll_length,)

        (new_training_state, _), metrics = learner.update_step((training_state, update_key), data)
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

    training_epoch = jax.pmap(
        training_epoch, axis_name=_PMAP_AXIS_NAME, devices=collective_devices(learner.devices)
    )

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
            num_training_steps_per_epoch * utd_ratio * max(self.num_resets_per_eval, 1)
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

    if not eval_env:
        eval_env = train_env
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn,
            rng=jax.random.split(eval_key, config.num_eval_envs),
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
    services = HostServices(evaluator, progress_fn, unwrapped_env, config.visualization_interval)

    # Run initial eval
    metrics = {}
    if process_id == 0 and config.num_evals > 1:
        init_params = _unpmap((training_state.normalizer_params, training_state.params.policy))
        metrics = services.publish(0, init_params, make_policy, init_params, {})

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
            env_state = reset_fn(key_envs) if self.num_resets_per_eval > 0 else env_state

            if process_id == 0:
                eval_params = _unpmap((training_state.normalizer_params, training_state.params.policy))
                metrics = services.publish(
                    current_step,
                    eval_params,
                    make_policy,
                    eval_params,
                    training_metrics,
                    epoch=eval_epoch_num,
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
    return make_policy, params, metrics
