import functools
import time
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple, Union, Generic

from absl import logging
from brax import base
from brax import envs
from brax.io import model
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.replay_buffers import QueueBase, Sample

from crl import losses as sac_losses
from crl import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax
from jax import random

# For debug purposes
CRL_TRAINING = True

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"

# Custom Replay buffer
class TrajectoryUniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue BUT WITH TRAJECTORIES."""

    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )

        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )
        batch = jnp.take(buffer_state.data, idx, axis=0, mode="wrap")

        transitions = self._unflatten_fn(batch)
        batch_keys = jax.random.split(sample_key, self._sample_batch_size)
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_to_s_a_g_fn, in_axes=(0, 0))(transitions, batch_keys)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @jax.jit
    def flatten_to_s_a_g_fn(transition: Transition, sample_key:PRNGKey) -> Transition:
        # TODO: we can take more transitions, but they will be from the same trajectory
        # TODO: move outside
        goal_key, transition_key = jax.random.split(sample_key)

        Config = namedtuple("Config", "discount obs_dim start_index end_index")
        config = Config(discount=0.99, obs_dim=10, start_index=0, end_index=9)

        seq_len = transition.observation.shape[
            0
        ]  # Because it's jitted and vmaped it's shape is (T,obs_dim)
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(
            arrangement[:, None] < arrangement[None], dtype=jnp.float32
        )
        discount = config.discount ** jnp.array(
            arrangement[None] - arrangement[:, None], dtype=jnp.float32
        )
        probs = is_future_mask * discount

        # TODO: probably use start_index and end_index if needed
        if CRL_TRAINING:
            goal_index = random.categorical(goal_key, jnp.log(probs))
            goal = transition.observation[:, 5: 8]
            goal = jnp.take(goal, goal_index[:-1], axis=0)
        else:
            goal = transition.observation[:-1, config.obs_dim :]

        state = transition.observation[:-1, : config.obs_dim]
        next_state = transition.observation[1:, : config.obs_dim]
        new_obs = jnp.concatenate([state, goal], axis=1)
        new_next_obs = jnp.concatenate([next_state, goal], axis=1)

        id_transition_to_take = random.randint(transition_key, (1,), 0, seq_len - 1)
        extras = {
            "next_action": jnp.squeeze(transition.action[1:][id_transition_to_take]),
            "policy_extras": {},
            "state_extras": {
                "truncation": jnp.squeeze(
                    transition.extras["state_extras"]["truncation"][:-1][
                        id_transition_to_take
                    ]
                )
            },
        }

        transition = Transition(
            observation=jnp.squeeze(new_obs[id_transition_to_take]),
            action=jnp.squeeze(transition.action[:-1][id_transition_to_take]),
            reward=jnp.squeeze(transition.reward[:-1][id_transition_to_take]),
            discount=jnp.squeeze(transition.discount[:-1][id_transition_to_take]),
            next_observation=jnp.squeeze(new_next_obs[id_transition_to_take]),
            extras=extras,
        )
        return transition


@flax.struct.dataclass
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
    crl_critic_params: Params
    crl_critic_optimizer_state: optax.OptState
    crl_policy_params: Params
    crl_policy_optimizer_state: optax.OptState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    crl_critics_optimizer: optax.GradientTransformation,
    crl_policy_optimizer: optax.GradientTransformation
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q, key_sa_enc, key_g_enc, key_policy_crl = jax.random.split(key, 5)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    sa_encoder_params = sac_network.sa_encoder.init(key_sa_enc)
    g_encoder_params = sac_network.g_encoder.init(key_g_enc)
    crl_critic_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
    crl_critic_state = crl_critics_optimizer.init(
        crl_critic_params
    )
    crl_policy_params = sac_network.crl_policy_network.init(key_policy_crl)
    crl_policy_state = crl_policy_optimizer.init(crl_policy_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype("float32"))
    )

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
        crl_critic_optimizer_state=crl_critic_state,
        crl_critic_params=crl_critic_params,
        crl_policy_params=crl_policy_params,
        crl_policy_optimizer_state=crl_policy_state,
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    unroll_length: int = 25,
    num_minibatches: int = 4
):
    """SAC training."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        local_devices_to_use,
        device_count,
    )

    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )

    if max_replay_size is None:
        max_replay_size = num_timesteps

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if isinstance(env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn,
            rng=jax.random.split(
                key, num_envs // jax.process_count() // local_devices_to_use
            ),
        )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    obs_size = env.observation_size
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)
    crl_critics_optimizer = optax.adam(learning_rate=3e-4)
    crl_actor_optimizer = optax.adam(learning_rate=3e-4)

    dummy_obs = jnp.zeros((unroll_length, obs_size))
    dummy_action = jnp.zeros((unroll_length, action_size))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros((unroll_length,)),
        discount=jnp.zeros((unroll_length,)),
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": jnp.zeros((unroll_length,))}, "policy_extras": {}},
    )

    replay_buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
    )

    alpha_loss, critic_loss, actor_loss, crl_critic_loss, crl_actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size,
    )
    alpha_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )
    critic_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )
    actor_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )
    crl_critic_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            crl_critic_loss, crl_critics_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )
    )
    crl_actor_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            crl_actor_loss, crl_actor_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )

    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        # TODO: separate keys for crl?
        key, key_alpha, key_critic, key_actor, key_crl_actor = jax.random.split(key, 5)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.crl_policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)

        (crl_critic_loss, metrics_crl), crl_critic_params, crl_critic_optimizer_state = crl_critic_update(
            training_state.crl_critic_params,
            training_state.normalizer_params,
            transitions,
            optimizer_state=training_state.crl_critic_optimizer_state,
        )
        crl_actor_loss, crl_policy_params, crl_policy_optimizer_state = crl_actor_update(
            training_state.crl_policy_params,
            training_state.normalizer_params,
            training_state.crl_critic_params,
            alpha,
            transitions,
            key_crl_actor,
            optimizer_state=training_state.crl_policy_optimizer_state,
        )

        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            "crl_critic_loss": crl_critic_loss,
            "crl_actor_loss": crl_actor_loss,
        }
        metrics.update(metrics_crl)

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
            crl_critic_params=crl_critic_params,
            crl_critic_optimizer_state=crl_critic_optimizer_state,
            crl_policy_params=crl_policy_params,
            crl_policy_optimizer_state=crl_policy_optimizer_state,
        )
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        crl_policy_params: Params,
        env_state: Union[envs.State, envs_v1.State],
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        Union[envs.State, envs_v1.State],
        ReplayBufferState,
    ]:
        if CRL_TRAINING:
            policy = make_policy((normalizer_params, crl_policy_params))
        else:
            policy = make_policy((normalizer_params, policy_params))

        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, data = acting.generate_unroll(
                env,
                env_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (env_state, next_key), data

        # Generate unroll_length rollouts in parallel.
        (env_state, _), data = jax.lax.scan(
            f, (env_state, key), (), length=1
        )
        # To have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        assert data.discount.shape[1:] == (unroll_length,)

        normalizer_params = running_statistics.update(
            normalizer_params, data.observation, pmap_axis_name=_PMAP_AXIS_NAME
        )
        buffer_state = replay_buffer.insert(buffer_state, data)
        return normalizer_params, env_state, buffer_state

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics
    ]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            training_state.crl_policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        # Sampling of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        transitions = jax.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                training_state.crl_policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            env_steps_per_actor_step * num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            buffer_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        crl_critics_optimizer=crl_critics_optimizer,
        crl_policy_optimizer=crl_actor_optimizer
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    if not eval_env:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
        )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    if CRL_TRAINING:
        policy = training_state.crl_policy_params
    else:
        policy = training_state.policy_params
    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, policy)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    replay_size = (
        jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Save current policy.
                params = _unpmap(
                    (training_state.normalizer_params, policy)
                )
                path = f"{checkpoint_logdir}_sac_{current_step}.pkl"
                model.save_params(path, params)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, policy)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps

    params = _unpmap((training_state.normalizer_params, policy))

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
