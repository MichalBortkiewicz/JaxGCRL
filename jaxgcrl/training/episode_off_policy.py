"""Compiled episode-boundary off-policy collection and training loop for CRL."""

import logging
import random
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from brax import envs

from jaxgcrl.agents.crl.crl import Transition, save_params
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .services import HostServices


def run(
    self,
    config,
    train_env,
    eval_env=None,
    randomization_fn=None,
    progress_fn=lambda *args, **kwargs: None,
):
    del randomization_fn
    self.check_config(config)

    unwrapped_env = train_env
    if eval_env is None:
        eval_env = train_env

    train_env = TrajectoryIdWrapper(train_env)
    train_env = envs.training.wrap(
        train_env,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
    )

    eval_env = TrajectoryIdWrapper(eval_env)
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
    )

    env_steps_per_actor_step = config.num_envs * self.unroll_length
    num_prefill_env_steps = self.min_replay_size * config.num_envs
    num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
    num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
        config.num_evals * env_steps_per_actor_step
    )

    assert num_training_steps_per_epoch > 0, "total_env_steps too small for given num_envs and episode_length"

    logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
    logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
    logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

    random.seed(config.seed)
    np.random.seed(config.seed)
    key = jax.random.PRNGKey(config.seed)
    key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = jax.random.split(key, 7)

    env_keys = jax.random.split(env_key, config.num_envs)
    env_state = jax.jit(train_env.reset)(env_keys)
    train_env.step = jax.jit(train_env.step)

    state_size = train_env.state_dim
    action_size = train_env.action_size
    obs_size = train_env.observation_size
    learner = self.create(
        config.seed,
        obs_size,
        action_size,
        config,
        train_env,
        key=(actor_key, sa_key, g_key),
    )
    training_state = learner.state
    make_policy = learner.make_policy
    actor, _, _ = learner.network
    update_networks = learner.update_step
    flatten_batch = learner.prepare_batch

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={
            "state_extras": {
                "truncation": 0.0,
                "traj_id": 0.0,
            }
        },
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=self.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=self.batch_size,
            num_envs=config.num_envs,
            episode_length=config.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
        actions = nn.tanh(means)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(actor_state, env, env_state, key, extra_fields):
        means, log_stds = actor.apply(actor_state.params, env_state.obs)
        stds = jnp.exp(log_stds)
        actions = nn.tanh(means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def get_experience(actor_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(
                actor_state,
                train_env,
                env_state,
                current_key,
                extra_fields=("truncation", "traj_id"),
            )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state.actor_state,
                env_state,
                buffer_state,
                key,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key = jax.random.split(key, 4)

        # update buffer
        env_state, buffer_state = get_experience(
            training_state.actor_state,
            env_state,
            buffer_state,
            experience_key1,
        )

        training_state = training_state.replace(
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        # sample actor-step worth of transitions
        buffer_state, transitions = replay_buffer.sample(buffer_state)

        # process transitions for training
        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
            (self.discounting, state_size, tuple(train_env.goal_indices)),
            transitions,
            batch_keys,
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
        )

        # permute transitions
        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
            transitions,
        )

        # take actor-step worth of training-step
        ((training_state, _), metrics) = jax.lax.scan(
            update_networks, (training_state, training_key), transitions
        )

        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            ((ts, es, bs), metrics) = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    evaluator = ActorEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=config.num_eval_envs,
        episode_length=config.episode_length,
        key=eval_env_key,
    )
    services = HostServices(evaluator, progress_fn, unwrapped_env, config.visualization_interval)

    training_walltime = 0
    current_step = 0
    metrics = {}
    logging.info("starting training....")
    for ne in range(config.num_evals):
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(
            training_state, env_state, buffer_state, epoch_key
        )

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        current_step = int(training_state.env_steps.item())
        logging.info("step: %d", current_step)

        metrics = services.publish(
            current_step,
            training_state,
            make_policy,
            training_state.actor_state.params,
            metrics,
            epoch=ne,
        )

        if config.checkpoint_logdir:
            periodic_params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )
            path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
            services.save(path, periodic_params, save_params)

    total_steps = current_step
    assert total_steps >= config.total_env_steps

    logging.info("total steps: %s", total_steps)
    return make_policy, training_state.actor_state.params, metrics
