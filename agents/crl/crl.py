import pickle
import random
import time
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from envs.wrappers import TrajectoryIdWrapper
from utils.evaluator import CrlEvaluator
from utils.replay_buffer import TrajectoryUniformSamplingQueue

from .networks import Actor, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRL:
    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    discounting: float = 0.99
    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1
    train_step_multiplier: int = 1
    use_her: bool = False
    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 256
    n_hidden: int = 2
    skip_connections: int = 4
    use_relu: bool = False
    repr_dim: int = 64
    # layer norm
    use_ln: bool = False

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert (
            config.num_envs * (config.episode_length - 1) % self.batch_size == 0
        ), "num_envs * (episode_length - 1) must be divisible by batch_size"

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        checkpoint_logdir: Optional[str] = None,
    ):

        self.check_config(config)

        unwrapped_env = train_env
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
        num_training_steps_per_epoch = (
            config.total_env_steps - num_prefill_env_steps
        ) // (config.num_evals * env_steps_per_actor_step)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key = (
            jax.random.split(key, 7)
        )

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert (
            obs_size == train_env.observation_size
        ), f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(
            sa_key, np.ones([1, state_size + action_size])
        )
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_state = TrainState.create(
            apply_fn=None,
            params={"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params},
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # Replay Buffer
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
            actions = nn.tanh(
                means
                + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            )

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

            (env_state, _), data = jax.lax.scan(
                f, (env_state, key), (), length=self.unroll_length
            )

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
        def update_actor_and_alpha(transitions, training_state, key):
            def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
                obs = (
                    transitions.observation
                )  # expected_shape = self.batch_size, obs_size + goal_size
                state = obs[:, :state_size]
                future_state = transitions.extras["future_state"]
                goal = future_state[:, train_env.goal_indices]
                observation = jnp.concatenate([state, goal], axis=1)

                means, log_stds = actor.apply(actor_params, observation)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(
                    key, shape=means.shape, dtype=means.dtype
                )
                action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
                log_prob = log_prob.sum(-1)  # dimension = B

                sa_encoder_params, g_encoder_params = (
                    critic_params["sa_encoder"],
                    critic_params["g_encoder"],
                )
                sa_repr = sa_encoder.apply(
                    sa_encoder_params, jnp.concatenate([state, action], axis=-1)
                )
                g_repr = g_encoder.apply(g_encoder_params, goal)

                qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

                actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - (qf_pi))

                return actor_loss, log_prob

            def alpha_loss(alpha_params, log_prob):
                alpha = jnp.exp(alpha_params["log_alpha"])
                alpha_loss = alpha * jnp.mean(
                    jax.lax.stop_gradient(-log_prob - target_entropy)
                )
                return jnp.mean(alpha_loss)

            (actorloss, log_prob), actor_grad = jax.value_and_grad(
                actor_loss, has_aux=True
            )(
                training_state.actor_state.params,
                training_state.critic_state.params,
                training_state.alpha_state.params["log_alpha"],
                transitions,
                key,
            )
            new_actor_state = training_state.actor_state.apply_gradients(
                grads=actor_grad
            )

            alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(
                training_state.alpha_state.params, log_prob
            )
            new_alpha_state = training_state.alpha_state.apply_gradients(
                grads=alpha_grad
            )

            training_state = training_state.replace(
                actor_state=new_actor_state, alpha_state=new_alpha_state
            )

            metrics = {
                "sample_entropy": -log_prob,
                "actor_loss": actorloss,
                "alph_aloss": alphaloss,
                "log_alpha": training_state.alpha_state.params["log_alpha"],
            }

            return training_state, metrics

        @jax.jit
        def update_critic(transitions, training_state, key):
            def critic_loss(critic_params, transitions, key):
                sa_encoder_params, g_encoder_params = (
                    critic_params["sa_encoder"],
                    critic_params["g_encoder"],
                )

                state = transitions.observation[:, :state_size]
                action = transitions.action

                sa_repr = sa_encoder.apply(
                    sa_encoder_params, jnp.concatenate([state, action], axis=-1)
                )
                g_repr = g_encoder.apply(
                    g_encoder_params, transitions.observation[:, state_size:]
                )

                # InfoNCE
                logits = -jnp.sqrt(
                    jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)
                )  # shape = BxB
                critic_loss = -jnp.mean(
                    jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)
                )

                # logsumexp regularisation
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += self.logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

                I = jnp.eye(logits.shape[0])
                correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
                logits_pos = jnp.sum(logits * I) / jnp.sum(I)
                logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

                return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg)

            (loss, (logsumexp, I, correct, logits_pos, logits_neg)), grad = (
                jax.value_and_grad(critic_loss, has_aux=True)(
                    training_state.critic_state.params, transitions, key
                )
            )
            new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
            training_state = training_state.replace(critic_state=new_critic_state)

            metrics = {
                "categorical_accuracy": jnp.mean(correct),
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "logsumexp": logsumexp.mean(),
                "critic_loss": loss,
            }

            return training_state, metrics

        @jax.jit
        def sgd_step(carry, transitions):
            training_state, key = carry
            (
                key,
                critic_key,
                actor_key,
            ) = jax.random.split(key, 3)

            training_state, actor_metrics = update_actor_and_alpha(
                transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                transitions, training_state, critic_key
            )
            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1
            )

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key = (
                jax.random.split(key, 4)
            )

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
            batch_keys = jax.random.split(
                sampling_key, transitions.observation.shape[0]
            )
            transitions = jax.vmap(
                TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0)
            )(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                transitions,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(
                experience_key2, len(transitions.observation)
            )
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
            (
                training_state,
                _,
            ), metrics = jax.lax.scan(
                sgd_step, (training_state, training_key), transitions
            )

            return (
                training_state,
                env_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    ts,
                    es,
                    bs,
                ), metrics = training_step(ts, es, bs, train_key)
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

        """Setting up evaluator"""
        evaluator = CrlEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        print("starting training....")
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

            sps = (
                env_steps_per_actor_step * num_training_steps_per_epoch
            ) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.ckpt_dir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        if config.checkpoint:
            # Save current policy and critic params.
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                training_state.critic_state.params,
            )
            path = f"{config.ckpt_dir}/final.pkl"
            save_params(path, params)
