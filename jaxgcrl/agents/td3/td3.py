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

"""TD3 training."""

import functools
import logging
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import base, envs
from brax.io import model
from brax.training import gradients, pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.acme.types import NestedArray
from brax.training.replay_buffers_test import jit_wrap
from brax.training.types import Params, Policy, PRNGKey
from brax.v1 import envs as envs_v1
from flax.struct import dataclass

from jaxgcrl.agents.common.her import flatten_batch  # noqa: F401  (moved from this file, see her.py)
from jaxgcrl.agents.common.her import flatten_batch_with_n_step
from jaxgcrl.agents.common import rep_metrics
from jaxgcrl.agents.common import categorical
from jaxgcrl.agents.common.scaled_net import get_activation
from jaxgcrl.agents.common.train_utils import Checkpointer, gradient_update_fn_with_norm
from jaxgcrl.envs.wrappers import TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import Evaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from . import losses as losses
from . import networks as networks

Metrics = types.Metrics
# Transition = types.Transition
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


def soft_update(target_params: Params, online_params: Params, tau) -> Params:
    return jax.tree_util.tree_map(lambda x, y: (1 - tau) * x + tau * y, target_params, online_params)


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


# flatten_batch (HER) now lives in jaxgcrl/agents/common/her.py and is imported above.


@dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    target_policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    td3_network: networks.TD3Networks,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)

    policy_params = td3_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = td3_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = running_statistics.init_state(specs.Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        target_policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


@dataclass
class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

    learning_rate: float = 3e-4
    discounting: float = 0.9
    batch_size: int = 256
    normalize_observations: bool = False
    reward_scaling: float = 1.0
    tau: float = 0.005
    min_replay_size: int = 0
    max_replay_size: Optional[int] = 10000
    deterministic_eval: bool = False
    train_step_multiplier: int = 1
    unroll_length: int = 50
    policy_delay: int = 2
    noise_clip: int = 0.5
    smoothing_noise: int = 0.2
    exploration_noise: float = 0.4
    use_her: bool = False

    # ---- scaled-depth study additions (defaults reproduce unmodified JaxGCRL) ----
    network_type: Literal["legacy", "scaled"] = "legacy"
    h_dim: int = 256
    n_hidden: int = 2
    use_ln: bool = False
    use_residual: bool = False
    activation: Literal["relu", "swish"] = "relu"
    actor_n_hidden: Optional[int] = None
    her_strategy: Literal["final", "future"] = "final"
    her_ratio: float = 0.8
    her_reward_obs: Literal["next", "current"] = "next"
    # n-step TD targets inside the sampled window (1 = original 1-step TD); see common/her.py
    n_step: int = 1
    # drop time-limit steps from the critic loss, as Brax SAC does (False = original TD3 behaviour)
    mask_truncation: bool = False
    # Original code (td3.py:389 at a2740ce) soft-updates the target policy from the *online* policy's old
    # params, so the target policy tracks the online policy almost exactly. True = standard TD3 target.
    fix_target_policy: bool = False
    diag_batch: int = 64
    # representation diagnostics once per epoch (weight norms, feature rank, dormant fraction); see rep_metrics.py
    rep_metrics: bool = True
    rep_batch: int = 1024
    # Categorical critic (LAC, Kang et al. 2026; see common/categorical.py). "mse" = original TD loss.
    # LAC-like preset: --critic-loss categorical --n-step 4 --actor-n-hidden 4 --critic-head-ln
    critic_loss: Literal["mse", "categorical"] = "mse"
    num_atoms: int = 51
    v_min: Optional[float] = None  # default 0 (rewards are 0/1)
    v_max: Optional[float] = None  # default reward_scaling / (1 - discounting)
    cat_projection: Literal["scalar", "distributional"] = "scalar"
    critic_head_ln: bool = False  # LayerNorm before the critic head (LAC Eq. 9)

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
        """TD3 training."""
        process_id = jax.process_index()
        local_devices_to_use = jax.local_device_count()
        if config.max_devices_per_host is not None:
            local_devices_to_use = min(local_devices_to_use, config.max_devices_per_host)
        device_count = local_devices_to_use * jax.process_count()
        logging.info(
            "local_device_count: %s; total_device_count: %s",
            local_devices_to_use,
            device_count,
        )
        network_factory = networks.make_td3_networks

        if self.min_replay_size >= config.total_env_steps:
            raise ValueError("No training will happen because min_replay_size >= total_env_steps")

        if self.max_replay_size is None:
            max_replay_size = config.total_env_steps
        else:
            max_replay_size = self.max_replay_size

        # The number of environment steps executed for every `actor_step()` call.
        env_steps_per_actor_step = config.action_repeat * config.num_envs * self.unroll_length
        num_prefill_actor_steps = self.min_replay_size // self.unroll_length + 1
        logging.info("Num_prefill_actor_steps: %s", num_prefill_actor_steps)
        num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
        assert config.total_env_steps - self.min_replay_size >= 0
        num_evals_after_init = max(config.num_evals - 1, 1)
        # The number of epoch calls per training
        # equals to
        # ceil(total_env_steps - num_prefill_env_steps /
        #      (num_evals_after_init * env_steps_per_actor_step))
        num_training_steps_per_epoch = -(
            -(config.total_env_steps - num_prefill_env_steps)
            // (num_evals_after_init * env_steps_per_actor_step)
        )

        logging.info("num_evals_after_init: %s", num_evals_after_init)
        logging.info("num_training_steps_per_epoch: %s", num_training_steps_per_epoch)

        assert config.num_envs % device_count == 0
        env = train_env
        if isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng = jax.random.PRNGKey(config.seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(key, self.num_envs // jax.process_count() // local_devices_to_use),
            )
        env = TrajectoryIdWrapper(env)
        env = wrap_for_training(
            env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            randomization_fn=v_randomization_fn,
        )
        unwrapped_env = train_env

        obs_size = env.observation_size
        action_size = env.action_size

        normalize_fn = lambda x, y: x
        if self.normalize_observations:
            normalize_fn = running_statistics.normalize
        cat_v_min, cat_v_max = categorical.default_support(self.v_min, self.v_max, self.discounting, self.reward_scaling)
        if self.critic_loss == "categorical" and self.network_type != "scaled":
            raise ValueError("--critic-loss categorical needs --network-type scaled")
        if self.network_type == "scaled":
            td3_network = networks.make_scaled_td3_networks(
                observation_size=obs_size,
                action_size=action_size,
                preprocess_observations_fn=normalize_fn,
                width=self.h_dim,
                depth=self.n_hidden,
                actor_depth=self.actor_n_hidden or self.n_hidden,
                use_ln=self.use_ln,
                use_residual=self.use_residual,
                activation=self.activation,
                num_atoms=self.num_atoms if self.critic_loss == "categorical" else 1,
                v_min=cat_v_min,
                v_max=cat_v_max,
                critic_head_ln=self.critic_head_ln,
            )
        else:
            td3_network = network_factory(
                observation_size=obs_size,
                action_size=action_size,
                preprocess_observations_fn=normalize_fn,
                hidden_layer_sizes=[self.h_dim] * self.n_hidden,
                activation=get_activation(self.activation),
                layer_norm=self.use_ln,
            )
        make_policy = networks.make_inference_fn(td3_network)

        policy_optimizer = optax.adam(learning_rate=self.learning_rate)
        q_optimizer = optax.adam(learning_rate=self.learning_rate)

        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))
        dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=dummy_obs,
            next_observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                },
                "policy_extras": {},
            },
        )
        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=max_replay_size // device_count,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size // device_count,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )

        critic_loss, actor_loss = losses.make_losses(
            td3_network=td3_network,
            reward_scaling=self.reward_scaling,
            discounting=self.discounting,
            smoothing=0.2,
            noise_clip=0.5,
            mask_truncation=self.mask_truncation,
        )
        if self.critic_loss == "categorical":
            critic_loss = categorical.make_td3_critic_loss(
                td3_network, self.reward_scaling, self.discounting, cat_v_min, cat_v_max, self.cat_projection,
                self.mask_truncation,
            )

        # Same math as brax gradients.gradient_update_fn; additionally returns the global gradient norm.
        critic_update = gradient_update_fn_with_norm(critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)
        actor_update = gradient_update_fn_with_norm(actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME)

        def cat_diag(target_q):
            # fraction of (scalar) TD targets outside the categorical support; empty for the MSE critic
            if self.critic_loss != "categorical":
                return {}
            out = jnp.logical_or(target_q < cat_v_min, target_q > cat_v_max)
            return {"cat_clip_frac": jnp.mean(out.astype(jnp.float32))}

        def q_diagnostics(training_state, transitions, key):
            sub = jax.tree_util.tree_map(lambda x: x[: self.diag_batch], transitions)
            q = td3_network.q_network.apply(
                training_state.normalizer_params, training_state.q_params, sub.observation, sub.action
            )
            next_actions = td3_network.policy_network.apply(
                training_state.normalizer_params, training_state.target_policy_params, sub.next_observation
            )
            noise = (jax.random.normal(key, next_actions.shape) * 0.2).clip(-0.5, 0.5)
            next_actions = (next_actions + noise).clip(-1.0, 1.0)
            next_q = td3_network.q_network.apply(
                training_state.normalizer_params, training_state.target_q_params, sub.next_observation, next_actions
            )
            target_q = sub.reward * self.reward_scaling + sub.discount * self.discounting * jnp.min(next_q, axis=-1)
            return jax.lax.stop_gradient(
                {"q_mean": jnp.mean(q), "q_max": jnp.max(q), "target_q_mean": jnp.mean(target_q), **cat_diag(target_q)}
            )

        def update_step(
            carry: Tuple[TrainingState, PRNGKey], transitions: Transition
        ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
            training_state, key = carry

            key, key_critic, key_actor = jax.random.split(key, 3)
            diag = q_diagnostics(training_state, transitions, jax.random.fold_in(key_critic, 1))

            critic_loss, q_params, q_optimizer_state, critic_grad_norm = critic_update(
                training_state.q_params,
                training_state.target_q_params,
                training_state.target_policy_params,
                training_state.normalizer_params,
                transitions,
                key_critic,
                optimizer_state=training_state.q_optimizer_state,
            )
            actor_loss, policy_params, policy_optimizer_state, actor_grad_norm = actor_update(
                training_state.policy_params,
                training_state.q_params,
                training_state.normalizer_params,
                transitions,
                optimizer_state=training_state.policy_optimizer_state,
            )

            def dont_policy_update(training_state):
                return (
                    0.0,
                    0.0,
                    training_state.policy_params,
                    training_state.policy_optimizer_state,
                    training_state.target_q_params,
                    training_state.target_policy_params,
                )

            def do_policy_update(training_state):
                actor_loss, policy_params, policy_optimizer_state, actor_grad_norm = actor_update(
                    training_state.policy_params,
                    training_state.q_params,
                    training_state.normalizer_params,
                    transitions,
                    optimizer_state=training_state.policy_optimizer_state,
                )
                new_target_q_params = soft_update(training_state.target_q_params, q_params, self.tau)
                target_policy_source = (
                    training_state.target_policy_params if self.fix_target_policy else training_state.policy_params
                )
                new_target_policy_params = soft_update(target_policy_source, policy_params, self.tau)
                return (
                    actor_loss,
                    actor_grad_norm,
                    policy_params,
                    policy_optimizer_state,
                    new_target_q_params,
                    new_target_policy_params,
                )

            update_policy = training_state.gradient_steps % self.policy_delay == 0
            (
                actor_loss,
                actor_grad_norm,
                policy_params,
                policy_optimizer_state,
                new_target_q_params,
                new_target_policy_params,
            ) = jax.lax.cond(update_policy, do_policy_update, dont_policy_update, training_state)

            metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "critic_grad_norm": critic_grad_norm,
                # averaged over all steps incl. the delayed ones, where it is 0 (as actor_loss is)
                "actor_grad_norm": actor_grad_norm,
                "reward_mean": jnp.mean(transitions.reward),
                "her_relabel_frac": jnp.mean(transitions.extras["her_relabeled"]),
                "truncation_frac": jnp.mean(transitions.extras["state_extras"]["truncation"]),
                **diag,
            }

            new_training_state = TrainingState(
                policy_optimizer_state=policy_optimizer_state,
                policy_params=policy_params,
                target_policy_params=new_target_policy_params,
                q_optimizer_state=q_optimizer_state,
                q_params=q_params,
                target_q_params=new_target_q_params,
                gradient_steps=training_state.gradient_steps + 1,
                env_steps=training_state.env_steps,
                normalizer_params=training_state.normalizer_params,
            )
            return (new_training_state, key), metrics

        def get_experience(
            normalizer_params: running_statistics.RunningStatisticsState,
            policy_params: Params,
            env_state: Union[envs.State, envs_v1.State],
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[
            running_statistics.RunningStatisticsState,
            Union[envs.State, envs_v1.State],
            ReplayBufferState,
        ]:
            policy = make_policy(
                (normalizer_params, policy_params),
                exploration_noise=self.exploration_noise,
                noise_clip=self.noise_clip,
            )

            @jax.jit
            def f(carry, unused_t):
                env_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    env,
                    env_state,
                    policy,
                    current_key,
                    extra_fields=(
                        "truncation",
                        "traj_id",
                    ),
                )
                return (env_state, next_key), transition

            (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=self.unroll_length)

            normalizer_params = running_statistics.update(
                normalizer_params,
                jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
                ).observation,  # so that batch size*unroll_length is the first dimension
                pmap_axis_name=_PMAP_AXIS_NAME,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return normalizer_params, env_state, buffer_state

        def training_step(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics]:
            experience_key, training_key = jax.random.split(key)
            normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                experience_key,
            )
            training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            training_state, buffer_state, metrics = train_steps(training_state, buffer_state, training_key)
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

        def train_steps(
            training_state: TrainingState,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
            experience_key, training_key, sampling_key = jax.random.split(key, 3)
            buffer_state, transitions = replay_buffer.sample(buffer_state)

            batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
            transitions = jax.vmap(
                flatten_batch_with_n_step,
                in_axes=(None, None, 0, 0),
            )(self, env, transitions, batch_keys)

            # Shuffle transitions and reshape them into (num_update_steps, batch_size, ...)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
                transitions,
            )
            permutation = jax.random.permutation(experience_key, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            (training_state, _), metrics = jax.lax.scan(
                update_step, (training_state, training_key), transitions
            )
            return training_state, buffer_state, metrics

        def scan_train_steps(n, ts, bs, update_key):
            def body(carry, unsued_t):
                ts, bs, update_key = carry
                new_key, update_key = jax.random.split(update_key)
                ts, bs, metrics = train_steps(ts, bs, update_key)
                return (ts, bs, new_key), metrics

            return jax.lax.scan(body, (ts, bs, update_key), (), length=n)

        def rep_metrics_fn(ts, bs, key):
            obs, act = rep_metrics.sample_states(replay_buffer, bs, key, self.rep_batch)
            norm = ts.normalizer_params
            qf, pf = td3_network.q_features, td3_network.policy_features
            m = {}
            m.update(
                rep_metrics.network_rep_metrics(
                    "rep_q", ts.q_params, None if qf is None else (lambda p, o, a: qf(norm, p, o, a)), obs, act
                )
            )
            m.update(
                rep_metrics.network_rep_metrics(
                    "rep_pi", ts.policy_params, None if pf is None else (lambda p, o: pf(norm, p, o)), obs
                )
            )
            # TD3 actor saturation: fraction of action components with |tanh(.)| > 0.995 (actor gradient ~ 0)
            m["rep_pi_tanh_sat"] = jnp.mean(jnp.abs(td3_network.policy_network.apply(norm, ts.policy_params, obs)) > 0.995)
            return jax.lax.stop_gradient(m)

        def training_epoch(
            training_state: TrainingState,
            env_state: envs.State,
            buffer_state: ReplayBufferState,
            key: PRNGKey,
        ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
            def f(carry, unused_t):
                ts, es, bs, k = carry
                k, new_key, update_key = jax.random.split(k, 3)
                ts, es, bs, metrics = training_step(ts, es, bs, k)
                (ts, bs, update_key), _ = scan_train_steps(self.train_step_multiplier - 1, ts, bs, update_key)
                return (ts, es, bs, new_key), metrics

            (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )
            if self.rep_metrics:
                metrics.update(rep_metrics_fn(training_state, buffer_state, jax.random.fold_in(key, 7)))
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
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
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
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
            td3_network=td3_network,
            policy_optimizer=policy_optimizer,
            q_optimizer=q_optimizer,
        )
        del global_key

        local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

        # Env init
        env_keys = jax.random.split(env_key, config.num_envs // jax.process_count())
        env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
        env_state = jax.vmap(env.reset)(env_keys)

        # Replay buffer init
        buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

        if not eval_env:
            eval_env = train_env
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, config.num_eval_envs)
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
                exploration_noise=0,
                noise_clip=0,
                deterministic=self.deterministic_eval,
            ),
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
            key=eval_key,
        )

        ckpt = Checkpointer(
            getattr(config, "run_dir", None),
            getattr(config, "ckpt_interval_min", 30.0),
            enabled=getattr(config, "auto_resume", False),
        )
        start_epoch = 0
        resumed = False
        if ckpt.exists():
            template = (training_state, env_state, buffer_state, local_key, evaluator._key)
            (training_state, env_state, buffer_state, local_key, evaluator._key), meta = ckpt.load(template)
            start_epoch = int(meta["epochs_done"])
            training_walltime = float(meta["training_walltime"])
            current_step = int(meta["env_steps"])
            resumed = True
            logging.info("resumed from step %d (epochs done %d)", current_step, start_epoch)

        if not resumed:
            # Run initial eval
            metrics = {}
            if process_id == 0 and config.num_evals > 1:
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    training_metrics={},
                )
                progress_fn(
                    0,
                    metrics,
                    make_policy,
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    unwrapped_env,
                )

            # Create and initialize the replay buffer.
            t = time.time()
            prefill_key, local_key = jax.random.split(local_key)
            prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
            training_state, env_state, buffer_state, _ = prefill_replay_buffer(
                training_state, env_state, buffer_state, prefill_keys
            )

            replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
            logging.info("replay size after prefill %s", replay_size)
            assert replay_size >= self.min_replay_size
            training_walltime = time.time() - t
            current_step = 0

        metrics = {}
        for eval_epoch_num in range(start_epoch, num_evals_after_init):
            logging.info("step %s", current_step)

            # Optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            # Eval and logging
            if process_id == 0:
                if config.checkpoint_logdir:
                    # Save current policy.
                    params = _unpmap((training_state.normalizer_params, training_state.policy_params))
                    path = f"{config.checkpoint_logdir}_td3_{current_step}.pkl"
                    model.save_params(path, params)

                # Run evals.
                metrics = evaluator.run_evaluation(
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    training_metrics,
                )
                do_render = (eval_epoch_num % config.visualization_interval) == 0
                progress_fn(
                    current_step,
                    metrics,
                    make_policy,
                    _unpmap((training_state.normalizer_params, training_state.policy_params)),
                    unwrapped_env,
                    do_render,
                )
                ckpt.maybe_save(
                    (training_state, env_state, buffer_state, local_key, evaluator._key),
                    {
                        "epochs_done": eval_epoch_num + 1,
                        "env_steps": current_step,
                        "training_walltime": training_walltime,
                    },
                )

        total_steps = current_step
        assert total_steps >= config.total_env_steps
        ckpt.finalize(training_state, {"env_steps": current_step, "training_walltime": training_walltime})

        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # If there was no mistakes the training_state should still be identical on all
        # devices.
        pmap.assert_is_replicated(training_state)
        logging.info("total steps: %s", total_steps)
        pmap.synchronize_hosts()
        return make_policy, params, metrics
