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

"""SAC networks."""

from typing import Any, Callable, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class SACNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    # feature hooks for rep_metrics (scaled networks only): fn(processor_params, params, *inputs) -> sown state
    q_features: Any = flax.struct.field(pytree_node=False, default=None)
    policy_features: Any = flax.struct.field(pytree_node=False, default=None)
    # categorical critic (num_atoms > 1): fn(processor_params, params, obs, actions) -> (B, C, I) logits; support z
    q_logits: Any = flax.struct.field(pytree_node=False, default=None)
    atoms: Any = flax.struct.field(pytree_node=False, default=None)


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    layer_norm: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                if self.layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    layer_norm=layer_norm,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a policy network."""

    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return networks.FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(observations: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return (
                sac_networks.parametric_action_distribution.sample(logits, key_sample),
                {},
            )

        return policy

    return make_policy


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# ---------------------------------------------------------------------------------------------
# Scaled networks (network_type="scaled"): ScaledMLP from jaxgcrl/agents/common/scaled_net.py.
# ---------------------------------------------------------------------------------------------


def make_scaled_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    width: int = 256,
    depth: int = 4,
    actor_depth: int = 4,
    use_ln: bool = True,
    use_residual: bool = True,
    activation: str = "swish",
    n_critics: int = 2,
    num_atoms: int = 1,
    v_min: float = 0.0,
    v_max: float = 100.0,
    critic_head_ln: bool = False,
) -> SACNetworks:
    """SAC networks built from ScaledMLP. Same interfaces as make_sac_networks, so Brax's SAC losses apply."""
    from jaxgcrl.agents.common.scaled_net import ScaledMLP
    from jaxgcrl.agents.common.categorical import expectation, support

    z = support(v_min, v_max, num_atoms) if num_atoms > 1 else None

    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    net_kwargs = dict(width=width, use_ln=use_ln, use_residual=use_residual, activation=activation)

    policy_module = ScaledMLP(depth=actor_depth, out_size=parametric_action_distribution.param_size, **net_kwargs)

    def policy_apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    class QModule(linen.Module):
        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            heads = [
                ScaledMLP(depth=depth, out_size=num_atoms, head_ln=critic_head_ln, **net_kwargs)(hidden)
                for _ in range(n_critics)
            ]
            if num_atoms == 1:
                return jnp.concatenate(heads, axis=-1)  # (B, n_critics) Q-values
            return jnp.stack(heads, axis=-2)  # (B, n_critics, num_atoms) logits (categorical critic)

    q_module = QModule()

    def q_apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        out = q_module.apply(q_params, obs, actions)
        if num_atoms == 1:
            return out
        return expectation(out, z)  # expected Q per critic, so the actor losses are unchanged

    def q_logits(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    def q_features(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions, mutable=["intermediates"])[1]

    def policy_features(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs, mutable=["intermediates"])[1]

    dummy_obs = jnp.zeros((1, observation_size))
    dummy_action = jnp.zeros((1, action_size))
    policy_network = networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=policy_apply
    )
    q_network = networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=q_apply
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
        q_features=q_features,
        policy_features=policy_features,
        q_logits=q_logits if num_atoms > 1 else None,
        atoms=z,
    )
