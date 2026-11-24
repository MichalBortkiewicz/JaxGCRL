"""TD3 networks."""

from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training import networks, types
from brax.training.networks import ActivationFn, FeedForwardNetwork, Initializer
from brax.training.types import PRNGKey
from flax import linen, struct


@struct.dataclass
class TD3Networks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    # feature hooks for rep_metrics (scaled networks only); the policy hook returns pre-tanh trunk features
    q_features: Any = struct.field(pytree_node=False, default=None)
    policy_features: Any = struct.field(pytree_node=False, default=None)
    # categorical critic (num_atoms > 1): fn(processor_params, params, obs, actions) -> (B, C, I) logits; support z
    q_logits: Any = struct.field(pytree_node=False, default=None)
    atoms: Any = struct.field(pytree_node=False, default=None)


def make_inference_fn(td3_networks: TD3Networks):
    """Creates params and inference function for the TD3 agent."""

    def make_policy(
        params: types.PolicyParams, exploration_noise=0.0, noise_clip=0.0, deterministic=False
    ) -> types.Policy:  # type: ignore
        def policy(observations: types.Observation, key_noise: PRNGKey) -> Tuple[types.Action, types.Extra]:
            actions = td3_networks.policy_network.apply(*params, observations)
            noise = (jax.random.normal(key_noise, actions.shape) * exploration_noise).clip(
                -noise_clip, noise_clip
            )
            return actions + noise, {}

        return policy

    return make_policy


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


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_actions = policy_module.apply(policy_params, obs)
        return linen.tanh(raw_actions)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_td3_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    layer_norm: bool = False,
) -> TD3Networks:
    """Make TD3 networks."""
    policy_network = make_policy_network(
        action_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )

    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        layer_norm=layer_norm,
    )

    return TD3Networks(policy_network=policy_network, q_network=q_network)


def make_scaled_td3_networks(
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
) -> TD3Networks:
    """TD3 networks built from ScaledMLP (network_type="scaled"). Policy output goes through tanh, as in
    make_policy_network above; the twin Q-network concatenates (obs, action) like Brax's make_q_network."""
    from jaxgcrl.agents.common.scaled_net import ScaledMLP
    from jaxgcrl.agents.common.categorical import expectation, support

    z = support(v_min, v_max, num_atoms) if num_atoms > 1 else None

    net_kwargs = dict(width=width, use_ln=use_ln, use_residual=use_residual, activation=activation)
    policy_module = ScaledMLP(depth=actor_depth, out_size=action_size, **net_kwargs)

    def policy_apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return linen.tanh(policy_module.apply(policy_params, obs))

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
    return TD3Networks(
        policy_network=FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=policy_apply),
        q_network=FeedForwardNetwork(init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=q_apply),
        q_features=q_features,
        policy_features=policy_features,
        q_logits=q_logits if num_atoms > 1 else None,
        atoms=z,
    )
