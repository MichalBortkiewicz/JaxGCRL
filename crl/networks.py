from typing import Callable, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.networks import ActivationFn, FeedForwardNetwork, MLP
from brax.training.types import PRNGKey
from flax import linen


def make_embedder(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    preprocess_observations_fn: types.PreprocessObservationFn = types,
) -> networks.FeedForwardNetwork:

    """Creates a model."""
    dummy_obs = jnp.zeros((1, obs_size))
    module = networks.MLP(layer_sizes=layer_sizes, activation=activation)

    # TODO: should we have a function to preprocess the observations?
    def apply(processor_params, policy_params, obs):
        # obs = preprocess_observations_fn(obs, processor_params)
        return module.apply(policy_params, obs)

    model = networks.FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=apply
    )
    return model

def make_crl_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        activate_final=True,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )

@flax.struct.dataclass
class SACNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    sa_encoder: networks.FeedForwardNetwork
    g_encoder: networks.FeedForwardNetwork
    crl_policy_network: networks.FeedForwardNetwork
    crl_parametric_action_distribution: distribution.ParametricDistribution

# TODO: here
def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.crl_policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.crl_parametric_action_distribution.mode(logits), {}
            return (
                sac_networks.crl_parametric_action_distribution.sample(logits, key_sample),
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
    repr_dim: int = 64,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    q_network = networks.make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    # TODO: refactor observation sizes
    sa_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [repr_dim],
        obs_size=2 + action_size,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    g_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [repr_dim],
        obs_size=2,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
    )

    crl_parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    crl_policy_network = make_crl_policy_network(
        crl_parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )

    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
        crl_policy_network=crl_policy_network,
        crl_parametric_action_distribution=crl_parametric_action_distribution,
    )