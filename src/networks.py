from typing import Sequence, Tuple, Callable, NamedTuple, Any

import jax
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class CRLNetworks:
    policy_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    sa_encoder: networks.FeedForwardNetwork
    g_encoder: networks.FeedForwardNetwork

class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    use_layer_norm: bool = False
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
                if self.use_layer_norm:
                    hidden = linen.LayerNorm()(hidden)
                hidden = self.activation(hidden)
        return hidden


class TauEncoder(linen.Module):
    output_dim: int
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, tau: jnp.ndarray):
        # Tau has shape [num_tau_samples]

        tau = jnp.cos(tau * jnp.arange(tau.shape[-1]) * jnp.pi)
        encoded_tau = linen.Dense(
            self.output_dim,
            name=f"tau_enc_dense",
            use_bias=True,
        )(tau)
        return encoded_tau



class IQN(linen.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    use_layer_norm: bool = False
    repr_dim: int = 64

    @linen.compact
    def __call__(self, data: jnp.ndarray, tau: jnp.ndarray):
        # data - [batch_size, input_dim]
        # tau_samples - [num_tau_samples]
        obs_encoder = MLP(layer_sizes=self.layer_sizes, activation=self.activation, kernel_init=self.kernel_init, activate_final=self.activate_final, bias=self.bias, use_layer_norm=self.use_layer_norm)
        combined_encoder = MLP(layer_sizes=self.layer_sizes, activation=self.activation, kernel_init=self.kernel_init, activate_final=self.activate_final, bias=self.bias, use_layer_norm=self.use_layer_norm)

        # encoded_obs - [batch_size, repr_dim]
        # encoded_tau - [repr_dim]
        encoded_obs = obs_encoder(data)
        encoded_tau = TauEncoder(output_dim=self.repr_dim,kernel_init=self.kernel_init)(tau)

        combined = encoded_obs * encoded_tau

        return combined_encoder(combined)


def make_embedder(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    preprocess_observations_fn: types.PreprocessObservationFn = types,
    use_ln: bool = False
) -> networks.FeedForwardNetwork:

    """Creates a model."""
    dummy_obs = jnp.zeros((1, obs_size))
    module = MLP(layer_sizes=layer_sizes, activation=activation, use_layer_norm=use_ln)

    # TODO: should we have a function to preprocess the observations?
    def apply(processor_params, policy_params, obs):
        # obs = preprocess_observations_fn(obs, processor_params)
        return module.apply(policy_params, obs)

    model = networks.FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs), apply=apply
    )
    return model



def make_iqn_embedder( 
    layer_sizes: Sequence[int],
    obs_size: int,
    tau_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    preprocess_observations_fn: types.PreprocessObservationFn = types,
    use_ln: bool = False,
) -> networks.FeedForwardNetwork:
    
    repr_dim = layer_sizes[-1]

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_tau = jnp.zeros((tau_size))
    module = IQN(layer_sizes=layer_sizes, activation=activation, use_layer_norm=use_ln, repr_dim=repr_dim)

    # TODO: should we have a function to preprocess the observations?
    def apply(processor_params, params, obs, tau):
        # obs = preprocess_observations_fn(obs, processor_params)
        return module.apply(params, obs, tau)

    model = networks.FeedForwardNetwork(
        init=lambda rng: module.init(rng, dummy_obs, dummy_tau), apply=apply
    )
    return model



def make_inference_fn(crl_networks: CRLNetworks):
    """Creates params and inference function for the CRL agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = crl_networks.policy_network.apply(*params[:2], observations)
            if deterministic:
                return crl_networks.parametric_action_distribution.mode(logits), {}
            return (
                crl_networks.parametric_action_distribution.sample(logits, key_sample),
                {},
            )

        return policy

    return make_policy


def make_crl_networks(
    config: NamedTuple,
    env: object,
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    use_ln: bool= False
) -> CRLNetworks:
    """Make CRL networks."""
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

    embedder_factory = make_embedder if not config.use_iqn else make_iqn_embedder
    sa_encoder = embedder_factory(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=env.state_dim + action_size,
        tau_size=config.num_tau,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
        use_ln=use_ln
    )
    g_encoder = embedder_factory(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=len(env.goal_indices),
        tau_size=config.num_tau,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
        use_ln=use_ln
    )

    return CRLNetworks(
        policy_network=policy_network,
        parametric_action_distribution=parametric_action_distribution,
        sa_encoder=sa_encoder,
        g_encoder=g_encoder,
    )
