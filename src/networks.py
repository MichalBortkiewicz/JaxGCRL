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
    """
    MLP class to build a Multi-Layer Perceptron.

    This class constructs a fully connected neural network given layer sizes, activation function, and other properties.
    Supports optional layer normalization, initialization, and activation on the final layer.

    Attributes:
        layer_sizes (Sequence[int]): A sequence of integers where each integer represents
                                     the number of neurons in that particular layer.
        activation (ActivationFn): Activation function to be applied after each hidden layer.
                                      Default is `linen.relu`.
        kernel_init (Initializer): Initializer function for the Dense layer's kernel.
                                   Defaults to `jax.nn.initializers.lecun_uniform()`.
        activate_final (bool): A flag to determine if activation should also be applied to
                               the final layer. Defaults to `False`.
        bias (bool): Whether to use bias in the Dense layers. Defaults to `True`.
        use_layer_norm (bool): Flag to determine if Layer Normalization should be used.
                               Defaults to `False`.

    Methods:
        __call__(data: jnp.ndarray) -> jnp.ndarray
            Forward pass through the network.

            Parameters:
            data (jnp.ndarray): Input data to the MLP.

            Returns:
            jnp.ndarray: Output of the MLP after passing through all layers and activation functions.
    """

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


def make_embedder(
    layer_sizes: Sequence[int],
    obs_size: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray] = linen.swish,
    preprocess_observations_fn: types.PreprocessObservationFn = types,
    use_ln: bool = False
) -> networks.FeedForwardNetwork:

    """
    Constructs a FeedForwardNetwork to embed observations using a specified Multi-Layer Perceptron (MLP).

    Summary:
    The `make_embedder` function creates a feedforward neural network given the layer sizes of the MLP.

    Args:
        layer_sizes (Sequence[int]): A sequence of integers indicating the sizes of the layers in the MLP.
        obs_size (int): An integer representing the size of the observation input.
        activation (Callable[[jnp.ndarray], jnp.ndarray]):
            A callable activation function applied to each layer of the MLP.
            Default is linen.swish.
        preprocess_observations_fn (types.PreprocessObservationFn):
            An optional function for preprocessing observations before embedding.
            Default is `types`.
        use_ln (bool): A boolean indicating whether to use layer normalization.
            Default is False.

    Returns:
        networks.FeedForwardNetwork: An instance of `FeedForwardNetwork` configured with the specified MLP.

    """
    dummy_obs = jnp.zeros((1, obs_size))
    module = MLP(layer_sizes=layer_sizes, activation=activation, use_layer_norm=use_ln)

    # TODO: should we have a function to preprocess the observations?
    def apply(processor_params, policy_params, obs):
        # obs = preprocess_observations_fn(obs, processor_params)
        return module.apply(policy_params, obs)

    model = networks.FeedForwardNetwork(init=lambda rng: module.init(rng, dummy_obs), apply=apply)
    return model

def make_inference_fn(crl_networks: CRLNetworks):
    """
    Creates and returns a policy function using a given CRL network.

    This function generates a policy function that will be used for making action
    predictions based on observations.

    Args:
        crl_networks (CRLNetworks): The CRL network used for creating the policy
            function.

    Returns:
        Callable: A function that creates the policy based on the provided
            parameters and observation.

    """
    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:
        def policy(obs: types.Observation, key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
            logits = crl_networks.policy_network.apply(*params[:2], obs)
            if deterministic:
                action = crl_networks.parametric_action_distribution.mode(logits)
            else:
                action = crl_networks.parametric_action_distribution.sample(logits, key_sample)
            return action, {}
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
    """
    Creates a set of CRL networks for a given
    environment configuration and observation-action setup.

    Parameters:
        config (NamedTuple): Configuration object containing 'repr_dim' attribute.
        env (object): The environment object must have 'state_dim' and 'goal_indices'
            attributes.
        observation_size (int): The size of the observation space.
        action_size (int): The size of the action space.
        preprocess_observations_fn (types.PreprocessObservationFn, optional): Function
            used to preprocess observations. Defaults to `types.identity_observation_preprocessor`.
        hidden_layer_sizes (Sequence[int], optional): Sizes of the hidden layers for the
            networks. Defaults to (256, 256).
        activation (networks.ActivationFn, optional): Activation function to use in the
            networks. Defaults to `linen.relu`.
        use_ln (bool, optional): Whether to use Layer Normalization. Defaults to False.

    Returns:
        CRLNetworks: An instance of the CRLNetworks containing policy network, parametric
            action distribution, state-action encoder, and goal encoder.
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    sa_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=env.state_dim + action_size,
        activation=activation,
        preprocess_observations_fn=preprocess_observations_fn,
        use_ln=use_ln
    )
    g_encoder = make_embedder(
        layer_sizes=list(hidden_layer_sizes) + [config.repr_dim],
        obs_size=len(env.goal_indices),
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
