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
    parametric_action_distribution: distribution.ParametricDistribution
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork

def make_inference_fn(crl_networks: CRLNetworks):
    """Creates params and inference function for the CRL agent."""
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

def make_crl_networks(config: NamedTuple, env: object, observation_size: int, action_size: int,
                      preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
                      hidden_layer_sizes: Sequence[int] = (256, 256), activation: networks.ActivationFn = linen.relu, 
                      use_ln: bool=False) -> CRLNetworks:
    """Make CRL networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    value_network = networks.make_value_network(
        obs_size=env.state_dim + action_size + len(env.goal_indices),
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    return CRLNetworks(
        parametric_action_distribution=parametric_action_distribution,
        policy_network=policy_network,
        value_network=value_network,
    )
