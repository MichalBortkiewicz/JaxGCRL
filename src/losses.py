from src import networks as crl_networks
from typing import Any, NamedTuple
from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition

def log_softmax(logits, axis, resubs):
    if not resubs:
        I = jnp.eye(logits.shape[0])
        big = 100
        eps = 1e-6
        return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
    else:
        return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)
    
def compute_contrastive_loss(contrastive_loss_fn, positive_logits, negative_logits):
    # Since we only have one negative example now, the contrastive loss function doesn't really matter
    l_align = positive_logits
    l_unif = -jax.nn.logsumexp(jnp.concatenate([positive_logits, negative_logits], axis=-1), axis=-1)
    loss = -jnp.mean(l_align + l_unif)
    
    # Removed l2 penalty from loss since it acted on the phi and psi representations which are gone; also removed logsumexp penalty
    return loss, l_align, l_unif

def compute_metrics(positive_logits, negative_logits, l_align, l_unif):
    l_align_log = -jnp.mean(l_align) # Removed diag since l_align is now a vector, not matrix
    l_unif_log = -jnp.mean(l_unif)

    metrics = {
        "binary_accuracy": jnp.mean(positive_logits > 0),
        "logits_pos": positive_logits, # These might not mean the same thing as they did before with multiple negative examples
        "logits_neg": negative_logits,
        "l_align": l_align_log,
        "l_unif": l_unif_log,
    }
    return metrics

def make_losses(
    config: NamedTuple,
    env: object,
    contrastive_loss_fn: str,
    energy_fn: str,
    logsumexp_penalty: float,
    l2_penalty: float,
    resubs: bool,
    crl_network: crl_networks.CRLNetworks,
    action_size: int,
    use_c_target: bool = False,
    exploration_coef: float = 0.0,
):
    """Creates the CRL losses."""

    target_entropy = -0.5 * action_size
    policy_network = crl_network.policy_network
    parametric_action_distribution = crl_network.parametric_action_distribution
    value_network = crl_network.value_network
    obs_dim = env.state_dim

    def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params, normalizer_params: Any, transitions: Transition, key: PRNGKey) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        obs = transitions.observation
        dist_params = policy_network.apply(normalizer_params, policy_params, obs)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def crl_critic_loss(crl_critic_params: Params, normalizer_params: Any, transitions: Transition, key: PRNGKey):
        value_network_params = crl_critic_params["value_network"]
        
        # 1 negative example (monolithic)
        sag_positive = jnp.concatenate([transitions.observation[:, :obs_dim], transitions.action, transitions.observation[:, obs_dim:]], axis=-1)
        positive_logits = value_network.apply(normalizer_params, value_network_params, sag_positive)
        sag_negative = jnp.concatenate([transitions.observation[:, :obs_dim], transitions.action, jnp.roll(transitions.observation[:, obs_dim:], 1, axis=0)], axis=-1)
        negative_logits = value_network.apply(normalizer_params, value_network_params, sag_negative)
        
        # Standard all-pairs negative examples, InfoNCE backward, monolithic.
        # SA is [batch_size, sa_dim] and G is [batch_size, g_dim]. Construct a [batch_size^2, sa_dim + g_dim] matrix SAG such that SAG[batch_size*i + j] = concat(SA[i], G[j]).
        # batch_size = transitions.observation.shape[0]
        # sa = jnp.concatenate([transitions.observation[:, :obs_dim], transitions.action])
        # g = transitions.observation[:, obs_dim:]
        # repeated_sa = jnp.repeat(sa, batch_size, axis=0)
        # tiled_g = jnp.tile(g, [batch_size, 1])
        # sag_matrix = jnp.concatenate([repeated_sa, tiled_g], axis=1)
        # logits = value_network.apply(normalizer_params, value_network_params, sag_matrix)
        
        
        loss, l_align, l_unif = compute_contrastive_loss(contrastive_loss_fn, positive_logits, negative_logits)
        
        # metrics = compute_metrics(positive_logits, negative_logits, l_align, l_unif)
        metrics = {}
        return loss, metrics

    def actor_loss(policy_params: Params, normalizer_params: Any, crl_critic_params: Params, 
                   alpha: jnp.ndarray, transitions: Transition, key: PRNGKey) -> jnp.ndarray:
        # State
        state = transitions.observation[:, :obs_dim]

        # Goal
        key, goal_key = jax.random.split(key)
        future_state = transitions.extras["future_state"]
        random_goal_mask = jax.random.bernoulli(goal_key, config.random_goals, shape=(future_state.shape[0], 1))
        future_rolled = jnp.roll(future_state, 1, axis=0)
        future_state = jnp.where(random_goal_mask, future_rolled, future_state)
        goal = future_state[:, env.goal_indices]

        # Action (sampled from policy)
        key, sample_key, entropy_key = jax.random.split(key, 3)
        sg_tuple = jnp.concatenate([state, goal], axis=1)
        dist_params = policy_network.apply(normalizer_params, policy_params, sg_tuple)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, sample_key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        entropy = parametric_action_distribution.entropy(dist_params, entropy_key)
        action = parametric_action_distribution.postprocess(action)
        
        # Critic value
        value_network_params = crl_critic_params["value_network"]
        sag_tuple = jnp.concatenate([state, action, goal], axis=-1)
        min_q = value_network.apply(normalizer_params, value_network_params, sag_tuple)
        
        # Loss (removed exploration bonus handling for size of phi representation)
        actor_loss = -jnp.mean(min_q)
        if not config.disable_entropy_actor:
            actor_loss += alpha * log_prob

        metrics = {
            "entropy": entropy.mean(),
        }
        return jnp.mean(actor_loss), metrics

    return alpha_loss, actor_loss, crl_critic_loss
