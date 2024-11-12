from src import networks as crl_networks

from typing import Any, NamedTuple
from brax.training import types
from brax.training.types import Params, PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition

def compute_energy(energy_fn, sa_repr, g_repr):
    if energy_fn == "l2":
        logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
    elif energy_fn == "l1":
        logits = -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1)
    elif energy_fn == "dot":
        logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr)
    else:
        raise ValueError(f"Unknown energy function: {energy_fn}")
        
def compute_actor_energy(energy_fn, sa_repr, g_repr):
    if energy_fn == "l2":
        min_q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
    elif energy_fn == "l1":
        min_q = -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1)
    elif energy_fn == "dot":
        min_q = jnp.einsum("ik,ik->i", sa_repr, g_repr)
    else:
        raise ValueError(f"Unknown energy function: {energy_fn}")
        
# Helper for compute_loss
def log_softmax(logits, axis, resubs):
    if not resubs:
        I = jnp.eye(logits.shape[0])
        big = 100
        eps = 1e-6
        return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
    else:
        return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)
    
def compute_loss(contrastive_loss_fn, logits, resubs):
    if contrastive_loss_fn == "symmetric_infonce":
        l_align1, l_unify1 = log_softmax(logits, axis=1, resubs=resubs)
        l_align2, l_unify2 = log_softmax(logits, axis=0, resubs=resubs)
        l_align = l_align1 + l_align2
        l_unif = l_unify1 + l_unify2
        loss = -jnp.mean(jnp.diag(l_align1 + l_unify1) + jnp.diag(l_align2 + l_unify2))
    elif contrastive_loss_fn == "infonce":
        l_align, l_unif = log_softmax(logits, axis=1, resubs=resubs)
        loss = -jnp.mean(jnp.diag(l_align + l_unif))
    elif contrastive_loss_fn == "infonce_backward":
        l_align, l_unif = log_softmax(logits, axis=0, resubs=resubs)
        loss = -jnp.mean(jnp.diag(l_align + l_unif))
    elif contrastive_loss_fn == "flatnce":
        # from https://arxiv.org/pdf/2107.01152
        logits_flat = logits - jnp.diag(logits)[:, None]
        clogits = jax.nn.logsumexp(logits_flat, axis=1)
        l_align = clogits
        l_unif = jnp.sum(logits_flat, axis=-1)
        loss = jnp.exp(clogits - jax.lax.stop_gradient(clogits)).mean()
    else:
        raise ValueError(f"Unknown contrastive loss function: {contrastive_loss_fn}")
    return loss, l_align, l_unif

def compute_metrics(logits, sa_repr, g_repr, l2_loss, c_target, l_align, l_unif):
    I = jnp.eye(logits.shape[0])
    correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
    logits_pos = jnp.sum(logits * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
    if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1) ** 2
    else:
        logsumexp = jax.nn.logsumexp(logits, axis=1) ** 2

    sa_repr_l2 = jnp.sqrt(jnp.sum(sa_repr**2, axis=-1))
    g_repr_l2 = jnp.sqrt(jnp.sum(g_repr**2, axis=-1))

    l_align_log = -jnp.mean(jnp.diag(l_align))
    l_unif_log = -jnp.mean(l_unif)

    metrics = {
        "binary_accuracy": jnp.mean((logits > 0) == I),
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "sa_repr_mean": jnp.mean(sa_repr_l2),
        "g_repr_mean": jnp.mean(g_repr_l2),
        "sa_repr_std": jnp.std(sa_repr_l2),
        "g_repr_std": jnp.std(g_repr_l2),
        "logsumexp": logsumexp.mean(),
        "l2_penalty": l2_loss,
        "c_target": c_target,
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
    def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params, normalizer_params: Any, transitions: Transition, key: PRNGKey) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = crl_network.policy_network.apply(normalizer_params, policy_params, transitions.observation)
        action = crl_network.parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        
        alpha = jnp.exp(log_alpha)
        log_prob = crl_network.parametric_action_distribution.log_prob(dist_params, action)
        target_entropy = -0.5 * action_size
        
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def crl_critic_loss(crl_critic_params: Params, normalizer_params: Any, transitions: Transition, key: PRNGKey):
        sa_encoder_params = crl_critic_params["sa_encoder"]
        g_encoder_params = crl_critic_params["g_encoder"]
        c_target = crl_critic_params["c"]

        # Compute representations
        sa = jnp.concatenate([transitions.observation[:, :env.state_dim], transitions.action], axis=-1)
        sa_repr = crl_network.sa_encoder.apply(normalizer_params, sa_encoder_params, sa)
        g = transitions.observation[:, env.state_dim:]
        g_repr = crl_network.g_encoder.apply(normalizer_params, g_encoder_params, g)
        
        # Compute energy and loss
        logits = compute_energy(energy_fn, sa_repr, g_repr)
        if use_c_target:
            logits *= c_target
        loss, l_align, l_unif = compute_loss(contrastive_loss_fn, logits, resubs)

        # Modify loss (logsumexp, L2 penalty)
        if logsumexp_penalty > 0:
            # For backward we can check jax.nn.logsumexp(logits, axis=0)
            # VM: we could also try removing the diagonal here when using logsumexp penalty + resubs=False
            logits_ = logits
            big = 100
            I = jnp.eye(logits.shape[0])

            if not resubs:
                logits_ = logits - big * I

            eps = 1e-6
            logsumexp = jax.nn.logsumexp(logits_ + eps, axis=1)
            loss += logsumexp_penalty * jnp.mean(logsumexp**2)

        if l2_penalty > 0:
            l2_loss = l2_penalty * (jnp.mean(sa_repr**2) + jnp.mean(g_repr**2))
            loss += l2_loss
        else:
            l2_loss = 0

        # Compute metrics
        metrics = compute_metrics(logits, sa_repr, g_repr, l2_loss, c_target, l_align, l_unif)
        return loss, metrics

    def actor_loss(policy_params: Params, normalizer_params: Any, crl_critic_params: Params, 
                   alpha: jnp.ndarray, transitions: Transition, key: PRNGKey) -> jnp.ndarray:
        sample_key, entropy_key, goal_key = jax.random.split(key, 3)
        sa_encoder_params = crl_critic_params["sa_encoder"]
        g_encoder_params = crl_critic_params["g_encoder"]

        # Compute future state (for goal)
        future_state = transitions.extras["future_state"]
        future_rolled = jnp.roll(future_state, 1, axis=0)
        random_goal_mask = jax.random.bernoulli(goal_key, config.random_goals, shape=(future_state.shape[0], 1))
        future_state = jnp.where(random_goal_mask, future_rolled, future_state)
        
        # Get state and goal
        state = transitions.observation[:, :env.state_dim]
        goal = future_state[:, env.goal_indices]
        sg = jnp.concatenate([state, goal], axis=1)

        # Compute action with policy, given state and goal
        dist_params = crl_network.policy_network.apply(normalizer_params, policy_params, sg)
        parametric_action_distribution = crl_network.parametric_action_distribution
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, sample_key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        entropy = parametric_action_distribution.entropy(dist_params, entropy_key)
        action = parametric_action_distribution.postprocess(action)

        # Compute representations
        sa = jnp.concatenate([state, action], axis=-1)
        sa_repr = crl_network.sa_encoder.apply(normalizer_params, sa_encoder_params, sa)
        g_repr = crl_network.g_encoder.apply(normalizer_params, g_encoder_params, goal)

        # Compute energy and loss
        min_q = compute_actor_energy(energy_fn, sa_repr, g_repr)
        actor_loss = -jnp.mean(min_q)

        # Modify loss (actor entropy and exploration coefficient)
        if not config.disable_entropy_actor:
            actor_loss += alpha * log_prob

        if exploration_coef != 0:
            if energy_fn == "l2":
                actor_loss -= exploration_coef * jnp.mean(jnp.sqrt(jnp.sum(sa_repr**2, axis=-1)))
            elif energy_fn == "dot":
                actor_loss += exploration_coef * jnp.mean(jnp.sqrt(jnp.sum(sa_repr**2, axis=-1)))
            else:
                raise ValueError(f"Unknown exploration_coef for energy function: {energy_fn}")
        
        # Compute metrics
        metrics = {"entropy": entropy.mean()}
        return jnp.mean(actor_loss), metrics

    return alpha_loss, actor_loss, crl_critic_loss