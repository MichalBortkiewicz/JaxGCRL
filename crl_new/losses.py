from typing import Any, NamedTuple

from brax.training import types
from optax import sigmoid_binary_cross_entropy

from crl_new import networks as sac_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

Transition = types.Transition


def make_losses(
    config:NamedTuple,
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
    """Creates the SAC losses."""

    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution
    sa_encoder = sac_network.sa_encoder
    g_encoder = sac_network.g_encoder

    obs_dim = config.obs_dim
    SAC = config.sac

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, transitions.action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling
            + transitions.discount * discounting * next_v
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def crl_critic_loss(
        crl_critic_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ):
        sa_encoder_params, g_encoder_params = (
            crl_critic_params["sa_encoder"],
            crl_critic_params["g_encoder"],
        )
        sa_repr = sa_encoder.apply(
            normalizer_params,
            sa_encoder_params,
            jnp.concatenate(
                [transitions.observation[:, :obs_dim], transitions.action], axis=-1
            ),
        )
        g_repr = g_encoder.apply(
            normalizer_params, g_encoder_params, transitions.observation[:, obs_dim:]
        )
        logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr)
        loss = jnp.mean(
            sigmoid_binary_cross_entropy(logits, labels=jnp.eye(logits.shape[0]))
        )  # shape[0] - is a batch size

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        if len(logits.shape) == 3:
            logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1) ** 2
        else:
            logsumexp = jax.nn.logsumexp(logits, axis=1) ** 2
        metrics = {
            "binary_accuracy": jnp.mean((logits > 0) == I),
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
        }

        return loss, metrics


    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        crl_critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)


        if SAC:
            q_action = q_network.apply(
                normalizer_params, q_params, transitions.observation, action
            )
            min_q = jnp.min(q_action, axis=-1)
        else:
            sa_encoder_params, g_encoder_params = (
                crl_critic_params["sa_encoder"],
                crl_critic_params["g_encoder"],
            )
            sa_repr = sa_encoder.apply(
                normalizer_params,
                sa_encoder_params,
                jnp.concatenate([transitions.observation[:, :obs_dim], action], axis=-1),
            )
            g_repr = g_encoder.apply(
                normalizer_params, g_encoder_params, transitions.observation[:, obs_dim:]
            )
            min_q = jnp.einsum("ik,ik->i", sa_repr, g_repr)

        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)

    return alpha_loss, critic_loss, actor_loss, crl_critic_loss
