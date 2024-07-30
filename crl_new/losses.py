from typing import Any, NamedTuple

from brax.training import types
from optax import sigmoid_binary_cross_entropy

from crl_new import networks as crl_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
from envs.wrappers import extract_info_from_obs


Transition = types.Transition


def make_losses(
    config: NamedTuple,
    contrastive_loss_fn: str,
    energy_fn: str,
    logsumexp_penalty: float,
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
    sa_encoder = crl_network.sa_encoder
    g_encoder = crl_network.g_encoder
    obs_dim = config.obs_dim

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        if config.use_old_trans_alpha:
            obs = transitions.extras["old_trans"].observation
        else:
            obs = transitions.observation

        dist_params = policy_network.apply(normalizer_params, policy_params, obs)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def crl_critic_loss(
        crl_critic_params: Params,
        normalizer_params: Any,
        transitions: Transition,
    ):
        # This is for debug purposes only
        if config.use_traj_idx_wrapper:
            old_obs, info_1, info_2 = extract_info_from_obs(transitions.observation, config)
            jax.debug.print(
                "OBS: \n{obs},\n\n info_1 \n{i_1},\n\n info_2 \n{i_2}\n\n", obs=old_obs, i_1=info_1, i_2=info_2
            )

        sa_encoder_params, g_encoder_params, c_target = (
            crl_critic_params["sa_encoder"],
            crl_critic_params["g_encoder"],
            crl_critic_params["c"],
        )
        sa_repr = sa_encoder.apply(
            normalizer_params,
            sa_encoder_params,
            jnp.concatenate([transitions.observation[:, :obs_dim], transitions.action], axis=-1),
        )
        g_repr = g_encoder.apply(normalizer_params, g_encoder_params, transitions.observation[:, obs_dim:])
        if energy_fn == "l2":
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
        elif energy_fn == "l2_no_sqrt":
            logits = -jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)
        elif energy_fn == "l1":
            logits = -jnp.sum(jnp.abs(sa_repr[:, None, :] - g_repr[None, :, :]), axis=-1)
        elif energy_fn == "dot":
            logits = jnp.einsum("ik,jk->ij", sa_repr, g_repr)
        elif energy_fn == "cos":
            sa_norm = jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
            g_norm = jnp.linalg.norm(g_repr, axis=1, keepdims=True)
            sa_normalized = sa_repr / sa_norm
            g_normalized = g_repr / g_norm
            logits = jnp.einsum("ik,jk->ij", sa_normalized, g_normalized)
        else:
            raise ValueError(f"Unknown energy function: {energy_fn}")

        def log_softmax(logits, axis, resubs):
            if not resubs:
                I = jnp.eye(logits.shape[0])
                big = 100
                eps = 1e-6
                return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
            else:
                return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)

        if use_c_target:
            logits = logits * c_target

        if contrastive_loss_fn == "binary":
            loss = jnp.mean(
                sigmoid_binary_cross_entropy(logits, labels=jnp.eye(logits.shape[0]))
            )  # shape[0] - is a batch size
            l_align, l_unify = log_softmax(logits, axis=1, resubs=resubs)
        elif contrastive_loss_fn == "symmetric_infonce":
            l_align1, l_unify1 = log_softmax(logits, axis=1, resubs=resubs)
            l_align2, l_unify2 = log_softmax(logits, axis=0, resubs=resubs)
            l_align = l_align1 + l_align2
            l_unify = l_unify1 + l_unify2
            loss = -jnp.mean(jnp.diag(l_align1 + l_unify1) + jnp.diag(l_align2 + l_unify2))
        elif contrastive_loss_fn == "infonce":
            l_align, l_unify = log_softmax(logits, axis=1, resubs=resubs)
            loss = -jnp.mean(jnp.diag(l_align + l_unify))
        elif contrastive_loss_fn == "infonce_backward":
            l_align, l_unify = log_softmax(logits, axis=0, resubs=resubs)
            loss = -jnp.mean(jnp.diag(l_align + l_unify))
        elif contrastive_loss_fn == "fb":
            # This is a Monte Carlo version of the loss from "Does Zero-Shot Reinforcement Learning Exist?"
            batch_size = logits.shape[0]
            I = jnp.eye(batch_size)
            l_align = -jnp.diag(logits)  # shape = (batch_size,)
            l_unif = 0.5 * jnp.sum(logits**2 * (1 - I) / (batch_size - 1), axis=-1)  # shape = (batch_size,)
            loss = (l_align + l_unif).mean()  # shape = ()
        else:
            raise ValueError(f"Unknown contrastive loss function: {contrastive_loss_fn}")

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
            "c_target": c_target,
            "l_align": -jnp.mean(jnp.diag(l_align)),
            "l_unif": -jnp.mean(l_unify),
        }
        return loss, metrics

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        crl_critic_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:

        sample_key, entropy_key, goal_key = jax.random.split(key, 3)

        if config.use_old_trans_actor:
            obs = transitions.extras["old_trans"].observation
        else:
            obs = transitions.observation

        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]

        random_goal_mask = jax.random.bernoulli(goal_key, config.random_goals, shape=(goal.shape[0], 1))
        goal_rolled = jnp.roll(goal, 1, axis=0)
        goal = jnp.where(random_goal_mask, goal_rolled, goal)

        observation = jnp.concatenate([state, goal], axis=1)

        dist_params = policy_network.apply(normalizer_params, policy_params, observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, sample_key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        entropy = parametric_action_distribution.entropy(dist_params, entropy_key)
        action = parametric_action_distribution.postprocess(action)

        sa_encoder_params, g_encoder_params = (
            crl_critic_params["sa_encoder"],
            crl_critic_params["g_encoder"],
        )

        sa_repr = sa_encoder.apply(
            normalizer_params,
            sa_encoder_params,
            jnp.concatenate([state, action], axis=-1),
        )
        g_repr = g_encoder.apply(normalizer_params, g_encoder_params, goal)

        if energy_fn == "l2":
            min_q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
        elif energy_fn == "l2_no_sqrt":
            min_q = -jnp.sum((sa_repr - g_repr) ** 2, axis=-1)
        elif energy_fn == "l1":
            min_q = -jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1)
        elif energy_fn == "dot":
            min_q = jnp.einsum("ik,ik->i", sa_repr, g_repr)
        elif energy_fn == "cos":
            sa_norm = jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
            g_norm = jnp.linalg.norm(g_repr, axis=1, keepdims=True)
            sa_normalized = sa_repr / sa_norm
            g_normalized = g_repr / g_norm
            min_q = jnp.einsum("ik,ik->i", sa_normalized, g_normalized)
        else:
            raise ValueError(f"Unknown energy function: {energy_fn}")

        if config.disable_entropy_actor:
            actor_loss = -jnp.mean(min_q)
        else:
            actor_loss = alpha * log_prob - jnp.mean(min_q)

        if exploration_coef != 0:
            if energy_fn == "l2":
                actor_loss -= exploration_coef * jnp.mean(jnp.sqrt(jnp.sum(sa_repr**2, axis=-1)))
            elif energy_fn == "dot":
                actor_loss += exploration_coef * jnp.mean(jnp.sqrt(jnp.sum(sa_repr**2, axis=-1)))
            else:
                raise ValueError(f"Unknown exploration_coef for energy function: {energy_fn}")

        metrics = {
            "entropy": entropy.mean(),
        }
        return jnp.mean(actor_loss), metrics

    return alpha_loss, actor_loss, crl_critic_loss
