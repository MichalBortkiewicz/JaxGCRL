from typing import Any, NamedTuple

from brax.training import types
from optax import sigmoid_binary_cross_entropy

from src import networks as crl_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp
from src import utils


Transition = types.Transition


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
    sa_encoder = crl_network.sa_encoder
    g_encoder = crl_network.g_encoder
    obs_dim = env.state_dim

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
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
        key: PRNGKey,
    ):

        sa_encoder_params, g_encoder_params, c_target = (
            crl_critic_params["sa_encoder"],
            crl_critic_params["g_encoder"],
            crl_critic_params["c"],
        )

        key, t1_key, t2_key = jax.random.split(key, 3)
        obs = transitions.observation[:, :obs_dim]
        action = transitions.action
        future_action = transitions.extras["future_action"]
        future_action_shuf = jax.random.permutation(key, future_action)
        # goal = transitions.observation[:, obs_dim:]
        # obs_shuf = jax.random.permutation(key2, obs)
        # goal_pad = jax.lax.dynamic_update_slice_in_dim(obs_shuf, goal, 0, -1)
        goal_pad = transitions.extras["future_state"]

        tau1 = jax.random.uniform(t1_key, (config.num_tau,))
        tau2 = jax.random.uniform(t2_key, (config.num_tau,))

        

        
        sa_repr = sa_encoder.apply(
            normalizer_params,
            sa_encoder_params,
            jnp.concatenate([obs, action], axis=-1),
            tau1,
        ) # [num_tau_samples, batch_size, repr_dim]
        g_repr = g_encoder.apply(
            normalizer_params, 
            g_encoder_params, 
            transitions.observation[:, obs_dim:], 
            tau2
        ) # [num_tau_samples, batch_size, repr_dim]

        # jax.debug.print("sa_shape {sa}, g_shape {g}", sa=sa_repr.shape, g=g_repr.shape)

        assert energy_fn == "l2"
        # Not every tau is compared against every other, the resulting logits matrix is
        # [num_tau_samples, batch_size, batch_size]


        if energy_fn == "l2":
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, :, None, :] - g_repr[:, None, :, :]) ** 2, axis=-1))
        else:
            raise ValueError(f"Unknown energy function: {energy_fn}")

        logits = jnp.moveaxis(logits, -1, 0)

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

        assert contrastive_loss_fn == "symmetric_infonce"

        if contrastive_loss_fn == "symmetric_infonce":
            l_align1, l_unify1 = log_softmax(logits, axis=1, resubs=resubs)
            l_align2, l_unify2 = log_softmax(logits, axis=0, resubs=resubs)
            l_align = l_align1 + l_align2
            l_unif = l_unify1 + l_unify2
            
            loss = -jnp.mean(jnp.diagonal(l_align1 + l_unify1, axis1=1, axis2=2)+ jnp.diagonal(l_align2 + l_unify2, axis1=1, axis2=2))
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

        if l2_penalty > 0:
            l2_loss = l2_penalty * (jnp.mean(sa_repr**2) + jnp.mean(g_repr**2))
            loss += l2_loss
        else:
            l2_loss = 0


        I = jnp.eye(logits.shape[0])
        correct = 0 #jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = 0 #jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = 0 #jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        logsumexp = 0
        # if len(logits.shape) == 3:
        #     logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1) ** 2
        # else:
        #     logsumexp = jax.nn.logsumexp(logits, axis=1) ** 2

        sa_repr_l2 = jnp.sqrt(jnp.sum(sa_repr**2, axis=-1))
        g_repr_l2 = jnp.sqrt(jnp.sum(g_repr**2, axis=-1))

        l_align_log = 0
        l_unif_log = 0



        metrics = {
            "binary_accuracy": 0,
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "sa_repr_mean": jnp.mean(sa_repr_l2),
            "g_repr_mean": jnp.mean(g_repr_l2),
            "sa_repr_std": jnp.std(sa_repr_l2),
            "g_repr_std": jnp.std(g_repr_l2),
            "logsumexp": 0,
            "l2_penalty": l2_loss,
            "c_target": c_target,
            "l_align": l_align_log,
            "l_unif": l_unif_log,
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

        sample_key, entropy_key, goal_key, t1_key, t2_key = jax.random.split(key, 5)

        obs = transitions.observation

        future_state = transitions.extras["future_state"]

        state = obs[:, :obs_dim]

        random_goal_mask = jax.random.bernoulli(goal_key, config.random_goals, shape=(future_state.shape[0], 1))
        future_rolled = jnp.roll(future_state, 1, axis=0)
        future_state = jnp.where(random_goal_mask, future_rolled, future_state)
        future_action = transitions.extras["future_action"]

        goal = future_state[:, env.goal_indices]

        observation = jnp.concatenate([state, goal], axis=1)

        dist_params = policy_network.apply(normalizer_params, policy_params, observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, sample_key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        entropy = parametric_action_distribution.entropy(dist_params, entropy_key)
        action = parametric_action_distribution.postprocess(action)

        extra_key, key = jax.random.split(key, 2)
        future_action_shuf = jax.random.permutation(extra_key, future_action)

        sa_encoder_params, g_encoder_params = (
            crl_critic_params["sa_encoder"],
            crl_critic_params["g_encoder"],
        )
        
        tau1 = jax.random.uniform(t1_key, (config.num_tau,))
        tau2 = jax.random.uniform(t2_key, (config.num_tau,))

        if config.risk_measure == "cpw":
            eta = config.eta or 0.71
            tau2 = tau2 ** eta / ((tau2 ** eta + (1. - tau2) ** eta) ** (1. / eta))
        elif config.risk_measure == "cvar":
            eta = config.eta or 0.1
            tau2 = eta * tau2
        elif config.risk_measure == "pow":
            eta = config.eta or -2.0
            if eta >= 0.:
                tau2 = tau2 ** (1. / (1. + eta))
            else:
                tau2 = 1. - (1. - tau2) ** (1. / (1. - eta))

        sa_repr = sa_encoder.apply(
            normalizer_params,
            sa_encoder_params,
            jnp.concatenate([state, action], axis=-1),
            tau1
        )
        g_repr = g_encoder.apply(normalizer_params, g_encoder_params, goal, tau2)

        goal_pad = future_state

        if energy_fn == "l2":
            min_q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
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
