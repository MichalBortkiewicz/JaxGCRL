"""Categorical (classification) critic losses for SAC and TD3, after LAC (Kang, Lee, Kwon, CIKM 2026,
arXiv 2608.26659, Sec. 4.2-4.5, Eq. 10, Algorithm 1).

Each Q-head outputs `num_atoms` logits over a fixed, uniform support z_1..z_I on [v_min, v_max]; the
value used by the actor (and for the twin-critic min) is the expectation Q = sum_i z_i softmax(logits)_i.

Two target constructions (config `cat_projection`):
  "scalar"         (paper, Algorithm 1): y = r + d * gamma * V(s') is computed as a scalar, exactly as in
                   the MSE critic (V' = min over the twin critics' expected values, minus alpha log pi
                   for SAC), then projected onto the support with the two-hot rule of Eq. 10.
  "distributional" (LAC code, agents/lac.py categorical_projection; C51): the target critic's whole
                   next-state distribution is shifted atom by atom, Tz_j = r + d * gamma * (z_j - alpha
                   log pi), clipped, and each atom's mass is split two-hot. With twin critics the next
                   distribution is the one of the critic whose expectation is smaller (LAC code,
                   q_agg='min').
Loss: cross-entropy between the projected target and the predicted distribution, per critic, with
time-limit steps masked out as in the MSE critics (Brax SAC; TD3 --mask-truncation). The agents log the
fraction of scalar targets outside the support (`cat_clip_frac`), to check the support is wide enough.

Rewards in our envs are 0/1, so the default support is [0, reward_scaling / (1 - gamma)] (the paper's
[-1/(1-gamma), 0] for its -1/0 rewards, mirrored). SAC targets also contain -alpha log pi; if
`cat_clip_frac` is large, widen the support with --v-min/--v-max.
"""

import jax
import jax.numpy as jnp


def support(v_min, v_max, num_atoms):
    return jnp.linspace(v_min, v_max, num_atoms)


def default_support(v_min, v_max, discounting, reward_scaling):
    lo = 0.0 if v_min is None else v_min
    hi = reward_scaling / (1.0 - discounting) if v_max is None else v_max
    return float(lo), float(hi)


def expectation(logits, z):
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * z, axis=-1)


def two_hot(y, v_min, v_max, num_atoms):
    """Eq. 10: y (any shape) -> probabilities (..., num_atoms) with mass on the two atoms around clip(y)."""
    dz = (v_max - v_min) / (num_atoms - 1)
    b = (jnp.clip(y, v_min, v_max) - v_min) / dz
    lo = jnp.floor(b)
    up = jnp.ceil(b)
    w_up = b - lo
    w_lo = jnp.where(up == lo, 1.0, up - b)
    w_up = jnp.where(up == lo, 0.0, w_up)
    idx = jnp.arange(num_atoms)
    return w_lo[..., None] * (idx == lo[..., None]) + w_up[..., None] * (idx == up[..., None])


def project_distribution(next_probs, shift, scale, z, v_min, v_max):
    """C51 projection. next_probs (B, I); atoms move to Tz = shift[:, None] + scale[:, None] * z."""
    num_atoms = z.shape[0]
    tz = shift[:, None] + scale[:, None] * z[None, :]
    per_atom = two_hot(tz, v_min, v_max, num_atoms)  # (B, I, I): source atom j -> target atoms
    return jnp.sum(next_probs[..., None] * per_atom, axis=-2)


def cross_entropy(logits, target_probs):
    return -jnp.sum(target_probs * jax.nn.log_softmax(logits, axis=-1), axis=-1)


def categorical_td_loss(
    q_logits, next_logits_target, reward, bootstrap_scale, next_correction, truncation, z, v_min, v_max, mode
):
    """Shared by SAC and TD3.

    q_logits: (B, C, I) online logits at (s, a); next_logits_target: (B, C, I) target logits at (s', a').
    Target: y = reward + bootstrap_scale * (V' + next_correction), where V' is the min over critics of the
    expected next value; bootstrap_scale = discount * gamma (n-step: gamma^m * discount);
    next_correction = -alpha log pi (SAC) or 0 (TD3). truncation: (B,), 1 at time-limit steps (masked).
    """
    num_atoms = z.shape[0]
    next_q = expectation(next_logits_target, z)  # (B, C)
    y = reward + bootstrap_scale * (jnp.min(next_q, axis=-1) + next_correction)
    if mode == "scalar":
        target = two_hot(y, v_min, v_max, num_atoms)
    elif mode == "distributional":
        c_star = jnp.argmin(next_q, axis=-1)
        probs = jax.nn.softmax(next_logits_target, axis=-1)
        next_probs = jnp.take_along_axis(probs, c_star[:, None, None], axis=1)[:, 0]
        target = project_distribution(
            next_probs, reward + bootstrap_scale * next_correction, bootstrap_scale, z, v_min, v_max
        )
    else:
        raise ValueError(f"Unknown cat_projection {mode!r}")
    target = jax.lax.stop_gradient(target)
    ce = cross_entropy(q_logits, target[:, None, :])  # (B, C)
    ce = ce * (1.0 - truncation)[:, None]
    return jnp.mean(ce)


def make_sac_critic_loss(sac_network, reward_scaling, discounting, v_min, v_max, mode):
    """Drop-in replacement for brax SAC's critic_loss (same signature and the same next-action sampling)."""
    policy_network = sac_network.policy_network
    dist = sac_network.parametric_action_distribution
    z = sac_network.atoms

    def critic_loss(q_params, policy_params, normalizer_params, target_q_params, alpha, transitions, key):
        q_logits = sac_network.q_logits(normalizer_params, q_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
        next_action = dist.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = dist.log_prob(next_dist_params, next_action)
        next_action = dist.postprocess(next_action)
        next_logits = sac_network.q_logits(
            normalizer_params, target_q_params, transitions.next_observation, next_action
        )
        return categorical_td_loss(
            q_logits,
            jax.lax.stop_gradient(next_logits),
            transitions.reward * reward_scaling,
            transitions.discount * discounting,
            jax.lax.stop_gradient(-alpha * next_log_prob),
            transitions.extras["state_extras"]["truncation"],  # Brax SAC always masks time-limit steps
            z,
            v_min,
            v_max,
            mode,
        )

    return critic_loss


def make_td3_critic_loss(
    td3_network, reward_scaling, discounting, v_min, v_max, mode, mask_truncation, smoothing=0.2, noise_clip=0.5
):
    """Drop-in replacement for td3/losses.py critic_loss (same signature, same target-policy smoothing)."""
    policy_network = td3_network.policy_network
    z = td3_network.atoms

    def critic_loss(q_params, target_q_params, target_policy_params, normalizer_params, transitions, key):
        q_logits = td3_network.q_logits(normalizer_params, q_params, transitions.observation, transitions.action)
        next_actions = policy_network.apply(normalizer_params, target_policy_params, transitions.next_observation)
        noise = (jax.random.normal(key, next_actions.shape) * smoothing).clip(-noise_clip, noise_clip)
        next_actions = (next_actions + noise).clip(-1.0, 1.0)
        next_logits = td3_network.q_logits(
            normalizer_params, target_q_params, transitions.next_observation, next_actions
        )
        trunc = transitions.extras["state_extras"]["truncation"]
        return categorical_td_loss(
            q_logits,
            jax.lax.stop_gradient(next_logits),
            transitions.reward * reward_scaling,
            transitions.discount * discounting,
            jnp.zeros_like(transitions.reward),
            trunc if mask_truncation else jnp.zeros_like(trunc),
            z,
            v_min,
            v_max,
            mode,
        )

    return critic_loss
