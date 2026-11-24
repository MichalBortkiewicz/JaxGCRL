"""Categorical critic (common/categorical.py; LAC Eq. 10) and its use in the scaled SAC/TD3 networks."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxgcrl.agents.common import categorical as cat
from jaxgcrl.agents.common.scaled_net import variant_flags
from jaxgcrl.agents.common.train_utils import gradient_update_fn_with_norm
from jaxgcrl.agents.sac import networks as sac_networks
from jaxgcrl.agents.td3 import networks as td3_networks
from jaxgcrl.agents.sac.sac import Transition

OBS, ACT, B = 31, 8, 16


def batch(key):
    k = jax.random.split(key, 4)
    return Transition(
        observation=jax.random.normal(k[0], (B, OBS)),
        next_observation=jax.random.normal(k[1], (B, OBS)),
        action=jnp.tanh(jax.random.normal(k[2], (B, ACT))),
        reward=(jax.random.uniform(k[3], (B,)) < 0.3).astype(jnp.float32),
        discount=jnp.ones((B,)),
        extras={"state_extras": {"truncation": jnp.zeros((B,)), "traj_id": jnp.zeros((B,))}, "policy_extras": {}},
    )

VMIN, VMAX, I = 0.0, 100.0, 51


def test_two_hot_expectation_and_mass():
    z = cat.support(VMIN, VMAX, I)
    y = jnp.array([0.0, 1.3, 2.0, 37.77, 99.9, 100.0, -5.0, 250.0])
    p = cat.two_hot(y, VMIN, VMAX, I)
    assert np.allclose(np.asarray(p.sum(-1)), 1.0)
    assert np.allclose(np.asarray((p * z).sum(-1)), np.clip(np.asarray(y), VMIN, VMAX), atol=1e-4)
    assert np.count_nonzero(np.asarray(p[2])) == 1  # y on an atom (2.0 = z_1) -> one-hot


def test_projection_preserves_mean_inside_support():
    z = cat.support(VMIN, VMAX, I)
    probs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(0), (4, I)) - 3 * jnp.arange(I) / I, axis=-1)
    shift = jnp.array([0.0, 1.0, 0.5, 1.0])
    scale = jnp.array([0.99, 0.5, 0.9, 0.0])
    out = cat.project_distribution(probs, shift, scale, z, VMIN, VMAX)
    assert np.allclose(np.asarray(out.sum(-1)), 1.0, atol=1e-5)
    want = np.asarray(shift + scale * (probs * z).sum(-1))
    assert np.allclose(np.asarray((out * z).sum(-1)), want, atol=1e-3)


def test_scalar_and_distributional_agree_for_point_mass():
    """With a one-atom next distribution and zero correction, both constructions give the same target."""
    z = cat.support(VMIN, VMAX, I)
    q_logits = jnp.zeros((B, 2, I))
    nxt = jnp.full((B, 2, I), -1e9).at[:, :, 10].set(0.0)  # point mass on z_10 = 20
    r, d, corr, tr = jnp.ones((B,)), jnp.full((B,), 0.99), jnp.zeros((B,)), jnp.zeros((B,))
    a = cat.categorical_td_loss(q_logits, nxt, r, d, corr, tr, z, VMIN, VMAX, "scalar")
    b = cat.categorical_td_loss(q_logits, nxt, r, d, corr, tr, z, VMIN, VMAX, "distributional")
    assert np.isclose(float(a), float(b), atol=1e-4)


def test_truncation_masks_loss():
    z = cat.support(VMIN, VMAX, I)
    lg = jax.random.normal(jax.random.PRNGKey(1), (B, 2, I))
    loss = cat.categorical_td_loss(lg, lg, jnp.ones((B,)), jnp.ones((B,)) * 0.99, jnp.zeros((B,)), jnp.ones((B,)), z, VMIN, VMAX, "scalar")
    assert float(loss) == 0.0


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_sac_categorical_step(mode):
    net = sac_networks.make_scaled_sac_networks(
        OBS, ACT, width=32, depth=8, actor_depth=4, num_atoms=I, v_min=VMIN, v_max=VMAX, critic_head_ln=True,
        **variant_flags("full"),
    )
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    assert net.q_network.apply(None, qp, jnp.zeros((B, OBS)), jnp.zeros((B, ACT))).shape == (B, 2)
    assert net.q_logits(None, qp, jnp.zeros((B, OBS)), jnp.zeros((B, ACT))).shape == (B, 2, I)
    loss = cat.make_sac_critic_loss(net, 1.0, 0.99, VMIN, VMAX, mode)
    opt = optax.adam(3e-4)
    cu = jax.jit(gradient_update_fn_with_norm(loss, opt, None))
    l, qp2, _, gn = cu(qp, pp, None, qp, 0.1, batch(k), k, optimizer_state=opt.init(qp))
    assert np.isfinite(float(l)) and np.isfinite(float(gn)) and float(gn) > 0


@pytest.mark.parametrize("mode", ["scalar", "distributional"])
def test_td3_categorical_step(mode):
    net = td3_networks.make_scaled_td3_networks(
        OBS, ACT, width=32, depth=8, actor_depth=4, num_atoms=I, v_min=VMIN, v_max=VMAX, **variant_flags("full")
    )
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    loss = cat.make_td3_critic_loss(net, 1.0, 0.99, VMIN, VMAX, mode, mask_truncation=True)
    opt = optax.adam(3e-4)
    cu = jax.jit(gradient_update_fn_with_norm(loss, opt, None))
    l, _, _, gn = cu(qp, qp, pp, None, batch(k), k, optimizer_state=opt.init(qp))
    assert np.isfinite(float(l)) and float(gn) > 0


def test_categorical_learns_constant_target():
    """Repeated updates on a fixed batch with terminal transitions (discount 0) drive E[Q] to the reward."""
    net = sac_networks.make_scaled_sac_networks(
        OBS, ACT, width=32, depth=4, actor_depth=4, num_atoms=I, v_min=VMIN, v_max=VMAX, **variant_flags("full")
    )
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    tr = batch(k)._replace(reward=jnp.full((B,), 0.7), discount=jnp.zeros((B,)))
    loss = cat.make_sac_critic_loss(net, 1.0, 0.99, VMIN, VMAX, "scalar")
    opt = optax.adam(3e-3)
    cu = jax.jit(gradient_update_fn_with_norm(loss, opt, None))
    st = opt.init(qp)
    first = None
    for _ in range(1000):
        l, qp, st, _ = cu(qp, pp, None, qp, 0.1, tr, k, optimizer_state=st)
        first = float(l) if first is None else first
    # target two-hot: 0.65 on z_0 = 0, 0.35 on z_1 = 2; check the distribution and the expectation
    probs = jax.nn.softmax(net.q_logits(None, qp, tr.observation, tr.action), axis=-1)
    assert float(jnp.min(probs[..., 0] + probs[..., 1])) > 0.97
    assert np.allclose(np.asarray(probs[..., 1]), 0.35, atol=0.05)
    q = net.q_network.apply(None, qp, tr.observation, tr.action)
    assert np.allclose(np.asarray(q), 0.7, atol=1.0) and float(l) < first


def test_default_network_unchanged():
    """num_atoms=1 and no head LN: the parameter tree is the one of Stage 1 (old checkpoints still load)."""
    a = sac_networks.make_scaled_sac_networks(OBS, ACT, width=32, depth=8, **variant_flags("full"))
    p = a.q_network.init(jax.random.PRNGKey(0))
    heads = [v["kernel"].shape for k, v in p["params"]["ScaledMLP_0"].items() if k.startswith("Dense")]
    assert heads == [(32, 1)]  # head Dense directly on the trunk, 1 output
    assert set(p["params"]["ScaledMLP_0"]) == {"ScaledTrunk_0", "Dense_0"}
    assert a.q_logits is None
