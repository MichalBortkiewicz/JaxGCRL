"""Representation diagnostics (common/rep_metrics.py) and the ScaledTrunk feature hooks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.agents.common import rep_metrics as rm
from jaxgcrl.agents.common.scaled_net import variant_flags
from jaxgcrl.agents.sac import networks as sac_networks
from jaxgcrl.agents.td3 import networks as td3_networks

OBS, ACT, B, W = 31, 8, 256, 32


def test_rank_of_k_orthogonal_directions():
    q, _ = np.linalg.qr(np.random.RandomState(0).randn(B, W))
    for k in (1, 5, 20):
        phi = jnp.asarray(q[:, :k] * 3.0)
        s = jnp.linalg.svd(phi, compute_uv=False)
        assert int(rm.srank(s)) == k
        assert abs(float(rm.erank(s)) - k) < 1e-3


def test_dormant_fraction():
    h = np.random.RandomState(1).randn(B, 10)
    h[:, :3] = 0.0  # 3 dead units
    h[:, 3] *= 1e-3  # 1 nearly dead unit (score << 0.1)
    assert abs(float(rm.dormant_fraction(jnp.asarray(h))) - 0.4) < 1e-6


def test_weight_norms():
    p = {"params": {"a": {"kernel": jnp.ones((2, 2)), "bias": jnp.ones(2) * 100}, "b": {"kernel": jnp.ones((3, 3))}}}
    total, mx = rm.weight_norms(p)
    assert abs(float(total) - np.sqrt(4 + 9)) < 1e-5 and abs(float(mx) - 3.0) < 1e-5


@pytest.mark.parametrize("variant", ["plain", "full"])
@pytest.mark.parametrize("depth", [4, 8])
def test_sac_scaled_hooks(variant, depth):
    net = sac_networks.make_scaled_sac_networks(OBS, ACT, width=W, depth=depth, actor_depth=4, **variant_flags(variant))
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    obs, act = jax.random.normal(k, (B, OBS)), jnp.tanh(jax.random.normal(k, (B, ACT)))
    # hooks do not change the normal forward pass
    q1 = net.q_network.apply(None, qp, obs, act)
    st = net.q_features(None, qp, obs, act)
    groups = rm._sown(st, "layer")
    assert len(groups) == 2 and all(len(g) == depth + 1 for g in groups)  # stem + depth layers, per critic
    assert np.allclose(np.asarray(q1), np.asarray(net.q_network.apply(None, qp, obs, act)))
    m = jax.jit(lambda p: rm.network_rep_metrics("rep_q", p, lambda pp_, o, a: net.q_features(None, pp_, o, a), obs, act))(qp)
    for key in ("wnorm", "wnorm_max", "srank", "erank", "dormant", "dormant_last", "fnorm"):
        v = float(m[f"rep_q_{key}"])
        assert np.isfinite(v)
    assert 1 <= float(m["rep_q_srank"]) <= W
    assert 0.0 <= float(m["rep_q_dormant"]) <= 1.0
    mp = rm.network_rep_metrics("rep_pi", pp, lambda p, o: net.policy_features(None, p, o), obs)
    assert 1 <= float(mp["rep_pi_srank"]) <= W


def test_td3_scaled_hooks():
    net = td3_networks.make_scaled_td3_networks(OBS, ACT, width=W, depth=8, actor_depth=8, **variant_flags("full"))
    k = jax.random.PRNGKey(0)
    pp = net.policy_network.init(k)
    obs = jax.random.normal(k, (B, OBS))
    mp = rm.network_rep_metrics("rep_pi", pp, lambda p, o: net.policy_features(None, p, o), obs)
    assert np.isfinite(float(mp["rep_pi_erank"]))


def test_legacy_only_weight_norms():
    net = sac_networks.make_sac_networks(OBS, ACT, hidden_layer_sizes=(W, W))
    qp = net.q_network.init(jax.random.PRNGKey(0))
    assert net.q_features is None
    m = rm.network_rep_metrics("rep_q", qp, None)
    assert set(m) == {"rep_q_wnorm", "rep_q_wnorm_max"}
