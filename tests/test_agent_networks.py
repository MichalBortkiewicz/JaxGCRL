"""One jitted critic/actor gradient step with the scaled networks of SAC and TD3 (ant_big_maze sizes)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.training.agents.sac import losses as sac_losses

from jaxgcrl.agents.common.scaled_net import VARIANTS, variant_flags
from jaxgcrl.agents.common.train_utils import gradient_update_fn_with_norm
from jaxgcrl.agents.sac import networks as sac_networks
from jaxgcrl.agents.sac.sac import Transition
from jaxgcrl.agents.td3 import losses as td3_losses
from jaxgcrl.agents.td3 import networks as td3_networks

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


def finite(tree):
    return all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("depth", [4, 8])
def test_sac_scaled_step(variant, depth):
    net = sac_networks.make_scaled_sac_networks(OBS, ACT, width=32, depth=depth, actor_depth=depth, **variant_flags(variant))
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    assert net.q_network.apply(None, qp, jnp.zeros((B, OBS)), jnp.zeros((B, ACT))).shape == (B, 2)
    _, critic_loss, actor_loss = sac_losses.make_losses(net, reward_scaling=1.0, discounting=0.99, action_size=ACT)
    opt = optax.adam(3e-4)
    cu = jax.jit(gradient_update_fn_with_norm(critic_loss, opt, None))
    au = jax.jit(gradient_update_fn_with_norm(actor_loss, opt, None))
    tr = batch(k)
    l, qp2, _, gn = cu(qp, pp, None, qp, 0.1, tr, k, optimizer_state=opt.init(qp))
    assert np.isfinite(l) and np.isfinite(gn) and finite(qp2)
    l, pp2, _, gn = au(pp, None, qp, 0.1, tr, k, optimizer_state=opt.init(pp))
    assert np.isfinite(l) and np.isfinite(gn) and finite(pp2)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_td3_scaled_step(variant):
    net = td3_networks.make_scaled_td3_networks(OBS, ACT, width=32, depth=8, actor_depth=8, **variant_flags(variant))
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    a = net.policy_network.apply(None, pp, jnp.zeros((B, OBS)))
    assert a.shape == (B, ACT) and np.all(np.abs(np.asarray(a)) <= 1)
    critic_loss, actor_loss = td3_losses.make_losses(net, 1.0, 0.99, 0.2, 0.5, mask_truncation=True)
    opt = optax.adam(3e-4)
    cu = jax.jit(gradient_update_fn_with_norm(critic_loss, opt, None))
    au = jax.jit(gradient_update_fn_with_norm(actor_loss, opt, None))
    tr = batch(k)
    l, qp2, _, gn = cu(qp, qp, pp, None, tr, k, optimizer_state=opt.init(qp))
    assert np.isfinite(l) and np.isfinite(gn) and finite(qp2)
    l, pp2, _, gn = au(pp, qp, None, tr, optimizer_state=opt.init(pp))
    assert np.isfinite(l) and np.isfinite(gn) and finite(pp2)


def test_td3_truncation_mask():
    net = td3_networks.make_scaled_td3_networks(OBS, ACT, width=16, depth=4, actor_depth=4, **variant_flags("full"))
    k = jax.random.PRNGKey(0)
    qp, pp = net.q_network.init(k), net.policy_network.init(k)
    tr = batch(k)
    tr_all_trunc = tr._replace(extras={**tr.extras, "state_extras": {"truncation": jnp.ones((B,)), "traj_id": jnp.zeros((B,))}})
    masked, _ = td3_losses.make_losses(net, 1.0, 0.99, 0.2, 0.5, mask_truncation=True)
    unmasked, _ = td3_losses.make_losses(net, 1.0, 0.99, 0.2, 0.5, mask_truncation=False)
    assert float(masked(qp, qp, pp, None, tr_all_trunc, k)) == 0.0
    assert float(unmasked(qp, qp, pp, None, tr_all_trunc, k)) > 0.0
    np.testing.assert_allclose(float(masked(qp, qp, pp, None, tr, k)), float(unmasked(qp, qp, pp, None, tr, k)))


def test_legacy_sac_networks_unchanged_param_structure():
    from flax import linen

    net = sac_networks.make_sac_networks(OBS, ACT, hidden_layer_sizes=[256, 256], activation=linen.relu)
    p = net.q_network.init(jax.random.PRNGKey(0))
    assert sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(p)) == 2 * ((OBS + ACT) * 256 + 256 + 256 * 256 + 256 + 256 + 1)
