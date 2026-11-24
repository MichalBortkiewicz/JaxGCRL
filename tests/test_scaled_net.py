import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from jaxgcrl.agents.common.scaled_net import VARIANTS, ScaledMLP, variant_flags


def make(depth, variant="full", width=32, out=3):
    return ScaledMLP(width=width, depth=depth, out_size=out, **variant_flags(variant))


def n_params(tree):
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("depth", [4, 16])
def test_output_shape(variant, depth):
    m = make(depth, variant)
    x = jnp.ones((5, 7))
    p = m.init(jax.random.PRNGKey(0), x)
    assert m.apply(p, x).shape == (5, 3)


@pytest.mark.parametrize("depth", [4, 8])
@pytest.mark.parametrize("use_ln", [False, True])
def test_param_count(depth, use_ln):
    w, i, o = 32, 7, 3
    m = ScaledMLP(width=w, depth=depth, out_size=o, use_ln=use_ln, use_residual=True, activation="swish")
    p = m.init(jax.random.PRNGKey(0), jnp.ones((1, i)))
    expected = (i * w + w) + depth * (w * w + w) + (w * o + o)
    if use_ln:
        expected += (depth + 1) * 2 * w
    assert n_params(p) == expected


@pytest.mark.parametrize("depth", [4, 8, 16])
def test_dense_count(depth):
    p = make(depth).init(jax.random.PRNGKey(0), jnp.ones((1, 7)))
    kernels = [path for path, _ in jax.tree_util.tree_flatten_with_path(p)[0] if "kernel" in jax.tree_util.keystr(path)]
    assert len(kernels) == depth + 2


def test_residual_changes_output_and_plain_loop_matches():
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 7))
    kw = dict(width=16, depth=8, out_size=2, use_ln=True, activation="swish")
    m_res = ScaledMLP(use_residual=True, **kw)
    m_nores = ScaledMLP(use_residual=False, **kw)
    p = m_res.init(jax.random.PRNGKey(0), x)
    # identical parameter structure, so the same params can be used for both
    out_res, out_nores = m_res.apply(p, x), m_nores.apply(p, x)
    assert not np.allclose(out_res, out_nores)

    # hand-written loop of Dense -> LN -> swish for the no-residual network
    params = p["params"]["ScaledTrunk_0"]
    h = x
    dense_names = sorted([k for k in params if k.startswith("Dense")], key=lambda k: int(k.split("_")[1]))
    ln_names = sorted([k for k in params if k.startswith("LayerNorm")], key=lambda k: int(k.split("_")[1]))
    for d, l in zip(dense_names, ln_names):
        h = h @ params[d]["kernel"] + params[d]["bias"]
        h = nn.LayerNorm().apply({"params": params[l]}, h)
        h = nn.swish(h)
    head = p["params"]["Dense_0"]
    h = h @ head["kernel"] + head["bias"]
    np.testing.assert_allclose(np.asarray(h), np.asarray(out_nores), rtol=1e-5, atol=1e-5)


def test_depth64_finite():
    m = make(64, "full", width=64)
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 31))
    p = m.init(jax.random.PRNGKey(0), x)
    y = m.apply(p, x)
    g = jax.grad(lambda p: jnp.sum(m.apply(p, x) ** 2))(p)
    assert np.all(np.isfinite(np.asarray(y)))
    assert all(np.all(np.isfinite(np.asarray(l))) for l in jax.tree_util.tree_leaves(g))


def test_bad_depth_raises():
    with pytest.raises(ValueError):
        make(6).init(jax.random.PRNGKey(0), jnp.ones((1, 7)))
